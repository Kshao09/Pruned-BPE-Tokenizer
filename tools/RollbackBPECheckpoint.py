from __future__ import annotations

"""
Rollback a PrunedBPETrainerCythonParallel checkpoint to an earlier exact
Standard-BPE vocabulary size without retraining.

Core idea
---------
At cutoff K, any token ID < K already existed. Any token ID >= K was created
later and can be recursively expanded through id_to_children until only IDs
< K remain. Applying that expansion to the saved corpus exactly reverses all
merges performed after K.

Verification
------------
1. Always checks byte preservation, ID ranges, merge-tree consistency, and
   final_token_counts.
2. Optionally compares the rolled state against a genuine checkpoint at K,
   including the full corpus segmentation. This is the recommended one-time
   validation: roll 13,996 -> 11,996 and compare with the real 11,996 file.
"""
import hashlib
import os
import struct
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, Tuple

from PrunedBPETrainerCythonParallel import PrunedBPETrainerCythonParallel

Pair = Tuple[int, int]


def _put_int(h, value: int) -> None:
    h.update(struct.pack("<q", int(value)))


def corpus_sha256(corpus: Sequence[Sequence[int]]) -> str:
    h = hashlib.sha256()
    _put_int(h, len(corpus))
    for ids in corpus:
        _put_int(h, len(ids))
        for token_id in ids:
            _put_int(h, token_id)
    return h.hexdigest()


def vocab_sha256(id_to_bytes: dict[int, bytes], id_to_children: dict[int, Pair]) -> str:
    h = hashlib.sha256()
    for token_id in sorted(id_to_bytes):
        token_bytes = id_to_bytes[token_id]
        left_id, right_id = id_to_children[token_id]
        _put_int(h, token_id)
        _put_int(h, len(token_bytes))
        h.update(token_bytes)
        _put_int(h, left_id)
        _put_int(h, right_id)
    return h.hexdigest()


def _first_corpus_mismatch(a, b) -> Optional[str]:
    if len(a) != len(b):
        return f"pretoken count differs: rolled={len(a):,}, reference={len(b):,}"

    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            p = 0
            common = min(len(x), len(y))
            while p < common and x[p] == y[p]:
                p += 1
            return (
                f"first mismatch at pretoken {i:,}, token position {p:,}; "
                f"rolled={x[max(0,p-3):p+4]}, "
                f"reference={y[max(0,p-3):p+4]}"
            )
    return None


def _compare_dict(name: str, rolled: dict, reference: dict) -> Optional[str]:
    if rolled == reference:
        print(f"{name:20s}: EXACT MATCH")
        return None

    rk = set(rolled)
    ek = set(reference)
    missing = sorted(ek - rk)[:5]
    extra = sorted(rk - ek)[:5]
    diffs = []
    for key in sorted(rk & ek):
        if rolled[key] != reference[key]:
            diffs.append((key, rolled[key], reference[key]))
            if len(diffs) == 5:
                break

    print(f"{name:20s}: MISMATCH")
    return (
        f"{name}: missing={missing}, extra={extra}, "
        f"first differing values={diffs}"
    )


def verify_against_reference_checkpoint(
    rolled: PrunedBPETrainerCythonParallel,
    reference_checkpoint_path: str | Path,
    *,
    target_vocab_size: int,
) -> None:
    """Strong verification against a genuine checkpoint saved at target K."""
    reference_checkpoint_path = Path(reference_checkpoint_path)

    print("\n=== Exact Reference Checkpoint Verification ===")
    print(f"Reference checkpoint : {reference_checkpoint_path}")

    reference = PrunedBPETrainerCythonParallel.load_checkpoint(
        str(reference_checkpoint_path),
        train_vocab_size=target_vocab_size,
        visible_vocab_size=target_vocab_size,
        min_exposure_count=0,
    )

    if reference._next_training_id() != target_vocab_size:
        raise RuntimeError(
            f"Reference checkpoint is at {reference._next_training_id():,}, "
            f"not {target_vocab_size:,}."
        )

    failures = []
    for name, x, y in (
        ("id_to_bytes", rolled.id_to_bytes, reference.id_to_bytes),
        ("id_to_children", rolled.id_to_children, reference.id_to_children),
        ("merges", rolled.merges, reference.merges),
        ("final_token_counts", dict(rolled.final_token_counts), dict(reference.final_token_counts)),
    ):
        mismatch = _compare_dict(name, dict(x), dict(y))
        if mismatch:
            failures.append(mismatch)

    rolled_mc = getattr(rolled, "merge_counts", None)
    ref_mc = getattr(reference, "merge_counts", None)
    if rolled_mc is not None and ref_mc is not None:
        mismatch = _compare_dict("merge_counts", dict(rolled_mc), dict(ref_mc))
        if mismatch:
            failures.append(mismatch)
    else:
        print(f"{'merge_counts':20s}: SKIPPED (not present in both)")

    corpus_mismatch = _first_corpus_mismatch(rolled.corpus, reference.corpus)
    if corpus_mismatch is None:
        print(f"{'corpus':20s}: EXACT MATCH")
    else:
        print(f"{'corpus':20s}: MISMATCH")
        failures.append(corpus_mismatch)

    rolled_ch = corpus_sha256(rolled.corpus)
    ref_ch = corpus_sha256(reference.corpus)
    rolled_vh = vocab_sha256(rolled.id_to_bytes, rolled.id_to_children)
    ref_vh = vocab_sha256(reference.id_to_bytes, reference.id_to_children)

    print(f"Rolled corpus SHA-256    : {rolled_ch}")
    print(f"Reference corpus SHA-256 : {ref_ch}")
    print(f"Rolled vocab SHA-256     : {rolled_vh}")
    print(f"Reference vocab SHA-256  : {ref_vh}")

    if failures:
        raise RuntimeError(
            "ROLLBACK REFERENCE VERIFICATION FAILED:\n  - "
            + "\n  - ".join(failures[:10])
        )

    print(
        "\nREFERENCE VERIFICATION PASSED: rolled-back state is an "
        "EXACT MATCH for all checked checkpoint fields."
    )


def rollback_checkpoint(
    source_checkpoint_path: str | Path,
    target_vocab_size: int,
    output_checkpoint_path: str | Path,
    *,
    reference_checkpoint_path: str | Path | None = None,
    progress_every: int = 250_000,
) -> PrunedBPETrainerCythonParallel:
    """
    Roll a later ordinary-BPE checkpoint back to target_vocab_size.

    target_vocab_size is the ordinary BPE size / next token ID. Reserved
    special tokens are external and are not included.
    """
    source_path = Path(source_checkpoint_path)
    output_path = Path(output_checkpoint_path)

    if target_vocab_size < 256:
        raise ValueError("target_vocab_size must be >= 256")
    if not source_path.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source_path}")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Refusing to overwrite the source checkpoint")
    if reference_checkpoint_path is not None:
        if output_path.resolve() == Path(reference_checkpoint_path).resolve():
            raise ValueError("Refusing to overwrite the reference checkpoint")

    print("\n=== Load Source Checkpoint ===")
    print(f"Source checkpoint : {source_path}")
    print(f"Target vocab size  : {target_vocab_size:,}")

    trainer = PrunedBPETrainerCythonParallel.load_checkpoint(
        str(source_path),
        train_vocab_size=target_vocab_size,
        visible_vocab_size=target_vocab_size,
        min_exposure_count=0,
    )

    source_size = trainer._next_training_id()
    if source_size <= target_vocab_size:
        raise ValueError(
            f"Source must be later than target: source={source_size:,}, "
            f"target={target_vocab_size:,}"
        )

    if sorted(trainer.id_to_bytes) != list(range(source_size)):
        raise RuntimeError("Source token IDs are not contiguous from 0")
    if not trainer.corpus:
        raise RuntimeError("Source checkpoint has no corpus state")

    original_bytes = trainer.id_to_bytes
    original_children = trainer.id_to_children

    # Validate merge-tree ordering first.
    for token_id in range(256, source_size):
        children = original_children.get(token_id)
        if children is None or children == (-1, -1):
            raise RuntimeError(f"Learned token {token_id} has no valid children")
        left_id, right_id = children
        if not (0 <= left_id < token_id and 0 <= right_id < token_id):
            raise RuntimeError(
                f"Token {token_id} has invalid children {children}; "
                "children must have earlier IDs"
            )

    @lru_cache(maxsize=None)
    def expand_to_cutoff(token_id: int) -> tuple[int, ...]:
        if token_id < target_vocab_size:
            return (token_id,)

        left_id, right_id = original_children[token_id]
        expansion = expand_to_cutoff(left_id) + expand_to_cutoff(right_id)

        # Validate the merge-tree expansion itself.
        expanded = b"".join(original_bytes[x] for x in expansion)
        if expanded != original_bytes[token_id]:
            raise RuntimeError(f"Byte-integrity failure for token {token_id}")
        return expansion

    print("\n=== Build Rollback Expansions ===")
    for token_id in range(target_vocab_size, source_size):
        expand_to_cutoff(token_id)
    print(f"Later token types to remove : {source_size - target_vocab_size:,}")
    print(f"Expansion cache entries     : {expand_to_cutoff.cache_info().currsize:,}")

    print("\n=== Restore Corpus Segmentation ===")
    before_count = 0
    after_count = 0
    changed_pretokens = 0
    expanded_occurrences = 0

    # Restore in place to avoid holding two complete Corpus-II states.
    for index, ids in enumerate(trainer.corpus):
        before_count += len(ids)

        if not any(token_id >= target_vocab_size for token_id in ids):
            after_count += len(ids)
        else:
            before_bytes = b"".join(original_bytes[x] for x in ids)
            restored: list[int] = []

            for token_id in ids:
                if token_id >= target_vocab_size:
                    expanded_occurrences += 1
                    restored.extend(expand_to_cutoff(token_id))
                else:
                    restored.append(token_id)

            if any(x >= target_vocab_size for x in restored):
                raise RuntimeError(f"Rollback left a later token in pretoken {index:,}")

            after_bytes = b"".join(original_bytes[x] for x in restored)
            if after_bytes != before_bytes:
                raise RuntimeError(f"Pretoken bytes changed at corpus index {index:,}")

            trainer.corpus[index] = restored
            after_count += len(restored)
            changed_pretokens += 1

        processed = index + 1
        if progress_every and processed % progress_every == 0:
            print(
                f"  processed={processed:,}/{len(trainer.corpus):,}; "
                f"changed={changed_pretokens:,}; "
                f"expanded occurrences={expanded_occurrences:,}"
            )

    print("\n=== Truncate BPE State ===")
    trainer.id_to_bytes = {
        token_id: value
        for token_id, value in original_bytes.items()
        if token_id < target_vocab_size
    }
    trainer.id_to_children = {
        token_id: value
        for token_id, value in original_children.items()
        if token_id < target_vocab_size
    }
    trainer.merges = {
        pair: new_id
        for pair, new_id in trainer.merges.items()
        if new_id < target_vocab_size
    }
    if hasattr(trainer, "merge_counts"):
        trainer.merge_counts = {
            token_id: count
            for token_id, count in trainer.merge_counts.items()
            if token_id < target_vocab_size
        }

    trainer.final_token_counts = Counter()
    for ids in trainer.corpus:
        trainer.final_token_counts.update(ids)

    # Clear export-only remapping state if present.
    if hasattr(trainer, "old_to_new_id"):
        trainer.old_to_new_id = {}
    if hasattr(trainer, "special_token_ids"):
        trainer.special_token_ids = {}

    # Save as a clean ordinary Standard-BPE state at K.
    trainer.train_vocab_size = target_vocab_size
    trainer.visible_vocab_size = target_vocab_size
    trainer.min_exposure_count = 0

    print("\n=== Internal Rollback Verification ===")
    if trainer._next_training_id() != target_vocab_size:
        raise RuntimeError("next training ID does not equal target")

    expected_ids = list(range(target_vocab_size))
    if sorted(trainer.id_to_bytes) != expected_ids:
        raise RuntimeError("id_to_bytes is not exactly 0..target-1")
    if sorted(trainer.id_to_children) != expected_ids:
        raise RuntimeError("id_to_children is not exactly 0..target-1")

    for token_id in range(256, target_vocab_size):
        left_id, right_id = trainer.id_to_children[token_id]
        if not (0 <= left_id < token_id and 0 <= right_id < token_id):
            raise RuntimeError(f"Retained token {token_id} has invalid children")

    recomputed = Counter()
    max_corpus_id = -1
    for ids in trainer.corpus:
        recomputed.update(ids)
        if ids:
            max_corpus_id = max(max_corpus_id, max(ids))

    if max_corpus_id >= target_vocab_size:
        raise RuntimeError(f"Corpus still contains later ID {max_corpus_id}")
    if recomputed != trainer.final_token_counts:
        raise RuntimeError("final_token_counts does not match restored corpus")
    if len(trainer.merges) != target_vocab_size - 256:
        raise RuntimeError(
            f"Expected {target_vocab_size - 256:,} merges, "
            f"found {len(trainer.merges):,}"
        )
    for pair, new_id in trainer.merges.items():
        if trainer.id_to_children.get(new_id) != pair:
            raise RuntimeError(f"Merge map mismatch at token {new_id}")

    print(f"Source ordinary vocab size : {source_size:,}")
    print(f"Rolled ordinary vocab size : {trainer._next_training_id():,}")
    print(f"Removed later token types  : {source_size - target_vocab_size:,}")
    print(f"Corpus pretokens           : {len(trainer.corpus):,}")
    print(f"Changed pretokens          : {changed_pretokens:,}")
    print(f"Expanded occurrences       : {expanded_occurrences:,}")
    print(f"Corpus tokens before       : {before_count:,}")
    print(f"Corpus tokens after        : {after_count:,}")
    print(f"Max corpus token ID        : {max_corpus_id:,}")
    print(f"Corpus SHA-256             : {corpus_sha256(trainer.corpus)}")
    print(f"Vocab/tree SHA-256         : {vocab_sha256(trainer.id_to_bytes, trainer.id_to_children)}")
    print("Byte preservation          : PASSED for every changed pretoken")
    print("Internal state checks      : PASSED")

    if reference_checkpoint_path is not None:
        verify_against_reference_checkpoint(
            trainer,
            reference_checkpoint_path,
            target_vocab_size=target_vocab_size,
        )

    # Save only after all requested verification passes.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(output_path))

    print("\n=== Rollback Complete ===")
    print(f"Saved checkpoint : {output_path}")
    print(f"next_id          : {trainer._next_training_id():,}")
    return trainer

# def verification_example_24k_to_23k() -> None:
#     """
#     Alternative/recommended validation method.
#
#     If checkpoint_vocab_23000.pkl and checkpoint_vocab_24000.pkl came from
#     the same Corpus-II Standard-BPE trajectory, rollback 24K -> 23K and
#     demand an exact match against the genuine 23K checkpoint.
#     """
#     from settings import PROJECT_ROOT
#
#     # Change this directory if your 23K/24K checkpoints are stored elsewhere.
#     d = os.path.join(PROJECT_ROOT, "checkpoints", "c1c2")
#     rollback_checkpoint(
#         os.path.join(d, "checkpoint_vocab_13500.pkl"),
#         12996,
#         os.path.join(d, "rollback_test_checkpoint_vocab_12996.pkl"),
#         reference_checkpoint_path=os.path.join(d, "checkpoint_vocab_12996.pkl")
#     )

if __name__ == "__main__":
    from settings import PROJECT_ROOT

    src_vocab_size = 12560
    tgt_vocab_size = 12420
    d = os.path.join(PROJECT_ROOT, "checkpoints", "c2")
    rollback_checkpoint(
        os.path.join(d, f"checkpoint_vocab_{src_vocab_size}.pkl"),
        tgt_vocab_size,
        os.path.join(d, f"checkpoint_vocab_{tgt_vocab_size}.pkl"),
    )
