from __future__ import annotations

"""
Standalone MinGram trainer adapted to the Pruned-BPE project.

Purpose
-------
Reproduce the DEFAULT MinGram training logic from sanderland/script_tok while:

1. using the Pruned-BPE byte vocabulary (0..255);
2. using the exact Pruned-BPE pretokenization already stored in / produced by
   PrunedBPETrainerCythonParallel;
3. allowing the BPE seed stage to resume from an existing Pruned-BPE checkpoint;
4. using the existing Pruned-BPE BPE trainer so resumed BPE tokens follow the
   same deterministic training path as the user's saved checkpoints;
5. exporting a simple model-visible vocab file that can be evaluated by the
   existing MaxLenTokenizerExp.py.

This file implements DEFAULT MinGram:
    - BPE seed with overshoot_factor (paper experiment: 1.15)
    - Hard-EM E-step using MinGram's DP path score
    - M-step removes zero-use non-required tokens and re-estimates log-probability
    - default pruning by usage_count / learned log-probability
    - NO PathPiece/MI pruning (that would be MinGram-PP)

It intentionally does not depend on script_tok.

Expected project files beside this file
---------------------------------------
    PrunedBPETrainer.py
    PrunedBPETrainerCythonParallel.py
    pruned_bpe_pretokenizer.py
    settings.py                         # only needed by the example main()

The BPE checkpoint must be at or below the requested MinGram BPE seed size.
"""

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pruned_bpe_pretokenizer import pretokenize
from PrunedBPETrainer import PrunedBPETrainer
from PrunedBPETrainerCythonParallel import PrunedBPETrainerCythonParallel


DEFAULT_SCORE_DELTA = 1.0 / 100_000.0
ATOMIC_TOKEN_COUNT = 256


@dataclass
class MinGramToken:
    """One token in the MinGram token-list model."""

    id: int
    token_bytes: bytes
    log_prob: float
    required: bool = False


class _Trie:
    TOKEN_KEY = -1

    def __init__(self, tokens: Iterable[MinGramToken]):
        self.root: dict = {}
        for token in tokens:
            node = self.root
            for byte_value in token.token_bytes:
                node = node.setdefault(byte_value, {})
            node[self.TOKEN_KEY] = token


class MinGramModel:
    """
    Minimal MinGram model matching the relevant logic in script_tok/model.py.

    Path score:
        score_delta * sum(log_prob) - number_of_tokens

    With the default score_delta=1e-5, minimum token count is dominant and
    learned log-probability is used as the tiebreaking signal, matching the
    reference MinGram implementation.
    """

    def __init__(
        self,
        tokens: Iterable[MinGramToken],
        *,
        score_delta: float = DEFAULT_SCORE_DELTA,
    ):
        token_list = list(tokens)
        self.tokens: Dict[int, MinGramToken] = {t.id: t for t in token_list}
        self.trie = _Trie(token_list)
        self.score_delta = score_delta

    def encode_chunk(self, chunk: bytes) -> List[MinGramToken]:
        """
        Encode one Pruned-BPE pretoken (already converted to UTF-8 bytes).

        This follows the forward DP in script_tok's MinGramModel.encode_chunk().
        """
        chunk_len = len(chunk)
        if chunk_len == 0:
            return []

        sd = self.score_delta
        best_score = [float("-inf")] * (chunk_len + 1)
        best_prev: List[Optional[MinGramToken]] = [None] * (chunk_len + 1)
        best_score[0] = 0.0

        trie_root = self.trie.root

        for pos in range(chunk_len):
            score = best_score[pos]

            # Reference implementation skips only genuinely unreachable
            # positions; -inf may still be reachable through an unused atomic
            # fallback token.
            if pos > 0 and best_prev[pos] is None:
                continue

            node = trie_root
            for i in range(pos, chunk_len):
                node = node.get(chunk[i])
                if node is None:
                    break

                token = node.get(_Trie.TOKEN_KEY)
                if token is None:
                    continue

                nxt = i + 1
                contrib = sd * token.log_prob if sd != 0.0 else 0.0
                new_score = score + contrib - 1.0

                # Match the reference implementation:
                # >= with epsilon; later write wins on a numerical tie.
                if new_score >= best_score[nxt] - 1e-12:
                    best_score[nxt] = new_score
                    best_prev[nxt] = token

        output: List[MinGramToken] = []
        pos = chunk_len

        while pos > 0:
            token = best_prev[pos]
            if token is None:
                raise RuntimeError(
                    f"No valid MinGram segmentation found at byte position {pos} "
                    f"for chunk {chunk!r}"
                )
            output.append(token)
            pos -= len(token.token_bytes)

        output.reverse()
        return output

    def encode(self, text: str) -> List[int]:
        """Encode text with the project's exact Pruned-BPE pretokenizer."""
        ids: List[int] = []
        chunks = pretokenize(text)

        if "".join(chunks) != text:
            raise RuntimeError("pretokenize() did not preserve the original input text")

        for chunk in chunks:
            if chunk:
                ids.extend(t.id for t in self.encode_chunk(chunk.encode("utf-8")))

        return ids


class MinGramTrainer:
    """
    Standalone implementation of the default MinGram training procedure.

    target_vocab_size is the EXTERNAL model-visible size, including the user's
    reserved special tokens. For the current project:

        target_vocab_size = 18_000
        reserved tokens   = 4
        base byte tokens  = 256
        learned target    = 17_740

    With overshoot_factor=1.15:

        BPE learned seed = int(17_740 * 1.15) = 20_401
        BPE ordinary size = 256 + 20_401 = 20_657

    The four reserved special tokens are not involved in BPE or MinGram
    training. They are appended only during final export, matching Pruned BPE.
    """
    MAX_ITERATIONS = 100

    def __init__(
        self,
        *,
        target_vocab_size: int = 18_000,
        overshoot_factor: float = 1.15,
        num_em_iterations: int = 2,
        score_delta: float = DEFAULT_SCORE_DELTA,
        special_token_texts: Sequence[str] = (
            "<_EOS_>",
            "<_SOS_>",
            "<_SEP_>",
            "<_PAD_>",
        ),
        num_workers: int = 20,
        progress_every: int = 100_000,
    ):
        self.target_vocab_size = int(target_vocab_size)
        self.overshoot_factor = float(overshoot_factor)
        self.num_em_iterations = int(num_em_iterations)
        self.score_delta = float(score_delta)
        self.special_token_texts = list(special_token_texts)
        self.num_workers = int(num_workers)
        self.progress_every = int(progress_every)

        if self.target_vocab_size <= ATOMIC_TOKEN_COUNT + len(self.special_token_texts):
            raise ValueError("target_vocab_size is too small")
        if self.overshoot_factor <= 1.0:
            raise ValueError("overshoot_factor must be > 1.0")
        if self.num_em_iterations < 1:
            raise ValueError("num_em_iterations must be >= 1")

        self.num_reserved = len(self.special_token_texts)
        self.final_ordinary_vocab_size = self.target_vocab_size - self.num_reserved
        self.final_learned_vocab_size = (
            self.final_ordinary_vocab_size - ATOMIC_TOKEN_COUNT
        )

        # Match script_tok:
        self.bpe_seed_learned_size = math.ceil(
            self.final_learned_vocab_size * self.overshoot_factor
        )

        self.bpe_seed_ordinary_size = (
            ATOMIC_TOKEN_COUNT + self.bpe_seed_learned_size
        )

        self.bpe_trainer: Optional[PrunedBPETrainerCythonParallel] = None
        self.atomic_corpus: List[Tuple[bytes, int]] = []
        self.model: Optional[MinGramModel] = None
        self.metadata: dict = {}

    # ------------------------------------------------------------------
    # BPE seed stage
    # ------------------------------------------------------------------
    def prepare_bpe_seed(
        self,
        *,
        checkpoint_path: Optional[str] = None,
        corpus_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_vocab_sizes: Optional[Sequence[int]] = None,
        verification_checkpoint_path: Optional[str] = None,
        verification_vocab_size: Optional[int] = None,
    ) -> PrunedBPETrainerCythonParallel:
        """
        Build the oversized BPE seed required by MinGram.

        Resume path:
            1. Load an existing Standard-BPE checkpoint.
            2. If verification is requested, train only to the early
               verification_vocab_size and save a NEW checkpoint there.
            3. Immediately compare the regenerated state with the OLD
               reference checkpoint.
            4. If verification fails, raise RuntimeError and stop.
            5. If verification passes, reload the newly generated checkpoint
               and continue to the final MinGram BPE seed size.
            6. Save a checkpoint at the final BPE seed size.

        IMPORTANT:
        checkpoint_dir should be a NEW output directory for this MinGram run.
        Do not use the directory containing the old verification reference.
        """
        print("\n=== MinGram BPE Seed Configuration ===")
        print(f"External target vocabulary : {self.target_vocab_size:,}")
        print(f"Reserved special tokens    : {self.num_reserved:,}")
        print(f"Final ordinary vocabulary  : {self.final_ordinary_vocab_size:,}")
        print(f"Final learned tokens       : {self.final_learned_vocab_size:,}")
        print(f"Overshoot factor           : {self.overshoot_factor:.4f}")
        print(f"BPE seed learned tokens    : {self.bpe_seed_learned_size:,}")
        print(f"BPE seed ordinary size     : {self.bpe_seed_ordinary_size:,}")

        requested_checkpoints = list(checkpoint_vocab_sizes or [])
        if self.bpe_seed_ordinary_size not in requested_checkpoints:
            requested_checkpoints.append(self.bpe_seed_ordinary_size)

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        if checkpoint_path:
            if verification_checkpoint_path and verification_vocab_size is None:
                raise ValueError(
                    "verification_vocab_size must be provided when "
                    "verification_checkpoint_path is provided."
                )

            first_target = (
                int(verification_vocab_size)
                if verification_checkpoint_path
                else self.bpe_seed_ordinary_size
            )

            if first_target > self.bpe_seed_ordinary_size:
                raise ValueError(
                    f"Verification vocabulary size {first_target:,} is larger "
                    f"than the final BPE seed size "
                    f"{self.bpe_seed_ordinary_size:,}."
                )

            trainer = PrunedBPETrainerCythonParallel.load_checkpoint(
                checkpoint_path,
                train_vocab_size=first_target,
                visible_vocab_size=first_target,
                min_exposure_count=0,
            )

            next_id = trainer._next_training_id()

            print(f"\nLoaded BPE checkpoint      : {checkpoint_path}")
            print(f"Checkpoint next_id         : {next_id:,}")

            if next_id > self.bpe_seed_ordinary_size:
                raise ValueError(
                    "The supplied BPE checkpoint is already larger than the "
                    f"required MinGram seed ({next_id:,} > "
                    f"{self.bpe_seed_ordinary_size:,}). Use an earlier checkpoint."
                )

            if verification_checkpoint_path and next_id > first_target:
                raise ValueError(
                    f"The supplied resume checkpoint is already at {next_id:,}, "
                    f"past verification point {first_target:,}. "
                    "Use an earlier resume checkpoint."
                )

            if next_id < first_target:
                print(
                    f"\nStage 1: resuming Standard BPE from "
                    f"{next_id:,} to {first_target:,}..."
                )

                trainer.train(
                    texts=None,
                    checkpoint_vocab_sizes=[first_target],
                    checkpoint_dir=checkpoint_dir,
                    num_workers=self.num_workers,
                )
            else:
                print(
                    f"\nResume checkpoint is already exactly at {first_target:,}."
                )

            current_size = trainer._next_training_id()
            if current_size != first_target:
                raise RuntimeError(
                    f"Stage 1 ended at {current_size:,}; expected {first_target:,}."
                )

            # Verify immediately. A mismatch raises RuntimeError and is
            # intentionally NOT caught, so Stage 2 never starts on failure.
            if verification_checkpoint_path:
                self.bpe_trainer = trainer

                print(
                    f"\nVerifying regenerated BPE at vocabulary size {first_target:,}..."
                )

                self.verify_bpe_against_checkpoint(verification_checkpoint_path)
                print(
                    "\nVerification PASSED. Continuing Standard BPE training..."
                )

            current_size = trainer._next_training_id()

            # Continue from the newly generated verified checkpoint.
            if current_size < self.bpe_seed_ordinary_size:
                if not checkpoint_dir:
                    raise ValueError(
                        "checkpoint_dir is required for staged resume training."
                    )

                generated_checkpoint = os.path.join(
                    checkpoint_dir, f"checkpoint_vocab_{current_size}.pkl",
                )

                if not os.path.exists(generated_checkpoint):
                    raise FileNotFoundError(
                        "Expected Stage-1 checkpoint was not created: "
                        f"{generated_checkpoint}"
                    )

                print(
                    f"\nStage 2: reloading newly generated checkpoint:\n"
                    f"  {generated_checkpoint}"
                )

                trainer = PrunedBPETrainerCythonParallel.load_checkpoint(
                    generated_checkpoint,
                    train_vocab_size=self.bpe_seed_ordinary_size,
                    visible_vocab_size=self.bpe_seed_ordinary_size,
                    min_exposure_count=0,
                )

                print(
                    f"Continuing Standard BPE from {current_size:,} to "
                    f"{self.bpe_seed_ordinary_size:,}..."
                )

                final_checkpoints = list(requested_checkpoints)
                if self.bpe_seed_ordinary_size not in final_checkpoints:
                    final_checkpoints.append(self.bpe_seed_ordinary_size)

                trainer.train(
                    texts=None,
                    checkpoint_vocab_sizes=final_checkpoints,
                    checkpoint_dir=checkpoint_dir,
                    num_workers=self.num_workers,
                )
            else:
                print(
                    "Checkpoint already has exactly the required MinGram BPE seed size."
                )
        else:
            if corpus_dir is None:
                raise ValueError(
                    "Provide either checkpoint_path or corpus_dir to prepare_bpe_seed()."
                )

            print(f"\nLoading fresh corpus from: {corpus_dir}")
            texts = PrunedBPETrainer.load_data(corpus_dir)

            trainer = PrunedBPETrainerCythonParallel(
                train_vocab_size=self.bpe_seed_ordinary_size,
                visible_vocab_size=self.bpe_seed_ordinary_size,
                min_exposure_count=0,
            )

            trainer.train(
                texts=texts,
                checkpoint_vocab_sizes=requested_checkpoints,
                checkpoint_dir=checkpoint_dir,
                num_workers=self.num_workers,
            )
            del texts

        actual_size = trainer._next_training_id()

        if actual_size != self.bpe_seed_ordinary_size:
            raise RuntimeError(
                f"BPE seed ended at size {actual_size:,}; expected "
                f"{self.bpe_seed_ordinary_size:,}"
            )

        if checkpoint_dir:
            final_checkpoint = os.path.join(
                checkpoint_dir,
                f"checkpoint_vocab_{self.bpe_seed_ordinary_size}.pkl",
            )
            if os.path.exists(final_checkpoint):
                print(f"\nFinal BPE checkpoint saved : {final_checkpoint}")
            else:
                print(
                    "\nWARNING: expected final BPE checkpoint was not found at:\n"
                    f"  {final_checkpoint}"
                )

        self.bpe_trainer = trainer
        self._build_atomic_corpus_from_bpe_checkpoint_state()
        return trainer

    def verify_bpe_against_checkpoint(self, reference_checkpoint_path: str) -> None:
        """
        Verify that the generated/resumed BPE seed matches a previously saved
        Pruned-BPE checkpoint token-for-token.

        This compares token bytes, BPE children, and (when available) merge
        counts for every ordinary BPE token in the common range.

        It is useful for the user's planned check that resuming from (say) an
        18K checkpoint reproduces the exact Standard-BPE tokens previously
        obtained at a later checkpoint.
        """
        if self.bpe_trainer is None:
            raise RuntimeError("prepare_bpe_seed() must be called first")

        reference = PrunedBPETrainerCythonParallel.load_checkpoint(
            reference_checkpoint_path,
            train_vocab_size=self.bpe_seed_ordinary_size,
            visible_vocab_size=self.bpe_seed_ordinary_size,
            min_exposure_count=0,
        )

        generated = self.bpe_trainer
        generated_ids = set(generated.id_to_bytes)
        reference_ids = set(reference.id_to_bytes)
        common_ids = sorted(generated_ids & reference_ids)

        mismatches = []

        for token_id in common_ids:
            if generated.id_to_bytes[token_id] != reference.id_to_bytes[token_id]:
                mismatches.append((token_id, "token_bytes"))
                continue

            if generated.id_to_children.get(token_id) != reference.id_to_children.get(token_id):
                mismatches.append((token_id, "children"))
                continue

            if hasattr(generated, "merge_counts") and hasattr(reference, "merge_counts"):
                g = generated.merge_counts.get(token_id)
                r = reference.merge_counts.get(token_id)
                if g is not None and r is not None and g != r:
                    mismatches.append((token_id, "merge_count"))

        if mismatches:
            preview = ", ".join(f"{tid}:{kind}" for tid, kind in mismatches[:10])
            raise RuntimeError(
                f"BPE verification FAILED with {len(mismatches):,} mismatch(es). "
                f"First mismatches: {preview}"
            )

        print("\n=== BPE Verification ===")
        print(f"Reference checkpoint      : {reference_checkpoint_path}")
        print(f"Common token IDs checked  : {len(common_ids):,}")
        print("Result                    : EXACT MATCH for checked fields")

    def _build_atomic_corpus_from_bpe_checkpoint_state(self) -> None:
        """
        Recover the original pretoken byte sequences from the BPE checkpoint.

        A Pruned-BPE checkpoint's corpus contains the current BPE token IDs for
        each pretoken. Concatenating id_to_bytes for those IDs exactly recovers
        the original pretoken bytes. This lets MinGram reuse the checkpoint
        without rereading/re-pretokenizing the raw corpus.

        Identical pretokens are aggregated to (bytes, frequency), matching the
        frequency-weighted PretokenizedCorpus used by the reference code.
        """
        if self.bpe_trainer is None:
            raise RuntimeError("BPE trainer is not available")
        if not self.bpe_trainer.corpus:
            raise RuntimeError(
                "The loaded/trained BPE checkpoint has no corpus state. "
                "MinGram needs the pretoken corpus for Hard-EM."
            )

        print("\nRecovering original pretoken bytes from BPE corpus state...")

        freq_by_chunk: Counter[bytes] = Counter()
        id_to_bytes = self.bpe_trainer.id_to_bytes

        for index, ids in enumerate(self.bpe_trainer.corpus, start=1):
            try:
                chunk = b"".join(id_to_bytes[token_id] for token_id in ids)
            except KeyError as exc:
                raise RuntimeError(
                    f"BPE corpus references unknown token ID {exc.args[0]}"
                ) from exc

            if chunk:
                freq_by_chunk[chunk] += 1

            if self.progress_every and index % self.progress_every == 0:
                print(
                    f"  Recovered {index:,} pretokens; unique={len(freq_by_chunk):,}"
                )

        self.atomic_corpus = list(freq_by_chunk.items())

        total_pretokens = sum(freq for _, freq in self.atomic_corpus)
        total_atomic_bytes = sum(len(chunk) * freq for chunk, freq in self.atomic_corpus)

        print(f"Recovered pretokens        : {total_pretokens:,}")
        print(f"Unique pretokens           : {len(self.atomic_corpus):,}")
        print(f"Total UTF-8 bytes          : {total_atomic_bytes:,}")

    # ------------------------------------------------------------------
    # MinGram initialization
    # ------------------------------------------------------------------
    def _build_initial_model(self) -> MinGramModel:
        if self.bpe_trainer is None:
            raise RuntimeError("prepare_bpe_seed() must be called first")

        # PrunedBPETrainerCythonParallel computes final_token_counts from its
        # final BPE corpus. This is the project's analogue of BPEToken.current_count
        # used by script_tok when initializing MinGram.
        counts = self.bpe_trainer.final_token_counts
        if not counts:
            counts = Counter()
            for ids in self.bpe_trainer.corpus:
                counts.update(ids)

        token_ids = sorted(self.bpe_trainer.id_to_bytes)
        expected_ids = list(range(self.bpe_seed_ordinary_size))
        if token_ids != expected_ids:
            raise RuntimeError(
                "Expected ordinary BPE token IDs to be contiguous from 0 to "
                f"{self.bpe_seed_ordinary_size - 1:,}"
            )

        total_count = sum(max(1, counts[token_id]) for token_id in token_ids)

        tokens = [
            MinGramToken(
                id=token_id,
                token_bytes=self.bpe_trainer.id_to_bytes[token_id],
                log_prob=math.log(max(1, counts[token_id]) / total_count),
                required=(token_id < ATOMIC_TOKEN_COUNT),
            )
            for token_id in token_ids
        ]

        return MinGramModel(tokens, score_delta=self.score_delta)

    # ------------------------------------------------------------------
    # Default MinGram Hard-EM
    # ------------------------------------------------------------------
    def run_e_step(
        self,
        model: MinGramModel,
        corpus_atomic_length: int,
        *,
        label: str = "",
    ) -> Tuple[Dict[int, float], float, int]:
        """
        Hard-EM E-step: use the MinGram DP best path and collect token usage.

        This mirrors script_tok's run_e_step().
        """
        expected_count: Dict[int, float] = defaultdict(float)
        objective = 0.0
        total_tokens = 0

        for index, (atomic_chunk, freq) in enumerate(self.atomic_corpus, start=1):
            path = model.encode_chunk(atomic_chunk)
            path_logprob = sum(token.log_prob for token in path)

            if math.isnan(path_logprob):
                raise RuntimeError(
                    f"NaN path log-prob for pretoken {atomic_chunk!r}, freq={freq}"
                )

            for token in path:
                expected_count[token.id] += freq

            total_tokens += len(path) * freq
            objective -= path_logprob * freq

            if self.progress_every and index % self.progress_every == 0:
                prefix = f"{label}: " if label else ""
                print(
                    f"  {prefix}processed {index:,}/{len(self.atomic_corpus):,} "
                    f"unique pretokens; tokens={total_tokens:,}"
                )

        objective /= corpus_atomic_length
        return expected_count, objective, total_tokens

    def run_m_step(
        self,
        model: MinGramModel,
        expected_count: Dict[int, float],
    ) -> Tuple[MinGramModel, int]:
        """
        Remove zero-use non-required tokens and update token log-probabilities.
        """
        filtered_tokens = [
            t for t in model.tokens.values() if expected_count[t.id] > 0 or t.required
        ]

        num_removed = len(model.tokens) - len(filtered_tokens)

        total_freq = sum(expected_count[t.id] for t in filtered_tokens)

        if total_freq <= 0:
            raise RuntimeError("MinGram M-step found zero total token frequency")

        for token in filtered_tokens:
            count = expected_count[token.id]
            token.log_prob = (
                math.log(count / total_freq) if count > 0 else float("-inf")
            )

        return (
            MinGramModel(filtered_tokens, score_delta=self.score_delta),
            num_removed,
        )

    def score_based_prune(
        self,
        model: MinGramModel,
        target_size: int,
    ) -> Tuple[MinGramModel, int]:
        """
        DEFAULT MinGram pruning: retain required atomic tokens, then keep
        highest-log-probability tokens until target_size is reached.

        After the M-step, log_prob is monotonic with expected usage count, so
        this reproduces the reference implementation's usage_count pruning.
        """
        kept: Dict[int, MinGramToken] = {
            token.id: token
            for token in model.tokens.values()
            if token.required
        }

        for token in sorted(model.tokens.values(), key=lambda x: -x.log_prob):
            if token.id in kept:
                continue
            if len(kept) >= target_size:
                break
            kept[token.id] = token

        removed = len(model.tokens) - len(kept)

        return (
            MinGramModel(kept.values(), score_delta=self.score_delta),
            removed,
        )

    def train_mingram(self) -> MinGramModel:
        """
        Run the default MinGram training loop as implemented in the supplied
        script_tok trainer.py.

        Note: because pruning_shrinking_factor=0.0 in default MinGram, the
        reference prune_tokens() jumps directly to the final ordinary target
        size after each block of EM iterations. This implementation does the
        same.
        """
        if self.bpe_trainer is None or not self.atomic_corpus:
            raise RuntimeError("prepare_bpe_seed() must be called first")

        model = self._build_initial_model()
        final_vocab_size = self.final_ordinary_vocab_size

        total_pretokens = sum(freq for _, freq in self.atomic_corpus)
        corpus_atomic_length = sum(
            len(chunk) * freq for chunk, freq in self.atomic_corpus
        )

        print("\n=== Default MinGram Training ===")
        print(f"Initial BPE seed tokens    : {len(model.tokens):,}")
        print(f"Final ordinary target     : {final_vocab_size:,}")
        print(f"EM iterations per round   : {self.num_em_iterations}")
        print(f"score_delta               : {self.score_delta:g}")
        print("Prune criterion           : usage_count")

        totals_removed: Dict[str, List[int]] = defaultdict(list)
        iter_idx = 0

        start_time = time.perf_counter()

        for iter_idx in range(self.MAX_ITERATIONS):
            for sub in range(self.num_em_iterations):
                label = f"EM {iter_idx + 1}.{sub + 1}"
                print(f"\n{label}. Model size={len(model.tokens):,}")

                expected_count, objective, total_tokens = self.run_e_step(
                    model,
                    corpus_atomic_length,
                    label=label,
                )

                model, removed = self.run_m_step(model, expected_count)
                totals_removed["M Step Low Count"].append(removed)

                print(
                    f"{label} complete: tokens={total_tokens:,}, "
                    f"zero-use removed={removed:,}, "
                    f"model size={len(model.tokens):,}"
                )

            if len(model.tokens) <= final_vocab_size:
                print("\nTarget vocabulary size reached during EM.")
                break

            # Default reference config has pruning_shrinking_factor=0.0,
            # therefore prune_tokens() goes directly to final_vocab_size.
            model, num_pruned = self.score_based_prune(
                model,
                final_vocab_size,
            )
            totals_removed["Prune/Score"].append(num_pruned)

            print(
                f"\nUsage-count prune: removed={num_pruned:,}; "
                f"model size={len(model.tokens):,}"
            )

        # Match reference trainer's final score_based_prune().
        model, finalize_removed = self.score_based_prune(
            model,
            final_vocab_size,
        )
        totals_removed["Finalize"].append(finalize_removed)

        # Match reference trainer's final E-step for final statistics.
        print("\nFinal E-step...")
        expected_count, objective, total_tokens = self.run_e_step(
            model,
            corpus_atomic_length,
            label="Final E",
        )

        elapsed = time.perf_counter() - start_time

        self.model = model
        self.metadata = {
            "tokenizer_variant": "mingram",
            "source": "standalone adaptation of script_tok default MinGram",
            "objective": objective,
            "total_tokens": total_tokens,
            "tokens_per_pretoken": total_tokens / total_pretokens,
            "num_iterations": iter_idx + 1,
            "totals_removed": dict(totals_removed),
            "config": {
                "target_vocab_size": self.target_vocab_size,
                "final_ordinary_vocab_size": self.final_ordinary_vocab_size,
                "final_learned_vocab_size": self.final_learned_vocab_size,
                "overshoot_factor": self.overshoot_factor,
                "bpe_seed_ordinary_size": self.bpe_seed_ordinary_size,
                "bpe_seed_learned_size": self.bpe_seed_learned_size,
                "num_em_iterations": self.num_em_iterations,
                "score_delta": self.score_delta,
                "prune_criterion": "usage_count",
                "num_workers": self.num_workers,
            },
            "elapsed_seconds": elapsed,
        }

        print("\n=== MinGram Training Complete ===")
        print(f"Final ordinary model size : {len(model.tokens):,}")
        print(f"Final corpus token count  : {total_tokens:,}")
        print(f"Tokens / pretoken         : {self.metadata['tokens_per_pretoken']:.6f}")
        print(f"Elapsed seconds           : {elapsed:.2f}")
        print(f"Elapsed hours             : {elapsed / 3600:.4f}")

        if len(model.tokens) != final_vocab_size:
            print(
                "WARNING: final ordinary model size is not exactly the requested "
                f"{final_vocab_size:,}. This can occur if an M-step removes "
                "zero-use tokens after pruning."
            )

        return model

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    @staticmethod
    def _token_literal(token_bytes: bytes) -> str:
        # bytes repr is unambiguous and is accepted by the project's vocab
        # readers / ast.literal_eval.
        return repr(token_bytes)

    def save_vocab(
        self,
        vocab_path: str,
        *,
        metadata_path: Optional[str] = None,
    ) -> None:
        """
        Export the final MinGram token list in the project's four-column shape:

            token_id<TAB>(-1, -1)<TAB>token_literal<TAB>log_prob

        MinGram is a token-list model after pruning, not a BPE merge-tree model,
        so children are deliberately exported as (-1, -1).

        IDs are remapped densely:
            0..255             base bytes
            256..             retained learned tokens
            final 4 IDs       reserved special tokens

        The four reserved tokens are appended only for model-vocabulary parity;
        they were not involved in MinGram training.
        """
        if self.model is None:
            raise RuntimeError("train_mingram() must be called before save_vocab()")

        path = Path(vocab_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        base_tokens = [
            self.model.tokens[token_id]
            for token_id in range(ATOMIC_TOKEN_COUNT)
            if token_id in self.model.tokens
        ]

        if len(base_tokens) != ATOMIC_TOKEN_COUNT:
            raise RuntimeError("Final MinGram model is missing one or more byte tokens")

        learned_tokens = sorted(
            (
                token
                for token in self.model.tokens.values()
                if not token.required
            ),
            key=lambda token: token.id,  # deterministic export order
        )

        rows: List[Tuple[int, bytes, str]] = []

        for token_id in range(ATOMIC_TOKEN_COUNT):
            token = self.model.tokens[token_id]
            rows.append((token_id, token.token_bytes, repr(token.log_prob)))

        next_id = ATOMIC_TOKEN_COUNT
        for token in learned_tokens:
            rows.append((next_id, token.token_bytes, repr(token.log_prob)))
            next_id += 1

        for special in self.special_token_texts:
            rows.append((next_id, special.encode("utf-8"), "special"))
            next_id += 1

        with path.open("w", encoding="utf-8", newline="\n") as f:
            for token_id, token_bytes, value in rows:
                f.write(
                    f"{token_id}\t(-1, -1)\t"
                    f"{self._token_literal(token_bytes)}\t{value}\n"
                )

        print(f"\nSaved MinGram vocabulary  : {path}")
        print(f"Exported rows             : {len(rows):,}")
        print(f"Base bytes                : {ATOMIC_TOKEN_COUNT:,}")
        print(f"Learned MinGram tokens    : {len(learned_tokens):,}")
        print(f"Reserved special tokens   : {self.num_reserved:,}")

        if len(rows) != self.target_vocab_size:
            print(
                f"WARNING: exported row count {len(rows):,} != requested "
                f"target {self.target_vocab_size:,}"
            )

        if metadata_path is None:
            metadata_path = str(path.with_suffix(path.suffix + ".metadata.json"))

        metadata_file = Path(metadata_path)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"Saved MinGram metadata    : {metadata_file}")


if __name__ == "__main__":
    from settings import PROJECT_ROOT

    TARGET_VOCAB_SIZE = 16_000
    OVERSHOOT_FACTOR = 1.15
    NUM_WORKERS = 8

    REFERENCE_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "c2")
    MINGRAM_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "c2_mingram")

    # Existing Standard-BPE checkpoint trained on Corpus II.
    RESUME_BPE_CHECKPOINT = os.path.join(REFERENCE_CHECKPOINT_DIR, "checkpoint_vocab_18357.pkl")

    OUTPUT_VOCAB = os.path.join(
        PROJECT_ROOT, "experiments", "c2", "16k",
        "vocab_mingram_16k_f1.15.txt",
    )

    trainer = MinGramTrainer(
        target_vocab_size=TARGET_VOCAB_SIZE,
        overshoot_factor=OVERSHOOT_FACTOR,
        num_em_iterations=2,
        score_delta=DEFAULT_SCORE_DELTA,
        num_workers=NUM_WORKERS,
    )

    trainer.prepare_bpe_seed(
        checkpoint_path=RESUME_BPE_CHECKPOINT,
        # Save newly generated checkpoints separately.
        checkpoint_dir=MINGRAM_CHECKPOINT_DIR,
        # 16K target with f=1.15 requires a BPE seed of 18357.
        checkpoint_vocab_sizes=[18357],
    )

    trainer.train_mingram()
    trainer.save_vocab(OUTPUT_VOCAB)