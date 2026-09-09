from __future__ import annotations

"""
Standalone MinGram-PP trainer for the Pruned-BPE project.

This file builds directly on MinGramTrainer.py in the same project.

MinGram-PP keeps the same:
    - Pruned-BPE byte vocabulary and pretokenization
    - BPE-derived seed vocabulary
    - minimum-token MinGram inference with log-probability tiebreak
    - Hard-EM E/M steps
    - checkpoint-resume / deterministic BPE-seed machinery
    - final four-column vocabulary export

The difference from default MinGram is the pruning rule.

MinGram-PP uses PathPiece-style Minimum-Increase (MI) pruning:
    1. compute, for each removable token, a lower bound on the corpus
       token-count increase caused by deleting it;
    2. remove the lowest-MI tokens in a batch;
    3. resegment / rerun Hard EM;
    4. recompute MI and repeat until the target vocabulary size is reached.

The current script_tok paper implementation uses:
    pruning_shrinking_factor = 0.9
    prune_criterion          = "mi"
    num_em_iterations        = 2

Thus each pruning round removes roughly 10% of the remaining non-atomic
candidate vocabulary (subject to the final target size), with MI recomputed
on the next round.

The paper's main MinGram-PP result uses a large BPE overshoot factor f=8 and
reports saturation around f=5..8.  The example main() below intentionally
uses f=5 so the factor can be changed to 2, 3, 4, 5, or 8 for this project's
Corpus-II experiments.

Reference implementation:
    https://github.com/sanderland/script_tok/
    script_bpe/tokenizers/mingram/trainer.py
    script_bpe/tokenizers/_mi_prune.py

Paper:
    Sander Land, "MinGram: A Minimalist Unigram Tokenizer with High
    Compression and Competitive Morphological Alignment", arXiv:2606.27019v2.

Runtime project dependencies
----------------------------
MinGramPPTrainer.py directly depends on:
    MinGramTrainer.py

MinGramTrainer.py in turn depends on:
    PrunedBPETrainer.py
    PrunedBPETrainerCythonParallel.py
    pruned_bpe_pretokenizer.py

settings.py is only required by the example main().
"""
import math
import os
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from MinGramTrainer import (
    ATOMIC_TOKEN_COUNT,
    DEFAULT_SCORE_DELTA,
    MinGramModel,
    MinGramToken,
    MinGramTrainer,
    _Trie,
)


DEFAULT_PRUNING_SHRINKING_FACTOR = 0.9


def compute_mi_table(
    model: MinGramModel,
    corpus: Iterable[Tuple[bytes, int]],
    *,
    max_token_width: Optional[int] = None,
    progress_every: int = 0,
) -> Tuple[Dict[int, float], int]:
    """
    Compute PathPiece-style Minimum Increase (MI) for MinGram pruning.

    mi_by_id[token_id] is the corpus-frequency-weighted lower bound on the
    additional number of minimum-path tokens that would be required if that
    token were removed.

    Required atomic byte tokens are excluded because they may not be pruned.

    This is adapted to this project's byte-based MinGramModel:
        - token width is len(token.token_bytes)
        - trie terminal key is _Trie.TOKEN_KEY
        - corpus entries are (pretoken_bytes, frequency)

    The computation follows the two PathPiece deletion cases used by the
    current script_tok implementation:
        (1) force a break inside the selected token;
        (2) replace it through a strict superset token spanning its range.

    Tokens not used by the model's current segmentation retain MI=0 and are
    therefore natural early pruning candidates.
    """
    mi_by_id: Dict[int, float] = {
        token_id: 0.0
        for token_id, token in model.tokens.items()
        if not token.required
    }

    if max_token_width is None:
        max_token_width = max(
            (len(token.token_bytes) for token in model.tokens.values()),
            default=1,
        )

    trie_root = model.trie.root
    token_key = _Trie.TOKEN_KEY
    total_ctc = 0
    processed = 0

    for chunk, freq in corpus:
        n = len(chunk)
        if n == 0:
            continue

        processed += 1
        unreachable = n + 1

        # --------------------------------------------------------------
        # Forward minimum-token DP.
        # pl[i] = minimum number of tokens needed for chunk[:i].
        #
        # Also cache all vocabulary tokens ending at each position.  The
        # superset case below reuses this information.
        # --------------------------------------------------------------
        pl = [unreachable] * (n + 1)
        pl[0] = 0

        tokens_ending_at: List[List[Tuple[int, MinGramToken]]] = [
            [] for _ in range(n + 1)
        ]

        for start in range(n):
            base = pl[start]
            node = trie_root
            limit = min(n, start + max_token_width)

            for i in range(start, limit):
                node = node.get(chunk[i])
                if node is None:
                    break

                token = node.get(token_key)
                if token is None:
                    continue

                end = i + 1
                width = end - start
                tokens_ending_at[end].append((width, token))

                if base < unreachable and base + 1 < pl[end]:
                    pl[end] = base + 1

        k_min = pl[n]
        if k_min >= unreachable:
            raise RuntimeError(
                "No valid minimum-token path during MI computation for "
                f"chunk={chunk!r}. Required byte tokens may be missing."
            )

        total_ctc += k_min * freq

        # --------------------------------------------------------------
        # Backward minimum-token DP.
        # bpl[i] = minimum number of tokens needed for chunk[i:].
        # --------------------------------------------------------------
        bpl = [unreachable] * (n + 1)
        bpl[n] = 0

        for start in range(n - 1, -1, -1):
            node = trie_root
            limit = min(n, start + max_token_width)
            best = unreachable

            for i in range(start, limit):
                node = node.get(chunk[i])
                if node is None:
                    break

                if node.get(token_key) is not None:
                    suffix = bpl[i + 1]
                    if suffix < unreachable and suffix + 1 < best:
                        best = suffix + 1

            bpl[start] = best

        # --------------------------------------------------------------
        # MinGram-chosen minimum-token segmentation.  The path has the same
        # minimum token count k_min; its log-probability term only breaks ties.
        # --------------------------------------------------------------
        path = model.encode_chunk(chunk)

        segmentation: List[Tuple[int, int, MinGramToken]] = []
        pos = 0

        for token in path:
            width = len(token.token_bytes)
            end = pos + width
            segmentation.append((pos, end, token))
            pos = end

        if pos != n:
            raise RuntimeError(
                f"MinGram segmentation width {pos} != pretoken width {n}"
            )

        if len(path) != k_min:
            raise RuntimeError(
                "MinGram path is not a minimum-token path during MI pruning: "
                f"model_path={len(path)}, minimum={k_min}, chunk={chunk!r}"
            )

        # --------------------------------------------------------------
        # MI for every token occurrence in the chosen path.
        # --------------------------------------------------------------
        for start, end, token in segmentation:
            if token.required:
                continue

            # Case 1: force at least one boundary strictly inside this token.
            if end - start >= 2:
                best_break = unreachable

                for boundary in range(start + 1, end):
                    candidate = pl[boundary] + bpl[boundary]
                    if candidate < best_break:
                        best_break = candidate

                mi_break = best_break - k_min
            else:
                mi_break = unreachable

            # Case 2: use a strict superset vocabulary token whose span
            # contains the selected token span.
            mi_superset = unreachable
            end_limit = min(start + max_token_width, n)

            for superset_end in range(end, end_limit + 1):
                for width, _superset_token in tokens_ending_at[superset_end]:
                    superset_start = superset_end - width

                    if superset_start > start:
                        continue

                    if superset_start == start and superset_end == end:
                        continue

                    candidate = (
                        pl[superset_start]
                        + 1
                        + bpl[superset_end]
                    )

                    if candidate < mi_superset:
                        mi_superset = candidate

            mi_superset -= k_min

            local_mi = min(mi_break, mi_superset)

            # If neither replacement case exists, deleting the token would
            # force an arbitrarily costly fallback relative to this bound.
            if local_mi >= unreachable:
                local_mi = float("inf")

            mi_by_id[token.id] += freq * local_mi

        if progress_every and processed % progress_every == 0:
            print(
                f"  MI scan: {processed:,} unique pretokens; "
                f"current CTC={total_ctc:,}"
            )

    return mi_by_id, total_ctc


def select_drop_batch(
    ordered_ids: Sequence[int],
    k: int,
    tokens_by_id: Dict[int, MinGramToken],
    *,
    skip_substring: bool = False,
    max_sub_len: int = 16,
) -> Set[int]:
    """
    Select up to k token IDs from an already MI-sorted removal list.

    By default this simply returns the first k IDs, which matches the paper
    experiment configuration.

    Optional skip_substring=True mirrors the reference implementation's
    conservative within-batch heuristic: after choosing a token for removal,
    avoid also removing a short contiguous substring of that token in the same
    stale-MI batch.
    """
    if k <= 0:
        return set()

    if not skip_substring:
        return set(ordered_ids[:k])

    dropped: Set[int] = set()
    covered_substrings: Set[bytes] = set()

    for token_id in ordered_ids:
        if len(dropped) >= k:
            break

        seq = tokens_by_id[token_id].token_bytes

        if len(seq) <= max_sub_len and seq in covered_substrings:
            continue

        dropped.add(token_id)

        n = len(seq)
        for start in range(n):
            max_end = min(n, start + max_sub_len)
            for end in range(start + 1, max_end + 1):
                covered_substrings.add(seq[start:end])

    return dropped


class MinGramPPTrainer(MinGramTrainer):
    """
    MinGram with PathPiece-style iterative Minimum-Increase pruning.

    The BPE seed, Hard EM, model scoring, and export format come from the
    project's existing MinGramTrainer.  Only vocabulary pruning differs.

    Reference paper configuration:
        num_em_iterations        = 2
        pruning_shrinking_factor = 0.9
        skip_substring_in_batch  = False

    For pruning_shrinking_factor=0.9, each intermediate pruning call retains
    roughly 90% of the current non-atomic candidate pool, while never going
    below the final requested vocabulary size.  MI is recomputed after the
    following Hard-EM/resegmentation round.
    """
    def __init__(
        self,
        *,
        target_vocab_size: int = 18_000,
        overshoot_factor: float = 8.0,
        num_em_iterations: int = 2,
        score_delta: float = DEFAULT_SCORE_DELTA,
        pruning_shrinking_factor: float = DEFAULT_PRUNING_SHRINKING_FACTOR,
        skip_substring_in_batch: bool = False,
        max_substring_skip_len: int = 16,
        special_token_texts: Sequence[str] = (
            "<_EOS_>",
            "<_SOS_>",
            "<_SEP_>",
            "<_PAD_>",
        ),
        num_workers: int = 20,
        progress_every: int = 100_000,
    ):
        super().__init__(
            target_vocab_size=target_vocab_size,
            overshoot_factor=overshoot_factor,
            num_em_iterations=num_em_iterations,
            score_delta=score_delta,
            special_token_texts=special_token_texts,
            num_workers=num_workers,
            progress_every=progress_every,
        )

        self.pruning_shrinking_factor = float(pruning_shrinking_factor)
        self.skip_substring_in_batch = bool(skip_substring_in_batch)
        self.max_substring_skip_len = int(max_substring_skip_len)

        if not 0.0 <= self.pruning_shrinking_factor < 1.0:
            raise ValueError(
                "pruning_shrinking_factor must satisfy 0.0 <= p < 1.0"
            )

        if self.max_substring_skip_len < 1:
            raise ValueError("max_substring_skip_len must be >= 1")

    # ------------------------------------------------------------------
    # MinGram-PP MI pruning
    # ------------------------------------------------------------------
    def _mi_based_prune(
        self,
        model: MinGramModel,
        target_size: int,
        *,
        round_index: Optional[int] = None,
    ) -> Tuple[MinGramModel, int, int, Dict[int, float]]:
        """
        Remove the lowest-MI non-required tokens until target_size is reached.

        Ties are broken by log_prob ascending, so less-probable tokens are
        removed first when their MI values are equal, matching script_tok.
        """
        required_ids = {
            token.id
            for token in model.tokens.values()
            if token.required
        }

        removable_count = len(model.tokens) - len(required_ids)
        n_remove = min(
            max(0, len(model.tokens) - target_size),
            removable_count,
        )

        if n_remove == 0:
            return model, 0, 0, {}

        max_width = max(
            (len(token.token_bytes) for token in model.tokens.values()),
            default=1,
        )

        round_label = (
            f" round {round_index}" if round_index is not None else ""
        )

        print(
            f"\nComputing MinGram-PP MI{round_label}: "
            f"model={len(model.tokens):,}, target={target_size:,}, "
            f"remove={n_remove:,}, max_width={max_width:,}"
        )

        mi_by_id, corpus_token_count = compute_mi_table(
            model,
            self.atomic_corpus,
            max_token_width=max_width,
            progress_every=self.progress_every,
        )

        candidates = [
            token
            for token in model.tokens.values()
            if token.id not in required_ids
        ]

        # Lowest MI first.  On an MI tie, remove the lowest-probability token.
        candidates.sort(
            key=lambda token: (
                mi_by_id.get(token.id, 0.0),
                token.log_prob,
            )
        )

        ordered_ids = [token.id for token in candidates]

        to_remove = select_drop_batch(
            ordered_ids,
            n_remove,
            model.tokens,
            skip_substring=self.skip_substring_in_batch,
            max_sub_len=self.max_substring_skip_len,
        )

        if len(to_remove) < n_remove:
            raise RuntimeError(
                "MinGram-PP substring-skip rule could not select the requested "
                f"batch size: requested={n_remove:,}, selected={len(to_remove):,}"
            )

        kept = [
            token
            for token in model.tokens.values()
            if token.id not in to_remove
        ]

        new_model = MinGramModel(
            kept,
            score_delta=self.score_delta,
        )

        finite_mi = [
            value
            for token_id, value in mi_by_id.items()
            if token_id in to_remove and math.isfinite(value)
        ]
        zero_mi_removed = sum(
            1
            for token_id in to_remove
            if mi_by_id.get(token_id, 0.0) == 0.0
        )
        infinite_mi_removed = sum(
            1
            for token_id in to_remove
            if math.isinf(mi_by_id.get(token_id, 0.0))
        )

        print(
            f"MI prune{round_label}: CTC before prune={corpus_token_count:,}; "
            f"removed={len(to_remove):,}; new model={len(new_model.tokens):,}; "
            f"zero-MI removed={zero_mi_removed:,}; "
            f"infinite-MI removed={infinite_mi_removed:,}"
        )

        if finite_mi:
            print(
                f"  Removed finite MI range: "
                f"{min(finite_mi):,.0f} .. {max(finite_mi):,.0f}"
            )

        return new_model, len(to_remove), corpus_token_count, mi_by_id

    def _next_prune_target(self, model: MinGramModel) -> int:
        """
        Compute the next iterative pruning target using script_tok's schedule.

        With p=0.9 this is approximately a 10% batch, with the final ordinary
        target acting as a floor.
        """
        num_non_atomic_tokens = (len(model.tokens) - ATOMIC_TOKEN_COUNT)

        shrink_n = int(
            num_non_atomic_tokens
            * (1.0 - self.pruning_shrinking_factor)
        )

        # Guarantee progress when the model is still above target.  The
        # reference formula normally yields >=1 for the large MinGram-PP
        # candidate pools used here; this guard prevents a tiny-model stall.
        if shrink_n <= 0 and len(model.tokens) > self.final_ordinary_vocab_size:
            shrink_n = 1

        return max(
            self.final_ordinary_vocab_size,
            num_non_atomic_tokens - shrink_n,
        )

    # ------------------------------------------------------------------
    # MinGram-PP training loop
    # ------------------------------------------------------------------
    def train_mingram_pp(self) -> MinGramModel:
        """
        Train MinGram-PP.

        Each outer round:
            1. run the configured number of MinGram Hard-EM iterations;
            2. remove zero-use non-required tokens during each M-step;
            3. if still above target, compute MI and prune one iterative batch;
            4. repeat, allowing the model to resegment before the next MI scan.

        The final pruning call also uses MI, as in the current script_tok
        MinGram implementation when prune_criterion="mi".
        """
        if self.bpe_trainer is None or not self.atomic_corpus:
            raise RuntimeError("prepare_bpe_seed() must be called first")

        model = self._build_initial_model()
        final_vocab_size = self.final_ordinary_vocab_size

        total_pretokens = sum(freq for _, freq in self.atomic_corpus)
        corpus_atomic_length = sum(
            len(chunk) * freq
            for chunk, freq in self.atomic_corpus
        )

        print("\n=== MinGram-PP Training ===")
        print(f"Initial BPE seed tokens     : {len(model.tokens):,}")
        print(f"Final ordinary target      : {final_vocab_size:,}")
        print(f"External target vocabulary : {self.target_vocab_size:,}")
        print(f"Overshoot factor           : {self.overshoot_factor:g}")
        print(f"EM iterations per round    : {self.num_em_iterations}")
        print(f"score_delta                : {self.score_delta:g}")
        print("Prune criterion            : Minimum Increase (MI)")
        print(
            f"Pruning shrinking factor   : "
            f"{self.pruning_shrinking_factor:g}"
        )
        print(
            f"Skip substring in batch    : "
            f"{self.skip_substring_in_batch}"
        )

        totals_removed: Dict[str, List[int]] = defaultdict(list)
        pruning_history: List[dict] = []

        start_time = time.perf_counter()

        for iter_idx in range(self.MAX_ITERATIONS):
            # Hard EM / resegmentation before every fresh MI computation.
            for sub in range(self.num_em_iterations):
                label = f"EM {iter_idx + 1}.{sub + 1}"

                print(
                    f"\n{label}. Model size={len(model.tokens):,}"
                )

                expected_count, objective, total_tokens = self.run_e_step(
                    model,
                    corpus_atomic_length,
                    label=label,
                )

                model, removed = self.run_m_step(
                    model,
                    expected_count,
                )

                totals_removed["M Step Low Count"].append(removed)

                print(
                    f"{label} complete: tokens={total_tokens:,}, "
                    f"zero-use removed={removed:,}, "
                    f"model size={len(model.tokens):,}"
                )

            if len(model.tokens) <= final_vocab_size:
                print("\nTarget vocabulary size reached during Hard EM.")
                break

            next_target = self._next_prune_target(model)

            # The schedule should always make progress while above target.
            if next_target >= len(model.tokens):
                raise RuntimeError(
                    "MinGram-PP pruning schedule made no progress: "
                    f"model={len(model.tokens):,}, next_target={next_target:,}"
                )

            model_before = len(model.tokens)

            model, num_pruned, ctc_before, mi_by_id = self._mi_based_prune(
                model,
                next_target,
                round_index=iter_idx + 1,
            )

            totals_removed["Prune/MI"].append(num_pruned)

            pruning_history.append(
                {
                    "round": iter_idx + 1,
                    "model_size_before": model_before,
                    "target_size": next_target,
                    "removed": num_pruned,
                    "model_size_after": len(model.tokens),
                    "corpus_token_count_before_prune": ctc_before,
                    "zero_mi_candidates": sum(
                        1 for value in mi_by_id.values() if value == 0.0
                    ),
                    "infinite_mi_candidates": sum(
                        1 for value in mi_by_id.values() if math.isinf(value)
                    ),
                }
            )

        else:
            raise RuntimeError(
                f"MinGram-PP exceeded MAX_ITERATIONS={self.MAX_ITERATIONS} "
                "before reaching the final vocabulary size."
            )

        # Match the reference trainer's final score-based prune, with the
        # configured criterion being MI for MinGram-PP.
        if len(model.tokens) > final_vocab_size:
            model_before = len(model.tokens)

            model, finalize_removed, ctc_before, mi_by_id = self._mi_based_prune(
                model,
                final_vocab_size,
                round_index=iter_idx + 2,
            )

            totals_removed["Finalize/MI"].append(finalize_removed)

            pruning_history.append(
                {
                    "round": "finalize",
                    "model_size_before": model_before,
                    "target_size": final_vocab_size,
                    "removed": finalize_removed,
                    "model_size_after": len(model.tokens),
                    "corpus_token_count_before_prune": ctc_before,
                    "zero_mi_candidates": sum(
                        1 for value in mi_by_id.values() if value == 0.0
                    ),
                    "infinite_mi_candidates": sum(
                        1 for value in mi_by_id.values() if math.isinf(value)
                    ),
                }
            )

        # Final E-step: statistics and final corpus token count under the
        # completed MinGram-PP vocabulary.
        print("\nFinal E-step...")

        expected_count, objective, total_tokens = self.run_e_step(
            model,
            corpus_atomic_length,
            label="Final E",
        )

        elapsed = time.perf_counter() - start_time

        self.model = model
        self.metadata = {
            "tokenizer_variant": "mingram_pp",
            "source": (
                "standalone adaptation of script_tok MinGram with "
                "PathPiece-style Minimum-Increase pruning"
            ),
            "objective": objective,
            "total_tokens": total_tokens,
            "tokens_per_pretoken": total_tokens / total_pretokens,
            "num_iterations": iter_idx + 1,
            "totals_removed": dict(totals_removed),
            "pruning_history": pruning_history,
            "config": {
                "target_vocab_size": self.target_vocab_size,
                "final_ordinary_vocab_size": self.final_ordinary_vocab_size,
                "final_learned_vocab_size": self.final_learned_vocab_size,
                "overshoot_factor": self.overshoot_factor,
                "bpe_seed_ordinary_size": self.bpe_seed_ordinary_size,
                "bpe_seed_learned_size": self.bpe_seed_learned_size,
                "num_em_iterations": self.num_em_iterations,
                "score_delta": self.score_delta,
                "prune_criterion": "mi",
                "pruning_shrinking_factor": self.pruning_shrinking_factor,
                "skip_substring_in_batch": self.skip_substring_in_batch,
                "max_substring_skip_len": self.max_substring_skip_len,
                "num_workers": self.num_workers,
            },
            "elapsed_seconds": elapsed,
        }

        print("\n=== MinGram-PP Training Complete ===")
        print(f"Final ordinary model size : {len(model.tokens):,}")
        print(f"Final corpus token count  : {total_tokens:,}")
        print(
            f"Tokens / pretoken         : "
            f"{self.metadata['tokens_per_pretoken']:.6f}"
        )
        print(f"MI pruning rounds         : {len(pruning_history):,}")
        print(f"Elapsed seconds           : {elapsed:.2f}")
        print(f"Elapsed hours             : {elapsed / 3600:.4f}")

        if len(model.tokens) != final_vocab_size:
            print(
                "WARNING: final ordinary model size is not exactly the "
                f"requested {final_vocab_size:,}."
            )

        return model

    # Keep the familiar parent-method name safe for direct project use.
    # On a MinGramPPTrainer instance this must never invoke default MinGram.
    def train_mingram(self) -> MinGramModel:
        return self.train_mingram_pp()

    # Preserve the parent pruning API as well, but dispatch it to MI pruning.
    def score_based_prune(
        self,
        model: MinGramModel,
        target_size: int,
    ) -> Tuple[MinGramModel, int]:
        pruned, removed, _ctc, _mi = self._mi_based_prune(
            model,
            target_size,
        )
        return pruned, removed


if __name__ == "__main__":
    from settings import PROJECT_ROOT

    # ----------------------------------------------------------------------
    # Example: Corpus II, 12K final visible vocabulary, f=5.
    #
    # For this project:
    #   final learned target = 12,000 - 256 - 4 = 11,740
    #   f=5 BPE learned seed = 58,700
    #   ordinary BPE seed    = 58,956
    #
    # Change OVERSHOOT_FACTOR to 2.0 / 3.0 / 4.0 / 5.0 as needed.
    # The supplied checkpoint must be at or below the resulting BPE seed size.
    # ----------------------------------------------------------------------
    TARGET_VOCAB_SIZE = 12_000
    TARGET_VOCAB_STR = "12k"
    OVERSHOOT_FACTOR = 5.0
    NUM_WORKERS = 12

    REFERENCE_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "c2")

    MINGRAM_PP_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "c2_mingram_pp")

    # Update this filename to the latest compatible Standard-BPE checkpoint.
    RESUME_BPE_CHECKPOINT = os.path.join(REFERENCE_CHECKPOINT_DIR, "checkpoint_vocab_58956.pkl")

    factor_text = f"{OVERSHOOT_FACTOR:g}"

    OUTPUT_VOCAB = os.path.join(
        PROJECT_ROOT, "experiments", "c2",
        TARGET_VOCAB_STR,
        f"vocab_mingram_pp_{TARGET_VOCAB_STR}_f{factor_text}.txt",
    )

    trainer = MinGramPPTrainer(
        target_vocab_size=TARGET_VOCAB_SIZE,
        overshoot_factor=OVERSHOOT_FACTOR,
        num_em_iterations=2,
        score_delta=DEFAULT_SCORE_DELTA,
        pruning_shrinking_factor=0.9,
        skip_substring_in_batch=False,
        num_workers=NUM_WORKERS,
    )

    trainer.prepare_bpe_seed(
        checkpoint_path=RESUME_BPE_CHECKPOINT,
        checkpoint_dir=MINGRAM_PP_CHECKPOINT_DIR,
        checkpoint_vocab_sizes=[trainer.bpe_seed_ordinary_size],
    )

    trainer.train_mingram_pp()
    trainer.save_vocab(OUTPUT_VOCAB)
