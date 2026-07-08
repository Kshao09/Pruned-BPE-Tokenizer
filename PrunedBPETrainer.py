import os
import math
import pickle
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from collections import Counter
from itertools import pairwise
from typing import Dict, List, Tuple, Optional

from pruned_bpe_pretokenizer import pretokenize

Pair = Tuple[int, int]
RESERVED_TOKENS = ["<_EOS_>", "<_SOS_>", "<_SEP_>", "<_PAD_>"]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)

class PrunedBPETrainer:
    def __init__(
        self,
        train_vocab_size: int = 1024,
        visible_vocab_size: int = 1024,
        min_exposure_count: int = 2,
    ):
        """
        Create a Pruned BPE trainer.

        train_vocab_size:
            Number of tokens to train in the original BPE training ID space.
            This includes the first 256 byte tokens, but excludes reserved
            special tokens. For example, if you want a final file-level
            vocabulary budget of 1000 entries including 4 reserved special
            tokens, pass 996 here, not 1000.

        visible_vocab_size:
            Number of model-visible tokens to export before reserved special
            tokens are appended. This includes the first 256 byte tokens, but
            excludes reserved special tokens. For example, if the model-visible
            vocabulary budget is 1000 entries including 4 reserved special
            tokens, pass 996 here, not 1000.

        min_exposure_count:
            Minimum final-corpus exposure count required for a learned token to
            be exported as model-visible. Lower-exposure learned tokens are
            kept only as internal construction tokens, unless they are trained
            after the visible vocabulary is already full; those later tokens are
            discarded completely.
        """
        if visible_vocab_size < 256:
            raise ValueError("visible_vocab_size must be at least 256")
        if train_vocab_size < visible_vocab_size:
            raise ValueError("train_vocab_size must be >= visible_vocab_size")
        # if min_exposure_count < 1:
        #     raise ValueError("min_exposure_count must be at least 1")

        # Both sizes include the 256 base byte tokens and exclude reserved
        # special tokens. Subtract NUM_RESERVED_TOKENS before passing these
        # values if your external target vocabulary size includes specials.
        self.train_vocab_size = train_vocab_size
        self.visible_vocab_size = visible_vocab_size
        self.min_exposure_count = min_exposure_count

        # Benchmark timing fields.
        self.benchmark_start_perf: Optional[float] = None
        self.benchmark_last_perf: Optional[float] = None

        # Original training-space structures.
        # pair -> original new token id
        self.merges: Dict[Pair, int] = {}

        # original token id -> raw bytes represented by that token
        self.id_to_bytes: Dict[int, bytes] = {
            i: bytes([i]) for i in range(256)
        }

        # original token id -> original children token ids
        self.id_to_children: Dict[int, Pair] = {
            i: (-1, -1) for i in range(256)
        }

        # Final corpus after all training merges.
        self.corpus: List[List[int]] = []

        # Final exposed-token counts from self.corpus.
        self.final_token_counts: Counter[int] = Counter()

        # old token id -> new remapped token id
        self.old_to_new_id: Dict[int, int] = {}

        # Special token text -> special token id
        self.special_token_ids: Dict[str, int] = {}

    def _text_to_ids(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def _merge_ids(self, ids: List[int], pair: Pair, new_id: int) -> List[int]:
        result = []
        i = 0

        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1

        return result

    def _next_training_id(self) -> int:
        """
        Return the next original training-space token ID.

        This is used for both fresh training and resumed training.
        Base byte tokens occupy IDs 0..255, so the first learned BPE token is 256.
        """
        learned_ids = [
            token_id for token_id in self.id_to_bytes.keys()
            if token_id >= 256
        ]

        if not learned_ids:
            return 256

        return max(learned_ids) + 1

    def mark_benchmark_start(self) -> None:
        """
        Record the benchmark start time.

        Call this right after load_data() finishes, before train().
        """
        now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        now_perf = time.perf_counter()

        self.benchmark_start_perf = now_perf
        self.benchmark_last_perf = now_perf

        print(f"## Benchmark start time: {now_text}")

    def train(
        self,
        texts: Optional[List[str]] = None,
        checkpoint_vocab_sizes: Optional[List[int]] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        """
        Train or resume training.

        Fresh training:
            pass texts.

        Resume training:
            load a checkpoint first, then call train(texts=None).

        checkpoint_vocab_sizes:
            Effective vocabulary sizes at which to save checkpoints.
            Example:
                If next_id becomes 12000 after creating token 11999,
                then the effective training vocabulary size is 12000.
                So checkpoint_vocab_sizes should contain 12000.

        checkpoint_dir:
            Folder where checkpoint files are written. If None, checkpoint
            files are written to the current working directory.
        """
        if checkpoint_vocab_sizes is None:
            checkpoint_vocab_sizes = []

        checkpoint_vocab_sizes = set(checkpoint_vocab_sizes)
        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Fresh training.
        if texts is not None:
            corpus: List[List[int]] = []

            for text in texts:
                if not text:
                    continue

                for chunk in pretokenize(text):
                    if chunk:
                        corpus.append(self._text_to_ids(chunk))
        # Resume training.
        else:
            if not self.corpus:
                raise ValueError("texts is None, but no checkpoint corpus exists.")

            corpus = self.corpus

        next_id = self._next_training_id()

        while next_id < self.train_vocab_size:
            pair_counts = Counter()
            for ids in corpus:
                pair_counts.update(pairwise(ids))

            if not pair_counts:
                break

            best_pair, count = pair_counts.most_common(1)[0]
            if count < 2:
                break

            left_id, right_id = best_pair

            self.merges[best_pair] = next_id
            self.id_to_children[next_id] = best_pair
            self.id_to_bytes[next_id] = (
                self.id_to_bytes[left_id] + self.id_to_bytes[right_id]
            )

            # Keep this field for compatibility with the Cython trainers.
            # The pure Python trainer can also use it.
            if not hasattr(self, "merge_counts"):
                self.merge_counts = {}
            self.merge_counts[next_id] = count

            corpus = [
                self._merge_ids(ids, best_pair, next_id)
                for ids in corpus
            ]

            token_text_repr = self._format_token_text(self.id_to_bytes[next_id])
            print(
                f"Merge {next_id}: {best_pair} -> {next_id}, "
                f"count={count}, token={token_text_repr}",
                flush=True,
            )

            tokens_generated = next_id - 255
            if tokens_generated % 10 == 0:
                now_perf = time.perf_counter()
                now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if self.benchmark_start_perf is None:
                    self.benchmark_start_perf = now_perf

                if self.benchmark_last_perf is None:
                    self.benchmark_last_perf = now_perf

                total_elapsed = now_perf - self.benchmark_start_perf
                elapsed_since_last = now_perf - self.benchmark_last_perf

                tokens_per_hour_last_10 = (
                    10 * 3600 / elapsed_since_last
                    if elapsed_since_last > 0
                    else 0.0
                )

                print(
                    f"[Benchmark] time={now_text}, "
                    f"latest_token={next_id}, "
                    f"tokens_generated={tokens_generated}, "
                    f"total_elapsed_sec={total_elapsed:.2f}, "
                    f"last_10_tokens_sec={elapsed_since_last:.2f}, "
                    f"tokens_per_hour_last_10={tokens_per_hour_last_10:.2f}",
                    flush=True,
                )

                self.benchmark_last_perf = now_perf

            next_id += 1

            # Effective vocab size is now next_id, because IDs 0..next_id-1 exist.
            if next_id in checkpoint_vocab_sizes:
                self.corpus = corpus

                self.final_token_counts = Counter()
                for ids in self.corpus:
                    self.final_token_counts.update(ids)

                if checkpoint_dir is None:
                    checkpoint_path = f"checkpoint_vocab_{next_id}.pkl"
                else:
                    checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f"checkpoint_vocab_{next_id}.pkl",
                    )

                self.save_checkpoint(checkpoint_path)

        self.corpus = corpus

        self.final_token_counts = Counter()
        for ids in self.corpus:
            self.final_token_counts.update(ids)

    def _recompute_final_token_counts(self) -> None:
        """
        Recompute final token exposure counts from the current trained corpus.
        """
        self.final_token_counts = Counter()

        for ids in self.corpus:
            self.final_token_counts.update(ids)

    def count_exportable_visible_learned_tokens(self) -> int:
        """
        Count how many learned tokens currently qualify as model-visible.
        Base byte tokens 0..255 are not counted here.

        Target visible learned tokens:
            visible_vocab_size - 256
        because visible_vocab_size includes the 256 base byte tokens but
        excludes reserved special tokens.
        """
        target_visible_learned = self.visible_vocab_size - 256
        visible_count = 0
        learned_old_ids = [
            old_id
            for old_id in sorted(self.id_to_bytes.keys())
            if old_id >= 256
        ]

        for old_id in learned_old_ids:
            exposure_count = self.final_token_counts.get(old_id, 0)
            if exposure_count >= self.min_exposure_count:
                visible_count += 1
                if visible_count >= target_visible_learned:
                    break

        return visible_count

    def train_until_visible_vocab_full(
        self,
        max_train_vocab_size: int,
        checkpoint_vocab_sizes: Optional[List[int]] = None,
        checkpoint_dir: Optional[str] = None,
        step_size: int = 25,
        **train_kwargs,
    ) -> None:
        """
        Continue training until enough sufficiently exposed learned tokens exist
        to fill the requested visible vocabulary.

        This method intentionally calls self.train(...), instead of containing
        its own training loop.

        Therefore:
            - In PrunedBPETrainer, it uses the pure Python train().
            - In PrunedBPETrainerCython, it uses the Cython train().
            - In PrunedBPETrainerCythonParallel, it uses the parallel Cython train().

        A final checkpoint is always saved before this method returns.
        """
        if checkpoint_vocab_sizes is None:
            checkpoint_vocab_sizes = []

        if step_size < 1:
            raise ValueError("step_size must be at least 1")

        if not self.corpus:
            raise ValueError(
                "No existing corpus state found. "
                "Use train(texts=...) first or load a checkpoint."
            )

        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        current_next_id = self._next_training_id()
        if max_train_vocab_size <= current_next_id:
            raise ValueError(
                f"max_train_vocab_size={max_train_vocab_size} must be larger "
                f"than current next_id={current_next_id}."
            )

        def save_final_visible_fill_checkpoint() -> None:
            actual_vocab_size = self._next_training_id()
            self._recompute_final_token_counts()
            self.train_vocab_size = actual_vocab_size

            if checkpoint_dir is None:
                checkpoint_path = f"checkpoint_vocab_{actual_vocab_size}.pkl"
            else:
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"checkpoint_vocab_{actual_vocab_size}.pkl",
                )

            print(
                f"\n[Visible-fill training] saving final checkpoint: "
                f"{checkpoint_path}",
                flush=True,
            )

            self.save_checkpoint(checkpoint_path)

        target_visible_learned = self.visible_vocab_size - 256
        self._recompute_final_token_counts()
        visible_learned = self.count_exportable_visible_learned_tokens()

        print(
            f"\n[Visible-fill training] initial visible learned tokens: "
            f"{visible_learned}/{target_visible_learned}",
            flush=True,
        )

        if visible_learned >= target_visible_learned:
            print(
                "[Visible-fill training] visible vocabulary is already full.",
                flush=True,
            )
            save_final_visible_fill_checkpoint()
            return

        while visible_learned < target_visible_learned:
            current_next_id = self._next_training_id()
            if current_next_id >= max_train_vocab_size:
                print(
                    f"\n[Visible-fill training] reached max_train_vocab_size="
                    f"{max_train_vocab_size} before visible vocabulary was full.",
                    flush=True,
                )
                break

            missing_visible = target_visible_learned - visible_learned
            extension_size = max(step_size, math.ceil(1.1 * missing_visible))
            next_target = min(current_next_id + extension_size, max_train_vocab_size)

            print(
                f"\n[Visible-fill training] extending training: "
                f"{current_next_id} -> {next_target}",
                flush=True,
            )

            self.train_vocab_size = next_target

            self.train(
                texts=None,
                checkpoint_vocab_sizes=checkpoint_vocab_sizes,
                checkpoint_dir=checkpoint_dir,
                **train_kwargs,
            )

            self.train_vocab_size = self._next_training_id()
            self._recompute_final_token_counts()
            visible_learned = self.count_exportable_visible_learned_tokens()

            print(
                f"[Visible-fill training] visible learned tokens: "
                f"{visible_learned}/{target_visible_learned}; "
                f"actual train_vocab_size={self._next_training_id()}",
                flush=True,
            )

            if self._next_training_id() == current_next_id:
                print(
                    "[Visible-fill training] no new token was trained. "
                    "Stopping to avoid an infinite loop.",
                    flush=True,
                )
                break

        self.train_vocab_size = self._next_training_id()
        self._recompute_final_token_counts()
        final_visible_learned = self.count_exportable_visible_learned_tokens()

        print(f"\n[Visible-fill training] finished.", flush=True)
        print(
            f"[Visible-fill training] actual train_vocab_size="
            f"{self.train_vocab_size}",
            flush=True,
        )
        print(
            f"[Visible-fill training] visible learned tokens="
            f"{final_visible_learned}/{target_visible_learned}",
            flush=True,
        )

        save_final_visible_fill_checkpoint()

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save the current training state.

        Important:
            train_vocab_size, visible_vocab_size, and min_exposure_count are
            saved only as metadata. They are not required restore state.

            When loading a checkpoint, the caller must explicitly provide the
            current experiment settings.
        """
        checkpoint = {
            # Required restore state.
            "merges": self.merges,
            "id_to_bytes": self.id_to_bytes,
            "id_to_children": self.id_to_children,
            "corpus": self.corpus,
            "final_token_counts": self.final_token_counts,

            # Optional but useful for reporting and analysis.
            "merge_counts": getattr(self, "merge_counts", {}),

            # Informational. The loader does not trust this as state.
            # It recomputes next_id from id_to_bytes.
            "next_id": self._next_training_id(),

            # Metadata only.
            # These values describe how the checkpoint was saved, but they do
            # not control how it is restored.
            "metadata": {
                "train_vocab_size_at_save_time": self.train_vocab_size,
                "visible_vocab_size_at_save_time": self.visible_vocab_size,
                "min_exposure_count_at_save_time": self.min_exposure_count,
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

        temp_path = checkpoint_path + ".tmp"

        with open(temp_path, "wb") as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

        os.replace(temp_path, checkpoint_path)

        print(f"\n[Checkpoint] saved to {checkpoint_path}", flush=True)
        print(f"[Checkpoint] next_id={checkpoint['next_id']}", flush=True)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        train_vocab_size: int,
        visible_vocab_size: int,
        min_exposure_count: int,
    ):
        """
        Load a checkpoint using current experiment settings.

        The checkpoint restores only the BPE training state.

        The following values must come from the current script/config:
            train_vocab_size
            visible_vocab_size
            min_exposure_count

        This avoids accidental reuse of old experiment settings saved inside
        the checkpoint.
        """
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        trainer = cls(
            train_vocab_size=train_vocab_size,
            visible_vocab_size=visible_vocab_size,
            min_exposure_count=min_exposure_count,
        )

        trainer.merges = checkpoint["merges"]
        trainer.id_to_bytes = checkpoint["id_to_bytes"]
        trainer.id_to_children = checkpoint["id_to_children"]
        trainer.corpus = checkpoint["corpus"]
        trainer.final_token_counts = checkpoint["final_token_counts"]
        trainer.merge_counts = checkpoint.get("merge_counts", {})

        print(f"[Checkpoint] loaded from {checkpoint_path}", flush=True)
        print(f"[Checkpoint] next_id={trainer._next_training_id()}", flush=True)
        print(f"[Checkpoint] train_vocab_size={trainer.train_vocab_size}", flush=True)
        print(f"[Checkpoint] visible_vocab_size={trainer.visible_vocab_size}", flush=True)
        print(f"[Checkpoint] min_exposure_count={trainer.min_exposure_count}", flush=True)

        metadata = checkpoint.get("metadata")
        if metadata:
            print(f"[Checkpoint] metadata={metadata}", flush=True)

        return trainer

    def _build_id_remap(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Build ID remapping after training.

        There are two ID spaces:

        1. Training IDs
            These are the original BPE IDs created during training. They are
            monotonic by merge order: 256 is the first learned merge, 257 is
            the second learned merge, and so on.

        2. Exported IDs
            These are the IDs written to vocab.txt and inter_vocab.txt.

        Exported model-visible IDs:
            0..255: byte tokens
            256..visible_vocab_size-1: sufficiently exposed learned tokens
            visible_vocab_size..visible_vocab_size+NUM_RESERVED_TOKENS-1:
                reserved special tokens

        Exported internal-only IDs:
            start after the visible tokens and reserved special tokens.

        Learned tokens are scanned in merge order. Once visible_vocab_size is
        full, all remaining trained tokens are discarded. They are not written
        to vocab.txt or inter_vocab.txt, because later merge tokens cannot be
        needed to construct earlier exported tokens.
        """
        if not self.id_to_bytes:
            raise ValueError("No vocabulary exists. Did you call train() first?")

        learned_old_ids = [
            old_id for old_id in sorted(self.id_to_bytes.keys())
            if old_id >= 256
        ]

        visible_learned_old_ids: List[int] = []
        internal_old_ids: List[int] = []
        discarded_old_ids: List[int] = []

        old_to_new: Dict[int, int] = {}

        # Base byte tokens keep their IDs.
        for old_id in range(256):
            old_to_new[old_id] = old_id

        # First pass over learned tokens in merge order.
        # Low-exposure tokens before the visible vocabulary is full are kept as
        # internal construction tokens, because later visible tokens may depend
        # on them. Once the visible vocabulary is full, all remaining trained
        # tokens are extra candidates and are discarded.
        next_visible_id = 256
        for old_id in learned_old_ids:
            if next_visible_id >= self.visible_vocab_size:
                discarded_old_ids.append(old_id)
                continue

            exposure_count = self.final_token_counts.get(old_id, 0)
            if exposure_count >= self.min_exposure_count:
                visible_learned_old_ids.append(old_id)
                old_to_new[old_id] = next_visible_id
                next_visible_id += 1
            else:
                internal_old_ids.append(old_id)

        # Reserved special token IDs are fixed immediately after the actual
        # visible learned token IDs produced so far.
        self.special_token_ids = {}
        next_special_id = next_visible_id
        for token_text in RESERVED_TOKENS:
            self.special_token_ids[token_text] = next_special_id
            next_special_id += 1

        # Internal-only tokens start after special token IDs. Their children are
        # remapped through old_to_new_id when the files are written.
        next_internal_id = next_special_id
        for old_id in internal_old_ids:
            old_to_new[old_id] = next_internal_id
            next_internal_id += 1

        self.old_to_new_id = old_to_new
        return visible_learned_old_ids, internal_old_ids, discarded_old_ids

    def _remap_children(self, old_token_id: int) -> Pair:
        old_children = self.id_to_children[old_token_id]

        if old_children == (-1, -1):
            return (-1, -1)

        left_old, right_old = old_children
        return (self.old_to_new_id[left_old], self.old_to_new_id[right_old])

    def save_vocab(self, vocab_path: str, inter_vocab_path: str) -> None:
        """
        Save two files:

        vocab.txt:
            model-visible tokens only:
            - base byte tokens 0..255
            - sufficiently exposed learned tokens
            - reserved special tokens

        inter_vocab.txt:
            internal-only/intermediate tokens

        4-column tab-separated format:
            token_id    children_tuple    token_text    merge_rank

        token_id:
            Exported/remapped token ID used by the tokenizer/model.
        merge_rank:
            Original training-space token ID.
            This preserves the true BPE merge order after token reallocation.

            For base byte tokens and reserved special tokens, merge_rank = -1.
            For learned BPE tokens, merge_rank = old_id.

        Important:
            The tokenizer must use merge_rank for choosing the next BPE merge,
            not token_id.
        """
        visible_learned_old_ids, internal_old_ids, discarded_old_ids = (
            self._build_id_remap()
        )

        # Entry format:
        #   exported_token_id, remapped_children, token_bytes, merge_rank
        vocab_entries: List[Tuple[int, Pair, bytes, int]] = []
        inter_entries: List[Tuple[int, Pair, bytes, int]] = []

        # Base byte tokens.
        # These are not learned BPE merges, so merge_rank = -1.
        for old_id in range(256):
            new_id = self.old_to_new_id[old_id]
            vocab_entries.append(
                (new_id, (-1, -1), self.id_to_bytes[old_id], -1)
            )

        # Visible learned BPE tokens.
        # merge_rank must remain the original training-space old_id.
        for old_id in visible_learned_old_ids:
            new_id = self.old_to_new_id[old_id]
            children = self._remap_children(old_id)
            vocab_entries.append(
                (new_id, children, self.id_to_bytes[old_id], old_id)
            )

        # Reserved special tokens.
        # These are not BPE merges, so merge_rank = -1.
        for token_text, token_id in self.special_token_ids.items():
            token_bytes = token_text.encode("utf-8")
            vocab_entries.append(
                (token_id, (-1, -1), token_bytes, -1)
            )

        # Internal-only learned BPE tokens.
        # These are still real BPE merges, so preserve original old_id as rank.
        for old_id in sorted(internal_old_ids):
            new_id = self.old_to_new_id[old_id]
            children = self._remap_children(old_id)
            inter_entries.append(
                (new_id, children, self.id_to_bytes[old_id], old_id)
            )

        self._write_vocab_file(vocab_path, vocab_entries)
        self._write_vocab_file(inter_vocab_path, inter_entries)

        print(f"\nSaved model-visible vocab to {vocab_path}")
        print(f"Saved internal-only vocab to {inter_vocab_path}")
        print(f"Visible learned tokens: {len(visible_learned_old_ids)}")
        print(f"Reserved special tokens: {len(self.special_token_ids)}")
        print(f"Internal-only tokens: {len(internal_old_ids)}")
        print(f"Discarded extra trained tokens: {len(discarded_old_ids)}")

    @staticmethod
    def _format_token_text(token_bytes: bytes) -> str:
        """
        Show readable UTF-8 text when possible.
        Otherwise fall back to bytes repr.

        The result is still written using repr(...), so the tokenizer can
        safely load it back with ast.literal_eval().
        """
        try:
            token_text = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return repr(token_bytes)

        # If decoded text contains control characters, keep bytes form.
        # This avoids making vocab.txt confusing for newline/tab/control bytes.
        if any(ord(ch) < 32 for ch in token_text):
            return repr(token_bytes)

        return repr(token_text)

    @staticmethod
    def _write_vocab_file(path: str, entries: List[Tuple[int, Pair, bytes, int]]) -> None:
        """
        Write vocab/inter_vocab file in 4-column format:
            token_id    children_tuple    token_text    merge_rank
        The file is sorted by exported token_id for readability and stable IDs.
        The tokenizer must use merge_rank, not token_id, for merge priority.
        """
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for token_id, children, token_bytes, merge_rank in sorted(
                entries,
                key=lambda x: x[0],
            ):
                token_text_repr = PrunedBPETrainer._format_token_text(token_bytes)
                f.write(
                    f"{token_id}\t{children}\t{token_text_repr}\t{merge_rank}\n"
                )

    @staticmethod
    def load_data(folder_path: str) -> List[str]:
        """
        Load all UTF-8 text files from a folder.

        For .txt and .log:
            read line by line and rstrip only '\\r' and '\\n'

        For other text files such as .java, .py, .json, .xml:
            keep all characters exactly, including spaces, '\\r', and '\\n'

        Binary / non-UTF-8 files are skipped.
        """
        texts: List[str] = []

        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder does not exist: {folder_path}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Expected a folder: {folder_path}")

        for file_path in sorted(folder.rglob("*")):
            if not file_path.is_file():
                continue

            try:
                # Supported input encoding UTF-8 only.
                # If your input files use other encodings, convert them to UTF-8 first.
                text = file_path.read_text(encoding="utf-8")

                if text.startswith("\ufeff"):
                    text = text[1:]

                text = unicodedata.normalize("NFC", text)
            except UnicodeDecodeError:
                print(f"Skipping non-UTF-8 or binary file: {file_path}")
                continue

            print(f"Loading data from file: {file_path}")

            suffix = file_path.suffix.lower()

            # Feel free to change the data processing logic here.
            # For example, you can parse .java and .js files line by line here.
            if suffix in {".txt", ".log", ".html", ".htm"}:
                lines = text.splitlines(keepends=True)
                for line in lines:
                    line = line.rstrip("\r\n")
                    if line and line != "===":
                        texts.append(line)
            else:
                texts.append(text)

        return texts