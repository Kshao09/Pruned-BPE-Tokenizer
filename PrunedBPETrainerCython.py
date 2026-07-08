"""
Patch-style trainer showing only the changed train() method.
Put this beside bpe_fast_core.pyx after building the extension.
"""
import os
from math import ceil
from pathlib import Path
from collections import Counter
from typing import List, Optional

from bpe_fast_core import find_best_pair, merge_corpus
from pruned_bpe_pretokenizer import pretokenize
from PrunedBPETrainer import PrunedBPETrainer, NUM_RESERVED_TOKENS


class PrunedBPETrainerCython(PrunedBPETrainer):
    def train(
            self,
            texts: Optional[List[str]] = None,
            checkpoint_vocab_sizes: Optional[List[int]] = None,
            checkpoint_dir: Optional[str] = None,
    ) -> None:
        if checkpoint_vocab_sizes is None:
            checkpoint_vocab_sizes = []

        checkpoint_vocab_sizes = set(checkpoint_vocab_sizes)

        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Fresh training.
        if texts is not None:
            corpus = []
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
            result = find_best_pair(corpus)
            if result is None:
                break

            left_id, right_id, count = result
            if count < 2:
                break

            best_pair = (left_id, right_id)

            self.merges[best_pair] = next_id
            self.id_to_children[next_id] = best_pair
            self.id_to_bytes[next_id] = (
                    self.id_to_bytes[left_id] + self.id_to_bytes[right_id]
            )

            if not hasattr(self, "merge_counts"):
                self.merge_counts = {}
            self.merge_counts[next_id] = count

            corpus = merge_corpus(corpus, left_id, right_id, next_id)
            token_text_repr = self._format_token_text(self.id_to_bytes[next_id])
            print(
                f"Merge {next_id}: {best_pair} -> {next_id}, count={count}, token={token_text_repr}",
                flush=True
            )

            next_id += 1

            # Effective vocab size is now next_id, because IDs 0...next_id-1 exist.
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
                        f"checkpoint_vocab_{next_id}.pkl"
                    )

                self.save_checkpoint(checkpoint_path)

        self.corpus = corpus

        self.final_token_counts = Counter()
        for ids in self.corpus:
            self.final_token_counts.update(ids)