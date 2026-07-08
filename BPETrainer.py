import os
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple


Pair = Tuple[int, int]


class BPETrainer:
    def __init__(self, vocab_size: int = 1024):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        self.vocab_size = vocab_size

        # pair -> new token id
        self.merges: Dict[Pair, int] = {}

        # token id -> raw bytes represented by that token
        self.id_to_bytes: Dict[int, bytes] = {
            i: bytes([i]) for i in range(256)
        }

        # token id -> children token ids
        # base byte tokens do not have children, so use (-1, -1)
        self.id_to_children: Dict[int, Pair] = {
            i: (-1, -1) for i in range(256)
        }

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

    def train(self, texts: List[str]) -> None:
        corpus: List[List[int]] = []

        for text in texts:
            if text:
                corpus.append(self._text_to_ids(text))

        next_id = 256

        while next_id < self.vocab_size:
            pair_counts = Counter()

            for ids in corpus:
                for i in range(len(ids) - 1):
                    pair_counts[(ids[i], ids[i + 1])] += 1

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

            corpus = [
                self._merge_ids(ids, best_pair, next_id)
                for ids in corpus
            ]

            print(f"Merge {next_id}: {best_pair} -> {next_id}, count={count}")

            next_id += 1

    def save_vocab(self, path: str) -> None:
        """
        Save vocab.txt with 3 tab-separated columns:

        token_id    children_tuple    token_text

        Example:
        256    (104, 101)    'he'
        """
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for token_id in sorted(self.id_to_bytes.keys()):
                raw_bytes = self.id_to_bytes[token_id]
                children = self.id_to_children[token_id]

                token_text = raw_bytes.decode("utf-8", errors="replace")

                f.write(
                    f"{token_id}\t{children}\t{repr(token_text)}\n"
                )

    @staticmethod
    def load_data(folder_path: str) -> List[str]:
        """
        Load all text files from a folder.

        For .txt and .log:
            read line by line and rstrip only '\\r' and '\\n'

        For other text files such as .java, .py, .json, .xml:
            keep all characters exactly, including spaces, '\\r', and '\\n'

        Binary / non-UTF-8 files are skipped.
        """
        texts: List[str] = []

        folder = Path(folder_path)
        for file_path in sorted(folder.rglob("*")):
            if not file_path.is_file():
                continue

            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                print(f"Skipping non-UTF-8 or binary file: {file_path}")
                continue

            print(f"Loading data from file: {file_path}")

            suffix = file_path.suffix.lower()
            if suffix in {".txt", ".log"}:
                lines = text.splitlines(keepends=True)

                for line in lines:
                    line = line.rstrip("\r\n")
                    if line and line != "===":
                        texts.append(line)
            else:
                texts.append(text)

        return texts

if __name__ == "__main__":
    from settings import PROJECT_ROOT

    corpus_dir = os.path.join(PROJECT_ROOT, "Corpus")
    vocab_path = os.path.join(PROJECT_ROOT, "vocab.txt")

    texts = BPETrainer.load_data(corpus_dir)

    trainer = BPETrainer(vocab_size=1024)
    trainer.train(texts)
    trainer.save_vocab(vocab_path)

    print(f"\nSaved vocab to {vocab_path}")