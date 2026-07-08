import os
import ast
from typing import Dict, List, Tuple

Pair = Tuple[int, int]

class BPETokenizer:
    def __init__(self, vocab_path: str):
        self.id_to_children: Dict[int, Pair] = {}
        self.id_to_bytes: Dict[int, bytes] = {
            i: bytes([i]) for i in range(256)
        }

        self.merges: Dict[Pair, int] = {}

        self._load_vocab(vocab_path)

    def _load_vocab(self, vocab_path: str) -> None:
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue

                parts = line.split("\t", maxsplit=2)

                if len(parts) != 3:
                    raise ValueError(f"Invalid vocab line: {line}")

                token_id_str, children_str, _token_text_repr = parts
                token_id = int(token_id_str)
                children = ast.literal_eval(children_str)

                if (
                    not isinstance(children, tuple)
                    or len(children) != 2
                    or not all(isinstance(x, int) for x in children)
                ):
                    raise ValueError(f"Invalid children tuple: {children_str}")

                self.id_to_children[token_id] = children

        # Rebuild bytes and merges from children.
        for token_id in sorted(self.id_to_children.keys()):
            children = self.id_to_children[token_id]

            if token_id < 256:
                self.id_to_bytes[token_id] = bytes([token_id])
                continue

            left_id, right_id = children

            if left_id not in self.id_to_bytes or right_id not in self.id_to_bytes:
                raise ValueError(
                    f"Token {token_id} depends on missing children: {children}"
                )

            self.id_to_bytes[token_id] = (
                self.id_to_bytes[left_id] + self.id_to_bytes[right_id]
            )

            self.merges[children] = token_id

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

    def encode(self, text: str) -> List[int]:
        ids = self._text_to_ids(text)

        while True:
            candidate_pairs = []

            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])

                if pair in self.merges:
                    candidate_pairs.append(pair)

            if not candidate_pairs:
                break

            # Use the earliest learned merge.
            best_pair = min(candidate_pairs, key=lambda p: self.merges[p])
            new_id = self.merges[best_pair]

            ids = self._merge_ids(ids, best_pair, new_id)

        return ids

    def decode(self, ids: List[int]) -> str:
        byte_string = b"".join(self.id_to_bytes[i] for i in ids)
        return byte_string.decode("utf-8", errors="replace")

if __name__ == "__main__":
    from settings import PROJECT_ROOT

    vocab_file = os.path.join(PROJECT_ROOT, "vocab.txt")
    tokenizer = BPETokenizer(vocab_file)

    test_texts = [
        "hello world",
        "I love machine learning.",
        "你好吗？Are you OK?",
        "我喜欢学习人工智能。",
        "hello世界, AI很有意思!",
        "这是一个非常简单的测试。",
        "THIS IS A REALLY SIMPLE TEST",
    ]

    print("\n=== Encoding / Decoding Test ===")

    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        print("\nOriginal:", text)
        print("Encoded :", encoded)
        print("Decoded :", decoded)
