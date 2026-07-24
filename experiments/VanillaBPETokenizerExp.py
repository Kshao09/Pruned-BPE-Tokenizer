import os
import ast
from typing import Dict, List

from BPETokenizer import BPETokenizer
from PrunedBPETrainer import PrunedBPETrainer
from pruned_bpe_pretokenizer import pretokenize


class VanillaBPETokenizerExp(BPETokenizer):
    """
    Experiment wrapper for standard / vanilla BPE tokenization.
    This class subclasses BPETokenizer, but overrides _load_vocab() and encode().
    Why override _load_vocab()?
        The vocab.txt generated from PrunedBPETrainer may include special tokens
        such as <_EOS_>, <_SOS_>, <_SEP_>, and <_PAD_>. These tokens have
        children (-1, -1) and token IDs >= 256.

        The original BPETokenizer assumes that every token ID >= 256 must be a
        normal merge token with valid children. Therefore, it fails on special
        tokens. For vanilla BPE experiments, these special tokens should not be
        used as BPE merges, so this loader keeps their bytes for decoding but
        does not add them to self.merges.
    Why override encode()?
        The original BPETokenizer.encode() applies BPE to the whole raw string.
        For the experiment, vanilla BPE should use the same pretokenization
        boundary as Pruned BPE, because both vocabularies come from the same
        PrunedBPETrainer training process.
    """
    def __init__(self, vocab_path: str):
        super().__init__(vocab_path)

    def _load_vocab(self, vocab_path: str) -> None:
        """
        Load vocab.txt.
        This version supports three token types:
        1. Base byte tokens:
            token_id < 256 and children == (-1, -1)
        2. Normal BPE merge tokens:
            token_id >= 256 and children != (-1, -1)
        3. Special tokens:
            token_id >= 256 and children == (-1, -1)
        Special tokens are loaded into id_to_bytes so decode() can still work,
        but they are not added to merges.
        """
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue

                parts = line.split("\t")

                if len(parts) < 3:  # can be 3 or 4 (if saved from Pruned BPE Trainer)
                    raise ValueError(f"Invalid vocab line: {line}")

                token_id_str, children_str, token_text_repr = parts[0], parts[1], parts[2]

                token_id = int(token_id_str)
                children = ast.literal_eval(children_str)
                token_text_or_bytes = ast.literal_eval(token_text_repr)

                if (
                    not isinstance(children, tuple)
                    or len(children) != 2
                    or not all(isinstance(x, int) for x in children)
                ):
                    raise ValueError(f"Invalid children tuple: {children_str}")

                if isinstance(token_text_or_bytes, bytes):
                    token_bytes = token_text_or_bytes
                elif isinstance(token_text_or_bytes, str):
                    token_bytes = token_text_or_bytes.encode("utf-8")
                else:
                    raise ValueError(
                        f"Invalid token text/bytes literal: {token_text_repr}"
                    )

                self.id_to_children[token_id] = children
                self.id_to_bytes[token_id] = token_bytes

        # Rebuild bytes and merges from children.
        for token_id in sorted(self.id_to_children.keys()):
            children = self.id_to_children[token_id]

            # Base byte tokens.
            if token_id < 256:
                self.id_to_bytes[token_id] = bytes([token_id])
                continue

            # Special tokens such as <_EOS_>, <_SOS_>, <_SEP_>, <_PAD_>.
            # They are not BPE merge rules.
            if children == (-1, -1):
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

    def _encode_pretokenized_chunk(self, chunk: str) -> List[int]:
        """
        Encode one pretokenized chunk using standard byte-level BPE.

        This is the same basic logic as BPETokenizer.encode(), but applied only
        to one chunk produced by pretokenize(...).
        """
        ids = self._text_to_ids(chunk)

        while True:
            candidate_pairs = []

            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])

                if pair in self.merges:
                    candidate_pairs.append(pair)

            if not candidate_pairs:
                break

            # Smaller token ID means earlier learned merge.
            best_pair = min(candidate_pairs, key=lambda p: self.merges[p])
            new_id = self.merges[best_pair]

            ids = self._merge_ids(ids, best_pair, new_id)

        return ids

    def encode(self, text: str) -> List[int]:
        """
        Encode text into vanilla BPE token IDs using the same pretokenization
        boundary as Pruned BPE.
        """
        result: List[int] = []

        for chunk in pretokenize(text):
            if not chunk:
                continue

            result.extend(self._encode_pretokenized_chunk(chunk))

        return result

    def tokenize_dataset(self, folder_paths: List[str]) -> Dict[str, int]:
        """
        Tokenize all text loaded from the given dataset folders and return counts.

        The encoded token lists are not stored. Only their lengths are accumulated.
        """
        total_documents = 0
        total_final_tokens = 0

        for folder_index, folder_path in enumerate(folder_paths, start=1):
            print(f"Loading folder {folder_index}/{len(folder_paths)}: {folder_path}")

            texts = PrunedBPETrainer.load_data(folder_path)

            print(f"Loaded {len(texts)} text records.")

            for text_index, text in enumerate(texts, start=1):
                encoded = self.encode(text)

                total_final_tokens += len(encoded)
                total_documents += 1

                if text_index % 100_000 == 0:
                    print(
                        f"  Processed {text_index} records from current folder; "
                        f"total documents={total_documents}, "
                        f"total final tokens={total_final_tokens}"
                    )

            if len(folder_paths) > 1:
                print(
                    f"total documents={total_documents}, "
                    f"total final tokens={total_final_tokens}"
                )

            del texts

        return {
            "folder_count": len(folder_paths),
            "document_count": total_documents,
            "final_token_count": total_final_tokens,
        }


if __name__ == "__main__":
    from settings import PROJECT_ROOT

    vocab_file = os.path.join(PROJECT_ROOT, "experiments", "c1c2", "20k", "vocab_s20k.txt")

    dataset_folders = [
        os.path.join(PROJECT_ROOT, "Corpus", "Corpus1"),
        os.path.join(PROJECT_ROOT, "Corpus", "Corpus2"),
    ]

    tokenizer = VanillaBPETokenizerExp(vocab_file)

    stats = tokenizer.tokenize_dataset(dataset_folders)

    print("\n=== Vanilla BPE Dataset Tokenization Stats ===")
    print(f"Folders processed  : {stats['folder_count']}")
    print(f"Documents processed: {stats['document_count']}")
    print(f"Final token count  : {stats['final_token_count']}")