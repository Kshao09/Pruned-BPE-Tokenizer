from __future__ import annotations

import os
from typing import Dict, List, TypedDict

from MinTokenDPTokenizer import MinTokenDPTokenizer
from PrunedBPETrainer import PrunedBPETrainer

class TokenizeDatasetResult(TypedDict):
    folder_count: int
    document_count: int
    final_token_count: int
    folder_token_counts: Dict[str, int]


class MaxLenTokenizerExp(MinTokenDPTokenizer):
    """
    Experiment wrapper for minimum-length vocabulary-only tokenization.

    This class uses the same pretokenization and dynamic-programming encoding
    behavior as MaxLenTokenizer, but it loads the original vocab.txt format
    produced by PrunedBPETrainer.

    Expected vocabulary format:

        token_id<TAB>children<TAB>token_literal<TAB>count

    Only the third column, token_literal, is used to build the tokenizer
    vocabulary. Token IDs are assigned by file order, which matches the
    sequential token IDs in the trainer's vocab.txt.

    This allows the same original vocab.txt to be used by:

        - VanillaBPETokenizerExp for standard ranked BPE encoding
        - MaxLenTokenizerExp for exact minimum-token DP encoding

    Both tokenizers also use the same pretokenization boundaries.
    """
    def __init__(self, vocab_path: str):
        super().__init__(vocab_path)

    def _load_vocab(self) -> None:
        """
        Load the original three- or four-column vocab.txt.

        Column meanings:

            0: token ID                 ignored
            1: child token IDs          ignored
            2: quoted token literal     used
            3: count/statistic          ignored, when present

        The third column may evaluate to either str or bytes. Readable strings
        are converted to UTF-8 bytes; bytes literals are preserved exactly.
        """
        if not self.vocab_path.is_file():
            raise FileNotFoundError(
                f"Vocabulary file not found: {self.vocab_path}"
            )

        token_to_id: dict[bytes, int] = {}
        id_to_token: list[bytes] = []

        with self.vocab_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as vocab_file:
            for line_number, raw_line in enumerate(vocab_file, start=1):
                # Remove only the physical line ending. Do not use strip(),
                # because whitespace inside a quoted token is meaningful.
                line = raw_line.rstrip("\r\n")

                if not line:
                    continue

                parts = line.split("\t", 3)

                if len(parts) < 3:
                    raise ValueError(
                        f"{self.vocab_path}:{line_number}: expected at least "
                        f"3 tab-separated columns, found {len(parts)}"
                    )

                token_literal = parts[2]
                token = self._parse_vocab_token(token_literal, line_number)

                if not token:
                    raise ValueError(
                        f"{self.vocab_path}:{line_number}: "
                        "empty tokens are not allowed"
                    )

                if token in token_to_id:
                    first_id = token_to_id[token]
                    raise ValueError(
                        f"{self.vocab_path}:{line_number}: duplicate token "
                        f"{token!r}; it already has token ID {first_id}"
                    )

                # The trainer writes tokens in token-ID order, beginning at 0.
                token_id = len(id_to_token)
                token_to_id[token] = token_id
                id_to_token.append(token)

        if not id_to_token:
            raise ValueError(
                f"Vocabulary is empty: {self.vocab_path}"
            )

        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.max_token_length = max(
            len(token) for token in id_to_token
        )
        self._trie = self._build_trie(id_to_token)

        if self.require_all_single_bytes:
            missing = [
                value
                for value in range(256)
                if bytes((value,)) not in self.token_to_id
            ]

            if missing:
                preview = ", ".join(
                    f"0x{value:02x}" for value in missing[:16]
                )

                if len(missing) > 16:
                    preview += ", ..."

                raise ValueError(
                    "The vocabulary does not contain every one-byte fallback "
                    f"token. Missing {len(missing)} byte values: {preview}."
                )

    def tokenize_dataset(
        self,
        folder_paths: List[str],
    ) -> TokenizeDatasetResult:
        """
        Tokenize all text loaded from the given dataset folders and return counts.

        The encoded token lists are not retained. Only their lengths are added,
        matching the experiment performed by VanillaBPETokenizerExp.
        """
        total_documents = 0
        total_final_tokens = 0
        folder_token_counts: Dict[str, int] = {}

        for folder_index, folder_path in enumerate(
            folder_paths,
            start=1,
        ):
            print(
                f"Loading folder {folder_index}/{len(folder_paths)}: "
                f"{folder_path}"
            )

            texts = PrunedBPETrainer.load_data(folder_path)

            print(f"Loaded {len(texts)} text records.")

            folder_final_tokens = 0

            for text_index, text in enumerate(texts, start=1):
                encoded = self.encode(text)
                token_count = len(encoded)

                folder_final_tokens += token_count
                total_final_tokens += token_count
                total_documents += 1

                if text_index % 100_000 == 0:
                    print(
                        f"  Processed {text_index} records from current folder; "
                        f"total documents={total_documents}, "
                        f"total final tokens={total_final_tokens}"
                    )

            folder_name = os.path.basename(os.path.normpath(folder_path))
            folder_token_counts[folder_name] = folder_final_tokens

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
            "folder_token_counts": folder_token_counts,
        }


if __name__ == "__main__":
    from settings import PROJECT_ROOT

    vocab_file = os.path.join(PROJECT_ROOT, "experiments", "DH_BPE", "c2", "12k", "vocab_pruned_bpe_0.6.txt")

    dataset_folders = [
        os.path.join(PROJECT_ROOT, "Corpus", "Corpus2"),
        os.path.join(PROJECT_ROOT, "Corpus", "Corpus1"),
        os.path.join(PROJECT_ROOT, "Corpus", "Corpus3")
        # os.path.join(PROJECT_ROOT, "Corpus", "Corpus3", "chinese_medicine"),
        # os.path.join(PROJECT_ROOT, "Corpus", "Corpus3", "chinese_weibo"),
        # os.path.join(PROJECT_ROOT, "Corpus", "Corpus3", "chinese_wiki"),
        # os.path.join(PROJECT_ROOT, "Corpus", "Corpus3", "english_reddit"),
        # os.path.join(PROJECT_ROOT, "Corpus", "Corpus3", "english_legal"),
    ]

    tokenizer = MaxLenTokenizerExp(vocab_file)

    stats = tokenizer.tokenize_dataset(dataset_folders)

    print("\n=== Max-Length DP Dataset Tokenization Stats ===")
    print(f"Folders processed  : {stats['folder_count']}")
    print(f"Documents processed: {stats['document_count']}")
    print(f"Final token count  : {stats['final_token_count']}")

    print("\nFinal token counts by corpus:")
    for folder_name, token_count in stats["folder_token_counts"].items():
        print(f"{folder_name}: {token_count:,}")
