import os
from typing import Dict, List, Optional

from PrunedBPETokenizer import PrunedBPETokenizer
from PrunedBPETrainer import PrunedBPETrainer


class PrunedBPETokenizerExp(PrunedBPETokenizer):
    """
    Experiment wrapper for PrunedBPETokenizer.

    This class reuses the normal PrunedBPETokenizer implementation and adds
    dataset-level final-token counting for experiments.

    It does not store encoded token lists, because large corpora may consume
    too much memory.
    """

    def __init__(self, vocab_path: str, inter_vocab_path: Optional[str] = None):
        super().__init__(vocab_path, inter_vocab_path)

    def tokenize_dataset(self, folder_paths: List[str]) -> Dict[str, int]:
        """
        Tokenize all text loaded from the given dataset folders and return counts.

        Parameters
        ----------
        folder_paths:
            List of folders. Each folder is passed to load_data(folder_path).

        Returns
        -------
        Dict[str, int]
            A dictionary containing:
                folder_count
                document_count
                final_token_count

        Notes
        -----
        The encoded token lists are not stored. Only their lengths are accumulated.
        This is safer for large experiment corpora.

        This method calls _encode_normal_text() directly because experiment data
        should be tokenized as ordinary corpus text. Special-token strings such
        as <_SOS_> and <_EOS_> are not treated as reserved special tokens here.
        """
        total_documents = 0
        total_final_tokens = 0

        for folder_index, folder_path in enumerate(folder_paths, start=1):
            print(f"Loading folder {folder_index}/{len(folder_paths)}: {folder_path}")

            texts = PrunedBPETrainer.load_data(folder_path)

            print(f"Loaded {len(texts)} text records.")

            for text_index, text in enumerate(texts, start=1):
                encoded = self._encode_normal_text(text)

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

    T_SIZE, P_RATIO = "18k", "0.4"
    vocab_file = os.path.join(PROJECT_ROOT, "experiments", "c1c2",
                              f"{T_SIZE}", f"vocab_p{T_SIZE}_{P_RATIO}.txt")
    inter_vocab_file = os.path.join(PROJECT_ROOT, "experiments", "c1c2",
                                    f"{T_SIZE}", f"inter_vocab_p{T_SIZE}_{P_RATIO}.txt")

    dataset_folders = [
        os.path.join(PROJECT_ROOT, "Corpus", "Corpus1"),
        os.path.join(PROJECT_ROOT, "Corpus", "Corpus2"),
    ]

    tokenizer = PrunedBPETokenizerExp(vocab_file, inter_vocab_file)

    stats = tokenizer.tokenize_dataset(dataset_folders)

    print("\n=== Pruned BPE Diagnostic Stats ===")
    print(f"Folders processed   : {stats['folder_count']}")
    print(f"Documents processed : {stats['document_count']}")
    print(f"Final token count   : {stats['final_token_count']}")