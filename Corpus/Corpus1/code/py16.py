import os
from typing import List, Optional, Tuple

from VanillaBPETokenizerExp import VanillaBPETokenizerExp
from PrunedBPETokenizerExp import PrunedBPETokenizerExp
from PrunedBPETrainer import PrunedBPETrainer
from pruned_bpe_pretokenizer import pretokenize


def get_pruned_raw_ids(tokenizer: PrunedBPETokenizerExp, text: str) -> List[int]:
    """
    Encode text with Pruned BPE but stop before internal-token expansion.

    This is the raw BPE merge result.
    """

    raw_ids: List[int] = []

    for chunk in pretokenize(text):
        if not chunk:
            continue

        raw_ids.extend(tokenizer._encode_pretokenized_chunk(chunk))

    return raw_ids


def compare_chunks(
    vanilla: VanillaBPETokenizerExp,
    pruned: PrunedBPETokenizerExp,
    text: str,
    max_mismatch_prints: int = 30,
) -> None:
    """
    Compare vanilla and pruned tokenization chunk by chunk.

    This function prints mismatching chunks and also reports the worst chunk.
    """

    chunks = list(pretokenize(text))

    mismatch_count = 0
    total_diff = 0

    worst_diff = 0
    worst_info: Optional[
        Tuple[int, str, List[int], List[int], List[int], int]
    ] = None

    for chunk_index, chunk in enumerate(chunks, start=1):
        if not chunk:
            continue

        vanilla_ids = vanilla.encode(chunk)

        pruned_raw_ids = pruned._encode_pretokenized_chunk(chunk)
        pruned_final_ids = pruned._remove_internal_tokens(pruned_raw_ids)

        diff = len(pruned_final_ids) - len(vanilla_ids)

        if diff != 0:
            mismatch_count += 1
            total_diff += diff

            if mismatch_count <= max_mismatch_prints:
                print("\n=== Chunk mismatch ===")
                print(f"Chunk index       : {chunk_index}")
                print(f"Chunk text repr   : {chunk!r}")
                print(f"Vanilla count     : {len(vanilla_ids)}")
                print(f"Pruned raw count  : {len(pruned_raw_ids)}")
                print(f"Pruned final count: {len(pruned_final_ids)}")
                print(f"Diff final-vanilla: {diff}")
                print(f"Vanilla ids       : {vanilla_ids}")
                print(f"Pruned raw ids    : {pruned_raw_ids}")
                print(f"Pruned final ids  : {pruned_final_ids}")

        if abs(diff) > abs(worst_diff):
            worst_diff = diff
            worst_info = (
                chunk_index,
                chunk,
                vanilla_ids,
                pruned_raw_ids,
                pruned_final_ids,
                diff,
            )

    print("\n=== Chunk Comparison Summary ===")
    print(f"Total chunks checked       : {len(chunks)}")
    print(f"Mismatch chunks            : {mismatch_count}")
    print(f"Total chunk diff           : {total_diff}")

    if mismatch_count > max_mismatch_prints:
        print(
            f"Only first {max_mismatch_prints} mismatching chunks were printed."
        )

    if worst_info is not None:
        (
            chunk_index,
            chunk,
            vanilla_ids,
            pruned_raw_ids,
            pruned_final_ids,
            diff,
        ) = worst_info

        print("\n=== Worst Chunk Mismatch ===")
        print(f"Chunk index       : {chunk_index}")
        print(f"Chunk text repr   : {chunk!r}")
        print(f"Vanilla count     : {len(vanilla_ids)}")
        print(f"Pruned raw count  : {len(pruned_raw_ids)}")
        print(f"Pruned final count: {len(pruned_final_ids)}")
        print(f"Diff final-vanilla: {diff}")
        print(f"Vanilla ids       : {vanilla_ids}")
        print(f"Pruned raw ids    : {pruned_raw_ids}")
        print(f"Pruned final ids  : {pruned_final_ids}")


def main() -> None:
    from settings import PROJECT_ROOT

    vanilla_vocab_file = os.path.join(
        PROJECT_ROOT,
        "experiments",
        "c2",
        "vocab_v8020.txt",
    )

    pruned_vocab_file = os.path.join(
        PROJECT_ROOT,
        "experiments",
        "c2",
        "vocab_p8020.txt",
    )

    inter_vocab_file = os.path.join(
        PROJECT_ROOT,
        "experiments",
        "c2",
        "inter_vocab_p8020.txt",
    )

    dataset_folders = [
        # os.path.join(PROJECT_ROOT, "Corpus", "Corpus1"),
        os.path.join(PROJECT_ROOT, "Corpus", "Corpus2"),
    ]

    # Stop when a text has this much difference.
    diff_threshold = 40

    vanilla = VanillaBPETokenizerExp(vanilla_vocab_file)
    pruned = PrunedBPETokenizerExp(pruned_vocab_file, inter_vocab_file)

    global_text_index = 0

    worst_diff = 0
    worst_text = ""
    worst_folder = ""
    worst_local_index = -1
    worst_global_index = -1
    worst_counts = None

    for folder_index, folder_path in enumerate(dataset_folders, start=1):
        print(f"\nLoading folder {folder_index}/{len(dataset_folders)}:")
        print(folder_path)

        texts = PrunedBPETrainer.load_data(folder_path)

        print(f"Loaded {len(texts)} text records.")

        for local_index, text in enumerate(texts, start=1):
            global_text_index += 1

            vanilla_ids = vanilla.encode(text)

            pruned_raw_ids = get_pruned_raw_ids(pruned, text)
            pruned_final_ids = pruned._remove_internal_tokens(pruned_raw_ids)

            diff = len(pruned_final_ids) - len(vanilla_ids)

            if abs(diff) > abs(worst_diff):
                worst_diff = diff
                worst_text = text
                worst_folder = folder_path
                worst_local_index = local_index
                worst_global_index = global_text_index
                worst_counts = (
                    len(vanilla_ids),
                    len(pruned_raw_ids),
                    len(pruned_final_ids),
                )

            if diff >= diff_threshold:
                print("\n=== Large mismatch found ===")
                print(f"Folder path       : {folder_path}")
                print(f"Local text index  : {local_index}")
                print(f"Global text index : {global_text_index}")
                print(f"Vanilla count     : {len(vanilla_ids)}")
                print(f"Pruned raw count  : {len(pruned_raw_ids)}")
                print(f"Pruned final count: {len(pruned_final_ids)}")
                print(f"Diff final-vanilla: {diff}")

                debug_dir = os.path.join(PROJECT_ROOT, "experiments", "debug")
                os.makedirs(debug_dir, exist_ok=True)

                bad_text_file = os.path.join(
                    debug_dir,
                    f"bad_text_global_{global_text_index}_diff_{diff}.txt",
                )

                with open(bad_text_file, "w", encoding="utf-8") as f:
                    f.write(text)

                print(f"\nSaved bad text to:")
                print(bad_text_file)

                print("\n=== Text preview repr ===")
                print(repr(text[:2000]))

                print("\n=== Text preview normal ===")
                print(text[:2000])

                print("\nNow comparing chunks from this text...")
                compare_chunks(vanilla, pruned, text)

                return

            if global_text_index % 50000 == 0:
                print(
                    f"Checked {global_text_index}; "
                    f"worst_diff={worst_diff} "
                    f"at global index {worst_global_index}"
                )

        del texts

    print("\n=== Finished without reaching threshold ===")
    print(f"Worst diff        : {worst_diff}")
    print(f"Worst folder      : {worst_folder}")
    print(f"Worst local index : {worst_local_index}")
    print(f"Worst global index: {worst_global_index}")

    if worst_counts is not None:
        vanilla_count, pruned_raw_count, pruned_final_count = worst_counts

        print(f"Vanilla count     : {vanilla_count}")
        print(f"Pruned raw count  : {pruned_raw_count}")
        print(f"Pruned final count: {pruned_final_count}")

    if worst_text:
        debug_dir = os.path.join(PROJECT_ROOT, "experiments", "debug")
        os.makedirs(debug_dir, exist_ok=True)

        worst_text_file = os.path.join(
            debug_dir,
            f"worst_text_global_{worst_global_index}_diff_{worst_diff}.txt",
        )

        with open(worst_text_file, "w", encoding="utf-8") as f:
            f.write(worst_text)

        print(f"\nSaved worst text to:")
        print(worst_text_file)

        print("\nNow comparing chunks from the worst text...")
        compare_chunks(vanilla, pruned, worst_text)


if __name__ == "__main__":
    main()