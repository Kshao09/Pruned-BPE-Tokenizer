from __future__ import annotations

import ast
import math
import os
from collections import Counter
from pathlib import Path
from typing import Sequence

from pruned_bpe_pretokenizer import pretokenize


class MinTokenDPSegmenter:
    """
    Minimum-token dynamic-programming segmenter used for DP exposure analysis.

    The segmenter loads exactly one four-column BPE vocabulary:

        token_id<TAB>(left_id, right_id)<TAB>token_literal<TAB>value

    It does not support a separate internal-only vocabulary and does not perform
    post-pruning expansion. Its only job is to find an exact minimum-token
    segmentation of the supplied text under the input vocabulary.

    For string input, ``pretokenize()`` is applied first and each pretoken is
    segmented independently. Raw bytes bypass pretokenization.

    Tie-breaking:
        If multiple segmentations use the same minimum number of tokens, prefer
        the longest token at the earliest differing position. If token lengths
        also tie, prefer the smaller token ID.
    """
    _TOKEN_ID_KEY = -1

    def __init__(
        self,
        vocab_path: str | Path = "vocab.txt",
        *,
        text_encoding: str = "utf-8",
        require_all_single_bytes: bool = True,
    ) -> None:
        self.vocab_path = Path(vocab_path)
        self.text_encoding = text_encoding
        self.require_all_single_bytes = require_all_single_bytes

        self.id_to_token: dict[int, bytes] = {}
        self.id_to_children: dict[int, tuple[int, int]] = {}
        self.token_to_id: dict[bytes, int] = {}

        self._trie: dict[int, dict] = {}
        self.max_token_length = 0

        self._load_vocab()

    def _load_vocab(self) -> None:
        if not self.vocab_path.is_file():
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")

        with self.vocab_path.open("r", encoding="utf-8", newline="") as vocab_file:
            for line_number, raw_line in enumerate(vocab_file, start=1):
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue

                parts = line.split("\t", maxsplit=3)
                if len(parts) != 4:
                    raise ValueError(
                        f"{self.vocab_path}:{line_number}: expected 4 "
                        f"tab-separated columns, found {len(parts)}"
                    )

                token_id_str, children_str, token_literal, _ = parts

                try:
                    token_id = int(token_id_str)
                except ValueError as exc:
                    raise ValueError(
                        f"{self.vocab_path}:{line_number}: invalid token ID: "
                        f"{token_id_str!r}"
                    ) from exc

                children = self._parse_children(
                    children_str,
                    path=self.vocab_path,
                    line_number=line_number,
                )
                token = self._parse_token_literal(
                    token_literal,
                    path=self.vocab_path,
                    line_number=line_number,
                )

                if not token:
                    raise ValueError(
                        f"{self.vocab_path}:{line_number}: empty tokens are "
                        "not allowed"
                    )

                if token_id in self.id_to_token:
                    raise ValueError(
                        f"{self.vocab_path}:{line_number}: duplicate token ID "
                        f"{token_id}"
                    )

                if token in self.token_to_id:
                    previous_id = self.token_to_id[token]
                    raise ValueError(
                        f"{self.vocab_path}:{line_number}: duplicate token "
                        f"{token!r}; already loaded as token ID {previous_id}"
                    )

                self.id_to_token[token_id] = token
                self.id_to_children[token_id] = children
                self.token_to_id[token] = token_id

        if not self.id_to_token:
            raise ValueError(f"Vocabulary is empty: {self.vocab_path}")

        self.max_token_length = max(
            len(token) for token in self.id_to_token.values()
        )
        self._trie = self._build_trie()

        if self.require_all_single_bytes:
            self._validate_single_byte_fallbacks()

    @staticmethod
    def _parse_children(
        children_str: str,
        *,
        path: Path,
        line_number: int,
    ) -> tuple[int, int]:
        try:
            children = ast.literal_eval(children_str)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid children tuple: "
                f"{children_str!r}"
            ) from exc

        if (
            not isinstance(children, tuple)
            or len(children) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in children
            )
        ):
            raise ValueError(
                f"{path}:{line_number}: children must be a pair of integers"
            )

        return children

    @staticmethod
    def _parse_token_literal(
        token_literal: str,
        *,
        path: Path,
        line_number: int,
    ) -> bytes:
        try:
            value = ast.literal_eval(token_literal)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid token literal: "
                f"{token_literal!r}"
            ) from exc

        if isinstance(value, bytes):
            return value

        if isinstance(value, str):
            return value.encode("utf-8")

        raise ValueError(
            f"{path}:{line_number}: token literal must evaluate to str or "
            f"bytes, not {type(value).__name__}"
        )

    def _build_trie(self) -> dict[int, dict]:
        root: dict[int, dict] = {}

        for token_id, token in self.id_to_token.items():
            node = root
            for byte_value in token:
                node = node.setdefault(byte_value, {})
            node[self._TOKEN_ID_KEY] = token_id

        return root

    def _validate_single_byte_fallbacks(self) -> None:
        missing = [
            byte_value
            for byte_value in range(256)
            if bytes((byte_value,)) not in self.token_to_id
        ]

        if missing:
            preview = ", ".join(f"0x{x:02x}" for x in missing[:16])
            if len(missing) > 16:
                preview += ", ..."

            raise ValueError(
                "The vocabulary must contain all 256 one-byte fallback "
                f"tokens. Missing {len(missing)} byte values: {preview}"
            )

    def segment(
        self,
        text: str | bytes | bytearray | memoryview,
    ) -> list[int]:
        """
        Return the exact minimum-token DP segmentation of ``text``.

        String input is pretokenized first. Byte-like input is segmented
        directly without pretokenization.
        """
        if isinstance(text, str):
            chunks = pretokenize(text)
            if "".join(chunks) != text:
                raise RuntimeError(
                    "pretokenize() did not preserve the original input text"
                )

            token_ids: list[int] = []
            for chunk_index, chunk in enumerate(chunks):
                if not chunk:
                    raise RuntimeError(
                        f"pretokenize() returned an empty chunk at index "
                        f"{chunk_index}"
                    )

                chunk_bytes = chunk.encode(self.text_encoding)
                try:
                    token_ids.extend(self._segment_bytes(chunk_bytes))
                except ValueError as exc:
                    raise ValueError(
                        f"Unable to segment pretoken {chunk_index} "
                        f"({chunk!r}): {exc}"
                    ) from exc

            return token_ids

        if isinstance(text, (bytes, bytearray, memoryview)):
            return self._segment_bytes(bytes(text))

        raise TypeError(
            "segment() expects str, bytes, bytearray, or memoryview; "
            f"received {type(text).__name__}"
        )

    def _segment_bytes(self, data: bytes) -> list[int]:
        """Find the exact minimum-token segmentation of one byte sequence."""
        n = len(data)
        if n == 0:
            return []

        unreachable = n + 1
        best_count = [unreachable] * (n + 1)
        best_count[n] = 0

        choice_id = [-1] * n
        choice_end = [-1] * n

        for start in range(n - 1, -1, -1):
            node = self._trie
            end = start
            selected_length = -1
            selected_id = -1

            while end < n:
                node = node.get(data[end])
                if node is None:
                    break

                end += 1
                token_id = node.get(self._TOKEN_ID_KEY)

                if token_id is None or best_count[end] == unreachable:
                    continue

                candidate_count = 1 + best_count[end]
                candidate_length = end - start

                is_better = candidate_count < best_count[start]
                if candidate_count == best_count[start]:
                    is_better = (
                        candidate_length > selected_length
                        or (
                            candidate_length == selected_length
                            and (
                                selected_id == -1
                                or token_id < selected_id
                            )
                        )
                    )

                if is_better:
                    best_count[start] = candidate_count
                    choice_id[start] = token_id
                    choice_end[start] = end
                    selected_length = candidate_length
                    selected_id = token_id

        if best_count[0] == unreachable:
            failure_offset = self._first_unencodable_offset(data)
            preview = data[failure_offset: failure_offset + 16]
            raise ValueError(
                "Input cannot be fully segmented with this vocabulary. "
                f"First failing byte offset: {failure_offset}; "
                f"next bytes: {preview!r}"
            )

        token_ids: list[int] = []
        position = 0

        while position < n:
            token_id = choice_id[position]
            next_position = choice_end[position]

            if token_id < 0 or next_position <= position:
                raise RuntimeError(
                    "Internal segmenter error while reconstructing the DP path"
                )

            token_ids.append(token_id)
            position = next_position

        return token_ids

    def _first_unencodable_offset(self, data: bytes) -> int:
        reachable = bytearray(len(data) + 1)
        reachable[0] = 1
        farthest = 0

        for start in range(len(data)):
            if not reachable[start]:
                continue

            node = self._trie
            upper_bound = min(len(data), start + self.max_token_length)

            for end in range(start, upper_bound):
                node = node.get(data[end])
                if node is None:
                    break

                if self._TOKEN_ID_KEY in node:
                    next_position = end + 1
                    reachable[next_position] = 1
                    if next_position > farthest:
                        farthest = next_position

        return farthest

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def __len__(self) -> int:
        return self.vocab_size


def tokenize_dataset_with_pruning(
    segmenter: MinTokenDPSegmenter,
    dataset_folders: Sequence[str | Path],
    *,
    target_vocab_size: int,
    min_exposure_count: int,
    protect_two_space_tokens: bool = True,
    output_vocab_path: str | Path = "vocab_pruned.txt",
    output_inter_vocab_path: str | Path = "inter_vocab_pruned.txt",
    recursive: bool = True,
    file_suffixes: Sequence[str] = (".txt",),
) -> dict[str, int]:
    """
    Run DP exposure analysis and create final pruned visible/internal vocabularies.

    ``protect_two_space_tokens=True`` reproduces the H-SP2 behavior of the old
    PruneDPVocab2Spaces.py: every learned token beginning with two consecutive
    ASCII spaces is protected from exposure-based pruning. This is the default.

    ``protect_two_space_tokens=False`` reproduces the pruning behavior of the
    old PruneDPVocab.py for the current one-vocabulary workflow.

    Exported ID layout:
        0..255                 base bytes
        256..                  visible learned tokens
        ...target_vocab_size-1 reserved/special tokens
        target_vocab_size..    retained internal-only tokens

    Child IDs are remapped consistently. The fourth input column is preserved
    unchanged in the exported files.
    """
    if not dataset_folders:
        raise ValueError("dataset_folders must contain at least one folder")
    if target_vocab_size <= 0:
        raise ValueError("target_vocab_size must be > 0")
    if min_exposure_count < 0:
        raise ValueError("min_exposure_count must be >= 0")

    suffixes = {
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in file_suffixes
    }

    dataset_files: list[Path] = []
    seen_files: set[Path] = set()

    for folder_value in dataset_folders:
        folder = Path(folder_value)
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

        iterator = folder.rglob("*") if recursive else folder.glob("*")
        for path in iterator:
            if not path.is_file():
                continue
            if suffixes and path.suffix.lower() not in suffixes:
                continue

            resolved = path.resolve()
            if resolved not in seen_files:
                seen_files.add(resolved)
                dataset_files.append(path)

    dataset_files.sort(key=lambda path: str(path).lower())

    if not dataset_files:
        raise ValueError("No corpus files were found in the supplied dataset folders")

    # Initialize all tokens to zero so zero-exposure tokens are included.
    exposure = Counter({token_id: 0 for token_id in segmenter.id_to_token})
    total_dp_tokens = 0
    total_characters = 0

    print("\n=== Minimum-Token DP Exposure Analysis ===")
    print(f"Dataset files : {len(dataset_files):,}")
    print(f"DP vocab      : {segmenter.vocab_size:,}")

    for file_index, path in enumerate(dataset_files, start=1):
        corpus_text = path.read_text(
            encoding=segmenter.text_encoding,
            errors="strict",
        )

        if not corpus_text:
            print(f"[{file_index}/{len(dataset_files)}] {path} (empty; skipped)")
            continue

        dp_ids = segmenter.segment(corpus_text)
        exposure.update(dp_ids)

        total_dp_tokens += len(dp_ids)
        total_characters += len(corpus_text)

        print(
            f"[{file_index}/{len(dataset_files)}] {path} "
            f"chars={len(corpus_text):,}, "
            f"dp_tokens={len(dp_ids):,}, "
            f"running_total={total_dp_tokens:,}"
        )

    # Fixed byte-level base vocabulary.
    base_ids = list(range(256))
    for token_id in base_ids:
        if token_id not in segmenter.id_to_token:
            raise ValueError(f"Missing required base byte token ID {token_id}")
        if segmenter.id_to_children[token_id] != (-1, -1):
            raise ValueError(
                f"Base byte token ID {token_id} unexpectedly has BPE children"
            )

    learned_ids = sorted(
        token_id
        for token_id, children in segmenter.id_to_children.items()
        if children != (-1, -1)
    )

    # Any no-children token outside 0..255 is treated as reserved/special.
    special_ids = sorted(
        token_id
        for token_id, children in segmenter.id_to_children.items()
        if children == (-1, -1) and token_id not in base_ids
    )

    fixed_visible_count = len(base_ids) + len(special_ids)
    required_visible_learned = target_vocab_size - fixed_visible_count

    if required_visible_learned < 0:
        raise ValueError(
            f"target_vocab_size={target_vocab_size} is smaller than the "
            f"{fixed_visible_count} fixed visible tokens"
        )

    print("\n=== Final Vocabulary Selection ===")
    print(f"Target visible vocabulary : {target_vocab_size:,}")
    print(f"Base byte tokens          : {len(base_ids):,}")
    print(f"Reserved/special tokens   : {len(special_ids):,}")
    print(f"Required visible learned  : {required_visible_learned:,}")
    print(f"Learned candidates        : {len(learned_ids):,}")
    print(f"Exposure threshold        : {min_exposure_count:,}")
    print(
        "Protect two-space tokens  : "
        f"{'yes' if protect_two_space_tokens else 'no'}"
    )

    protected_ids: set[int] = set()

    if protect_two_space_tokens:
        # H-SP2 behavior: learned tokens beginning with two consecutive ASCII
        # spaces are exempt from exposure-based pruning.
        protected_ids = {
            token_id
            for token_id in learned_ids
            if segmenter.id_to_token[token_id].startswith(b"  ")
        }

        print(f"Protected whitespace tokens: {len(protected_ids):,}")
        for token_id in sorted(protected_ids):
            print(
                f"  ID={token_id:,}, "
                f"token={segmenter.id_to_token[token_id]!r}, "
                f"exposure={exposure[token_id]:,}"
            )

        if len(protected_ids) > required_visible_learned:
            raise RuntimeError(
                "Protected token count exceeds available visible learned slots"
            )

        eligible_nonprotected_ids = [
            token_id
            for token_id in learned_ids
            if token_id not in protected_ids
            and exposure[token_id] >= min_exposure_count
        ]

        required_nonprotected = required_visible_learned - len(protected_ids)

        if len(eligible_nonprotected_ids) < required_nonprotected:
            shortfall = required_nonprotected - len(eligible_nonprotected_ids)

            print("\n=== Candidate Pool Too Small ===")
            print(f"Protected learned tokens  : {len(protected_ids):,}")
            print(
                f"Eligible non-protected    : "
                f"{len(eligible_nonprotected_ids):,}"
            )
            print(f"Required visible learned  : {required_visible_learned:,}")
            print(f"Shortfall                 : {shortfall:,}")
            print("Continue BPE training and run this pruning step again.")

            raise RuntimeError(
                "Not enough sufficiently exposed learned tokens to fill the "
                f"target vocabulary. Need {shortfall} more visible learned "
                "token(s)."
            )

        selected_nonprotected_ids = set(
            eligible_nonprotected_ids[:required_nonprotected]
        )
        visible_learned_set = protected_ids | selected_nonprotected_ids

        visible_learned_old_ids = [
            token_id
            for token_id in learned_ids
            if token_id in visible_learned_set
        ]

        if len(visible_learned_old_ids) != required_visible_learned:
            raise RuntimeError(
                "Internal error: selected visible learned-token count does not "
                "equal required_visible_learned"
            )

        if visible_learned_old_ids:
            # A protected token may occur later than the ordinary stopping
            # point. Retain all preceding non-visible learned tokens as
            # internal-only so child dependencies remain valid.
            last_visible_id = max(visible_learned_old_ids)
            stop_index = learned_ids.index(last_visible_id) + 1
        else:
            stop_index = 0

        internal_old_ids = [
            token_id
            for token_id in learned_ids[:stop_index]
            if token_id not in visible_learned_set
        ]

        discarded_old_ids = learned_ids[stop_index:]
    else:
        # Original PruneDPVocab.py behavior: scan candidates in BPE merge order,
        # retaining low-exposure candidates as internal-only until enough
        # visible learned tokens have been selected.
        visible_learned_old_ids: list[int] = []
        internal_old_ids: list[int] = []

        stop_index = len(learned_ids)

        for index, token_id in enumerate(learned_ids):
            if len(visible_learned_old_ids) >= required_visible_learned:
                stop_index = index
                break

            if exposure[token_id] < min_exposure_count:
                internal_old_ids.append(token_id)
            else:
                visible_learned_old_ids.append(token_id)

        discarded_old_ids = learned_ids[stop_index:]

        if len(visible_learned_old_ids) < required_visible_learned:
            shortfall = required_visible_learned - len(visible_learned_old_ids)

            eligible_total = sum(
                1
                for token_id in learned_ids
                if exposure[token_id] >= min_exposure_count
            )

            print("\n=== Candidate Pool Too Small ===")
            print(f"Eligible learned tokens   : {eligible_total:,}")
            print(f"Required visible learned  : {required_visible_learned:,}")
            print(f"Shortfall                 : {shortfall:,}")
            print("Continue BPE training and run this pruning step again.")

            raise RuntimeError(
                "Not enough sufficiently exposed learned tokens to fill the "
                f"target vocabulary. Need {shortfall} more visible learned "
                "token(s)."
            )

    # Build old-ID -> exported-ID mapping.
    old_to_new: dict[int, int] = {}

    for old_id in base_ids:
        old_to_new[old_id] = old_id

    next_visible_id = 256
    for old_id in visible_learned_old_ids:
        old_to_new[old_id] = next_visible_id
        next_visible_id += 1

    for old_id in special_ids:
        old_to_new[old_id] = next_visible_id
        next_visible_id += 1

    if next_visible_id != target_vocab_size:
        raise RuntimeError(
            "Final visible ID count does not equal target_vocab_size"
        )

    next_internal_id = target_vocab_size
    for old_id in internal_old_ids:
        old_to_new[old_id] = next_internal_id
        next_internal_id += 1

    retained_old_ids = (
        set(base_ids)
        | set(visible_learned_old_ids)
        | set(special_ids)
        | set(internal_old_ids)
    )

    # A retained learned token must never depend on a discarded later token.
    for old_id in visible_learned_old_ids + internal_old_ids:
        left_old, right_old = segmenter.id_to_children[old_id]

        if left_old not in retained_old_ids:
            raise RuntimeError(
                f"Retained token {old_id} depends on discarded/missing "
                f"left child {left_old}"
            )
        if right_old not in retained_old_ids:
            raise RuntimeError(
                f"Retained token {old_id} depends on discarded/missing "
                f"right child {right_old}"
            )

    # Read original four-column rows from the one input vocabulary.
    rows_by_id: dict[int, tuple[str, str]] = {}

    with segmenter.vocab_path.open("r", encoding="utf-8", newline="") as vocab_file:
        for line_number, raw_line in enumerate(vocab_file, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue

            parts = line.split("\t", maxsplit=3)
            if len(parts) != 4:
                raise ValueError(
                    f"{segmenter.vocab_path}:{line_number}: expected 4 "
                    "tab-separated columns"
                )

            try:
                old_id = int(parts[0])
            except ValueError as exc:
                raise ValueError(
                    f"{segmenter.vocab_path}:{line_number}: invalid token ID "
                    f"{parts[0]!r}"
                ) from exc

            if old_id in rows_by_id:
                raise ValueError(
                    f"Duplicate token ID {old_id} while reading input vocabulary"
                )

            # Preserve token literal and fourth-column metadata unchanged.
            rows_by_id[old_id] = (parts[2], parts[3])

    if set(rows_by_id) != set(segmenter.id_to_token):
        missing = set(segmenter.id_to_token) - set(rows_by_id)
        extra = set(rows_by_id) - set(segmenter.id_to_token)
        raise RuntimeError(
            "Vocabulary rows do not match segmenter contents. "
            f"Missing={sorted(missing)[:10]}, "
            f"extra={sorted(extra)[:10]}"
        )

    def make_row(old_id: int) -> str:
        new_id = old_to_new[old_id]
        old_children = segmenter.id_to_children[old_id]

        if old_children == (-1, -1):
            new_children = (-1, -1)
        else:
            left_old, right_old = old_children

            if left_old not in old_to_new or right_old not in old_to_new:
                raise RuntimeError(
                    f"Cannot remap children of retained token {old_id}: "
                    f"{old_children}"
                )

            new_children = (
                old_to_new[left_old],
                old_to_new[right_old],
            )

        token_literal, value = rows_by_id[old_id]

        return (
            f"{new_id}\t"
            f"({new_children[0]}, {new_children[1]})\t"
            f"{token_literal}\t"
            f"{value}"
        )

    output_vocab = Path(output_vocab_path)
    output_inter_vocab = Path(output_inter_vocab_path)

    output_vocab.parent.mkdir(parents=True, exist_ok=True)
    output_inter_vocab.parent.mkdir(parents=True, exist_ok=True)

    visible_export_order = base_ids + visible_learned_old_ids + special_ids

    with output_vocab.open("w", encoding="utf-8", newline="\n") as visible_file:
        for old_id in visible_export_order:
            visible_file.write(make_row(old_id) + "\n")

    with output_inter_vocab.open("w", encoding="utf-8", newline="\n") as internal_file:
        for old_id in internal_old_ids:
            internal_file.write(make_row(old_id) + "\n")

    visible_new_ids = [old_to_new[old_id] for old_id in visible_export_order]
    if visible_new_ids != list(range(target_vocab_size)):
        raise RuntimeError(
            "Final visible token IDs are not contiguous from 0 to "
            f"{target_vocab_size - 1}"
        )

    print("\n=== DP Pruning / Final Export Result ===")
    print(f"Minimum exposure threshold : {min_exposure_count:,}")
    print(f"Learned tokens examined    : {stop_index:,}")
    print(f"Visible learned retained   : {len(visible_learned_old_ids):,}")
    print(f"Internal-only retained     : {len(internal_old_ids):,}")
    print(f"Later candidates discarded : {len(discarded_old_ids):,}")
    print(f"Final visible vocabulary   : {target_vocab_size:,}")
    print(f"Visible ID range           : 0..{target_vocab_size - 1:,}")

    if internal_old_ids:
        print(
            f"Internal ID range          : "
            f"{target_vocab_size:,}..{next_internal_id - 1:,}"
        )
    else:
        print("Internal ID range          : none")

    print(f"Total raw DP tokens        : {total_dp_tokens:,}")
    print(f"Output vocab               : {output_vocab}")
    print(f"Output internal vocab      : {output_inter_vocab}")

    summary = {
        "dataset_files": len(dataset_files),
        "total_characters": total_characters,
        "total_dp_tokens": total_dp_tokens,
        "min_exposure_count": min_exposure_count,
        "target_vocab_size": target_vocab_size,
        "base_tokens": len(base_ids),
        "special_tokens": len(special_ids),
        "learned_candidates": len(learned_ids),
        "learned_tokens_examined": stop_index,
        "visible_learned_tokens": len(visible_learned_old_ids),
        "internal_tokens": len(internal_old_ids),
        "discarded_tokens": len(discarded_old_ids),
        "visible_tokens": target_vocab_size,
    }

    if protect_two_space_tokens:
        summary["protected_whitespace_tokens"] = len(protected_ids)

    return summary


def create_overshoot_vocab(
    source_vocab_path: str | Path,
    output_vocab_path: str | Path,
    *,
    target_vocab_size: int,
    overshoot_factor: float,
) -> None:
    source_vocab_path = Path(source_vocab_path)
    output_vocab_path = Path(output_vocab_path)

    rows = []

    with source_vocab_path.open("r", encoding="utf-8", newline="") as vocab_file:
        for raw_line in vocab_file:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue

            parts = line.split("\t", maxsplit=3)
            if len(parts) != 4:
                raise ValueError(f"Invalid vocab row: {line!r}")

            token_id = int(parts[0])
            children = parts[1]
            rows.append((token_id, children, parts[2], parts[3]))

    base_count = 256

    special_rows = [
        row
        for row in rows
        if row[0] >= base_count and row[1] == "(-1, -1)"
    ]

    num_reserved = len(special_rows)
    learned_target = target_vocab_size - base_count - num_reserved

    if learned_target < 0:
        raise ValueError(
            f"target_vocab_size={target_vocab_size} is smaller than the "
            f"{base_count + num_reserved} fixed base/special tokens"
        )

    learned_candidates = math.ceil(learned_target * overshoot_factor)
    ordinary_vocab_size = base_count + learned_candidates

    ordinary_rows = [
        row
        for row in rows
        if row[0] < ordinary_vocab_size
    ]

    if len(ordinary_rows) != ordinary_vocab_size:
        raise RuntimeError(
            f"Source vocab is too small. Need {ordinary_vocab_size:,} "
            f"ordinary tokens, found {len(ordinary_rows):,}."
        )

    output_vocab_path.parent.mkdir(parents=True, exist_ok=True)

    with output_vocab_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for token_id, children, token_literal, value in ordinary_rows:
            output_file.write(
                f"{token_id}\t{children}\t{token_literal}\t{value}\n"
            )

        next_id = ordinary_vocab_size

        for _, children, token_literal, value in special_rows:
            output_file.write(
                f"{next_id}\t{children}\t{token_literal}\t{value}\n"
            )
            next_id += 1

    print("\n=== Overshoot Vocabulary Created ===")
    print(f"Source vocab        : {source_vocab_path}")
    print(f"Overshoot factor    : {overshoot_factor}")
    print(f"Learned candidates  : {learned_candidates:,}")
    print(f"Ordinary vocab size : {ordinary_vocab_size:,}")
    print(f"Reserved tokens     : {num_reserved:,}")
    print(f"Total vocab size    : {next_id:,}")
    print(f"Output vocab        : {output_vocab_path}")


def create_overshoot_vocab_main() -> None:
    TARGET_VOCAB_SIZE = 12_000
    TARGET_VOCAB_STR = "12k"

    SOURCE_FACTOR = "1.15"
    DESIRED_FACTOR = 1.09

    folder = os.path.join(PROJECT_ROOT, "experiments", "c2", TARGET_VOCAB_STR)

    source_vocab_file = os.path.join(
        folder, f"vocab_s{TARGET_VOCAB_STR}_{SOURCE_FACTOR}.txt",
    )
    vocab_file = os.path.join(
        folder, f"vocab_s{TARGET_VOCAB_STR}_{DESIRED_FACTOR:.2f}.txt",
    )

    create_overshoot_vocab(
        source_vocab_file,
        vocab_file,
        target_vocab_size=TARGET_VOCAB_SIZE,
        overshoot_factor=DESIRED_FACTOR,
    )


def prune_dp_vocab_main() -> None:
    TARGET_VOCAB_SIZE = 16_000
    TARGET_VOCAB_STR = "16k"

    DESIRED_FACTOR = 1.10
    RATIO = 0.75

    # True  -> old PruneDPVocab2Spaces.py / H-SP2 behavior (default)
    # False -> old PruneDPVocab.py behavior
    PROTECT_TWO_SPACE_TOKENS = True

    MIN_EXPOSURE_COUNT = math.ceil(3008 * RATIO)

    folder = os.path.join(PROJECT_ROOT, "experiments", "c2", TARGET_VOCAB_STR)

    vocab_file = os.path.join(
        folder, f"vocab_s{TARGET_VOCAB_STR}_{DESIRED_FACTOR:.2f}.txt",
    )

    segmenter = MinTokenDPSegmenter(vocab_file)

    dataset_folders = [
        # os.path.join(PROJECT_ROOT, "Corpus", "Corpus1"),
        os.path.join(PROJECT_ROOT, "Corpus", "Corpus2"),
    ]

    mode_tag = "_2sp" if PROTECT_TWO_SPACE_TOKENS else ""

    output_vocab_file = os.path.join(
        folder,
        f"vocab_pruned{mode_tag}_s{TARGET_VOCAB_STR}_"
        f"{DESIRED_FACTOR:.2f}_{RATIO}.txt",
    )
    output_inter_vocab_file = os.path.join(
        folder,
        f"inter_vocab_pruned{mode_tag}_s{TARGET_VOCAB_STR}_"
        f"{DESIRED_FACTOR:.2f}_{RATIO}.txt",
    )

    summary = tokenize_dataset_with_pruning(
        segmenter,
        dataset_folders,
        target_vocab_size=TARGET_VOCAB_SIZE,
        min_exposure_count=MIN_EXPOSURE_COUNT,
        protect_two_space_tokens=PROTECT_TWO_SPACE_TOKENS,
        output_vocab_path=output_vocab_file,
        output_inter_vocab_path=output_inter_vocab_file,
    )

    print("\nSummary:", summary)


if __name__ == "__main__":
    from settings import PROJECT_ROOT

    # Pick one main workflow to run.
    # create_overshoot_vocab_main()
    prune_dp_vocab_main()
