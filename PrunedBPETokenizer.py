import os
import ast
import re
from typing import Dict, List, Tuple, Set, Optional, Pattern

from pruned_bpe_pretokenizer import pretokenize

Pair = Tuple[int, int]


class PrunedBPETokenizer:
    def __init__(self, vocab_path: str, inter_vocab_path: Optional[str] = None):
        # token id -> children token ids
        self.id_to_children: Dict[int, Pair] = {}

        # token id -> raw bytes represented by that token
        self.id_to_bytes: Dict[int, bytes] = {
            i: bytes([i]) for i in range(256)
        }

        # pair -> exported token id, includes both model-visible and internal-only merges
        self.merges: Dict[Pair, int] = {}

        # pair -> original BPE merge rank.
        # This is normally the original training-space token ID.
        # It must be used for choosing merge order.
        self.merge_ranks: Dict[Pair, int] = {}

        # IDs that should not be exposed to the model output from encode().
        self.internal_token_ids: Set[int] = set()

        # IDs reserved for special tokens such as EOS/SOS/SEP/PAD.
        self.special_token_ids: Dict[str, int] = {}
        self.id_to_special_token: Dict[int, str] = {}

        # Regex pattern for special tokens.
        # Built after vocab files are loaded.
        self._special_token_pattern: Optional[Pattern[str]] = None

        self._load_vocab_file(vocab_path, is_internal=False)

        if inter_vocab_path is not None and os.path.exists(inter_vocab_path):
            self._load_vocab_file(inter_vocab_path, is_internal=True)

        self._rebuild_bytes_and_merges()
        self._build_special_token_pattern()

    def _build_special_token_pattern(self) -> None:
        """
        Build a regex pattern that can split text around known special tokens.

        Example:
            special tokens: <_SOS_>, <_EOS_>, <_PAD_>
            text: "<_SOS_>Hello<_EOS_>"
            parts: ["", "<_SOS_>", "Hello", "<_EOS_>", ""]
        """
        if not self.special_token_ids:
            self._special_token_pattern = None
            return

        escaped_specials = [
            re.escape(token_text)
            for token_text in sorted(
                self.special_token_ids.keys(),
                key=len,
                reverse=True,
            )
        ]

        self._special_token_pattern = re.compile(
            "(" + "|".join(escaped_specials) + ")"
        )

    def _rebuild_bytes_and_merges(self) -> None:
        visiting: Set[int] = set()

        def build_bytes(token_id: int) -> bytes:
            if token_id in self.id_to_bytes:
                return self.id_to_bytes[token_id]

            if token_id in self.id_to_special_token:
                return self.id_to_bytes[token_id]

            if token_id in visiting:
                raise ValueError(f"Cycle detected while rebuilding token {token_id}")

            if token_id not in self.id_to_children:
                raise ValueError(f"Missing token id in vocab files: {token_id}")

            visiting.add(token_id)
            left_id, right_id = self.id_to_children[token_id]

            if (left_id, right_id) == (-1, -1):
                if token_id < 256:
                    self.id_to_bytes[token_id] = bytes([token_id])
                    visiting.remove(token_id)
                    return self.id_to_bytes[token_id]

                if token_id in self.id_to_special_token:
                    visiting.remove(token_id)
                    return self.id_to_bytes[token_id]

                raise ValueError(
                    f"Token {token_id} has no children but is not a base byte "
                    f"or special token"
                )

            self.id_to_bytes[token_id] = build_bytes(left_id) + build_bytes(right_id)
            visiting.remove(token_id)
            return self.id_to_bytes[token_id]

        for token_id in sorted(self.id_to_children.keys()):
            build_bytes(token_id)

        # Rebuild pair -> exported token id.
        # merge_ranks were loaded from vocab/inter_vocab.
        self.merges.clear()

        for token_id, children in self.id_to_children.items():
            if children != (-1, -1):
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

    def _expand_internal_token(self, token_id: int) -> List[int]:
        if token_id not in self.internal_token_ids:
            return [token_id]

        left_id, right_id = self.id_to_children[token_id]
        return (
            self._expand_internal_token(left_id)
            + self._expand_internal_token(right_id)
        )

    def _remove_internal_tokens(self, ids: List[int]) -> List[int]:
        result: List[int] = []
        for token_id in ids:
            result.extend(self._expand_internal_token(token_id))
        return result

    def _encode_pretokenized_chunk(self, chunk: str) -> List[int]:
        """
        Encode one pretokenized chunk using byte-level BPE.

        This must receive one chunk from pretokenize(...), not the whole
        original text. This keeps tokenization boundaries consistent with
        training, where BPE merges were learned only inside pretokenized chunks.
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

            # Use the original merge rank, not the exported token id.
            # This is critical for Pruned BPE, because internal tokens may be
            # remapped to high exported IDs while still being early BPE merges.
            best_pair = min(candidate_pairs, key=lambda p: self.merge_ranks[p])
            new_id = self.merges[best_pair]
            ids = self._merge_ids(ids, best_pair, new_id)

        return ids

    def _encode_normal_text(self, text: str) -> List[int]:
        """
        Encode ordinary text using the same pretokenization boundary as training.
        This method does not treat <_SOS_>, <_EOS_>, etc. specially.
        """
        result: List[int] = []

        for chunk in pretokenize(text):
            if not chunk:
                continue

            result.extend(self._encode_pretokenized_chunk(chunk))

        # Internal-only tokens may be useful during merging, but should not be
        # exposed as final model input IDs.
        return self._remove_internal_tokens(result)

    def encode(self, text: str, encode_special_tokens: bool = False) -> List[int]:
        """
        Encode text into token IDs.
        Recommended usage:
            Raw user input:
                encode(user_text, encode_special_tokens=False)
            System-built prompt:
                encode(prompt_text, encode_special_tokens=True)

        When encode_special_tokens=False:
            "<_SOS_>" is treated as normal text characters:
            '<', '_', 'S', 'O', 'S', '_', '>'

        When encode_special_tokens=True:
            "<_SOS_>" is treated as the reserved special token ID.
        """
        if (
            not encode_special_tokens
            or not self.special_token_ids
            or self._special_token_pattern is None
        ):
            return self._encode_normal_text(text)

        # Fast path for your current special-token style: <_SOS_>, <_EOS_>, <_SEP_>, <_PAD_>.
        # If there is no '<_', no special token can appear.
        # Modify this in case your special tokens do not contain "<_"
        if "<_" not in text:
            return self._encode_normal_text(text)

        result: List[int] = []

        parts = self._special_token_pattern.split(text)
        for part in parts:
            if not part:
                continue

            special_id = self.special_token_ids.get(part)
            if special_id is not None:
                result.append(special_id)
            else:
                result.extend(self._encode_normal_text(part))

        return result

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        chunks: List[bytes] = []

        for token_id in ids:
            if token_id in self.id_to_special_token:
                if skip_special_tokens:
                    continue
                chunks.append(self.id_to_special_token[token_id].encode("utf-8"))
                continue

            if token_id not in self.id_to_bytes:
                raise ValueError(f"Unknown token id: {token_id}")

            chunks.append(self.id_to_bytes[token_id])

        byte_string = b"".join(chunks)
        return byte_string.decode("utf-8", errors="replace")

    def _load_vocab_file(self, path: str, is_internal: bool) -> None:
        """
        Load vocab.txt or inter_vocab.txt.

        Support 4-column format:
            token_id    children_tuple    token_text    merge_rank

        For learned BPE tokens, merge_rank is the original training-space token ID.
        The tokenizer must use merge_rank for merge priority.
        """
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) != 4:
                    raise ValueError(f"Invalid vocab line: {line}")

                token_id_str, children_str, token_text_repr, merge_rank_str = parts
                merge_rank = int(merge_rank_str)

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

                if is_internal:
                    self.internal_token_ids.add(token_id)

                # Base byte tokens also have (-1, -1), but special tokens are >= 256.
                if children == (-1, -1) and token_id >= 256:
                    token_text = token_bytes.decode("utf-8", errors="replace")
                    self.special_token_ids[token_text] = token_id
                    self.id_to_special_token[token_id] = token_text
                    continue

                # Learned BPE token.
                if children != (-1, -1):
                    if merge_rank is None:
                        # Old 3-column fallback.
                        # This is safe only when token_id still equals merge order.
                        merge_rank = token_id

                    if merge_rank < 0:
                        raise ValueError(
                            f"Invalid merge_rank for learned token {token_id}: "
                            f"{merge_rank}"
                        )

                    self.merge_ranks[children] = merge_rank

if __name__ == "__main__":
    from settings import PROJECT_ROOT

    vocab_file = os.path.join(PROJECT_ROOT, "vocab.txt")
    inter_vocab_file = os.path.join(PROJECT_ROOT, "inter_vocab.txt")

    tokenizer = PrunedBPETokenizer(vocab_file, inter_vocab_file)

    test_texts = [
        "能",
        "可再生能源",
        "能满足现有消耗",
        "hello world, I love machine learning.",
        "你好吗？Are you OK?",
        "我喜欢学习人工智能。",
        "hello世界, AI很有意思!",
        "<_SOS_>简单测试<_EOS_><_PAD_>",
        "THIS IS A REALLY SIMPLE TEST",
        "蟁螻欬蚷洉",
        "ӳԂԃ",
        "쌍아안애",
        "西班牙首相访问中华人民共和国。",
        "刚果人民共和国",
        "The word environment is constructed from token en and vironment.",
        "是啊姻宙扶昏",
        "' Color',' Light',' Hill','bur'",
        "<div><p>这是一个测试小段落（英语：This is a short paragraph for test.）</p></div>",
        """
        {
            "compilerOptions": {
                "target": "ES2020",
                "useDefineForClassFields": true,
                "lib": ["ES2020", "DOM", "DOM.Iterable"],
                "module": "ESNext",
                "skipLibCheck": true,
                "moduleResolution": "bundler",
                "allowImportingTsExtensions": true,
                "resolveJsonModule": true,
                "isolatedModules": true,
                "noEmit": true,
                "jsx": "react-jsx",
                "strict": true,
                "noUnusedLocals": true,
                "noUnusedParameters": true,
                "noFallthroughCasesInSwitch": true,
                "baseUrl": ".",
                "paths": {
                  "@/*": ["./src/*"]
                }
            },
            "include": ["src", "tests", "playwright.config.ts"],
            "references": [
                {
                  "path": "./tsconfig.node.json"
                }
            ]
        }
        """,
    ]

    print("\n=== Encoding / Decoding Test: Raw User Input Mode ===")
    print("In this mode, <_SOS_>/<_EOS_>/<_PAD_> are treated as normal text.")

    for text in test_texts:
        encoded = tokenizer.encode(text, encode_special_tokens=False)
        decoded = tokenizer.decode(encoded, skip_special_tokens=False)

        print("\nOriginal:", text)
        print("Encoded :", encoded)
        print("Decoded :", decoded)

        assert text == decoded

    print("\n=== Encoding / Decoding Test: System Prompt Mode ===")
    print("In this mode, <_SOS_>/<_EOS_>/<_PAD_> are encoded as reserved special IDs.")

    system_prompt_text = "<_SOS_>简单测试<_EOS_><_PAD_>"

    encoded = tokenizer.encode(system_prompt_text, encode_special_tokens=True)
    decoded_keep_specials = tokenizer.decode(encoded, skip_special_tokens=False)
    decoded_skip_specials = tokenizer.decode(encoded, skip_special_tokens=True)

    print("\nOriginal:", system_prompt_text)
    print("Encoded :", encoded)
    print("Decoded with specials:", decoded_keep_specials)
    print("Decoded skip specials:", decoded_skip_specials)