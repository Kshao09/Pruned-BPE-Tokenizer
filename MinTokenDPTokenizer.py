from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, Sequence

from pruned_bpe_pretokenizer import pretokenize


class MinTokenDPTokenizer:
    """
    Vocabulary-only byte tokenizer.

    Encoding:
        String input is first split with ``pretokenize()`` from
        ``pruned_bpe_pretokenizer.py``. Each pretoken is then encoded
        independently using a trie plus dynamic programming to find its
        minimum-token segmentation. Raw bytes input bypasses pretokenization.

    Tie-breaking:
        If multiple segmentations use the same minimum number of tokens, the
        tokenizer prefers the longest token at the earliest differing position.
        If token lengths also tie, the smaller token ID is preferred.

    Vocabulary format:
        One quoted token per line, for example:

            'a'
            'the'
            ' hello'
            '\\x00'
            '\\xe8\\x91\\xa1'
            b'bytes are also accepted'

        Ordinary quoted strings are encoded as UTF-8. Explicit ``b'...'``
        literals are preserved exactly, which supports partial or invalid
        UTF-8 byte sequences without making readable Unicode tokens opaque.
    """

    _TOKEN_ID_KEY = -1

    def __init__(
        self,
        vocab_path: str | Path = "vocab.txt",
        *,
        text_encoding: str = "utf-8",
        vocab_string_encoding: str = "utf-8",
        require_all_single_bytes: bool = True,
    ) -> None:
        self.vocab_path = Path(vocab_path)
        self.text_encoding = text_encoding
        if vocab_string_encoding.lower().replace("_", "-") != "utf-8":
            raise ValueError(
                "vocab_string_encoding must be 'utf-8' for the mixed "
                "quoted str/bytes vocabulary format"
            )
        self.vocab_string_encoding = "utf-8"
        self.require_all_single_bytes = require_all_single_bytes

        self.token_to_id: dict[bytes, int] = {}
        self.id_to_token: list[bytes] = []
        self._trie: dict[int, dict] = {}
        self.max_token_length = 0

        self._load_vocab()

    def _load_vocab(self) -> None:
        """
        Load tokens from ``vocab.txt`` and build the byte trie.

        Token IDs are assigned in file order, starting at zero.
        """
        if not self.vocab_path.is_file():
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")

        token_to_id: dict[bytes, int] = {}
        id_to_token: list[bytes] = []

        with self.vocab_path.open("r", encoding="utf-8", newline="") as vocab_file:
            for line_number, raw_line in enumerate(vocab_file, start=1):
                line = raw_line.strip()

                # Empty physical lines are ignored. The empty token must not be
                # used; a space token should be written as ' '.
                if not line:
                    continue

                token = self._parse_vocab_token(line, line_number)

                if not token:
                    raise ValueError(
                        f"{self.vocab_path}:{line_number}: empty tokens are not allowed"
                    )

                if token in token_to_id:
                    first_id = token_to_id[token]
                    raise ValueError(
                        f"{self.vocab_path}:{line_number}: duplicate token "
                        f"{token!r}; it already has token ID {first_id}"
                    )

                token_id = len(id_to_token)
                token_to_id[token] = token_id
                id_to_token.append(token)

        if not id_to_token:
            raise ValueError(f"Vocabulary is empty: {self.vocab_path}")

        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.max_token_length = max(len(token) for token in id_to_token)
        self._trie = self._build_trie(id_to_token)

        if self.require_all_single_bytes:
            missing = [
                value
                for value in range(256)
                if bytes((value,)) not in self.token_to_id
            ]
            if missing:
                preview = ", ".join(f"0x{value:02x}" for value in missing[:16])
                if len(missing) > 16:
                    preview += ", ..."
                raise ValueError(
                    "The vocabulary does not contain every one-byte fallback "
                    f"token. Missing {len(missing)} byte values: {preview}. "
                    "Add all 256 one-byte tokens, or construct the tokenizer with "
                    "require_all_single_bytes=False."
                )

    def _parse_vocab_token(self, line: str, line_number: int) -> bytes:
        try:
            value = ast.literal_eval(line)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"{self.vocab_path}:{line_number}: invalid quoted token: {line!r}"
            ) from exc

        if isinstance(value, bytes):
            return value

        if not isinstance(value, str):
            raise ValueError(
                f"{self.vocab_path}:{line_number}: each line must evaluate to "
                f"str or bytes, not {type(value).__name__}"
            )

        # String literals represent readable Unicode and are converted to
        # UTF-8. Exact arbitrary bytes must use a bytes literal such as
        # b'\\xe8\\x91'.
        return value.encode("utf-8")

    def _build_trie(self, tokens: Sequence[bytes]) -> dict[int, dict]:
        root: dict[int, dict] = {}

        for token_id, token in enumerate(tokens):
            node = root
            for byte_value in token:
                node = node.setdefault(byte_value, {})
            node[self._TOKEN_ID_KEY] = token_id

        return root

    def encode(self, text: str | bytes | bytearray | memoryview) -> list[int]:
        """
        Encode input into token IDs.

        For ``str`` input, ``pretokenize()`` is applied first. Dynamic
        programming is then run independently on every pretoken, so no token
        can cross a pretoken boundary.

        For ``bytes``, ``bytearray``, or ``memoryview`` input, pretokenization
        is skipped because arbitrary bytes are not necessarily valid Unicode.

        Within each encoded byte sequence, the dynamic program minimizes the
        total output-token count. Its worst-case time is O(n * L), where n is
        the byte length and L is the maximum vocabulary-token length.
        """
        if isinstance(text, str):
            chunks = pretokenize(text)

            # The attached pretokenizer is intended to preserve the original
            # text exactly. Check that invariant before encoding.
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
                    token_ids.extend(self._encode_bytes(chunk_bytes))
                except ValueError as exc:
                    raise ValueError(
                        f"Unable to encode pretoken {chunk_index} "
                        f"({chunk!r}): {exc}"
                    ) from exc

            return token_ids

        if isinstance(text, (bytes, bytearray, memoryview)):
            return self._encode_bytes(bytes(text))

        raise TypeError(
            "encode() expects str, bytes, bytearray, or memoryview; "
            f"received {type(text).__name__}"
        )

    def _encode_bytes(self, data: bytes) -> list[int]:
        """
        Find the exact minimum-token segmentation of one byte sequence.

        This helper does not apply pretokenization. It is used once per
        pretoken for string input and once for the complete value for raw-byte
        input.
        """
        n = len(data)
        if n == 0:
            return []

        # best_count[i] is the minimum token count needed for data[i:].
        unreachable = n + 1
        best_count = [unreachable] * (n + 1)
        best_count[n] = 0

        # choice_id[i] and choice_end[i] reconstruct the optimal path.
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
                            and (selected_id == -1 or token_id < selected_id)
                        )
                    )

                if is_better:
                    best_count[start] = candidate_count
                    choice_id[start] = token_id
                    choice_end[start] = end
                    selected_length = candidate_length
                    selected_id = token_id

        if best_count[0] == unreachable:
            failure_offset = self._first_unencodable_offset(data, best_count)
            preview = data[failure_offset : failure_offset + 16]
            raise ValueError(
                "Input cannot be fully encoded with this vocabulary. "
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
                    "Internal tokenizer error while reconstructing the DP path"
                )

            token_ids.append(token_id)
            position = next_position

        return token_ids

    def _first_unencodable_offset(
        self, data: bytes, best_count: Sequence[int]
    ) -> int:
        """
        Return the farthest byte offset reachable from the beginning.

        ``best_count`` is accepted because this helper is called from the DP
        error path; reachability itself is recomputed forward to identify the
        most informative failure location.
        """
        del best_count  # The forward reachability pass is clearer for errors.

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

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        errors: str = "strict",
        return_bytes: bool = False,
    ) -> str | bytes:
        """
        Decode token IDs back to the original byte sequence.

        Set ``return_bytes=True`` to receive bytes directly. Otherwise the
        bytes are decoded with ``text_encoding``.
        """
        pieces: list[bytes] = []

        for position, token_id in enumerate(token_ids):
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError(
                    f"Token ID at position {position} must be int, "
                    f"not {type(token_id).__name__}"
                )

            if token_id < 0 or token_id >= len(self.id_to_token):
                raise ValueError(
                    f"Token ID at position {position} is out of range: {token_id}"
                )

            pieces.append(self.id_to_token[token_id])

        data = b"".join(pieces)
        if return_bytes:
            return data

        return data.decode(self.text_encoding, errors=errors)

    def __len__(self) -> int:
        return len(self.id_to_token)


if __name__ == "__main__":
    tokenizer = MinTokenDPTokenizer("vocab_new.txt")

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
        """
/* eslint global-require: 0 */

const { isArray } = Array;
const { entries } = Object;
const { CLIEngine } = require('eslint');

if (CLIEngine) {
  /* eslint no-inner-declarations: 0 */
  const whitespaceRules = require('./whitespaceRules');

  const baseConfig = require('.');

  const severities = ['off', 'warn', 'error'];

  function getSeverity(ruleConfig) {
    if (isArray(ruleConfig)) {
      return getSeverity(ruleConfig[0]);
    }
    if (typeof ruleConfig === 'number') {
      return severities[ruleConfig];
    }
    return ruleConfig;
  }

  function onlyErrorOnRules(rulesToError, config) {
    const errorsOnly = { ...config };
    const cli = new CLIEngine({ baseConfig: config, useEslintrc: false });
    const baseRules = cli.getConfigForFile(require.resolve('./')).rules;

    entries(baseRules).forEach((rule) => {
      const ruleName = rule[0];
      const ruleConfig = rule[1];
      const severity = getSeverity(ruleConfig);

      if (rulesToError.indexOf(ruleName) === -1 && severity === 'error') {
        if (isArray(ruleConfig)) {
          errorsOnly.rules[ruleName] = ['warn'].concat(ruleConfig.slice(1));
        } else if (typeof ruleConfig === 'number') {
          errorsOnly.rules[ruleName] = 1;
        } else {
          errorsOnly.rules[ruleName] = 'warn';
        }
      }
    });

    return errorsOnly;
  }

  module.exports = onlyErrorOnRules(whitespaceRules, baseConfig);
} else {
  const path = require('path');
  const { execSync } = require('child_process');

  // NOTE: ESLint adds runtime statistics to the output (so it's no longer JSON) if TIMING is set
  module.exports = JSON.parse(String(execSync(path.join(__dirname, 'whitespace-async.js'), {
    env: {
      ...process.env,
      TIMING: undefined,
    }
  })));
}
        """,
        """
相关网页摘要：
来源：中国外交部国家概况
URL：https://www.mfa.gov.cn/web/gjhdq_676201/gj_676203/oz_678770/1206_679110/1206x0_679112/
摘要：俄罗斯国家概况，最近更新时间为2026年3月。国名为俄罗斯联邦，亦称俄罗斯。面积1709.82万平方公里。人口1.46亿人。首都莫斯科，常住人口约1330万。国家元首为俄罗斯联邦总统弗拉基米尔·弗拉基米罗维奇·普京。官方语言为俄语，货币为卢布。
        """,
    ]

    print("\n=== Encoding / Decoding Test: Raw User Input Mode ===")
    print("In this mode, <_SOS_>/<_EOS_>/<_PAD_> are treated as normal text.")

    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        print("\nOriginal:", text)
        print("Encoded :", encoded)
        print("Decoded :", decoded)

        assert text == decoded
