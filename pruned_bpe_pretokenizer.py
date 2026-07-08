import unicodedata
from typing import List


def _is_latin_letter(ch: str) -> bool:
    """
    Treat ASCII letters and accented Latin letters as the same broad Latin class.

    Examples:
        a, Z, é, è, à, ç, ô, ü, ñ
    """
    if ch == "_": return True

    if "A" <= ch <= "Z" or "a" <= ch <= "z": return True

    cat = unicodedata.category(ch)

    if cat[0] not in {"L", "M"}: return False

    try:
        name = unicodedata.name(ch)
    except ValueError:
        return False

    return "LATIN" in name


def _char_kind(ch: str) -> str:
    if ch.isspace(): return "SPACE"

    if _is_latin_letter(ch): return "LATIN_WORD"

    if "0" <= ch <= "9": return "DIGIT"

    cat = unicodedata.category(ch)

    # Unicode numbers other than ASCII digits.
    if cat[0] == "N": return "DIGIT"

    # Unicode letters/marks that are not Latin:
    # Chinese, Japanese, Korean, Greek, Cyrillic, etc.
    if cat[0] in {"L", "M"}: return "UNICODE_WORD"

    # Unicode punctuation:
    # (), （）, :, ：, comma, quotes, etc.
    if cat[0] == "P": return "PUNCT"

    # Unicode symbols:
    # +, =, <, >, $, arrows, emoji, etc.
    if cat[0] == "S": return "SYMBOL"

    return "OTHER"


def _is_latin_word_char(ch: str) -> bool:
    return _char_kind(ch) == "LATIN_WORD"


def _is_digit_char(ch: str) -> bool:
    return _char_kind(ch) == "DIGIT"


def _is_unicode_word_char(ch: str) -> bool:
    return _char_kind(ch) == "UNICODE_WORD"


def _is_latin_word_or_digit(ch: str) -> bool:
    return _is_latin_word_char(ch) or _is_digit_char(ch)


def _scan_latin_word_like(text: str, start: int) -> int:
    """
    Scan Latin/code-like word span.

    Desired examples:
        abc123      -> abc123
        GPT4        -> GPT4
        IPv6        -> IPv6
        user_id     -> user_id
        GPT-4       -> GPT-4
        COVID-19    -> COVID-19
        Node.js     -> Node.js
        file.java   -> file.java
        don't       -> don't
        C'était     -> C'était
        C’était     -> C’était
        déjà        -> déjà
        Montréal    -> Montréal
        C++         -> C++
        C#          -> C#
    """
    n = len(text)
    i = start + 1

    while i < n:
        ch = text[i]

        # Latin letters, accented Latin letters, digits, underscore.
        if _is_latin_word_or_digit(ch):
            i += 1
            continue

        # Dot/apostrophe/hyphen stay inside only if followed by Latin letter/digit/underscore.
        # Examples:
        #   Node.js
        #   don't
        #   C'était
        #   C’était
        #   GPT-4
        #   COVID-19
        if ch in {".", "'", "’", "-"} and i + 1 < n and _is_latin_word_or_digit(text[i + 1]):
            i += 1
            continue

        break

    return i


def _scan_digit_like(text: str, start: int) -> int:
    """
    Scan number-first span.

    Desired examples:
        123abc           -> 123 | abc
        1949年6月24日     -> 1949 | 年 | 6 | 月 | 24 | 日
        3.14             -> 3.14
        10-20            -> 10-20
    """
    n = len(text)
    i = start + 1

    while i < n:
        ch = text[i]

        if _is_digit_char(ch):
            i += 1
            continue

        # Keep numeric forms like 3.14 or 10-20.
        if ch in {".", "-"} and i + 1 < n and _is_digit_char(text[i + 1]):
            i += 1
            continue

        break

    return i


def _scan_unicode_word_like(text: str, start: int) -> int:
    """
    Scan non-Latin Unicode word span.

    Desired examples:
        英语              -> 英语
        1949年6月24日     -> 1949 | 年 | 6 | 月 | 24 | 日
        美国共有50个州    -> 美国共有 | 50 | 个州
        中文abc           -> 中文 | abc
        日本語2025        -> 日本語 | 2025
        한국어123         -> 한국어 | 123
        Αθήνα2024         -> Αθήνα | 2024
    """
    n = len(text)
    i = start + 1

    while i < n:
        ch = text[i]

        # Continue only with non-Latin Unicode word chars.
        # Stop before digits and Latin words.
        if _is_unicode_word_char(ch):
            i += 1
            continue

        break

    return i


def _scan_file_extension(text: str, start: int) -> int:
    """
    Scan file-extension-like chunk.

    Desired examples:
        .js
        .java
        .py
        .cpp
        .html
        .json
        .csv
    """
    n = len(text)
    i = start + 2  # skip "." and first Latin letter

    while i < n and _is_latin_word_or_digit(text[i]):
        i += 1

    return i


def _scan_repeated_punct_symbol(text: str, start: int) -> int:
    """
    Group repeated same punctuation/symbol characters.

    Desired examples:
        ......  -> ......
        ------  -> ------
        ======  -> ======
        ))))    -> ))))
    """
    n = len(text)
    ch = text[start]
    i = start + 1

    while i < n and text[i] == ch:
        i += 1

    return i


def pretokenize(text: str) -> List[str]:
    """
    Unicode-aware, code-friendly, structural-boundary-aware pretokenizer.

    Main goals:
        - preserve all original text exactly
        - attach one ordinary leading space to following word/code chunks
        - keep Latin/code terms like Node.js, GPT-4, C++, C#, abc123
        - keep French/accented Latin words like C'était, déjà, Montréal
        - split number-first mixed terms like 123abc
        - split Chinese date-like text:
              1949年6月24日 -> 1949 | 年 | 6 | 月 | 24 | 日
        - split structural punctuation:
              （英语： -> （ | 英语 | ：
        - keep HTML closing opener:
              </div> -> </ | div | >
        - group repeated same punctuation:
              ...... -> ......
    """
    chunks: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        kind = _char_kind(ch)

        # Whitespace run.
        if kind == "SPACE":
            j = i + 1

            while j < n and _char_kind(text[j]) == "SPACE":
                j += 1

            whitespace = text[i:j]

            # Attach exactly one normal space to the next word/code-like chunk.
            # Do not attach tabs/newlines/multiple spaces, so indentation is preserved.
            if whitespace == " " and j < n:
                next_ch = text[j]
                next_kind = _char_kind(next_ch)

                if next_kind == "LATIN_WORD":
                    k = _scan_latin_word_like(text, j)
                    chunks.append(" " + text[j:k])
                    i = k
                    continue

                if next_kind == "DIGIT":
                    k = _scan_digit_like(text, j)
                    chunks.append(" " + text[j:k])
                    i = k
                    continue

                if next_kind == "UNICODE_WORD":
                    k = _scan_unicode_word_like(text, j)
                    chunks.append(" " + text[j:k])
                    i = k
                    continue

                # File extension with leading space:
                # " .js", " .java", " .py"
                if next_ch == "." and j + 1 < n and _is_latin_word_char(text[j + 1]):
                    k = _scan_file_extension(text, j)
                    chunks.append(" " + text[j:k])
                    i = k
                    continue

            chunks.append(whitespace)
            i = j
            continue

        # HTML/XML closing-tag opener:
        # </div> -> "</", "div", ">"
        if ch == "<" and i + 1 < n and text[i + 1] == "/":
            chunks.append("</")
            i += 2
            continue

        # Latin/code-like word.
        if kind == "LATIN_WORD":
            j = _scan_latin_word_like(text, i)
            chunks.append(text[i:j])
            i = j
            continue

        # Number-first span.
        if kind == "DIGIT":
            j = _scan_digit_like(text, i)
            chunks.append(text[i:j])
            i = j
            continue

        # Non-Latin Unicode word span, such as Chinese/Japanese/Korean/Greek.
        if kind == "UNICODE_WORD":
            j = _scan_unicode_word_like(text, i)
            chunks.append(text[i:j])
            i = j
            continue

        # File extension-like chunk at start or after punctuation:
        # .js, .java, .py, .html
        if ch == "." and i + 1 < n and _is_latin_word_char(text[i + 1]):
            j = _scan_file_extension(text, i)
            chunks.append(text[i:j])
            i = j
            continue

        # Repeated same punctuation/symbol/other:
        # ......, ------, ======, ))))
        if kind in {"PUNCT", "SYMBOL", "OTHER"}:
            j = _scan_repeated_punct_symbol(text, i)
            chunks.append(text[i:j])
            i = j
            continue

        # Safe fallback.
        chunks.append(ch)
        i += 1

    return chunks


if __name__ == "__main__":
    # A few test cases:
    tests = [
        "123abc 123abc, abc123 abc123，中文abc, abc中文",
        "<ul><li>Item 1</li><li>Item 2</li><li>Item 3</li></ul>",
        "https://www.123.com/123abc/try_my_best.jsp?abc=123&cde=456.2&hh=1+2.2+a+c",
        "def __init__(self, value: Any):\n    self.value = value",
        "def test(x):\n    return x + 1\n",
        "<div class=\"main\"></div>",
        "10+10-20*30/40**5",
        "(2^-3)-2^-3-3*2Pi----2--3",
        "看看这个，2×(4-3)/5×(7-6)/8×9=",
        "I love Node.js and Python.",
        "file.java file.c file.py file.html",
        "C++ C# GPT-4 COVID-19 don't abc-def abc+def abc*def abc/def",
        "王青（英语：Wong Ching，1949年6月24日－）",
        "Eta（大寫Η，小寫η，中文音譯：艾塔或者伊塔）",
        "这里是24个大写希腊字母： _nl_ 1、Α - Alpha _nl_ 2、Β - Beta _nl_ 3、Γ - Gamma _nl_ 4、Δ - Delta _nl_ 5、Ε - Epsilon，......",
        "与汉字数字一千、两千、三千、四千、五千、六千、七千、八千、九千、一万对应的阿拉伯数字分别是： 1000，2000，3000，4000，5000，6000，7000，8000，9000，10000。",
        "美国共有50个州，英文名分别是：Alabama, Alaska, Arizona, Arkansas, California, Colorado, Connecticut,",
        # French + English + numbers
        "C'était déjà l'été 2024 à Montréal, but the API response took 3.14 seconds and was cached in file_v2.json.",
        # Japanese + English + numbers
        "私は2025年6月10日にNode.jsで小さなweb serverを2つ作りました。The endpoint is /api/users/123.",
        # Korean + English + numbers
        "오늘은 PyTorch 모델을 3번 training하고, 결과를 output_2025.csv 파일에 저장했다.",
        # Greek + English + numbers
        "Η Αθήνα έχει 3.7 εκατομμύρια people, and the Python script parsed data_2024.json in 12 seconds.",
    ]

    for t in tests:
        print("\nInput :", repr(t))
        print("Chunks:", pretokenize(t))
        print("Joined:", "".join(pretokenize(t)))
        assert "".join(pretokenize(t)) == t