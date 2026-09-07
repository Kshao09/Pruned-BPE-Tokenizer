import ast


def parse_token(token_field: str) -> bytes:
    token_field = token_field.strip()

    value = ast.literal_eval(token_field)

    if isinstance(value, bytes):
        return value

    if isinstance(value, str):
        return value.encode("utf-8")

    raise ValueError(f"Unsupported token format: {token_field}")


def load_vocab_tokens(vocab_file: str):
    tokens = {}

    with open(vocab_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip("\n")

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 4:
                raise ValueError(
                    f"Expected 4 tab-separated columns at line {line_num}:\n"
                    f"{line}"
                )

            token_id = int(parts[0])
            token = parse_token(parts[2])

            tokens[token] = token_id

    return tokens


def display_token(token: bytes) -> str:
    try:
        return repr(token.decode("utf-8"))
    except UnicodeDecodeError:
        return repr(token)


def main():
    vocab_file_1 = r"C:\Python\Tokenize\experiments\c2\12k\vocab_mingram_pp_12k_f5.txt"
    vocab_file_2 = r"C:\Python\Tokenize\experiments\c2\12k\vocab_pruned_2sp_s12k_1.07_0.7.txt"

    vocab1 = load_vocab_tokens(vocab_file_1)
    vocab2 = load_vocab_tokens(vocab_file_2)

    tokens1 = set(vocab1.keys())
    tokens2 = set(vocab2.keys())

    common = tokens1 & tokens2
    only1 = tokens1 - tokens2
    only2 = tokens2 - tokens1

    print("=== Vocabulary Comparison ===")
    print(f"Vocab 1 tokens : {len(tokens1):,}")
    print(f"Vocab 2 tokens : {len(tokens2):,}")
    print(f"Common tokens  : {len(common):,}")
    print(f"Only in vocab1 : {len(only1):,}")
    print(f"Only in vocab2 : {len(only2):,}")

    print("\n=== Only in Vocab 1 ===")
    for token in sorted(only1):
        print(
            f"ID={vocab1[token]:5d}  "
            f"token={display_token(token)}  "
            f"bytes={token!r}"
        )

    print("\n=== Only in Vocab 2 ===")
    for token in sorted(only2):
        print(
            f"ID={vocab2[token]:5d}  "
            f"token={display_token(token)}  "
            f"bytes={token!r}"
        )


if __name__ == "__main__":
    main()