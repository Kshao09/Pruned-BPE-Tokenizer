from __future__ import annotations

from pathlib import Path


def convert_vocab(
    input_path: str | Path = "vocab_old.txt",
    output_path: str | Path = "vocab_new.txt",
) -> int:
    """
    Convert the old four-column vocabulary into the one-token-per-line format
    expected by MaxLenTokenizer.

    The third tab-separated column is copied exactly as written.

    Examples of output lines:

        'a'
        ' hello'
        '中国'
        b'\\xe4\\xb8'
        b'\\xff'

    The original line order—and therefore the token IDs—is preserved.

    Returns:
        The number of tokens written.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input vocabulary not found: {input_path}")

    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different")

    converted_tokens: list[str] = []

    with input_path.open("r", encoding="utf-8", newline="") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            # Remove only the physical line ending.
            # Do not use strip(), because spaces inside token literals matter.
            line = raw_line.rstrip("\r\n")

            if not line:
                continue

            parts = line.split("\t", 3)

            if len(parts) != 4:
                raise ValueError(
                    f"{input_path}:{line_number}: "
                    f"expected 4 tab-separated columns, "
                    f"but found {len(parts)}"
                )

            token_literal = parts[2]

            # token_literal already contains the correct quotes and,
            # when applicable, the b prefix.
            converted_tokens.append(token_literal)

    if not converted_tokens:
        raise ValueError(f"Input vocabulary is empty: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(output_path.name + ".tmp")

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for token_literal in converted_tokens:
                # Do not use repr(token_literal).
                # The token literal is already formatted correctly.
                output_file.write(token_literal)
                output_file.write("\n")

        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return len(converted_tokens)


if __name__ == "__main__":
    token_count = convert_vocab("vocab_old.txt","vocab_new.txt")

    print(f"Converted {token_count:,} tokens.")