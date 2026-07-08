from pathlib import Path
import re


INPUT_FILE = Path(r"C:\tokenizer_corpus\cc_news_english_clean.txt")
OUTPUT_FILE = Path(r"C:\tokenizer_corpus\cc_news_english_clean2.txt")


# Lines that are almost always boilerplate in scraped news pages.
EXACT_BOILERPLATE_LINES = {
    "follow us on twitter",
    "like us on facebook",
    "share this:",
    "share this",
    "facebook",
    "twitter",
    "whatsapp",
    "email",
    "google",
    "google+",
    "linkedin",
    "skype",
    "pocket",
    "reddit",
    "tumblr",
    "pinterest",
    "print",
    "___",
    "New York Post",
    "AFP/CC",
    "AFP"
}


# Regex patterns for common noisy lines.
BOILERPLATE_PATTERNS = [
    r"^follow us on .*$",
    r"^like us on .*$",
    r"^share this:?$",
    r"^share this: .*$",
    r"^click here to read .*$",
    r"^click here .*$",
    r"^do you have any question.*$",
    r"^do you something awesome to share.*$",
    r"^also, like us on facebook.*$",
    r"^follow us on twitter.*$",
    r"^subscribe to .*$",
    r"^sign up for .*$",
    r"^newsletter.*$",
    r"^advertisement$",
    r"^advertisements$",
    r"^read more:?$",
    r"^related articles:?$",
    r"^related posts:?$",
    r"^all rights reserved.*$",
    r"^copyright .*$",
    r"^© .*$",
]


compiled_patterns = [re.compile(p, re.IGNORECASE) for p in BOILERPLATE_PATTERNS]


def normalize_for_compare(line: str) -> str:
    """
    Normalize a line for duplicate comparison.
    This should be stricter than full text cleaning.
    """
    line = line.strip()
    line = re.sub(r"\s+", " ", line)
    return line.lower()


def is_boilerplate_line(line: str) -> bool:
    raw = line.strip()
    if not raw:
        return False

    normalized = normalize_for_compare(raw)

    if normalized in EXACT_BOILERPLATE_LINES:
        return True

    for p in compiled_patterns:
        if p.match(normalized):
            return True

    return False


def remove_social_share_blocks(lines):
    """
    Remove a whole block starting with 'Follow us...', 'Like us...', or 'Share this',
    and continuing through common social-media names.
    """
    result = []
    i = 0

    social_words = {
        "facebook",
        "twitter",
        "whatsapp",
        "email",
        "google",
        "google+",
        "linkedin",
        "skype",
        "pocket",
        "reddit",
        "tumblr",
        "pinterest",
        "print",
    }

    while i < len(lines):
        line = lines[i].strip()
        norm = normalize_for_compare(line)

        # Start of a social/share boilerplate block.
        if (
            norm.startswith("follow us on")
            or norm.startswith("like us on")
            or norm.startswith("share this")
        ):
            i += 1

            # Skip following short social-media lines.
            while i < len(lines):
                next_norm = normalize_for_compare(lines[i])

                if not next_norm:
                    i += 1
                    continue

                # Remove short social platform lines.
                if next_norm in social_words:
                    i += 1
                    continue

                # Also remove repeated social/share boilerplate patterns.
                if is_boilerplate_line(lines[i]):
                    i += 1
                    continue

                break

            continue

        result.append(lines[i])
        i += 1

    return result


def remove_adjacent_duplicate_lines(lines):
    """
    Remove directly repeated lines.
    Example:
        Title
        Title
    becomes:
        Title
    """
    result = []
    prev_norm = None

    for line in lines:
        norm = normalize_for_compare(line)

        if norm and norm == prev_norm:
            continue

        result.append(line)
        prev_norm = norm if norm else None

    return result


def remove_nearby_duplicate_short_titles(lines, window=5):
    """
    Some news dumps repeat a title within a few lines, not always exactly adjacent.
    This removes repeated short lines within a small window.
    """
    result = []
    recent = []

    for line in lines:
        norm = normalize_for_compare(line)

        # Treat relatively short lines as possible titles/menu lines.
        is_short_title_like = 8 <= len(norm) <= 160

        if is_short_title_like and norm in recent:
            continue

        result.append(line)

        if norm:
            recent.append(norm)
            if len(recent) > window:
                recent.pop(0)

    return result


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")

    # Strip line edges but keep paragraph structure.
    lines = [line.strip() for line in lines]

    # Remove exact boilerplate lines first.
    lines = [line for line in lines if not is_boilerplate_line(line)]

    # Remove social-share blocks.
    lines = remove_social_share_blocks(lines)

    # Remove adjacent duplicates.
    lines = remove_adjacent_duplicate_lines(lines)

    # Remove repeated short title-like lines within a small window.
    lines = remove_nearby_duplicate_short_titles(lines, window=5)

    # Remove very short isolated menu-like lines.
    # Keep normal short words only if you really want them.
    cleaned = []
    for line in lines:
        norm = normalize_for_compare(line)

        if not norm:
            cleaned.append("")
            continue

        # Remove one-word menu/social/navigation leftovers.
        if len(norm) <= 20 and norm in EXACT_BOILERPLATE_LINES:
            continue

        cleaned.append(line)

    text = "\n".join(cleaned)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip() + "\n"


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {INPUT_FILE}")
    text = INPUT_FILE.read_text(encoding="utf-8", errors="ignore")

    before_bytes = len(text.encode("utf-8"))
    before_lines = text.count("\n") + 1

    print("Cleaning...")
    cleaned = clean_text(text)

    after_bytes = len(cleaned.encode("utf-8"))
    after_lines = cleaned.count("\n") + 1

    OUTPUT_FILE.write_text(cleaned, encoding="utf-8", newline="\n")

    print(f"Written: {OUTPUT_FILE}")
    print(f"Before: {before_bytes / 1024 / 1024:.2f} MB, {before_lines:,} lines")
    print(f"After:  {after_bytes / 1024 / 1024:.2f} MB, {after_lines:,} lines")
    print(f"Removed: {(before_bytes - after_bytes) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()