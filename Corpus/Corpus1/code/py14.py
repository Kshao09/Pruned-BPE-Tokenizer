import re
import json
import shutil
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List

import requests
import py7zr
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names


# ============================================================
# User settings
# ============================================================

OUTPUT_DIR = Path("C:/tokenizer_corpus/news_data")

# Rough output targets. Change these numbers as needed.
CC_NEWS_TARGET_MB = 300          # English CC-News
XLSUM_EN_TARGET_MB = 50         # BBC English news
XLSUM_ZH_TARGET_MB = 50         # BBC Chinese news, if available
THUCNEWS_TARGET_MB = 200        # Chinese THUCNews

HF_CACHE_DIR = "C:/hf_datasets_cache"

# Figshare article id for THUCNews mirror
THUCNEWS_FIGSHARE_ARTICLE_ID = "28279964"


# ============================================================
# General helpers
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utf8_size(text: str) -> int:
    return len(text.encode("utf-8"))


def clean_text(text: str) -> str:
    """
    Light cleaning only. This keeps the text mostly natural for tokenizer training.
    """
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip each line but do not destroy paragraph structure.
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines).strip()

    return text


def pick_text_field(row: Dict[str, Any]) -> str:
    """
    Try common text/article/content fields from Hugging Face datasets.
    """
    preferred_fields = [
        "text",
        "article",
        "document",
        "maintext",
        "content",
        "body",
        "description",
        "summary",
    ]

    parts = []

    # Prefer article-like fields first.
    for key in preferred_fields:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
            break

    # Optionally add title at the beginning if available.
    title = row.get("title")
    if isinstance(title, str) and title.strip():
        title = title.strip()
        if parts:
            return title + "\n" + parts[0]
        return title

    return parts[0] if parts else ""


def write_texts_to_file(
    texts: Iterable[str],
    output_path: Path,
    target_mb: Optional[int] = None,
    min_chars: int = 200,
) -> int:
    """
    Write texts into one UTF-8 text file until target_mb is reached.
    Returns bytes written.
    """
    ensure_dir(output_path.parent)

    target_bytes = None if target_mb is None else target_mb * 1024 * 1024
    written = 0
    count = 0

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for text in texts:
            text = clean_text(text)
            if len(text) < min_chars:
                continue

            b = utf8_size(text)
            if target_bytes is not None and written >= target_bytes:
                break

            f.write(text)
            f.write("\n\n")
            written += b + 2
            count += 1

            if count % 1000 == 0:
                print(f"  wrote {count:,} records, {written / 1024 / 1024:.1f} MB")

    print(f"Finished: {output_path}")
    print(f"  records: {count:,}")
    print(f"  size: {written / 1024 / 1024:.1f} MB")
    return written


def load_hf_dataset_streaming_or_normal(
    dataset_name: str,
    config_name: Optional[str],
    split: str,
):
    """
    Try streaming first. If the dataset does not support streaming,
    fall back to normal loading.
    """
    try:
        print(f"Loading {dataset_name}, config={config_name}, split={split}, streaming=True")
        if config_name is None:
            return load_dataset(
                dataset_name,
                split=split,
                streaming=True,
                cache_dir=HF_CACHE_DIR,
            )
        return load_dataset(
            dataset_name,
            config_name,
            split=split,
            streaming=True,
            cache_dir=HF_CACHE_DIR,
        )
    except Exception as e:
        print("Streaming failed. Falling back to normal download/cache loading.")
        print(f"Reason: {e}")

        if config_name is None:
            return load_dataset(
                dataset_name,
                split=split,
                cache_dir=HF_CACHE_DIR,
            )
        return load_dataset(
            dataset_name,
            config_name,
            split=split,
            cache_dir=HF_CACHE_DIR,
        )


# ============================================================
# 1. CC-News English
# ============================================================

def export_cc_news_english() -> None:
    """
    Exports English CC-News from Hugging Face to a plain text file.
    """
    output_path = OUTPUT_DIR / "cc_news_english.txt"

    ds = load_hf_dataset_streaming_or_normal(
        dataset_name="vblagoje/cc_news",
        config_name="plain_text",
        split="train",
    )

    def text_iter():
        for row in ds:
            yield pick_text_field(row)

    write_texts_to_file(
        texts=text_iter(),
        output_path=output_path,
        target_mb=CC_NEWS_TARGET_MB,
        min_chars=300,
    )


# ============================================================
# 2. XL-Sum BBC news
# ============================================================

def find_xlsum_config(language_keywords: List[str]) -> Optional[str]:
    """
    Finds an XL-Sum config by matching language keywords.
    The actual config names can differ between GEM/xlsum and csebuetnlp/xlsum,
    so this function discovers them dynamically.
    """
    dataset_name_candidates = [
        "GEM/xlsum",
        "csebuetnlp/xlsum",
    ]

    for dataset_name in dataset_name_candidates:
        try:
            configs = get_dataset_config_names(dataset_name)
            print(f"\nAvailable configs for {dataset_name}:")
            print(configs[:50], "..." if len(configs) > 50 else "")

            lowered = [(c, c.lower()) for c in configs]

            for keyword in language_keywords:
                keyword = keyword.lower()
                for original, low in lowered:
                    if keyword == low or keyword in low:
                        return dataset_name + "::" + original

        except Exception as e:
            print(f"Could not list configs for {dataset_name}: {e}")

    return None


def export_xlsum_language(
    language_name: str,
    language_keywords: List[str],
    target_mb: int,
    output_filename: str,
) -> None:
    """
    Exports one XL-Sum language subset to plain text.
    """
    found = find_xlsum_config(language_keywords)

    if found is None:
        print(f"\nCould not find XL-Sum config for {language_name}. Skipping.")
        print("You can manually inspect configs by running:")
        print("  from datasets import get_dataset_config_names")
        print("  print(get_dataset_config_names('GEM/xlsum'))")
        print("  print(get_dataset_config_names('csebuetnlp/xlsum'))")
        return

    dataset_name, config_name = found.split("::", 1)
    print(f"\nUsing XL-Sum config for {language_name}: {dataset_name}, {config_name}")

    # Usually XL-Sum has train/validation/test.
    ds = load_hf_dataset_streaming_or_normal(
        dataset_name=dataset_name,
        config_name=config_name,
        split="train",
    )

    output_path = OUTPUT_DIR / output_filename

    def text_iter():
        for row in ds:
            # For summarization datasets, article/text is useful.
            # We avoid using only the summary because it is too short and summary-like.
            text = pick_text_field(row)
            yield text

    write_texts_to_file(
        texts=text_iter(),
        output_path=output_path,
        target_mb=target_mb,
        min_chars=200,
    )


def export_xlsum_english_and_chinese() -> None:
    """
    Attempts to export English and Chinese XL-Sum subsets.
    Chinese may or may not exist depending on the Hugging Face version.
    """
    export_xlsum_language(
        language_name="English",
        language_keywords=["english", "en"],
        target_mb=XLSUM_EN_TARGET_MB,
        output_filename="xlsum_bbc_english.txt",
    )

    export_xlsum_language(
        language_name="Chinese",
        language_keywords=[
            "chinese",
            "zh",
            "zh-cn",
            "chinese_simplified",
            "simplified_chinese",
        ],
        target_mb=XLSUM_ZH_TARGET_MB,
        output_filename="xlsum_bbc_chinese.txt",
    )


# ============================================================
# 3. THUCNews Chinese news from Figshare
# ============================================================

def get_figshare_files(article_id: str) -> List[Dict[str, Any]]:
    """
    Query Figshare public API for files attached to an article.
    """
    api_url = f"https://api.figshare.com/v2/articles/{article_id}"
    print(f"Querying Figshare API: {api_url}")

    r = requests.get(api_url, timeout=60)
    r.raise_for_status()

    data = r.json()
    files = data.get("files", [])
    print("Figshare files:")
    for f in files:
        print(f"  - {f.get('name')}  {f.get('size')} bytes")
    return files


def download_file(url: str, output_path: Path) -> None:
    """
    Download a file with progress bar.
    """
    ensure_dir(output_path.parent)

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Already exists, skipping download: {output_path}")
        return

    print(f"Downloading:\n  {url}\n  -> {output_path}")

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        with output_path.open("wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=output_path.name,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    """
    Extract .7z archive.
    """
    ensure_dir(extract_dir)

    marker = extract_dir / ".extracted"
    if marker.exists():
        print(f"Already extracted, skipping: {extract_dir}")
        return

    print(f"Extracting {archive_path} -> {extract_dir}")

    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extractall(path=extract_dir)

    marker.write_text("done", encoding="utf-8")
    print("Extraction finished.")


def iter_text_files(root_dir: Path) -> Iterable[Path]:
    """
    Recursively iterate text-like files.
    """
    for path in root_dir.rglob("*"):
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix in [".txt", ".text"]:
                yield path


def read_text_file_guess_encoding(path: Path) -> str:
    """
    THUCNews should be UTF-8, but this makes reading more robust.
    """
    encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc, errors="strict")
        except UnicodeDecodeError:
            continue

    return path.read_text(encoding="utf-8", errors="ignore")


def export_thucnews_chinese() -> None:
    """
    Downloads THUCNews mirror from Figshare, extracts it, and merges .txt files.
    """
    raw_dir = OUTPUT_DIR / "_raw_thucnews"
    extract_dir = raw_dir / "extracted"
    output_path = OUTPUT_DIR / "thucnews_chinese.txt"

    ensure_dir(raw_dir)

    files = get_figshare_files(THUCNEWS_FIGSHARE_ARTICLE_ID)
    if not files:
        print("No files found from Figshare article. Cannot download THUCNews.")
        return

    # Prefer .7z, .zip, or the largest file.
    selected = None
    for f in files:
        name = f.get("name", "").lower()
        if name.endswith(".7z"):
            selected = f
            break

    if selected is None:
        for f in files:
            name = f.get("name", "").lower()
            if name.endswith(".zip"):
                selected = f
                break

    if selected is None:
        selected = max(files, key=lambda x: int(x.get("size", 0)))

    file_name = selected.get("name", "thucnews_download")
    download_url = selected.get("download_url")

    if not download_url:
        print("No download_url found in Figshare file metadata.")
        print(json.dumps(selected, indent=2, ensure_ascii=False))
        return

    archive_path = raw_dir / file_name
    download_file(download_url, archive_path)

    if archive_path.suffix.lower() == ".7z":
        extract_archive(archive_path, extract_dir)
    elif archive_path.suffix.lower() == ".zip":
        import zipfile
        marker = extract_dir / ".extracted"
        if not marker.exists():
            print(f"Extracting {archive_path} -> {extract_dir}")
            with zipfile.ZipFile(archive_path, "r") as z:
                z.extractall(extract_dir)
            marker.write_text("done", encoding="utf-8")
    else:
        print(f"Unknown archive type: {archive_path}")
        print("Please extract it manually, then place extracted files under:")
        print(extract_dir)
        return

    text_paths = list(iter_text_files(extract_dir))
    print(f"Found {len(text_paths):,} text files under {extract_dir}")

    def text_iter():
        for p in text_paths:
            text = read_text_file_guess_encoding(p)
            yield text

    write_texts_to_file(
        texts=text_iter(),
        output_path=output_path,
        target_mb=THUCNEWS_TARGET_MB,
        min_chars=100,
    )


# ============================================================
# 4. Optional: combine selected files into one news corpus
# ============================================================

def combine_outputs() -> None:
    combined_path = OUTPUT_DIR / "combined_news_corpus.txt"

    input_files = [
        OUTPUT_DIR / "cc_news_english.txt",
        OUTPUT_DIR / "xlsum_bbc_english.txt",
        OUTPUT_DIR / "xlsum_bbc_chinese.txt",
        OUTPUT_DIR / "thucnews_chinese.txt",
    ]

    print(f"\nCombining files -> {combined_path}")

    with combined_path.open("w", encoding="utf-8", newline="\n") as out:
        for path in input_files:
            if not path.exists():
                print(f"Skipping missing file: {path}")
                continue

            print(f"Adding: {path}")
            out.write(f"\n\n===== SOURCE: {path.name} =====\n\n")
            with path.open("r", encoding="utf-8", errors="ignore") as inp:
                shutil.copyfileobj(inp, out)

    print(f"Combined file written: {combined_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    ensure_dir(OUTPUT_DIR)

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Hugging Face cache: {HF_CACHE_DIR}")

    print("\n=== 1. Exporting CC-News English ===")
    export_cc_news_english()

    print("\n=== 2. Exporting XL-Sum English and Chinese if available ===")
    export_xlsum_english_and_chinese()

    print("\n=== 3. Exporting THUCNews Chinese ===")
    export_thucnews_chinese()

    print("\n=== 4. Combining outputs ===")
    combine_outputs()

    print("\nAll done.")


if __name__ == "__main__":
    main()