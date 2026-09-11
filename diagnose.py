"""Identify what unreadable files in the dataset actually are.

    python tools/diagnose.py             # check everything Pillow cannot open
    python tools/diagnose.py --all       # check every file, not just the failures

Reads the magic bytes rather than trusting the extension, so a HEIC named
.jpg or a Git LFS pointer that never got fetched is named for what it is.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import config

SOURCE_DIR = config.DATASET_ROOT
WORK_DIR = config.WORK_DIR

IMAGE_EXTENSIONS = config.IMAGE_EXTENSIONS

# (offset, magic bytes, verdict, is_recoverable)
SIGNATURES = [
    (0, b"\xff\xd8\xff", "JPEG", True),
    (0, b"\x89PNG\r\n\x1a\n", "PNG", True),
    (0, b"GIF8", "GIF", True),
    (0, b"BM", "BMP", True),
    (0, b"II*\x00", "TIFF/DNG (little-endian)", True),
    (0, b"MM\x00*", "TIFF/DNG (big-endian)", True),
    (0, b"%PDF", "PDF", False),
    (0, b"PK\x03\x04", "ZIP archive", False),
    (4, b"ftypheic", "HEIC", True),
    (4, b"ftypheix", "HEIC", True),
    (4, b"ftyphevc", "HEIC", True),
    (4, b"ftypmif1", "HEIF", True),
    (4, b"ftypmsf1", "HEIF", True),
    (4, b"ftypavif", "AVIF", True),
    (4, b"ftypmp4", "MP4 video", False),
    (4, b"ftypqt", "QuickTime video", False),
    (8, b"WEBP", "WebP", True),
]

LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def identify(path: Path):
    """Return (verdict, recoverable, detail)."""
    size = path.stat().st_size
    if size == 0:
        return "EMPTY (0 bytes)", False, "download never wrote any data"

    with path.open("rb") as f:
        head = f.read(64)

    if head.startswith(LFS_PREFIX):
        return ("GIT LFS POINTER", False,
                "the real image was never fetched -- this is a 130-byte text stub")

    for offset, magic, verdict, recoverable in SIGNATURES:
        if head[offset:offset + len(magic)] == magic:
            detail = ""
            if verdict == "JPEG":
                with path.open("rb") as f:
                    f.seek(-2, 2)
                    if f.read(2) != b"\xff\xd9":
                        return ("JPEG (TRUNCATED)", False,
                                "starts as JPEG but has no end-of-image marker")
                detail = "valid JPEG -- if Pillow still fails, the pixel data is corrupt"
            elif verdict.startswith(("HEIC", "HEIF", "AVIF")):
                detail = "install pillow-heif, or convert with convert.py"
            return verdict, recoverable, detail

    if head[:1] in (b"<", b"{") or head[:5] == b"<!DOC":
        return ("HTML/JSON (error page)", False,
                "an HTTP error response was saved instead of the image")

    return f"UNKNOWN (starts with {head[:8]!r})", False, ""


def candidates(check_all: bool):
    failed = WORK_DIR / "failed.csv"
    if not check_all and failed.exists():
        return [SOURCE_DIR / r["path"]
                for r in csv.DictReader(failed.open(encoding="utf-8"))]

    if not check_all:
        print("No _work/failed.csv found -- checking every file instead.\n")
    paths = []
    return list(config.source_images())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="check every image, not just the ones scan failed on")
    args = parser.parse_args()

    paths = candidates(args.all)
    if not paths:
        print("Nothing to check.")
        return

    print(f"Inspecting magic bytes of {len(paths)} files...\n")
    WORK_DIR.mkdir(exist_ok=True)
    rows, tally = [], Counter()

    for path in paths:
        if not path.exists():
            continue
        verdict, recoverable, detail = identify(path)
        tally[verdict] += 1
        rows.append({
            "path": str(path.relative_to(SOURCE_DIR)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "verdict": verdict,
            "recoverable": recoverable,
            "detail": detail,
        })

    out = WORK_DIR / "diagnosis.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "bytes", "verdict", "recoverable", "detail"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {'verdict':<32} {'count':>7}")
    for verdict, count in tally.most_common():
        print(f"  {verdict:<32} {count:>7}")

    print(f"\nPer-file detail -> {out}")

    if tally["GIT LFS POINTER"]:
        print(f"\n{tally['GIT LFS POINTER']} files are LFS pointers. The image bytes "
              f"were never downloaded.\nRe-fetch them with:\n"
              f"  hf download {config.REPO_ID} --repo-type {config.REPO_TYPE} "
              f"--local-dir . --force-download")
    if tally["EMPTY (0 bytes)"]:
        print(f"\n{tally['EMPTY (0 bytes)']} files are empty -- delete them and "
              f"re-download.")
    heif = sum(v for k, v in tally.items() if k.startswith(("HEIC", "HEIF", "AVIF")))
    if heif:
        print(f"\n{heif} files are HEIC/HEIF despite their extension. "
              f"Run `pip install pillow-heif` and re-run scan.")


if __name__ == "__main__":
    main()
