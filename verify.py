"""Decide whether a broken local file is a bad download or bad upstream data.

    python tools/verify.py                     # check everything in _work/failed.csv
    python tools/verify.py Plastic/abc.jpg     # check specific paths

For each file it compares the local size and SHA-256 against the LFS object
the Hub says should be there:

  REDOWNLOAD  local bytes differ from the Hub -> your copy is incomplete
  FAITHFUL    local bytes match the Hub exactly -> your copy is correct, so a
              file that still will not decode is broken in the dataset itself.
              Re-downloading changes nothing. Drop it.
  MISSING     the path is not in the repo at all

Requires `hf auth login` for a private repo.
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

import config

SOURCE_DIR = config.DATASET_ROOT
WORK_DIR = config.WORK_DIR

REPO_ID = config.REPO_ID
REPO_TYPE = config.REPO_TYPE
BATCH = 200


def sha256_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*",
                        help="repo-relative paths; defaults to _work/failed.csv")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--all", action="store_true",
                        help="check every local image, not just the failures "
                             "(full SHA-256 pass -- slow, but decisive)")
    args = parser.parse_args()

    paths = [p.replace("\\", "/") for p in args.paths]
    if args.all:
        paths = [
            str(p.relative_to(SOURCE_DIR)).replace("\\", "/")
            for p in config.source_images()
        ]
    elif not paths:
        failed = WORK_DIR / "failed.csv"
        if not failed.exists():
            sys.exit("No paths given and no _work/failed.csv -- run `tools/rebuild.py scan` first.")
        paths = [r["path"] for r in csv.DictReader(failed.open(encoding="utf-8"))]

    print(f"Checking {len(paths)} file(s) against {args.repo_id}...\n")
    api = HfApi()

    remote = {}
    for start in range(0, len(paths), BATCH):
        chunk = paths[start:start + BATCH]
        try:
            for info in api.get_paths_info(
                args.repo_id, chunk, repo_type=REPO_TYPE, expand=True
            ):
                remote[info.path] = info
        except HfHubHTTPError as exc:
            sys.exit(f"Hub request failed: {exc}\nAre you logged in? Try `hf auth login`.")

    rows = []
    verdicts = {"REDOWNLOAD": 0, "FAITHFUL": 0, "MISSING": 0}

    for path in paths:
        local = SOURCE_DIR / path
        info = remote.get(path)

        if info is None:
            verdict, detail = "MISSING", "not present in the repo"
            local_size = local.stat().st_size if local.exists() else 0
            expected_size = 0
        elif not local.exists():
            verdict, detail = "REDOWNLOAD", "file absent locally"
            local_size, expected_size = 0, info.size
        else:
            local_size = local.stat().st_size
            expected_size = info.size
            expected_sha = info.lfs.sha256 if info.lfs else None

            if local_size != expected_size:
                verdict = "REDOWNLOAD"
                detail = f"size mismatch: have {local_size}, expected {expected_size}"
            elif expected_sha is None:
                verdict = "FAITHFUL"
                detail = "size matches (file is not LFS-tracked, no hash to compare)"
            elif sha256_of(local) == expected_sha:
                verdict = "FAITHFUL"
                detail = "byte-for-byte identical to the Hub copy"
            else:
                verdict = "REDOWNLOAD"
                detail = "same size but different hash -- corrupted in transit"

        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        rows.append({
            "path": path, "verdict": verdict,
            "local_bytes": local_size, "expected_bytes": expected_size,
            "detail": detail,
        })
        if not args.all or verdict != "FAITHFUL":
            print(f"  {verdict:<11} {path}")
            print(f"              {detail}")
        elif len(rows) % 500 == 0:
            print(f"  ...{len(rows)}/{len(paths)} checked")

    out = WORK_DIR / "verify.csv"
    WORK_DIR.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "verdict", "local_bytes", "expected_bytes", "detail"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  {'verdict':<12} {'count':>6}")
    for verdict, count in verdicts.items():
        if count:
            print(f"  {verdict:<12} {count:>6}")
    print(f"\nDetail -> {out}")

    if verdicts.get("REDOWNLOAD"):
        print("\nRe-fetch the incomplete files:")
        print(f"  hf download {args.repo_id} --repo-type dataset --local-dir . "
              f"--force-download \\")
        print("    " + " ".join(
            f'--include "{r["path"]}"' for r in rows if r["verdict"] == "REDOWNLOAD"
        )[:400] + (" ..." if verdicts["REDOWNLOAD"] > 5 else ""))
    if verdicts.get("FAITHFUL"):
        print(f"\n{verdicts['FAITHFUL']} file(s) are broken in the dataset itself. "
              f"Re-downloading will not help.\nDrop them -- `rebuild.py build` already "
              f"skips them, and the upload replaces the repo contents,\nso they "
              f"disappear from the published dataset automatically.")


if __name__ == "__main__":
    main()
