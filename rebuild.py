"""Rebuild the published dataset: deduplicate, downscale, and re-upload.

Run every command from the dataset root, not from tools/:

    python tools/rebuild.py ingest <folder> --label Plastic-Bottle
    python tools/rebuild.py scan       # hash + measure every image -> _work/scan.csv
    python tools/rebuild.py resync     # re-point scan.csv after moving files
    python tools/rebuild.py dedup      # cluster near-duplicates
    python tools/rebuild.py build      # resize survivors -> _build/
    python tools/rebuild.py upload     # push _build/ to the Hub
    python tools/rebuild.py squash     # reclaim old storage, after verifying

See tools/README.md for the full workflow.

Source images are never modified. `dedup` only writes a report unless you
pass --apply, which moves duplicates to _duplicates/ (still not deleted).

Output JPEGs carry no EXIF at all: orientation is baked into the pixels, and
capture time is preserved in metadata.csv instead. That makes the double-
rotation bug structurally impossible and strips GPS coordinates by construction.
"""

import argparse
import contextlib
import csv
import datetime
import hashlib
import random
import shutil
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ExifTags, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass  # .heic sources will be reported as unreadable by `scan`

try:
    import rawpy
except ImportError:
    rawpy = None  # .dng sources will be reported as unreadable by `scan`

# Configuration -- all tunables live in .env, see tools/.env.example
import config

SOURCE_DIR = config.DATASET_ROOT
WORK_DIR = config.WORK_DIR
BUILD_DIR = config.BUILD_DIR
QUARANTINE_DIR = config.QUARANTINE_DIR

MAX_EDGE = config.MAX_EDGE
JPEG_QUALITY = config.JPEG_QUALITY

DUP_DISTANCE = config.DUP_DISTANCE      # <= this is "the same photo"
GROUP_DISTANCE = config.GROUP_DISTANCE  # <= this is "the same object"

REPO_ID = config.REPO_ID
REPO_TYPE = config.REPO_TYPE

IMAGE_EXTENSIONS = config.IMAGE_EXTENSIONS
RAW_EXTENSIONS = config.RAW_EXTENSIONS
EXIF_DATE_TAGS = {"DateTimeOriginal", "DateTimeDigitized", "DateTime"}
SWAPPED_ORIENTATIONS = {5, 6, 7, 8}
GREGORIAN_OFFSET = 0x01B21DD213814000

RESERVED_DIRS = config.RESERVED_DIRS


@contextlib.contextmanager
def open_image(path: Path):
    """Open any supported source as a PIL image.

    RAW files are demosaiced with rawpy (Pillow cannot read them); everything
    else goes through Pillow directly. Used by every stage so that adding a
    format only has to be handled here.
    """
    if path.suffix.lower() in RAW_EXTENSIONS:
        if rawpy is None:
            raise RuntimeError("rawpy is not installed; cannot read RAW files")
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True)
        image = Image.fromarray(rgb)
        try:
            yield image
        finally:
            image.close()
    else:
        with Image.open(path) as image:
            yield image


class_dirs = config.class_dirs
source_images = config.source_images


def uuid6_to_datetime(stem: str):
    """Recover the capture time embedded in a UUIDv6 filename, if it is one."""
    try:
        value = uuid.UUID(stem)
    except (ValueError, AttributeError):
        return None
    if value.version != 6:
        return None
    i = value.int
    intervals = (
        (((i >> 96) & 0xFFFFFFFF) << 28)
        | (((i >> 80) & 0xFFFF) << 12)
        | ((i >> 64) & 0x0FFF)
    )
    seconds = (intervals - GREGORIAN_OFFSET) / 10_000_000
    try:
        return datetime.datetime.fromtimestamp(seconds)
    except (OverflowError, OSError, ValueError):
        return None


def capture_time(img, path: Path) -> datetime.datetime:
    """EXIF capture date, else the UUIDv6 timestamp, else filesystem mtime."""
    try:
        exif_data = img.getexif()
        for tag_id, value in exif_data.items():
            if ExifTags.TAGS.get(tag_id) in EXIF_DATE_TAGS and value:
                return datetime.datetime.strptime(str(value).strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass

    from_uuid = uuid6_to_datetime(path.stem)
    if from_uuid is not None:
        return from_uuid

    return datetime.datetime.fromtimestamp(path.stat().st_mtime)


def make_uuid6(dt: datetime.datetime) -> str:
    """A UUIDv6 whose embedded timestamp is the image's capture time.

    Sorts chronologically, is globally unique, and lets `resync` recover a
    file's identity after it has been moved between class folders.
    """
    intervals = int(dt.timestamp() * 10_000_000) + GREGORIAN_OFFSET
    time_low_and_version = (0x6 << 12) | (intervals & 0x0FFF)
    clock_seq_and_node = (
        random.getrandbits(64) & 0x3FFFFFFFFFFFFFFF | 0x8000000000000000
    )
    value = (
        (((intervals >> 28) & 0xFFFFFFFF) << 96)
        | (((intervals >> 12) & 0xFFFF) << 80)
        | (time_low_and_version << 64)
        | clock_seq_and_node
    )
    return str(uuid.UUID(int=value))


# ---------------------------------------------------------
# Stage 0: ingest
# ---------------------------------------------------------

def stage_ingest(args):
    """Bring new photos into a class folder under UUIDv6 names.

    Replaces the old rename.py/convert.py pair. Files keep their original
    format and full resolution -- transcoding and downscaling happen later in
    `build`, so the class folders stay the archival master. RAW is kept as RAW.
    """
    src_dir = Path(args.source).resolve()
    if not src_dir.is_dir():
        sys.exit(f"{src_dir} is not a directory")

    dest = SOURCE_DIR / args.label
    if not args.create and not dest.is_dir():
        sys.exit(f"{dest.name}/ does not exist. Pass --create to make it, "
                 f"or check the label spelling.")
    dest.mkdir(parents=True, exist_ok=True)

    incoming = sorted(
        p for p in src_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not incoming:
        sys.exit(f"No supported images in {src_dir} "
                 f"({', '.join(sorted(IMAGE_EXTENSIONS))})")

    # Refuse to re-ingest anything already in the dataset, by content hash.
    existing = set()
    scan_path = WORK_DIR / "scan.csv"
    if scan_path.exists():
        existing = {r["sha256"] for r in csv.DictReader(scan_path.open(encoding="utf-8"))}
        print(f"Checking against {len(existing)} hashes already in scan.csv")

    print(f"Ingesting {len(incoming)} file(s) from {src_dir} -> {dest.name}/")
    added = skipped = failed = 0
    for path in incoming:
        try:
            hasher = hashlib.sha256()
            with path.open("rb") as f:
                while chunk := f.read(1024 * 1024):
                    hasher.update(chunk)
            sha = hasher.hexdigest()
            if sha in existing:
                skipped += 1
                continue

            with open_image(path) as img:
                img.draft("RGB", (64, 64))   # cheap validity check
                taken = capture_time(img, path)

            suffix = path.suffix.lower()
            if suffix == ".jpeg":
                suffix = ".jpg"
            target = dest / f"{make_uuid6(taken)}{suffix}"
            while target.exists():
                target = dest / f"{make_uuid6(taken)}{suffix}"

            if args.move:
                shutil.move(str(path), str(target))
            else:
                shutil.copy2(path, target)
            existing.add(sha)
            added += 1
        except Exception as exc:
            print(f"  ! {path.name}: {exc}", file=sys.stderr)
            failed += 1

    print(f"\n  added   {added}")
    print(f"  skipped {skipped} (already in the dataset)")
    print(f"  failed  {failed}")
    if added:
        print(f"\nNext: `python tools/rebuild.py resync` to fold them into scan.csv "
              f"without re-reading everything.")


# ---------------------------------------------------------
# Stage 1: scan
# ---------------------------------------------------------

def scan_one(path_str: str):
    path = Path(path_str)
    try:
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)
        sha = hasher.hexdigest()

        with open_image(path) as img:
            width, height = img.size
            orientation = img.getexif().get(274)
            if orientation in SWAPPED_ORIENTATIONS:
                width, height = height, width

            taken = capture_time(img, path)

            # draft() decodes JPEGs at a reduced DCT scale -- much faster, and
            # phash downsamples to 32x32 anyway.
            img.draft("RGB", (512, 512))
            phash = imagehash.phash(ImageOps.exif_transpose(img))

        return {
            "path": str(path.relative_to(SOURCE_DIR)).replace("\\", "/"),
            "label": path.relative_to(SOURCE_DIR).parts[0],
            "sha256": sha,
            "phash": str(phash),
            "width": width,
            "height": height,
            "bytes": path.stat().st_size,
            "captured": taken.isoformat(timespec="seconds"),
        }
    except Exception as exc:
        print(f"  ! failed {path.name}: {exc}", file=sys.stderr)
        return {
            "path": str(path.relative_to(SOURCE_DIR)).replace("\\", "/"),
            "error": f"{type(exc).__name__}: {exc}",
            "bytes": path.stat().st_size if path.exists() else 0,
        }


def stage_scan(args):
    WORK_DIR.mkdir(exist_ok=True)
    paths = [str(p) for p in source_images()]
    if not paths:
        sys.exit("No source images found. Download the dataset first.")

    print(f"Scanning {len(paths)} images with {args.workers} workers...")
    rows, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, row in enumerate(pool.map(scan_one, paths, chunksize=16), start=1):
            (failures if "error" in row else rows).append(row)
            if n % 500 == 0:
                print(f"  {n}/{len(paths)}")

    if failures:
        bad = WORK_DIR / "failed.csv"
        with bad.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "bytes", "error"])
            writer.writeheader()
            writer.writerows(failures)
        print(f"\n!! {len(failures)} unreadable files -> {bad}")
        print("   Run `python tools/diagnose.py` to find out what they actually are.")

    if not rows:
        sys.exit("No images could be read at all.")

    out = WORK_DIR / "scan.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total_gb = sum(r["bytes"] for r in rows) / 1024**3
    print(f"\nScanned {len(rows)} images ({total_gb:.1f} GB) -> {out}")
    by_label = {}
    for r in rows:
        by_label[r["label"]] = by_label.get(r["label"], 0) + 1
    for label in sorted(by_label):
        print(f"  {label:<20} {by_label[label]:>6}")


# ---------------------------------------------------------
# Stage 2: dedup
# ---------------------------------------------------------

# uint8 keeps the (chunk x N x 8) lookup result small: at 24k images an int64
# table would balloon each chunk to ~400MB instead of ~50MB.
POPCOUNT = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1
).sum(axis=1).astype(np.uint8)


def cluster_by_distance(hashes: np.ndarray, threshold: int):
    """Union-find over all pairs within `threshold` Hamming distance."""
    n = len(hashes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    chunk = 256
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        xor = np.bitwise_xor(hashes[start:stop, None, :], hashes[None, :, :])
        dist = POPCOUNT[xor].sum(axis=2)
        # only compare against later images to halve the work
        for local, i in enumerate(range(start, stop)):
            for j in np.nonzero(dist[local, i + 1:] <= threshold)[0]:
                union(i, i + 1 + int(j))

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return groups


def stage_resync(args):
    """Re-point scan.csv at where files actually live now, without re-reading them.

    Moving files between class folders (the presort, or a manual pass) changes
    their paths but not their contents, so the expensive part of `scan` -- the
    hash and the decode -- is still valid. Filenames are unique UUIDs, so they
    are a safe join key. Any file with no existing row is scanned for real and
    appended, so the result always matches what is on disk.
    """
    scan_path = WORK_DIR / "scan.csv"
    if not scan_path.exists():
        sys.exit("No _work/scan.csv -- run `scan` first.")

    rows = list(csv.DictReader(scan_path.open(encoding="utf-8")))
    by_stem = {Path(r["path"]).stem: r for r in rows}

    on_disk = {}
    for path in source_images():
        rel = str(path.relative_to(SOURCE_DIR)).replace("\\", "/")
        on_disk[Path(rel).stem] = rel

    kept, moved, new_stems = [], 0, []
    for stem, rel in on_disk.items():
        row = by_stem.get(stem)
        if row is None:
            new_stems.append(rel)
            continue
        if row["path"] != rel:
            row["path"] = rel
            moved += 1
        row["label"] = Path(rel).parts[0]
        kept.append(row)

    gone = len(rows) - len(kept)
    print(f"scan.csv rows        : {len(rows)}")
    print(f"files on disk        : {len(on_disk)}")
    print(f"  re-pointed (moved) : {moved}")
    print(f"  dropped (not on disk): {gone}")
    print(f"  new, need scanning : {len(new_stems)}")

    if new_stems:
        print(f"\nScanning {len(new_stems)} new file(s)...")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(scan_one, [str(SOURCE_DIR / r) for r in new_stems]):
                if "error" in row:
                    print(f"  ! {row['path']}: {row['error']}")
                else:
                    kept.append(row)

    kept.sort(key=lambda r: r["path"])
    backup = WORK_DIR / "scan.previous.csv"
    shutil.copy2(scan_path, backup)
    with scan_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(kept[0].keys()))
        writer.writeheader()
        writer.writerows(kept)

    # groups.csv carries group_id for leakage-safe splits; re-point it too and
    # clear the automatic dropped flags, which resync has now made redundant.
    groups_path = WORK_DIR / "groups.csv"
    if groups_path.exists():
        groups = list(csv.DictReader(groups_path.open(encoding="utf-8")))
        stem_to_group = {Path(g["path"]).stem: g["group_id"] for g in groups}
        with groups_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "group_id", "dropped"])
            for row in kept:
                stem = Path(row["path"]).stem
                writer.writerow([row["path"], stem_to_group.get(stem, ""), False])
        print(f"Re-pointed {groups_path.name} as well")

    print(f"\nscan.csv now lists {len(kept)} files, matching disk. "
          f"(previous version kept as {backup.name})")
    labels = {}
    for r in kept:
        labels[r["label"]] = labels.get(r["label"], 0) + 1
    for label in sorted(labels):
        print(f"  {label:<22}{labels[label]:>7}")


def stage_dedup(args):
    rows = list(csv.DictReader((WORK_DIR / "scan.csv").open(encoding="utf-8")))
    if not rows:
        sys.exit("scan.csv is empty -- run `scan` first.")

    hashes = np.stack([
        np.frombuffer(bytes.fromhex(r["phash"]), dtype=np.uint8) for r in rows
    ])
    print(f"Clustering {len(rows)} perceptual hashes...")

    # Duplicates: same photo, possibly at a different size. Keep the largest.
    dup_clusters = cluster_by_distance(hashes, DUP_DISTANCE)
    drop = []
    for members in dup_clusters.values():
        if len(members) < 2:
            continue
        members.sort(
            key=lambda i: (int(rows[i]["width"]) * int(rows[i]["height"]), int(rows[i]["bytes"])),
            reverse=True,
        )
        keeper = members[0]
        for i in members[1:]:
            drop.append((i, keeper))

    dropped = {i for i, _ in drop}

    # Groups: same physical object across shots. Kept, but tagged so that a
    # random train/test split cannot leak one shot into both halves.
    group_clusters = cluster_by_distance(hashes, GROUP_DISTANCE)
    group_of = {}
    for gid, (_, members) in enumerate(sorted(group_clusters.items())):
        for i in members:
            group_of[i] = f"g{gid:06d}"

    with (WORK_DIR / "duplicates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["drop_path", "drop_pixels", "kept_path", "kept_pixels", "label_match"])
        for i, keeper in sorted(drop):
            writer.writerow([
                rows[i]["path"],
                f'{rows[i]["width"]}x{rows[i]["height"]}',
                rows[keeper]["path"],
                f'{rows[keeper]["width"]}x{rows[keeper]["height"]}',
                rows[i]["label"] == rows[keeper]["label"],
            ])

    with (WORK_DIR / "groups.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "group_id", "dropped"])
        for i, row in enumerate(rows):
            writer.writerow([row["path"], group_of[i], i in dropped])

    multi = sum(1 for m in group_clusters.values() if len(m) > 1)
    cross = sum(1 for i, k in drop if rows[i]["label"] != rows[k]["label"])
    print(f"\n  {len(drop)} duplicates (distance <= {DUP_DISTANCE})")
    print(f"  {multi} multi-shot object groups (distance <= {GROUP_DISTANCE})")
    if cross:
        print(f"  !! {cross} duplicate pairs sit in DIFFERENT class folders -- "
              f"these are labelling conflicts, review duplicates.csv before applying")
    print(f"\nReports written to {WORK_DIR}")

    if args.apply:
        QUARANTINE_DIR.mkdir(exist_ok=True)
        for i, _ in sorted(drop):
            src = SOURCE_DIR / rows[i]["path"]
            dst = QUARANTINE_DIR / rows[i]["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.move(str(src), str(dst))
        print(f"Moved {len(drop)} duplicates to {QUARANTINE_DIR} (not deleted).")
    else:
        print("Nothing moved. Re-run with --apply once the report looks right.")


# ---------------------------------------------------------
# Stage 3: build
# ---------------------------------------------------------

def build_one(job):
    src_rel, dst_rel = job
    src = SOURCE_DIR / src_rel
    dst = BUILD_DIR / dst_rel
    if dst.exists():
        return dst_rel, None  # resume: already built

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open_image(src) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            if max(img.size) > MAX_EDGE:
                img.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
            # No exif= argument: orientation is in the pixels, GPS is gone.
            img.save(dst, "JPEG", quality=JPEG_QUALITY, optimize=True)
            return dst_rel, img.size
    except Exception as exc:
        print(f"  ! failed {src.name}: {exc}", file=sys.stderr)
        return dst_rel, None


def stage_build(args):
    rows = list(csv.DictReader((WORK_DIR / "scan.csv").open(encoding="utf-8")))
    groups = {
        r["path"]: r["group_id"]
        for r in csv.DictReader((WORK_DIR / "groups.csv").open(encoding="utf-8"))
    }
    dropped = {
        r["path"]
        for r in csv.DictReader((WORK_DIR / "groups.csv").open(encoding="utf-8"))
        if r["dropped"] == "True"
    }

    # A manual review in review_duplicates.py overrides the automatic verdict.
    # Clusters you have not reviewed yet keep their automatic one, so a partial
    # review is safe.
    decisions_path = WORK_DIR / "duplicate_decisions.csv"
    if decisions_path.exists():
        overrides = 0
        for r in csv.DictReader(decisions_path.open(encoding="utf-8")):
            was_dropped = r["path"] in dropped
            if r["decision"] == "keep" and was_dropped:
                dropped.discard(r["path"])
                overrides += 1
            elif r["decision"] == "drop" and not was_dropped:
                dropped.add(r["path"])
                overrides += 1
        print(f"Applied {decisions_path.name}: {overrides} manual override(s) "
              f"of the automatic dedup verdict")

    survivors = [r for r in rows if r["path"] not in dropped]
    BUILD_DIR.mkdir(exist_ok=True)

    jobs = []
    for r in survivors:
        src_rel = r["path"]
        # Normalise .jpeg/.png/.heic -> .jpg, keep the existing UUIDv6 stem.
        dst_rel = str(Path(src_rel).with_suffix(".jpg")).replace("\\", "/")
        jobs.append((src_rel, dst_rel))

    print(f"Building {len(jobs)} images at {MAX_EDGE}px / q{JPEG_QUALITY}...")
    sizes = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, (dst_rel, size) in enumerate(pool.map(build_one, jobs, chunksize=16), start=1):
            if size:
                sizes[dst_rel] = size
            if n % 500 == 0:
                print(f"  {n}/{len(jobs)}")

    with (BUILD_DIR / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_name", "label", "material", "subtype", "group_id",
            "captured", "source_sha256", "source_width", "source_height",
        ])
        for (src_rel, dst_rel), r in zip(jobs, survivors):
            if not (BUILD_DIR / dst_rel).exists():
                continue
            label = r["label"]
            material, _, subtype = label.partition("-")
            writer.writerow([
                dst_rel, label, material, subtype,
                groups.get(src_rel, ""), r["captured"], r["sha256"],
                r["width"], r["height"],
            ])

    for extra in ("README.md", ".gitattributes"):
        source = SOURCE_DIR / extra
        if source.exists():
            shutil.copy2(source, BUILD_DIR / extra)

    built = sum(1 for _, d in jobs if (BUILD_DIR / d).exists())
    total_gb = sum(
        (BUILD_DIR / d).stat().st_size for _, d in jobs if (BUILD_DIR / d).exists()
    ) / 1024**3
    print(f"\nBuilt {built} images ({total_gb:.2f} GB) -> {BUILD_DIR}")


# ---------------------------------------------------------
# Stage 4: upload
# ---------------------------------------------------------

def stage_upload(args):
    from huggingface_hub import HfApi

    if not (BUILD_DIR / "metadata.csv").exists():
        sys.exit("No _build/metadata.csv -- run `build` first.")

    api = HfApi()
    print(f"Uploading {BUILD_DIR} to {REPO_ID}...")
    api.upload_folder(
        folder_path=str(BUILD_DIR),
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        delete_patterns="*",
        commit_message=f"Deduplicate, downscale to {MAX_EDGE}px, add metadata.csv",
    )
    print("Upload complete.")

    if args.squash:
        print("Squashing history (quota reflects this within 36h)...")
        api.super_squash_history(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=f"Dataset rebuild: {MAX_EDGE}px, deduplicated",
        )
        print("Squashed. Old full-resolution blobs will be reclaimed.")
    else:
        print("History not squashed. Re-run with --squash once you have verified "
              "the upload AND backed up the full-resolution originals.")


def stage_squash(args):
    """Reclaim the old full-resolution blobs, once the new upload is verified.

    upload_folder(delete_patterns="*") already replaced the repo contents; the
    previous files survive only in git history. This collapses that history to
    a single commit so they are dereferenced. The quota reflects it within 36h.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(REPO_ID, repo_type=REPO_TYPE, files_metadata=False)
    files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    images = [f for f in files if f.lower().endswith(".jpg")]

    print(f"{REPO_ID} currently holds {len(images)} .jpg files "
          f"({len(files)} total), head = {info.sha[:8]}")
    if not args.yes:
        print("\nThis is IRREVERSIBLE: commit history and all previous file "
              "versions are destroyed.")
        print("Check the repo looks right on the Hub first, then re-run with --yes")
        return

    api.super_squash_history(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message=f"Dataset rebuild: {MAX_EDGE}px, deduplicated, subtyped",
    )
    print("Squashed. Storage is reclaimed within 36 hours.")


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workers", type=int, default=8)

    parser = argparse.ArgumentParser(description=__doc__, parents=[common])
    sub = parser.add_subparsers(dest="stage", required=True)

    ingest = sub.add_parser("ingest", parents=[common])
    ingest.add_argument("source", help="folder of new photos to bring in")
    ingest.add_argument("--label", required=True,
                        help="destination class folder, e.g. Plastic-Bottle")
    ingest.add_argument("--move", action="store_true",
                        help="move instead of copy (default copies, leaving the source intact)")
    ingest.add_argument("--create", action="store_true",
                        help="allow creating a class folder that does not exist yet")
    ingest.set_defaults(func=stage_ingest)

    sub.add_parser("scan", parents=[common]).set_defaults(func=stage_scan)

    sub.add_parser("resync", parents=[common]).set_defaults(func=stage_resync)

    dedup = sub.add_parser("dedup", parents=[common])
    dedup.add_argument("--apply", action="store_true",
                       help="move duplicates to _duplicates/ instead of only reporting")
    dedup.set_defaults(func=stage_dedup)

    sub.add_parser("build", parents=[common]).set_defaults(func=stage_build)

    squash = sub.add_parser("squash", parents=[common])
    squash.add_argument("--yes", action="store_true",
                        help="actually do it; without this the command only reports")
    squash.set_defaults(func=stage_squash)

    upload = sub.add_parser("upload", parents=[common])
    upload.add_argument("--squash", action="store_true",
                        help="squash git history to reclaim storage (irreversible)")
    upload.set_defaults(func=stage_upload)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
