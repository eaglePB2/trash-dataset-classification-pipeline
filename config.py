"""Shared configuration, loaded from a .env file.

Every tunable the pipeline has lives here, so nothing has to be edited inside
the scripts. Values are resolved in this order, first match wins:

    1. a real environment variable        (export REPO_ID=... / $env:REPO_ID)
    2. .env in the dataset root           (where you run the commands)
    3. .env in tools/                     (next to this file)
    4. the default below

Run it directly to see what is actually in effect and where each value came
from:

    python tools/config.py
"""

import os
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent

_ENV_FILES = [Path(".").resolve() / ".env", TOOLS_DIR / ".env"]
_loaded = {}
_origin = {}


def _parse(path: Path):
    """Minimal .env reader -- no dependency on python-dotenv."""
    values = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.split(" #")[0].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


for _path in _ENV_FILES:
    if _path.exists():
        for _k, _v in _parse(_path).items():
            if _k not in _loaded:            # earlier file wins
                _loaded[_k] = _v
                _origin[_k] = str(_path)


def _raw(name, default):
    """Real environment variables outrank .env, which outranks the default."""
    if name in os.environ:
        _origin[name] = "environment"
        return os.environ[name]
    if name in _loaded:
        return _loaded[name]
    _origin[name] = "default"
    return default


def get_str(name, default):
    return str(_raw(name, default))


def get_int(name, default):
    try:
        return int(str(_raw(name, default)).strip())
    except ValueError:
        raise SystemExit(f"{name} in .env must be a whole number, "
                         f"got {_raw(name, default)!r}")


def get_float(name, default):
    try:
        return float(str(_raw(name, default)).strip())
    except ValueError:
        raise SystemExit(f"{name} in .env must be a number, "
                         f"got {_raw(name, default)!r}")


# --- Where the dataset lives ----------------------------------------------
# Defaults to the current directory, so running the commands from inside the
# dataset folder just works. Set it if you keep the code somewhere else --
# then the tools can be cloned anywhere and pointed at the data.
DATASET_ROOT = Path(get_str("DATASET_ROOT", ".")).expanduser().resolve()

# --- Hub -------------------------------------------------------------------
REPO_ID = get_str("REPO_ID", "eaglepb2/Trash_Classification")
REPO_TYPE = get_str("REPO_TYPE", "dataset")

# --- Published image format ------------------------------------------------
MAX_EDGE = get_int("MAX_EDGE", 1024)
JPEG_QUALITY = get_int("JPEG_QUALITY", 88)

# --- Deduplication ---------------------------------------------------------
DUP_DISTANCE = get_int("DUP_DISTANCE", 5)
GROUP_DISTANCE = get_int("GROUP_DISTANCE", 12)

# --- CLIP pre-sort (shared by presort.py and contact_sheet.py, so the sheets
#     always grade against the same thresholds the sort used) ---------------
CLIP_MODEL = get_str("CLIP_MODEL", "openai/clip-vit-large-patch14")
CLIP_BATCH_SIZE = get_int("CLIP_BATCH_SIZE", 32)
MIN_PROB = get_float("MIN_PROB", 0.50)
MIN_MARGIN = get_float("MIN_MARGIN", 0.10)

# --- Working directories ---------------------------------------------------
WORK_DIR = DATASET_ROOT / get_str("WORK_DIR", "_work")
BUILD_DIR = DATASET_ROOT / get_str("BUILD_DIR", "_build")
QUARANTINE_DIR = DATASET_ROOT / get_str("QUARANTINE_DIR", "_duplicates")
BROKEN_DIR = DATASET_ROOT / get_str("BROKEN_DIR", "_broken")

# --- Shared file/folder rules ----------------------------------------------
# Defined once so every tool agrees on what counts as an image and which
# directories are scaffolding rather than class labels.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".dng"}
RAW_EXTENSIONS = {".dng"}
RESERVED_DIRS = {"tools", ".git", ".claude", "converted_jpgs"}


def class_dirs():
    """Top-level directories that represent class labels.

    Anything starting with "_" or "." is scaffolding, so new working
    directories never need adding to RESERVED_DIRS.
    """
    return sorted(
        p for p in DATASET_ROOT.iterdir()
        if p.is_dir() and p.name not in RESERVED_DIRS
        and not p.name.startswith((".", "_"))
    )


def source_images():
    """Every image inside a class folder, in a stable order."""
    for class_dir in class_dirs():
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path

SETTINGS = [
    ("DATASET_ROOT", DATASET_ROOT),
    ("REPO_ID", REPO_ID), ("REPO_TYPE", REPO_TYPE),
    ("MAX_EDGE", MAX_EDGE), ("JPEG_QUALITY", JPEG_QUALITY),
    ("DUP_DISTANCE", DUP_DISTANCE), ("GROUP_DISTANCE", GROUP_DISTANCE),
    ("CLIP_MODEL", CLIP_MODEL), ("CLIP_BATCH_SIZE", CLIP_BATCH_SIZE),
    ("MIN_PROB", MIN_PROB), ("MIN_MARGIN", MIN_MARGIN),
    ("WORK_DIR", WORK_DIR), ("BUILD_DIR", BUILD_DIR),
    ("QUARANTINE_DIR", QUARANTINE_DIR), ("BROKEN_DIR", BROKEN_DIR),
]


def describe():
    found = [str(p) for p in _ENV_FILES if p.exists()]
    print(f"dataset root : {DATASET_ROOT}")
    print(f".env files   : {', '.join(found) if found else 'none found (using defaults)'}")
    print()
    print(f"  {'setting':<16} {'value':<42} source")
    for name, value in SETTINGS:
        src = _origin.get(name, _origin.get(name, ".env"))
        if name in _loaded and name not in os.environ:
            src = _origin.get(name, ".env")
        print(f"  {name:<16} {str(value):<42} {src}")
    token = "set" if os.environ.get("HF_TOKEN") else "not set (use `hf auth login`)"
    print(f"\n  HF_TOKEN         {token}")


if __name__ == "__main__":
    describe()
