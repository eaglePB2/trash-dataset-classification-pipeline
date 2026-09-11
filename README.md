# Trash Classification Dataset pipeline

Tooling that maintains the `EaglePB2/Trash_Classification` dataset: it turns a
folder of raw phone photos into a deduplicated, labelled, downscaled dataset and
publishes it to the Hugging Face Hub.

Requires **Python 3.9+** (tested on 3.14). Works on Windows, macOS and Linux.

## Install

Clone into your dataset folder as `tools/`, so commands read naturally:

```bash
git clone https://github.com/eaglePB2/trash-dataset-classification-pipeline tools
```

```bash
pip install -r tools/requirements.txt
```

**Run every command from the dataset root, not from inside `tools/`** — the
scripts treat the current directory as the dataset root:

```bash
python tools/rebuild.py scan
```

If you would rather keep the code somewhere separate from the images, clone it
anywhere and set `DATASET_ROOT` in `.env` (or per-command):

```bash
DATASET_ROOT=/data/trash python /code/trash-pipeline/rebuild.py scan
```

---

## Configuration

Everything tunable lives in a `.env` file in the dataset root. Nothing needs to
be edited inside the scripts.

```bash
cp tools/.env.example .env
```

**`REPO_ID` is the one you must change** if you are reusing this pipeline for
your own dataset. The rest have working defaults.

```bash
python tools/config.py
```

Prints every setting actually in effect and where each came from — useful when
a value isn't what you expected.

Resolution order, first match wins:

1. a real environment variable — `REPO_ID=me/mine python tools/rebuild.py upload`
2. `.env` in the dataset root
3. `.env` in `tools/`
4. the built-in default

| Setting | Default | Notes |
|---|---|---|
| `DATASET_ROOT` | current directory | Where the images live |
| `REPO_ID` | `eaglepb2/Trash_Classification` | Target repository |
| `REPO_TYPE` | `dataset` | `dataset` or `model` |
| `MAX_EDGE` | `1024` | Longest side of published images |
| `JPEG_QUALITY` | `88` | Visually lossless at training resolution |
| `DUP_DISTANCE` | `5` | phash distance treated as the same photo |
| `GROUP_DISTANCE` | `12` | phash distance treated as the same object |
| `CLIP_MODEL` | `openai/clip-vit-large-patch14` | Any CLIP checkpoint on the Hub |
| `CLIP_BATCH_SIZE` | `32` | Lower if you run out of VRAM |
| `MIN_PROB` / `MIN_MARGIN` | `0.50` / `0.10` | Pre-sort acceptance thresholds |
| `WORK_DIR` etc. | `_work`, `_build`, `_duplicates`, `_broken` | Keep the leading underscore |

`MIN_PROB` and `MIN_MARGIN` are read by both `presort.py` and
`contact_sheet.py`, so the QA sheets always grade against the thresholds the
sort actually used — they cannot drift apart.

Bad values fail immediately with a clear message rather than silently falling
back, and `.env` is never copied into `_build/`, so it cannot be published.

**Authentication is deliberately not in `.env`.** Run `hf auth login` once; the
token is stored in your Hugging Face cache instead of a plain-text file that is
easy to leak into a commit or an upload.

---

## How the dataset is laid out

| Path | What it is |
|---|---|
| `Clothes/`, `Plastic-Bottle/`, … | **Class folders.** The folder name *is* the label. Holds full-resolution originals — this is the archival master. |
| `_work/` | Intermediate state: `scan.csv`, `duplicates.csv`, decisions, reports. Safe to inspect, never uploaded. |
| `_build/` | What actually gets published: 1024px JPEGs plus `metadata.csv` and the dataset card. |
| `_duplicates/`, `_broken/` | Quarantine. Files are **moved** here, never deleted. |
| `tools/` | This directory. |
| `README.md` (root) | The **dataset card**, published to the Hub. Not this file. |

Any folder starting with `_` or `.`, plus `tools/`, is ignored when discovering
class labels. A new top-level folder becomes a new class automatically.

---

## Workflow A — adding new photos

The common case. Roughly:

```bash
python tools/rebuild.py ingest ~/Pictures/phone_dump --label Plastic-Bottle
```

Copies the photos in under UUIDv6 names, skipping anything already in the
dataset by content hash. Originals keep their format and resolution — RAW stays
RAW. Add `--move` to move rather than copy, `--create` to allow a new class.

```bash
python tools/rebuild.py resync
```

Folds the new files into `scan.csv` — hashing only the new ones rather than
re-reading all 57 GB. Then continue at **dedup** in Workflow B.

---

## Workflow B — a full rebuild

### 1. Get the data

```bash
hf auth login
```

```bash
hf download eaglepb2/Trash_Classification --repo-type dataset --local-dir .
```

### 2. Scan

```bash
python tools/rebuild.py scan
```

Reads every image once: SHA-256, perceptual hash, dimensions, capture time →
`_work/scan.csv`. Unreadable files go to `_work/failed.csv` rather than being
silently dropped. This is the slow step (~20–30 min for 23k images).

If anything failed:

```bash
python tools/diagnose.py
```

Identifies files by **magic bytes**, not extension — Git LFS pointers, truncated
JPEGs, HEIC misnamed `.jpg`, zero-byte files, saved HTML error pages.

```bash
python tools/verify.py
```

Compares failures against the Hub's recorded LFS hashes to settle the only
question that matters: **`REDOWNLOAD`** (your copy is incomplete — re-fetch) or
**`FAITHFUL`** (your copy is correct, so the file is broken in the dataset
itself — drop it). `--all` checks every file.

### 3. Deduplicate

```bash
python tools/rebuild.py dedup
```

Clusters by perceptual hash. Distance ≤5 is "the same photo" (drop all but the
largest); ≤12 is "the same object" (keep, but tag with `group_id`). **Writes a
report and moves nothing.**

```bash
python tools/review_duplicates.py
```

A tkinter GUI showing each cluster side by side. `Enter` accepts, `K` keeps all,
number keys toggle, `Q` saves and quits. Resumable — decisions are saved after
every cluster. Label-conflict clusters, where the same photo sits under two
different classes, are sorted to the front; those are labelling errors, not
duplicate decisions.

```bash
python tools/review_duplicates.py --apply
```

Moves the files you marked drop into `_duplicates/`. Use `--reconcile` if the
filesystem and your decisions ever disagree — it forces them to match in both
directions.

> Don't run `rebuild.py dedup --apply` if you've used the reviewer. It quarantines
> by the *automatic* verdict and will overwrite your choices.

### 4. Sub-classify (optional)

```bash
python tools/presort.py
```

Zero-shot CLIP, sorting `Plastic` and `Glass` into subtypes. Material comes from
the existing folder, so the model only chooses among subtypes of that material —
a far easier problem than open-set classification. Report-only by default.

```bash
python tools/contact_sheet.py
```

Renders `_work/sheets/*.jpg` — for each proposed folder, the **lowest-confidence
accepted** predictions (the decision boundary, where errors concentrate) above a
random sample. Check these before applying; if the boundary looks right,
everything above it is fine.

```bash
python tools/presort.py --apply
```

Moves confident predictions into `<Material>-<Subtype>/` and the rest into
`<Material>-Unsorted/`. **Then review every folder by hand** and empty the
`-Unsorted` ones — folder names become labels, so `Plastic-Unsorted` would ship
as a class. Afterwards run `rebuild.py resync` to re-point `scan.csv`.

Tuning: `--min-prob` / `--min-margin` change the thresholds. With `--from-csv`
they re-apply to cached scores instantly, with no GPU pass — but that does *not*
pick up prompt edits, which need a full re-run.

### 5. Build

```bash
python tools/rebuild.py build
```

Writes `_build/`: 1024px long edge, JPEG q88, plus `metadata.csv`. Resumable —
re-running skips files already built. Roughly 3.2 GB from 23k images.

### 6. Publish

```bash
python tools/rebuild.py upload
```

One commit that replaces the repo contents entirely (`delete_patterns="*"`).
**Verify on the Hub before the next step** — file count, that `metadata.csv`
parses, that the card renders.

```bash
python tools/rebuild.py squash --yes
```

Collapses history so the old files stop counting against your quota. Storage
frees within 36 hours. Irreversible; without `--yes` it only reports.

---

## What `build` guarantees about output

- **Always a real JPEG**, whatever came in. JPEG, MPO (multi-frame phone
  captures — primary frame kept), HEIF and DNG all converge on single-frame
  `.jpg`. Extensions stop lying.
- **Orientation baked into the pixels**, orientation tag removed. The original
  pipeline transposed the pixels and then wrote the *old* EXIF back, so every
  phone photo was rotated twice; writing no EXIF makes that impossible.
- **No EXIF at all** — which also strips the GPS coordinates present in about a
  third of the sources. Capture time survives in `metadata.csv` and in the UUIDv6
  filename.
- **Filenames are stable.** A file keeps its UUID for life. Renaming would create
  new LFS objects on every upload and quietly multiply your storage.

---

## Things worth knowing before you change something

**Why UUIDv6.** The filename encodes capture time, so it sorts chronologically and
`resync` can recover a file's identity after it moves between class folders —
which is what makes re-labelling cheap instead of a 30-minute re-scan.

**Use `group_id` when splitting.** 1,283 groups hold more than one photo of the
same physical object. A random train/test split leaks them across both sides and
inflates accuracy.

**Prompt wording in `presort.py` is load-bearing.** CLIP has no sense of scale, so
a prompt mentioning "bottle cap" matches whole bottles that *have* a cap. A prompt
saying "white foam cup" matches every cup. If a class starts absorbing things it
shouldn't, look at the prompts before the thresholds.

**Frequent items with no class get absorbed by the nearest one.** That's what a
`-Misc` bucket is really for. If a class fills with confident nonsense, the usual
cause is a missing class, not a bad threshold.

**Nothing is deleted.** Every destructive-looking operation moves files to
`_duplicates/` or `_broken/`. The exceptions are `upload` and `squash`, which do
change the Hub — `squash` irreversibly.

---

## Extending the taxonomy

Numeric and Hub settings are in `.env` (see **Configuration** above). The one
thing still defined in code is the subtype vocabulary: `PROMPTS` at the top of
`tools/presort.py`. Adding a class means adding an entry there and re-running
the pre-sort — prompt changes are not picked up by `--from-csv`.

Shared file rules — `IMAGE_EXTENSIONS`, `RESERVED_DIRS`, and the `class_dirs()`
/ `source_images()` helpers — live in `tools/config.py` and are imported by every
script, so a new source format or working directory only has to be handled once.

---

## License

MIT — see [LICENSE](LICENSE). The dataset itself is published separately on the
Hugging Face Hub under its own license.
