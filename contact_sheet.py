"""Contact sheets for judging the CLIP pre-sort at a glance.

    python tools/contact_sheet.py

For every proposed subtype it renders one JPEG into _work/sheets/:

  top rows    the LOWEST-confidence predictions that were still accepted --
              the decision boundary, where mistakes concentrate. If these
              look right, everything above them is fine.
  bottom rows a random sample, for a fair view of typical quality.

Also renders the <Material>-Unsorted bucket, so you can see whether the
threshold is throwing away work you could have kept.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

import config

SOURCE_DIR = config.DATASET_ROOT
WORK_DIR = config.WORK_DIR
SHEET_DIR = WORK_DIR / "sheets"

# Same source as presort.py, so the sheets cannot grade against different
# thresholds than the sort actually used.
MIN_PROB = config.MIN_PROB
MIN_MARGIN = config.MIN_MARGIN

CELL = 220
COLS = 6
BOUNDARY_ROWS = 2   # lowest-confidence accepted
RANDOM_ROWS = 2     # random sample
CAPTION = 22
HEADER = 46
SEED = 0


def font(size):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def thumb(path: Path, box: int):
    try:
        with Image.open(path) as im:
            im.draft("RGB", (box * 2, box * 2))
            im = ImageOps.exif_transpose(im)
            if im.mode != "RGB":
                im = im.convert("RGB")
            im.thumbnail((box, box), Image.LANCZOS)
            return im.copy()
    except Exception:
        return Image.new("RGB", (box, box), (60, 30, 30))


def sheet(title, entries, subtitle):
    rows = max(1, -(-len(entries) // COLS))
    width = COLS * CELL
    height = HEADER + rows * (CELL + CAPTION)
    canvas = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)

    draw.text((10, 8), title, fill=(240, 240, 240), font=font(20))
    draw.text((10, 30), subtitle, fill=(150, 150, 150), font=font(12))

    for i, (rel_path, prob, note) in enumerate(entries):
        col, row = i % COLS, i // COLS
        x = col * CELL
        y = HEADER + row * (CELL + CAPTION)

        image = thumb(SOURCE_DIR / rel_path, CELL - 8)
        canvas.paste(image, (x + (CELL - image.width) // 2,
                             y + (CELL - 8 - image.height) // 2))

        colour = (230, 120, 120) if prob < 0.62 else (150, 200, 150)
        draw.text((x + 6, y + CELL - 6), f"p={prob:.3f} {note}",
                  fill=colour, font=font(12))

    return canvas


def main():
    rows = list(csv.DictReader((WORK_DIR / "presort.csv").open(encoding="utf-8")))
    if not rows:
        raise SystemExit("No _work/presort.csv -- run `python tools/presort.py` first.")

    accepted = defaultdict(list)
    rejected = defaultdict(list)
    for r in rows:
        prob, margin = float(r["prob"]), float(r["margin"])
        key = f'{r["material"]}-{r["subtype"]}'
        if prob >= MIN_PROB and margin >= MIN_MARGIN:
            accepted[key].append((r["path"], prob))
        else:
            rejected[f'{r["material"]}-Unsorted'].append((r["path"], prob, r["subtype"]))

    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    written = []

    for key in sorted(accepted):
        items = sorted(accepted[key], key=lambda t: t[1])
        boundary = items[:COLS * BOUNDARY_ROWS]
        pool = items[COLS * BOUNDARY_ROWS:]
        sample = rng.sample(pool, min(COLS * RANDOM_ROWS, len(pool)))

        entries = ([(p, q, "weakest") for p, q in boundary]
                   + [(p, q, "") for p, q in sorted(sample, key=lambda t: t[1])])
        image = sheet(
            key,
            entries,
            f"{len(items)} accepted  |  top {len(boundary)} = lowest confidence "
            f"(check these), bottom {len(sample)} = random sample",
        )
        out = SHEET_DIR / f"{key}.jpg"
        image.save(out, "JPEG", quality=88)
        written.append(out)

    for key in sorted(rejected):
        items = sorted(rejected[key], key=lambda t: -t[1])[:COLS * 4]
        entries = [(p, q, f"->{s}") for p, q, s in items]
        image = sheet(
            key,
            entries,
            f"{len(rejected[key])} below threshold  |  showing the highest-scoring "
            f"rejects -- if these look obviously right, lower the threshold",
        )
        out = SHEET_DIR / f"{key}.jpg"
        image.save(out, "JPEG", quality=88)
        written.append(out)

    print(f"Wrote {len(written)} sheets to {SHEET_DIR}")
    for p in written:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
