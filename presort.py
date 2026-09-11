"""Zero-shot CLIP pre-sort: propose subtype folders for Plastic and Glass.

This does NOT produce final labels. It gets each image into a probable subtype
folder so that reviewing is a correction pass rather than a sorting pass.
Anything the model is not confident about goes to <Material>-Unsorted for you
to place by hand.

    python tools/presort.py --limit 200          # trial run, report only, no moves
    python tools/presort.py                      # full run, report only
    python tools/presort.py --apply              # move files into subtype folders

Material is taken from the existing folder, so the model only ever chooses
among subtypes of that material -- a much easier problem than open-set
classification, and considerably more accurate.

Run this AFTER `tools/rebuild.py dedup --apply` (no point labelling duplicates) and
BEFORE `tools/rebuild.py build` (folder names become the labels).
"""

import argparse
import csv
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

import config

SOURCE_DIR = config.DATASET_ROOT
WORK_DIR = config.WORK_DIR

MODEL_ID = config.CLIP_MODEL
BATCH_SIZE = config.CLIP_BATCH_SIZE

# Accept the model's proposal only when it is both confident and decisive.
MIN_PROB = config.MIN_PROB      # top-1 softmax probability
MIN_MARGIN = config.MIN_MARGIN  # top-1 minus top-2

IMAGE_EXTENSIONS = config.IMAGE_EXTENSIONS

# Deliberately no "Misc" prompt: a generic catch-all absorbs probability mass
# from every real class and wrecks the others. Low-confidence images land in
# <Material>-Unsorted instead, and you decide what is genuinely Misc.
PROMPTS = {
    "Plastic": {
        # Bottle vs Container is decided by NECK GEOMETRY, not contents:
        # narrow neck = Bottle, wide mouth = Container. Without that rule the
        # two prompts overlap (is a pump-top shampoo bottle a bottle or a
        # container?) and probability splits, so neither wins by a margin.
        "Bottle": [
            "a plastic bottle with a narrow neck and a screw cap",
            "a tall plastic drink bottle with a narrow opening",
            "a plastic shampoo or detergent bottle with a narrow neck",
            "a complete plastic bottle with its cap still on",
            "an empty PET water bottle",
        ],
        "Container": [
            "a wide-mouthed plastic tub with no neck",
            "a plastic food tray or clamshell punnet",
            "an empty plastic yogurt or ice cream tub",
            "a shallow plastic takeaway container",
        ],
        # Film vs Laminate is a real disposal-stream boundary: mono-material
        # film is often collected, multi-layer laminate is not. The cue is
        # transparency versus a metallic sheen, so the prompts lean on that.
        "Film": [
            "a photo of a crumpled transparent plastic bag",
            "a piece of clear plastic cling film",
            "a translucent plastic shopping bag",
            "a clear plastic bread bag",
        ],
        "Laminate": [
            "a photo of a snack packet with a shiny silver metallic inner lining",
            "an empty metallised crisp packet",
            "a foil-lined biscuit wrapper",
            "a flexible coffee pouch with a metallic interior",
        ],
        # Caps and cups are frequent enough that, without their own classes,
        # CLIP absorbed them into whatever was nearest (it made Foam a magnet
        # for every small white object). Foam was dropped: once caps and cups
        # are pulled out, genuine polystyrene is too rare here to train on, so
        # it belongs in Misc.
        # Every prompt naming "bottle" made CLIP match whole bottles that have
        # a cap on them -- it has no sense of scale. Describe the isolated disc
        # instead, and never say "bottle" here.
        "Cap": [
            "a close-up of a single small plastic screw cap",
            "one small round plastic lid lying alone on a plain surface",
            "a tiny circular plastic closure disc",
            "a single detached plastic cap photographed by itself",
        ],
        "Cup": [
            "a disposable plastic drinking cup",
            "a clear plastic cup with a domed lid and a straw",
            "a plastic bubble tea cup",
            "an empty disposable cold drink cup",
        ],
    },
    "Glass": {
        "Bottle": [
            "a photo of a glass bottle",
            "an empty glass beer bottle",
            "a glass wine bottle",
            "a tall glass drink bottle",
        ],
        "Jar": [
            "a photo of a glass jar",
            "an empty glass food jar with a lid",
            "a mason jar",
            "a short wide glass jar",
        ],
    },
}


def load_image(path: Path):
    try:
        with Image.open(path) as img:
            img.draft("RGB", (336, 336))  # fast reduced-scale JPEG decode
            return path, img.convert("RGB")
    except Exception as exc:
        print(f"  ! failed {path.name}: {exc}", file=sys.stderr)
        return path, None


def features(out):
    """transformers >= 5 returns an output object here; older versions a tensor.

    Verified equivalent: pooler_output is the *projected* embedding, and
    reproduces model(**inputs).logits_per_image exactly.
    """
    return out if torch.is_tensor(out) else out.pooler_output


@torch.no_grad()
def text_embeddings(model, processor, subtypes, device, dtype):
    """One ensembled, normalised embedding per subtype."""
    vectors = []
    for prompts in subtypes.values():
        batch = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        feats = features(model.get_text_features(**batch)).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        mean = feats.mean(dim=0)
        vectors.append(mean / mean.norm())
    return torch.stack(vectors).to(dtype)


@torch.no_grad()
def classify_material(material, model, processor, device, dtype, limit):
    folder = SOURCE_DIR / material
    if not folder.is_dir():
        print(f"  {material}/ not found, skipping")
        return []

    paths = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit:
        paths = paths[:limit]
    if not paths:
        print(f"  {material}/ is empty, skipping")
        return []

    subtypes = PROMPTS[material]
    names = list(subtypes.keys())
    text_vecs = text_embeddings(model, processor, subtypes, device, dtype)
    scale = model.logit_scale.exp().to(dtype)

    print(f"  {material}: {len(paths)} images over {names}")
    rows = []
    with ThreadPoolExecutor(max_workers=8) as io_pool:
        for start in range(0, len(paths), BATCH_SIZE):
            chunk = paths[start:start + BATCH_SIZE]
            loaded = [(p, im) for p, im in io_pool.map(load_image, chunk) if im is not None]
            if not loaded:
                continue

            batch = processor(
                images=[im for _, im in loaded], return_tensors="pt"
            ).to(device, dtype)
            feats = features(model.get_image_features(**batch)).to(dtype)
            feats = feats / feats.norm(dim=-1, keepdim=True)

            probs = (scale * feats @ text_vecs.T).float().softmax(dim=-1).cpu()
            ordered = probs.argsort(dim=-1, descending=True)

            for (path, _), prob, order in zip(loaded, probs, ordered):
                top, second = int(order[0]), int(order[1]) if len(names) > 1 else int(order[0])
                rows.append({
                    "path": str(path.relative_to(SOURCE_DIR)).replace("\\", "/"),
                    "material": material,
                    "subtype": names[top],
                    "prob": round(float(prob[top]), 4),
                    "margin": round(float(prob[top] - prob[second]), 4),
                    **{f"p_{n}": round(float(prob[i]), 4) for i, n in enumerate(names)},
                })

            done = min(start + BATCH_SIZE, len(paths))
            if done % (BATCH_SIZE * 10) == 0 or done == len(paths):
                print(f"    {done}/{len(paths)}")

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", nargs="+", default=list(PROMPTS),
                        help="which material folders to sort")
    parser.add_argument("--limit", type=int, default=0,
                        help="only process the first N images per material (trial run)")
    parser.add_argument("--min-prob", type=float, default=MIN_PROB)
    parser.add_argument("--min-margin", type=float, default=MIN_MARGIN)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--apply", action="store_true",
                        help="move files into <Material>-<Subtype>/ folders")
    parser.add_argument("--from-csv", action="store_true",
                        help="re-apply thresholds to the existing _work/presort.csv "
                             "instead of re-running the model (instant; use when "
                             "tuning --min-prob/--min-margin, not after prompt edits)")
    args = parser.parse_args()

    if args.from_csv:
        existing = WORK_DIR / "presort.csv"
        if not existing.exists():
            sys.exit(f"{existing} not found -- run a full pass first.")
        all_rows = [
            {**r, "prob": float(r["prob"]), "margin": float(r["margin"])}
            for r in csv.DictReader(existing.open(encoding="utf-8"))
        ]
        print(f"Re-thresholding {len(all_rows)} cached scores "
              f"(no model run; prompt edits are NOT reflected)")
        report_and_apply(all_rows, args, rewrite_csv=False)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {args.model} on {device} ({dtype})...")
    model = CLIPModel.from_pretrained(args.model, dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    WORK_DIR.mkdir(exist_ok=True)
    all_rows = []
    for material in args.materials:
        if material not in PROMPTS:
            print(f"  no prompts defined for {material}, skipping")
            continue
        all_rows.extend(
            classify_material(material, model, processor, device, dtype, args.limit)
        )

    if not all_rows:
        sys.exit("Nothing classified.")

    report_and_apply(all_rows, args, rewrite_csv=True)


def report_and_apply(all_rows, args, rewrite_csv=True):
    WORK_DIR.mkdir(exist_ok=True)
    out = WORK_DIR / "presort.csv"
    if rewrite_csv:
        fields = sorted({k for r in all_rows for k in r},
                        key=lambda k: (k.startswith("p_"), k))
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(all_rows)

    confident = [
        r for r in all_rows
        if r["prob"] >= args.min_prob and r["margin"] >= args.min_margin
    ]
    print(f"\n{len(confident)}/{len(all_rows)} confident "
          f"(prob >= {args.min_prob}, margin >= {args.min_margin})")

    counts = {}
    for r in all_rows:
        key = f'{r["material"]}-{r["subtype"]}'
        accepted = r["prob"] >= args.min_prob and r["margin"] >= args.min_margin
        counts.setdefault(key, [0, 0])[0 if accepted else 1] += 1
    print(f"\n  {'proposed folder':<22} {'confident':>9} {'unsorted':>9}")
    for key in sorted(counts):
        ok, no = counts[key]
        print(f"  {key:<22} {ok:>9} {no:>9}")
    print(f"\nFull scores -> {out}")

    if not args.apply:
        print("\nNothing moved. Review presort.csv (sort by prob ascending to see "
              "the model's weakest calls), then re-run with --apply.")
        return

    moved = unsorted_count = 0
    for r in all_rows:
        src = SOURCE_DIR / r["path"]
        if not src.exists():
            continue
        accepted = r["prob"] >= args.min_prob and r["margin"] >= args.min_margin
        if accepted:
            dest_dir = SOURCE_DIR / f'{r["material"]}-{r["subtype"]}'
            moved += 1
        else:
            dest_dir = SOURCE_DIR / f'{r["material"]}-Unsorted'
            unsorted_count += 1
        dest_dir.mkdir(exist_ok=True)
        shutil.move(str(src), str(dest_dir / src.name))

    print(f"\nMoved {moved} into subtype folders, "
          f"{unsorted_count} into <Material>-Unsorted.")
    print("Review every folder by eye, then resolve the -Unsorted folders "
          "(rename to -Misc or redistribute) before running `rebuild.py scan`.")


if __name__ == "__main__":
    main()
