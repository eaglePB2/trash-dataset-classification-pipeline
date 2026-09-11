"""Visual review of duplicate clusters: decide what to keep, one cluster at a time.

    python tools/review_duplicates.py            # open the reviewer
    python tools/review_duplicates.py --apply    # move everything marked drop to _duplicates/

Reads _work/duplicates.csv (from `rebuild.py dedup`) and shows each cluster of
near-identical photos side by side. Click a thumbnail, or press its number, to
toggle keep/drop. Decisions are written to _work/duplicate_decisions.csv after
every cluster, so you can close the window and pick up where you left off.

Clusters where the same photo was filed under two different classes are shown
first -- those are labelling errors, not just duplicates.

Keys:  Enter/Right = accept and advance     Left = go back
       1-9 = toggle that image              K = keep all
       D = keep only the largest            Q = save and quit
"""

import argparse
import csv
import shutil
import sys
import tkinter as tk
from collections import OrderedDict
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox

from PIL import Image, ImageOps, ImageTk

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

import config

SOURCE_DIR = config.DATASET_ROOT
WORK_DIR = config.WORK_DIR
QUARANTINE_DIR = config.QUARANTINE_DIR

DUPLICATES_CSV = WORK_DIR / "duplicates.csv"
DECISIONS_CSV = WORK_DIR / "duplicate_decisions.csv"

THUMB_MAX = 360
BG = "#1e1e1e"
FG = "#e8e8e8"
KEEP_COLOR = "#2ea043"
DROP_COLOR = "#8b2c2c"
WARN_COLOR = "#d29922"


def load_clusters():
    """Group duplicates.csv into clusters, label-conflict ones first."""
    if not DUPLICATES_CSV.exists():
        sys.exit(f"{DUPLICATES_CSV} not found -- run `python tools/rebuild.py dedup` first.")

    rows = list(csv.DictReader(DUPLICATES_CSV.open(encoding="utf-8")))
    grouped = OrderedDict()
    conflict = {}
    for r in rows:
        keeper = r["kept_path"]
        grouped.setdefault(keeper, [])
        grouped[keeper].append(r["drop_path"])
        if r["label_match"] == "False":
            conflict[keeper] = True

    clusters = []
    for keeper, drops in grouped.items():
        clusters.append({
            "id": keeper,
            "members": [keeper] + drops,
            "suggested_keep": keeper,
            "conflict": conflict.get(keeper, False),
        })
    # conflicts first, order otherwise stable
    clusters.sort(key=lambda c: not c["conflict"])
    return clusters


def load_decisions():
    if not DECISIONS_CSV.exists():
        return {}
    return {
        r["path"]: r["decision"]
        for r in csv.DictReader(DECISIONS_CSV.open(encoding="utf-8"))
    }


def save_decisions(decisions):
    WORK_DIR.mkdir(exist_ok=True)
    with DECISIONS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "decision"])
        for path, decision in decisions.items():
            writer.writerow([path, decision])


def file_info(path: Path):
    try:
        size = path.stat().st_size
    except OSError:
        return "missing", 0
    return f"{size / 1024:.0f} KB", size


class Reviewer:
    def __init__(self, root, clusters, decisions):
        self.root = root
        self.clusters = clusters
        self.decisions = decisions
        self.index = self.first_undecided()
        self.thumb_cache = {}
        self.photo_refs = []
        self.panels = []

        root.title("Duplicate review")
        root.configure(bg=BG)
        root.geometry("1500x950")

        self.bold = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self.small = tkfont.Font(family="Segoe UI", size=9)

        self.header = tk.Label(root, bg=BG, fg=FG, font=self.bold, pady=8)
        self.header.pack(fill="x")

        self.warning = tk.Label(root, bg=BG, fg=WARN_COLOR, font=self.bold)
        self.warning.pack(fill="x")

        self.strip = tk.Frame(root, bg=BG)
        self.strip.pack(fill="both", expand=True, padx=12, pady=6)

        controls = tk.Frame(root, bg=BG, pady=10)
        controls.pack(fill="x")
        for text, command in (
            ("← Back", self.back),
            ("Keep only largest  (D)", self.keep_largest),
            ("Keep ALL  (K)", self.keep_all),
            ("Accept →  (Enter)", self.advance),
            ("Save & quit  (Q)", self.quit_save),
        ):
            tk.Button(controls, text=text, command=command,
                      bg="#2d2d2d", fg=FG, activebackground="#3d3d3d",
                      relief="flat", padx=14, pady=6).pack(side="left", padx=6)

        self.status = tk.Label(root, bg=BG, fg="#888", font=self.small, pady=6)
        self.status.pack(fill="x")

        root.bind("<Return>", lambda e: self.advance())
        root.bind("<Right>", lambda e: self.advance())
        root.bind("<Left>", lambda e: self.back())
        root.bind("k", lambda e: self.keep_all())
        root.bind("d", lambda e: self.keep_largest())
        root.bind("q", lambda e: self.quit_save())
        for n in range(1, 10):
            root.bind(str(n), lambda e, n=n: self.toggle(n - 1))

        self.render()

    def first_undecided(self):
        for i, cluster in enumerate(self.clusters):
            if any(m not in self.decisions for m in cluster["members"]):
                return i
        return len(self.clusters) - 1

    def cluster(self):
        return self.clusters[min(self.index, len(self.clusters) - 1)]

    def ensure_defaults(self):
        """Apply the automatic suggestion for any member not yet decided."""
        cluster = self.cluster()
        for member in cluster["members"]:
            if member not in self.decisions:
                self.decisions[member] = (
                    "keep" if member == cluster["suggested_keep"] else "drop"
                )

    def thumb(self, rel_path, box):
        key = (rel_path, box)
        if key in self.thumb_cache:
            return self.thumb_cache[key]
        path = SOURCE_DIR / rel_path
        try:
            with Image.open(path) as im:
                im.draft("RGB", (box * 2, box * 2))
                im = ImageOps.exif_transpose(im)
                if im.mode != "RGB":
                    im = im.convert("RGB")
                im.thumbnail((box, box), Image.LANCZOS)
                result = im.copy()
        except Exception as exc:
            result = Image.new("RGB", (box, box // 2), "#442222")
            print(f"  ! preview failed {rel_path}: {exc}", file=sys.stderr)
        self.thumb_cache[key] = result
        return result

    def render(self):
        self.ensure_defaults()
        cluster = self.cluster()
        members = cluster["members"]

        for widget in self.strip.winfo_children():
            widget.destroy()
        self.photo_refs.clear()
        self.panels.clear()

        decided = sum(
            1 for c in self.clusters
            if all(m in self.decisions for m in c["members"])
        )
        self.header.config(
            text=f"Cluster {self.index + 1} of {len(self.clusters)}    "
                 f"•    {len(members)} near-identical images    "
                 f"•    {decided} clusters decided"
        )
        self.warning.config(
            text="LABEL CONFLICT — the same photo is filed under different "
                 "classes. Fix the label, do not just drop one."
            if cluster["conflict"] else ""
        )

        box = max(180, min(THUMB_MAX, (1500 - 60) // max(1, len(members))))

        for n, rel_path in enumerate(members):
            image = self.thumb(rel_path, box)
            photo = ImageTk.PhotoImage(image)
            self.photo_refs.append(photo)

            panel = tk.Frame(self.strip, bg=BG, highlightthickness=4, bd=0)
            panel.pack(side="left", padx=6, fill="y")

            tk.Label(panel, image=photo, bg=BG).pack()
            size_text, _ = file_info(SOURCE_DIR / rel_path)
            name = Path(rel_path).name
            folder = Path(rel_path).parent.as_posix()

            tk.Label(panel, text=f"[{n + 1}]  {folder}", bg=BG, fg=FG,
                     font=self.bold).pack(pady=(6, 0))
            tk.Label(panel, text=f"{name[:24]}...\n{image.width}x{image.height} "
                                 f"preview • {size_text}",
                     bg=BG, fg="#999", font=self.small, justify="center").pack()

            state = tk.Label(panel, text="", bg=BG, fg=FG, font=self.bold, pady=4)
            state.pack()

            panel.bind("<Button-1>", lambda e, i=n: self.toggle(i))
            for child in panel.winfo_children():
                child.bind("<Button-1>", lambda e, i=n: self.toggle(i))

            self.panels.append((panel, state, rel_path))

        self.paint_states()
        kept = sum(1 for v in self.decisions.values() if v == "keep")
        dropped = sum(1 for v in self.decisions.values() if v == "drop")
        self.status.config(
            text=f"{kept} marked keep • {dropped} marked drop      "
                 f"click an image or press its number to toggle"
        )

    def paint_states(self):
        for panel, state, rel_path in self.panels:
            if self.decisions.get(rel_path) == "keep":
                panel.config(highlightbackground=KEEP_COLOR,
                             highlightcolor=KEEP_COLOR)
                state.config(text="KEEP", fg=KEEP_COLOR)
            else:
                panel.config(highlightbackground=DROP_COLOR,
                             highlightcolor=DROP_COLOR)
                state.config(text="drop", fg="#c97b7b")

    def toggle(self, i):
        if i >= len(self.panels):
            return
        rel_path = self.panels[i][2]
        self.decisions[rel_path] = (
            "drop" if self.decisions.get(rel_path) == "keep" else "keep"
        )
        self.paint_states()

    def keep_all(self):
        for member in self.cluster()["members"]:
            self.decisions[member] = "keep"
        self.paint_states()

    def keep_largest(self):
        cluster = self.cluster()
        for member in cluster["members"]:
            self.decisions[member] = (
                "keep" if member == cluster["suggested_keep"] else "drop"
            )
        self.paint_states()

    def advance(self):
        save_decisions(self.decisions)
        if self.index >= len(self.clusters) - 1:
            messagebox.showinfo(
                "Done",
                f"All {len(self.clusters)} clusters reviewed.\n\n"
                f"Decisions saved to {DECISIONS_CSV.name}.\n\n"
                f"`tools/rebuild.py build` will honour them. To move the dropped "
                f"files out of the class folders, run:\n\n"
                f"    python tools/review_duplicates.py --apply",
            )
            return
        self.index += 1
        self.render()

    def back(self):
        if self.index > 0:
            self.index -= 1
            self.render()

    def quit_save(self):
        save_decisions(self.decisions)
        self.root.destroy()


def stage_apply():
    decisions = load_decisions()
    if not decisions:
        sys.exit(f"No {DECISIONS_CSV} yet -- run the reviewer first.")

    to_drop = [p for p, d in decisions.items() if d == "drop"]
    missing = [p for p in to_drop if not (SOURCE_DIR / p).exists()]
    present = [p for p in to_drop if (SOURCE_DIR / p).exists()]

    print(f"{len(decisions)} decisions: "
          f"{sum(1 for d in decisions.values() if d == 'keep')} keep, "
          f"{len(to_drop)} drop")
    if missing:
        print(f"  ({len(missing)} already moved)")

    for rel_path in present:
        src = SOURCE_DIR / rel_path
        dst = QUARANTINE_DIR / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    print(f"Moved {len(present)} files to {QUARANTINE_DIR} (not deleted).")


def stage_reconcile():
    """Make the filesystem match the decisions exactly, in both directions.

    Needed if the automatic `dedup --apply` also ran: it quarantines by the
    automatic verdict, which overwrites any cluster where you picked a
    different keeper. Safe to re-run.
    """
    decisions = load_decisions()
    if not decisions:
        sys.exit(f"No {DECISIONS_CSV} yet -- nothing to reconcile against.")

    restore, quarantine, missing = [], [], []
    for rel_path, decision in decisions.items():
        in_place = (SOURCE_DIR / rel_path).exists()
        in_quarantine = (QUARANTINE_DIR / rel_path).exists()
        if decision == "keep" and not in_place:
            (restore if in_quarantine else missing).append(rel_path)
        elif decision == "drop" and in_place:
            quarantine.append(rel_path)

    print(f"{len(decisions)} decisions")
    print(f"  restore to class folders : {len(restore)}")
    print(f"  move to _duplicates      : {len(quarantine)}")
    if missing:
        print(f"  !! unaccounted for       : {len(missing)}")
        for p in missing[:10]:
            print(f"       {p}")

    for rel_path in restore:
        dst = SOURCE_DIR / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(QUARANTINE_DIR / rel_path), str(dst))
    for rel_path in quarantine:
        dst = QUARANTINE_DIR / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(SOURCE_DIR / rel_path), str(dst))

    print(f"\nReconciled: {len(restore)} restored, {len(quarantine)} quarantined.")
    if missing:
        sys.exit("Some decided files are in neither place -- investigate before building.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="move files marked drop into _duplicates/")
    parser.add_argument("--reconcile", action="store_true",
                        help="force the filesystem to match decisions, both directions")
    parser.add_argument("--selftest", type=int, default=0,
                        help="build the UI, step through N clusters, exit")
    args = parser.parse_args()

    if args.reconcile:
        stage_reconcile()
        return

    if args.apply:
        stage_apply()
        return

    clusters = load_clusters()
    decisions = load_decisions()
    print(f"{len(clusters)} clusters to review "
          f"({sum(1 for c in clusters if c['conflict'])} label conflicts first)")

    root = tk.Tk()
    app = Reviewer(root, clusters, decisions)

    if args.selftest:
        for _ in range(args.selftest):
            root.update()
            app.advance()
        root.update()
        print(f"selftest ok: rendered {args.selftest} clusters, "
              f"{len(app.decisions)} decisions recorded")
        save_decisions(app.decisions)
        root.destroy()
        return

    root.mainloop()


if __name__ == "__main__":
    main()
