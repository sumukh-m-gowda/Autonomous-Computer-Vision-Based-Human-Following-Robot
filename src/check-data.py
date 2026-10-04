"""Find the files we need inside the downloaded folder and report what's missing.

Run it from the terminal with the folder path, or with no argument and it will ask.
"""
import sys
from pathlib import Path
from PIL import Image

TRAIN = ["left0586", "left25343", "left22331", "left25057", "left24875", "left23945",
         "left23761", "left23075", "left22741", "left22088", "left24496"]
TEST = ["left23590", "left22156"]          # photos only, no masks exist for these
MASK_SUFFIXES = ["_sup", "_supervised"]    # both names appear in the original code

# Use the command-line argument if given, otherwise ask for the path.
if len(sys.argv) > 1:
    raw = sys.argv[1]
else:
    raw = input("Paste the path of the downloaded folder: ")
root = Path(raw.strip().strip('"').strip("'"))

if not root.exists():
    sys.exit(f"Folder not found: {root}")

# Index every jpg/png once: lowercase filename -> full path
files = {p.name.lower(): p for p in root.rglob("*")
         if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
print(f"Found {len(files)} image files under {root}\n")


def find(stem, suffix=""):
    for ext in (".jpg", ".jpeg", ".png"):
        hit = files.get((stem + suffix + ext).lower())
        if hit:
            return hit
    return None


missing = 0
for stem in TRAIN + TEST:
    photo = find(stem)
    mask = None
    if stem in TRAIN:
        for s in MASK_SUFFIXES:
            mask = find(stem, s)
            if mask:
                break
    size = Image.open(photo).size if photo else None
    msize = Image.open(mask).size if mask else None
    need_mask = stem in TRAIN
    ok = bool(photo) and (bool(mask) or not need_mask)
    missing += 0 if ok else 1
    print(f"{'OK     ' if ok else 'MISSING'} {stem:11s} photo={size}  "
          f"mask={msize if need_mask else 'n/a (test)'}")

print(f"\n{missing} problem(s).")