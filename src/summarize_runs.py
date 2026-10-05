"""Average the Output/metrics_seed*.txt files written by evaluate.py."""
import glob
import re
import numpy as np

KEYS = {
    "pixel accuracy":        r"pixel accuracy \(6 classes\): ([\d.]+)",
    "mean IoU":              r"mean IoU over classes present: ([\d.]+)",
    "traversable IoU":       r"traversable IoU: ([\d.]+)",
    "traversable precision": r"precision: ([\d.]+)",
    "traversable recall":    r"recall: ([\d.]+)",   # colon only matches the traversable line
}

files = sorted(glob.glob("Output/metrics_seed*.txt"))
res, base = {}, []
for f in files:
    text = open(f).read()
    base.append(float(re.search(r"baseline accuracy on test: ([\d.]+)", text).group(1)))
    parts = re.split(r"^--- (.+?) ---$", text, flags=re.M)   # [pre, title, body, title, body...]
    for title, body in zip(parts[1::2], parts[2::2]):
        for key, pat in KEYS.items():
            res.setdefault(title, {}).setdefault(key, []).append(float(re.search(pat, body).group(1)))

print(f"{len(files)} runs: {files}")
print(f"majority baseline (test): {np.mean(base):.3f} +/- {np.std(base, ddof=1):.3f}")
for title, d in res.items():
    print(f"\n{title}")
    for key, v in d.items():
        print(f"  {key:22s} {np.mean(v):.3f} +/- {np.std(v, ddof=1):.3f}   per-run: {v}")