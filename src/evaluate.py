"""Held-out evaluation of the conv + softmax terrain classifier.

Key idea: split by IMAGE, not by pixel. Neighbouring pixels in one photo are
almost identical, so a pixel-level split would leak test information into
training and give falsely high accuracy.

Usage (from project root):
    python src/evaluate.py                 # seed 0, 3 test images
    python src/evaluate.py --seed 1        # a different random split
"""
import argparse
import os
import random
import time

import matplotlib
matplotlib.use("Agg")  # draw straight to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np

import image_processing as ip
import softmax_reg

STEMS = ["left0586", "left25343", "left22331", "left25057", "left24875", "left23945",
         "left23761", "left23075", "left22741", "left22088", "left24496"]
# Class colours (RGB), index = class id
MU = np.array([[0, 0, 0], [237, 27, 36], [35, 177, 77],
               [163, 73, 163], [253, 240, 11], [63, 71, 204]])
MU8 = MU.astype(np.uint8)
CLASS_NAMES = ["sky", "traversable grass", "putting green",
               "untraversable", "pavement", "golfer"]
TRAVERSABLE = [1, 4]          # the paper only values these two classes
K = len(CLASS_NAMES)
W, H, R = 200, 112, 17        # image width/height after resizing, conv window size


def load_image(stem):
    """Return features X (pixels x 867), labels y (pixels,), h, w, trimmed photo."""
    photo = ip.resize_img(f"./Data/{stem}.jpg", W, H)
    feats = ip.ConvoluteImage(photo, R)            # (h, w, R*R*3), edges trimmed
    h, w = feats.shape[:2]
    X = feats.reshape(h * w, -1)
    mask = ip.trim_edges(ip.resize_img(f"./Data/{stem}_sup.jpg", W, H), R)
    # Snap each mask pixel to the nearest class colour -> integer class id
    y = ip.ConvertWithKeyFlat(mask.reshape(h * w, 3), MU)
    return X, y, h, w, ip.trim_edges(photo, R)


def predict(mods, X, decision):
    """decision='original': the Stanford code's behaviour (forward_softmax as written).
    decision='argmax': plain one-vs-rest, pick the class with the highest probability."""
    probs = np.zeros((X.shape[0], len(mods)))
    for i, m in enumerate(mods):
        if m is not None:
            probs[:, i] = m.predict(X)
    if decision == "original":
        probs = softmax_reg.forward_softmax(probs)
    return np.argmax(probs, axis=1)


def confusion(y_true, y_pred):
    """cm[i, j] = number of pixels whose true class is i and predicted class is j."""
    return np.bincount(y_true * K + y_pred, minlength=K * K).reshape(K, K)


def summarise(title, cm):
    total = cm.sum()
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp          # predicted as this class, but wrong
    fn = cm.sum(axis=1) - tp          # really this class, but missed
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (tp + fp + fn)
        recall = tp / cm.sum(axis=1)

    # Binary view: traversable (red or yellow) vs everything else
    T = TRAVERSABLE
    btp = cm[np.ix_(T, T)].sum()
    bfn = cm[T, :].sum() - btp
    bfp = cm[:, T].sum() - btp
    btn = total - btp - bfn - bfp

    lines = [f"--- {title} ---",
             f"pixels: {total}",
             f"pixel accuracy (6 classes): {tp.sum() / total:.3f}",
             f"mean IoU over classes present: {np.nanmean(iou):.3f}"]
    for c in range(K):
        lines.append(f"  {CLASS_NAMES[c]:18s} IoU={iou[c]:.3f}  recall={recall[c]:.3f}  true pixels={cm[c].sum()}")
    lines += [f"traversable-vs-rest accuracy: {(btp + btn) / total:.3f}",
              f"traversable IoU: {btp / (btp + bfp + bfn):.3f}  "
              f"precision: {btp / (btp + bfp):.3f}  recall: {btp / (btp + bfn):.3f}"]
    return lines, iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_test", type=int, default=3)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    test_stems = rng.sample(STEMS, args.n_test)
    train_stems = [s for s in STEMS if s not in test_stems]
    out = [f"seed={args.seed}",
           f"train images ({len(train_stems)}): {train_stems}",
           f"TEST images  ({len(test_stems)}): {test_stems}"]

    print("Loading images...", flush=True)
    train = [load_image(s) for s in train_stems]
    test = [load_image(s) for s in test_stems]
    X_train = np.vstack([t[0] for t in train])
    y_train = np.concatenate([t[1] for t in train])
    X_test = np.vstack([t[0] for t in test])
    y_test = np.concatenate([t[1] for t in test])
    del train  # free memory; we only need test images for the figure

    t0 = time.time()
    mods = softmax_reg.train_softmax(X_train, y_train)
    out.append(f"training time: {time.time() - t0:.0f}s")

    # Baseline: predict the most common training class everywhere
    majority = np.bincount(y_train, minlength=K).argmax()
    out.append(f"majority-class baseline accuracy on test: {(y_test == majority).mean():.3f} "
               f"(always predicting '{CLASS_NAMES[majority]}')")

    preds = {}
    for decision in ("original", "argmax"):
        preds[decision] = predict(mods, X_test, decision)
        cm = confusion(y_test, preds[decision])
        lines, _ = summarise(f"TEST (unseen images), decision={decision}", cm)
        out += [""] + lines
        # Row-normalised confusion matrix: row = true class, % predicted as each column
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.nan_to_num(100 * cm / cm.sum(axis=1, keepdims=True)).round().astype(int)
        out.append("confusion matrix (% of each TRUE class; columns = predicted, order: "
                   + ", ".join(CLASS_NAMES) + ")")
        out += ["  " + " ".join(f"{v:4d}" for v in row) for row in pct]

    # Same model on its own training images: shows how much is memorisation
    # (original decision rule only)
    y_pred_train = predict(mods, X_train, "original")
    lines, _ = summarise("TRAIN (seen images), decision=original", confusion(y_train, y_pred_train))
    out += [""] + lines

    os.makedirs("Output", exist_ok=True)
    text = "\n".join(out)
    print("\n" + text)
    with open(f"Output/metrics_seed{args.seed}.txt", "w") as f:
        f.write(text)

    # Figure: photo | ground truth | prediction (original) | prediction (argmax)
    fig, axes = plt.subplots(len(test), 4, figsize=(12, 2.4 * len(test)), squeeze=False)
    titles = ["photo", "ground truth", "predicted (original)", "predicted (argmax)"]
    off = 0
    for r, (_, y, h, w, photo) in enumerate(test):
        n = h * w
        panels = [photo, MU8[y].reshape(h, w, 3),
                  MU8[preds["original"][off:off + n]].reshape(h, w, 3),
                  MU8[preds["argmax"][off:off + n]].reshape(h, w, 3)]
        off += n
        for c, img in enumerate(panels):
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=9)
    plt.tight_layout()
    plt.savefig(f"Output/eval_seed{args.seed}.png", dpi=150)
    print(f"\nSaved Output/metrics_seed{args.seed}.txt and Output/eval_seed{args.seed}.png")


if __name__ == "__main__":
    main()