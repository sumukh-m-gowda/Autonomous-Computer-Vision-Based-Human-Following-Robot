# Autonomous Computer-Vision-Based Human-Following Robot

**UE24CS352A Machine Learning mini-project, Team 26 (Problem 19)**
Sushanth Bhat (PES2UG24CS544), Sumukh M Gowda (PES2UG24CS537)

A software implementation of the Stanford CS229 (2018) project _"Golf Bag Carrier Robot Computer Vision"_ by Akshay Gupta and Nicholas Gloria. A robot with one camera must (1) classify terrain so it stays on traversable ground, and (2) detect and follow a person.

## Credit and what we added

The terrain-classification code (`image_processing.py`, `softmax_reg.py`, `linear_model.py`) and the dataset come from the original authors' project files, which the report links. The originals are kept unmodified in `legacy/`.

**Our changes and additions:**

- Fixed the original code so it runs on current libraries (`PIL.Image.ANTIALIAS` was removed from Pillow, so it now uses `LANCZOS`).
- Added an iteration cap and a pseudo-inverse fallback to the Newton solver, and vectorised `predict`.
- Removed broken or unused functions and added progress printing.
- **`evaluate.py`:** held-out evaluation by image (pixel accuracy, per-class IoU, confusion matrix, traversable-terrain metrics, majority-class baseline, and a comparison of two decision rules). The original report gives no quantitative results.
- **`summarize_runs.py`:** averages the metrics over several splits.
- **`track_demo.py`:** live webcam person detection with a follow command (steering and speed) [STATUS: confirm after testing].

## Method (short)

**Terrain classification.** Images are resized to 200x112. Each pixel is described by the RGB values of its 17x17 neighbourhood (867 features), so the classifier sees local colour gradients. One logistic regression per class is trained with Newton's method (one-vs-rest), and a softmax-style rule picks one class per pixel. Six classes: sky, traversable grass, putting green, untraversable, pavement, golfer.

**Person following.** A pretrained SSDLite-MobileNetV3 detector (COCO "person" class, via torchvision) finds the target. The horizontal offset of the box sets the steering command, and the box height serves as a rough distance proxy.

## Repository layout

```
src/
  image_processing.py   conv features, training and prediction pipeline (original code, fixed)
  softmax_reg.py        one-vs-rest logistic regression + softmax (original code, fixed)
  linear_model.py       base class (original)
  check-data.py         verifies the dataset files are in place
  evaluate.py           held-out evaluation by image
  summarize_runs.py     averages metrics over splits
  track_demo.py         webcam person-following demo
legacy/                 original files as received, for reference only (not run)
Data/                   images and masks (NOT in git, see below)
Output/                 generated predictions and metrics (NOT in git)
```

## Setup (Windows, Python 3.10+; tested on 3.13)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If activation is blocked, run `Set-ExecutionPolicy -Scope Process Bypass` first. The webcam demo also needs PyTorch (`pip install torch torchvision`), which is included in `requirements.txt` if it was installed when that file was generated; otherwise install it separately.

## Data

Place the files in a folder called `Data/` in the project root. We use 11 photos with hand-painted masks (`leftXXXXX.jpg` and `leftXXXXX_sup.jpg`) and 2 unlabelled test photos. Check with:

```powershell
python src/check-data.py "path\to\Data"
```

All 13 lines should say `OK`.

## How to run (always from the project root, with `.venv` active)

| Task                                         | Command                           | Output                                              |
| -------------------------------------------- | --------------------------------- | --------------------------------------------------- |
| Baseline: train and predict on all 13 images | `python src/image_processing.py`  | `Output/*.jpg`                                      |
| Held-out evaluation (seed = split)           | `python src/evaluate.py --seed 0` | `Output/metrics_seed0.txt`, `Output/eval_seed0.png` |
| Summarise several splits                     | `python src/summarize_runs.py`    | printed table                                       |
| Live demo (press `q` to quit)                | `python src/track_demo.py`        | webcam window                                       |

Each evaluation run takes about 2 minutes on a laptop CPU and a few GB of RAM. Running `evaluate.py` and `image_processing.py` also writes resized `*_red.jpg` copies next to the originals in `Data/`.

## Results

Evaluation uses 3 random splits, each training on 8 labelled images and testing on 3 unseen ones (mean ± std). Splits are by image, never by pixel, to avoid leakage between neighbouring pixels.

| Metric (unseen images)     | Original decision rule | Plain argmax  |
| -------------------------- | ---------------------- | ------------- |
| Pixel accuracy (6 classes) | 0.610 ± 0.033          | 0.644 ± 0.055 |
| Mean IoU                   | 0.370 ± 0.046          | 0.354 ± 0.056 |
| Traversable IoU            | 0.281 ± 0.092          | 0.283 ± 0.087 |
| Traversable precision      | 0.325 ± 0.127          | 0.407 ± 0.164 |
| Traversable recall         | 0.705 ± 0.046          | 0.496 ± 0.101 |

Majority-class baseline accuracy: 0.459 ± 0.048.

**Findings.** The classifier beats the baseline but is only moderately accurate. The main failure is putting green vs grass: in every split, 52% to 89% of true putting-green pixels were labelled traversable grass. Traversable precision is low, so predicted "safe" regions often are not.

## Limitations

- Only 11 labelled images, and the three test sets overlap (6 of 11 images were ever held out), so the standard deviations are rough.
- Masks are JPEG-compressed, which blurs class borders.
- The person detector has no memory between frames, so it may switch between people when several are visible.
- Box height is only a crude distance proxy.
- The torchvision detector is similar to, not identical to, the TensorFlow MobileNet SSD used in the report.

## References

1. Gupta, A. and Gloria, N. _Golf Bag Carrier Robot Computer Vision_, Stanford CS229, 2018. https://cs229.stanford.edu/proj2018/report/12.pdf
2. Felzenszwalb, P. F. and Huttenlocher, D. P. Efficient graph-based image segmentation. _IJCV_ 59, 2004.
3. Howard, A. et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv:1704.04861, 2017.
