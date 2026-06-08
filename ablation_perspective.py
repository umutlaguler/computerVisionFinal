"""
Controlled ablation: does the perspective correction actually help OCR?

The dataset receipts are mostly photographed flat, so on them the pipeline only
*matches* raw OCR. To isolate and prove the value of the homography step, we run
a controlled experiment:

    1. Take each (flat) receipt and apply a KNOWN synthetic perspective warp,
       simulating a photo taken from an angle.
    2. RAW   : OCR the distorted image directly.
    3. SCAN  : run the full pipeline (which detects the receipt and undoes the
               perspective) on the distorted image, then OCR.

If the classical-CV perspective correction works, SCAN should clearly beat RAW
on the distorted inputs. This is the honest demonstration of the homography's
contribution that the already-flat real photos can't show.

Usage:
    python ablation_perspective.py
    python ablation_perspective.py --tilt 0.30
"""

import argparse
import os
import re
from difflib import SequenceMatcher

import cv2
import numpy as np

from src import pipeline, ocr
from src.annotations import load_annotations


IMAGES_DIR = "data/raw/images"
ANN_PATH = "data/raw/annotations.xml"


def normalize(s):
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def field_similarity(gt_fields, ocr_text):
    lines = [normalize(l) for l in ocr_text.splitlines() if normalize(l)]
    if not lines:
        return 0.0
    sims = []
    for _, raw in gt_fields:
        target = normalize(raw)
        if target:
            sims.append(max(SequenceMatcher(None, target, l).ratio()
                            for l in lines))
    return sum(sims) / len(sims) if sims else 0.0


def synthetic_tilt(image, tilt=0.25):
    """Apply a known perspective distortion to simulate an angled photo.

    The top edge is pushed inward by `tilt` (fraction of width) and the image is
    shrunk onto a larger black canvas so the whole tilted receipt stays visible
    (and easy to re-detect against the black background).
    """
    h, w = image.shape[:2]
    pad = int(0.15 * max(h, w))
    canvas_w, canvas_h = w + 2 * pad, h + 2 * pad

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dx = tilt * w
    dst = np.float32([
        [pad + dx, pad],            # top-left  pushed right
        [pad + w - dx, pad],        # top-right pushed left  (top edge narrower)
        [pad + w, pad + h],         # bottom-right
        [pad, pad + h],             # bottom-left
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (canvas_w, canvas_h),
                               borderValue=(0, 0, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tilt", type=float, default=0.25,
                    help="perspective strength (fraction of width)")
    args = ap.parse_args()

    anns = load_annotations(ANN_PATH)
    stems = sorted(anns.keys(), key=lambda s: int(s) if s.isdigit() else s)

    print(f"Synthetic perspective ablation (tilt={args.tilt})")
    print(f"{'img':>4} | {'sim_raw':>8} {'sim_scan':>9}  {'delta':>7}")
    print("-" * 40)

    raws, scans = [], []
    for stem in stems:
        for ext in (".jpg", ".JPG", ".png"):
            path = os.path.join(IMAGES_DIR, stem + ext)
            if os.path.exists(path):
                break
        else:
            continue
        image = cv2.imread(path)
        distorted = synthetic_tilt(image, tilt=args.tilt)

        raw_sim = field_similarity(anns[stem].texts, ocr.image_to_text(distorted))
        res = pipeline.scan(distorted, lang="eng")
        scan_sim = field_similarity(anns[stem].texts, res.text)

        raws.append(raw_sim)
        scans.append(scan_sim)
        print(f"{stem:>4} | {raw_sim:8.3f} {scan_sim:9.3f}  "
              f"{scan_sim - raw_sim:+7.3f}")

    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print("-" * 40)
    print(f"MEAN | {mean(raws):8.3f} {mean(scans):9.3f}  "
          f"{mean(scans) - mean(raws):+7.3f}")
    print("\nOn angled inputs the perspective-correcting pipeline should beat "
          "raw OCR\nby a clear margin -> this is the value of the homography "
          "step.")


if __name__ == "__main__":
    main()
