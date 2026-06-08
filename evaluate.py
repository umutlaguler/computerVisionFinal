"""
Quantitative evaluation of the pipeline against the CVAT ground truth.

Two metrics, plus an ablation:

  1. Document detection quality  (geometry):
       IoU between the detected quad and the ground-truth `receipt` polygon.
       A detection counts as "success" if IoU >= --iou-thresh (default 0.80).

  2. OCR accuracy  (text), measured as FIELD RECALL:
       The dataset only annotates selected fields (shop / items / date_time /
       total), not the whole receipt, while OCR returns *all* text. A global
       CER/WER would therefore be dominated by "insertions" of real-but-
       unannotated lines. Instead, for each ground-truth field we find its best
       matching line in the OCR output and score the similarity in [0, 1]
       (1 = perfect). This directly answers "did we recover the key receipt
       content?". We report it for:
         - RAW   : OCR run directly on the original photo (no pipeline)
         - SCAN  : OCR run on our warped + binarised output (the pipeline)
       The RAW-vs-SCAN gap is the ablation that shows the value of the
       classical-CV pre-processing.

Usage:
    python evaluate.py
    python evaluate.py --iou-thresh 0.7 --limit 5
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

# A field counts as "recovered" when its best line similarity is at least this.
FIELD_HIT_THRESH = 0.6


def normalize(s):
    """Lowercase, strip punctuation, collapse whitespace - compare content."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def field_scores(gt_fields, ocr_text):
    """For each GT field, best similarity (0..1) to any line in the OCR text.

    Returns (mean_similarity, recall) where recall is the fraction of fields
    whose best similarity >= FIELD_HIT_THRESH.
    """
    lines = [normalize(l) for l in ocr_text.splitlines() if normalize(l)]
    if not lines:
        return 0.0, 0.0
    sims = []
    for _, raw in gt_fields:
        target = normalize(raw)
        if not target:
            continue
        best = max(SequenceMatcher(None, target, line).ratio() for line in lines)
        sims.append(best)
    if not sims:
        return None, None
    mean_sim = sum(sims) / len(sims)
    recall = sum(1 for s in sims if s >= FIELD_HIT_THRESH) / len(sims)
    return mean_sim, recall


def polygon_mask(points, shape):
    """Rasterise a polygon (N,2) into a binary mask of `shape` (h, w)."""
    mask = np.zeros(shape, dtype="uint8")
    pts = np.round(points).astype("int32").reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def iou(poly_a, poly_b, shape):
    a = polygon_mask(poly_a, shape) > 0
    b = polygon_mask(poly_b, shape) > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def find_image_path(stem):
    for ext in (".jpg", ".JPG", ".png", ".jpeg"):
        p = os.path.join(IMAGES_DIR, stem + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description="Evaluate the receipt pipeline")
    ap.add_argument("--iou-thresh", type=float, default=0.70)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N images (debug)")
    args = ap.parse_args()

    anns = load_annotations(ANN_PATH)
    stems = sorted(anns.keys(), key=lambda s: int(s) if s.isdigit() else s)
    if args.limit:
        stems = stems[:args.limit]

    print(f"{'img':>4} {'detect':>8} {'IoU':>6} | "
          f"{'sim_raw':>7} {'sim_scan':>8} | {'rec_raw':>7} {'rec_scan':>8}")
    print("-" * 70)

    ious, sim_raw, sim_scan, rec_raw, rec_scan = [], [], [], [], []
    n_success = 0

    for stem in stems:
        ann = anns[stem]
        path = find_image_path(stem)
        if path is None:
            continue
        image = cv2.imread(path)

        res = pipeline.scan(image, lang="eng")

        # --- detection IoU ---
        iou_val = float("nan")
        if res.quad is not None and ann.receipt_polygon is not None:
            shape = (image.shape[0], image.shape[1])
            iou_val = iou(res.quad, ann.receipt_polygon, shape)
            ious.append(iou_val)
            if iou_val >= args.iou_thresh:
                n_success += 1

        # --- OCR field recall: raw vs pipeline ---
        raw_text = ocr.image_to_text(image, lang="eng")
        sr, rr = field_scores(ann.texts, raw_text)
        ss, rs = field_scores(ann.texts, res.text)
        for lst, val in ((sim_raw, sr), (sim_scan, ss),
                         (rec_raw, rr), (rec_scan, rs)):
            if val is not None:
                lst.append(val)

        def fmt(x):
            return f"{x:6.3f}" if isinstance(x, float) else f"{'-':>6}"

        print(f"{stem:>4} {res.detection_method:>8} {iou_val:6.3f} | "
              f"{fmt(sr):>7} {fmt(ss):>8} | {fmt(rr):>7} {fmt(rs):>8}")

    print("-" * 70)
    n_det = len(ious)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    print(f"Detection success (IoU >= {args.iou_thresh:.2f}): "
          f"{n_success}/{len(stems)}  ({100*n_success/max(len(stems),1):.0f}%)")
    print(f"Mean IoU (detected):          {mean(ious):.3f}  (n={n_det})")
    print(f"Mean field similarity raw -> scan:  "
          f"{mean(sim_raw):.3f} -> {mean(sim_scan):.3f}")
    print(f"Mean field recall     raw -> scan:  "
          f"{mean(rec_raw):.3f} -> {mean(rec_scan):.3f}")
    print("\n(Higher is better; scan > raw = classical-CV pre-processing benefit.)")


if __name__ == "__main__":
    main()
