"""
Build a single labelled montage of all pipeline stages for one image, ready to
drop into the report / slides.

Usage:
    python make_figure.py data/raw/images/0.jpg
    python make_figure.py data/raw/images/5.jpg --out figure_5.png
"""

import argparse
import os

import cv2
import numpy as np

from src import pipeline


PANELS = [
    ("original_proxy", "1. Input"),
    ("gray", "2. Grayscale"),
    ("edges", "3. Canny edges"),
    ("mask", "4. Otsu segmentation"),
    ("detected", "5. 4-corner detection"),
    ("warped", "6. Perspective warp"),
    ("binary", "7. Adaptive threshold"),
]

PANEL_W, PANEL_H = 360, 480
COLS = 4
LABEL_H = 30


def to_bgr(img):
    if img is None:
        return np.full((PANEL_H, PANEL_W, 3), 40, np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def fit_panel(img):
    """Letterbox an image into a fixed PANEL_W x PANEL_H tile."""
    img = to_bgr(img)
    h, w = img.shape[:2]
    scale = min(PANEL_W / w, PANEL_H / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh))
    tile = np.full((PANEL_H, PANEL_W, 3), 255, np.uint8)
    y0, x0 = (PANEL_H - nh) // 2, (PANEL_W - nw) // 2
    tile[y0:y0 + nh, x0:x0 + nw] = resized
    return tile


def labelled(tile, text):
    strip = np.full((LABEL_H, PANEL_W, 3), 30, np.uint8)
    cv2.putText(strip, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([strip, tile])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        raise SystemExit(f"could not read {args.input}")
    res = pipeline.scan(image, lang="eng")
    res.stages["original_proxy"] = res.original

    tiles = [labelled(fit_panel(res.stages.get(key)), title)
             for key, title in PANELS]

    rows = []
    for i in range(0, len(tiles), COLS):
        row = tiles[i:i + COLS]
        while len(row) < COLS:
            row.append(labelled(fit_panel(None), ""))
        rows.append(np.hstack(row))
    montage = np.vstack(rows)

    out = args.out or f"figure_{os.path.splitext(os.path.basename(args.input))[0]}.png"
    out = os.path.join("data/results", out)
    cv2.imwrite(out, montage)
    print(f"detection={res.detection_method}  ->  {out}  ({montage.shape[1]}x{montage.shape[0]})")


if __name__ == "__main__":
    main()
