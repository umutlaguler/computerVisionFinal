"""
CLI entry point for the receipt scanner + OCR pipeline.

Usage:
    python main.py data/raw/images/0.jpg
    python main.py data/raw/images/0.jpg --outdir data/results --lang eng

For every input it writes, into <outdir>/<stem>/:
    01_gray.png 02_denoised.png 03_contrast.png 04_edges.png 05_closed.png
    06_detected.png 07_warped.png 08_flattened.png 09_binary.png  text.txt

so each classical-CV stage can be dropped straight into the report/slides.
"""

import argparse
import os

import cv2

from src import pipeline


STAGE_ORDER = [
    ("gray", "01_gray"),
    ("denoised", "02_denoised"),
    ("contrast", "03_contrast"),
    ("edges", "04_edges"),
    ("closed", "05_closed"),
    ("mask", "05b_mask"),
    ("detected", "06_detected"),
    ("warped", "07_warped"),
    ("flattened", "08_flattened"),
    ("binary", "09_binary"),
]


def process_one(path, outdir, lang):
    image = cv2.imread(path)
    if image is None:
        print(f"  [skip] could not read {path}")
        return

    result = pipeline.scan(image, lang=lang)

    stem = os.path.splitext(os.path.basename(path))[0]
    dest = os.path.join(outdir, stem)
    os.makedirs(dest, exist_ok=True)

    for key, name in STAGE_ORDER:
        img = result.stages.get(key)
        if img is not None:
            cv2.imwrite(os.path.join(dest, name + ".png"), img)

    with open(os.path.join(dest, "text.txt"), "w", encoding="utf-8") as f:
        f.write(result.text)

    n_chars = len(result.text.strip())
    print(f"  {stem}: detection={result.detection_method:8s} "
          f"chars={n_chars:5d}  -> {dest}")


def main():
    ap = argparse.ArgumentParser(description="Receipt scanner + OCR (classical CV)")
    ap.add_argument("inputs", nargs="+", help="image file(s)")
    ap.add_argument("--outdir", default="data/results")
    ap.add_argument("--lang", default="eng")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Processing {len(args.inputs)} image(s) -> {args.outdir}")
    for path in args.inputs:
        process_one(path, args.outdir, args.lang)


if __name__ == "__main__":
    main()
