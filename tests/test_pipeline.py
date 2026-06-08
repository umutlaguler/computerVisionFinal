"""
Minimal smoke tests for the receipt pipeline.

Run directly (no pytest required):
    python tests/test_pipeline.py
or with pytest if installed:
    pytest tests/
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import pipeline, transform, annotations
import numpy as np


SAMPLE = "data/raw/images/0.jpg"


def test_order_corners():
    """Corners given out of order must come back as TL, TR, BR, BL."""
    pts = np.array([[100, 100], [10, 10], [10, 100], [100, 10]], dtype="float32")
    o = transform.order_corners(pts)
    assert tuple(o[0]) == (10, 10)     # top-left
    assert tuple(o[1]) == (100, 10)    # top-right
    assert tuple(o[2]) == (100, 100)   # bottom-right
    assert tuple(o[3]) == (10, 100)    # bottom-left


def test_pipeline_runs_and_detects():
    image = cv2.imread(SAMPLE)
    assert image is not None, f"missing sample {SAMPLE}"
    res = pipeline.scan(image, lang="eng")
    assert res.quad is not None, "no document detected"
    assert res.detection_method in ("quad", "minarea")
    assert res.warped is not None and res.warped.size > 0
    assert res.binary is not None
    assert len(res.text.strip()) > 20, "OCR produced too little text"


def test_annotations_parse():
    anns = annotations.load_annotations("data/raw/annotations.xml")
    assert len(anns) == 20
    a0 = anns["0"]
    assert a0.receipt_polygon is not None
    assert a0.ground_truth_text().strip(), "no ground-truth text"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print(f"\n{'OK' if not failures else 'FAILURES: ' + str(failures)}")
    sys.exit(1 if failures else 0)
