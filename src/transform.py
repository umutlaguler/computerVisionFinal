"""
Step 8-9 of the pipeline: order the corners and warp to a top-down view.

Classical CV concept demonstrated here:
  - Projective geometry / homography. A planar document seen by a camera from
    an angle is related to its fronto-parallel ("scanned") view by a 3x3
    homography with 8 degrees of freedom, recoverable from 4 point matches.
    cv2.getPerspectiveTransform solves it; cv2.warpPerspective applies it.
"""

import cv2
import numpy as np


def order_corners(pts):
    """Order 4 points consistently as top-left, top-right, bottom-right,
    bottom-left.

    Uses the classic sum/diff trick:
      - top-left has the smallest (x+y), bottom-right the largest.
      - top-right has the smallest (x-y)... we use (y-x): top-right smallest,
        bottom-left largest.
    """
    pts = np.asarray(pts, dtype="float32").reshape(4, 2)
    ordered = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()  # y - x
    ordered[0] = pts[np.argmin(s)]       # top-left
    ordered[2] = pts[np.argmax(s)]       # bottom-right
    ordered[1] = pts[np.argmin(diff)]    # top-right
    ordered[3] = pts[np.argmax(diff)]    # bottom-left
    return ordered


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def warp_to_top_down(image, quad):
    """Warp the quadrilateral region of `image` to a rectangular top-down view.

    The output size is derived from the quad's own side lengths so the aspect
    ratio of the (tall, narrow) receipt is preserved.
    Returns (warped_bgr, homography_matrix).
    """
    rect = order_corners(quad)
    (tl, tr, br, bl) = rect

    width = int(round(max(_dist(br, bl), _dist(tr, tl))))
    height = int(round(max(_dist(tr, br), _dist(tl, bl))))
    width = max(width, 1)
    height = max(height, 1)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped, M
