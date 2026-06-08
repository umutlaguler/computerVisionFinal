"""
Step 6-7 of the pipeline: find the receipt and its four corners.

Classical CV concepts demonstrated here:
  - Contour extraction (boundary representation of connected shapes)
  - Polygonal approximation via Douglas-Peucker (cv2.approxPolyDP): test the
    hypothesis "this contour is a 4-sided document".
  - Geometric sanity checks (area ratio, convexity) to reject distractors
    such as table edges.
"""

import cv2
import numpy as np


def find_contours(closed_edges):
    """Return external contours sorted by area, largest first."""
    contours, _ = cv2.findContours(
        closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return sorted(contours, key=cv2.contourArea, reverse=True)


def approx_quad(contour, eps_frac=0.02):
    """Approximate a contour with a polygon; return it if it has 4 corners.

    eps_frac scales the approximation tolerance by the contour perimeter
    (arcLength), so it adapts to contour size. Returns a (4, 2) float array or
    None.
    """
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps_frac * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        return approx.reshape(4, 2).astype("float32")
    return None


def find_document_quad(closed_edges, image_area, min_area_ratio=0.10,
                       eps_candidates=(0.02, 0.04, 0.06, 0.08)):
    """Search the largest contours for a convex 4-corner quadrilateral.

    Strategy:
      1. Take contours largest-first.
      2. Skip contours smaller than `min_area_ratio` of the image (noise).
      3. Try a few approxPolyDP tolerances; accept the first 4-corner convex
         polygon that is large enough.

    Returns (quad, contour) where quad is (4, 2) float32 in the *resized* image
    coordinate space, or (None, largest_contour_or_None) on failure so the
    caller can fall back.
    """
    contours = find_contours(closed_edges)
    if not contours:
        return None, None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_ratio * image_area:
            break  # all remaining are smaller -> stop early
        for eps in eps_candidates:
            quad = approx_quad(cnt, eps)
            if quad is not None and cv2.contourArea(quad.astype("int32")) >= \
                    min_area_ratio * image_area:
                return quad, cnt

    # No clean quad: return the largest contour for a fallback / diagnostics.
    return None, contours[0]


def min_area_quad(contour):
    """Fallback: wrap a contour in its minimum-area rotated rectangle.

    Used when approxPolyDP can't find a clean 4-gon (curved/occluded edges).
    Gives a usable 4 corners at the cost of some accuracy.
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return box.astype("float32")


def scale_quad(quad, scale):
    """Map quad coordinates from resized space back to the original image."""
    return quad * float(scale)


def draw_quad(image, quad, color=(0, 255, 0), thickness=3):
    """Return a copy of image with the quad drawn (for visualisation)."""
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    pts = quad.astype("int32").reshape(-1, 1, 2)
    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)
    for (x, y) in quad.astype("int32"):
        cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 255), -1)
    return vis
