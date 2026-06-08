"""
Step 1-5 of the pipeline: image pre-processing for document detection.

Classical CV concepts demonstrated here:
  - Grayscale conversion (drop colour, work on intensity)
  - Gaussian / bilateral smoothing (noise model, scale-space basics)
  - Canny edge detection (gradient-based edges: Sobel + NMS + hysteresis)
  - Morphological closing (binary morphology, structuring elements)

Every function takes an image and returns an image (no side effects), so each
intermediate result can be visualised and dropped into the report.
"""

import cv2
import numpy as np


# Target height we resize to before detection. Smaller = faster and less
# sensitive to fine texture/noise; we keep the scale factor to map corners
# back to the full-resolution image later.
PROCESS_HEIGHT = 800


def resize_keep_aspect(image, target_height=PROCESS_HEIGHT):
    """Resize so height == target_height, keeping aspect ratio.

    Returns (resized_image, scale) where scale maps a coordinate in the
    resized image back to the original: original = resized * scale.
    """
    h, w = image.shape[:2]
    if h <= target_height:
        return image.copy(), 1.0
    scale = h / float(target_height)
    new_w = int(round(w / scale))
    resized = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def to_gray(image):
    """BGR -> single-channel grayscale (cv2.cvtColor)."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray, use_bilateral=True):
    """Smooth noise before edge detection.

    bilateralFilter preserves edges while removing noise (good for receipts on
    textured backgrounds); GaussianBlur is the classic, faster alternative.
    """
    if use_bilateral:
        return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def enhance_contrast(gray):
    """CLAHE: local (adaptive) histogram equalisation.

    Helps when the receipt barely separates from a low-contrast background
    (e.g. white paper on a pale/tan surface).
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def auto_canny(gray, sigma=0.33):
    """Canny with thresholds derived automatically from the image median.

    Avoids hand-tuning hi/lo thresholds per image, which is brittle across
    different lighting conditions.
    """
    v = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)


def close_edges(edges, kernel_size=5, iterations=2):
    """Morphological closing (dilate then erode) to join broken edge segments.

    Canny often leaves gaps along a receipt border; closing bridges them so the
    contour can be found as a single closed loop.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed


def segment_foreground(gray, close_kernel=25):
    """Segment the bright paper (receipt) from a contrasting background.

    Edge detection alone struggles on receipts: the dense interior text breaks
    into many blobs while the low-contrast outer border may not close. A receipt
    is, however, the largest *bright* region in the frame, so a global Otsu
    threshold + a large morphological close (to fill the text holes) collapses
    the whole receipt into one solid blob whose outer contour is the document
    boundary.

    Returns a filled binary mask (uint8, 0/255).
    """
    # Otsu picks the bright/dark split automatically (bimodal paper-vs-bg).
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Close text gaps so the receipt becomes a single solid region.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (close_kernel, close_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Drop small specks in the background.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Fill any remaining holes inside the largest blob so approxPolyDP sees a
    # clean filled polygon.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)
        mask = filled
    return mask


def preprocess_for_detection(image, use_bilateral=True, use_clahe=True):
    """Run the full pre-processing chain and return every intermediate.

    Returns a dict so the caller (pipeline / Streamlit / notebook) can show
    each stage. Keys: gray, denoised, contrast, edges, closed, mask.

    `closed` (Canny + morphological close) is kept for the report to illustrate
    gradient-based edge detection; `mask` (Otsu segmentation) is what detection
    actually uses, as it is far more robust for text-dense receipts.
    """
    gray = to_gray(image)
    denoised = denoise(gray, use_bilateral=use_bilateral)
    contrast = enhance_contrast(denoised) if use_clahe else denoised
    edges = auto_canny(contrast)
    closed = close_edges(edges)
    mask = segment_foreground(denoised)
    return {
        "gray": gray,
        "denoised": denoised,
        "contrast": contrast,
        "edges": edges,
        "closed": closed,
        "mask": mask,
    }
