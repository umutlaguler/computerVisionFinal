"""
Step 10 of the pipeline: turn the warped colour crop into a clean, scanned-
looking binary image that OCR reads well.

Classical CV concepts demonstrated here:
  - Adaptive (local) thresholding: under uneven illumination a single global
    threshold fails; a per-region threshold (Gaussian-weighted neighbourhood)
    keeps text crisp. This is local image segmentation.
  - Optional shadow removal via background estimation (large-kernel blur).
  - Light morphological opening to drop salt-and-pepper speckle.
"""

import cv2
import numpy as np


def remove_shadow(gray):
    """Flatten uneven illumination by dividing out an estimated background.

    The background is approximated by a heavily dilated + blurred copy; dividing
    the image by it normalises slow brightness gradients (shadows/vignetting).
    """
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(gray, bg)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype("uint8")


def prepare_for_ocr(warped, upscale=1.5):
    """Build the image we actually feed to Tesseract.

    Empirically, Tesseract reads a *grayscale* crop better than a hard
    adaptive-threshold binary (it runs its own internal Otsu and a harsh binary
    throws away information). A mild upscale nudges small receipt fonts towards
    Tesseract's preferred ~300 DPI. So OCR runs on grayscale-upscaled, while the
    binary below is kept only as the "scanned document" visual.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
    if upscale and upscale != 1.0:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale,
                          interpolation=cv2.INTER_CUBIC)
    return gray


def adaptive_binarize(warped, block_size=35, C=15, do_shadow=True):
    """Produce a black-text-on-white binary scan from a warped crop.

    Returns a dict of intermediates: gray, flattened, binary.
    block_size must be odd; it is the neighbourhood size for the local threshold.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
    flattened = remove_shadow(gray) if do_shadow else gray

    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(
        flattened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C,
    )

    # Remove tiny speckle without eroding the thin receipt font.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return {"gray": gray, "flattened": flattened, "binary": binary}
