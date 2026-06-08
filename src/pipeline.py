"""
End-to-end receipt scanning + OCR pipeline.

Ties together: preprocess -> detect -> transform -> postprocess -> ocr.
Returns a result object holding every intermediate image so the CLI, the
Streamlit app and the notebook can all visualise the same stages.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import preprocess, detect, transform, postprocess, ocr


@dataclass
class PipelineResult:
    original: np.ndarray
    stages: dict = field(default_factory=dict)   # name -> image (for display)
    quad: np.ndarray = None                       # 4 corners in ORIGINAL coords
    detection_method: str = "none"                # quad | minarea | fallback
    warped: np.ndarray = None
    binary: np.ndarray = None
    text: str = ""

    @property
    def detected(self):
        return self.quad is not None


def scan(image, lang="eng", use_bilateral=True, use_clahe=True,
         block_size=35, C=15, do_shadow=True):
    """Run the full pipeline on a BGR image and return a PipelineResult.

    Detection is done on a resized copy for speed/robustness, then the corners
    are scaled back and the warp is applied to the full-resolution image so OCR
    gets maximum detail.
    """
    result = PipelineResult(original=image, stages={})

    # --- Steps 1-5: pre-processing on a resized copy ----------------------
    small, scale = preprocess.resize_keep_aspect(image)
    pre = preprocess.preprocess_for_detection(
        small, use_bilateral=use_bilateral, use_clahe=use_clahe
    )
    result.stages.update(pre)
    small_area = small.shape[0] * small.shape[1]

    # --- Steps 6-7: detect the document quad ------------------------------
    # Detection runs on the Otsu segmentation mask (robust for text-dense
    # receipts); the Canny `closed` image is kept only for visualisation.
    quad_small, largest = detect.find_document_quad(pre["mask"], small_area)

    if quad_small is not None:
        result.detection_method = "quad"
    elif largest is not None:
        # Fallback: wrap the biggest contour in a min-area rectangle.
        quad_small = detect.min_area_quad(largest)
        result.detection_method = "minarea"

    if quad_small is not None:
        result.stages["detected"] = detect.draw_quad(small, quad_small)
        quad = detect.scale_quad(quad_small, scale)          # back to original
        result.quad = quad

        # --- Steps 8-9: perspective warp on the full-res image ------------
        warped, _ = transform.warp_to_top_down(image, quad)
        result.warped = warped
        result.stages["warped"] = warped
    else:
        # Total failure: OCR the original so the user still gets *something*.
        result.detection_method = "fallback"
        result.warped = image
        result.stages["warped"] = image

    # --- Step 10: binarise (the "scanned document" deliverable) -----------
    post = postprocess.adaptive_binarize(
        result.warped, block_size=block_size, C=C, do_shadow=do_shadow
    )
    result.binary = post["binary"]
    result.stages["flattened"] = post["flattened"]
    result.stages["binary"] = post["binary"]

    # --- Step 11: OCR ------------------------------------------------------
    # OCR runs on the grayscale-upscaled crop (better for Tesseract than the
    # hard binary); the binary above is kept purely as the scanned visual.
    ocr_input = postprocess.prepare_for_ocr(result.warped)
    result.text = ocr.image_to_text(ocr_input, lang=lang)
    return result
