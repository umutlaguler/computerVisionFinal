"""
Step 11 of the pipeline: text extraction with Tesseract (pytesseract).

Tesseract is a classical (LSTM-based) OCR engine. We deliberately use it rather
than a heavy deep-learning OCR so that the *contribution of our pre-processing*
is visible: a weak-ish engine on a well-cleaned image is the whole point of the
ablation study (raw vs pipeline).
"""

import pytesseract


# --psm 6  -> assume a single uniform block of text (good for a whole receipt).
# --oem 1  -> LSTM engine only.
DEFAULT_CONFIG = "--oem 1 --psm 6"
DEFAULT_LANG = "eng"


def image_to_text(image, lang=DEFAULT_LANG, config=DEFAULT_CONFIG):
    """Run OCR on a binary/gray/BGR image and return the recognised text."""
    return pytesseract.image_to_string(image, lang=lang, config=config)


def image_to_data(image, lang=DEFAULT_LANG, config=DEFAULT_CONFIG):
    """Return Tesseract's per-word data (incl. confidences) as a dict.

    Useful for drawing word boxes / reporting mean confidence in the demo.
    """
    import pytesseract as pt
    return pt.image_to_data(image, lang=lang, config=config,
                            output_type=pt.Output.DICT)


def mean_confidence(image, lang=DEFAULT_LANG, config=DEFAULT_CONFIG):
    """Mean confidence over recognised words (ignores -1 / empty tokens)."""
    data = image_to_data(image, lang=lang, config=config)
    confs = [int(c) for c, t in zip(data["conf"], data["text"])
             if str(c).lstrip("-").isdigit() and int(c) >= 0 and t.strip()]
    return sum(confs) / len(confs) if confs else 0.0
