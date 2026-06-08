"""
Parse the CVAT annotations.xml that ships with the dataset.

For every image it exposes:
  - the `receipt` polygon  -> ground truth for the document region (for IoU)
  - the text of shop/item/date_time/total boxes -> ground truth for OCR
    accuracy (CER/WER), so we never have to transcribe receipts by hand.
"""

import os
from dataclasses import dataclass, field

import numpy as np
from lxml import etree


TEXT_LABELS = ("shop", "item", "date_time", "total")


@dataclass
class ImageAnnotation:
    name: str                       # e.g. "images/0.jpg"
    width: int
    height: int
    receipt_polygon: np.ndarray = None   # (N, 2) float32, or None
    texts: list = field(default_factory=list)  # list of (label, text)

    @property
    def stem(self):
        return os.path.splitext(os.path.basename(self.name))[0]

    def ground_truth_text(self):
        """Concatenate all annotated text fields into one reference string.

        Order: shop -> items -> date_time -> total, which roughly follows the
        physical layout of a receipt.
        """
        order = {label: i for i, label in enumerate(TEXT_LABELS)}
        ordered = sorted(self.texts, key=lambda lt: order.get(lt[0], 99))
        return "\n".join(t for _, t in ordered if t.strip())


def _parse_points(points_str):
    pts = [p.split(",") for p in points_str.strip().split(";") if p]
    return np.array([[float(x), float(y)] for x, y in pts], dtype="float32")


def load_annotations(xml_path):
    """Return {stem: ImageAnnotation} for every <image> in the file."""
    tree = etree.parse(xml_path)
    out = {}
    for img in tree.findall(".//image"):
        ann = ImageAnnotation(
            name=img.get("name"),
            width=int(img.get("width")),
            height=int(img.get("height")),
        )
        for child in img:
            if child.tag == "polygon" and child.get("label") == "receipt":
                ann.receipt_polygon = _parse_points(child.get("points"))
            elif child.tag == "box" and child.get("label") in TEXT_LABELS:
                text_el = child.find("attribute[@name='text']")
                if text_el is not None and text_el.text:
                    ann.texts.append((child.get("label"), text_el.text))
        out[ann.stem] = ann
    return out
