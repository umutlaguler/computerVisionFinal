"""
Streamlit GUI for the receipt scanner + OCR pipeline.

Run with:
    streamlit run app.py

Upload a receipt photo (or pick a sample); the app shows every classical-CV
stage of the pipeline and the extracted text, so it doubles as a live demo for
the presentation.
"""

import glob
import os

import cv2
import numpy as np
import streamlit as st

from src import pipeline


SAMPLE_DIR = "data/raw/images"

st.set_page_config(page_title="Receipt Scanner + OCR", layout="wide")


def to_rgb(img):
    """Convert a BGR / gray OpenCV image to RGB for Streamlit display."""
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False)
def run_pipeline(img_bytes, lang, use_bilateral, use_clahe, block_size, C,
                 do_shadow):
    """Decode bytes -> BGR and run the pipeline. Cached on the inputs."""
    arr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    res = pipeline.scan(image, lang=lang, use_bilateral=use_bilateral,
                        use_clahe=use_clahe, block_size=block_size, C=C,
                        do_shadow=do_shadow)
    return image, res


st.title("🧾 Receipt Scanner + OCR")
st.caption("Classical computer-vision pipeline: edge/segmentation → contour → "
           "4-corner detection → perspective warp (homography) → adaptive "
           "threshold → Tesseract OCR")

# ---- Sidebar: input + parameters ----------------------------------------
with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload a receipt photo",
                                type=["jpg", "jpeg", "png"])
    samples = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.jpg")) +
                     glob.glob(os.path.join(SAMPLE_DIR, "*.JPG")),
                     key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
                     if os.path.splitext(os.path.basename(p))[0].isdigit() else 0)
    sample_names = ["(none)"] + [os.path.basename(p) for p in samples]
    chosen_sample = st.selectbox("…or pick a sample", sample_names)

    st.header("Parameters")
    lang = st.text_input("Tesseract language", value="eng")
    use_bilateral = st.checkbox("Bilateral filter (edge-preserving)", value=True)
    use_clahe = st.checkbox("CLAHE contrast boost", value=True)
    do_shadow = st.checkbox("Shadow removal (binary view)", value=True)
    block_size = st.slider("Adaptive threshold block size", 11, 71, 35, step=2)
    C = st.slider("Adaptive threshold C", 1, 30, 15)

# ---- Resolve the chosen image to bytes ----------------------------------
img_bytes = None
if uploaded is not None:
    img_bytes = uploaded.getvalue()
elif chosen_sample != "(none)":
    with open(os.path.join(SAMPLE_DIR, chosen_sample), "rb") as f:
        img_bytes = f.read()

if img_bytes is None:
    st.info("⬅️ Upload a receipt photo or pick a sample from the sidebar.")
    st.stop()

# ---- Run pipeline --------------------------------------------------------
with st.spinner("Running pipeline…"):
    image, res = run_pipeline(img_bytes, lang, use_bilateral, use_clahe,
                              block_size, C, do_shadow)

# ---- Status banner -------------------------------------------------------
method_msg = {
    "quad": "✅ Document detected as a clean 4-corner quadrilateral.",
    "minarea": "⚠️ No clean quad found — used a min-area rectangle fallback.",
    "fallback": "❌ Detection failed — OCR ran on the original image.",
}
banner = method_msg.get(res.detection_method, "")
(st.success if res.detection_method == "quad" else st.warning)(banner)

# ---- Tabs ----------------------------------------------------------------
tab_overview, tab_stages, tab_text = st.tabs(
    ["Overview", "Pipeline stages", "Extracted text"])

with tab_overview:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("1. Input")
        st.image(to_rgb(res.original), width="stretch")
    with c2:
        st.subheader("2. Detected document")
        det = res.stages.get("detected")
        st.image(to_rgb(det) if det is not None else to_rgb(res.original),
                 width="stretch")
    with c3:
        st.subheader("3. Scanned (binary)")
        st.image(to_rgb(res.binary), width="stretch")

with tab_stages:
    st.write("Each panel is one classical-CV step, in pipeline order.")
    stage_layout = [
        ("gray", "Grayscale"),
        ("denoised", "Denoised (bilateral)"),
        ("contrast", "CLAHE contrast"),
        ("edges", "Canny edges"),
        ("closed", "Morphological close"),
        ("mask", "Otsu segmentation (used for detection)"),
        ("detected", "Detected 4 corners"),
        ("warped", "Perspective warp (homography)"),
        ("flattened", "Shadow-flattened"),
        ("binary", "Adaptive threshold"),
    ]
    available = [(k, t) for k, t in stage_layout if res.stages.get(k) is not None]
    cols = st.columns(3)
    for i, (key, title) in enumerate(available):
        with cols[i % 3]:
            st.image(to_rgb(res.stages[key]), caption=f"{i+1}. {title}",
                     width="stretch")

with tab_text:
    st.subheader("OCR output")
    st.text_area("Recognised text", res.text, height=400)
    st.download_button("⬇️ Download text", res.text,
                       file_name="receipt.txt", mime="text/plain")
