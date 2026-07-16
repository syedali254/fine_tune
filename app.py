"""
Streamlit UI for the fine-tuned pediatric pneumonia classifier.

Shows prediction results and a Grad-CAM heatmap overlay.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import skimage.io

from pneumonia_config import Config
from inference import (
    load_model,
    preprocess_image,
    predict,
    GradCAM,
    overlay_heatmap,
)

st.set_page_config(
    page_title="Pediatric Pneumonia Classifier",
    page_icon="🫁",
    layout="centered",
)

CHECKPOINT_PATH = Path("checkpoints/best_model.pth")
CONFIG = Config()
DEVICE = CONFIG.device


@st.cache_resource(show_spinner="Loading fine-tuned model...")
def _load_model():
    if not CHECKPOINT_PATH.is_file():
        st.error(f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}")
        st.info("Train a model first with: `python train.py`")
        st.stop()
    # Load model once, then wrap it in GradCAM
    mdl = load_model(str(CHECKPOINT_PATH), CONFIG)
    return mdl


model = _load_model()

# Create GradCAM wrapper once (hooks are lightweight)
@st.cache_resource
def _get_gradcam(_model):
    return GradCAM(_model)


gradcam = _get_gradcam(model)

st.title("🫁 Pediatric Pneumonia Classifier")
st.markdown(
    "Fine-tuned DenseNet121 (TorchXRayVision) on Kaggle Pediatric Chest X-Rays."
)

with st.sidebar:
    st.markdown("### 🧠 Model")
    st.markdown("- **Current:** DenseNet121 (fine-tuned)")
    st.markdown("- *BioBERT* · *XGBoost* · *Fusion* — coming soon")
    st.divider()
    st.caption("MSc Research Project — Domain Shift in Chest X-Ray Analysis")

uploaded = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Supported: JPG, JPEG, PNG",
)

if uploaded is not None:
    # Save uploaded file temporarily
    image_bytes = uploaded.read()
    temp_path = Path("_temp_inference.png")
    temp_path.write_bytes(image_bytes)

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.image(uploaded, caption="Uploaded X-Ray", use_container_width=True)

    with col2:
        with st.spinner("Running inference..."):
            try:
                tensor = preprocess_image(temp_path)
                predicted_class, prob, confidence = predict(model, tensor, DEVICE)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                temp_path.unlink(missing_ok=True)
                st.stop()

        normal_prob = 1.0 - prob

        if predicted_class == "PNEUMONIA":
            st.error(f"**{predicted_class}**", icon="⚠️")
        else:
            st.success(f"**{predicted_class}**", icon="✅")

        st.metric("Pneumonia probability", f"{prob:.1%}")
        st.metric("Normal probability", f"{normal_prob:.1%}")
        st.progress(int(prob * 100), text="Pneumonia confidence")
        st.markdown(f"**Overall confidence:** {confidence:.1f}%")

        with st.expander("Raw model output"):
            st.code(f"Pneumonia probability: {prob:.6f}")
            st.code(f"Normal probability:    {normal_prob:.6f}")

    # ── Grad-CAM ──────────────────────────────────────────────────────────
    with st.spinner("Generating Grad-CAM heatmap..."):
        try:
            heatmap = gradcam.generate(tensor)
            original_img = skimage.io.imread(str(temp_path))
            overlay = overlay_heatmap(original_img, heatmap, alpha=0.45)
        except Exception as e:
            st.warning(f"Grad-CAM could not be generated: {e}")
            overlay = None

    if overlay is not None:
        st.image(overlay, caption="Grad-CAM: highlighted regions show what the model focused on", use_container_width=True)

    # Clean up temp file
    temp_path.unlink(missing_ok=True)

else:
    st.info("👆 Upload a chest X-ray to get started.")

st.divider()
st.caption(
    "**Note:** This is a research prototype. Not for clinical use.  "
    "Predictions should be validated by a qualified medical professional."
)
