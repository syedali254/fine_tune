"""
Streamlit UI for the pediatric pneumonia classifier.

Runs three independent branches on a single button click:
    - Image branch:    fine-tuned DenseNet121 (TorchXRayVision) + Grad-CAM
    - Text branch:     fine-tuned BioBERT symptom classifier (Experiment 1)
    - Evidence branch: source-grounded clinical finding extraction (Experiment 2)

The branches are independent evidence outputs; no fusion is applied.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
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
from text_inference import (
    CHECKPOINT_PATH as TEXT_CHECKPOINT_PATH,
    load_text_model,
    load_text_tokenizer,
    predict_text,
)
from evidence_inference import extract_evidence

st.set_page_config(
    page_title="Pediatric Pneumonia Classifier",
    page_icon="🫁",
    layout="centered",
)

CHECKPOINT_PATH = Path("checkpoints/best_model.pth")
CONFIG = Config()
DEVICE = CONFIG.device


@st.cache_resource(show_spinner="Loading fine-tuned model...")
def _load_image_model():
    if not CHECKPOINT_PATH.is_file():
        st.error(f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}")
        st.info("Train a model first with: `python train.py`")
        st.stop()
    return load_model(str(CHECKPOINT_PATH), CONFIG)


image_model = _load_image_model()


@st.cache_resource
def _get_gradcam(_model):
    return GradCAM(_model)


gradcam = _get_gradcam(image_model)


@st.cache_resource(show_spinner="Loading BioBERT model...")
def _load_text_model():
    if not TEXT_CHECKPOINT_PATH.is_file():
        st.error(f"BioBERT checkpoint not found: {TEXT_CHECKPOINT_PATH.resolve()}")
        st.stop()
    return load_text_model(str(TEXT_CHECKPOINT_PATH), DEVICE)


@st.cache_resource(show_spinner="Loading BioBERT tokenizer...")
def _load_text_tokenizer():
    return load_text_tokenizer()


def run_image_branch(image_path: Path):
    """Preprocess, predict and generate Grad-CAM for the image branch."""
    gradcam_error = None
    try:
        tensor = preprocess_image(image_path)
        predicted_class, prob, confidence = predict(image_model, tensor, DEVICE)
        try:
            heatmap = gradcam.generate(tensor)
            original_img = skimage.io.imread(str(image_path))
            overlay = overlay_heatmap(original_img, heatmap, alpha=0.45)
        except Exception as e:
            overlay = None
            gradcam_error = str(e)
        return {
            "ok": True,
            "predicted_class": predicted_class,
            "prob": prob,
            "normal_prob": 1.0 - prob,
            "confidence": confidence,
            "overlay": overlay,
            "gradcam_error": gradcam_error,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def run_text_branch(text: str, text_model, tokenizer):
    """Tokenize and run BioBERT inference for the text branch."""
    try:
        p_normal, p_pneumonia, prediction = predict_text(
            text_model, tokenizer, text, DEVICE
        )
        return {
            "ok": True,
            "p_normal": p_normal,
            "p_pneumonia": p_pneumonia,
            "prediction": prediction,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def run_evidence_branch(text: str):
    """Run Experiment 2 clinical finding extraction on the symptom text."""
    try:
        result = extract_evidence(text)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _image_label(predicted_class: str) -> str:
    return "Pneumonia" if predicted_class == "PNEUMONIA" else "Normal"


def display_evidence_section(text, ev_result):
    """Render the Experiment 2 structured clinical evidence output."""
    st.markdown("---")
    st.markdown("### 🧬 Evidence-Based Assessment (Experiment 2)")
    if not ev_result["ok"]:
        st.error(f"Evidence extraction failed: {ev_result['error']}")
        return
    res = ev_result["result"]
    st.markdown(f'**Input symptoms:** "{text}"')

    supportive = res["supportive_findings"]
    concerning = res["concerning_findings"]
    contextual = (
        res["upper_respiratory_findings"]
        + res["influenza_like_findings"]
        + res["nonspecific_findings"]
    )
    unmapped = res["unmapped_findings"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Pneumonia-associated — supportive**")
        if supportive:
            for f in supportive:
                st.markdown(f"✅ {f}")
        else:
            st.caption("None detected")
    with col2:
        st.markdown("**Respiratory severity indicators**")
        if concerning:
            for f in concerning:
                st.markdown(f"⚠️ {f}")
        else:
            st.caption("None detected")

    st.markdown("**Contextual / overlapping features**")
    if contextual:
        for f in contextual:
            st.markdown(f"- {f}")
    else:
        st.caption("None detected")

    if unmapped:
        with st.expander(f"Unmapped findings ({len(unmapped)})"):
            for f in unmapped:
                st.markdown(f"- {f}")
            st.caption("Not mapped to the knowledge base — no pneumonia significance assigned.")

    s = res["summary"]
    st.markdown(
        f"**Summary:** {s['supportive_count']} supportive · "
        f"{s['concerning_count']} concerning · "
        f"{s['upper_respiratory_count']} upper-respiratory · "
        f"{s['influenza_like_count']} influenza-like"
    )
    st.markdown(f"*{res['interpretation']}*")
    st.caption(res["disclaimer"])


st.title("🫁 Multimodal Pediatric Pneumonia Classifier")
st.markdown(
    "**Image:** fine-tuned DenseNet121 (TorchXRayVision) on Kaggle pediatric chest X-rays.  "
    "**Text:** fine-tuned BioBERT symptom classifier (Experiment 1).  "
    "**Evidence:** source-grounded clinical finding extraction (Experiment 2)."
)

with st.sidebar:
    st.markdown("### 🧠 Models")
    st.markdown("- **Image:** DenseNet121 (fine-tuned)")
    st.markdown("- **Text:** BioBERT (fine-tuned)")
    st.markdown("- **Evidence:** knowledge-base extractor (deterministic)")
    st.divider()
    st.caption("MSc Research Project — Domain Shift in Chest X-Ray Analysis")

uploaded = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Supported: JPG, JPEG, PNG",
)

symptoms = st.text_area(
    "Patient symptoms (free text)",
    placeholder="e.g. I have fever, cough, shortness of breath and chills.",
)

if uploaded is not None:
    st.image(uploaded, caption="Uploaded X-Ray", use_container_width=True)

run = st.button("Run Multimodal Assessment", type="primary", use_container_width=True)

if run:
    image_path = None
    if uploaded is not None:
        image_bytes = uploaded.read()
        image_path = Path("_temp_inference.png")
        image_path.write_bytes(image_bytes)

    text = symptoms.strip() if symptoms and symptoms.strip() else None

    if image_path is None and text is None:
        st.info("Please provide at least one input: a chest X-ray image and/or a symptom text description.")
    else:
        text_model = None
        text_tokenizer = None
        if text is not None:
            text_model = _load_text_model()
            text_tokenizer = _load_text_tokenizer()

        with ThreadPoolExecutor(max_workers=3) as pool:
            image_future = (
                pool.submit(run_image_branch, image_path) if image_path is not None else None
            )
            text_future = (
                pool.submit(run_text_branch, text, text_model, text_tokenizer)
                if text is not None
                else None
            )
            evidence_future = (
                pool.submit(run_evidence_branch, text) if text is not None else None
            )
            image_result = image_future.result() if image_future is not None else None
            text_result = text_future.result() if text_future is not None else None
            evidence_result = evidence_future.result() if evidence_future is not None else None

        # ── Image-based assessment ─────────────────────────────────────────
        if image_result is not None:
            st.markdown("---")
            st.markdown("### 📷 Image-Based Assessment")
            if image_result["ok"]:
                predicted_class = image_result["predicted_class"]
                if predicted_class == "PNEUMONIA":
                    st.error(f"**Prediction: {predicted_class}**", icon="⚠️")
                else:
                    st.success(f"**Prediction: {predicted_class}**", icon="✅")
                col1, col2 = st.columns(2)
                col1.metric("P(Pneumonia)", f"{image_result['prob']:.1%}")
                col2.metric("P(Non-pneumonia)", f"{image_result['normal_prob']:.1%}")
                st.markdown(f"**Overall confidence:** {image_result['confidence']:.1f}%")
                if image_result["overlay"] is not None:
                    st.image(
                        image_result["overlay"],
                        caption="Grad-CAM: regions the model focused on",
                        use_container_width=True,
                    )
                else:
                    st.warning(f"Grad-CAM could not be generated: {image_result['gradcam_error']}")
            else:
                st.error(f"Image prediction failed: {image_result['error']}")
        else:
            st.markdown("---")
            st.markdown("### 📷 Image-Based Assessment")
            st.info("No chest X-ray image provided — image branch skipped.")

        # ── Text-based ML assessment ───────────────────────────────────────
        if text_result is not None:
            st.markdown("---")
            st.markdown("### 📝 Text-Based ML Assessment")
            if text_result["ok"]:
                st.markdown(f'**Input symptoms:** "{text}"')
                if text_result["prediction"] == "Pneumonia":
                    st.error(f"**Prediction: {text_result['prediction']}**", icon="⚠️")
                else:
                    st.success(f"**Prediction: {text_result['prediction']}**", icon="✅")
                col1, col2 = st.columns(2)
                col1.metric("P(Pneumonia)", f"{text_result['p_pneumonia']:.1%}")
                col2.metric("P(Non-pneumonia)", f"{text_result['p_normal']:.1%}")
            else:
                st.error(f"Text prediction failed: {text_result['error']}")
        else:
            st.markdown("---")
            st.markdown("### 📝 Text-Based ML Assessment")
            st.info("No symptom text provided — text branch skipped.")

        # ── Evidence-based assessment (Experiment 2) ──────────────────────
        if evidence_result is not None:
            display_evidence_section(text, evidence_result)
        else:
            st.markdown("---")
            st.markdown("### 🧬 Evidence-Based Assessment (Experiment 2)")
            st.info("No symptom text provided — evidence branch skipped.")

        # ── Multimodal view (independent evidence, no fusion) ──────────────
        if (
            image_result is not None
            and image_result["ok"]
            and text_result is not None
            and text_result["ok"]
        ):
            st.markdown("---")
            st.markdown("### 🔬 Multimodal View")
            st.markdown("Independent evidence outputs (no fusion is applied).")
            cols = st.columns(3)
            cols[0].metric("Image evidence", _image_label(image_result["predicted_class"]))
            cols[1].metric("Text evidence", text_result["prediction"])
            if evidence_result is not None and evidence_result["ok"]:
                ev = evidence_result["result"]["summary"]
                ev_total = ev["supportive_count"] + ev["concerning_count"]
                cols[2].metric(
                    "Evidence findings (Exp 2)",
                    f"{ev_total} "
                    f"({ev['supportive_count']} supportive, {ev['concerning_count']} concerning)",
                )

    if image_path is not None:
        image_path.unlink(missing_ok=True)

else:
    st.info("👆 Upload a chest X-ray and/or enter symptoms, then press **Run Multimodal Assessment**.")

st.divider()
st.caption(
    "**Note:** This is a research prototype. Not for clinical use.  "
    "Predictions should be validated by a qualified medical professional."
)