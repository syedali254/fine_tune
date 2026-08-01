import os
import tempfile
import concurrent.futures

import streamlit as st

from text_module.inference import predict_text
from image_branch.inference import predict_image
from image_branch.gradcam import grad_cam_heatmap
from text_module.symptom_explain import explain_symptoms

DISEASE_NAMES = [
    "Atelectasis",
    "Emphysema",
    "Hiatal Hernia",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
    "Pulmonary Congestion",
    "Pulmonary Fibrosis",
]


st.set_page_config(page_title="Chest Disease Prediction", layout="centered")
st.title("Chest Disease Prediction")

xray = st.file_uploader("Chest X-ray", type=["png", "jpg", "jpeg"])
symptoms = st.text_area("Symptoms", placeholder="Enter symptoms...")

if st.button("Predict"):
    if not xray or not symptoms:
        st.error("Please provide both a chest X-ray and symptoms.")
    else:
        suffix = os.path.splitext(xray.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(xray.getbuffer())
            tmp_path = tmp.name

        with concurrent.futures.ThreadPoolExecutor() as ex:
            img_future = ex.submit(predict_image, tmp_path)
            text_future = ex.submit(predict_text, symptoms)
            img_probs = img_future.result()
            text_probs = text_future.result()

        st.markdown("---")
        st.markdown("**Image Branch**")
        for d in DISEASE_NAMES:
            st.write(f"{d:<30} {img_probs[d]:.4f}")

        st.markdown("---")
        st.markdown("**Image Explainability**")
        img_disease = max(img_probs, key=img_probs.get)
        #st.write(f"Predicted disease: {img_disease}")
        try:
            heatmap = grad_cam_heatmap(tmp_path, img_disease)
            st.image(heatmap, caption="Grad-CAM", use_container_width=True)
        except Exception as e:
            st.warning(f"Grad-CAM failed: {e}")

        st.markdown("---")
        st.markdown("**Text Branch**")
        for d in DISEASE_NAMES:
            st.write(f"{d:<30} {text_probs[d]:.4f}")

        st.markdown("---")
        st.markdown("**Text Explainability**")
        text_disease = max(text_probs, key=text_probs.get)
        expl = explain_symptoms(symptoms, text_disease)

        st.write("**Detected Clinical Symptoms**")
        for s in expl["detected"]:
            st.write(f"✓ {s}")

        st.write(f"**Associated with Top Prediction ({expl['predicted']})**")
        if expl["associated"]:
            for s in expl["associated"]:
                st.write(f"• {s}")
        else:
            st.write("• None of the detected symptoms are commonly associated with this prediction.")

        st.write("**Note:**")
        st.write("BioBERT prediction is primarily influenced by the semantic representation "
                 "of the detected symptoms rather than individual words.")

        os.unlink(tmp_path)
