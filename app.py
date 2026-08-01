import os
import tempfile
import concurrent.futures

import streamlit as st

from text_module.inference import predict_text
from image_branch.inference import predict_image

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

        os.unlink(tmp_path)

        st.markdown("---")
        st.markdown("**Image Branch**")
        for d in DISEASE_NAMES:
            st.write(f"{d:<30} {img_probs[d]:.4f}")

        st.markdown("---")
        st.markdown("**Text Branch**")
        for d in DISEASE_NAMES:
            st.write(f"{d:<30} {text_probs[d]:.4f}")
