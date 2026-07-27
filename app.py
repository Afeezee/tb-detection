"""
Streamlit demonstration interface for the TB detection system.

Run:
    streamlit run app.py
"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import streamlit as st
import torch

import config
from src.dataset import get_eval_transforms
from src.model import get_model
from src.gradcam import generate_gradcam_overlay, save_overlay
from src.train import get_device
from src import db

st.set_page_config(page_title="TB Detection — Demo", layout="wide")

MODEL_NAME = config.MODEL_NAME
CLASS_NAMES = {0: "Normal", 1: "TB-positive"}


@st.cache_resource
def load_model():
    device = get_device()
    checkpoint_path = config.MODELS_DIR / f"{MODEL_NAME}_best.pt"
    if not checkpoint_path.exists():
        return None, device
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = get_model(MODEL_NAME, num_classes=config.NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, device


def preprocess_for_inference(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if config.APPLY_CLAHE:
        clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID_SIZE
        )
        gray = clahe.apply(gray)
    resized_gray = cv2.resize(gray, (config.IMG_SIZE, config.IMG_SIZE))
    resized_rgb = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2RGB)

    transform = get_eval_transforms()
    tensor = transform(image=resized_rgb)["image"].unsqueeze(0)
    return tensor, resized_rgb


def main():
    st.title("🫁 Deep Learning-Based Tuberculosis Detection")
    st.caption(
        "Final-year project demo — Adesanlu Martins (U/22/CS/0011), "
        "Supervisor: Miss Shadare"
    )

    model, device = load_model()
    if model is None:
        st.warning(
            f"No trained checkpoint found at "
            f"`models/{MODEL_NAME}_best.pt`. Run `python -m src.train` first."
        )
        return

    tab_predict, tab_history = st.tabs(["Predict", "History"])

    with tab_predict:
        col_input, col_result = st.columns(2)

        with col_input:
            patient_ref = st.text_input("Patient reference / ID (optional)")
            uploaded_file = st.file_uploader(
                "Upload a chest X-ray (PNG/JPEG)", type=["png", "jpg", "jpeg"]
            )
            clinician_notes = st.text_area("Clinician notes (optional)")

        if uploaded_file is not None:
            input_tensor, display_rgb = preprocess_for_inference(uploaded_file)

            with col_input:
                st.image(display_rgb, caption="Preprocessed input", use_column_width=True)

            with torch.no_grad():
                outputs = model(input_tensor.to(device))
                probs = torch.softmax(outputs, dim=1)[0]
                pred_class = int(probs.argmax())
                confidence = float(probs[pred_class])

            overlay = generate_gradcam_overlay(
                model, MODEL_NAME, input_tensor, display_rgb, device, target_class=1
            )

            with col_result:
                label = CLASS_NAMES[pred_class]
                if pred_class == 1:
                    st.error(f"Prediction: **{label}**  (confidence: {confidence:.1%})")
                else:
                    st.success(f"Prediction: **{label}**  (confidence: {confidence:.1%})")

                st.image(overlay, caption="Grad-CAM — regions driving the prediction",
                          use_column_width=True)

                if st.button("Save this result"):
                    filename = f"{Path(uploaded_file.name).stem}_{datetime.now():%Y%m%d%H%M%S}.png"
                    gradcam_path = save_overlay(overlay, f"gradcam_{filename}")
                    db.insert_prediction(
                        image_filename=uploaded_file.name,
                        prediction=label,
                        confidence=confidence,
                        model_name=MODEL_NAME,
                        patient_ref=patient_ref or None,
                        gradcam_path=str(gradcam_path),
                        clinician_notes=clinician_notes or None,
                    )
                    st.toast("Saved to database.")

    with tab_history:
        st.subheader("Recent predictions (Neon Postgres)")
        try:
            records = db.fetch_history(limit=50)
            if records:
                st.dataframe(records, use_container_width=True)
            else:
                st.info("No predictions saved yet.")
        except RuntimeError as e:
            st.warning(str(e))


if __name__ == "__main__":
    main()
