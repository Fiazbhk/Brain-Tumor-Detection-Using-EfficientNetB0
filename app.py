import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from streamlit_option_menu import option_menu
import io

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="NeuroScan AI",
    layout="centered"
)

# -----------------------------
# CONSTANTS
# -----------------------------
MODEL_PATH = "models/best_model_phase2.keras"
IMAGE_SIZE = (224, 224)
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

CLASS_INFO = {
    "Glioma": {
        "color": "#e74c3c",
        "bg": "#fdf0ef",
        "severity": "High",
        "desc": "Gliomas arise from glial cells in the brain or spine. They are the most common type of primary brain tumors and can range from slow-growing (grade I) to highly aggressive (grade IV)."
    },
    "Meningioma": {
        "color": "#e67e22",
        "bg": "#fef6ec",
        "severity": "Moderate",
        "desc": "Meningiomas originate from the meninges surrounding the brain and spinal cord. Most are benign and slow-growing, but some can be atypical or malignant."
    },
    "No Tumor": {
        "color": "#27ae60",
        "bg": "#eafaf1",
        "severity": "None",
        "desc": "No tumor was detected in this MRI scan. The image appears to show normal brain tissue. Always confirm results with a certified radiologist."
    },
    "Pituitary": {
        "color": "#2980b9",
        "bg": "#eaf4fb",
        "severity": "Low-Moderate",
        "desc": "Pituitary tumors develop in the pituitary gland at the base of the brain. Most are non-cancerous adenomas that can affect hormone production and vision."
    }
}

# -----------------------------
# LOAD MODEL (cached)
# -----------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def run_inference(model, image: Image.Image):
    img = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASSES[idx], float(preds[idx]), preds

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("NeuroScan AI")

st.sidebar.markdown(
    """
**AI-Powered Brain Tumor Detection**

This system uses a deep learning model trained on brain MRI scans
to classify tumors into one of four categories.

**Tech Stack:** Python, TensorFlow, EfficientNetB0, Streamlit
"""
)

st.sidebar.markdown("### Capabilities")
st.sidebar.markdown(
    """
- MRI image upload and analysis
- 4-class tumor classification
- Per-class confidence breakdown
- Research-grade inference pipeline
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
[GitHub](https://github.com/Fiazbhk) |
[LinkedIn](https://www.linkedin.com/in/fiazbhk/) |
[Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
"""
)

# -----------------------------
# TITLE
# -----------------------------
st.title("NeuroScan AI")

# -----------------------------
# TOP HORIZONTAL MENU
# -----------------------------
selected_tab = option_menu(
    menu_title=None,
    options=["Tumor Detection", "About the Model"],
    icons=["activity", "info-circle"],
    orientation="horizontal",
    default_index=0
)

# -----------------------------
# TAB 1: TUMOR DETECTION
# -----------------------------
if selected_tab == "Tumor Detection":
    st.subheader("MRI Image Analysis")

    st.warning(
        "Medical Disclaimer: This tool is for educational and research purposes only. "
        "It is not a substitute for professional medical diagnosis. "
        "Always consult a qualified radiologist or physician."
    )

    uploaded_file = st.file_uploader(
        "Upload Brain MRI Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(io.BytesIO(uploaded_file.read()))

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
            st.caption(f"File: {uploaded_file.name} · {uploaded_file.size // 1024} KB")

        with col2:
            st.markdown("#### Run Analysis")
            st.markdown(
                "The model will classify this MRI into one of four categories: "
                "Glioma, Meningioma, Pituitary Tumor, or No Tumor."
            )

            st.divider()

            if st.button("Generate Diagnostic Result"):
                with st.spinner("Analysing MRI scan..."):
                    try:
                        model = load_model()
                        label, confidence, all_probs = run_inference(model, image)
                        info = CLASS_INFO[label]

                        result_msg = (
                            f"### ASSESSMENT: {label.upper()}\n"
                            f"Confidence: {confidence * 100:.2f}% · Severity: {info['severity']}"
                        )

                        if label == "No Tumor":
                            st.success(result_msg)
                        elif info["severity"] == "High":
                            st.error(result_msg)
                        else:
                            st.warning(result_msg)

                        st.markdown(
                            f"<div style='border-left: 4px solid {info['color']};"
                            f"padding: 0.6rem 0.9rem; background: {info['bg']};"
                            f"border-radius: 6px; font-size: 0.85rem; color: #374151; margin-top: 0.5rem'>"
                            f"{info['desc']}"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                        st.markdown("#### Confidence Breakdown")
                        for cls, prob in zip(CLASSES, all_probs):
                            pct = float(prob) * 100
                            is_top = cls == label
                            bar_color = CLASS_INFO[cls]["color"] if is_top else "#d1d5db"
                            fw = "700" if is_top else "400"
                            fc = "#1a1a2e" if is_top else "#6b7280"
                            st.markdown(
                                f"""<div style="margin-bottom: 0.6rem;">
                                    <div style="display: flex; justify-content: space-between;
                                                font-size: 0.82rem; font-weight: {fw}; color: {fc};">
                                        <span>{cls}</span>
                                        <span>{pct:.2f}%</span>
                                    </div>
                                    <div style="background: #f3f4f6; border-radius: 4px; height: 8px; margin-top: 3px;">
                                        <div style="width: {min(pct, 100):.1f}%; height: 8px; border-radius: 4px;
                                                    background: {bar_color};"></div>
                                    </div>
                                </div>""",
                                unsafe_allow_html=True
                            )

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
            else:
                st.info("Click Generate Diagnostic Result to analyse the scan.")

# -----------------------------
# TAB 2: ABOUT
# -----------------------------
if selected_tab == "About the Model":
    st.subheader("Model Information")

    st.markdown(
        """
**Algorithm:** EfficientNetB0 with Transfer Learning (ImageNet weights)
**Training Strategy:** Two-phase training — frozen base then full fine-tuning
**Preprocessing:** EfficientNet-specific preprocess_input
**Input Size:** 224 x 224 x 3
**Dataset:** Brain Tumor MRI Dataset by Masoud Nickparvar (Kaggle)
**Target:**
- 0 → Glioma
- 1 → Meningioma
- 2 → No Tumor
- 3 → Pituitary

**Performance:**
- Training Accuracy: 97.43%
- Validation Accuracy: 96.34%
- Test Accuracy: 93.00%

**Notes:**
- Predictions are probabilistic, not diagnostic
- Intended for educational and research use only
- Not a substitute for professional medical evaluation
- Meningioma is the most challenging class due to visual similarity with Glioma
"""
    )