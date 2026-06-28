import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# Import the core logic from our existing system
from end_to_end_system import (
    load_stage1_classifier,
    load_stage2_regressor,
    preprocess_for_classification,
    extract_topographical_roi
)

# Set page config
st.set_page_config(
    page_title="Iridology Diabetes AI",
    page_icon="👁️",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stAlert {
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Application Header
st.title("👁️ AI Iridology: Diabetes Screening System")
st.markdown("Upload a high-resolution image of a patient's iris to run the two-stage clinical pipeline.")

# Load models (Cached so they don't reload on every interaction)
@st.cache_resource
def init_models():
    clf = load_stage1_classifier()
    reg = load_stage2_regressor()
    return clf, reg

try:
    stage1_model, stage2_model = init_models()
    st.sidebar.success("✅ AI Models Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"Error loading models: {str(e)}")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Choose an iris image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Read Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Patient Iris', use_container_width=True)
    
    # Convert PIL Image to OpenCV format
    img_cv = np.array(image.convert('RGB'))
    # OpenCV expects BGR
    img_cv = img_cv[:, :, ::-1].copy()

    if st.button("Run Clinical Analysis", type="primary"):
        with st.spinner('Running AI Pipeline...'):
            # 2. STAGE 1: Diabetes Check
            clf_input = preprocess_for_classification(img_cv)
            diabetes_prob = stage1_model.predict(clf_input, verbose=0)[0][0]
            
            st.markdown("---")
            st.subheader("📋 Stage 1: Binary Classification")
            
            THRESHOLD = 0.5
            if diabetes_prob < THRESHOLD:
                st.success(f"**Result:** Patient DOES NOT have diabetes. (Confidence: {(1-diabetes_prob)*100:.1f}%)")
                st.info("💡 **Action:** No further continuous prediction required. Recommend standard yearly checkup.")
            else:
                st.warning(f"**Result:** Patient IS DIABETIC. (Confidence: {diabetes_prob*100:.1f}%)")
                st.info("💡 **Action:** Diabetes detected. Passing to Stage 2 Regression Pipeline for continuous glucose estimation...")
                
                # 3. STAGE 2: Continuous Glucose Prediction
                st.markdown("---")
                st.subheader("🩸 Stage 2: Continuous Glucose Regression")
                
                reg_input = extract_topographical_roi(img_cv)
                predicted_glucose = stage2_model.predict(reg_input, verbose=0)[0][0]
                
                # The model is now trained on real numbers, so we use the prediction directly!
                scaled_glucose = float(predicted_glucose)
                
                # Display final result nicely
                st.markdown(f'<p class="big-font">Current Estimated Blood Sugar: <span style="color:#ff4b4b">{scaled_glucose:.1f} mg/dL</span></p>', unsafe_allow_html=True)
                
                if scaled_glucose > 180:
                    st.error("🚨 **Medical Advice:** Blood sugar is HIGH. Recommend immediate review of insulin protocol according to doctor's plan.")
                else:
                    st.warning("⚠️ **Medical Advice:** Blood sugar is elevated but stable. Maintain current diet and medication.")
