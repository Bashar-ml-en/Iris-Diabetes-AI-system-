import cv2
import numpy as np
import tensorflow as tf
import os

# ==========================================
# Mock Models Loading for Demonstration
# ==========================================
def load_stage1_classifier():
    """
    Loads your Stage 1 Binary Classification Model.
    """
    print("[System] Loading Stage 1 Binary Classifier...")
    try:
        # Load with compile=False since we only use it for inference, bypassing metric serialization bugs
        model = tf.keras.models.load_model('stage1_diabetes_classifier.h5', compile=False)
    except OSError:
        print("Warning: stage1_diabetes_classifier.h5 not found! Falling back to random mock.")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(150, 150, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    return model

def load_stage2_regressor():
    """
    Loads the Stage 2 Continuous Regression Model.
    """
    print("[System] Loading Stage 2 Glucose Regressor...")
    try:
        # Load with compile=False since we only use it for inference, bypassing metric serialization bugs
        model = tf.keras.models.load_model('stage2_glucose_regressor.h5', compile=False)
    except OSError:
        print("Warning: stage2_glucose_regressor.h5 not found! Falling back to random mock.")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(150, 150, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='linear')
        ])
    return model

# ==========================================
# Preprocessing functions
# ==========================================
def preprocess_for_classification(image):
    """Prepares the whole iris for the classifier (Stage 1)"""
    # Assuming the classifier takes 150x150 images
    resized = cv2.resize(image, (150, 150))
    return np.expand_dims(resized.astype('float32') / 255.0, axis=0)

def extract_topographical_roi(image):
    """
    Simulates Daugman's rubber sheet model to unwrap the iris 
    and extract the specific sector corresponding to the pancreas for Stage 2.
    """
    h, w = image.shape[:2]
    sector = image[h//2:h, w//2:w]
    resized = cv2.resize(sector, (150, 150))
    return np.expand_dims(resized.astype('float32') / 255.0, axis=0)

# ==========================================
# The Master Cascaded Pipeline
# ==========================================
def run_clinical_pipeline(image_path, classifier, regressor):
    print(f"\n--- Processing Patient Iris: {os.path.basename(image_path)} ---")
    
    # 1. Read Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return
        
    # 2. STAGE 1: Diabetes Check
    clf_input = preprocess_for_classification(img)
    diabetes_prob = classifier.predict(clf_input, verbose=0)[0][0]
    
    # 3. Decision Logic
    THRESHOLD = 0.5
    if diabetes_prob < THRESHOLD:
        print(f"Result: Patient DOES NOT have diabetes. (Confidence: {(1-diabetes_prob)*100:.1f}%)")
        print("Action: No further continuous prediction required. Recommend standard yearly checkup.")
        return
    else:
        print(f"Result: Patient IS DIABETIC. (Confidence: {diabetes_prob*100:.1f}%)")
        print("Action: Passing to Stage 2 Regression Pipeline...")
        
        # 4. STAGE 2: Continuous Glucose Prediction
        reg_input = extract_topographical_roi(img)
        predicted_glucose = regressor.predict(reg_input, verbose=0)[0][0]
        
        # The model is now trained on real numbers, so we use the prediction directly!
        scaled_glucose = float(predicted_glucose)
        
        print(f"\n>>> FINAL DIAGNOSIS <<<")
        print(f"Current Blood Sugar Estimate: {scaled_glucose:.1f} mg/dL")
        
        if scaled_glucose > 180:
            print("Medical Advice: Blood sugar is HIGH. Recommend immediate insulin adjustment according to doctor's plan.")
        else:
            print("Medical Advice: Blood sugar is elevated but stable. Maintain current diet and medication.")


if __name__ == "__main__":
    # Initialize the system
    stage1_model = load_stage1_classifier()
    stage2_model = load_stage2_regressor()
    
    # Test with a mock image if it exists
    test_img = "synthetic_iris_dataset/images/control_000.png"
    if os.path.exists(test_img):
        run_clinical_pipeline(test_img, stage1_model, stage2_model)
    else:
        print(f"Test image {test_img} not found. Please run generate_mock_dataset.py first.")
