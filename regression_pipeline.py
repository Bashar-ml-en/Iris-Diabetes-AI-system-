import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

# ==========================================
# Phase 1: Topographical ROI (Daugman's Rubber Sheet Simulation)
# ==========================================
def extract_topographical_roi(image):
    """
    Simulates Daugman's rubber sheet model to unwrap the iris 
    and extract the specific sector corresponding to the pancreas.
    For this mock implementation, we crop the bottom-right quadrant.
    """
    h, w = image.shape[:2]
    # Simple simulated sector crop (e.g., Pancreas sector in iridology)
    sector = image[h//2:h, w//2:w]
    return cv2.resize(sector, (150, 150))

# ==========================================
# Phase 2: Data Loading & Preprocessing
# ==========================================
def load_and_preprocess_data(base_dir="synthetic_iris_dataset"):
    csv_path = os.path.join(base_dir, "glucose_labels.csv")
    images_dir = os.path.join(base_dir, "images")
    
    df = pd.read_csv(csv_path)
    
    X = []
    y = []
    
    for index, row in df.iterrows():
        img_path = os.path.join(images_dir, row["image_filename"])
        img = cv2.imread(img_path)
        if img is not None:
            # 1. Extract Sector
            roi = extract_topographical_roi(img)
            # 2. Normalize
            roi = roi.astype('float32') / 255.0
            X.append(roi)
            # 3. Load continuous glucose label
            y.append(row["glucose_mg_dl"])
            
    X = np.array(X)
    y = np.array(y)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# Phase 3: Regression Model Architecture
# ==========================================
def build_regression_model(input_shape=(150, 150, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        # CRITICAL DIFFERENCE: Linear activation for continuous value prediction
        Dense(1, activation='linear') 
    ])
    
    # Loss is MAE (Mean Absolute Error) for regression, not Binary Crossentropy
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mae', metrics=['mse'])
    return model

# ==========================================
# Phase 4: Clarke Error Grid & Evaluation
# ==========================================
def plot_clarke_error_grid(y_true, y_pred, save_path="clarke_error_grid.png"):
    """
    Plots a simplified Clarke Error Grid to evaluate clinical accuracy of glucose predictions.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.7, color='blue')
    
    # Perfect prediction line
    plt.plot([50, 300], [50, 300], 'k--', label='Perfect Agreement')
    
    # 20% margin (Zone A boundary)
    plt.plot([50, 300], [60, 360], 'g--', label='+20% Margin')
    plt.plot([50, 300], [40, 240], 'g--')
    
    plt.title("Clinical Benchmark: Clarke Error Grid")
    plt.xlabel("Reference Glucose (mg/dL)")
    plt.ylabel("Predicted Glucose (mg/dL)")
    plt.xlim(50, 300)
    plt.ylim(50, 300)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    print(f"Saved Clarke Error Grid to {save_path}")

# ==========================================
# Main Execution Pipeline
# ==========================================
if __name__ == "__main__":
    print("Loading data and mapping continuous glucose values...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples.")
    
    model = build_regression_model()
    
    print("Training Stage 2 Regression Model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=8,
        verbose=1
    )
    
    print("Evaluating Model...")
    y_pred = model.predict(X_test).flatten()
    
    # Calculate Pearson Correlation (Benchmark requirement)
    corr, _ = pearsonr(y_test, y_pred)
    print(f"Pearson Correlation Coefficient: {corr:.3f}")
    
    plot_clarke_error_grid(y_test, y_pred)
    print("Pipeline Execution Complete!")
