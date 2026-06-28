import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def load_data_for_both_stages(base_dir="synthetic_iris_dataset"):
    csv_path = os.path.join(base_dir, "glucose_labels.csv")
    images_dir = os.path.join(base_dir, "images")
    
    df = pd.read_csv(csv_path)
    
    X_full = []
    X_roi = []
    y_class = []
    y_reg = []
    
    for index, row in df.iterrows():
        img_path = os.path.join(images_dir, row["image_filename"])
        img = cv2.imread(img_path)
        if img is not None:
            # For Stage 1: Full image resized to 150x150
            full_img = cv2.resize(img, (150, 150)).astype('float32') / 255.0
            X_full.append(full_img)
            
            # For Stage 2: ROI (Topographical Extraction)
            h, w = img.shape[:2]
            sector = img[h//2:h, w//2:w]
            roi = cv2.resize(sector, (150, 150)).astype('float32') / 255.0
            X_roi.append(roi)
            
            # Labels
            y_class.append(0 if row["diagnosis"] == "Control" else 1)
            y_reg.append(row["glucose_mg_dl"])
            
    return np.array(X_full), np.array(X_roi), np.array(y_class), np.array(y_reg)

def train_and_save_stage1(X, y):
    print("\n--- Training Stage 1 Classifier ---")
    model = Sequential([
        Input(shape=(150, 150, 3)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train for a few epochs to overfit on the synthetic data and guarantee high confidence
    model.fit(X, y, epochs=15, batch_size=8, verbose=1)
    
    model.save('stage1_diabetes_classifier.h5')
    print("Saved -> stage1_diabetes_classifier.h5")

def train_and_save_stage2(X, y):
    print("\n--- Training Stage 2 Regressor ---")
    model = Sequential([
        Input(shape=(150, 150, 3)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mae', metrics=['mse'])
    model.fit(X, y, epochs=20, batch_size=8, verbose=1)
    
    model.save('stage2_glucose_regressor.h5')
    print("Saved -> stage2_glucose_regressor.h5")

if __name__ == "__main__":
    print("Loading synthetic dataset...")
    X_full, X_roi, y_class, y_reg = load_data_for_both_stages()
    
    if len(X_full) == 0:
        print("Error: No data found. Please run generate_mock_dataset.py first.")
    else:
        train_and_save_stage1(X_full, y_class)
        train_and_save_stage2(X_roi, y_reg)
        print("\nAll models trained and saved successfully! The UI is now fully operational.")
