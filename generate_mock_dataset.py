import os
import csv
import random
import cv2
import numpy as np

def generate_mock_dataset(base_dir="synthetic_iris_dataset", num_control=20, num_diabetes=20):
    """
    Generates a synthetic dataset of mock iris images and a CSV with continuous glucose values.
    Control: Glucose levels randomly assigned between 70.0 and 125.0 mg/dL.
    Diabetes: Glucose levels randomly assigned between 126.0 and 250.0 mg/dL.
    """
    os.makedirs(base_dir, exist_ok=True)
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    csv_path = os.path.join(base_dir, "glucose_labels.csv")
    
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_filename", "diagnosis", "glucose_mg_dl"])
        
        # Generate Control
        for i in range(num_control):
            filename = f"control_{i:03d}.png"
            glucose = round(random.uniform(70.0, 125.0), 1)
            create_dummy_image(os.path.join(images_dir, filename), diagnosis="Control")
            writer.writerow([filename, "Control", glucose])
            
        # Generate Diabetes
        for i in range(num_diabetes):
            filename = f"diabetes_{i:03d}.png"
            glucose = round(random.uniform(126.0, 250.0), 1)
            create_dummy_image(os.path.join(images_dir, filename), diagnosis="Diabetes")
            writer.writerow([filename, "Diabetes", glucose])
            
    print(f"Synthetic dataset generated at: {base_dir}")
    print(f"Total images: {num_control + num_diabetes}")
    print(f"Labels CSV: {csv_path}")

def create_dummy_image(filepath, size=(300, 300), diagnosis="Control"):
    """Creates a simple dummy image with a circle resembling an iris/pupil."""
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 200  # Gray background
    center = (size[1]//2, size[0]//2)
    
    if diagnosis == "Diabetes":
        # Add a slight red tint to diabetic irises so the CNN can distinguish the classes
        cv2.circle(img, center, 100, (100, 50, 150), -1) 
    else:
        # Normal iris color
        cv2.circle(img, center, 100, (100, 50, 50), -1) 
        
    cv2.circle(img, center, 30, (20, 20, 20), -1)   # Pupil
    cv2.imwrite(filepath, img)

if __name__ == "__main__":
    generate_mock_dataset()
