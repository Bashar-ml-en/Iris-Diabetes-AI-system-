# AI Iridology: Non-Invasive Diabetes Screening Pipeline

This repository contains the complete software architecture for an end-to-end, two-stage clinical AI pipeline designed to screen for diabetes and estimate continuous blood glucose levels non-invasively using high-resolution images of the human iris. 

This project successfully replicates the architectural concepts of state-of-the-art non-invasive glucose monitoring systems (such as EasyGlucose) by leveraging Topographical Iridology markers and Deep Convolutional Neural Networks (CNNs).

## 🏗️ System Architecture

The system is built as a **Two-Stage Cascaded Pipeline** to maximize clinical safety and computational efficiency.

### Stage 1: Binary Classification (Diabetes Detection)
* **Goal:** Determine if the patient has diabetes or is a healthy control.
* **Input:** A full 150x150 RGB image of the patient's eye.
* **Model:** Deep CNN with a `Sigmoid` output activation.
* **Logic:** If the model predicts "Control" (Confidence < 0.5), the pipeline halts and recommends standard yearly checkups. If it predicts "Diabetic", the image is automatically passed to Stage 2.

### Stage 2: Continuous Glucose Regression (EasyGlucose Replication)
* **Goal:** Estimate the exact current blood sugar level (mg/dL).
* **Topographical ROI Extraction:** Simulating Daugman's Rubber Sheet Model, the system automatically crops and isolates the specific sector of the iris corresponding to the pancreas in Iridology charts.
* **Model:** Deep CNN with a `Linear` output activation.
* **Loss Function:** Trained using **Mean Absolute Error (MAE)** to penalize large clinical deviations, optimizing for continuous scalar output rather than categorical buckets.
* **Evaluation:** Clinical accuracy is measured using **Pearson Correlation** and plotted against a standard **Clarke Error Grid** (measuring predictions falling into safe Zones A and B).

## 🚀 How to Run the System

The entire pipeline is wrapped in a beautiful, interactive web interface using Streamlit.

**1. Install Requirements:**
```bash
pip install tensorflow opencv-python pandas numpy streamlit
```

**2. Launch the Application:**
```bash
streamlit run streamlit_app.py
```
This will launch a local web server. You can upload an image of an iris, and the system will pass it through the cascading pipeline, outputting the final clinical diagnosis and medical advice.

## 🧬 Handling Dataset Limitations (Synthetic Data Generation)
Currently, a public dataset containing paired images of human irises and their exact, simultaneous continuous glucose monitor (CGM) readings does not exist. 

To overcome this and finalize the software architecture:
1. We built a **Synthetic Data Generator** (`generate_mock_dataset.py`) to create mock images and assign realistic continuous glucose values.
2. We trained the Stage 2 Regression model on this synthetic data (`train_mock_models.py`) to validate the mathematical pipeline, loss functions, and tensor shapes.
3. The software is **100% structurally complete**. When real clinical data is acquired in the future, it can be seamlessly swapped into the `synthetic_iris_dataset` folder without changing a single line of the pipeline architecture.
