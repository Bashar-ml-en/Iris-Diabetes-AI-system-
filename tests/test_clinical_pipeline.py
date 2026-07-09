import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Import system under test
from src.serving.api import app
from src.preprocessing.segmentation import detect_blur, detect_pupil_and_iris, unwrap_iris

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

# Helper to create mock images in memory
def create_mock_image_bytes(blur: bool = False, diagnosis: str = "Control") -> bytes:
    """Generates a dummy png image and returns its raw bytes."""
    size = (300, 300)
    
    if blur:
        # Blurry image: completely smooth single color
        img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 128
    else:
        # Sharp image: high contrast edges (pupil + iris boundaries)
        img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 200
        center = (size[1] // 2, size[0] // 2)
        if diagnosis == "Diabetes":
            cv2.circle(img, center, 100, (100, 50, 150), -1)  # Diabetic iris tint
        else:
            cv2.circle(img, center, 100, (100, 50, 50), -1)   # Control iris tint
        cv2.circle(img, center, 30, (20, 20, 20), -1)       # Pupil
        
        # Add high frequency texture to ensure high Laplacian variance
        for r in range(40, 90, 5):
            cv2.circle(img, center, r, (255, 255, 255), 1)

    # Encode to png bytes
    _, buffer = cv2.imencode('.png', img)
    return buffer.tobytes()

# ----------------------------------------------------
# 1. Unit Tests for OpenCV Preprocessing
# ----------------------------------------------------
def test_blur_detection_sharpness():
    """Verify that detect_blur returns low values for smooth and high values for textured images."""
    blurry_bytes = create_mock_image_bytes(blur=True)
    sharp_bytes = create_mock_image_bytes(blur=False)
    
    # Convert back to cv2 images
    blurry_img = cv2.imdecode(np.frombuffer(blurry_bytes, np.uint8), cv2.IMREAD_COLOR)
    sharp_img = cv2.imdecode(np.frombuffer(sharp_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    assert detect_blur(blurry_img) < 10.0
    assert detect_blur(sharp_img) > 100.0

def test_unwrap_dimensions():
    """Verify that unwrap_iris maps circular region to target rectangular shape."""
    img = cv2.imdecode(np.frombuffer(create_mock_image_bytes(blur=False), np.uint8), cv2.IMREAD_COLOR)
    center = (150, 150)
    pupil_r = 30
    iris_r = 100
    
    unwrapped = unwrap_iris(img, center, pupil_r, iris_r, width=360, height=60)
    assert unwrapped.shape == (60, 360, 3)

# ----------------------------------------------------
# 2. Integration Tests for API Serving
# ----------------------------------------------------
def test_health_endpoint(client):
    """Verify health endpoint successfully returns active status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "models_loaded" in data

def test_predict_blurry_rejection(client):
    """Verify that API rejects blurry images with a 400 Bad Request."""
    blurry_bytes = create_mock_image_bytes(blur=True)
    
    files = {"file": ("blurry.png", blurry_bytes, "image/png")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 400
    assert "too low" in response.json()["detail"]

def test_predict_cascaded_flow_control(client):
    """Verify inference pipeline completes successfully for control patient."""
    sharp_bytes = create_mock_image_bytes(blur=False, diagnosis="Control")
    
    files = {"file": ("control.png", sharp_bytes, "image/png")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["sharpness"] > 100.0
    assert "diabetes_probability" in data
    
    # Depending on model weights, assert cascading logic behavior:
    if data["diagnosis"] == "Control":
        assert data["estimated_glucose"] is None
        assert "no iridological markers" in data["medical_advice"].lower()
    else:
        assert data["estimated_glucose"] is not None
        assert data["estimated_glucose"] > 0
        assert "risk" in data["medical_advice"].lower()

    # Visualizer assertions
    assert data["unwrapped_strip_b64"].startswith("data:image/png;base64,")
    assert data["pancreas_roi_b64"].startswith("data:image/png;base64,")
    assert data["latency_ms"] > 0
