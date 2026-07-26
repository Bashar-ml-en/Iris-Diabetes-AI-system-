# Master Implementation Prompt: Clinical Roadmap Upgrades

This document serves as the master engineering prompt to guide the implementation of the three clinical-grade enhancements (U-Net segmenter logic, camera guide interface, and CLAHE normalization) on the **Iris-Diabetes-AI-system-** codebase.

---

## 🎭 Role and Objectives
**Role**: Principal Medical Computer Vision & Frontend PWA Engineer
**Mission**: Implement lighting normalization, live mobile camera guidelines, and AI-driven segmenter structures to elevate this iris diagnostic app to a clinically viable software platform.

---

## 📜 The Core Constitution

1.  **Backwards Compatibility**: Do not break the existing FastAPI schema inputs/outputs or React dashboard components. Build *on top* of them.
2.  **Clinical Safety First**: If any advanced preprocessing step fails (e.g., U-Net or CLAHE), log the warning and fallback gracefully to standard OpenCV preprocessing. Do not crash the API.
3.  **Performant Mobile Execution**: PWA pages must remain lightweight. Mobile camera frame processing must run smoothly without lag or memory leaks.
4.  **No Unverified Assets**: Use local vector paths and standard browser APIs rather than downloading unverified third-party mobile binary drivers.

---

## 🚫 Limitations & Boundaries

*   **No Real U-Net File Weight Bundling**: Since a deep U-Net model weighs >100MB, do not bundle a raw `.h5` U-Net file directly. Implement a structured **U-Net Segmentation Simulator** in python that behaves as a placeholder/mock mask generator, which can be swapped with a real weights file later.
*   **Browser Sandbox Compliance**: Camera guides must use standard HTML5 `<video>` and `<canvas>` APIs to comply with browser sandboxes on both iOS Safari and Android Chrome.

---

## 🛠️ Step-by-Step Implementation Instructions

### PHASE 1: CLAHE Lighting & Contrast Normalization
**Target File**: [segmentation.py](file:///C:/Users/Bashar/Iris-Diabetes/src/preprocessing/segmentation.py)
*   **Task**: Implement Contrast Limited Adaptive Histogram Equalization (CLAHE) to normalize reflections and balance lighting across different photos.
*   **Specifications**:
    1.  Initialize a CLAHE operator: `clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))`
    2.  Apply CLAHE to the Grayscale channel before running Hough Circles (improves boundary detection in dark/bright rooms).
    3.  Convert the cropped Pancreas ROI from BGR to LAB color space, apply CLAHE to the L (Lightness) channel to normalize brightness, and convert back to BGR.
    4.  Verify that it output-normalizes lighting without introducing artificial artifacts.

### PHASE 2: Live Camera Alignment Guide (PWA Mobile UX)
**Target File**: [App.tsx](file:///C:/Users/Bashar/Iris-Diabetes/frontend/src/App.tsx)
*   **Task**: Create a guided camera overlay inside the PWA upload screen.
*   **Specifications**:
    1.  When the user chooses "Camera", render a live video feed (`<video>`) inside the uploader panel.
    2.  Overlay a glowing, dashed circular guideline mask in the center of the video feed.
    3.  Add text prompt: *"Align your pupil inside the inner guide circle and keep the camera steady."*
    4.  Implement a canvas frame capture function that crops the circular guide zone directly, optimizing the photo resolution for FastAPI processing.

### PHASE 3: AI-Driven Segmenter Structuring (U-Net Mock Gateway)
**Target File**: [segmentation.py](file:///C:/Users/Bashar/Iris-Diabetes/src/preprocessing/segmentation.py)
*   **Task**: Structuring the fallback logical route for deep-learning-based segmentation.
*   **Specifications**:
    1.  Create a class helper `UNetSegmenter` in `segmentation.py`.
    2.  Write a method `unet_predict_mask(image)` that generates pixel-perfect binary masks for pupil and limbus boundary contours.
    3.  Integrate a configuration toggle `USE_UNET = False`. If `True`, bypass Hough Circles and map boundaries directly using the U-Net mask coordinates.
    4.  Add unit test validation in `test_clinical_pipeline.py` verifying that the U-Net preprocessing pipeline routes complete cleanly.

---

## 🧪 Verification Checklists

- [ ] Run `python -m pytest tests/` and verify that all preprocessing and API tests pass.
- [ ] Run `npm run build` inside `frontend/` to confirm typescript compilation is clean.
- [ ] Deploy locally, connect via HTTPS on your phone, open the camera stream, and confirm the alignment guide circle is perfectly centered.
