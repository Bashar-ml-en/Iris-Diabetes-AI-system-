import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def detect_blur(image: np.ndarray) -> float:
    """
    Calculates the sharpness of the image using the variance of the Laplacian.
    Higher values mean the image is sharper. Blurry images will have very low values.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    logger.info(f"Laplacian variance (sharpness): {variance:.2f}")
    return float(variance)

def detect_pupil_and_iris(image: np.ndarray) -> tuple[tuple[int, int], int, int]:
    """
    Automatically detects the boundaries of the pupil (inner circle) and iris (outer circle)
    using the OpenCV Hough Circle Transform.
    
    Returns:
        center: (cx, cy) coordinates of the pupil center
        pupil_radius: radius of the pupil in pixels
        iris_radius: radius of the iris in pixels
        
    Fallback:
        If Hough Circles fails to find circles, falls back to a central bounding crop.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Preprocess image to enhance circular structures
    gray_blurred = cv2.medianBlur(gray, 9)
    
    # 1. Detect Pupil (typically dark and central)
    # Threshold to isolate the dark pupil region
    _, thresh = cv2.threshold(gray_blurred, 50, 255, cv2.THRESH_BINARY_INV)
    pupil_circles = cv2.HoughCircles(
        thresh,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=w // 2,
        param1=100,
        param2=10,
        minRadius=int(w * 0.05),
        maxRadius=int(w * 0.25)
    )
    
    pupil_center = (w // 2, h // 2)
    pupil_radius = int(w * 0.1)
    
    if pupil_circles is not None:
        pupil_circles = np.uint16(np.around(pupil_circles))
        best_circle = pupil_circles[0][0]
        pupil_center = (int(best_circle[0]), int(best_circle[1]))
        pupil_radius = int(best_circle[2])
        logger.info(f"Pupil detected at {pupil_center} with radius {pupil_radius}")
    else:
        logger.warning("Hough Circles failed to detect pupil boundary. Using central default.")
        
    # 2. Detect Iris boundary (outer limbus)
    iris_circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=w // 2,
        param1=50,
        param2=30,
        minRadius=int(pupil_radius * 1.5),
        maxRadius=int(w * 0.45)
    )
    
    iris_radius = int(pupil_radius * 3)
    
    if iris_circles is not None:
        iris_circles = np.uint16(np.around(iris_circles))
        # Find circle closest to the pupil center
        min_dist = float('inf')
        for circle in iris_circles[0]:
            cx, cy, r = circle
            dist = np.sqrt((cx - pupil_center[0])**2 + (cy - pupil_center[1])**2)
            if dist < min_dist:
                min_dist = dist
                iris_radius = int(r)
        logger.info(f"Iris detected with radius {iris_radius}")
    else:
        logger.warning("Hough Circles failed to detect iris boundary. Using default radius ratio.")
        
    return pupil_center, pupil_radius, iris_radius

def unwrap_iris(
    image: np.ndarray, 
    center: tuple[int, int], 
    inner_r: int, 
    outer_r: int, 
    width: int = 360, 
    height: int = 60
) -> np.ndarray:
    """
    Simulates Daugman's Rubber Sheet Model by unwarping the circular iris ring
    into a normalized 2D rectangular strip of size (height, width).
    """
    cx, cy = center
    theta = np.linspace(0, 2 * np.pi, width)
    r_val = np.linspace(0, 1, height)
    
    # Meshgrid for polar mapping coordinates
    r_grid, theta_grid = np.meshgrid(r_val, theta)
    
    # Target radii mapping from pupil boundary to outer limbus boundary
    R = inner_r + r_grid * (outer_r - inner_r)
    
    # Convert polar to Cartesian mapping coordinates relative to the detected center
    map_x = cx + R * np.cos(theta_grid)
    map_y = cy + R * np.sin(theta_grid)
    
    # Transpose map matrices to match height x width output shape
    map_x = map_x.T.astype(np.float32)
    map_y = map_y.T.astype(np.float32)
    
    # Remap circular structures using bilinear interpolation
    unwrapped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    return unwrapped

def extract_pancreas_roi(unwrapped_strip: np.ndarray) -> np.ndarray:
    """
    Extracts the specific sector of the unwrapped iris corresponding to the
    pancreas zone in Iridology maps. 
    Typically, the pancreas zone is mapped to the lower right sector 
    of the iris (approx. 270 to 315 degrees, or the last quarter of the flat strip).
    
    Returns a resized (150, 150) normalized ROI.
    """
    h, w = unwrapped_strip.shape[:2]
    # Extract the last 15% to 25% of the horizontal strip (approx 270-324 degrees)
    start_col = int(w * 0.75)
    end_col = int(w * 0.90)
    
    # Crop the pancreas sector
    pancreas_crop = unwrapped_strip[0:h, start_col:end_col]
    
    # Resize to match TF Stage 2 Input dimension requirements (150, 150)
    resized_roi = cv2.resize(pancreas_crop, (150, 150))
    return resized_roi

def preprocess_pipeline(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Executes the full preprocessing pipeline:
    1. Sharpness blur check
    2. Pupil & Iris localization
    3. Unwrapping (Daugman's model)
    4. Pancreas ROI extraction
    
    Returns:
        unwrapped_strip: the unwarped full 2D iris strip
        pancreas_roi: the extracted and resized pancreas sector image
        sharpness: Laplacian variance score
    """
    sharpness = detect_blur(image)
    center, pupil_r, iris_r = detect_pupil_and_iris(image)
    unwrapped = unwrap_iris(image, center, pupil_r, iris_r)
    pancreas_roi = extract_pancreas_roi(unwrapped)
    return unwrapped, pancreas_roi, sharpness
