import os
from functools import lru_cache
import numpy as np
from PIL import Image, ImageStat

# Constants
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_FORMATS = {"PNG", "JPEG"}
XRAY_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "xray_classifier.h5"))

@lru_cache(maxsize=1)
def _get_xray_model():
    """
    Load the X-ray classifier model.
    """
    if not os.path.exists(XRAY_MODEL_PATH):
        return None  
    
    try:
        from tensorflow.keras.models import load_model
        return load_model(XRAY_MODEL_PATH)
    except Exception:
        return None


def _is_grayscale_aesthetic(image):
    
    if image.mode == 'L':
        return True
    
    # Check RGB variance
    stat = ImageStat.Stat(image)
    if len(stat.sum) >= 3:
        r, g, b = stat.mean[:3]
        if abs(r - g) < 15 and abs(g - b) < 15 and abs(r - b) < 15:
            return True
    return False


def _heuristic_check(image):
    # Convert to grayscale for consistent analysis
    img_l = image.convert("L")
    stat = ImageStat.Stat(img_l)
    
    mean_val = stat.mean[0]
    std_val = stat.stddev[0]

    if std_val < 20:
        return False, "Low dynamic range (possibly a screenshot or plain color)."
    
    if mean_val > 220 or mean_val < 30:
        return False, "Intensity out of range for typical Chest X-ray scans."
        
    return True, "Passed statistical heuristic."


def is_xray_ai(image):
    
    model = _get_xray_model()
    if model is None:
        return None, "AI validator missing (xray_classifier.h5 not found)."

    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    batch = np.expand_dims(arr, axis=0)
    
    prediction = model.predict(batch, verbose=0)
    score = float(np.array(prediction).reshape(-1)[0])
    
    if score > 0.5:
        return True, f"AI confirmed X-ray scan (Score: {score:.2f})"
    return False, f"AI rejected: likely not a medical scan (Score: {score:.2f})"


def validate_image_upload(uploaded_file):
    """
    Multi-stage validation: File Metadata -> Heuristic -> AI Layer
    """
    try:
        file_name = getattr(uploaded_file, "name", "")
        ext = os.path.splitext(file_name)[1].lower()
        if ext and ext not in ALLOWED_EXTENSIONS:
            return False, "Unsupported file extension. Please use PNG, JPG or JPEG."

        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        if img.format and img.format.upper() not in ALLOWED_FORMATS:
            return False, "Unsupported image format (MIME mismatch)."

        if not _is_grayscale_aesthetic(img):
            return False, "Color detected. Chest X-rays should be grayscale clinical images."
        
        ok, h_msg = _heuristic_check(img)
        if not ok:
            return False, h_msg

        is_valid_ai, ai_msg = is_xray_ai(img)
        
        if is_valid_ai is True:
            return True, ai_msg
        elif is_valid_ai is False:
            return False, ai_msg
        else:
            return True, f"Heuristic valid (AI layer inactive: {ai_msg})"
            
    except Exception as e:
        return False, f"Validation failure: {str(e)}"

