import cv2
import numpy as np
import pytesseract
import logging
from pytesseract.pytesseract import TesseractNotFoundError

logger = logging.getLogger(__name__)

def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Preprocess image and extract text using Tesseract OCR.
    """
    # 1. Convert uploaded image bytes to a numpy array, then to a cv2 image.
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Ensure it is a valid image file.")

    # 1.5 Optimize performance: Resize image if it's too large (saves CPU time)
    MAX_DIM = 1500
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 2. Preprocessing
    # Convert the image to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Adaptive Thresholding to binarize the image
    processed_img = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # 3. OCR Inference
    try:
        # --oem 1 forces the LSTM engine (faster/more accurate), --psm 3 is default automatic page segmentation
        custom_config = r'--oem 1 --psm 3'
        text = pytesseract.image_to_string(processed_img, lang='eng+ara', config=custom_config)
    except TesseractNotFoundError:
        logger.error("Tesseract is not installed or not in PATH.")
        raise RuntimeError("OCR Engine (Tesseract) is not installed on the server.")
        
    # Clean up the output text (remove excessive newlines, strip whitespaces)
    cleaned_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return cleaned_text
