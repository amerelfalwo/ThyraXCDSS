"""
Ultrasound Image Processing Pipeline (Node 4).

Handles the full segmentation → classification workflow:
  1. Decode and preprocess the uploaded image.
  2. Run ONNX segmentation to produce a binary mask.
  3. Extract the region of interest (ROI) via bounding-box crop.
  4. Run ONNX classification on the ROI.
  5. Map output to clinically accurate risk assessment.
  6. Save result images (mask, overlay, ROI) to the media directory.

Clinical Standards:
  - Classification labels: "benign" / "suspicious" (NOT "malignant")
  - Risk stratification mapped to ACR TI-RADS analogues (TR2–TR5)
  - TR1 is reserved for "no nodule detected" (handled separately)
  - Recommendations follow ACR TI-RADS evidence-based guidelines

Note on Segmentation:
  The segmentation mask is used for ROI extraction (bounding-box crop)
  only. No radiomic feature analysis (margin regularity, echogenicity,
  calcification, shape) is extracted from the mask at this stage.
"""

import gc
import uuid
import logging

import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ═══════════════════════════════════════════════════════════════
# Radiomics & Risk Assessment — ACR TI-RADS Compliant Mapping
# ═══════════════════════════════════════════════════════════════

def extract_radiomic_features(mask: np.ndarray, img_gray: np.ndarray, bbox: list) -> dict:
    """
    Extracts radiomic features from the segmentation mask to estimate TI-RADS points.
    1. Shape: Taller-than-wide.
    2. Margin: Irregularity (solidity / circularity).
    3. Echogenicity: Hypoechoic vs Isoechoic.
    """
    features = {}
    x_min, y_min, x_max, y_max = bbox
    w = x_max - x_min
    h = y_max - y_min
    
    # 1. Shape
    taller_than_wide = bool(h > w)
    features["taller_than_wide"] = taller_than_wide
    
    # 2. Margin
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        
        solidity = area / hull_area if hull_area > 0 else 1.0
        circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 1.0
        
        features["solidity"] = round(float(solidity), 3)
        features["circularity"] = round(float(circularity), 3)
        features["irregular_margin"] = bool(solidity < 0.85 or circularity < 0.6)
    else:
        features["solidity"] = 1.0
        features["circularity"] = 1.0
        features["irregular_margin"] = False

    # 3. Echogenicity
    roi_mask = mask[y_min:y_max+1, x_min:x_max+1]
    roi_img = img_gray[y_min:y_max+1, x_min:x_max+1]
    
    nodule_mean = np.mean(roi_img[roi_mask > 0]) if np.any(roi_mask > 0) else 0.0
    
    h_img, w_img = img_gray.shape
    exp_x_min = max(0, x_min - 20)
    exp_y_min = max(0, y_min - 20)
    exp_x_max = min(w_img, x_max + 20)
    exp_y_max = min(h_img, y_max + 20)
    
    bg_mask = np.ones_like(img_gray[exp_y_min:exp_y_max, exp_x_min:exp_x_max], dtype=bool)
    local_mask = mask[exp_y_min:exp_y_max, exp_x_min:exp_x_max]
    bg_mask[local_mask > 0] = False
    
    tissue_mean = np.mean(img_gray[exp_y_min:exp_y_max, exp_x_min:exp_x_max][bg_mask]) if np.any(bg_mask) else 0.0
    
    features["nodule_intensity"] = round(float(nodule_mean), 2)
    features["tissue_intensity"] = round(float(tissue_mean), 2)
    features["hypoechoic"] = bool(nodule_mean < tissue_mean * 0.8)
    features["markedly_hypoechoic"] = bool(nodule_mean < tissue_mean * 0.5)

    return features


def assess_risk_level(class_idx: int, confidence: float, features: dict = None) -> dict:
    """
    Map radiomic features and classification output to a points-based TI-RADS score.

    Args:
        class_idx: Predicted class (0 = benign, 1 = suspicious).
        confidence: Model confidence (0.0–1.0 scale).
        features: Extracted radiomic features dict.

    Returns:
        Dict with risk_level, acr_tirads_level, and clinical_recommendation.
    """
    if features is None:
        features = {}

    points = 0
    
    # 1. Shape
    if features.get("taller_than_wide", False):
        points += 3
        
    # 2. Margin
    if features.get("irregular_margin", False):
        points += 2
        
    # 3. Echogenicity
    if features.get("markedly_hypoechoic", False):
        points += 3
    elif features.get("hypoechoic", False):
        points += 2
    else:
        points += 1 # Isoechoic
        
    # 4. Neural Network Suspicion (acts as composition / calcifications proxy)
    if class_idx == 1:
        if confidence >= 0.85:
            points += 4  # Very suspicious -> Solid + Calcifications
        elif confidence >= 0.65:
            points += 2  # Moderately suspicious
        else:
            points += 1
            
    # Map points to TI-RADS
    # TR1: 0 points, TR2: 2 points, TR3: 3 points, TR4: 4-6 points, TR5: >=7 points
    if points <= 2:
        acr_tirads = "TR2"
        risk_level = "Benign / Very Low Suspicion"
        rec = "Imaging findings are consistent with a benign-appearing nodule. Follow-up ultrasound in 12–24 months."
        next_step = "Routine follow-up ultrasound in 12-24 months."
    elif points == 3:
        acr_tirads = "TR3"
        risk_level = "Mildly Suspicious"
        rec = "Imaging findings suggest a mildly suspicious nodule. Consider FNA if nodule is >= 2.5 cm."
        next_step = "Consider FNA if nodule >= 2.5 cm; otherwise 12 month follow-up."
    elif 4 <= points <= 6:
        acr_tirads = "TR4"
        risk_level = "Moderately Suspicious"
        rec = "Imaging findings are moderately suspicious. FNA biopsy is recommended for nodules >= 1.5 cm."
        next_step = "Perform FNA if nodule >= 1.5 cm; otherwise, follow-up ultrasound in 6-12 months."
    else:
        acr_tirads = "TR5"
        risk_level = "Highly Suspicious"
        rec = "Imaging findings are highly suspicious for malignancy. FNA biopsy strongly recommended for nodules >= 1.0 cm."
        next_step = "Immediate referral for FNA biopsy (if nodule >= 1.0 cm) and endocrinology evaluation."

    return {
        "risk_level": risk_level,
        "acr_tirads_level": acr_tirads,
        "clinical_recommendation": rec,
        "next_step": next_step,
        "points": points,
    }


def perform_segmentation(img_color: np.ndarray, threshold: float = 0.6):
    """
    Phase 1 & 2: Runs ONNX segmentation and overlay generation.
    Returns:
        tuple: (mask_full, bbox, roi, blended)
        If no nodule detected, returns None.
    """
    from app.core.models import load_segmentation_model
    
    orig_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    img_seg_in = cv2.resize(img_gray, (256, 256)).astype(np.float32) / 255.0
    img_seg_in = np.expand_dims(img_seg_in, axis=(0, -1))

    seg_session = load_segmentation_model()
    seg_input_name = seg_session.get_inputs()[0].name
    mask_pred = seg_session.run(None, {seg_input_name: img_seg_in})[0][0, :, :, 0]
    
    logger.debug("Segmentation model kept in memory cache")

    mask = (mask_pred > threshold).astype(np.uint8)
    mask_full = cv2.resize(
        mask, (orig_rgb.shape[1], orig_rgb.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    ys, xs = np.where(mask_full > 0)
    if len(xs) < 50:
        return None

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bbox = [x_min, y_min, x_max, y_max]
    roi = orig_rgb[y_min : y_max + 1, x_min : x_max + 1]
    
    overlay = orig_rgb.copy()
    overlay[mask_full > 0] = [0, 255, 0]
    blended = cv2.addWeighted(orig_rgb, 0.7, overlay, 0.3, 0)
    
    return mask_full, bbox, roi, blended


def perform_classification(roi: np.ndarray):
    """
    Phase 3: Runs ONNX classification on the extracted ROI.
    Returns:
        tuple: (class_idx, confidence, raw_logit)
    """
    from app.core.models import load_classification_model
    
    # medical_final model expects (380, 380) in NCHW format.
    roi_cls_in = cv2.resize(roi, (380, 380)).astype(np.float32) / 255.0
    
    # ImageNet Mean/Std normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    roi_cls_in = (roi_cls_in - mean) / std
    
    # Transpose to (Batch, Channel, Height, Width) - NCHW format
    roi_cls_in = np.transpose(roi_cls_in, (2, 0, 1))
    roi_cls_in = np.expand_dims(roi_cls_in, axis=0)

    cls_session = load_classification_model()
    cls_input_name = cls_session.get_inputs()[0].name
    cls_pred = cls_session.run(None, {cls_input_name: roi_cls_in})[0][0]

    logger.debug("Classification model kept in memory cache")

    # Robust output handling: works with 1-output (logits) or 2-output (softmax)
    if len(cls_pred) > 1:
        # Multi-class or 2-class Softmax
        exp_vals = np.exp(cls_pred - np.max(cls_pred))
        probs = exp_vals / exp_vals.sum()
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        raw_logit = None
    else:
        # Single output Logit -> Apply Sigmoid
        logit = float(cls_pred[0])
        prob = 1.0 / (1.0 + np.exp(-logit))
        class_idx = 1 if prob > 0.5 else 0
        confidence = prob if class_idx == 1 else 1.0 - prob
        raw_logit = logit

    return class_idx, confidence, raw_logit


def process_full_pipeline(
    image_bytes: bytes,
    img_hash: str,
    threshold: float = 0.6,
) -> dict:
    """
    Process an ultrasound image through the full segmentation + classification pipeline.

    This function is CPU-bound and MUST be called inside
    ``run_in_threadpool`` from async endpoints (brandguideline §5).

    Memory Management:
      - Models are cached in memory via @functools.lru_cache after the first load.

    Args:
        image_bytes: Raw bytes of the uploaded ultrasound image.
        img_hash: Full SHA256 hash of the image.
        threshold: Binarisation threshold for the segmentation mask.

    Returns:
        Dict matching the ``ImagePredictionResponse`` schema with keys:
        status, bbox, classification, segmentation, images, and medical_disclaimer.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # ────────────────────────────────────────────────────────────
    # Phase 1 & 2: Segmentation and Overlay
    # ────────────────────────────────────────────────────────────
    seg_result = perform_segmentation(img_color, threshold)
    
    if seg_result is None:
        del img_color, nparr
        gc.collect()
        return {
            "status": "success",
            "classification": "No nodule detected in the provided ultrasound image.",
            "confidence": None,
            "message": (
                "The segmentation model could not identify any thyroid nodules "
                "or relevant regions of interest. This corresponds to ACR TI-RADS "
                "TR1 (Benign — no nodule). If clinical suspicion remains, consider "
                "re-imaging or referral."
            ),
            "bbox": None,
        }

    mask_full, bbox, roi, blended = seg_result

    # ────────────────────────────────────────────────────────────
    # Phase 3: Classification
    # ────────────────────────────────────────────────────────────
    class_idx, confidence, raw_logit = perform_classification(roi)

    unique_id = str(uuid.uuid4())

    # ────────────────────────────────────────────────────────────
    # Phase 4: Prepare result images (in-memory bytes)
    # ────────────────────────────────────────────────────────────
    # Convert mask_full to BGR so it encodes cleanly as color
    mask_bgr = cv2.cvtColor(mask_full * 255, cv2.COLOR_GRAY2BGR)
    _, mask_enc = cv2.imencode(".png", mask_bgr)
    
    overlay_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    _, overlay_enc = cv2.imencode(".png", overlay_bgr)
    
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    _, roi_enc = cv2.imencode(".png", roi_bgr)

    # ────────────────────────────────────────────────────────────
    # Phase 5: Clinical risk assessment & Radiomics
    # ────────────────────────────────────────────────────────────
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    features = extract_radiomic_features(mask_full, img_gray, bbox)
    risk = assess_risk_level(class_idx, confidence, features)
    needs_review = confidence < 0.65

    # ── Memory Management for heavy arrays ──
    del img_color, img_gray, nparr
    del mask_full, roi, blended, mask_bgr, overlay_bgr, roi_bgr
    gc.collect()

    return {
        "status": "success",
        "bbox": bbox,
        "input_md5": img_hash,
        "classification": {
            "prediction": class_idx,
            "label": "suspicious" if class_idx == 1 else "benign",
            "confidence_pct": round(confidence * 100, 2),
            "raw_logit": round(raw_logit, 4) if raw_logit is not None else None,
            "risk_level": risk["risk_level"],
            "acr_tirads_level": risk["acr_tirads_level"],
            "clinical_recommendation": risk["clinical_recommendation"],
            "next_step": risk["next_step"],
            "needs_manual_review": needs_review,
            "radiomic_features": features,
        },
        "segmentation": {
            "method": "U-Net ONNX",
            "roi_extraction": "bounding_box_crop",
        },
        "images": {
            "mask_bytes": mask_enc.tobytes(),
            "overlay_bytes": overlay_enc.tobytes(),
            "roi_bytes": roi_enc.tobytes(),
            "unique_id": unique_id,
        },
    }