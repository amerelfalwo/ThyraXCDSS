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


def save_image_to_media(image_np: np.ndarray, filename: str, base_url: str) -> str:
    """
    Save a NumPy image array to the media directory and return its public URL.

    Args:
        image_np: Image as a NumPy array (BGR format).
        filename: Target filename (e.g. ``mask_<uuid>.png``).
        base_url: Base URL of the running API server.

    Returns:
        Fully qualified public URL to the saved image.
    """
    media_dir = BASE_DIR / "media"
    media_dir.mkdir(exist_ok=True)

    filepath = media_dir / filename
    cv2.imwrite(str(filepath), image_np)

    return f"{base_url}media/{filename}"


# ═══════════════════════════════════════════════════════════════
# Risk Assessment — ACR TI-RADS Compliant Mapping
# ═══════════════════════════════════════════════════════════════
#
# The binary classifier outputs: 0 = benign, 1 = suspicious.
# We map the (class, confidence) pair to a clinically appropriate
# risk level and an ACR TI-RADS analogue.
#
# IMPORTANT: True ACR TI-RADS scoring requires a radiologist to
# evaluate 5 ultrasonographic feature categories:
#   1. Composition (cystic / mixed / solid)
#   2. Echogenicity (anechoic / hyper / iso / hypoechoic)
#   3. Shape (wider-than-tall / taller-than-wide)
#   4. Margin (smooth / ill-defined / lobulated / irregular / extrathyroidal)
#   5. Echogenic foci (none / comet-tail / macrocalcifications / rim / punctate)
#
# Since our binary classifier does NOT evaluate these features
# individually, the TI-RADS mapping is an APPROXIMATION based on
# the model's confidence level — not a formal ACR TI-RADS score.
# ═══════════════════════════════════════════════════════════════

def assess_risk_level(class_idx: int, confidence: float) -> dict:
    """
    Map binary classification output to clinically appropriate risk assessment.

    Args:
        class_idx: Predicted class (0 = benign, 1 = suspicious).
        confidence: Model confidence (0.0–1.0 scale).

    Returns:
        Dict with risk_level, acr_tirads_level, and clinical_recommendation.
    """
    if class_idx == 0:
        # ── Benign predictions ──
        if confidence >= 0.90:
            return {
                "risk_level": "Very Low Suspicion",
                "acr_tirads_level": "TR2 (AI-estimated)",
                "clinical_recommendation": (
                    "Imaging findings are consistent with a benign-appearing nodule. "
                    "No Fine Needle Aspiration (FNA) is recommended based on imaging alone. "
                    "Follow-up ultrasound in 12–24 months if clinically indicated."
                ),
            }
        elif confidence >= 0.70:
            return {
                "risk_level": "Low Suspicion",
                "acr_tirads_level": "TR3 (AI-estimated)",
                "clinical_recommendation": (
                    "Imaging findings suggest a probably benign nodule. "
                    "Consider follow-up ultrasound in 12 months. "
                    "FNA may be considered if nodule is ≥ 2.5 cm per ACR TI-RADS guidelines."
                ),
            }
        else:
            return {
                "risk_level": "Indeterminate",
                "acr_tirads_level": "TR3 (AI-estimated)",
                "clinical_recommendation": (
                    "AI confidence is below threshold for a reliable benign classification. "
                    "Clinical and sonographic correlation is required. "
                    "Consider FNA if nodule is ≥ 2.5 cm, or follow-up ultrasound in 6–12 months."
                ),
            }
    else:
        # ── Suspicious predictions ──
        if confidence >= 0.85:
            return {
                "risk_level": "Very High Suspicion",
                "acr_tirads_level": "TR5 (AI-estimated)",
                "clinical_recommendation": (
                    "Imaging findings are highly suspicious for malignancy. "
                    "Fine Needle Aspiration (FNA) biopsy is strongly recommended "
                    "for nodules ≥ 1.0 cm per ACR TI-RADS TR5 guidelines. "
                    "Refer to endocrinology/thyroid surgery for evaluation."
                ),
            }
        elif confidence >= 0.70:
            return {
                "risk_level": "High Suspicion",
                "acr_tirads_level": "TR4 (AI-estimated)",
                "clinical_recommendation": (
                    "Imaging findings are suspicious. "
                    "FNA biopsy is recommended for nodules ≥ 1.0 cm. "
                    "If nodule is < 1.0 cm, consider follow-up ultrasound in 6 months. "
                    "Clinical correlation with patient history is advised."
                ),
            }
        else:
            return {
                "risk_level": "Intermediate Suspicion",
                "acr_tirads_level": "TR4 (AI-estimated)",
                "clinical_recommendation": (
                    "AI confidence is moderate for suspicious classification. "
                    "FNA biopsy may be considered for nodules ≥ 1.5 cm. "
                    "Follow-up ultrasound in 6–12 months is recommended. "
                    "Clinical and sonographic correlation is required."
                ),
            }


def process_full_pipeline(
    image_bytes: bytes,
    base_url: str = "http://localhost:8000/",
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
        base_url: Base URL of the running API server (for media URLs).
        threshold: Binarisation threshold for the segmentation mask.

    Returns:
        Dict matching the ``ImagePredictionResponse`` schema with keys:
        status, bbox, classification, segmentation, images, and medical_disclaimer.
    """
    # ── Lazy import: ONNX model loaders ──
    from app.core.models import load_segmentation_model, load_classification_model

    nparr = np.frombuffer(image_bytes, np.uint8)
    img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    orig_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # ────────────────────────────────────────────────────────────
    # Phase 1: Segmentation
    # ────────────────────────────────────────────────────────────
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

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bbox = [x_min, y_min, x_max, y_max]
    roi = orig_rgb[y_min : y_max + 1, x_min : x_max + 1]

    # ────────────────────────────────────────────────────────────
    # Phase 2: Overlay generation
    # ────────────────────────────────────────────────────────────
    overlay = orig_rgb.copy()
    overlay[mask_full > 0] = [0, 255, 0]
    blended = cv2.addWeighted(orig_rgb, 0.7, overlay, 0.3, 0)

    # ────────────────────────────────────────────────────────────
    # Phase 3: Classification
    # ────────────────────────────────────────────────────────────
    roi_cls_in = cv2.resize(roi, (224, 224)).astype(np.float32) / 255.0
    roi_cls_in = np.expand_dims(roi_cls_in, axis=0)

    cls_session = load_classification_model()
    cls_input_name = cls_session.get_inputs()[0].name
    cls_pred = cls_session.run(None, {cls_input_name: roi_cls_in})[0][0]

    logger.debug("Classification model kept in memory cache")

    prob = float(cls_pred[0])
    class_idx = 1 if prob > 0.5 else 0
    confidence = prob if class_idx == 1 else 1 - prob

    unique_id = str(uuid.uuid4())

    # ────────────────────────────────────────────────────────────
    # Phase 4: Save result images
    # ────────────────────────────────────────────────────────────
    mask_url = save_image_to_media(mask_full * 255, f"mask_{unique_id}.png", base_url)
    overlay_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    overlay_url = save_image_to_media(overlay_bgr, f"overlay_{unique_id}.png", base_url)
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    roi_url = save_image_to_media(roi_bgr, f"roi_{unique_id}.png", base_url)

    # ────────────────────────────────────────────────────────────
    # Phase 5: Clinical risk assessment
    # ────────────────────────────────────────────────────────────
    risk = assess_risk_level(class_idx, confidence)
    needs_review = confidence < 0.65

    return {
        "status": "success",
        "bbox": bbox,
        "classification": {
            "prediction": class_idx,
            "label": "suspicious" if class_idx == 1 else "benign",
            "confidence_pct": round(confidence * 100, 2),
            "risk_level": risk["risk_level"],
            "acr_tirads_level": risk["acr_tirads_level"],
            "clinical_recommendation": risk["clinical_recommendation"],
            "needs_manual_review": needs_review,
        },
        "segmentation": {
            "method": "U-Net ONNX",
            "roi_extraction": "bounding_box_crop",
        },
        "images": {
            "mask_url": mask_url,
            "overlay_url": overlay_url,
            "roi_url": roi_url,
        },
    }