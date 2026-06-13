"""
Inference Service — Pure Synchronous CPU-Bound Model Execution.

This module isolates ALL heavy ONNX / CV inference into pure synchronous
functions.  These functions:

  • Are NEVER called directly from an ``async def`` endpoint.
  • Are ALWAYS called via ``starlette.concurrency.run_in_threadpool(…)``
    so they execute on a worker thread and never block the asyncio loop.
  • Accept raw ``bytes`` and return plain dicts — no DB, no async I/O.
  • Are unit-testable in isolation without any FastAPI or DB fixture.

Architecture Pattern (Pipeline Interceptor):
    ┌──────────────────────────────────────────────────┐
    │  async route handler (event loop)                │
    │    1. await file.read()        ← async I/O       │
    │    2. await run_in_threadpool(  ← offload ──┐    │
    │         run_ultrasound_inference, …)         │    │
    │    3. await storage.upload(…)   ← async I/O  │    │
    │    4. await db.commit()         ← async I/O  │    │
    └─────────────────────────────────────────────┼────┘
                                                  │
    ┌─────────────────────────────────────────────▼────┐
    │  worker thread (threadpool)                      │
    │    • Image decoding (PIL / cv2)                   │
    │    • ONNX .run() — segmentation & classification │
    │    • NumPy post-processing & risk mapping        │
    │    → returns plain dict                           │
    └──────────────────────────────────────────────────┘
"""

import io
import time
import hashlib
import logging

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1. Ultrasound Inference (U-Net Segmentation + Classification)
# ═══════════════════════════════════════════════════════════════

def run_ultrasound_inference(image_bytes: bytes, base_url: str) -> dict:
    """
    Execute the full ultrasound segmentation → classification pipeline.

    This is a **pure synchronous, CPU-bound** function.  It MUST be
    called via ``run_in_threadpool`` from async endpoints.

    Steps:
        1. Decode raw bytes into an image array.
        2. Run ONNX U-Net segmentation to generate a binary mask.
        3. Extract the ROI via bounding-box crop.
        4. Run ONNX classification on the ROI.
        5. Map the result to ACR TI-RADS risk levels.
        6. Encode mask/overlay/roi images as raw bytes.

    Args:
        image_bytes: Raw bytes of the uploaded ultrasound image.
        base_url:    The server's base URL for constructing media paths.

    Returns:
        A plain dict containing segmentation masks (as raw bytes),
        classification result, risk level, and clinical recommendation.

    Raises:
        ValueError: If the image cannot be decoded or the model fails.
    """
    from app.segmentation.model import process_full_pipeline

    img_hash = hashlib.sha256(image_bytes).hexdigest()[:12]
    logger.info(f"[ultrasound] Starting inference — hash={img_hash}")

    t0 = time.perf_counter()
    result = process_full_pipeline(image_bytes, base_url)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        f"[ultrasound] Inference complete — hash={img_hash} "
        f"elapsed={elapsed_ms:.1f}ms "
        f"label={result.get('classification', {}).get('label', 'N/A')}"
    )
    return result


# ═══════════════════════════════════════════════════════════════
# 2. FNAC Cytopathology Inference (EfficientNet-B4 ONNX)
# ═══════════════════════════════════════════════════════════════

# ── ImageNet normalization constants (matches EfficientNet training) ──
_FNAC_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_FNAC_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# Bethesda System Mapping (2023 Edition, Categories I–VI)
BETHESDA_MAP = {
    0: {
        "category": "I",
        "label": "Bethesda I — Non-diagnostic / Unsatisfactory",
        "malignancy_risk": "1–4%",
        "recommendation": (
            "Specimen is non-diagnostic. Repeat FNA with ultrasound guidance "
            "is recommended. If repeatedly non-diagnostic, consider surgical "
            "excision for histological diagnosis."
        ),
    },
    1: {
        "category": "II",
        "label": "Bethesda II — Benign",
        "malignancy_risk": "0–3%",
        "recommendation": (
            "Findings are consistent with a benign lesion. No immediate "
            "intervention required. Clinical and sonographic follow-up "
            "in 12–24 months is recommended per ATA guidelines."
        ),
    },
    2: {
        "category": "III",
        "label": "Bethesda III — Atypia of Undetermined Significance (AUS/FLUS)",
        "malignancy_risk": "6–18%",
        "recommendation": (
            "Atypical cells identified. Consider molecular testing (e.g., "
            "Afirma, ThyroSeq) to refine risk. Repeat FNA in 3–6 months "
            "or consider diagnostic lobectomy based on clinical context."
        ),
    },
    3: {
        "category": "IV",
        "label": "Bethesda IV — Follicular Neoplasm / Suspicious for FN (FN/SFN)",
        "malignancy_risk": "10–40%",
        "recommendation": (
            "Follicular-patterned lesion identified. Molecular testing is "
            "recommended if available. Diagnostic surgical lobectomy is the "
            "standard of care for definitive diagnosis."
        ),
    },
    4: {
        "category": "V",
        "label": "Bethesda V — Suspicious for Malignancy",
        "malignancy_risk": "45–60%",
        "recommendation": (
            "Cytological findings are suspicious for malignancy (likely "
            "papillary thyroid carcinoma). Near-total thyroidectomy or "
            "lobectomy is recommended. Pre-surgical staging with neck "
            "ultrasound and CT is advised."
        ),
    },
    5: {
        "category": "VI",
        "label": "Bethesda VI — Malignant",
        "malignancy_risk": "94–96%",
        "recommendation": (
            "Cytological findings are diagnostic of malignancy. Total "
            "thyroidectomy with central neck dissection is the standard "
            "treatment. Pre-operative staging, multidisciplinary tumor "
            "board review, and surgical referral are indicated."
        ),
    },
}


def _preprocess_fnac_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess raw FNAC image bytes into the tensor expected by
    the EfficientNet-B4 ONNX model.

    Pipeline:
        1. Decode bytes → PIL RGB image
        2. Resize to (380, 380) using BILINEAR
        3. Scale to [0, 1]
        4. Apply ImageNet mean/std normalization
        5. Transpose HWC → CHW
        6. Add batch dimension → (1, 3, 380, 380)

    Args:
        image_bytes: Raw bytes of the cytopathology slide image.

    Returns:
        np.ndarray of shape (1, 3, 380, 380) and dtype float32.
    """
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img = img.resize((380, 380), Image.Resampling.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - _FNAC_MEAN) / _FNAC_STD
    arr = arr.transpose(2, 0, 1)         # HWC → CHW
    arr = np.expand_dims(arr, axis=0)    # → (1, 3, 380, 380)
    return arr.astype(np.float32)


def run_fnac_inference(image_bytes: bytes) -> dict:
    """
    Execute the FNAC cytopathology ONNX classification pipeline.

    This is a **pure synchronous, CPU-bound** function.  It MUST be
    called via ``run_in_threadpool`` from async endpoints.

    Steps:
        1. Preprocess the cytopathology slide image.
        2. Run ONNX inference (EfficientNet-B4 binary classifier).
        3. Apply sigmoid to raw logit.
        4. Map prediction to Bethesda System categories (II vs VI).

    Args:
        image_bytes: Raw bytes of the cytopathology slide image.

    Returns:
        A plain dict containing Bethesda category, malignancy risk,
        confidence percentage, clinical recommendation, and review flag.

    Raises:
        ValueError: If the model returns unexpected output.
        FileNotFoundError: If the ONNX model file is missing.
    """
    from app.core.models import load_classification_model

    img_hash = hashlib.sha256(image_bytes).hexdigest()[:12]
    logger.info(f"[fnac] Starting inference — hash={img_hash}")

    t0 = time.perf_counter()

    session = load_classification_model()
    input_name = session.get_inputs()[0].name

    # Preprocess for medical_final (380x380, NCHW)
    tensor = _preprocess_fnac_image(image_bytes)

    # Run inference
    raw_outputs = session.run(None, {input_name: tensor})
    if not raw_outputs:
        raise ValueError("FNAC model returned no outputs.")

    output = raw_outputs[0]
    if output.ndim == 2:
        output = output[0]  # (1, 1) → (1,)

    # Single output Logit → Apply Sigmoid
    logit = float(output[0])
    prob = 1.0 / (1.0 + np.exp(-logit))

    raw_class_idx = 1 if prob > 0.5 else 0
    confidence = prob if raw_class_idx == 1 else 1.0 - prob

    # Model Classes: 0 = "Benign", 1 = "Malignant (Papillary)"
    # Map to BETHESDA_MAP indices: 1 = Bethesda II, 5 = Bethesda VI
    bethesda_idx = 5 if raw_class_idx == 1 else 1
    bethesda = BETHESDA_MAP.get(bethesda_idx, BETHESDA_MAP[0])
    needs_review = confidence < 0.65

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"[fnac] Inference complete — hash={img_hash} "
        f"elapsed={elapsed_ms:.1f}ms "
        f"bethesda={bethesda['category']} confidence={confidence:.4f}"
    )

    return {
        "input_md5": hashlib.sha256(image_bytes).hexdigest(),
        "prediction": bethesda_idx,
        "bethesda_category": bethesda["category"],
        "bethesda_label": bethesda["label"],
        "confidence_pct": round(confidence * 100, 2),
        "raw_logit": round(logit, 4),
        "malignancy_risk": bethesda["malignancy_risk"],
        "recommendation": bethesda["recommendation"],
        "needs_manual_review": needs_review,
    }


# ═══════════════════════════════════════════════════════════════
# 3. Gatekeeper Inference (MobileNetV2 — Ultrasound validation)
# ═══════════════════════════════════════════════════════════════

def run_gatekeeper_inference(image_bytes: bytes) -> dict:
    """
    Run the ONNX gatekeeper model on raw image bytes.

    This is a **pure synchronous, CPU-bound** function.  It MUST be
    called via ``run_in_threadpool`` from async endpoints.

    Delegates to the existing image_service.run_gatekeeper() which
    already follows the correct synchronous pattern.

    Args:
        image_bytes: Raw bytes of the uploaded image.

    Returns:
        Dict with ``is_ultrasound`` (bool), ``reason`` (str),
        and ``confidence`` (float, 0–1).
    """
    from app.services.image_service import run_gatekeeper

    img_hash = hashlib.sha256(image_bytes).hexdigest()[:12]
    logger.info(f"[gatekeeper] Starting inference — hash={img_hash}")

    t0 = time.perf_counter()
    result = run_gatekeeper(image_bytes)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        f"[gatekeeper] Inference complete — hash={img_hash} "
        f"elapsed={elapsed_ms:.1f}ms "
        f"is_ultrasound={result['is_ultrasound']} "
        f"confidence={result['confidence']:.4f}"
    )
    return result
