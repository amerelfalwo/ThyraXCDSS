"""
Image Validation Service (Node 3 — Ultrasound Gatekeeper).

Uses a locally-cached MobileNetV2 ONNX model (gatekeeper.onnx) to
classify whether an uploaded image is a valid medical ultrasound scan.
Zero external API calls — fully offline, deterministic, and fast.

Classes: Index 0 = 'other', Index 1 = 'ultrasound'
Threshold: prob >= 0.60 → accepted as ultrasound (Fail-Closed default)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ── ImageNet normalization constants (matches MobileNetV2 training) ──
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Classification threshold ──
_ULTRASOUND_THRESHOLD = 0.60


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess raw image bytes into the tensor expected by the ONNX gatekeeper.

    Pipeline:
      1. Decode bytes → PIL RGB image
      2. Resize to (224, 224) using BILINEAR (matches PyTorch defaults)
      3. Scale to [0, 1]
      4. Apply ImageNet mean/std normalization
      5. Transpose HWC → CHW
      6. Add batch dimension → (1, 3, 224, 224)
      7. Cast to float32

    Args:
        image_bytes: Raw bytes of the uploaded image.

    Returns:
        np.ndarray of shape (1, 3, 224, 224) and dtype float32.
    """
    import io
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        # Critical: PyTorch defaults to BILINEAR resampling
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        
    arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1], HWC

    # Reshape mean/std for explicit broadcasting over (H, W, C)
    mean = _MEAN.reshape(1, 1, 3)
    std = _STD.reshape(1, 1, 3)
    
    arr = (arr - mean) / std            # Normalize
    arr = arr.transpose(2, 0, 1)        # HWC → CHW
    arr = np.expand_dims(arr, axis=0)   # → (1, 3, 224, 224)
    return arr.astype(np.float32)


def run_gatekeeper(image_bytes: bytes) -> dict:
    """
    Run the ONNX gatekeeper model on raw image bytes.

    This is a CPU-bound synchronous function and MUST be called
    inside ``run_in_threadpool`` from async endpoints.

    Args:
        image_bytes: Raw bytes of the uploaded image.

    Returns:
        Dict with ``is_ultrasound`` (bool), ``reason`` (str),
        and ``confidence`` (float, 0–1).
    """
    from app.core.models import load_gatekeeper_model

    session = load_gatekeeper_model()
    input_name = session.get_inputs()[0].name

    tensor = preprocess_image(image_bytes)
    raw_output = session.run(None, {input_name: tensor})[0][0]  # shape: (2,)

    probs = _softmax(raw_output)
    ultrasound_prob = float(probs[1])  # Index 1 = 'ultrasound'

    logger.info(
        f"Gatekeeper inference — ultrasound_prob={ultrasound_prob:.4f} "
        f"(threshold={_ULTRASOUND_THRESHOLD})"
    )

    if ultrasound_prob >= _ULTRASOUND_THRESHOLD:
        return {
            "is_ultrasound": True,
            "confidence": round(ultrasound_prob, 4),
            "reason": "Verified by AI Gatekeeper",
        }
    else:
        return {
            "is_ultrasound": False,
            "confidence": round(ultrasound_prob, 4),
            "reason": "Image does not appear to be a medical ultrasound.",
        }
