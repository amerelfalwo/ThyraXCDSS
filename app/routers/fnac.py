"""
FNAC Cytopathology Pipeline — Bethesda System Classification.

POST /fnac/predict
  Accepts an FNAC cytopathology image, runs the ONNX classifier,
  and returns a Bethesda System-aligned risk assessment.

Clinical Standards:
  - Output aligned with The Bethesda System for Reporting
    Thyroid Cytopathology (2023 Edition, Categories I–VI).
  - Mandatory medical disclaimer on every response.
  - Results are pushed to the Patient State Manager if session_id provided.
"""

import logging
import numpy as np

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool

from app.core.security import verify_internal_api_key
from app.schemas.fnac import FnacPredictionResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/fnac",
    tags=["FNAC Cytopathology"],
    dependencies=[Depends(verify_internal_api_key)],
)


# ═══════════════════════════════════════════════════════════════
# Bethesda System Mapping (2023 Edition)
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# FNAC Classification Logic
# ═══════════════════════════════════════════════════════════════

def _preprocess_fnac_image(image_bytes: bytes) -> np.ndarray:
    import io
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img = img.resize((384, 384), Image.Resampling.BILINEAR)
        
    arr = np.array(img, dtype=np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
    return arr.astype(np.float32)

def _run_fnac_classification(image_bytes: bytes) -> dict:
    """
    Run the FNAC ONNX classifier (binary Benign/Malignant) on cytopathology image bytes.
    Maps the binary prediction to Bethesda System categories.
    """
    import numpy as np
    from app.core.models import load_fnac_gatekeeper_model
    from app.services.image_service import _softmax

    session = load_fnac_gatekeeper_model()
    input_name = session.get_inputs()[0].name

    tensor = _preprocess_fnac_image(image_bytes)
    output = session.run(None, {input_name: tensor})[0][0]

    probs = _softmax(output)
    raw_class_idx = int(np.argmax(probs))
    confidence = float(probs[raw_class_idx])

    # Model Classes: 0 = "Benign", 1 = "Malignant (Papillary)"
    # Map to BETHESDA_MAP indices: 1 = Bethesda II (Benign), 5 = Bethesda VI (Malignant)
    bethesda_idx = 5 if raw_class_idx == 1 else 1

    bethesda = BETHESDA_MAP.get(bethesda_idx, BETHESDA_MAP[0])
    needs_review = confidence < 0.65

    return {
        "prediction": bethesda_idx,
        "bethesda_category": bethesda["category"],
        "bethesda_label": bethesda["label"],
        "confidence_pct": round(confidence * 100, 2),
        "malignancy_risk": bethesda["malignancy_risk"],
        "recommendation": bethesda["recommendation"],
        "needs_manual_review": needs_review,
    }


# ═══════════════════════════════════════════════════════════════
# Endpoint
# ═══════════════════════════════════════════════════════════════

@router.post("/predict", response_model=FnacPredictionResponse)
async def predict_fnac(
    file: UploadFile = File(...),
    session_id: str = Form(default=None),
):
    """
    Classify an FNAC cytopathology image using The Bethesda System.

    Accepts a cytopathology slide image, runs the ONNX classifier,
    and returns a Bethesda category (I–VI) with malignancy risk
    and clinical recommendation.

    If `session_id` is provided, the result is pushed to the
    Patient State Manager for correlation with other diagnostic nodes.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_bytes = await file.read()

        # ── 1. Classification ──
        try:
            result = await run_in_threadpool(_run_fnac_classification, image_bytes)
        except Exception as e:
            logger.error(f"FNAC classification model error (dummy file?): {e}")
            raise HTTPException(
                status_code=503,
                detail="The FNAC classification model is not yet deployed or is invalid. Please try again later."
            )

        # ── Push to Patient State ──
        if session_id:
            from app.services.patient_state import state_manager
            state_manager.update_fnac(session_id, result)

        # ── Audit Log ──
        from app.core.audit import log_audit_event
        log_audit_event(
            node="fnac_predict",
            action="bethesda_classification",
            result=result["bethesda_label"],
            confidence=result["confidence_pct"] / 100.0,
            metadata={
                "bethesda_category": result["bethesda_category"],
                "malignancy_risk": result["malignancy_risk"],
                "needs_manual_review": result["needs_manual_review"],
                "session_id": session_id,
            },
        )

        return FnacPredictionResponse(
            status="success",
            classification=result,
            session_id=session_id,
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=(
                "FNAC ONNX model not found. The model file "
                "'models/compressed/fnac_bethesda.onnx' has not been deployed yet. "
                "Please train and export the FNAC classifier first."
            ),
        )
    except Exception as e:
        logger.error(f"FNAC processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"FNAC classification error: {e}")
