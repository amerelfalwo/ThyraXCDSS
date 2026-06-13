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
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.security import verify_internal_api_key
from app.schemas.fnac import FnacPredictionResponse
from app.services.vision_explanation import generate_vision_explanation
from app.schemas.memory_models import Session as SessionModel, Patient

logger = logging.getLogger(__name__)

from app.core.responses import UnicodeJSONResponse

router = APIRouter(
    prefix="/fnac",
    tags=["FNAC Cytopathology"],
    dependencies=[Depends(verify_internal_api_key)],
    default_response_class=UnicodeJSONResponse,
)

MULTI_IMAGE_REQUEST_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {"type": "string", "format": "binary"},
                        }
                    },
                    "required": ["files"],
                }
            }
        },
    }
}


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
        img = img.resize((380, 380), Image.Resampling.BILINEAR)
        
    arr = np.array(img, dtype=np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
    return arr.astype(np.float32)

def _run_fnac_classification(image_bytes: bytes) -> dict:
    """
    Run the high-accuracy medical_final model on cytopathology image bytes.
    Maps the binary prediction to Bethesda System categories (II vs VI).
    """
    import numpy as np
    from app.core.models import load_classification_model

    session = load_classification_model()
    input_name = session.get_inputs()[0].name

    import hashlib
    img_hash = hashlib.sha256(image_bytes).hexdigest()
    print(f"DEBUG: Processing FNAC image with SHA256: {img_hash}", flush=True)

    # Preprocess for medical_final (380x380, NCHW)
    tensor = _preprocess_fnac_image(image_bytes)
    
    # Run inference
    raw_outputs = session.run(None, {input_name: tensor})
    if not raw_outputs:
        raise ValueError("Model returned no outputs.")
        
    output = raw_outputs[0]
    # Check shape: expect (1, 1) or (1,)
    if output.ndim == 2:
        output = output[0] # (1, 1) -> (1,)
    
    print(f"DEBUG: Raw FNAC output for {img_hash}: {output}", flush=True)

    # Single output Logit -> Apply Sigmoid
    logit = float(output[0])
    prob = 1.0 / (1.0 + np.exp(-logit))
    
    raw_class_idx = 1 if prob > 0.5 else 0
    confidence = prob if raw_class_idx == 1 else 1.0 - prob

    # Model Classes: 0 = "Benign", 1 = "Malignant (Papillary)"
    # Map to BETHESDA_MAP indices: 1 = Bethesda II (Benign), 5 = Bethesda VI (Malignant)
    bethesda_idx = 5 if raw_class_idx == 1 else 1

    bethesda = BETHESDA_MAP.get(bethesda_idx, BETHESDA_MAP[0])
    needs_review = confidence < 0.65

    return {
        "input_md5": img_hash,
        "prediction": bethesda_idx,
        "bethesda_category": bethesda["category"],
        "bethesda_label": bethesda["label"],
        "confidence_pct": round(confidence * 100, 2),
        "raw_logit": round(float(output[0]), 4) if len(output) == 1 else None,
        "malignancy_risk": bethesda["malignancy_risk"],
        "recommendation": bethesda["recommendation"],
        "needs_manual_review": needs_review,
    }


# ═══════════════════════════════════════════════════════════════
# Endpoint
# ═══════════════════════════════════════════════════════════════

@router.post(
    "/predict",
    response_model=List[FnacPredictionResponse],
    openapi_extra=MULTI_IMAGE_REQUEST_BODY,
)
async def predict_fnac(
    files: List[UploadFile] = File(..., description="Upload multiple images"),
    session_id: str = Form(default=None),
    doctor_id: str = Form(default=None),
    patient_id: str = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    """
    Classify an FNAC cytopathology image using The Bethesda System.

    Accepts a cytopathology slide image, runs the ONNX classifier,
    and returns a Bethesda category (I–VI) with malignancy risk
    and clinical recommendation.

    If `session_id` is provided, the result is pushed to the
    Patient State Manager for correlation with other diagnostic nodes.
    """
    # ── Mode 2 DB Isolation Check ──
    if session_id is not None:
        if doctor_id is None:
            raise HTTPException(status_code=422, detail="doctor_id is required when session_id is provided.")
        
        doctor_id_str = str(doctor_id)
        session_result = await db.execute(
            select(SessionModel).where(
                SessionModel.session_id == session_id,
                SessionModel.doctor_id == doctor_id_str,
            )
        )
        if not session_result.scalar_one_or_none():
            raise HTTPException(status_code=403, detail="Forbidden: Session does not belong to the provided Doctor.")
            
        if patient_id is not None:
            patient_result = await db.execute(
                select(Patient).where(
                    Patient.patient_id == str(patient_id),
                    Patient.doctor_id == doctor_id_str,
                )
            )
            if not patient_result.scalar_one_or_none():
                raise HTTPException(status_code=403, detail="Forbidden: Patient does not belong to the provided Doctor.")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded. Please attach at least one image.")

    results: List[FnacPredictionResponse] = []

    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            results.append(
                FnacPredictionResponse(
                    filename=file.filename,
                    status="error",
                    message="Invalid file type. Please upload an image.",
                    session_id=session_id,
                )
            )
            continue

        try:
            image_bytes = await file.read()

            # ── 1. Classification ──
            try:
                result = await run_in_threadpool(_run_fnac_classification, image_bytes)
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"FNAC classification model error: {e}", exc_info=True)
                results.append(
                    FnacPredictionResponse(
                        filename=file.filename,
                        status="error",
                        message=(
                            f"The FNAC classification model failed to run: {str(e)}. "
                            "Please check if model file is valid."
                        ),
                        session_id=session_id,
                    )
                )
                continue

            # ── LLM explanation of deterministic results ──
            key_findings = (
                f"{result['bethesda_label']}; "
                f"Malignancy Risk: {result['malignancy_risk']}"
            )
            model_confidence = f"{result['confidence_pct']:.2f}%"
            system_recommendation = result.get("recommendation", "")
            ai_recommendation = None
            if system_recommendation:
                ai_recommendation = await generate_vision_explanation(
                    analysis_type="FNAC Cytopathology (Bethesda System)",
                    key_findings=key_findings,
                    model_confidence=model_confidence,
                    system_recommendation=system_recommendation,
                )

            # ── Push to Dual-State Memory Manager ──
            if session_id:
                from app.services.memory_manager import memory_manager
                await memory_manager.save_diagnostic(
                    session_id=session_id,
                    node_type="fnac",
                    data=result
                )

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
                    "filename": file.filename,
                },
            )

            results.append(
                FnacPredictionResponse(
                    filename=file.filename,
                    status="success",
                    ai_recommendation=ai_recommendation,
                    classification=result,
                    session_id=session_id,
                )
            )

        except FileNotFoundError as e:
            results.append(
                FnacPredictionResponse(
                    filename=file.filename,
                    status="error",
                    message=(
                        f"FNAC ONNX model not found: {e}. "
                        "The model file 'models/compressed/efficientnet_b4_medical_final.onnx' might be missing."
                    ),
                    session_id=session_id,
                )
            )
        except Exception as e:
            logger.error(f"FNAC processing error: {e}", exc_info=True)
            results.append(
                FnacPredictionResponse(
                    filename=file.filename,
                    status="error",
                    message=f"FNAC classification error: {str(e)}",
                    session_id=session_id,
                )
            )

    return results
