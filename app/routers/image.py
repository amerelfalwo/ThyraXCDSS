"""
Image Pipeline — Ultrasound Validation & ONNX Prediction (Nodes 3 & 4).

POST /image/validate  — Local ONNX gatekeeper (Node 3).
POST /image/predict   — ONNX segmentation + classification pipeline (Node 4).

Architecture Notes:
  - All ONNX inference is CPU-bound and runs inside run_in_threadpool.
  - Models are cached in memory via @functools.lru_cache (first-load only).
  - No external API calls in the validation pipeline.
  - Results pushed to Patient State Manager if session_id provided.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool

from app.core.security import verify_internal_api_key
from app.schemas.image import ImagePredictionResponse, ImageValidationResponse
from app.services.image_service import run_gatekeeper

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/image",
    tags=["Image Pipeline"],
    dependencies=[Depends(verify_internal_api_key)],
)


# ═══════════════════════════════════════════════════════════════
# Node 3: /image/validate — Local ONNX Gatekeeper
# ═══════════════════════════════════════════════════════════════

@router.post("/validate", response_model=ImageValidationResponse)
async def validate_ultrasound_image(
    file: UploadFile = File(...),
    force: bool = False,
):
    """
    Gatekeeper — verify the uploaded image is a valid medical ultrasound
    using the locally-cached MobileNetV2 ONNX model (gatekeeper.onnx).

    Classes: 0 = 'other', 1 = 'ultrasound'. Threshold: prob >= 0.60.

    Args:
        file:  Uploaded image file.
        force: If True, bypass ONNX inference and immediately accept the image.
               Response will include a warning in /predict.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    # ── Human-in-the-loop override ──
    if force:
        logger.warning("Gatekeeper bypassed by user (force=True)")
        return ImageValidationResponse(
            is_ultrasound=True,
            reason="Forced by user",
        )

    image_bytes = await file.read()

    try:
        # ONNX inference is CPU-bound — must run in threadpool
        result = await run_in_threadpool(run_gatekeeper, image_bytes)

        logger.info(
            f"Gatekeeper result: is_ultrasound={result['is_ultrasound']} "
            f"confidence={result['confidence']:.4f}"
        )
        return ImageValidationResponse(
            is_ultrasound=result["is_ultrasound"],
            confidence=result["confidence"],
            reason=result["reason"],
        )

    except Exception as e:
        logger.error(f"Gatekeeper inference error: {e}", exc_info=True)
        # Fail-closed: do NOT accept the image if model crashes
        return ImageValidationResponse(
            is_ultrasound=False,
            reason="Gatekeeper model error. Could not verify image — please try again.",
        )


# ═══════════════════════════════════════════════════════════════
# Node 4: /image/predict — ONNX Segmentation + Classification
# ═══════════════════════════════════════════════════════════════

@router.post("/predict", response_model=ImagePredictionResponse)
async def predict_ultrasound_image(
    request: Request,
    file: UploadFile = File(...),
    force: bool = False,
    session_id: str = Form(default=None),
):
    """
    Run the full ONNX segmentation → classification pipeline.

    If `session_id` is provided, the result is pushed to the
    Patient State Manager for downstream correlation.

    Args:
        request: FastAPI Request (used to build media URLs).
        file:    Uploaded ultrasound image.
        force:   If True, marks the response with a bypass warning.
        session_id: Optional session ID for patient state tracking.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_bytes = await file.read()
        base_url = str(request.base_url)

        from app.segmentation.model import process_full_pipeline

        result = await run_in_threadpool(process_full_pipeline, image_bytes, base_url)

        # ── Forced bypass red flag ──
        validation_bypassed = force
        if validation_bypassed:
            result["validation_bypassed"] = True
            result["warning"] = (
                "Warning: Image validation was manually bypassed. "
                "The system assumes the input is a valid ultrasound, "
                "but results may be unreliable if it is not."
            )

        # ── Push to Patient State Manager ──
        if session_id:
            from app.services.patient_state import state_manager
            state_manager.update_ultrasound(session_id, result)

        # ── Audit log ──
        from app.core.audit import log_audit_event

        cls = result.get("classification", {})
        if isinstance(cls, dict):
            log_audit_event(
                node="image_predict",
                action="onnx_classification",
                result=cls.get("label", "no_detection"),
                confidence=cls.get("confidence_pct", 0) / 100.0 if cls else None,
                metadata={
                    "risk_level": cls.get("risk_level"),
                    "acr_tirads_level": cls.get("acr_tirads_level"),
                    "clinical_recommendation": cls.get("clinical_recommendation"),
                    "needs_manual_review": cls.get("needs_manual_review", False),
                    "session_id": session_id,
                },
            )
        else:
            log_audit_event(
                node="image_predict",
                action="onnx_classification",
                result=str(cls),
                confidence=None,
                metadata={"empty_mask": True, "session_id": session_id},
            )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {e}")
