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
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.security import verify_internal_api_key
from app.core.storage import upload_image_to_storage, get_signed_url
from app.core.database import get_db
from app.core.db_models import PatientSession
from app.schemas.image import ImagePredictionResponse, ImageValidationResponse
from app.services.image_service import run_gatekeeper
from app.services.vision_explanation import generate_vision_explanation

logger = logging.getLogger(__name__)

from app.core.responses import UnicodeJSONResponse

router = APIRouter(
    prefix="/image",
    tags=["Image Pipeline"],
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
# Node 3: /image/validate — Local ONNX Gatekeeper
# ═══════════════════════════════════════════════════════════════

@router.post(
    "/validate",
    response_model=List[ImageValidationResponse],
    openapi_extra=MULTI_IMAGE_REQUEST_BODY,
)
async def validate_ultrasound_image(
    files: List[UploadFile] = File(..., description="Upload multiple images"),
    force: bool = False,
):
    """
    Gatekeeper — verify the uploaded image is a valid medical ultrasound
    using the locally-cached MobileNetV2 ONNX model (gatekeeper.onnx).

    Classes: 0 = 'other', 1 = 'ultrasound'. Threshold: prob >= 0.60.

    Args:
        files: Uploaded image files.
        force: If True, bypass ONNX inference and immediately accept the image.
            Response will include a warning in /predict.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded. Please attach at least one image.")

    results: List[ImageValidationResponse] = []

    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            results.append(
                ImageValidationResponse(
                    filename=file.filename,
                    is_ultrasound=False,
                    confidence=0.0,
                    reason="Invalid file type. Please upload an image.",
                    status="error",
                )
            )
            continue

        # ── Human-in-the-loop override ──
        if force:
            logger.warning("Gatekeeper bypassed by user (force=True)")
            results.append(
                ImageValidationResponse(
                    filename=file.filename,
                    is_ultrasound=True,
                    reason="Forced by user",
                )
            )
            continue

        image_bytes = await file.read()

        try:
            # ONNX inference is CPU-bound — must run in threadpool
            result = await run_in_threadpool(run_gatekeeper, image_bytes)

            logger.info(
                f"Gatekeeper result: is_ultrasound={result['is_ultrasound']} "
                f"confidence={result['confidence']:.4f}"
            )
            results.append(
                ImageValidationResponse(
                    filename=file.filename,
                    is_ultrasound=result["is_ultrasound"],
                    confidence=result["confidence"],
                    reason=result["reason"],
                )
            )

        except Exception as e:
            logger.error(f"Gatekeeper inference error: {e}", exc_info=True)
            # Fail-closed: do NOT accept the image if model crashes
            results.append(
                ImageValidationResponse(
                    filename=file.filename,
                    is_ultrasound=False,
                    reason="Gatekeeper model error. Could not verify image — please try again.",
                    status="error",
                )
            )

    return results


# ═══════════════════════════════════════════════════════════════
# Node 4: /image/predict — ONNX Segmentation + Classification
# ═══════════════════════════════════════════════════════════════

@router.post(
    "/predict",
    response_model=List[ImagePredictionResponse],
    openapi_extra=MULTI_IMAGE_REQUEST_BODY,
)
async def predict_ultrasound_image(
    request: Request,
    files: List[UploadFile] = File(..., description="Upload multiple images"),
    force: bool = False,
    session_id: str = Form(default=None),
    doctor_id: str = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    """
    Run the full ONNX segmentation → classification pipeline.

    If `session_id` is provided, the result is pushed to the
    Patient State Manager for downstream correlation.

    Args:
        request: FastAPI Request (used to build media URLs).
        files:   Uploaded ultrasound images.
        force:   If True, marks the response with a bypass warning.
        session_id: Optional session ID for patient state tracking.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded. Please attach at least one image.")

    base_url = str(request.base_url)
    results: List[ImagePredictionResponse] = []

    from app.segmentation.model import process_full_pipeline

    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            results.append(
                ImagePredictionResponse(
                    filename=file.filename,
                    status="error",
                    message="Invalid file type. Please upload an image.",
                )
            )
            continue

        try:
            image_bytes = await file.read()
            result = await run_in_threadpool(process_full_pipeline, image_bytes, base_url)
            result["filename"] = file.filename

            # ── Upload Images to Storage (or fallback to base64) ──
            images_data = result.pop("images", None)
            if images_data and "unique_id" in images_data:
                uid = images_data["unique_id"]
                folder = session_id or "unassigned"

                try:
                    mask_path = await upload_image_to_storage(images_data.get("mask_bytes"), f"{uid}_mask.png", folder_path=folder)
                    overlay_path = await upload_image_to_storage(images_data.get("overlay_bytes"), f"{uid}_overlay.png", folder_path=folder)
                    roi_path = await upload_image_to_storage(images_data.get("roi_bytes"), f"{uid}_roi.png", folder_path=folder)

                    mask_url = await get_signed_url(mask_path)
                    overlay_url = await get_signed_url(overlay_path)
                    roi_url = await get_signed_url(roi_path)

                    result["images"] = {
                        "mask_url": mask_url,
                        "overlay_url": overlay_url,
                        "roi_url": roi_url,
                    }
                except Exception as storage_err:
                    # Fallback: encode images as base64 data URLs
                    import base64
                    logger.warning(f"Supabase storage unavailable, falling back to base64: {storage_err}")

                    def _to_data_url(raw: bytes | None) -> str | None:
                        if not raw:
                            return None
                        return f"data:image/png;base64,{base64.b64encode(raw).decode()}"

                    result["images"] = {
                        "mask_url": _to_data_url(images_data.get("mask_bytes")),
                        "overlay_url": _to_data_url(images_data.get("overlay_bytes")),
                        "roi_url": _to_data_url(images_data.get("roi_bytes")),
                    }

            # ── Forced bypass red flag ──
            if force:
                result["validation_bypassed"] = True
                result["warning"] = (
                    "Warning: Image validation was manually bypassed. "
                    "The system assumes the input is a valid ultrasound, "
                    "but results may be unreliable if it is not."
                )

            # ── LLM explanation of deterministic results ──
            cls = result.get("classification", {})
            analysis_type = "Ultrasound Segmentation + Classification (ACR TI-RADS)"
            if isinstance(cls, dict):
                key_findings = (
                    f"Label: {cls.get('label')}; "
                    f"Risk Level: {cls.get('risk_level')}; "
                    f"ACR TI-RADS: {cls.get('acr_tirads_level')}"
                )
                model_confidence = f"{cls.get('confidence_pct', 0):.2f}%"
                system_recommendation = cls.get("clinical_recommendation", "")
            else:
                key_findings = str(cls) if cls else "No classification result."
                model_confidence = "N/A"
                system_recommendation = result.get("message", "") or str(cls)

            if system_recommendation:
                ai_recommendation = await generate_vision_explanation(
                    analysis_type=analysis_type,
                    key_findings=key_findings,
                    model_confidence=model_confidence,
                    system_recommendation=system_recommendation,
                )
                if ai_recommendation:
                    result["ai_recommendation"] = ai_recommendation

            # ── Push to Database (Phase 3) ──
            if session_id and doctor_id:
                stmt = select(PatientSession).where(
                    PatientSession.session_id == session_id,
                    PatientSession.doctor_id == doctor_id
                )
                session_result = await db.execute(stmt)
                patient_session = session_result.scalar_one_or_none()

                if patient_session:
                    # If multiple images are processed, we can store them in a list or just the latest one.
                    # Here we update with the latest result as a JSON dictionary.
                    patient_session.ultrasound_result = result
                    await db.commit()
                else:
                    logger.warning(f"PatientSession not found for session_id={session_id} and doctor_id={doctor_id}")
            elif session_id:
                logger.warning("session_id provided without doctor_id, skipping database update.")

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
                        "filename": file.filename,
                    },
                )
            else:
                log_audit_event(
                    node="image_predict",
                    action="onnx_classification",
                    result=str(cls),
                    confidence=None,
                    metadata={"empty_mask": True, "session_id": session_id, "filename": file.filename},
                )

            results.append(result)

        except Exception as e:
            results.append(
                ImagePredictionResponse(
                    filename=file.filename,
                    status="error",
                    message=f"Image processing error: {e}",
                )
            )

    return results
