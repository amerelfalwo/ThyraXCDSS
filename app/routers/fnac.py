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

Threading Pattern:
  Async route handler (event loop):
    1. await file.read()                   ← async I/O
    2. await run_in_threadpool(…)          ← offload CPU to worker thread
    3. await generate_vision_explanation() ← async LLM call
    4. await memory_manager.save(…)        ← async I/O
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from starlette.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.security import verify_internal_api_key
from app.schemas.fnac import FnacPredictionResponse
from app.services.inference import run_fnac_inference
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

    Threading Architecture:
      1. ``await file.read()`` — non-blocking file I/O on event loop.
      2. ``await run_in_threadpool(run_fnac_inference, …)`` —
         offloads ALL CPU-bound ONNX work (preprocessing, EfficientNet-B4
         inference, sigmoid + Bethesda mapping) to a worker thread.
      3. Async I/O (LLM explanation, memory manager, audit) stays on
         the event loop after the threadpool returns.

    If ``session_id`` is provided, the result is pushed to the
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
            # ── Step 1: Async I/O — read file bytes without blocking ──
            image_bytes = await file.read()

            # ── Step 2: Offload CPU-bound inference to threadpool ──
            # This calls the pure synchronous function from
            # app.services.inference which runs EfficientNet-B4 ONNX
            # classification entirely on a worker thread.
            try:
                result = await run_in_threadpool(run_fnac_inference, image_bytes)
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

            # ── Step 3: Async I/O — LLM explanation ──
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

            # ── Step 4: Async I/O — Push to Dual-State Memory Manager ──
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
