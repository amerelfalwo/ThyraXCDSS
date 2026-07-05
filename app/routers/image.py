"""
Image Pipeline — Ultrasound Validation & ONNX Prediction (Nodes 3 & 4).

POST /image/validate  — Local ONNX gatekeeper (Node 3).
POST /image/predict   — ONNX segmentation + classification pipeline (Node 4).

Architecture Notes:
- All ONNX inference is CPU-bound and runs inside run_in_threadpool
via the centralized ``app.services.inference`` module.
- Models are cached in memory via @functools.lru_cache (first-load only).
- No external API calls in the validation pipeline.
- Results pushed to Patient State Manager if session_id provided.

Threading Pattern:
Async route handler (event loop):
    1. await file.read()                   ← async I/O
    2. await run_in_threadpool(…)          ← offload CPU to worker thread
    3. await storage.upload(…)             ← async I/O
    4. await db.commit()                   ← async I/O
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, File, Form
from starlette.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.security import verify_internal_api_key
from app.core.storage import upload_image_to_storage, get_signed_url
from app.core.database import get_db
from app.schemas.image import ImagePredictionResponse, ImageValidationResponse
from app.schemas.ai_nodes import MULTI_IMAGE_REQUEST_BODY
from app.services.inference import run_ultrasound_inference, run_gatekeeper_inference
from app.services.vision_explanation import generate_vision_explanation

logger = logging.getLogger(__name__)

def _sanitize_numpy(obj):
    """Convert numpy types to native Python for Pydantic serialization.

    Uses a JSON round-trip with a custom encoder so the C-level json
    module handles traversal — no Python-level recursion limit issues.
    """
    import json
    import numpy as np

    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.bool_):
                return bool(o)
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (bytes, bytearray)):
                return None
            return super().default(o)

    return json.loads(json.dumps(obj, cls=_Enc))


from app.core.responses import UnicodeJSONResponse

router = APIRouter(
    prefix="/image",
    tags=["Image Pipeline"],
    dependencies=[Depends(verify_internal_api_key)],
    default_response_class=UnicodeJSONResponse,
)
from app.services.memory_manager import memory_manager
from fastapi import Response

@router.get("/view/{image_id}")
async def get_image_view(image_id: int):
    """
    Retrieves a diagnostic image from the database.
    """
    image_data = await memory_manager.get_image(image_id)
    if not image_data:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=image_data, media_type="image/png")

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

        # ── Step 1: Async I/O — read file bytes without blocking ──
        image_bytes = await file.read()

        try:
            # ── Step 2: Offload CPU-bound inference to threadpool ──
            result = await run_in_threadpool(run_gatekeeper_inference, image_bytes)

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
    files: List[UploadFile] = File(..., description="Upload one or more ultrasound images"),
    force: bool = Query(False, description="Force prediction even if gatekeeper fails"),
    session_id: Optional[str] = Form(None, description="Enter Session ID for automated Synthesis trigger"),
    db: AsyncSession = Depends(get_db),
):
    """
    Run the full ONNX segmentation → classification pipeline.

    Threading Architecture:
      1. ``await file.read()`` — non-blocking file I/O on event loop.
      2. ``await run_in_threadpool(run_ultrasound_inference, …)`` —
         offloads ALL CPU-bound ONNX work (U-Net segmentation, ROI
         extraction, classification) to a worker thread.
      3. Async I/O (storage upload, DB commit, LLM call) stays on
         the event loop after the threadpool returns.

    If ``session_id`` is provided, the result is pushed to the
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
            # ── Step 1: Async I/O — read file bytes without blocking ──
            image_bytes = await file.read()

            # ── Step 2: Offload CPU-bound inference to threadpool ──
            # This calls the pure synchronous function from
            # app.services.inference which runs U-Net segmentation +
            # classification entirely on a worker thread.
            result = await run_in_threadpool(
                run_ultrasound_inference, image_bytes
            )
            result["filename"] = file.filename

            # ── Step 3: Async I/O — Upload images to database ──
            # (runs on the event loop, non-blocking)
            images_data = result.pop("images", None)
            if images_data and "unique_id" in images_data:
                uid = images_data["unique_id"]
                folder = session_id or "unassigned"

                async def _save_db(raw: bytes | None, image_type: str) -> str | None:
                    if not raw:
                        return None
                    try:
                        image_id = await memory_manager.save_image(
                            session_id=folder,
                            image_data=raw,
                            image_type=image_type
                        )
                        return f"/image/view/{image_id}"
                    except Exception as err:
                        logger.error(f"Failed to save {image_type} to DB: {err}")
                        return None

                result["images"] = {
                    "mask_url": await _save_db(images_data.get("mask_bytes"), "mask"),
                    "overlay_url": await _save_db(images_data.get("overlay_bytes"), "overlay"),
                    "roi_url": await _save_db(images_data.get("roi_bytes"), "roi"),
                }

            # ── Forced bypass red flag ──
            if force:
                result["validation_bypassed"] = True
                result["warning"] = (
                    "Warning: Image validation was manually bypassed. "
                    "The system assumes the input is a valid ultrasound, "
                    "but results may be unreliable if it is not."
                )

            # ── Step 4: Async I/O — LLM explanation ──
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

            # ── Step 5: Async I/O — Push to Database ──
            if session_id:
                await memory_manager.save_diagnostic(
                    session_id=session_id,
                    node_type="ultrasound",
                    data=dict(result)  # copy to prevent circular ref
                )

                # Automated synthesis trigger has been removed to keep Node 4 independent.


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

            results.append(_sanitize_numpy(result))

        except Exception as e:
            results.append(
                ImagePredictionResponse(
                    filename=file.filename,
                    status="error",
                    message=f"Image processing error: {e}",
                )
            )

    return results
