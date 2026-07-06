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
from app.schemas.ai_nodes import MULTI_IMAGE_REQUEST_BODY
from app.services.inference import run_fnac_inference
logger = logging.getLogger(__name__)


def _sanitize_numpy(obj):
    """Convert numpy types to native Python for Pydantic serialization."""
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
    prefix="/fnac",
    tags=["FNAC Cytopathology"],
    dependencies=[Depends(verify_internal_api_key)],
    default_response_class=UnicodeJSONResponse,
)




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

            # ── Step 3: LLM explanation removed for speed ──
            # (If AI explanation is needed, it will be generated in Node 6 or via Node 5)
            ai_recommendation = None

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
                    classification=_sanitize_numpy(result),
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
