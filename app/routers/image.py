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

    Multi-Image Consensus Logic:
      When multiple images are uploaded, each image is processed
      independently through the full pipeline. After all individual
      results are collected, a unified CONSENSUS classification is
      computed using a clinically-safe aggregation strategy:
        - Highest suspicion wins (worst-case approach per ATA guidelines).
        - Confidence is averaged across all suspicious findings.
        - The consensus is appended to each individual result AND
          saved to the Patient State Manager as the authoritative decision.

    Threading Architecture:
      1. ``await file.read()`` — non-blocking file I/O on event loop.
      2. ``await run_in_threadpool(run_ultrasound_inference, …)`` —
         offloads ALL CPU-bound ONNX work to a worker thread.
      3. Async I/O (storage upload, DB commit) stays on the event loop.

    Args:
        request: FastAPI Request (used to build media URLs).
        files:   Uploaded ultrasound images.
        force:   If True, marks the response with a bypass warning.
        session_id: Optional session ID for patient state tracking.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded. Please attach at least one image.")

    base_url = str(request.base_url)
    results: List[dict] = []

    # ── Track classification votes for consensus ──
    classification_votes: List[dict] = []

    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            results.append(
                ImagePredictionResponse(
                    filename=file.filename,
                    status="error",
                    message="Invalid file type. Please upload an image.",
                ).model_dump()
            )
            continue

        try:
            # ── Step 1: Async I/O — read file bytes without blocking ──
            image_bytes = await file.read()

            # ── Step 2: Offload CPU-bound inference to threadpool ──
            result = await run_in_threadpool(
                run_ultrasound_inference, image_bytes
            )
            result["filename"] = file.filename

            # ── Step 3: Async I/O — Upload images to database ──
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
                    "original_url": await _save_db(images_data.get("original_bytes"), "original"),
                    "mask_overlay_url": await _save_db(images_data.get("mask_overlay_bytes"), "mask_overlay"),
                    "annotated_url": await _save_db(images_data.get("annotated_bytes"), "annotated"),
                }

            # ── Forced bypass red flag ──
            if force:
                result["validation_bypassed"] = True
                result["warning"] = (
                    "Warning: Image validation was manually bypassed. "
                    "The system assumes the input is a valid ultrasound, "
                    "but results may be unreliable if it is not."
                )

            # ── Collect classification vote for consensus ──
            cls = result.get("classification", {})
            if isinstance(cls, dict) and "prediction" in cls:
                classification_votes.append({
                    "filename": file.filename,
                    "prediction": cls["prediction"],
                    "label": cls.get("label", "unknown"),
                    "confidence_pct": cls.get("confidence_pct", 0),
                    "risk_level": cls.get("risk_level", ""),
                    "ata_level": cls.get("ata_level", ""),
                    "acr_tirads_level": cls.get("acr_tirads_level", ""),
                    "clinical_recommendation": cls.get("clinical_recommendation", ""),
                    "next_step": cls.get("next_step", ""),
                })

            # ── Audit log ──
            from app.core.audit import log_audit_event

            if isinstance(cls, dict):
                log_audit_event(
                    node="image_predict",
                    action="onnx_classification",
                    result=cls.get("label", "no_detection"),
                    confidence=cls.get("confidence_pct", 0) / 100.0 if cls else None,
                    metadata={
                        "risk_level": cls.get("risk_level"),
                        "ata_level": cls.get("ata_level"),
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
                ).model_dump()
            )

    # ══════════════════════════════════════════════════════════════
    # Multi-Image Consensus Classification
    # ══════════════════════════════════════════════════════════════
    consensus = None
    if len(classification_votes) > 1:
        consensus = _compute_consensus(classification_votes)

        # Attach consensus to every individual result
        for r in results:
            if isinstance(r, dict) and r.get("status") == "success":
                r["consensus_classification"] = consensus

    elif len(classification_votes) == 1:
        # Single image — the individual result IS the consensus
        consensus = {
            "total_images_analyzed": 1,
            "consensus_source": "single_image",
            **{k: v for k, v in classification_votes[0].items() if k != "filename"},
        }

    # ── Save consensus to Patient State Manager ──
    if session_id and consensus:
        save_data = {"individual_results": results}
        save_data["consensus_classification"] = consensus
        await memory_manager.save_diagnostic(
            session_id=session_id,
            node_type="ultrasound",
            data=save_data,
        )

    return results


# ═══════════════════════════════════════════════════════════════
# Consensus Aggregation Logic
# ═══════════════════════════════════════════════════════════════

# ATA risk hierarchy for worst-case aggregation (higher index = higher risk)
_ATA_RISK_ORDER = [
    "Very Low Suspicion",
    "Low Suspicion",
    "Intermediate Suspicion",
    "High Suspicion",
]

_TIRADS_ORDER = ["TR1", "TR2", "TR3", "TR4", "TR5"]


def _compute_consensus(votes: List[dict]) -> dict:
    """
    Compute a unified consensus classification from multiple image votes.

    Clinical Aggregation Strategy (Worst-Case / Highest Suspicion Wins):
      - If ANY image is classified as suspicious → consensus is suspicious.
      - The ATA/TI-RADS level is the HIGHEST across all images.
      - Confidence is averaged across all images with the consensus label.
      - This follows the clinical principle of erring on the side of caution.

    Args:
        votes: List of classification dicts from individual images.

    Returns:
        Dict with consensus classification, source images count, and agreement stats.
    """
    total = len(votes)
    suspicious_count = sum(1 for v in votes if v["prediction"] == 1)
    benign_count = total - suspicious_count

    # ── Consensus label: worst-case wins ──
    if suspicious_count > 0:
        consensus_prediction = 1
        consensus_label = "suspicious"
        # Average confidence only from suspicious images
        suspicious_confs = [v["confidence_pct"] for v in votes if v["prediction"] == 1]
        avg_confidence = sum(suspicious_confs) / len(suspicious_confs)
    else:
        consensus_prediction = 0
        consensus_label = "benign"
        all_confs = [v["confidence_pct"] for v in votes]
        avg_confidence = sum(all_confs) / len(all_confs)

    # ── Highest ATA level ──
    best_ata_idx = -1
    best_ata = votes[0].get("ata_level", "Very Low Suspicion")
    best_tirads = votes[0].get("acr_tirads_level", "TR2")
    best_rec = votes[0].get("clinical_recommendation", "")
    best_next = votes[0].get("next_step", "")
    best_risk = votes[0].get("risk_level", "")

    for v in votes:
        ata = v.get("ata_level", "Very Low Suspicion")
        tirads = v.get("acr_tirads_level", "TR2")

        ata_idx = _ATA_RISK_ORDER.index(ata) if ata in _ATA_RISK_ORDER else -1
        if ata_idx > best_ata_idx:
            best_ata_idx = ata_idx
            best_ata = ata
            best_tirads = tirads
            best_rec = v.get("clinical_recommendation", "")
            best_next = v.get("next_step", "")
            best_risk = v.get("risk_level", "")

    agreement_pct = round(max(suspicious_count, benign_count) / total * 100, 1)

    return {
        "total_images_analyzed": total,
        "consensus_source": "multi_image_aggregation",
        "consensus_method": "highest_suspicion_wins",
        "prediction": consensus_prediction,
        "label": consensus_label,
        "confidence_pct": round(avg_confidence, 2),
        "risk_level": best_risk,
        "ata_level": best_ata,
        "acr_tirads_level": best_tirads,
        "clinical_recommendation": best_rec,
        "next_step": best_next,
        "agreement": {
            "suspicious_images": suspicious_count,
            "benign_images": benign_count,
            "agreement_pct": agreement_pct,
        },
        "needs_manual_review": agreement_pct < 100 or avg_confidence < 65,
    }

