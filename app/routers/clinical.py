"""
Phase 1 & 2 — Clinical Assessment & Medically-Driven Agentic Routing.

POST /clinical/assess
  Delegates to the clinical service for disease prediction and routing.
  Pushes results to Patient State Manager if session_id is provided.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.core.security import verify_internal_api_key
from app.schemas.clinical import ClinicalAssessmentRequest, ClinicalAssessmentResponse
from app.services.clinical_service import run_clinical_assessment

from app.core.responses import UnicodeJSONResponse

router = APIRouter(
    prefix="/clinical",
    tags=["Clinical Assessment"],
    dependencies=[Depends(verify_internal_api_key)],
    default_response_class=UnicodeJSONResponse,
)


@router.post("/assess", response_model=ClinicalAssessmentResponse)
async def assess_clinical(req: ClinicalAssessmentRequest):
    # ── Mode 2 DB Isolation Check removed ──

    """
    Run the full CDSS clinical workflow.

    Phase 1: Disease model inference (XGBoost, threadpooled).
    Phase 2: Medically-driven agentic routing based on prediction.

    If `session_id` is provided in the request, the results are
    pushed to the Patient State Manager for downstream correlation.

    Args:
        req: Validated clinical assessment request payload.

    Returns:
        ClinicalAssessmentResponse with prediction, probabilities, and routing.
    """
    try:
        result = await run_clinical_assessment(req)

        # ── Push to Dual-State Memory Manager ──
        if req.session_id:
            from app.services.memory_manager import memory_manager
            await memory_manager.save_diagnostic(
                session_id=req.session_id,
                node_type="clinical",
                data={
                    "functional_status": result.functional_status,
                    "risk_level": result.risk_level,
                    "probabilities": result.probabilities,
                    "model_confidence": result.model_confidence,
                    "clinical_recommendation": result.clinical_recommendation,
                    "next_step": result.next_step,
                    "next_step_details": result.next_step_details,
                }
            )

        return result

    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Disease model error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Disease model error: {e}",
        )
