"""
Phase 1 & 2 — Clinical Assessment & Medically-Driven Agentic Routing.

POST /clinical/assess
  Delegates to the clinical service for disease prediction and routing.
  Pushes results to Patient State Manager if session_id is provided.
"""

from fastapi import APIRouter, Depends, HTTPException
from app.core.database import get_db
from app.core.security import verify_internal_api_key
from app.schemas.clinical import ClinicalAssessmentRequest, ClinicalAssessmentResponse
from app.services.clinical_service import run_clinical_assessment
from app.schemas.memory_models import Session as SessionModel, Patient

from app.core.responses import UnicodeJSONResponse

router = APIRouter(
    prefix="/clinical",
    tags=["Clinical Assessment"],
    dependencies=[Depends(verify_internal_api_key)],
    default_response_class=UnicodeJSONResponse,
)


@router.post("/assess", response_model=ClinicalAssessmentResponse)
async def assess_clinical(req: ClinicalAssessmentRequest, db: AsyncSession = Depends(get_db)):
    # ── Mode 2 DB Isolation Check ──
    from app.core.security import verify_doctor_session_ownership
    await verify_doctor_session_ownership(
        session_id=req.session_id,
        doctor_id=req.doctor_id,
        patient_id=req.patient_id,
        db=db
    )

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
    import asyncio

    MAX_RETRIES = 3
    RETRY_DELAYS = [5, 15, 30]
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
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
                    },
                    doctor_id=req.doctor_id
                )

            return result

        except Exception as e:
            last_error = e
            error_str = str(e)
            is_transient = any(
                code in error_str
                for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "rate_limit")
            )
            if is_transient and attempt < MAX_RETRIES - 1:
                import logging
                logging.getLogger(__name__).warning(
                    f"LLM transient error on /clinical/assess "
                    f"(attempt {attempt + 1}/{MAX_RETRIES}), "
                    f"retrying in {RETRY_DELAYS[attempt]}s"
                )
                await asyncio.sleep(RETRY_DELAYS[attempt])
                continue
            break

    is_overload = any(
        code in str(last_error)
        for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "rate_limit")
    )
    raise HTTPException(
        status_code=503 if is_overload else 500,
        detail=(
            "The AI model is currently experiencing high demand. Please try again in a moment."
            if is_overload
            else f"Disease model error: {last_error}"
        ),
    )
