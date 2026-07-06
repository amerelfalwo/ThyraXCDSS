"""
Clinical Assessment Service.

Encapsulates all business logic for Phase 1 (disease prediction)
and Phase 2 (medically-driven agentic routing).  Routers call into
this module instead of implementing logic inline.

Brand Guideline Compliance:
  - Models cached in memory.
  - MLflow local fallback via core.model_loader.
  - Threadpooled CPU-bound inference.
"""

import gc
import logging

from app.core.inference import run_clinical_inference
from app.core.model_loader import load_production_model
from app.schemas.clinical import ClinicalAssessmentRequest, ClinicalAssessmentResponse

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────
LABEL_MAP = {0: "normal", 1: "hypothyroid", 2: "hyperthyroid"}

FEATURE_NAMES = [
    "TT4", "TSH", "T3", "FTI", "T4U",
    "age", "on_thyroxine", "thyroid_surgery", "query_hyperthyroid",
]


async def route_clinical_decision(
    functional_status: str,
    nodule_present: bool,
    model_confidence: float,
) -> dict:
    """
    Medically-driven routing based on disease model output.

    Attempts to enhance the recommendation with an LLM-generated explanation.
    Falls back to deterministic rules if the LLM is unavailable.

    Args:
        functional_status: Predicted thyroid status (normal / hypothyroid / hyperthyroid).
        nodule_present: Whether a palpable nodule was detected.
        model_confidence: Max probability from the model (0.0–1.0).

    Returns:
        Dict with risk_level, recommendation, next_step, next_step_details,
        and optional ai_recommendation.
    """
    # ── Rule-based routing ──
    if functional_status == "hyperthyroid":
        if nodule_present:
            base_recommendation = (
                "Patient shows signs of HYPERTHYROIDISM with a PALPABLE NODULE. "
                "The recommended next step is a Radionuclide (Iodine-123) Scan "
                "to evaluate for an autonomously functioning thyroid nodule (Hot Nodule). "
                "Hot nodules are RARELY malignant (<1% risk). "
                "Cancer workup is NOT immediately indicated unless cold nodules are "
                "identified on the scan."
            )
            result = {
                "risk_level": "moderate",
                "recommendation": base_recommendation,
                "next_step": "radionuclide_scan",
                "next_step_details": {
                    "action": "Order Radionuclide Scan (I-123 uptake)",
                    "rationale": "Differentiate hot vs. cold nodules in hyperthyroid state",
                    "cancer_pipeline_triggered": False,
                    "urgency": "routine",
                },
            }
        else:
            base_recommendation = (
                "Patient shows signs of HYPERTHYROIDISM with NO palpable nodule. "
                "The recommended next step is to evaluate etiology (e.g., Graves' disease "
                "or thyroiditis) via TSH Receptor Antibodies (TRAb) or a generic uptake scan. "
                "Since no structural nodules are present, cancer workup is NOT indicated."
            )
            result = {
                "risk_level": "low",
                "recommendation": base_recommendation,
                "next_step": "biochemical_workup",
                "next_step_details": {
                    "action": "Order TRAb / evaluate etiology",
                    "rationale": "Hyperthyroidism without discrete nodules suggests autoimmune or systemic etiology",
                    "cancer_pipeline_triggered": False,
                    "urgency": "routine",
                },
            }

    elif functional_status in ("hypothyroid", "normal") and nodule_present:
        risk = "high" if functional_status == "hypothyroid" else "elevated"
        status_label = (
            "HYPOTHYROID" if functional_status == "hypothyroid" else "EUTHYROID (normal)"
        )
        base_recommendation = (
            f"Patient is {status_label} with a PALPABLE NODULE detected on physical "
            f"examination. Cold nodules in {functional_status} patients carry a "
            f"HIGHER malignancy risk (5-15%). The recommended next step is a "
            f"HIGH-RESOLUTION THYROID ULTRASOUND to evaluate the nodule "
            f"characteristics per ACR TI-RADS criteria."
        )
        result = {
            "risk_level": risk,
            "recommendation": base_recommendation,
            "next_step": "upload_ultrasound",
            "next_step_details": {
                "action": "Upload thyroid ultrasound image for AI analysis",
                "endpoint": "/predict/image",
                "rationale": (
                    "Evaluate cold nodule for malignancy using "
                    "segmentation + classification pipeline"
                ),
                "cancer_pipeline_triggered": True,
                "urgency": "priority",
            },
        }

    else:
        # normal + no nodule
        base_recommendation = (
            "Patient thyroid function is NORMAL with NO palpable nodule detected. "
            "No immediate imaging is required. Recommend routine clinical follow-up "
            "with repeat thyroid function tests in 6-12 months, or sooner if symptoms "
            "develop."
        )
        result = {
            "risk_level": "low",
            "recommendation": base_recommendation,
            "next_step": "routine_followup",
            "next_step_details": {
                "action": "Schedule follow-up in 6-12 months",
                "rationale": "Normal function, no structural abnormality",
                "cancer_pipeline_triggered": False,
                "urgency": "routine",
            },
        }

    # ── LLM explanation removed for speed ──
    # Node 1 & 2 return raw ML + rule-based decisions instantly.
    # LLM generation is deferred to later nodes.

    return result


# ═══════════════════════════════════════════════════════════════
# Main Assessment Orchestrator (Node 1 + Node 2)
# ═══════════════════════════════════════════════════════════════

async def run_clinical_assessment(
    req: ClinicalAssessmentRequest,
) -> ClinicalAssessmentResponse:
    """
    Full CDSS clinical workflow:
      Node 1: Load cached XGBoost model, run inference in a threadpool (CPU-bound).
      Node 2: Route the patient based on clinical rules + LLM advice.

    Args:
        req: Validated clinical assessment request.

    Returns:
        ClinicalAssessmentResponse with prediction + routing.

    Raises:
        RuntimeError: If model loading or inference fails.
    """
    import pandas as pd

    feature_values = [
        req.TT4, req.TSH, req.T3, req.FTI, req.T4U,
        req.age, req.on_thyroxine, req.thyroid_surgery, req.query_hyperthyroid,
    ]

    # ── Node 1: Lazy-load model with MLflow local fallback ──
    model = load_production_model("ThyraX_Disease_Classifier")
    df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)

    # Run inference in threadpool to avoid blocking the event loop
    pred, probs = await run_clinical_inference(model, df)

    # ── Models are cached in memory for speed ──
    del df

    prob_dict = {LABEL_MAP[i]: float(probs[i]) for i in range(len(probs))}
    functional_status = LABEL_MAP[pred]
    max_confidence = float(max(probs))
    needs_review = max_confidence < 0.65

    # ── Node 2: Medically-driven agentic routing ──
    routing = await route_clinical_decision(
        functional_status,
        req.nodule_present,
        max_confidence,
    )

    # ── Audit Log ──
    from app.core.audit import log_audit_event
    log_audit_event(
        node="clinical_assess",
        action="xgboost_prediction",
        result=functional_status,
        confidence=max_confidence,
        metadata={
            "probabilities": prob_dict,
            "needs_manual_review": needs_review,
            "risk_level": routing["risk_level"],
        },
    )

    return ClinicalAssessmentResponse(
        status="success",
        functional_status=functional_status,
        probabilities=prob_dict,
        model_confidence=max_confidence,
        needs_manual_review=needs_review,
        risk_level=routing["risk_level"],
        clinical_recommendation=routing["recommendation"],
        ai_recommendation=routing.get("ai_recommendation"),
        next_step=routing["next_step"],
        next_step_details=routing["next_step_details"],
    )
