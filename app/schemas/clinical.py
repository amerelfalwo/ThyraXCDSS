from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ClinicalAssessmentRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID to link results to the patient's diagnostic journey.",
    )
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    on_thyroxine: int = Field(..., ge=0, le=1, description="Is patient on thyroxine? 1=yes, 0=no")
    thyroid_surgery: int = Field(..., ge=0, le=1, description="History of thyroid surgery? 1=yes, 0=no")
    query_hyperthyroid: int = Field(..., ge=0, le=1, description="Clinical suspicion of hyperthyroidism? 1=yes, 0=no")
    TSH: float = Field(..., ge=0, description="Thyroid Stimulating Hormone (µIU/mL)")
    T3: float = Field(..., ge=0, description="Triiodothyronine (ng/mL)")
    TT4: float = Field(..., ge=0, description="Total T4 (µg/dL)")
    FTI: float = Field(..., ge=0, description="Free Thyroxine Index")
    T4U: float = Field(..., ge=0, description="T4 Uptake")
    nodule_present: bool = Field(..., description="Palpable nodule detected? true=yes, false=no")

class ClinicalAssessmentResponse(BaseModel):
    status: str

    # ── Disease model output (Node 1) ──
    functional_status: str
    probabilities: Dict[str, float]

    # ── Confidence guard ──
    model_confidence: float = Field(
        ..., description="Max probability from the XGBoost model (0.0–1.0)"
    )
    needs_manual_review: bool = Field(
        False,
        description="True if model confidence < 0.65 — physician must verify manually",
    )

    # ── Agentic routing (Node 2) ──
    risk_level: str
    clinical_recommendation: str
    next_step: str
    next_step_details: Dict[str, Any]
