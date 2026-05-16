"""
Schemas for FNAC Cytopathology Pipeline.

Aligned with The Bethesda System for Reporting Thyroid Cytopathology (2023).

Reference:
  Cibas ES, Ali SZ. The 2023 Bethesda System for Reporting Thyroid
  Cytopathology. Thyroid. 2023;33(9):1039-1044.

Categories:
  I   — Non-diagnostic / Unsatisfactory
  II  — Benign
  III — Atypia of Undetermined Significance (AUS) / Follicular Lesion (FLUS)
  IV  — Follicular Neoplasm / Suspicious for Follicular Neoplasm (FN/SFN)
  V   — Suspicious for Malignancy
  VI  — Malignant
"""

from pydantic import BaseModel, Field
from typing import Optional

_FNAC_DISCLAIMER = (
    "⚕️ DISCLAIMER: This is an AI-assisted cytopathological risk assessment. "
    "It does NOT constitute a definitive pathological diagnosis. "
    "Final diagnosis must be made by a board-certified cytopathologist "
    "after reviewing the original specimen slides. This tool is intended "
    "to support — not replace — expert pathological evaluation."
)


class FnacPredictionRequest(BaseModel):
    """Request for FNAC image classification."""
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID to link this result to the patient's diagnostic journey.",
    )


class FnacClassificationResult(BaseModel):
    """
    FNAC classification output aligned with The Bethesda System.
    """
    prediction: int = Field(
        ..., description="Raw model output class index (0–5 mapping to Bethesda I–VI)"
    )
    bethesda_category: str = Field(
        ..., description="Bethesda category (I through VI)"
    )
    bethesda_label: str = Field(
        ...,
        description=(
            "Full Bethesda label, e.g. "
            "'Bethesda IV — Follicular Neoplasm / Suspicious for FN'"
        ),
    )
    confidence_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Model confidence as a percentage (0.00 – 100.00)",
    )
    malignancy_risk: str = Field(
        ...,
        description="Implied malignancy risk range per Bethesda guidelines (e.g. '15–30%')",
    )
    recommendation: str = Field(
        ..., description="Evidence-based clinical recommendation per Bethesda category."
    )
    needs_manual_review: bool = Field(
        False,
        description="True if confidence < 65% — cytopathologist must verify.",
    )


class FnacPredictionResponse(BaseModel):
    """Full response from the FNAC cytopathology classification pipeline."""
    filename: Optional[str] = None
    status: str
    ai_recommendation: Optional[str] = None
    classification: Optional[FnacClassificationResult] = None
    message: Optional[str] = None
    session_id: Optional[str] = None
    medical_disclaimer: str = Field(
        default=_FNAC_DISCLAIMER,
        description="Mandatory clinical safety disclaimer.",
    )
