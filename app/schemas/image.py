"""
Schemas for Image Pipeline (Nodes 3 & 4).

Clinical Standards Compliance:
  - ATA Guidelines for
    ultrasound-based risk stratification.
  - Labels use imaging-appropriate terminology ("suspicious" not
    "malignant") per ATA Guidelines lexicon.
  - Mandatory medical disclaimer on every prediction response.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Union


# ═══════════════════════════════════════════════════════════════
# Node 3: Gatekeeper Response
# ═══════════════════════════════════════════════════════════════

class ImageValidationResponse(BaseModel):
    filename: Optional[str] = None
    is_ultrasound: bool
    confidence: float = 0.0
    reason: str = ""
    status: str = "success"


# ═══════════════════════════════════════════════════════════════
# Node 4: Classification — Clinically Accurate Response
# ═══════════════════════════════════════════════════════════════

_MEDICAL_DISCLAIMER = (
    "⚕️ DISCLAIMER: This is an AI-assisted risk assessment based on "
    "ultrasound image analysis. It does NOT constitute a medical diagnosis. "
    "Definitive diagnosis requires histopathological examination (e.g., "
    "Fine Needle Aspiration Biopsy). This tool is intended to support — "
    "not replace — clinical judgment by a qualified physician."
)


class ClassificationResult(BaseModel):
    """
    Clinically validated classification output.

    Terminology follows ATA Guidelines lexicon:
      - Labels: "benign" / "suspicious" (not "malignant" — that requires histopathology)
      - Risk levels: Based on AI confidence mapped to ATA Guidelines-analogous categories
      - Recommendations: Evidence-based follow-up guidance per ATA guidelines
    """
    prediction: int = Field(
        ..., description="Raw model output: 0 = benign, 1 = suspicious"
    )
    label: str = Field(
        ...,
        description=(
            "Imaging-appropriate classification label. "
            "'benign' or 'suspicious' — NOT 'malignant' "
            "(which is a histopathological diagnosis)."
        ),
    )
    confidence_pct: float = Field(
        ...,
        description="Model confidence as a percentage (0.00 – 100.00)",
        ge=0.0,
        le=100.0,
    )
    risk_level: str = Field(
        ...,
        description=(
            "AI-estimated risk level: "
            "'Very Low Suspicion' | 'Low Suspicion' | 'Indeterminate' | "
            "'Intermediate Suspicion' | 'High Suspicion' | 'Very High Suspicion'"
        ),
    )
    ata_level: str = Field(
        ...,
        description=(
            "AI-estimated ATA (American Thyroid Association) risk level. "
            "NOTE: This is a model-derived approximation based on ultrasonographic "
            "features extracted from the segmentation mask, NOT a formal clinical assessment."
        ),
    )
    clinical_recommendation: str = Field(
        ...,
        description="Evidence-based follow-up recommendation per ATA guidelines.",
    )
    next_step: str = Field(
        ...,
        description="Actionable next step based on the clinical recommendation.",
    )
    needs_manual_review: bool = Field(
        False,
        description="True if classification confidence < 65% — physician must verify.",
    )
    radiomic_features: Optional[dict] = Field(
        None,
        description="Extracted radiomic features from the segmentation mask (shape, margin, echogenicity).",
    )


class ImageUrlsResponse(BaseModel):
    original_url: Optional[str] = Field(None, description="URL to the original ultrasound image")
    mask_overlay_url: Optional[str] = Field(None, description="URL to the mask overlaid on the original ultrasound")
    annotated_url: Optional[str] = Field(None, description="URL to the annotated image with bounding box and label")


class SegmentationInfo(BaseModel):
    """Metadata about the segmentation phase."""
    method: str = Field(
        default="U-Net ONNX",
        description="Segmentation model architecture used.",
    )
    roi_extraction: str = Field(
        default="bounding_box_crop",
        description=(
            "How the ROI was extracted from the segmentation mask. "
            "Currently: bounding-box crop. No radiomic feature extraction "
            "(margin analysis, echogenicity scoring, calcification detection) "
            "is performed at this stage."
        ),
    )


class ImagePredictionResponse(BaseModel):
    """
    Full response from the ONNX segmentation + classification pipeline.

    Includes clinically validated risk assessment, segmentation metadata,
    result images, and a mandatory medical disclaimer.
    """
    filename: Optional[str] = None
    status: str
    ai_recommendation: Optional[str] = None
    bbox: Optional[List[int]] = Field(
        None, description="Bounding box of the detected nodule [x_min, y_min, x_max, y_max]"
    )
    classification: Optional[Union[ClassificationResult, str]] = Field(
        None, description="Classification result or status message if no nodule detected"
    )
    segmentation: Optional[SegmentationInfo] = Field(
        None, description="Metadata about the segmentation method and ROI extraction"
    )
    images: Optional[ImageUrlsResponse] = None
    message: Optional[str] = None
    validation_bypassed: bool = False
    warning: Optional[str] = None
    medical_disclaimer: str = Field(
        default=_MEDICAL_DISCLAIMER,
        description="Mandatory clinical safety disclaimer included in every response.",
    )
    synthesis_report: Optional[dict] = Field(
        None, description="Optional synthesis report generated if clinical data was available in the session."
    )

