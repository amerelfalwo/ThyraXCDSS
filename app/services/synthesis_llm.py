"""
Synthesis Node - LLM Reviewer Service.

This module acts as the final decision layer (Expert Endocrinologist),
cross-referencing clinical lab values and AI ultrasound classification results
to produce a cohesive, structured final medical report.

It does NOT process raw images — only numerical/textual data.
"""

import json
import logging
from typing import Dict, Any
from pydantic import BaseModel, Field, field_validator
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.config import settings

logger = logging.getLogger(__name__)


class FinalMedicalReport(BaseModel):
    """Structured output for the LLM Synthesis Node."""
    is_consistent: bool = Field(
        ...,
        description="Whether the clinical labs and ultrasound classification agree."
    )
    corrected_classification: str = Field(
        ...,
        description="Final classification: 'Benign', 'Suspicious', 'Malignant', etc."
    )
    tumor_stage: str = Field(
        ...,
        description="TNM stage estimate (e.g. T1aN0M0) or 'N/A' if insufficient data."
    )
    comprehensive_report: str = Field(
        ...,
        description="Professional summary cross-referencing labs and imaging."
    )
    needs_manual_review: bool = Field(
        ...,
        description="True if contradictions or high uncertainty require human review."
    )

    @field_validator("is_consistent", "needs_manual_review", mode="before")
    @classmethod
    def _coerce_bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return v


def _extract_clinical_summary(clinical_data: Dict[str, Any]) -> str:
    """Extract a clean, readable summary of lab values from clinical data."""
    if not clinical_data:
        return "No clinical data available."

    lines = []
    # Direct lab values
    lab_keys = ["tsh", "t3", "t4", "free_t3", "free_t4", "tg", "anti_tpo",
                "anti_tg", "calcitonin", "calcium", "pth"]
    for key in lab_keys:
        val = clinical_data.get(key)
        if val is not None:
            lines.append(f"  {key.upper()}: {val}")

    # Risk assessment
    risk = clinical_data.get("risk_level")
    if risk:
        lines.append(f"  Clinical Risk Level: {risk}")

    next_step = clinical_data.get("next_step")
    if next_step:
        lines.append(f"  Recommended Next Step: {next_step}")

    # Interpretation text
    interp = clinical_data.get("interpretation") or clinical_data.get("assessment")
    if interp:
        lines.append(f"  Clinical Interpretation: {interp}")

    timestamp = clinical_data.get("timestamp")
    if timestamp:
        lines.append(f"  Assessment Date: {timestamp}")

    if not lines:
        # Fallback: dump all keys
        for k, v in clinical_data.items():
            if k != "timestamp" and v is not None:
                lines.append(f"  {k}: {v}")

    return "\n".join(lines) if lines else "No clinical lab values found."


def _extract_ultrasound_summary(us_data: Dict[str, Any]) -> str:
    """Extract classification results and numerical scores from ultrasound data."""
    if not us_data:
        return "No ultrasound data available."

    lines = []

    cls = us_data.get("classification", {})
    if isinstance(cls, dict):
        label = cls.get("label", "Unknown")
        confidence = cls.get("confidence_pct", "N/A")
        risk = cls.get("risk_level", "N/A")
        tirads = cls.get("acr_tirads_level", "N/A")
        recommendation = cls.get("clinical_recommendation", "")

        lines.append(f"  AI Classification: {label}")
        lines.append(f"  Confidence: {confidence}%")
        lines.append(f"  Risk Level: {risk}")
        lines.append(f"  ACR TI-RADS Level: {tirads}")
        if recommendation:
            lines.append(f"  System Recommendation: {recommendation}")

        radiomic_features = cls.get("radiomic_features")
        if radiomic_features:
            lines.append("  Radiomic Features:")
            for feature, value in radiomic_features.items():
                if isinstance(value, float):
                    lines.append(f"    - {feature}: {value:.2f}")
                else:
                    lines.append(f"    - {feature}: {value}")
    elif cls:
        lines.append(f"  Raw Classification: {cls}")

    # Bounding box info (size indicator)
    bbox = us_data.get("bbox")
    if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        lines.append(f"  Nodule Bounding Box: {w}x{h} pixels")

    # Segmentation summary (area, not image)
    seg = us_data.get("segmentation", {})
    if isinstance(seg, dict):
        area = seg.get("area_pct") or seg.get("area")
        if area is not None:
            lines.append(f"  Segmentation Area: {area}")

    ai_rec = us_data.get("ai_recommendation")
    if ai_rec:
        lines.append(f"  AI Recommendation: {ai_rec}")

    return "\n".join(lines) if lines else "No ultrasound classification data found."


def _extract_fnac_summary(fnac_data: Dict[str, Any]) -> str:
    """Extract cytopathology classification results from FNAC data."""
    if not fnac_data:
        return "No FNAC data available."

    lines = []
    
    cls = fnac_data.get("classification", fnac_data)
    if isinstance(cls, dict):
        label = cls.get("bethesda_label", "Unknown")
        risk = cls.get("malignancy_risk", "N/A")
        confidence = cls.get("confidence_pct", "N/A")
        
        lines.append(f"  Bethesda Classification: {label}")
        lines.append(f"  Malignancy Risk: {risk}")
        if confidence != "N/A":
            lines.append(f"  Confidence: {confidence:.2f}%" if isinstance(confidence, float) else f"  Confidence: {confidence}%")
        
        rec = cls.get("recommendation")
        if rec:
            lines.append(f"  Recommendation: {rec}")
            
    return "\n".join(lines) if lines else "No FNAC classification data found."


async def generate_final_report(
    clinical_data: Dict[str, Any],
    ultrasound_data: Dict[str, Any],
    fnac_data: Dict[str, Any] = None,
) -> FinalMedicalReport:
    """
    Cross-references clinical lab values, AI ultrasound classification results,
    and FNAC cytopathology data to generate a final structured medical report.

    This function works with NUMBERS and TEXT only — no image processing.
    """
    try:
        llm = ChatGroq(
            temperature=0.1,
            model="llama-3.3-70b-versatile",
            max_tokens=4096,
            api_key=settings.GROQ_API_KEY,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        json_schema = (
            "You MUST respond with ONLY a JSON object matching this exact schema:\n"
            "{\n"
            '  "is_consistent": <boolean true or false>,\n'
            '  "corrected_classification": "<string>",\n'
            '  "tumor_stage": "<string e.g. T1aN0M0 or N/A>",\n'
            '  "comprehensive_report": "<string>",\n'
            '  "needs_manual_review": <boolean true or false>\n'
            "}\n"
            "IMPORTANT: is_consistent and needs_manual_review MUST be JSON booleans "
            "(true/false), NOT strings.\n"
        )

        system_prompt = (
            "You are an Expert Endocrinologist and Thyroid Oncologist.\n\n"
            "YOUR TASK: Cross-reference the patient's clinical lab results, "
            "AI ultrasound classification output, and FNAC cytopathology results (if available) to produce a precise clinical synthesis.\n\n"
            "ANALYSIS RULES:\n"
            "1. Compare thyroid function labs (TSH, T3, T4, Free T3/T4) against the "
            "ultrasound classification (TI-RADS level, risk level, confidence score, and extracted Radiomic Features) and FNAC Bethesda classification.\n"
            "2. If labs are normal but imaging shows high TI-RADS (≥4) or FNAC shows Bethesda IV-VI, flag inconsistency "
            "and recommend appropriate next steps.\n"
            "3. If labs show hypothyroidism/hyperthyroidism AND imaging/FNAC is suspicious, "
            "escalate risk classification.\n"
            "4. Estimate TNM tumor stage based on:\n"
            "   - Nodule size (from bounding box or description)\n"
            "   - Classification aggressiveness from Ultrasound and FNAC\n"
            "   - Lab markers (elevated calcitonin = medullary carcinoma concern)\n"
            "5. Set needs_manual_review=true if:\n"
            "   - TI-RADS ≥ 4 with confidence < 70%\n"
            "   - Labs, imaging, and FNAC contradict each other significantly\n"
            "   - Calcitonin is elevated\n"
            "   - Classification is 'Malignant' regardless of confidence\n"
            "6. Provide an INTERPRETIVE REPORT: In your `comprehensive_report`, explicitly mention how the extracted radiomic features (e.g., taller-than-wide shape, margin irregularity/solidity, echogenicity) support the final TI-RADS level and risk assessment. Explain *why* a specific classification or stage was chosen based on these features and the clinical context.\n\n"
            "Be precise and data-driven. Reference specific numerical values and stages in your report. Do not reference raw image data or masks.\n\n"
            + json_schema
        )

        # Build structured, clean data summary
        clinical_summary = _extract_clinical_summary(clinical_data)
        ultrasound_summary = _extract_ultrasound_summary(ultrasound_data)
        fnac_summary = _extract_fnac_summary(fnac_data or {})

        user_text = (
            "Cross-reference the following data and generate the final synthesis report.\n\n"
            f"=== CLINICAL LAB RESULTS ===\n{clinical_summary}\n\n"
            f"=== ULTRASOUND AI CLASSIFICATION ===\n{ultrasound_summary}\n\n"
            f"=== FNAC CYTOPATHOLOGY ===\n{fnac_summary}"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_text),
        ]

        ai_message = await llm.ainvoke(messages)
        raw = json.loads(ai_message.content)
        result = FinalMedicalReport(**raw)

        logger.info(
            f"Synthesis complete: consistent={result.is_consistent}, "
            f"classification={result.corrected_classification}, "
            f"stage={result.tumor_stage}, review={result.needs_manual_review}"
        )
        return result

    except Exception as e:
        logger.error(f"Error generating final report in synthesis LLM: {e}")
        return FinalMedicalReport(
            is_consistent=False,
            corrected_classification="Unknown",
            tumor_stage="Unknown",
            comprehensive_report=f"System error during synthesis: {str(e)}",
            needs_manual_review=True,
        )
