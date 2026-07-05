"""
Synthesis Node — LLM Reviewer Service (Node 7).

Accepts outputs from ALL preceding nodes (1-6) and cross-references them
to produce a unified, authoritative clinical report.

Node inputs (all optional, pulled from session diagnostic_context):
    - clinical   → Node 1+2  XGBoost assessment
    - ultrasound → Node 3+4  ONNX gatekeeper + segmentation / TI-RADS
    - fnac       → Node 6    Bethesda cytopathology
    - agent_chat → Node 5    last AI-assistant exchange (context only)

No raw image data is sent to the LLM — only numerical / textual features.
"""

import json
import logging
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Output Schema
# ═══════════════════════════════════════════════════════════════

class FinalMedicalReport(BaseModel):
    """Structured synthesis report produced by the Node 7 LLM reviewer."""

    # ── Core judgements ──────────────────────────────────────
    is_consistent: bool = Field(
        ...,
        description="True when all available node outputs agree; False on contradiction.",
    )
    corrected_classification: str = Field(
        ...,
        description=(
            "Final authoritative classification: "
            "'Benign', 'Suspicious', 'Malignant', 'Indeterminate', etc."
        ),
    )
    tumor_stage: str = Field(
        ...,
        description="Best-effort TNM stage estimate (e.g. T1aN0M0) or 'N/A'.",
    )
    needs_manual_review: bool = Field(
        ...,
        description=(
            "True when contradictions, low confidence, or high-risk findings "
            "require a human expert review before clinical action."
        ),
    )

    # ── Evidence cross-reference ─────────────────────────────
    nodes_available: list[str] = Field(
        default_factory=list,
        description="List of nodes whose data was included in this synthesis.",
    )
    comprehensive_report: str = Field(
        ...,
        description=(
            "Professional narrative that explicitly cites lab values, "
            "TI-RADS level, radiomic features, Bethesda category, and "
            "AI-chat insights to justify the final classification and stage."
        ),
    )
    recommended_next_steps: str = Field(
        default="",
        description=(
            "Concrete clinical actions: e.g. 'Surgical referral', "
            "'Repeat ultrasound in 6 months', 'Molecular testing'."
        ),
    )

    # ── Bool coercion (robustness against LLM string output) ─
    @field_validator("is_consistent", "needs_manual_review", mode="before")
    @classmethod
    def _coerce_bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return bool(v)


# ═══════════════════════════════════════════════════════════════
# Data extractors — one per node type
# ═══════════════════════════════════════════════════════════════

def _extract_clinical_summary(data: Dict[str, Any]) -> str:
    if not data:
        return "(no data)"

    lines: list[str] = []
    lab_keys = [
        "tsh", "t3", "t4", "free_t3", "free_t4",
        "tg", "anti_tpo", "anti_tg", "calcitonin", "calcium", "pth",
    ]
    for key in lab_keys:
        val = data.get(key)
        if val is not None:
            lines.append(f"  {key.upper()}: {val}")

    for label, keys in [
        ("Risk Level", ["risk_level"]),
        ("Functional Status", ["functional_status"]),
        ("Model Confidence", ["model_confidence"]),
        ("Clinical Recommendation", ["clinical_recommendation"]),
        ("Next Step", ["next_step"]),
        ("Interpretation", ["interpretation", "assessment"]),
    ]:
        for k in keys:
            val = data.get(k)
            if val:
                lines.append(f"  {label}: {val}")
                break

    if not lines:
        for k, v in data.items():
            if k != "timestamp" and v is not None:
                lines.append(f"  {k}: {v}")

    return "\n".join(lines) or "(no lab values found)"


def _extract_ultrasound_summary(data: Dict[str, Any]) -> str:
    if not data:
        return "(no data)"

    lines: list[str] = []

    cls = data.get("classification", {})
    if isinstance(cls, dict):
        lines.append(f"  AI Classification : {cls.get('label', 'Unknown')}")
        lines.append(f"  Confidence        : {cls.get('confidence_pct', 'N/A')}%")
        lines.append(f"  Risk Level        : {cls.get('risk_level', 'N/A')}")
        lines.append(f"  ACR TI-RADS       : {cls.get('acr_tirads_level', 'N/A')}")
        rec = cls.get("clinical_recommendation")
        if rec:
            lines.append(f"  System Rec.       : {rec}")

        rf = cls.get("radiomic_features")
        if rf and isinstance(rf, dict):
            lines.append("  Radiomic Features :")
            for feat, val in rf.items():
                fmt_val = f"{val:.3f}" if isinstance(val, float) else val
                lines.append(f"    - {feat}: {fmt_val}")
    elif cls:
        lines.append(f"  Raw Classification: {cls}")

    bbox = data.get("bbox")
    if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        lines.append(
            f"  Nodule Size       : {bbox[2] - bbox[0]}×{bbox[3] - bbox[1]} px"
        )

    seg = data.get("segmentation", {})
    if isinstance(seg, dict):
        area = seg.get("area_pct") or seg.get("area")
        if area is not None:
            lines.append(f"  Segmentation Area : {area}")

    ai_rec = data.get("ai_recommendation")
    if ai_rec:
        lines.append(f"  AI Recommendation : {ai_rec}")

    return "\n".join(lines) or "(no ultrasound features found)"


def _extract_fnac_summary(data: Dict[str, Any]) -> str:
    if not data:
        return "(no data)"

    lines: list[str] = []
    # data may wrap the payload under a "classification" key
    cls = data.get("classification", data)
    if isinstance(cls, dict):
        label = cls.get("bethesda_label") or cls.get("label", "Unknown")
        risk = cls.get("malignancy_risk", "N/A")
        conf = cls.get("confidence_pct", "N/A")
        rec = cls.get("recommendation")

        lines.append(f"  Bethesda Category : {label}")
        lines.append(f"  Malignancy Risk   : {risk}")
        if conf != "N/A":
            conf_str = f"{conf:.2f}" if isinstance(conf, float) else conf
            lines.append(f"  Confidence        : {conf_str}%")
        if rec:
            lines.append(f"  Recommendation    : {rec}")

    return "\n".join(lines) or "(no FNAC data found)"


def _extract_agent_chat_summary(data: Dict[str, Any]) -> str:
    """Extract the most recent AI-assistant insight from Node 5."""
    if not data:
        return "(no data)"

    lines: list[str] = []

    # chat router stores last exchange under "last_response" or similar
    last_response = data.get("last_response") or data.get("response")
    if last_response:
        snippet = str(last_response)[:600]
        lines.append(f"  Last AI Response  : {snippet}")

    query = data.get("last_query") or data.get("query")
    if query:
        lines.append(f"  Last Query        : {query}")

    return "\n".join(lines) or "(no agent-chat context found)"


# ═══════════════════════════════════════════════════════════════
# Main synthesis function
# ═══════════════════════════════════════════════════════════════

async def generate_final_report(
    clinical_data:   Dict[str, Any],
    ultrasound_data: Dict[str, Any],
    fnac_data:       Optional[Dict[str, Any]] = None,
    agent_chat_data: Optional[Dict[str, Any]] = None,
) -> FinalMedicalReport:
    """
    Cross-reference all node outputs and produce the final structured report.

    Parameters:
        clinical_data   – Node 1+2 XGBoost assessment dict.
        ultrasound_data – Node 3+4 ONNX results dict.
        fnac_data       – Node 6 Bethesda cytopathology dict (optional).
        agent_chat_data – Node 5 last AI-assistant exchange (optional, context).

    Returns:
        FinalMedicalReport Pydantic model.
    """
    # ── Detect which nodes contributed data ──
    nodes_available: list[str] = []
    if clinical_data:
        nodes_available.append("Node 1+2 (Clinical Assessment)")
    if ultrasound_data:
        nodes_available.append("Node 3+4 (Ultrasound / TI-RADS)")
    if fnac_data:
        nodes_available.append("Node 6 (FNAC / Bethesda)")
    if agent_chat_data:
        nodes_available.append("Node 5 (AI Assistant Chat)")

    # ── Build human-readable sections ──
    clinical_summary   = _extract_clinical_summary(clinical_data)
    ultrasound_summary = _extract_ultrasound_summary(ultrasound_data)
    fnac_summary       = _extract_fnac_summary(fnac_data or {})
    chat_summary       = _extract_agent_chat_summary(agent_chat_data or {})

    # ── JSON schema instruction ──
    json_schema = (
        "You MUST respond with ONLY a valid JSON object matching this exact schema:\n"
        "{\n"
        '  "is_consistent": <boolean>,\n'
        '  "corrected_classification": "<string>",\n'
        '  "tumor_stage": "<string, e.g. T1aN0M0 or N/A>",\n'
        '  "needs_manual_review": <boolean>,\n'
        '  "comprehensive_report": "<string>",\n'
        '  "recommended_next_steps": "<string>"\n'
        "}\n"
        "IMPORTANT: is_consistent and needs_manual_review MUST be JSON booleans "
        "(true/false), NOT strings. Do not wrap the JSON in markdown fences.\n"
    )

    # ── System prompt ──
    system_prompt = (
        "You are an Expert Endocrinologist and Thyroid Oncologist acting as the "
        "final synthesis layer of an AI-powered Clinical Decision Support System.\n\n"
        "YOUR TASK:\n"
        "Cross-reference ALL available diagnostic data from up to 6 AI nodes and "
        "produce the definitive clinical synthesis for this patient.\n\n"
        "ANALYSIS RULES:\n"
        "1. Compare thyroid function labs (TSH, T3, T4, Free T3/T4) against "
        "ultrasound AI classification (TI-RADS level, risk, confidence, radiomic "
        "features) and FNAC Bethesda category.\n"
        "2. Flag `is_consistent = false` and explain contradictions if:\n"
        "   - Normal labs + high TI-RADS (≥4) or Bethesda IV-VI.\n"
        "   - Benign ultrasound + Bethesda V-VI cytology.\n"
        "   - Clinical model says high-risk but imaging says benign.\n"
        "3. Escalate risk classification when:\n"
        "   - Hypothyroidism/hyperthyroidism + suspicious imaging/FNAC.\n"
        "   - Elevated calcitonin (medullary carcinoma risk).\n"
        "4. Estimate TNM stage from:\n"
        "   - Nodule size (bounding box pixels or description).\n"
        "   - Classification aggressiveness across all nodes.\n"
        "   - Elevated calcitonin → medullary carcinoma consideration.\n"
        "5. Set `needs_manual_review = true` if:\n"
        "   - TI-RADS ≥ 4 AND confidence < 70%.\n"
        "   - Labs, imaging, and FNAC significantly contradict each other.\n"
        "   - Calcitonin is elevated.\n"
        "   - Final classification is 'Malignant' regardless of confidence.\n"
        "6. In `comprehensive_report`, explicitly cite:\n"
        "   - Specific lab values and their clinical meaning.\n"
        "   - TI-RADS level and the radiomic features that support it.\n"
        "   - Bethesda category and its malignancy risk percentage.\n"
        "   - Any AI-chat insights that added clinical context.\n"
        "   - WHY a specific classification and stage were chosen.\n"
        "7. In `recommended_next_steps`, provide concrete clinical guidance "
        "(e.g., 'Surgical referral for total thyroidectomy', "
        "'FNA biopsy under ultrasound guidance', "
        "'Follow-up ultrasound in 6 months').\n\n"
        "Be precise and data-driven. Do NOT reference raw pixel data or masks.\n\n"
        + json_schema
    )

    # ── User message ──
    node_block = "  " + "\n  ".join(nodes_available) if nodes_available else "  (none)"
    user_text = (
        "Nodes that contributed data to this synthesis:\n"
        f"{node_block}\n\n"
        "Cross-reference the data below and generate the final synthesis report.\n\n"
        f"=== NODE 1+2 — CLINICAL ASSESSMENT ===\n{clinical_summary}\n\n"
        f"=== NODE 3+4 — ULTRASOUND / TI-RADS ===\n{ultrasound_summary}\n\n"
        f"=== NODE 6   — FNAC / BETHESDA CYTOPATHOLOGY ===\n{fnac_summary}\n\n"
        f"=== NODE 5   — AI ASSISTANT CHAT (context) ===\n{chat_summary}"
    )

    try:
        llm = ChatGroq(
            temperature=0.1,
            model="llama-3.3-70b-versatile",
            max_tokens=4096,
            api_key=settings.GROQ_API_KEY,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_text),
        ]

        ai_message = await llm.ainvoke(messages)
        raw = json.loads(ai_message.content)

        result = FinalMedicalReport(
            **raw,
            nodes_available=nodes_available,
        )

        logger.info(
            "Synthesis complete | nodes=%s | consistent=%s | "
            "classification=%s | stage=%s | review=%s",
            len(nodes_available),
            result.is_consistent,
            result.corrected_classification,
            result.tumor_stage,
            result.needs_manual_review,
        )
        return result

    except Exception as exc:
        logger.error("Synthesis LLM error: %s", exc, exc_info=True)
        return FinalMedicalReport(
            is_consistent=False,
            corrected_classification="Unknown",
            tumor_stage="N/A",
            needs_manual_review=True,
            nodes_available=nodes_available,
            comprehensive_report=f"System error during synthesis: {exc}",
            recommended_next_steps="Manual clinical review required.",
        )
