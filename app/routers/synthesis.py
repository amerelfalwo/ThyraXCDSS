"""
Synthesis Router — Node 7.

GET  /synthesis/review/{session_id}   → Full synthesis from all 6 nodes.
GET  /synthesis/image/{image_id}      → Retrieve a stored diagnostic image.

The endpoint reads ALL diagnostic data saved by Nodes 1-6 under the given
session_id and passes them to the LLM synthesis service. No file uploads —
everything flows through the shared session state.
"""

import logging
from fastapi import APIRouter, HTTPException, status

from app.services.memory_manager import memory_manager
from app.services.synthesis_llm import generate_final_report

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/synthesis", tags=["Synthesis Node — Node 7"])


# ═══════════════════════════════════════════════════════════════
# Node 7: POST /synthesis/review/{session_id}
# ═══════════════════════════════════════════════════════════════

@router.post(
    "/review/{session_id}",
    summary="Run Node 7 synthesis using all prior node results for this session",
    response_description=(
        "Unified final medical report cross-referencing "
        "Node 1+2 (clinical), Node 3+4 (ultrasound), "
        "Node 5 (agent chat), and Node 6 (FNAC)."
    ),
)
async def synthesis_review(session_id: str):
    """
    **Node 7 — Final Clinical Synthesis.**

    Reads the diagnostic context stored by all previous nodes under
    `session_id` and asks the LLM to produce an authoritative, cross-referenced
    medical report.

    Partial sessions are supported: if only some nodes have run,
    the LLM will note which data sources were available and work with
    whatever is present.

    Returns a `FinalMedicalReport` JSON plus the `session_id`.
    Raises:
        - **404** if the session does not exist in the database.
        - **400** if the session has no diagnostic data at all.
        - **500** on LLM or unexpected errors.
    """
    # ── 1. Load session context from the shared DB ──────────────
    try:
        session_ctx = await memory_manager.load_context(session_id)
    except Exception as exc:
        logger.error("Failed to load session %s: %s", session_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load session: {exc}",
        )

    if not session_ctx:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    diag = session_ctx.diagnostic_context or {}

    # ── 2. Extract per-node data ─────────────────────────────────
    clinical_data    = diag.get("clinical", {})
    ultrasound_data  = diag.get("ultrasound", {})
    fnac_data        = diag.get("fnac", {})
    agent_chat_data  = diag.get("agent_chat", {})   # Node 5 — optional

    # Fallback: if Node 5 never ran save_diagnostic("agent_chat"), but there IS
    # conversation history or a session summary available, build a synthetic
    # snapshot so Node 7 still benefits from any clinical chat context.
    if not agent_chat_data:
        chat_history = session_ctx.chat_history or []
        session_summary = session_ctx.session_summary or ""
        synthetic_parts: list[str] = []
        if session_summary:
            synthetic_parts.append(f"Session summary: {session_summary}")
        if chat_history:
            # Take the two most recent AI responses as context snippets
            ai_msgs = [
                m.get("content", "") for m in chat_history
                if isinstance(m, dict) and m.get("role") == "assistant"
            ]
            if ai_msgs:
                synthetic_parts.append(
                    f"Recent AI insight: {ai_msgs[-1][:600]}"
                )
        if synthetic_parts:
            agent_chat_data = {
                "last_response": " | ".join(synthetic_parts),
                "source": "session_history_fallback",
            }

    # Guard: at least one node must have produced results
    if not any([clinical_data, ultrasound_data, fnac_data, agent_chat_data]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"No diagnostic data found for session '{session_id}'. "
                "Run at least one of the preceding nodes (1-6) with this session_id first."
            ),
        )

    # ── 3. Call LLM synthesis service ────────────────────────────
    try:
        final_report = await generate_final_report(
            clinical_data=clinical_data,
            ultrasound_data=ultrasound_data,
            fnac_data=fnac_data,
            agent_chat_data=agent_chat_data,
        )
    except Exception as exc:
        logger.error("Synthesis LLM failed for session %s: %s", session_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate final synthesis report.",
        )

    # ── 4. Build response payload ────────────────────────────────
    response_data = final_report.model_dump()
    response_data["session_id"] = session_id

    # ── 5. Persist synthesis result back into the session ────────
    try:
        await memory_manager.save_diagnostic(
            session_id=session_id,
            node_type="synthesis",
            data=response_data,
        )
    except Exception as exc:
        # Non-fatal: log and continue — the response is still returned
        logger.warning(
            "Could not persist synthesis result for session %s: %s",
            session_id,
            exc,
        )

    logger.info(
        "Synthesis complete for session=%s | nodes=%s | classification=%s | "
        "stage=%s | consistent=%s | review=%s",
        session_id,
        final_report.nodes_available,
        final_report.corrected_classification,
        final_report.tumor_stage,
        final_report.is_consistent,
        final_report.needs_manual_review,
    )

    return response_data


# ═══════════════════════════════════════════════════════════════
# GET /synthesis/image/{image_id}
# ═══════════════════════════════════════════════════════════════

from fastapi import Response

@router.get(
    "/image/{image_id}",
    summary="Retrieve a stored diagnostic image by ID",
)
async def get_synthesis_image(image_id: int):
    """Return the raw PNG bytes of a previously-stored diagnostic image."""
    image_data = await memory_manager.get_image(image_id)
    if not image_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image ID {image_id} not found.",
        )
    return Response(content=image_data, media_type="image/png")
