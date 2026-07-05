"""
Internal Synthesis Orchestrator.

Called by other parts of the system (e.g., after Node 4 or Node 6 completes)
to trigger an automated synthesis without going through the HTTP layer.

Uses the shared session_id to pull ALL available node results from the
MemoryManager and delegates to the LLM synthesis service.
"""

import logging
from typing import Any, Dict, Optional

from app.services.synthesis_llm import generate_final_report

logger = logging.getLogger(__name__)


async def run_internal_synthesis(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Trigger Node 7 synthesis programmatically using the shared session_id.

    Loads all diagnostic data (Nodes 1-6) from the MemoryManager, calls the
    LLM synthesis service, persists the result, and returns the report dict.

    Args:
        session_id: The shared session identifier used across all nodes.

    Returns:
        The FinalMedicalReport as a dict, or None on failure.
    """
    from app.services.memory_manager import memory_manager

    try:
        # ── Load all node data from session state ──
        session_ctx = await memory_manager.load_context(session_id)
        if not session_ctx:
            logger.warning("run_internal_synthesis: session %s not found", session_id)
            return None

        diag = session_ctx.diagnostic_context or {}

        clinical_data    = diag.get("clinical", {})
        ultrasound_data  = diag.get("ultrasound", {})
        fnac_data        = diag.get("fnac", {})
        agent_chat_data  = diag.get("agent_chat", {})

        if not any([clinical_data, ultrasound_data, fnac_data, agent_chat_data]):
            logger.info(
                "run_internal_synthesis: no diagnostic data in session %s — skipping",
                session_id,
            )
            return None

        logger.info(
            "Internal synthesis triggered for session=%s | nodes_with_data=%s",
            session_id,
            [k for k in ("clinical", "ultrasound", "fnac", "agent_chat") if diag.get(k)],
        )

        # ── Call LLM synthesis ──
        final_report = await generate_final_report(
            clinical_data=clinical_data,
            ultrasound_data=ultrasound_data,
            fnac_data=fnac_data,
            agent_chat_data=agent_chat_data,
        )

        # ── Build and persist result ──
        response_data = final_report.model_dump()
        response_data["session_id"] = session_id

        await memory_manager.save_diagnostic(
            session_id=session_id,
            node_type="synthesis",
            data=response_data,
        )

        logger.info(
            "Internal synthesis saved | session=%s | classification=%s | stage=%s",
            session_id,
            final_report.corrected_classification,
            final_report.tumor_stage,
        )
        return response_data

    except Exception as exc:
        logger.error(
            "run_internal_synthesis failed for session %s: %s",
            session_id,
            exc,
            exc_info=True,
        )
        return None
