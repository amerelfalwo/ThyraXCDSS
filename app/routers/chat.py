"""
AI Agent Chat Endpoints (Node 5).

POST /agent/chat  (Dual-Mode)
  Mode 1 — General Medical Chat:
    Send only ``user_message``.  No session_id needed.
    Uses a generic medical-assistant persona; no persistence.
  Mode 2 — Contextual Patient Chat:
    Supply ``session_id``, ``patient_id``, and ``doctor_id``.
    Data-isolation is enforced, patient context injected,
    and conversation persisted to the sessions table.

Features:
  - Circuit breaker protection for LLM API.
  - Audit logging for every interaction.
"""

import json
import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import datetime

from app.core.security import verify_internal_api_key
from app.core.database import get_db
from app.core.config import settings
from app.schemas.chat import ChatResponse, AgentChatRequest

logger = logging.getLogger(__name__)

from app.core.responses import UnicodeJSONResponse

router = APIRouter(
    prefix="/agent",
    tags=["AI Agent"],
    dependencies=[Depends(verify_internal_api_key)],
    default_response_class=UnicodeJSONResponse,
)


@router.post("/chat", response_model=ChatResponse)
async def agent_chat(
    request: AgentChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    **Dual-Mode** chat endpoint.

    **Mode 1 — General Medical Chat** (``session_id`` is ``None``):
      • Skips database validation.
      • Uses the general agent persona without persistence.

    **Mode 2 — Contextual Patient Chat** (``session_id`` provided):
      • Validates data-isolation (doctor owns session).
      • Injects full patient context (long-term + short-term memory).
      • Persists the exchange to the ``sessions`` table.
    """
    from app.agent.agent import run_agent
    from app.core.audit import log_audit_event

    # ═══════════════════════════════════════════════════════════
    # MODE 1 — General Medical Chat  (no session)
    # ═══════════════════════════════════════════════════════════
    if request.session_id is None:
        try:
            result = await run_agent(query=request.user_message)
            content = result["output"]
            tools_used = result.get("tools_used", [])

            log_audit_event(
                node="agent_chat_general",
                action="general_medical_query",
                result=content[:200],
                metadata={
                    "query": request.user_message[:200],
                    "mode": "general",
                },
            )

            return ChatResponse(status="success", response=content, tools_used=tools_used)
        except Exception as e:
            logger.error(f"General-mode error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ═══════════════════════════════════════════════════════════
    # MODE 2 — Contextual Patient Chat  (session present)
    # ═══════════════════════════════════════════════════════════
    from app.schemas.memory_models import Session as SessionModel, Patient

    # doctor_id is mandatory for contextual mode validation
    if request.doctor_id is None:
        raise HTTPException(
            status_code=422,
            detail="doctor_id is required when session_id is provided.",
        )

    doctor_id_str = str(request.doctor_id)

    # ── Data-Isolation Guard ──
    session_result = await db.execute(
        select(SessionModel).where(
            SessionModel.session_id == request.session_id
        )
    )
    session = session_result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Session does not belong to the provided Doctor.",
        )

    if request.patient_id is not None:
        patient_id_str = str(request.patient_id)
        patient_result = await db.execute(
            select(Patient).where(
                Patient.patient_id == patient_id_str,
                Patient.doctor_id == doctor_id_str,
            )
        )
        if not patient_result.scalar_one_or_none():
            raise HTTPException(
                status_code=403,
                detail="Forbidden: Patient does not belong to the provided Doctor.",
            )

    try:
        # run_agent handles context loading and persistence
        result = await run_agent(
            query=request.user_message,
            session_id=request.session_id,
            patient_id=request.patient_id,
        )
        content = result["output"]
        tools_used = result.get("tools_used", [])

        log_audit_event(
            node="agent_chat_contextual",
            action="agent_invocation",
            result=content[:200],
            metadata={
                "query": request.user_message[:200],
                "patient_id": request.patient_id,
                "session_id": request.session_id,
                "doctor_id": request.doctor_id,
                "mode": "contextual",
            },
        )

        return ChatResponse(status="success", response=content, tools_used=tools_used)
    except Exception as e:
        logger.error(f"Contextual-mode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))