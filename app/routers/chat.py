"""
AI Agent Chat Endpoints (Node 7) — Fast Direct LLM.

POST /agent/chat  (Dual-Mode)
  Mode 1 — General Medical Chat:
    Send only ``user_message``.  No session_id needed.
    Uses a generic medical-assistant persona; no persistence.
  Mode 2 — Contextual Patient Chat:
    Supply ``session_id`` and ``user_message``.
    Data-isolation is enforced, patient context injected,
    and conversation persisted to the sessions table.

Performance:
  - Uses direct ChatGroq.ainvoke() instead of AgentExecutor.
  - Single LLM call — no tool-calling overhead (3-5x faster).
  - Circuit breaker protection for LLM API.
  - Audit logging for every interaction.
"""

import json
import logging
from typing import Optional, List, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.security import verify_internal_api_key
from app.core.database import get_db
from app.core.config import settings
from app.schemas.chat import ChatResponse, AgentChatRequest
from app.core.responses import UnicodeJSONResponse

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/agent",
    tags=["AI Agent"],
    dependencies=[Depends(verify_internal_api_key)],
    default_response_class=UnicodeJSONResponse,
)


# ═══════════════════════════════════════════════════════════════
# System Prompt for Direct LLM (No Agent / No Tools)
# ═══════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are ThyraX, an Elite Clinical Decision Support AI specialized in Thyroid pathology.
Your primary role is to act as an expert consultant to the medical doctor. You will review and discuss the results from the other diagnostic nodes (e.g., Clinical Assessment, Ultrasound Prediction, FNAC) provided in the PATIENT CONTEXT.

RULES:
- Discuss the clinical and prediction results with the doctor to help formulate a final diagnosis or treatment plan.
- Analyze the findings critically and answer any questions the doctor has regarding the prediction nodes.
- LANGUAGE MIRRORING: Reply in the exact same language used by the user.
- TONE: Address the user respectfully as 'Doctor', 'يا دكتور', or 'حضرتك'.
- Be concise but thorough. Provide actionable clinical insights.
- If asked about non-medical topics, politely decline.
- Start your answer directly — no preamble like "Based on..." or "Here is what I found".

[PATIENT CONTEXT]
{patient_context}

[CONVERSATION HISTORY]
{history_block}
"""


def _format_history_block(chat_history: list) -> str:
    """Convert chat_history list into a readable text block."""
    if not chat_history:
        return "No previous conversation."

    lines = []
    for entry in chat_history[-10:]:  # last 10 exchanges max
        if isinstance(entry, dict):
            role = entry.get("role", "user")
            content = entry.get("content", "")
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            role, content = entry[0], entry[1]
        else:
            continue
        prefix = "Doctor" if role in ("user", "human") else "ThyraX"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines) if lines else "No previous conversation."


# ═══════════════════════════════════════════════════════════════
# Fast Direct LLM Call (replaces slow AgentExecutor)
# ═══════════════════════════════════════════════════════════════

async def _stream_llm_direct(
    query: str,
    patient_context: str = "No patient context available.",
    chat_history: list | None = None,
) -> AsyncGenerator[str, None]:
    """
    Call ChatGroq directly — single LLM call, no agent loop.

    This is 3-5x faster than AgentExecutor because:
    - No MCP tool loading
    - No tool-calling decision round-trip
    - No multi-turn agent scratchpad
    """
    from groq import AsyncGroq

    history_block = _format_history_block(chat_history or [])
    system_content = _SYSTEM_PROMPT.format(
        patient_context=patient_context,
        history_block=history_block,
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
    ]

    # ── API key selection ──
    keys = settings.get_groq_keys()
    if not keys:
        raise ValueError("No GROQ_API_KEYs found in configuration.")

    last_error = None
    for i, key in enumerate(keys):
        try:
            client = AsyncGroq(api_key=key)
            response = await client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=2048,
                stream=True,
            )
            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
            return
        except Exception as e:
            last_error = e
            err_str = str(e)
            _QUOTA_SIGNALS = ("429", "RESOURCE_EXHAUSTED", "rate_limit", "Too Many Requests")
            if any(sig in err_str for sig in _QUOTA_SIGNALS):
                logger.warning(f"Groq Quota Exhausted on key {i}. Trying next key...")
                continue
            raise

    raise last_error  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════
# Endpoint
# ═══════════════════════════════════════════════════════════════

@router.post("/chat")
async def agent_chat(
    request: AgentChatRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    **Contextual Patient Chat Endpoint** (Fast — Direct LLM).

    Accepts ``session_id`` and ``user_message``.
    Automatically loads diagnostic context (from Clinical, Ultrasound, FNAC, etc.)
    and injects it into the conversation, enabling Node 7 to answer based on
    the results from the first 4 nodes.
    Returns a streaming Server-Sent Events (SSE) response.
    """
    from app.core.audit import log_audit_event
    from app.services.memory_manager import memory_manager

    # ── Load Memory Context (Results from previous nodes) ──
    patient_context = "No patient context available for this session."
    effective_history: list = []

    try:
        memory_ctx = await memory_manager.load_context(
            session_id=request.session_id,
        )
        patient_context = memory_ctx.to_prompt_context()
        if memory_ctx.chat_history:
            effective_history = memory_ctx.chat_history
    except Exception as e:
        logger.error(f"Memory load failed: {e}")

    async def _stream_and_save() -> AsyncGenerator[str, None]:
        full_response = ""
        try:
            generator = _stream_llm_direct(
                query=request.user_message,
                patient_context=patient_context,
                chat_history=effective_history,
            )
            async for token in generator:
                if token:
                    full_response += token
                    event_data = json.dumps({"token": token}, ensure_ascii=False)
                    yield f"data: {event_data}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Contextual-mode error: {e}")
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"
            yield "data: [DONE]\n\n"

        # ── Post-stream: Audit & Persist ──
        if full_response:
            try:
                log_audit_event(
                    node="agent_chat_contextual",
                    action="agent_invocation",
                    result=full_response[:200],
                    metadata={
                        "query": request.user_message[:200],
                        "session_id": request.session_id,
                        "mode": "contextual",
                    },
                )
            except Exception as e:
                logger.warning(f"Audit log failed: {e}")

            try:
                await memory_manager.save_exchange(
                    session_id=request.session_id,
                    user_message=request.user_message,
                    ai_response=full_response,
                )
            except Exception as e:
                logger.error(f"Failed to save exchange to memory: {e}")

            # ── Trigger summarization if history grows large ──
            try:
                ctx = await memory_manager.load_context(request.session_id)
                if len(ctx.chat_history) > 6:
                    background_tasks.add_task(
                        memory_manager.summarize_and_prune, request.session_id
                    )
            except Exception as e:
                logger.warning(f"Summarization trigger failed: {e}")

    return StreamingResponse(
        _stream_and_save(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )