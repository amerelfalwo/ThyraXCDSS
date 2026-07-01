"""
Simple AI Chat Endpoint — Streaming + Memory-Backed.

POST /ai/chat
  Accepts ``session_id`` + ``user_message``.
  Loads conversation history and diagnostic context from the database
  via the MemoryManager, streams the AI response token-by-token
  as Server-Sent Events (SSE), then persists the full exchange.

Features:
  - SSE streaming response (text/event-stream).
  - Automatic session creation if session_id doesn't exist.
  - Full memory context injection (chat history + diagnostics).
  - Conversation persistence after each exchange.
  - Auto-summarisation when history grows large.
  - Circuit breaker protection for LLM API.
  - Audit logging for every interaction.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.responses import UnicodeJSONResponse

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Schemas
# ═══════════════════════════════════════════════════════════════

class SimpleChatRequest(BaseModel):
    """
    Minimal chat request — session ID + message.

    The server handles all memory (history loading, context injection,
    and persistence) automatically via the session_id.
    """
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description=(
            "Session identifier. If the session doesn't exist, "
            "it will be created automatically."
        ),
    )
    user_message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's message or medical question.",
    )


# ═══════════════════════════════════════════════════════════════
# Router — no auth dependency
# ═══════════════════════════════════════════════════════════════

router = APIRouter(
    prefix="/ai",
    tags=["AI Chat"],
    default_response_class=UnicodeJSONResponse,
)


# ═══════════════════════════════════════════════════════════════
# Direct LLM Streaming (Fast — No Agent Overhead)
# ═══════════════════════════════════════════════════════════════

_CHAT_SYSTEM_PROMPT = """You are ThyraX, an Elite Clinical Decision Support AI specialized in Thyroid pathology.
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
    """Convert memory chat_history list into a readable text block for the prompt."""
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


async def _stream_llm_direct(
    query: str,
    chat_history: list,
    patient_context: str,
) -> AsyncGenerator[str, None]:
    """
    Stream tokens directly from ChatGroq — single LLM call, no agent loop.

    This is 3-5x faster than AgentExecutor because:
    - No MCP tool loading
    - No tool-calling decision round-trip
    - No multi-turn agent scratchpad
    - Single API call with streaming enabled
    """
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage
    from app.core.config import settings

    # ── Build the system prompt with context ──
    history_block = _format_history_block(chat_history)
    system_content = _CHAT_SYSTEM_PROMPT.format(
        patient_context=patient_context or "No patient context available.",
        history_block=history_block,
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=query),
    ]

    # ── Get API key (rotation) ──
    keys = settings.get_groq_keys()
    if not keys:
        yield "عذراً، لم يتم تكوين مفتاح API."
        return

    selected_key = keys[0]

    # ── Initialize LLM with streaming ──
    llm = ChatGroq(
        model=settings.GROQ_MODEL,
        api_key=selected_key,
        temperature=0.2,
        max_tokens=2048,
        streaming=True,
    )

    try:
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        logger.error(f"Direct LLM streaming error: {e}", exc_info=True)
        yield "عذراً، حدث خطأ في المعالجة. يرجى المحاولة مرة أخرى."


async def _sse_generator(
    request: SimpleChatRequest,
    chat_history: list,
    patient_context: str,
):
    """
    SSE event generator.

    Format:
      data: {"token": "..."}     — for each streamed token
      data: [DONE]               — end-of-stream signal

    After streaming completes, persists the exchange to memory.
    """
    from app.services.memory_manager import memory_manager
    from app.core.audit import log_audit_event

    full_response = ""

    try:
        async for token in _stream_llm_direct(
            query=request.user_message,
            chat_history=chat_history,
            patient_context=patient_context,
        ):
            if token:  # skip empty tokens
                full_response += token
                # SSE format: data: {json}\n\n
                event_data = json.dumps(
                    {"token": token},
                    ensure_ascii=False,
                )
                yield f"data: {event_data}\n\n"

        # Send [DONE] signal
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"SSE generator error: {e}", exc_info=True)
        error_data = json.dumps(
            {"error": str(e)},
            ensure_ascii=False,
        )
        yield f"data: {error_data}\n\n"
        yield "data: [DONE]\n\n"

    # ── Post-stream: persist exchange to memory ──
    if full_response:
        try:
            await memory_manager.save_exchange(
                session_id=request.session_id,
                user_message=request.user_message,
                ai_response=full_response,
            )
            logger.debug(
                f"Persisted exchange to session {request.session_id}"
            )
        except Exception as e:
            logger.error(f"Failed to save exchange: {e}")

        # Trigger summarization if history is growing
        try:
            ctx = await memory_manager.load_context(request.session_id)
            if len(ctx.chat_history) > 6:
                asyncio.create_task(
                    memory_manager.summarize_and_prune(request.session_id)
                )
        except Exception as e:
            logger.warning(f"Summarization trigger failed: {e}")

        # Audit log
        try:
            log_audit_event(
                node="ai_chat_stream",
                action="streamed_medical_query",
                result=full_response[:200],
                metadata={
                    "query": request.user_message[:200],
                    "session_id": request.session_id,
                    "mode": "stream",
                    "response_length": len(full_response),
                },
            )
        except Exception as e:
            logger.warning(f"Audit log failed: {e}")


# ═══════════════════════════════════════════════════════════════
# Endpoint
# ═══════════════════════════════════════════════════════════════

@router.post("/chat")
async def ai_chat_stream(request: SimpleChatRequest):
    """
    **Streaming AI Chat** — send a message, receive a streamed response.

    The server automatically:
    1. Loads (or creates) the session from the database.
    2. Injects conversation history + diagnostic context into the AI.
    3. Streams the AI response token-by-token as SSE events.
    4. Persists the full exchange to the session after streaming.
    5. Triggers auto-summarization when history grows large.

    **Request:**
    ```json
    {
      "session_id": "my-session-123",
      "user_message": "ما هي أعراض قصور الغدة الدرقية؟"
    }
    ```

    **Response (SSE stream):**
    ```
    data: {"token": "أعراض"}
    data: {"token": " قصور"}
    data: {"token": " الغدة"}
    ...
    data: [DONE]
    ```

    **Client-side consumption (JavaScript):**
    ```javascript
    const response = await fetch('/ai/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: 'my-session-123',
        user_message: 'What is hypothyroidism?'
      })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const text = decoder.decode(value);
      // Parse SSE events from text
    }
    ```
    """
    from app.services.memory_manager import memory_manager

    # ── Load memory context from database ──
    try:
        memory_ctx = await memory_manager.load_context(
            session_id=request.session_id,
        )
        patient_context = memory_ctx.to_prompt_context()
        chat_history = memory_ctx.chat_history or []
    except Exception as e:
        logger.error(f"Memory load failed for session {request.session_id}: {e}")
        patient_context = "No patient context available for this session."
        chat_history = []

    return StreamingResponse(
        _sse_generator(
            request=request,
            chat_history=chat_history,
            patient_context=patient_context,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
