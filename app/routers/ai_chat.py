"""
AI Chat Endpoints — Medical Information Dictionary.

Provides a unified endpoint:
1. /ai/chat/dictionary: A friendly medical knowledge repository using RAG & MCP tools.

This endpoint has NO access to patient data and is strictly for general medical inquiries.
"""

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.responses import UnicodeJSONResponse
from app.core.audit import log_audit_event

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Schemas
# ═══════════════════════════════════════════════════════════════

class SimpleChatRequest(BaseModel):
    """
    Minimal chat request — session ID + message.
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
        description="The user's medical question.",
    )


# ═══════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════

router = APIRouter(
    prefix="/ai",
    tags=["AI Chat"],
    default_response_class=UnicodeJSONResponse,
)


async def _sse_wrapper(
    agent_generator: AsyncGenerator[str, None],
    session_id: str,
    query: str,
    mode: str,
) -> AsyncGenerator[str, None]:
    """
    Consumes tokens from the agent's stream, wraps them in SSE format,
    and captures the full response to record an audit log.
    """
    full_response = ""

    try:
        async for token in agent_generator:
            if token:
                full_response += token
                event_data = json.dumps({"token": token}, ensure_ascii=False)
                yield f"data: {event_data}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"SSE generator error in {mode} mode: {e}", exc_info=True)
        error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
        yield f"data: {error_data}\n\n"
        yield "data: [DONE]\n\n"

    # ── Post-stream: Audit Logging ──
    if full_response:
        try:
            log_audit_event(
                node=f"ai_chat_{mode}",
                action=f"streamed_{mode}_query",
                result=full_response[:200],
                metadata={
                    "query": query[:200],
                    "session_id": session_id,
                    "mode": mode,
                    "response_length": len(full_response),
                },
            )
        except Exception as e:
            logger.warning(f"Audit log failed: {e}")


# ═══════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════

@router.post("/chat/dictionary")
async def ai_chat_dictionary(request: SimpleChatRequest, background_tasks: BackgroundTasks):
    """
    **Streaming Medical Dictionary AI Chat**
    Acts as a friendly medical information repository using RAG and MCP tools.
    Has no access to patient data.
    """
    from app.agent.research_agent import stream_research_agent

    # The agent will handle loading memory and saving the exchange
    # using the provided background_tasks.
    generator = stream_research_agent(
        query=request.user_message,
        session_id=request.session_id,
        fast_path=False,  # Set to False to enable full AgentExecutor (RAG + Web Search tools)
        background_tasks=background_tasks,
    )

    return StreamingResponse(
        _sse_wrapper(
            agent_generator=generator,
            session_id=request.session_id,
            query=request.user_message,
            mode="dictionary",
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
