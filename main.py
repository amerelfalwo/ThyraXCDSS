"""
ThyraX CDSS — Unified Clinical Decision Support System API.

v4.0.0 — Continuous Context Orchestrator

Combines:
    - Node 1+2: Clinical Assessment (XGBoost) + Agentic Routing
    - Node 3: Ultrasound Gatekeeper (ONNX MobileNetV2)
    - Node 4: ONNX Segmentation & Classification (ACR TI-RADS)
    - Node 5: Medical AI Assistant Chat (Groq/Llama-3 + RAG + Web Search)
    - NEW: FNAC Cytopathology (Bethesda System I–VI)
    - NEW: Patient State Manager (Dynamic Context Orchestration)

Production Features:
    - Groq (Llama-3) powered agent — fast inference, low cost
    - Dynamic Patient State tracking across all nodes
    - Circuit Breaker pattern for all LLM-dependent services
    - JSONL audit logging for clinical traceability
    - Confidence threshold guards (needs_manual_review)
    - Lightweight RAG re-ranking + Web fallback

Architecture:
    All ML/LLM imports are lazy-loaded per function call.
    No heavy models at module scope (512 MB RAM mandate).
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.security import verify_internal_api_key

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Lifespan — startup & shutdown (NO model loading here)
# ═══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    
    # ── 1. Initialize LangChain LLM Cache (Redis) ──
    redis_client = None
    try:
        import os
        from langchain.globals import set_llm_cache
        from langchain_community.cache import AsyncRedisCache
        import redis.asyncio as redis_async

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        logger.info(f"Initializing LangChain LLM Cache with Redis at {redis_url}")
        
        # Initialize async Redis client to prevent event loop blocking
        redis_client = redis_async.from_url(redis_url)
        
        # Set the global async cache for LangChain
        set_llm_cache(AsyncRedisCache(redis_client))
        logger.info("LangChain global AsyncRedisCache configured successfully.")
    except Exception as e:
        logger.warning(f"Error initializing LLM Cache: {e}")

    # ── 2. Verify Database Connectivity ──
    try:
        from app.core.database import engine
        from sqlalchemy import text
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connectivity verified successfully.")
    except Exception as e:
        logger.warning(f"Error connecting to database during startup: {e}")

    # Ensure required directories exist
    Path("media").mkdir(exist_ok=True)
    Path("data/audit").mkdir(parents=True, exist_ok=True)

    logger.info("ThyraX CDSS v4.0.0 — Continuous Context Orchestrator ready.")

    yield

    logger.info("Shutdown sequence initiated for ThyraX CDSS.")

    # ── Graceful shutdown: close connections ──
    try:
        from app.agent.mcp_servers.mcp_client import mcp_client_manager
        await mcp_client_manager.shutdown()
        logger.info("MCP servers shut down.")
    except Exception as e:
        logger.warning(f"Error shutting down MCP servers: {e}")

    try:
        from app.core.database import engine
        await engine.dispose()
        logger.info("SQLAlchemy Async Engine disposed safely.")
    except Exception as e:
        logger.warning(f"Error disposing SQLAlchemy engine: {e}")

    try:
        if redis_client:
            await redis_client.close()
            logger.info("Redis cache connection closed securely.")
    except Exception as e:
        logger.warning(f"Error closing Redis connection: {e}")

    logger.info("ThyraX CDSS shutdown complete.")


# ═══════════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════════

from app.core.responses import UnicodeJSONResponse

app = FastAPI(
    title="ThyraX CDSS API",
    description=(
        "Clinical Decision Support System for Thyroid Cancer Diagnosis.\n\n"
        "## Core AI Nodes\n"
        "- **Node 1+2** `POST /clinical/assess` — XGBoost prediction + agentic routing\n"
        "- **Node 3** `POST /image/validate` — Ultrasound gatekeeper (ONNX)\n"
        "- **Node 4** `POST /image/predict` — ONNX segmentation + classification (ACR TI-RADS)\n"
        "- **Node 5** `POST /agent/chat` — Medical AI assistant (Groq/Llama-3)\n"
        "- **Node 5** `POST /agent/chat/stream` — Medical AI assistant (SSE streaming)\n"
        "- **NEW** `POST /fnac/predict` — FNAC cytopathology (Bethesda I–VI)\n\n"
        "## Context Orchestration\n"
        "- `GET /state/{session_id}` — Retrieve patient diagnostic context\n"
        "- `DELETE /state/{session_id}` — Clear patient session\n\n"
        "## Production Features\n"
        "- Groq/Llama-3 Agent, Dynamic Patient State, Circuit Breaker, Audit Logging\n"
    ),
    version="4.0.0",
    lifespan=lifespan,
    default_response_class=UnicodeJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure required directories exist before mounting static files
Path("media").mkdir(exist_ok=True)

# Mount media directory for static file serving
app.mount("/media", StaticFiles(directory="media"), name="media")


# ═══════════════════════════════════════════════════════════════
# Register Routers — All Nodes
# ═══════════════════════════════════════════════════════════════

from app.routers import chat, clinical, image, fnac

app.include_router(clinical.router)
app.include_router(image.router)
app.include_router(fnac.router)
app.include_router(chat.router)


# ═══════════════════════════════════════════════════════════════
# Patient State Endpoints
# ═══════════════════════════════════════════════════════════════

@app.get(
    "/state/{session_id}",
    tags=["Patient State"],
    dependencies=[Depends(verify_internal_api_key)],
)
async def get_patient_state(session_id: str):
    """Retrieve the current diagnostic state for a patient session."""
    from app.services.patient_state import state_manager

    state = state_manager.get_state(session_id)
    if state is None:
        return {
            "status": "not_found",
            "session_id": session_id,
            "message": "No active session found. The session may have expired.",
        }
    return {
        "status": "success",
        "session_id": session_id,
        "state": state,
    }


@app.delete(
    "/state/{session_id}",
    tags=["Patient State"],
    dependencies=[Depends(verify_internal_api_key)],
)
async def clear_patient_state(session_id: str):
    """Clear a patient session's diagnostic state."""
    from app.services.patient_state import state_manager

    cleared = state_manager.clear_session(session_id)
    return {
        "status": "cleared" if cleared else "not_found",
        "session_id": session_id,
    }


@app.get(
    "/state",
    tags=["Patient State"],
    dependencies=[Depends(verify_internal_api_key)],
)
async def list_sessions():
    """List all active patient session IDs."""
    from app.services.patient_state import state_manager

    sessions = state_manager.list_sessions()
    return {
        "active_sessions": len(sessions),
        "session_ids": sessions,
    }


# ═══════════════════════════════════════════════════════════════
# Health Check + System Status
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """Return service health status including circuit breaker states."""
    from app.core.circuit_breaker import get_circuit_status
    from app.services.patient_state import state_manager

    return {
        "status": "healthy",
        "service": "ThyraX AI Engine",
        "version": "4.0.0",
        "llm_backend": "Groq (Llama-3)",
        "nodes": [
            "clinical_assessment",
            "agentic_routing",
            "ultrasound_gatekeeper",
            "onnx_segmentation",
            "fnac_cytopathology",
            "medical_agent_chat",
        ],
        "active_sessions": len(state_manager.list_sessions()),
        "circuit_breakers": get_circuit_status(),
    }


# ═══════════════════════════════════════════════════════════════
# Audit Log Endpoint
# ═══════════════════════════════════════════════════════════════

@app.get("/audit/logs", dependencies=[Depends(verify_internal_api_key)])
async def get_audit_logs(limit: int = 50):
    """Return recent audit log entries for clinical traceability."""
    from app.core.audit import read_recent_audits

    entries = read_recent_audits(limit=min(limit, 200))
    return {"entries": entries, "total": len(entries)}