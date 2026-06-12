"""
AI Agent Chat Endpoint (Node 5).

POST /agent/chat
  Accepts a medical query, optional chat_history, and optional image
  for multi-modal analysis. Uses StreamingResponse for real-time UX.

Supports three input modes:
  - Text only:   {"query": "What is TSH?"}
  - Image only:  {"image_base64": "...", "image_content_type": "image/png"}
  - Multimodal:  {"query": "Interpret this lab report", "image_base64": "..."}

Features:
  - Streaming token output via StreamingResponse.
  - Multi-modal input (text + optional image).
  - Circuit breaker protection for Gemini API.
  - Audit logging for every interaction.
  - Medical guardrails — non-medical queries rejected pre-LLM.
"""

import re
import json
import logging
import base64
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Form, File, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import datetime

from app.core.security import verify_internal_api_key
from app.core.database import get_db
from app.core.config import settings
from app.schemas.chat import ChatResponse, AgentChatRequest
from app.agent.agent import run_agent

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agent",
    tags=["AI Agent"],
    dependencies=[Depends(verify_internal_api_key)],
)

# ═══════════════════════════════════════════════════════════════
# Medical Guardrail — Pre-LLM Filter
# ═══════════════════════════════════════════════════════════════

_NON_MEDICAL_PATTERNS = [
    r"\b(write|generate|create)\s+(code|script|program|function|class)\b",
    r"\b(python|javascript|java|c\+\+|html|css|sql)\s+(code|script|program)\b",
    r"\b(recipe|cook|bake|ingredient)\b",
    r"\b(weather|forecast|temperature)\s+(in|for|today)\b",
    r"\b(stock|crypto|bitcoin|trading|forex)\b",
    r"\b(joke|funny|humor|riddle)\b",
    r"\b(poem|story|novel|fiction|song|lyrics)\b",
    r"\b(sports?|football|soccer|basketball|tennis)\s+(score|result|game)\b",
    r"\b(movie|film|tv\s*show|series|anime)\s+(recommend|review)\b",
    r"\b(travel|hotel|flight|vacation|tourism)\b",
    r"\b(math|calcul|algebra|geometry|equation)\b",
    r"\b(translate|translation)\b",
]

# Broadened to cover all medical specialties (not just thyroid)
_MEDICAL_KEYWORDS = [
    # ── Thyroid & Endocrinology ──
    "thyroid", "tsh", "t3", "t4", "nodule", "goiter", "biopsy", "fna",
    "tirads", "ata", "thyroxine", "levothyroxine", "methimazole", "ptu",
    "hashimoto", "graves", "papillary", "follicular", "medullary",
    "anaplastic", "calcitonin", "thyroglobulin", "tpo",
    "hypothyroid", "hyperthyroid", "endocrin", "hormone", "insulin",
    "diabetes", "adrenal", "pituitary", "cortisol", "testosterone",
    # ── General Medicine ──
    "patient", "symptom", "diagnos", "treat", "medic", "clinic",
    "lab", "blood", "test", "scan", "imaging", "prognos", "patholog",
    "disease", "disorder", "health", "pharma", "dose", "drug",
    "prescription", "histolog", "cytolog", "biopsy",
    # ── Oncology ──
    "cancer", "tumor", "malignan", "benign", "metastas", "lymph node",
    "chemotherap", "radiation", "oncolog", "staging", "carcinoma",
    "neoplasm", "remission", "relapse",
    # ── Cardiology ──
    "cardiac", "heart", "ecg", "ekg", "arrhythmi", "hypertens",
    "cholesterol", "statin", "atrial", "ventricular", "murmur",
    # ── Neurology ──
    "neurolog", "brain", "stroke", "seizure", "epileps", "migraine",
    "neuropath", "dementia", "alzheimer", "parkinson",
    # ── Surgery & Emergency ──
    "surgery", "surgical", "operat", "anesthes", "emergency", "trauma",
    "fracture", "wound", "resuscitat",
    # ── Radiology & Imaging ──
    "ultrasound", "x-ray", "xray", "mri", "ct scan", "radiol",
    "mammogra", "radionuclide", "iodine", "contrast",
    # ── Pharmacology ──
    "antibiotic", "antiviral", "analgesic", "nsaid", "opioid",
    "contraindic", "adverse effect", "side effect", "interaction",
    # ── Other specialties ──
    "pediatric", "obstetric", "gynecol", "dermatol", "ophthalm",
    "pulmonar", "respiratory", "gastro", "hepat", "renal", "kidney",
    "urolog", "orthoped", "rheumatol", "immunol", "allerg", "infect",
    "hematol", "anemia", "coagul", "vitamin", "mineral", "nutrition",
    # ── General clinical terms ──
    "guideline", "protocol", "recommend", "risk", "referral",
    "differential", "etiology", "comorbid", "chronic", "acute",
    "vital sign", "fever", "pain", "inflam", "edema", "fatigue",
    "nausea", "vomit", "diarrhea", "constipat",
]

def _get_rejection_message(query: str) -> str:
    """Returns the rejection message in the language of the query."""
    if query and any("\u0600" <= char <= "\u06FF" for char in query):
        return (
            "أنا أقدر سؤالك، لكني ThyraX — مساعد ذكاء اصطناعي طبي "
            "مصمم للمساعدة في الأسئلة السريرية والرعاية الصحية.\n\n"
            "يمكنني المساعدة في:\n"
            "• المبادئ التوجيهية الطبية والبروتوكولات السريرية\n"
            "• تفسير نتائج المختبر\n"
            "• تحليل الأعراض والتشخيص التفريقي\n"
            "• معلومات الأدوية وعلم الصيدلة\n"
            "• إرشادات تفسير الصور الطبية\n\n"
            "يرجى إعادة صياغة سؤالك في سياق طبي أو صحي."
        )
    return (
        "I appreciate your question, but I'm ThyraX — a medical AI assistant "
        "designed to help with clinical and healthcare questions.\n\n"
        "I can assist with:\n"
        "• Medical guidelines and clinical protocols\n"
        "• Lab result interpretation\n"
        "• Symptom analysis and differential diagnoses\n"
        "• Drug information and pharmacology\n"
        "• Imaging interpretation guidance\n\n"
        "Please rephrase your question in a medical or healthcare context."
    )


def _is_medical_query(query: str) -> bool:
    """
    Check whether a query is medical/clinical in nature.
    """
    query_lower = query.lower()

    # If it contains medical keywords, allow it
    for keyword in _MEDICAL_KEYWORDS:
        if keyword in query_lower:
            return True

    # If it matches non-medical patterns, reject it
    for pattern in _NON_MEDICAL_PATTERNS:
        if re.search(pattern, query_lower):
            return False

    # Default: allow ambiguous queries through to the LLM
    return True


def _get_effective_query(query: str | None, has_image: bool) -> str:
    """
    Derive the effective text query for guardrails and logging.
    """
    if query:
        return query
    if has_image:
        return "Analyze the attached medical image"
    return "No query provided"


# ═══════════════════════════════════════════════════════════════
# Conversation Persistence Helper
# ═══════════════════════════════════════════════════════════════

async def _persist_conversation(
    session_id: str,
    user_message: str,
    assistant_response: str,
) -> None:
    """
    Append user + assistant messages to ``sessions.conversation_history``.

    Uses a **fresh** ``AsyncSession`` (not the request-scoped one) so that
    the write succeeds even after FastAPI has closed the DI session.
    """
    from app.core.database import AsyncSessionLocal
    from app.schemas.memory_models import Session as SessionModel

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(SessionModel).where(SessionModel.session_id == session_id)
        )
        session = result.scalar_one_or_none()
        if not session:
            logger.warning(f"Session {session_id} not found — skipping persistence.")
            return

        history = list(session.conversation_history or [])
        history.append({"role": "user", "content": user_message, "ts": now_iso})
        history.append({"role": "assistant", "content": assistant_response, "ts": now_iso})

        session.conversation_history = history
        await db.commit()
        logger.info(
            f"Persisted conversation for session {session_id} "
            f"({len(history)} messages total)."
        )


# ═══════════════════════════════════════════════════════════════
# Streaming Generator
# ═══════════════════════════════════════════════════════════════

async def _stream_agent_response(
    query: Optional[str],
    chat_history: List[dict],
    image_base64: Optional[str],
    image_content_type: Optional[str],
    session_id: Optional[str]
):
    """
    Generator that runs the agent and yields the response as SSE.

    After the stream completes successfully, the full user message and
    accumulated assistant response are persisted to the ``sessions``
    table's ``conversation_history`` JSONB column via a fresh
    ``AsyncSession``.
    """
    from app.core.circuit_breaker import is_circuit_open, record_success, record_failure
    from app.core.audit import log_audit_event
    import asyncio

    effective_query = _get_effective_query(query, bool(image_base64))

    # ── Circuit breaker check ──
    if is_circuit_open("agent_chat"):
        error_payload = json.dumps({
            "status": "circuit_open",
            "response": (
                "The AI service is temporarily unavailable due to repeated errors. "
                "The system will automatically retry in ~2 minutes."
            ),
            "tools_used": [],
        })
        yield f"data: {error_payload}\n\n"
        return

    MAX_RETRIES = 3
    RETRY_DELAYS = [5, 15, 30]
    last_error = None

    # Send initial "thinking" event
    yield f"data: {json.dumps({'status': 'thinking', 'message': 'Processing your query...'})}\n\n"

    for attempt in range(MAX_RETRIES):
        try:
            result = await run_agent(
                query=query,
                chat_history=chat_history,
                image_base64=image_base64,
                image_content_type=image_content_type,
                session_id=session_id,
            )

            output_text = result["output"]
            if isinstance(output_text, list):
                output_text = "".join(
                    block.get("text", "")
                    for block in output_text
                    if isinstance(block, dict)
                )
            elif not isinstance(output_text, str):
                output_text = str(output_text)

            record_success("agent_chat")

            # ── Audit Log ──
            log_audit_event(
                node="agent_chat",
                action="agent_invocation",
                result=output_text[:200],
                metadata={
                    "query": effective_query[:200],
                    "tools_used": result["tools_used"],
                    "has_image": bool(image_base64),
                    "session_id": session_id,
                    "mode": "image_only" if not query else (
                        "multimodal" if image_base64 else "text_only"
                    ),
                },
            )

            payload = json.dumps({
                "status": "success",
                "query": query,
                "response": output_text,
                "tools_used": result["tools_used"],
            })
            yield f"data: {payload}\n\n"

            # ── Persist conversation to sessions table ──
            if session_id and output_text:
                try:
                    await _persist_conversation(
                        session_id=session_id,
                        user_message=effective_query,
                        assistant_response=output_text,
                    )
                except Exception as persist_err:
                    logger.error(
                        f"Failed to persist conversation for session {session_id}: {persist_err}",
                        exc_info=True,
                    )
            return

        except Exception as e:
            last_error = e
            error_str = str(e)

            is_transient = any(
                code in error_str
                for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")
            )

            if is_transient and attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                logger.warning(
                    f"Gemini transient error (attempt {attempt + 1}/{MAX_RETRIES}), "
                    f"retrying in {delay}s: {error_str[:120]}"
                )
                yield f"data: {json.dumps({'status': 'retrying', 'attempt': attempt + 2})}\n\n"
                await asyncio.sleep(delay)
                continue

            logger.error(f"Agent error: {e}", exc_info=True)
            record_failure("agent_chat")
            break

    # All retries exhausted
    record_failure("agent_chat")
    error_payload = json.dumps({
        "status": "error",
        "response": (
            "The AI model is currently experiencing high demand. "
            "Please try again in a moment."
        ),
        "tools_used": [],
    })
    yield f"data: {error_payload}\n\n"


# ═══════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════

@router.post("/chat/stream")
async def agent_chat_stream(
    query: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    chat_history: str = Form("[]"),
    image: Optional[UploadFile] = File(None),
):
    """
    Chat with ThyraX via Server-Sent Events (streaming).
    Now supports multipart/form-data for direct file uploads.
    """
    # Parse chat_history
    try:
        history_list = json.loads(chat_history)
    except Exception:
        history_list = []

    # Process image
    image_base64 = None
    image_content_type = None
    if image:
        image_bytes = await image.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_content_type = image.content_type

    # Only run guardrail on text queries
    if query and not _is_medical_query(query):
        logger.info(f"Non-medical query rejected: {query[:80]}...")
        async def _reject():
            payload = json.dumps({
                "status": "rejected",
                "query": query,
                "response": _get_rejection_message(query),
                "tools_used": [],
            })
            yield f"data: {payload}\n\n"
        return StreamingResponse(_reject(), media_type="text/event-stream")

    return StreamingResponse(
        _stream_agent_response(query, history_list, image_base64, image_content_type, session_id),
        media_type="text/event-stream",
    )


@router.post("/chat")
async def agent_chat(
    request: AgentChatRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Refactored chat endpoint for Phase 4.
    Accepts JSON payload with patient_id, session_id, doctor_id, and user_message.
    Validates data isolation, fetches memory context, and streams Groq LLM response.
    """
    from app.schemas.memory_models import Session as SessionModel, Patient
    from app.services.memory_manager import memory_manager
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage

    patient_id_str = str(request.patient_id)
    doctor_id_str = str(request.doctor_id)

    # 1. Validate data isolation
    patient_result = await db.execute(
        select(Patient).where(Patient.patient_id == patient_id_str, Patient.doctor_id == doctor_id_str)
    )
    patient = patient_result.scalar_one_or_none()

    session_result = await db.execute(
        select(SessionModel).where(SessionModel.session_id == request.session_id, SessionModel.doctor_id == doctor_id_str)
    )
    session = session_result.scalar_one_or_none()

    if not patient or not session:
        raise HTTPException(
            status_code=403, 
            detail="Forbidden: Patient or Session does not belong to the provided Doctor."
        )

    # 2. Fetch context
    system_prompt = await memory_manager.get_injected_context(request.patient_id, request.session_id, db)

    # 3. Call Groq LLM
    keys = settings.get_groq_keys()
    if not keys:
        raise HTTPException(status_code=500, detail="No GROQ_API_KEYs found in configuration.")
    
    llm = ChatGroq(model=settings.GROQ_MODEL, api_key=keys[0], temperature=settings.LLM_TEMPERATURE)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=request.user_message)
    ]
    
    async def generate_response():
        agent_response = ""
        try:
            async for chunk in llm.astream(messages):
                if chunk.content:
                    agent_response += chunk.content
                    yield f"data: {json.dumps({'status': 'streaming', 'chunk': chunk.content})}\n\n"
            
            # Post-processing: Persist via fresh session (safe for generators)
            await _persist_conversation(
                session_id=request.session_id,
                user_message=request.user_message,
                assistant_response=agent_response,
            )
            
            # Also log audit event
            from app.core.audit import log_audit_event
            log_audit_event(
                node="agent_chat_phase4",
                action="agent_invocation",
                result=agent_response[:200],
                metadata={
                    "query": request.user_message[:200],
                    "patient_id": request.patient_id,
                    "session_id": request.session_id,
                    "doctor_id": request.doctor_id
                },
            )
            
            yield f"data: {json.dumps({'status': 'success', 'response': agent_response})}\n\n"
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            yield f"data: {json.dumps({'status': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(generate_response(), media_type="text/event-stream")