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

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.core.security import verify_internal_api_key
from app.schemas.chat import ChatRequest, ChatResponse

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

_REJECTION_RESPONSE = (
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

    Returns True if the query appears medical, False if it's clearly
    non-medical (coding, recipes, sports, etc.).
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
    # (the agent's system prompt has its own guardrails)
    return True


def _get_effective_query(req: ChatRequest) -> str:
    """
    Derive the effective text query for guardrails and logging.

    - Text mode:  returns query directly
    - Image mode: returns a default medical analysis prompt
    - Both mode:  returns query directly
    """
    if req.query:
        return req.query
    return "Analyze the attached medical image"


# ═══════════════════════════════════════════════════════════════
# Streaming Generator
# ═══════════════════════════════════════════════════════════════

async def _stream_agent_response(req: ChatRequest):
    """
    Generator that runs the agent and yields the full response
    as a single SSE-style data chunk. For true token-level streaming,
    the LangChain agent would need a streaming callback; this approach
    provides immediate "processing" feedback + final result.
    """
    from app.core.circuit_breaker import is_circuit_open, record_success, record_failure
    from app.core.audit import log_audit_event
    import asyncio

    effective_query = _get_effective_query(req)

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
            from app.agent.agent import run_agent

            history_dicts = [
                {"role": msg.role, "content": msg.content}
                for msg in (req.chat_history or [])
            ]

            result = await run_agent(
                query=req.query,
                chat_history=history_dicts,
                image_base64=req.image_base64,
                image_content_type=req.image_content_type,
                session_id=req.session_id,
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
                    "has_image": bool(req.image_base64),
                    "session_id": req.session_id,
                    "mode": "image_only" if not req.query else (
                        "multimodal" if req.image_base64 else "text_only"
                    ),
                },
            )

            payload = json.dumps({
                "status": "success",
                "query": req.query,
                "response": output_text,
                "tools_used": result["tools_used"],
            })
            yield f"data: {payload}\n\n"
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
async def agent_chat_stream(req: ChatRequest):
    """
    Chat with ThyraX via Server-Sent Events (streaming).

    Supports three input modes:
      - Text only:   `{"query": "What is TSH?"}`
      - Image only:  `{"image_base64": "...", "image_content_type": "image/png"}`
      - Multimodal:  `{"query": "Interpret this", "image_base64": "..."}`

    Returns a StreamingResponse with SSE events:
      - `{"status": "thinking"}` — query received, processing
      - `{"status": "retrying", "attempt": N}` — transient error, retrying
      - `{"status": "success", "response": "...", "tools_used": [...]}` — final result
      - `{"status": "error", "response": "..."}` — all retries exhausted
    """
    # Only run guardrail on text queries; image-only requests are always medical
    if req.query and not _is_medical_query(req.query):
        logger.info(f"Non-medical query rejected: {req.query[:80]}...")
        async def _reject():
            payload = json.dumps({
                "status": "rejected",
                "query": req.query,
                "response": _REJECTION_RESPONSE,
                "tools_used": [],
            })
            yield f"data: {payload}\n\n"
        return StreamingResponse(_reject(), media_type="text/event-stream")

    return StreamingResponse(
        _stream_agent_response(req),
        media_type="text/event-stream",
    )


@router.post("/chat", response_model=ChatResponse)
async def agent_chat(req: ChatRequest):
    """
    Chat with the ThyraX AI medical assistant (Node 5).

    Supports three input modes:
      - Text only:   `{"query": "What is TSH?"}`
      - Image only:  `{"image_base64": "...", "image_content_type": "image/png"}`
      - Multimodal:  `{"query": "Interpret this", "image_base64": "..."}`

    Standard JSON response (non-streaming). For real-time UX,
    use POST /agent/chat/stream instead.
    """
    effective_query = _get_effective_query(req)

    # ── Medical Guardrail Pre-filter (only for text queries) ──
    if req.query and not _is_medical_query(req.query):
        logger.info(f"Non-medical query rejected: {req.query[:80]}...")
        return ChatResponse(
            status="rejected",
            query=req.query,
            response=_REJECTION_RESPONSE,
            tools_used=[],
        )

    from app.core.circuit_breaker import is_circuit_open, record_success, record_failure
    from app.core.audit import log_audit_event
    import asyncio

    if is_circuit_open("agent_chat"):
        raise HTTPException(
            status_code=503,
            detail="AI service temporarily unavailable (circuit breaker open). Retrying in ~2 minutes.",
        )

    MAX_RETRIES = 3
    RETRY_DELAYS = [5, 15, 30]
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            from app.agent.agent import run_agent

            history_dicts = [
                {"role": msg.role, "content": msg.content}
                for msg in (req.chat_history or [])
            ]

            result = await run_agent(
                query=req.query,
                chat_history=history_dicts,
                image_base64=req.image_base64,
                image_content_type=req.image_content_type,
                session_id=req.session_id,
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

            log_audit_event(
                node="agent_chat",
                action="agent_invocation",
                result=output_text[:200],
                metadata={
                    "query": effective_query[:200],
                    "tools_used": result["tools_used"],
                    "has_image": bool(req.image_base64),
                    "session_id": req.session_id,
                    "mode": "image_only" if not req.query else (
                        "multimodal" if req.image_base64 else "text_only"
                    ),
                },
            )

            return ChatResponse(
                status="success",
                query=req.query,
                response=output_text,
                tools_used=result["tools_used"],
            )

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
                await asyncio.sleep(delay)
                continue

            logger.error(f"Agent error: {e}", exc_info=True)
            record_failure("agent_chat")
            break

    record_failure("agent_chat")
    is_overload = any(
        code in str(last_error)
        for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")
    )
    if is_overload:
        raise HTTPException(
            status_code=503,
            detail=(
                "The AI model is currently experiencing high demand. "
                "Please try again in a moment."
            ),
        )
    raise HTTPException(status_code=500, detail=f"Agent error: {last_error}")