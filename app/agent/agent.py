"""
LangChain Agent for the ThyraX CDSS (Node 5).

Powered by Groq (Llama-3) for high-speed, low-cost inference.

Features:
  - Open medical scope (all specialties, not just thyroid).
  - Multi-modal context: text + patient state + RAG + web search.
  - Strict RAG-first directive: always search guidelines before answering.
  - Conversational memory via chat_history message list.
  - Dynamic patient context injection from the State Manager.

Memory Management:
  - Agent executor is lazily initialized on first call and cached.
  - LangChain imports are inside function bodies.
"""
import logging
import typing

from app.core.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# System Prompt — Open Medical Scope + Strict RAG + Patient Context
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are ThyraX, an advanced Clinical Decision Support AI assistant. Your user is a medical doctor (physician/endocrinologist). You must speak with utmost professional respect, always addressing the user as 'Doctor' (e.g., 'يا دكتور', 'حضرتك'). Provide concise, evidence-based medical insights. You are fluent in professional Arabic and English medical terminology.
CRITICAL RULE: You must ALWAYS end your response by asking the doctor how you can further assist with the patient's case, or what the next diagnostic step should be. (e.g., in Arabic: 'كيف يمكنني مساعدتك في هذه الحالة يا دكتور؟', 'هل تحب أن أستعرض لك التقرير الطبي المفصل؟', or 'أقدر أساعد حضرتك في إيه تاني؟').

## CURRENT PATIENT CONTEXT
{patient_context}

## Your Tools (USE THEM IN THIS ORDER)
You have TWO search tools. You MUST follow this strict order:

1. **search_medical_guidelines** — Searches the LOCAL medical knowledge base (ChromaDB) containing ingested clinical guidelines, ATA protocols, published literature, drug references, and evidence-based documents.
2. **search_medical_web** — Searches the internet via DuckDuckGo, prioritizing trusted medical sources (PubMed, WHO, NIH, Mayo Clinic, ATA, UpToDate, medical journals).

## MANDATORY SEARCH WORKFLOW — FOLLOW THIS EXACTLY

For EVERY clinical/medical question, you MUST follow these steps:

**Step 1 — ALWAYS search local RAG first:**
Call `search_medical_guidelines` with a well-formed, specific query.

**Step 2 — Check RAG results:**
- ✅ If RAG returns relevant guidelines/documents → Base your answer on them. Cite the source. STOP here (no web search needed).
- ❌ If RAG returns "NO RESULTS", "empty", or irrelevant content → Proceed to Step 3.

**Step 3 — Search the web as fallback:**
Call `search_medical_web` with the medical query. This searches trusted medical websites.
- ✅ If web search returns results → Use them to answer. Cite the URLs.
- ❌ If web search also fails → Proceed to Step 4.

**Step 4 — General medical knowledge (last resort):**
Only if BOTH tools returned no results, answer from your training data. You MUST clearly state:
"No specific guidelines were found in the knowledge base or online sources. Based on general medical knowledge: ..."

## CRITICAL RULES
- NEVER skip Step 1. Even if you think you know the answer, search RAG first.
- NEVER call search_medical_web BEFORE search_medical_guidelines.
- NEVER fabricate citations, guideline numbers, or study references.
- ALWAYS label your sources: [From Knowledge Base], [From Web Search], or [From General Medical Knowledge].
- When patient context is available, ALWAYS correlate your answer with the patient's current diagnostic state.

## Patient Context Usage
When the CURRENT PATIENT CONTEXT section above contains data:
- Reference the patient's existing results in your answers.
- Correlate new findings with prior diagnostic steps.
- Provide recommendations that account for the full diagnostic journey.
- If the patient has ultrasound results showing TI-RADS 4/5, prioritize FNA guidance.
- If the patient has FNAC results, correlate Bethesda category with prior imaging.

## Conversation Memory
You have access to the full conversation history. Use it to:
- Maintain continuity across questions on the same topic.
- Avoid asking the user to repeat information.
- Reference earlier findings when relevant.

## Clinical Response Standards
- Present information in a structured, clinically relevant format.
- Clearly distinguish between **evidence from sources** (cite them) and **AI analysis** (label it).
- Frame findings as "suggestive of" or "consistent with" — NEVER provide definitive diagnoses.
- NEVER recommend specific drug dosages or surgical procedures.

## STRICT MEDICAL GUARDRAIL
You ONLY answer questions related to medicine, healthcare, clinical diagnostics, pharmacology, and biomedical sciences.
If a user asks about coding, recipes, sports, entertainment, politics, or ANY non-medical topic, respond ONLY with:
"I appreciate your question, but I'm ThyraX — a medical AI assistant. I can only help with medical and clinical questions. Please rephrase your question in a healthcare context."
"""


# ═══════════════════════════════════════════════════════════════
# Agent Setup — Lazy Singleton (Groq / Llama-3)
# ═══════════════════════════════════════════════════════════════

_agent_executor = None


def get_agent_executor() -> typing.Any:
    """Returns a singleton AgentExecutor (lazy initialized).

    All heavy LangChain imports happen here, NOT at module level,
    to keep the startup memory footprint minimal.

    Uses Groq (Llama-3) for high-speed inference.
    """
    global _agent_executor
    if _agent_executor is not None:
        return _agent_executor

    from langchain_groq import ChatGroq
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from app.agent.tools import ALL_TOOLS

    if not settings.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file. "
            "Get a free key at https://console.groq.com/"
        )

    # Initialize the LLM — Groq Llama-3
    llm = ChatGroq(
        model=settings.GROQ_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=2048,
    )

    # Build the prompt — chat_history and patient_context are required
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create the agent
    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)

    _agent_executor = AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
        return_intermediate_steps=True,
    )

    logger.info("✅ LangChain AgentExecutor initialized (Groq/Llama-3, lazy singleton)")
    return _agent_executor


def _convert_chat_history(raw_history: list) -> list:
    """
    Convert raw chat history dicts into LangChain message objects.

    Accepts a list of dicts like:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

    Returns a list of HumanMessage / AIMessage objects that LangChain
    can process in the prompt's MessagesPlaceholder.
    """
    from langchain_core.messages import HumanMessage, AIMessage

    messages = []
    for msg in raw_history:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        if not content:
            continue
        if role == "user" or role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "assistant" or role == "ai":
            messages.append(AIMessage(content=content))
        # Skip system or unknown roles
    return messages


def _build_multimodal_input(
    query: str | None,
    image_base64: str | None = None,
    image_content_type: str | None = None,
) -> str | list:
    """
    Build the agent input.

    Handles three modes:
      - Text only:  returns the query string
      - Image only: returns multimodal block with default analysis prompt
      - Both:       returns multimodal block with user's query

    Note: Groq/Llama-3 is text-only. Image analysis requires a vision model.
    When images are provided, we describe the context but defer vision to
    dedicated endpoints (Node 4, FNAC).
    """
    # Default prompt when only an image is sent
    effective_query = query or (
        "Analyze this medical image in detail. "
        "Identify any clinical findings, extract values if it's a lab report, "
        "and provide a structured medical interpretation."
    )

    if not image_base64:
        return effective_query

    # Groq/Llama-3 is text-only — include image context as text reference
    return (
        f"{effective_query}\n\n"
        "[Note: An image was provided with this query. "
        "Image analysis has been handled by the dedicated vision pipeline. "
        "Please focus on the text query and patient context.]"
    )


async def run_agent(
    query: str | None = None,
    chat_history: list | None = None,
    image_base64: str | None = None,
    image_content_type: str | None = None,
    session_id: str | None = None,
) -> dict:
    """
    Run the ThyraX agent with the given query, conversation history,
    and patient context.

    Args:
        query: The medical question to answer (optional if image provided).
        chat_history: Optional list of previous messages as dicts
            with 'role' and 'content' keys.
        image_base64: Optional base64-encoded image for multi-modal analysis.
        image_content_type: MIME type of the attached image.
        session_id: Optional session ID to retrieve patient context.

    Returns:
        dict with 'output' (the agent's response) and 'tools_used'
        (list of tool names invoked).
    """
    executor = get_agent_executor()

    # Convert raw history dicts to LangChain message objects
    lc_history = _convert_chat_history(chat_history or [])

    # Build multi-modal or plain text input
    enhanced_input = _build_multimodal_input(
        query, image_base64, image_content_type
    )

    # ── Retrieve patient context from State Manager ──
    patient_context = "No patient context available for this session."
    if session_id:
        from app.services.patient_state import state_manager
        patient_context = state_manager.get_state_summary(session_id)

    _QUOTA_SIGNALS = ("429", "RESOURCE_EXHAUSTED", "rate_limit", "Too Many Requests")

    try:
        result = await executor.ainvoke({
            "input": enhanced_input,
            "chat_history": lc_history,
            "patient_context": patient_context,
        })
    except Exception as e:
        err_str = str(e)
        # Surface quota exhaustion as a clean ValueError the router can handle
        if any(sig in err_str for sig in _QUOTA_SIGNALS):
            raise ValueError(
                "The AI agent's API quota has been exhausted. "
                "Please wait a moment before retrying, "
                "or check your GROQ_API_KEY usage at https://console.groq.com/"
            ) from e
        raise  # preserve original traceback for unexpected errors

    # Extract which tools were used
    tools_used = []
    for step in result.get("intermediate_steps", []):
        if step and len(step) >= 1:
            action = step[0]
            if hasattr(action, "tool"):
                tools_used.append(action.tool)

    return {
        "output": result["output"],
        "tools_used": list(set(tools_used)),
    }
