import logging
from functools import lru_cache
from langchain_groq import ChatGroq
from app.core.config import settings

logger = logging.getLogger(__name__)

@lru_cache()
def get_shared_llm(temperature: float = 0.1, max_tokens: int = 512) -> ChatGroq:
    """
    Returns a cached instance of ChatGroq for non-agentic tasks (e.g. explanations).
    This avoids redundant initialization overhead on every request.
    """
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not configured")

    logger.debug("Initializing shared ChatGroq LLM client")
    return ChatGroq(
        model=settings.GROQ_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
    )

async def generate_llm_explanation(
    circuit_name: str,
    system_msg: str,
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> str | None:
    """
    Generate a strict LLM explanation.
    Returns None if the LLM is unavailable or a circuit breaker is open.
    """
    from app.core.circuit_breaker import is_circuit_open, record_success, record_failure

    if is_circuit_open(circuit_name):
        logger.info(f"Circuit OPEN for {circuit_name} — skipping LLM explanation")
        return None

    try:
        from langchain_core.messages import SystemMessage
        llm = get_shared_llm(temperature=temperature, max_tokens=max_tokens)
        response = await llm.ainvoke([SystemMessage(content=system_msg)])
        record_success(circuit_name)
        return response.content.strip()
    except Exception as e:
        record_failure(circuit_name)
        logger.warning(f"LLM explanation failed ({e}), skipping")
        return None
