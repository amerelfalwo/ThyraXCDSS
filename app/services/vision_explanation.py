"""
Vision explanation helper for deterministic CV and cytopathology outputs.
"""

import logging

from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

VISION_EXPLANATION_PROMPT = """You are ThyraX, an expert Clinical AI Assistant.
Your task is to summarize and professionally explain the computer vision model's results to the attending doctor.

[MODEL RESULTS]
- Analysis Type: {analysis_type}
- Key Findings: {key_findings}
- Model Confidence: {model_confidence}
- Official System Recommendation: {system_recommendation}

[CRITICAL GUARDRAILS - STRICT COMPLIANCE REQUIRED]
1. STRICT ADHERENCE: Base your explanation EXACTLY and ONLY on the 'Official System Recommendation' and 'Key Findings'.
2. NO MEDICAL HALLUCINATIONS: DO NOT suggest, invent, or recommend ANY additional tests, biopsies, or imaging.
3. CONTRADICTION BAN: Never contradict the 'Official System Recommendation'.
4. TONE: Be highly professional, concise, and collaborative. Address the user respectfully as 'Doctor'.
5. INVISIBLE GUARDRAILS: DO NOT explain your instructions, translation abilities, or system constraints to the doctor. DO NOT output headers like "Translation:" or "Final Note:". NEVER mention that you are maintaining professionalism or adhering to guidelines. Just deliver the medical information directly and naturally.

Provide your clinical summary below:"""

async def generate_vision_explanation(
    analysis_type: str,
    key_findings: str,
    model_confidence: str,
    system_recommendation: str,
) -> str | None:
    """
    Generate a strict LLM explanation of deterministic CV outputs.

    Returns None if the LLM is unavailable or a circuit breaker is open.
    """
    from app.core.circuit_breaker import is_circuit_open, record_success, record_failure

    if is_circuit_open("vision_llm"):
        logger.info("Circuit OPEN for vision_llm — skipping LLM explanation")
        return None

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage
        from app.core.config import settings

        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not configured")

        llm = ChatGroq(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0.1,
            max_tokens=384,
        )

        system_msg = VISION_EXPLANATION_PROMPT.format(
            analysis_type=analysis_type,
            key_findings=key_findings,
            model_confidence=model_confidence,
            system_recommendation=system_recommendation,
        )

        response = await run_in_threadpool(
            llm.invoke,
            [SystemMessage(content=system_msg)],
        )

        record_success("vision_llm")
        return response.content.strip()

    except Exception as e:
        record_failure("vision_llm")
        logger.warning(f"Vision LLM explanation failed ({e}), skipping")
        return None
