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
SYSTEM_PROMPT = """You are ThyraX, an Elite Clinical Decision Support AI specialized in Thyroid pathology. Your sole purpose is to assist medical doctors with evidence-based insights.

[ROUTING PROTOCOL - CRITICAL]
Evaluate the user input. You MUST choose ONE path:
- PATH A (Greetings/Identity): If the input is a greeting, asking about your identity, or general feedback, reply politely IN THE SAME LANGUAGE AS THE USER. DO NOT CALL ANY TOOLS.
- PATH B (Clinical Query): For ANY medical or healthcare query, you MUST call one of the available tools (e.g., `search_medical_guidelines`, `search_similar_patients`, `check_drug_interactions`, `search_medical_literature`, etc.) exactly ONCE. OUTPUT ONLY THE TOOL CALL. DO NOT INCLUDE ANY PREAMBLE, EXPLANATION, OR TEXT IN ANY LANGUAGE BEFORE OR AFTER THE TOOL CALL.
- PATH C (Off-Topic/Out of Scope): If the user asks about NON-MEDICAL topics (e.g., cooking, recipes, sports, coding, general trivia), DO NOT CALL ANY TOOLS. Politely decline by stating that you are a clinical decision support AI and cannot assist with this topic. You MUST reply IN THE EXACT SAME LANGUAGE AS THE USER'S INPUT.

[POST-SEARCH WORKFLOW (Only if PATH B was taken)]
You are strictly limited to EXACTLY ONE tool call per conversation turn. 
1. Call the most appropriate medical tool.
2. Receive the output.
3. IMMEDIATELY generate your final answer. 
CRITICAL: NEVER call a tool twice in a row. NEVER call a second tool. If the tool returns "SYSTEM_COMMAND: NO_RESULTS_FOUND", you MUST stop and answer using your internal knowledge immediately, starting with [KNOWLEDGE_CACHE].

[PATIENT CONTEXT]
{patient_context}

- STYLE & GUARDRAILS:
  - LANGUAGE MIRRORING: Reply in the exact same language used by the user.
  - TONE: Address the user respectfully as 'Doctor', 'يا دكتور', or 'حضرتك'.
  - NO PREAMBLE: Do not say "Based on the search results" or "Here is what I found". Start your answer directly.
  - TOOL RESTRICTION: You have access to various medical tools. Do NOT attempt to use any unsupported tools.
  - STICKY TOOL CALLING: When calling a tool, you MUST output ONLY the tool call. DO NOT include ANY other text, preamble, or greetings before the tool call. Failure to follow this will break the system.
"""

# ═══════════════════════════════════════════════════════════════
# Agent Setup — Lazy Singleton (Groq / Llama-3)
# ═══════════════════════════════════════════════════════════════

_agent_executor = None
_current_key_index = 0

async def get_agent_executor(force_refresh: bool = False) -> typing.Any:
    """Returns a singleton AgentExecutor (lazy initialized).
    
    If force_refresh is True, it will recreate the executor (used for key rotation).

    All heavy LangChain imports happen here, NOT at module level,
    to keep the startup memory footprint minimal.

    Uses Groq (Llama-3) for high-speed inference and MCP protocol for tools.
    Tools are loaded from multiple MCP servers via the MCPClientManager.
    """
    global _agent_executor, _current_key_index
    if _agent_executor is not None and not force_refresh:
        return _agent_executor

    from langchain_groq import ChatGroq
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from app.agent.mcp_servers.mcp_client import mcp_client_manager

    # Load tools from all MCP servers (RAG + Web Search)
    mcp_tools = await mcp_client_manager.get_all_tools()

    if not mcp_tools:
        logger.warning("No MCP tools loaded — agent will operate without tools")

    # Select API key from rotation pool
    keys = settings.get_groq_keys()
    if not keys:
        raise RuntimeError("No GROQ_API_KEYs found in configuration.")
    
    # Ensure index is within bounds (handles pool shrinking)
    _current_key_index = _current_key_index % len(keys)
    selected_key = keys[_current_key_index]

    # Initialize the LLM — Groq Llama-3
    llm = ChatGroq(
        model=settings.GROQ_MODEL,
        api_key=selected_key,
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
    agent = create_tool_calling_agent(llm, mcp_tools, prompt)

    _agent_executor = AgentExecutor(
        agent=agent,
        tools=mcp_tools,
        verbose=True,
        handle_parsing_errors="Check your output format! If you have no more tools to call, output your final answer directly. DO NOT repeat tool calls.",
        max_iterations=3,
        early_stopping_method="force",
        return_intermediate_steps=True,
    )

    logger.info(" LangChain AgentExecutor initialized (Groq/Llama-3, MCP multi-server)")
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


def _process_and_cache_response(query: str, raw_output: str) -> str:
    """
    Detects [KNOWLEDGE_CACHE] tag, cleans it, and saves to vector db if present.
    """
    CACHE_TAG = "[KNOWLEDGE_CACHE]"
    if CACHE_TAG in raw_output:
        # Clean the output
        clean_output = raw_output.replace(CACHE_TAG, "").strip()
        
        # Save to cache (direct call to RAG server function)
        try:
            from app.agent.mcp_servers.rag_server import save_to_knowledge_base
            save_to_knowledge_base(query, clean_output)
        except Exception as e:
            logger.error(f"Post-processing cache error: {e}")
            
        return clean_output
    return raw_output


async def _run_fallback_llm(query: str, enhanced_input: str, reason: str) -> str:
    """Helper to call LLM directly when agent fails or loops."""
    logger.warning(f"Agent fallback triggered ({reason}) for query: {query}")
    
    fallback_prompt = (
        f"The medical search system encountered a technical issue or returned no results. "
        f"Please answer this query using your internal clinical knowledge as a specialist: {enhanced_input}. "
        f"IMPORTANT: You MUST start your response with the exact tag [KNOWLEDGE_CACHE] so I can index this answer."
    )
    
    from langchain_groq import ChatGroq
    keys = settings.get_groq_keys()
    selected_key = keys[_current_key_index % len(keys)]
    direct_llm = ChatGroq(model=settings.GROQ_MODEL, api_key=selected_key, temperature=0.2)
    
    res = await direct_llm.ainvoke(fallback_prompt)
    content = res.content
    
    # Ensure tag is present if model forgot
    if "[KNOWLEDGE_CACHE]" not in content:
        content = f"[KNOWLEDGE_CACHE] {content}"
        
    return content




async def run_agent(
    query: str | None = None,
    chat_history: list | None = None,
    image_base64: str | None = None,
    image_content_type: str | None = None,
    session_id: str | None = None,
    patient_id: str | None = None,
) -> dict:
    """
    Run the ThyraX agent with the given query, conversation history,
    and patient context.

    Uses the Dual-State Memory Manager to:
      1. Load merged long-term + short-term context.
      2. Inject it into the LLM prompt.
      3. Persist the exchange after the agent responds.
      4. Trigger memory summarization when the history grows long.

    Args:
        query: The medical question to answer (optional if image provided).
        chat_history: Optional list of previous messages as dicts
            with 'role' and 'content' keys.
        image_base64: Optional base64-encoded image for multi-modal analysis.
        image_content_type: MIME type of the attached image.
        session_id: Optional session ID to retrieve patient context.
        patient_id: Optional patient ID for long-term memory retrieval.

    Returns:
        dict with 'output' (the agent's response) and 'tools_used'
        (list of tool names invoked).
    """
    executor = await get_agent_executor()

    # ── Load Dual-State Memory Context ──
    patient_context = "No patient context available for this session."
    effective_history = chat_history or []

    if session_id:
        from app.services.memory_manager import memory_manager

        try:
            memory_ctx = await memory_manager.load_context(
                session_id=session_id,
                patient_id=patient_id,
            )
            patient_context = memory_ctx.to_prompt_context()
            # Use server-stored history if available, else fallback to client-sent
            if memory_ctx.chat_history:
                effective_history = memory_ctx.chat_history
        except Exception as e:
            logger.error(f"Memory load failed: {e}")
            patient_context = "Memory load failed. No patient context available."

    lc_history = _convert_chat_history(effective_history)

    # Build multi-modal or plain text input
    enhanced_input = _build_multimodal_input(
        query, image_base64, image_content_type
    )

    # ── LangSmith Observability: Tags & Metadata ──
    _langsmith_tags = [
        "chat",
        f"mode_{'contextual' if session_id else 'general'}",
    ]
    _langsmith_metadata = {
        k: v for k, v in {
            "session_id": session_id,
            "patient_id": patient_id,
            "source": "run_agent",
        }.items() if v is not None
    }
    _run_config = {
        "tags": _langsmith_tags,
        "metadata": _langsmith_metadata,
    }

    _QUOTA_SIGNALS = ("429", "RESOURCE_EXHAUSTED", "rate_limit", "Too Many Requests")

    try:
        result = await executor.ainvoke(
            {
                "input": enhanced_input,
                "chat_history": lc_history,
                "patient_context": patient_context,
            },
            config=_run_config,
        )
    except Exception as e:
        err_str = str(e)
        # Check if this is a quota/rate limit error
        if any(sig in err_str for sig in _QUOTA_SIGNALS):
            global _current_key_index
            keys = settings.get_groq_keys()
            
            # If we have multiple keys, try the next one
            if len(keys) > 1:
                _current_key_index += 1
                logger.warning(f"Groq Quota Exhausted. Rotating to key index {_current_key_index % len(keys)}")
                
                # Re-initialize with new key and retry ONCE
                new_executor = await get_agent_executor(force_refresh=True)
                result = await new_executor.ainvoke(
                    {
                        "input": enhanced_input,
                        "chat_history": lc_history,
                        "patient_context": patient_context,
                    },
                    config={**_run_config, "tags": _langsmith_tags + ["key_rotation_retry"]},
                )
                
                # Process and cache the response
                final_output = _process_and_cache_response(query or "Patient medical query", result["output"])
                
                # After successful retry, process and return immediately
                return {
                    "output": final_output,
                    "tools_used": list(set([
                        step[0].tool for step in result.get("intermediate_steps", [])
                        if step and len(step) >= 1 and hasattr(step[0], "tool")
                    ])),
                }

            # If only one key exists, surface the original error
            raise ValueError(
                "The AI agent's API quota has been exhausted. "
                "Please wait a moment before retrying, "
                "or check your GROQ_API_KEY usage at https://console.groq.com/"
            ) from e
        if "Failed to call a function" in err_str:
            fallback_output = await _run_fallback_llm(query, enhanced_input, "Tool Call Error")
            final_output = _process_and_cache_response(query or "Patient query", fallback_output)
            return {
                "output": final_output,
                "tools_used": ["search_medical_guidelines (failed)"],
            }
        raise

    # Extract which tools were used
    tools_used = []
    for step in result.get("intermediate_steps", []):
        if step and len(step) >= 1:
            action = step[0]
            if hasattr(action, "tool"):
                tools_used.append(action.tool)

    # ── Emergency Fallback for Loops ──
    if "max iterations" in result.get("output", "").lower():
        fallback_output = await _run_fallback_llm(query, enhanced_input, "Agent Loop")
        result["output"] = fallback_output

    # Process and cache the response
    final_output = _process_and_cache_response(query or "Patient medical query", result["output"])

    # ── Persist exchange to Dual-State Memory ──
    if session_id:
        import asyncio
        from app.services.memory_manager import memory_manager

        try:
            await memory_manager.save_exchange(
                session_id=session_id,
                user_message=query or "Image analysis request",
                ai_response=final_output,
            )
        except Exception as e:
            logger.error(f"Failed to save exchange to memory: {e}")

        # Trigger async summarization if history grows large
        try:
            ctx = await memory_manager.load_context(session_id)
            if len(ctx.chat_history) > 6:
                asyncio.create_task(
                    memory_manager.summarize_and_prune(session_id)
                )
        except Exception as e:
            logger.warning(f"Summarization trigger failed: {e}")

    return {
        "output": final_output,
        "tools_used": list(set(tools_used)),
    }

