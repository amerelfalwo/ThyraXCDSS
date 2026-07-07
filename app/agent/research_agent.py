"""
LangChain Agent for the ThyraX CDSS (Node 8 - General Medical Research).

Powered by Groq (Llama-3) for high-speed, low-cost inference.

Features:
  - Open medical scope (all specialties).
  - Multi-modal context: text + RAG + web search.
  - Strict RAG-first directive: always search guidelines before answering.
  - Conversational memory via chat_history message list.
  - Highly interactive, transparent reasoning, and professional medical tone.

Memory Management:
  - Agent executor is lazily initialized on first call and cached.
  - LangChain imports are inside function bodies.
"""
import logging
import typing

from app.core.config import settings

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════

# System Prompt — General Medical Researcher + Transparent Reasoning

# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are ThyraX (Node 8), an Elite Medical Research AI and Clinical Knowledge Assistant.

*** SCOPE & BOUNDARY RULES ***
1. MEDICAL DOMAIN ONLY: You answer ONLY medical, clinical, and health-science questions. Your expertise covers all medical specialties including endocrinology, oncology, radiology, pathology, pharmacology, surgery, internal medicine, and related fields.
2. NON-MEDICAL REJECTION: If the user asks anything outside medicine or health science (programming, cooking, politics, math, jokes, general knowledge, sports, entertainment, etc.), respond ONLY with: "أنا متخصص في البحث الطبي فقط. لا أستطيع المساعدة في هذا الموضوع. / I am a medical research assistant only. I cannot help with this topic."
3. GREETINGS: Reply naturally in user's language.
4. MEDICAL QUERY: Use tools to search for guidelines or medical knowledge when necessary.
5. POST-SEARCH: Use the search results to provide a comprehensive, evidence-based answer. If NO_RESULTS_FOUND, use internal medical knowledge.
6. NO PATIENT DATA: You do NOT have access to any specific patient's data. If the user asks about a specific patient's results, redirect them: "لتحليل بيانات مريض محدد، يرجى استخدام محادثة المريض (Node 7). / To analyze a specific patient's data, please use Patient Chat (Node 7)."

[CRITICAL OUTPUT RULES]
- NEVER mention the tools you are using.
- NEVER describe the process of searching or retrieving data.
- Do NOT say "I will search...", "Based on search results...", or "I am calling a function...".
- Go directly to the medical answer as a professional consultant would.
- If you need to perform a search, do it silently in the background and output only the synthesis.

STYLE:
- RESEARCHER: Walk the user explicitly through your thought process and differential diagnosis.
- LANGUAGE: Match user's language natively. Mix English medical terms if Arabic.
- TONE: Professional, analytical. Address as colleague ('يا دكتور').
- DIRECT: No preamble ("Based on..."). Just answer.
"""
# ═══════════════════════════════════════════════════════════════
# Agent Setup — Lazy Singleton (Groq / Llama-3)
# ═══════════════════════════════════════════════════════════════

_research_agent_executor = None
_current_key_index = 0

async def get_research_agent_executor(force_refresh: bool = False) -> typing.Any:
    """Returns a singleton AgentExecutor (lazy initialized).
    
    If force_refresh is True, it will recreate the executor (used for key rotation).

    All heavy LangChain imports happen here, NOT at module level,
    to keep the startup memory footprint minimal.

    Uses Groq (Llama-3) for high-speed inference and MCP protocol for tools.
    Tools are loaded from multiple MCP servers via the MCPClientManager.
    """
    global _research_agent_executor, _current_key_index
    if _research_agent_executor is not None and not force_refresh:
        return _research_agent_executor

    from langchain_groq import ChatGroq
    from app.core.http_client import get_shared_async_client
    from langgraph.prebuilt import create_react_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from app.agent.mcp_servers.mcp_client import mcp_client_manager

    # Load tools from all MCP servers (RAG + Web Search)
    mcp_tools = await mcp_client_manager.get_all_tools()

    if not mcp_tools:
        logger.warning("No MCP tools loaded — agent will operate without tools")

    keys = settings.get_groq_keys()
    if not keys:
        raise RuntimeError("No GROQ_API_KEYs found in configuration.")
    
    # Ensure index is within bounds (handles pool shrinking)
    _current_key_index = _current_key_index % len(keys)
    selected_key = keys[_current_key_index]

    # Initialize the LLM — Groq
    llm = ChatGroq(
        model=settings.GROQ_MODEL,
        api_key=selected_key,
        temperature=settings.LLM_TEMPERATURE,
    )

    _research_agent_executor = create_react_agent(
        model=llm,
        tools=mcp_tools,
    )

    logger.info(" LangChain Research AgentExecutor initialized (Groq/Llama-3, MCP multi-server)")
    return _research_agent_executor


def _convert_chat_history(raw_history: list) -> list:
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
    return messages


def _build_multimodal_input(
    query: str | None,
    image_base64: str | None = None,
    image_content_type: str | None = None,
) -> str | list:
    effective_query = query or (
        "Analyze this medical image in detail. "
        "Identify any clinical findings, extract values if it's a lab report, "
        "and provide a structured medical interpretation."
    )

    if not image_base64:
        return effective_query

    return (
        f"{effective_query}\n\n"
        "[Note: An image was provided with this query. "
        "Image analysis has been handled by the dedicated vision pipeline. "
        "Please focus on the text query.]"
    )


def _process_and_cache_response(query: str, raw_output: str) -> str:
    CACHE_TAG = "[KNOWLEDGE_CACHE]"
    if CACHE_TAG in raw_output:
        clean_output = raw_output.replace(CACHE_TAG, "").strip()
        try:
            from app.agent.mcp_servers.rag_server import save_to_knowledge_base
            save_to_knowledge_base(query, clean_output)
        except Exception as e:
            logger.error(f"Post-processing cache error: {e}")
        return clean_output
    return raw_output


async def _run_fallback_llm(query: str, enhanced_input: str, reason: str) -> str:
    logger.warning(f"Research Agent fallback triggered ({reason}) for query: {query}")
    
    fallback_prompt = (
        f"You are ThyraX (Node 8), an Elite Medical Research AI. "
        f"SCOPE: You answer ONLY medical and health-science questions. "
        f"If the query is non-medical, respond ONLY with: 'I am a medical research assistant only. I cannot help with this topic.'\n\n"
        f"The medical search system encountered a technical issue or returned no results. "
        f"Please answer this query using your internal medical knowledge as an expert researcher: {enhanced_input}. "
        f"IMPORTANT: You MUST start your response with the exact tag [KNOWLEDGE_CACHE] so I can index this answer.\n\n"
        f"You MUST reply IN THE EXACT SAME LANGUAGE AS THE USER'S INPUT. "
        f"Adopt an interactive, highly analytical research style. Show your reasoning explicitly. "
        f"If replying in Arabic, use natural medical phrasing and keep English terms for technical accuracy where appropriate."
    )
    
    from langchain_groq import ChatGroq
    from app.core.http_client import get_shared_async_client
    keys = settings.get_groq_keys()
    selected_key = keys[_current_key_index % len(keys)]
    direct_llm = ChatGroq(
        model=settings.GROQ_MODEL, 
        api_key=selected_key, 
        temperature=0.2
    )
    
    res = await direct_llm.ainvoke(fallback_prompt)
    content = res.content
    
    if "[KNOWLEDGE_CACHE]" not in content:
        content = f"[KNOWLEDGE_CACHE] {content}"
        
    return content


async def run_agent(
    query: str | None = None,
    chat_history: list | None = None,
    image_base64: str | None = None,
    image_content_type: str | None = None,
    session_id: str | None = None,
) -> dict:
    executor = await get_research_agent_executor()

    # ── Load Dual-State Memory Context ──
    effective_history = chat_history or []

    if session_id:
        from app.services.memory_manager import memory_manager
        try:
            memory_ctx = await memory_manager.load_context(
                session_id=session_id,
                patient_id=None,
            )
            if memory_ctx.chat_history:
                effective_history = memory_ctx.chat_history
        except Exception as e:
            logger.error(f"Memory load failed: {e}")

    lc_history = _convert_chat_history(effective_history)

    enhanced_input = _build_multimodal_input(
        query, image_base64, image_content_type
    )

    _langsmith_tags = [
        "chat",
        "mode_research",
    ]
    _langsmith_metadata = {
        k: v for k, v in {
            "session_id": session_id,
            "source": "run_research_agent",
        }.items() if v is not None
    }
    _run_config = {
        "tags": _langsmith_tags,
        "metadata": _langsmith_metadata,
    }

    _QUOTA_SIGNALS = ("429", "RESOURCE_EXHAUSTED", "rate_limit", "Too Many Requests")

    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + lc_history + [HumanMessage(content=enhanced_input)]

    try:
        result = await executor.ainvoke(
            {
                "messages": messages,
            },
            config=_run_config,
        )
    except Exception as e:
        err_str = str(e)
        if any(sig in err_str for sig in _QUOTA_SIGNALS):
            global _current_key_index
            keys = settings.get_groq_keys()
            
            if len(keys) > 1:
                _current_key_index += 1
                logger.warning(f"Groq Quota Exhausted. Rotating to key index {_current_key_index % len(keys)}")
                
                new_executor = await get_research_agent_executor(force_refresh=True)
                result = await new_executor.ainvoke(
                    {
                        "messages": messages,
                    },
                    config={**_run_config, "tags": _langsmith_tags + ["key_rotation_retry"]},
                )
                
                output_content = result["messages"][-1].content
                final_output = _process_and_cache_response(query or "General medical query", output_content)
                
                tools_used = []
                for msg in result.get("messages", []):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tools_used.append(tc.get("name"))

                return {
                    "output": final_output,
                    "tools_used": list(set(tools_used)),
                }

            raise ValueError(
                "The AI agent's API quota has been exhausted. "
                "Please wait a moment before retrying, "
                "or check your GROQ_API_KEY usage at https://console.groq.com/"
            ) from e
        if "Failed to call a function" in err_str:
            fallback_output = await _run_fallback_llm(query, enhanced_input, "Tool Call Error")
            final_output = _process_and_cache_response(query or "General query", fallback_output)
            return {
                "output": final_output,
                "tools_used": ["tool_call_failed"],
            }
        raise

    tools_used = []
    for msg in result.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_used.append(tc.get("name"))

    output_content = result["messages"][-1].content
    if "max iterations" in output_content.lower():
        fallback_output = await _run_fallback_llm(query, enhanced_input, "Agent Loop")
        output_content = fallback_output

    final_output = _process_and_cache_response(query or "General medical query", output_content)

    if "search_medical_web" in tools_used and "[KNOWLEDGE_CACHE]" not in output_content:
        try:
            from app.agent.mcp_servers.rag_server import save_to_knowledge_base
            save_to_knowledge_base(query or "Web Search", final_output)
            logger.info("Auto-embedded web search results into RAG knowledge base.")
        except Exception as e:
            logger.error(f"Failed to auto-embed web search: {e}")

    # ── Persist exchange to Dual-State Memory ──
    if session_id:
        import asyncio
        from app.services.memory_manager import memory_manager

        try:
            await memory_manager.save_exchange(
                session_id=session_id,
                user_message=query or "General analysis request",
                ai_response=final_output,
            )
        except Exception as e:
            logger.error(f"Failed to save exchange to memory: {e}")

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


async def stream_research_agent(
    query: str | None = None,
    chat_history: list | None = None,
    image_base64: str | None = None,
    image_content_type: str | None = None,
    session_id: str | None = None,
    fast_path: bool = False,
    background_tasks: typing.Any = None,
) -> typing.AsyncGenerator[str, None]:
    global _current_key_index
    import asyncio
    from app.services.semantic_cache import check_semantic_cache, save_semantic_cache

    if query and not image_base64:
        def _check_cache():
            return check_semantic_cache(query)
        cached_response = await asyncio.get_running_loop().run_in_executor(None, _check_cache)
        
        if cached_response:
            logger.info("Semantic cache hit, bypassing Agent/LLM.")
            chunk_size = 20
            for i in range(0, len(cached_response), chunk_size):
                yield cached_response[i:i+chunk_size]
                
            if session_id:
                from app.services.memory_manager import memory_manager
                async def _background_save_cached():
                    try:
                        await memory_manager.save_exchange(session_id, query, cached_response)
                        ctx = await memory_manager.load_context(session_id)
                        if len(ctx.chat_history) > 6:
                            await memory_manager.summarize_and_prune(session_id)
                    except Exception as e:
                        logger.error(f"Background save for cached exchange failed: {e}")
                
                if background_tasks:
                    background_tasks.add_task(_background_save_cached)
                else:
                    asyncio.create_task(_background_save_cached())
            return
    
    effective_history = chat_history or []

    async def _get_memory_ctx():
        if not session_id:
            return None
        from app.services.memory_manager import memory_manager
        return await memory_manager.load_context(session_id=session_id, patient_id=None)

    executor, memory_ctx = await asyncio.gather(
        get_research_agent_executor(),
        _get_memory_ctx(),
        return_exceptions=True
    )
    
    if isinstance(executor, Exception):
        logger.error(f"Failed to load executor: {executor}")
        raise executor

    if session_id:
        if isinstance(memory_ctx, Exception):
            logger.error(f"Memory load failed: {memory_ctx}")
        elif memory_ctx and memory_ctx.chat_history:
            effective_history = memory_ctx.chat_history

    lc_history = _convert_chat_history(effective_history)
    enhanced_input = _build_multimodal_input(query, image_base64, image_content_type)

    _langsmith_tags = ["chat", "mode_research", "streaming"]
    _langsmith_metadata = {
        k: v for k, v in {
            "session_id": session_id,
            "source": "stream_research_agent",
        }.items() if v is not None
    }
    _run_config = {"tags": _langsmith_tags, "metadata": _langsmith_metadata}

    full_output = ""
    tools_used = []

    try:
        if fast_path:
            logger.info("Using FAST PATH LCEL Stream for Research Agent")
            import redis.asyncio as aioredis
            import hashlib
            from groq import AsyncGroq
            from app.agent.mcp_servers.rag_server import search_medical_guidelines
            
            # Redis Caching logic
            redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
            query_hash = hashlib.sha256(query.encode()).hexdigest()
            cache_key = f"thyrax:chat:cache:{query_hash}"
            
            cached_response = await redis_client.get(cache_key)
            if cached_response:
                logger.info(f"Redis Cache HIT for query: {query}")
                chunk_size = 20
                for i in range(0, len(cached_response), chunk_size):
                    chunk = cached_response[i:i+chunk_size]
                    full_output += chunk
                    yield chunk
                await redis_client.close()
                return

            # Perform RAG lookup directly to keep it fast
            try:
                rag_context = search_medical_guidelines(query)
                logger.info("Successfully fetched RAG context directly.")
            except Exception as e:
                logger.error(f"Failed to fetch direct RAG context: {e}")
                rag_context = "No additional guidelines retrieved."
            
            keys = settings.get_groq_keys()
            selected_key = keys[_current_key_index % len(keys)]
            client = AsyncGroq(api_key=selected_key)
            
            # Format messages for Groq API
            system_msg = (
                "You are ThyraX (Node 8), an Elite Medical Research AI.\n"
                "SCOPE: You answer ONLY medical and health-science questions. "
                "If the user asks anything non-medical (programming, cooking, politics, jokes, etc.), "
                "respond ONLY with: 'I am a medical research assistant only. I cannot help with this topic.'\n"
                "You do NOT have access to specific patient data. For patient-specific analysis, "
                "redirect to Patient Chat (Node 7).\n"
                "Answer in the exact same language as the user.\n\n"
                "MEDICAL GUIDELINES & CONTEXT:\n"
                f"{rag_context}\n\n"
                "Use the above guidelines to inform your answer if relevant. Do not mention that you performed a search."
            )
            
            groq_messages = [
                {"role": "system", "content": system_msg}
            ]
            for msg in lc_history:
                role = "assistant" if msg.type == "ai" else "user"
                groq_messages.append({"role": role, "content": msg.content})
            groq_messages.append({"role": "user", "content": enhanced_input})
            
            completion = await client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=groq_messages,
                temperature=settings.LLM_TEMPERATURE,
                stream=True,
            )
            
            async for chunk in completion:
                content = chunk.choices[0].delta.content
                if content:
                    full_output += content
                    yield content
                    
            # Save to Redis cache for future
            if full_output:
                await redis_client.setex(cache_key, 3600 * 24, full_output) # Cache for 24 hours
            await redis_client.close()
        else:
            logger.info("Using AGENT EXECUTOR Stream for Research Agent")
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + lc_history + [HumanMessage(content=enhanced_input)]
            async for event in executor.astream_events(
                {
                    "messages": messages,
                },
                version="v2",
                config=_run_config
            ):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        full_output += content
                        yield content
                elif kind == "on_tool_start":
                    tools_used.append(event["name"])
                    
    except Exception as e:
        err_str = str(e)
        logger.error(f"Streaming error: {err_str}")
        
        _QUOTA_SIGNALS = ("429", "RESOURCE_EXHAUSTED", "rate_limit", "Too Many Requests")
        if any(sig in err_str for sig in _QUOTA_SIGNALS):
            keys = settings.get_groq_keys()
            if len(keys) > 1:
                _current_key_index += 1
                logger.warning(f"Groq Quota Exhausted in stream. Rotating to key index {_current_key_index % len(keys)}")
                
                fallback_output = await _run_fallback_llm(query, enhanced_input, "Quota Exhausted Retry")
                chunk_size = 20
                for i in range(0, len(fallback_output), chunk_size):
                    yield fallback_output[i:i+chunk_size]
                full_output += fallback_output
            else:
                error_msg = "\n[Error: The AI agent's API quota has been exhausted. Please check your GROQ_API_KEY.]"
                yield error_msg
                full_output += error_msg
                
        elif "Failed to call a function" in err_str:
            fallback_output = await _run_fallback_llm(query, enhanced_input, "Tool Call Error")
            chunk_size = 20
            for i in range(0, len(fallback_output), chunk_size):
                yield fallback_output[i:i+chunk_size]
            full_output += fallback_output
            
        else:
            error_msg = f"\n[Error: {err_str}]"
            yield error_msg
            full_output += error_msg

    # Background Saving & Caching
    final_output = _process_and_cache_response(query or "Patient medical query", full_output)

    if not fast_path and "search_medical_web" in tools_used and "[KNOWLEDGE_CACHE]" not in final_output:
        def _auto_embed_task():
            try:
                from app.agent.mcp_servers.rag_server import save_to_knowledge_base
                save_to_knowledge_base(query or "Web Search", final_output)
            except Exception as e:
                logger.error(f"Failed to auto-embed web search: {e}")
        
        if background_tasks:
            background_tasks.add_task(_auto_embed_task)
        else:
            try:
                asyncio.get_running_loop().run_in_executor(None, _auto_embed_task)
            except RuntimeError:
                _auto_embed_task()

    if query and not image_base64 and "[Error:" not in final_output:
        def _save_cache_task():
            save_semantic_cache(query, final_output)
        
        if background_tasks:
            background_tasks.add_task(_save_cache_task)
        else:
            try:
                asyncio.get_running_loop().run_in_executor(None, _save_cache_task)
            except RuntimeError:
                pass

    if session_id:
        from app.services.memory_manager import memory_manager
        async def _background_save_and_prune():
            try:
                await memory_manager.save_exchange(
                    session_id=session_id,
                    user_message=query or "Medical request",
                    ai_response=final_output,
                )
                if len(effective_history) + 2 > 6:
                    await memory_manager.summarize_and_prune(session_id)
            except Exception as e:
                logger.error(f"Background save/prune failed: {e}")
                
        if background_tasks:
            background_tasks.add_task(_background_save_and_prune)
        else:
            asyncio.create_task(_background_save_and_prune())
