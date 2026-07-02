# 🚀 ThyraX CDSS System Acceleration Plan (Zero-Latency Roadmap)

This document outlines the step-by-step roadmap to reduce the latency of all AI nodes in the ThyraX CDSS from multi-second delays to near real-time (sub-second) responses.

## 🟢 Phase 1: Quick Wins (Non-Blocking Operations)
**Goal:** Remove bottlenecks caused by database writes and logging that block the HTTP response.

- [x] **Background Tasks for AI Chat:** In `app/routers/ai_chat.py`, move `memory_manager.save_exchange`, `save_to_knowledge_base`, and `log_audit_event` to `FastAPI BackgroundTasks`.
- [x] **Background Tasks for Image endpoints:** In `app/routers/image.py` (and similar nodes), ensure that database image saving and audit logging do not block the primary API response, or at least run them in parallel.
- [x] **Parallel Context Loading:** In `app/agent/agent.py` and `app/agent/research_agent.py`, use `asyncio.gather` to fetch `patient_context` and load `MCP tools` concurrently instead of sequentially.

## 🟡 Phase 2: Native Streaming & Bypassing AgentExecutor
**Goal:** Deliver the "Time to First Token" (TTFT) in < 300ms by dropping the rigid `AgentExecutor` ReAct loop when tools aren't strictly necessary.

- [x] **Smart Routing (Classifier):** Implement a lightweight classifier (e.g., regex or a fast 0-temperature prompt) at the router level.
    - If the user query is purely conversational -> Route to Direct LLM with Native Streaming.
    - If the user query requires lookup -> Route to AgentExecutor.
- [x] **Implement LCEL Native Streaming:** Create a fast-path in the agents that utilizes LangChain Expression Language (LCEL) and `astream()` for real-time chunk delivery to the user.
- [x] **Remove Fake Streaming:** Remove the synthetic `await asyncio.sleep(0.005)` loop from `ai_chat.py` once true native streaming is implemented.

## 🟠 Phase 3: Token Optimization & Prompt Compression
**Goal:** Reduce the payload size sent to Groq Llama-3 to speed up input token processing.

- [x] **History Summarization:** Update `MemoryManager` to aggressively summarize history. Pass only the last 2-3 messages fully + 1 compressed summary to the LLM prompt.
- [x] **Context Compression:** Optimize the string returned by `patient_context.to_prompt_context()` to be highly terse. Remove redundant human-readable formatting and use compact JSON/YAML.
- [x] **System Prompt Tuning:** Trim down `SYSTEM_PROMPT` in `agent.py` and `research_agent.py`. Fewer tokens = faster TTFT.

## 🔴 Phase 4: Advanced Caching & Connection Pooling
**Goal:** Prevent repetitive processing for common queries and reduce network overhead.

- [x] **Semantic Caching:** Implement an in-memory or DB-backed semantic cache for Node 8.
    - E.g., If a user asks "What is TI-RADS 4?", check if a similar query exists. Return the cached response instantly to bypass the LLM entirely.
- [x] **Connection Pooling:** Ensure Supabase HTTP clients and Groq HTTP sessions are pooled and reused efficiently across FastAPI requests to eliminate TCP handshake delays.
