"""
MCP Servers Package — ThyraX CDSS Tool Abstraction Layer.

Each MCP server encapsulates a single data-source concern:

  rag_server.py   — ChromaDB RAG retrieval + re-ranking + cache write-back
  web_server.py   — DuckDuckGo medical web search (trusted domains)

The FastAPI backend acts as an MCP *Client* and connects to each
server via stdio transport. The client factory in `mcp_client.py`
manages the lifecycle of all server connections.

Architecture:
    ┌─────────────────────────────────────────────┐
    │  FastAPI  (MCP Client)                      │
    │                                             │
    │  mcp_client.py                              │
    │    ├── connect_to(rag_server)   → session_1 │
    │    └── connect_to(web_server)   → session_2 │
    │                                             │
    │  load_mcp_tools(session_1, session_2)       │
    │    → unified tool list for LangChain Agent  │
    └─────────────────────────────────────────────┘
"""
