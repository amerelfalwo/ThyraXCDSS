"""
MCP Client Factory — Multi-Server Connection Manager.

Manages the lifecycle of MCP server connections. The FastAPI backend
acts as an MCP *Client* that spawns each server as a subprocess
(stdio transport) and aggregates their tools into a single list
for the LangChain Agent.

Architecture:
    ┌──────────────────────────────────────────────────┐
    │  MCPClientManager (singleton)                    │
    │                                                  │
    │  _servers = {                                    │
    │    "rag":  {session, exit_stack, module_path}     │
    │    "web":  {session, exit_stack, module_path}     │
    │  }                                               │
    │                                                  │
    │  get_all_tools() → merged LangChain tool list    │
    │  shutdown()      → graceful cleanup              │
    └──────────────────────────────────────────────────┘

Usage:
    from app.agent.mcp_servers.mcp_client import mcp_client_manager

    # At agent init time:
    tools = await mcp_client_manager.get_all_tools()

    # At shutdown:
    await mcp_client_manager.shutdown()
"""

import contextlib
import logging
import sys
import typing

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


# ─── Server Registry ──────────────────────────────────────────
# Each entry is a (name, module_path) pair. The module must have
# `if __name__ == "__main__": mcp.run(transport="stdio")` at the
# bottom so it can be launched as a subprocess.

_SERVER_REGISTRY: list[tuple[str, str]] = [
    ("rag", "app.agent.mcp_servers.rag_server"),
    ("web", "app.agent.mcp_servers.web_server"),
]


class MCPClientManager:
    """
    Manages connections to multiple MCP tool servers.

    Each server is spawned as a subprocess via stdio transport.
    The manager holds the async exit stacks and sessions so they
    can be cleanly shut down on app teardown.
    """

    def __init__(self):
        self._servers: dict[str, dict[str, typing.Any]] = {}
        self._initialized = False

    async def _connect_server(self, name: str, module_path: str) -> ClientSession:
        """
        Spawn a single MCP server subprocess and establish a session.
        """
        if name in self._servers:
            return self._servers[name]["session"]

        exit_stack = contextlib.AsyncExitStack()

        server_params = StdioServerParameters(
            command=sys.executable,          # use the current Python interpreter
            args=["-m", module_path],
        )

        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport[0], stdio_transport[1]

        session = await exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()

        self._servers[name] = {
            "session": session,
            "exit_stack": exit_stack,
            "module_path": module_path,
        }

        logger.info(f"MCP server '{name}' connected ({module_path})")
        return session

    async def initialize(self) -> None:
        """Connect to all registered MCP servers."""
        if self._initialized:
            return

        for name, module_path in _SERVER_REGISTRY:
            try:
                await self._connect_server(name, module_path)
            except Exception as e:
                logger.error(
                    f"Failed to connect MCP server '{name}' "
                    f"({module_path}): {e}"
                )

        self._initialized = True
        logger.info(
            f"MCPClientManager initialized with "
            f"{len(self._servers)}/{len(_SERVER_REGISTRY)} servers"
        )

    async def get_session(self, name: str) -> ClientSession | None:
        """Get a specific server's session by name."""
        await self.initialize()
        entry = self._servers.get(name)
        return entry["session"] if entry else None

    async def get_all_sessions(self) -> list[ClientSession]:
        """Return sessions for all connected servers."""
        await self.initialize()
        return [entry["session"] for entry in self._servers.values()]

    async def get_all_tools(self) -> list:
        """
        Load and merge tools from ALL connected MCP servers.

        Returns a flat list of LangChain-compatible tool objects
        that can be passed directly to the AgentExecutor.
        """
        from langchain_mcp_adapters.tools import load_mcp_tools

        await self.initialize()

        all_tools = []
        for name, entry in self._servers.items():
            try:
                tools = await load_mcp_tools(entry["session"])
                all_tools.extend(tools)
                logger.info(
                    f"Loaded {len(tools)} tools from MCP server '{name}': "
                    f"{[t.name for t in tools]}"
                )
            except Exception as e:
                logger.error(f"Failed to load tools from '{name}': {e}")

        return all_tools

    async def shutdown(self) -> None:
        """Gracefully close all MCP server connections."""
        for name, entry in self._servers.items():
            try:
                await entry["exit_stack"].aclose()
                logger.info(f"MCP server '{name}' disconnected")
            except Exception as e:
                logger.warning(f"Error closing MCP server '{name}': {e}")

        self._servers.clear()
        self._initialized = False
        logger.info("MCPClientManager shut down")

    @property
    def connected_servers(self) -> list[str]:
        """List names of currently connected servers."""
        return list(self._servers.keys())


# ── Module-level singleton ──
mcp_client_manager = MCPClientManager()
