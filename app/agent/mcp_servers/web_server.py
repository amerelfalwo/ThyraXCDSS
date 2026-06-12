"""
MCP Server: Web Search — DuckDuckGo Medical Search.

Exposes tool:
  - search_medical_web : Searches the internet for medical info,
                         prioritizing trusted clinical sources.

This server runs as a separate process and communicates via stdio
with the MCP client in the FastAPI backend.

Run standalone:  python -m app.agent.mcp_servers.web_server
"""

import logging

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("ThyraX_WebSearch")

# ─── Trusted medical domains to prioritize ────────────────────

_MEDICAL_SITE_SUFFIXES = (
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "who.int",
    "mayoclinic.org",
    "uptodate.com",
    "medscape.com",
    "nih.gov",
    "cdc.gov",
    "thyroid.org",        # ATA
    "endocrine.org",
    "wiley.com",
    "springer.com",
    "thelancet.com",
    "nejm.org",
    "bmj.com",
    "nature.com",
)


# ═══════════════════════════════════════════════════════════════
# Tool: search_medical_web
# ═══════════════════════════════════════════════════════════════

@mcp.tool()
def search_medical_web(query: str) -> str:
    """
    Search the internet for medical information using DuckDuckGo.
    Results are filtered to prioritize trusted medical sources like
    PubMed, WHO, Mayo Clinic, NIH, UpToDate, ATA, and medical journals.

    ONLY use this tool when search_medical_guidelines returns no results
    or when the local knowledge base does not have sufficient information
    to answer the clinical question.

    This tool is RESTRICTED to medical queries only. Do NOT use it for
    non-medical topics.

    Args:
        query: A specific medical search query. Add "medical" or
            "clinical guidelines" to improve relevance.
            Example: "TSH reference range subclinical hypothyroidism
            clinical guidelines"

    Returns:
        Top medical search results with titles, snippets, and URLs
        from trusted medical sources.
    """
    try:
        from ddgs import DDGS

        # Append "medical" to bias results towards clinical content
        medical_query = f"{query} medical clinical"

        with DDGS() as ddgs:
            raw_results = list(ddgs.text(medical_query, max_results=8))

        if not raw_results:
            return (
                f"No web results found for: '{query}'. "
                "Answer based on your general medical training data, "
                "but clearly state that no external sources were found."
            )

        # Separate trusted vs general results
        trusted = []
        general = []
        for r in raw_results:
            href = r.get("href", "").lower()
            is_trusted = any(domain in href for domain in _MEDICAL_SITE_SUFFIXES)
            entry = (
                f"**{r.get('title', 'No title')}**\n"
                f"{r.get('body', 'No snippet')}\n"
                f"🔗 {r.get('href', 'No URL')}"
            )
            if is_trusted:
                trusted.append(entry)
            else:
                general.append(entry)

        # Prioritize trusted sources, then fill with general
        sorted_results = trusted + general
        top_results = sorted_results[:5]

        header = (
            f"🌐 Web Search Results ({len(trusted)} from trusted medical sources, "
            f"{len(general)} from general sources):\n\n"
        )

        return header + "\n\n---\n\n".join(top_results)

    except ImportError:
        return (
            "Web search is not available — the duckduckgo-search package "
            "is not installed. Answer based on general medical knowledge."
        )
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return (
            f"Web search failed: {e}. "
            "Answer based on your general medical training data, "
            "but clearly state that external search was unavailable."
        )


# ═══════════════════════════════════════════════════════════════
# Entry point — run as standalone MCP server via stdio
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mcp.run(transport="stdio")
