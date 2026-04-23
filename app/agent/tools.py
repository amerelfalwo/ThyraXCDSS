"""
LangChain Tools for the ThyraX CDSS Agent.

Tool 1: search_medical_guidelines  –  RAG over ChromaDB (local knowledge base)
Tool 2: search_medical_web          –  Web search via DuckDuckGo (medical only)

Workflow enforced by the system prompt:
  1. ALWAYS call search_medical_guidelines first.
  2. If RAG returns no relevant results, call search_medical_web.
  3. Combine both sources in the final answer.

Features:
  - Lightweight keyword re-ranking for RAG results.
  - Web search restricted to trusted medical domains.
  - Open medical scope (not limited to thyroid-specific queries).
"""
import os
import logging
import numpy as np
from langchain_core.tools import tool

from app.core.config import settings

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Shared: Embedding Manager & ChromaDB Client (lazy loaded)
# ═══════════════════════════════════════════════════════════════

_embedding_model = None
_chroma_client = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _embedding_model


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    return _chroma_client


def _embed_query(query: str) -> list:
    model = _get_embedding_model()
    embedding = model.encode([query], normalize_embeddings=True)
    return embedding[0].tolist()


# ═══════════════════════════════════════════════════════════════
# Re-Ranking — lightweight keyword-overlap scorer
# ═══════════════════════════════════════════════════════════════

def _rerank_results(
    query: str,
    documents: list[str],
    metadatas: list[dict],
    top_k: int = 3,
) -> tuple[list[str], list[dict]]:
    """
    Re-rank retrieved chunks by keyword overlap with the query.

    This is a lightweight alternative to a cross-encoder that adds
    zero memory overhead — critical for the 512MB constraint.

    Scoring:
      - Exact phrase matches score highest.
      - Individual keyword overlap counts as a secondary signal.
      - Results are sorted by combined score, top_k returned.
    """
    query_lower = query.lower()
    query_terms = set(query_lower.split())

    scored = []
    for i, doc in enumerate(documents):
        doc_lower = doc.lower()

        # Phrase match bonus (high value)
        phrase_bonus = 10.0 if query_lower in doc_lower else 0.0

        # Keyword overlap (Jaccard-like)
        doc_terms = set(doc_lower.split())
        overlap = len(query_terms & doc_terms)
        keyword_score = overlap / max(len(query_terms), 1)

        # Combined score
        total = phrase_bonus + keyword_score
        scored.append((total, i))

    scored.sort(key=lambda x: x[0], reverse=True)

    reranked_docs = [documents[idx] for _, idx in scored[:top_k]]
    reranked_meta = [metadatas[idx] for _, idx in scored[:top_k]]

    return reranked_docs, reranked_meta


# ═══════════════════════════════════════════════════════════════
# Tool 1: Medical Guidelines RAG (Open Scope + Re-Ranking)
# ═══════════════════════════════════════════════════════════════

@tool
def search_medical_guidelines(query: str) -> str:
    """
    Search the LOCAL medical knowledge base for clinical guidelines,
    published literature, drug references, diagnostic criteria,
    and evidence-based protocols across ALL medical specialties.

    This tool searches the ChromaDB vector store containing ingested
    medical documents, ATA guidelines, WHO protocols, drug formularies,
    PubMed references, and any other documents added via the RAG
    ingestion pipeline.

    IMPORTANT: You MUST call this tool FIRST before any other tool
    for every clinical question.

    Args:
        query: The medical question or topic to search for.
            Be specific — e.g. "ATA guidelines for FNA biopsy
            of thyroid nodules >1cm" rather than just "thyroid".

    Returns:
        Relevant excerpts from the medical knowledge base with
        source document references. Returns a message if no
        relevant documents are found — in that case, you should
        call search_medical_web as a fallback.
    """
    try:
        client = _get_chroma_client()

        # Use get_or_create so the tool works even before any docs are ingested
        collection = client.get_or_create_collection(settings.CHROMA_GUIDELINES_COLLECTION)

        # Guard: collection is empty — no docs ingested yet
        total_docs = collection.count()
        if total_docs == 0:
            return (
                "⚠️ NO RESULTS — The local knowledge base is empty. "
                "No medical documents have been ingested yet. "
                "You MUST now call the search_medical_web tool to find "
                "relevant medical information online."
            )

        query_embedding = _embed_query(query)

        # Clamp n_results to the actual collection size to avoid ChromaDB crash
        n_results = min(10, total_docs)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        if not results["documents"] or not results["documents"][0]:
            return (
                "⚠️ NO RESULTS — No relevant medical guidelines found in "
                f"the knowledge base for: '{query}'. "
                "You MUST now call the search_medical_web tool to find "
                "relevant medical information online."
            )

        # Re-rank results for better precision
        reranked_docs, reranked_meta = _rerank_results(
            query=query,
            documents=results["documents"][0],
            metadatas=results["metadatas"][0],
            top_k=3,
        )

        # Format results with sources
        formatted = []
        for i, (doc, metadata) in enumerate(
            zip(reranked_docs, reranked_meta), 1
        ):
            source = metadata.get("source", "Unknown source")
            # Extract just the filename for cleaner display
            source_name = os.path.basename(source) if source else "Unknown"
            formatted.append(
                f"[Source {i}: {source_name}]\n{doc}"
            )

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return (
            f"⚠️ RAG ERROR — Could not search the local knowledge base: {e}. "
            "You MUST now call the search_medical_web tool as a fallback."
        )


# ═══════════════════════════════════════════════════════════════
# Tool 2: Medical Web Search (DuckDuckGo — Free, No API Key)
# ═══════════════════════════════════════════════════════════════

# Trusted medical domains to prioritize in web search
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


@tool
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


# ── Export all tools for the agent ──
ALL_TOOLS = [search_medical_guidelines, search_medical_web]
