"""
MCP Server: RAG — ChromaDB Medical Guidelines Retrieval.

Exposes tools:
  - search_medical_guidelines : Semantic search + keyword re-ranking
  - save_to_knowledge_base   : Write-back mechanism for self-updating RAG

This server runs as a separate process and communicates via stdio
with the MCP client in the FastAPI backend.

Run standalone:  python -m app.agent.mcp_servers.rag_server
"""

import os
import logging
import uuid

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("ThyraX_RAG")

# ─── Configuration (read from env, not importing app.core.config
# to keep the server process lightweight) ───────────────────────

_CHROMA_PERSIST_DIR = os.environ.get(
    "CHROMA_PERSIST_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"),
)
_CHROMA_COLLECTION = os.environ.get("CHROMA_GUIDELINES_COLLECTION", "pdf_documents")
_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ─── Lazy-loaded singletons ───────────────────────────────────

_embedding_model = None
_chroma_client = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(_EMBEDDING_MODEL)
    return _embedding_model


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        persist_dir = os.path.abspath(_CHROMA_PERSIST_DIR)
        _chroma_client = chromadb.PersistentClient(path=persist_dir)
    return _chroma_client


def _embed_query(query: str) -> list:
    model = _get_embedding_model()
    embedding = model.encode([query], normalize_embeddings=True)
    return embedding[0].tolist()


# ─── Re-Ranking — lightweight keyword-overlap scorer ──────────

def _rerank_results(
    query: str,
    documents: list[str],
    metadatas: list[dict],
    top_k: int = 3,
) -> tuple[list[str], list[dict]]:
    """
    Re-rank retrieved chunks by keyword overlap with the query.

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

        total = phrase_bonus + keyword_score
        scored.append((total, i))

    scored.sort(key=lambda x: x[0], reverse=True)

    reranked_docs = [documents[idx] for _, idx in scored[:top_k]]
    reranked_meta = [metadatas[idx] for _, idx in scored[:top_k]]

    return reranked_docs, reranked_meta


# ═══════════════════════════════════════════════════════════════
# Tool 1: search_medical_guidelines
# ═══════════════════════════════════════════════════════════════

@mcp.tool()
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
        source document references. Returns a specific SYSTEM_COMMAND
        if no relevant documents are found, instructing you to
        fallback to your internal knowledge.
    """
    try:
        client = _get_chroma_client()

        # Use get_or_create so the tool works even before any docs are ingested
        collection = client.get_or_create_collection(_CHROMA_COLLECTION)

        # Guard: collection is empty — no docs ingested yet
        total_docs = collection.count()
        if total_docs == 0:
            return (
                "SYSTEM_COMMAND: NO_RESULTS_FOUND. DO NOT CALL THIS TOOL AGAIN. "
                "Stop searching and generate your final answer immediately using "
                "your internal knowledge. You MUST start your response with the "
                "exact tag: [KNOWLEDGE_CACHE]."
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
                "SYSTEM_COMMAND: NO_RESULTS_FOUND. DO NOT CALL THIS TOOL AGAIN. "
                "Stop searching and generate your final answer immediately using "
                "your internal knowledge. You MUST start your response with the "
                "exact tag: [KNOWLEDGE_CACHE]."
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
            source_name = os.path.basename(source) if source else "Unknown"
            formatted.append(
                f"[Source {i}: {source_name}]\n{doc}"
            )

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return (
            "SYSTEM_COMMAND: NO_RESULTS_FOUND. DO NOT CALL THIS TOOL AGAIN. "
            "Stop searching and generate your final answer immediately using "
            "your internal knowledge. You MUST start your response with the "
            "exact tag: [KNOWLEDGE_CACHE]."
        )


# ═══════════════════════════════════════════════════════════════
# Tool 2: save_to_knowledge_base (Self-Updating RAG)
# ═══════════════════════════════════════════════════════════════

@mcp.tool()
def save_to_knowledge_base(question: str, answer: str) -> str:
    """
    Save a question-answer pair to the medical knowledge base.

    This enables the 'Self-Updating RAG' mechanism where the LLM's
    internal knowledge is externalized for future retrieval.

    Args:
        question: The original medical question.
        answer: The comprehensive answer to cache.

    Returns:
        Confirmation message.
    """
    try:
        client = _get_chroma_client()
        collection = client.get_or_create_collection(_CHROMA_COLLECTION)

        embedding = _embed_query(question)

        collection.add(
            ids=[f"cache_{uuid.uuid4()}"],
            embeddings=[embedding],
            metadatas=[{"question": question, "source": "AI Knowledge Cache"}],
            documents=[answer],
        )
        logger.info(f"Cached response for query: '{question[:50]}...'")
        return "Successfully cached the answer for future retrieval."
    except Exception as e:
        logger.error(f"Failed to save to vector db cache: {e}")
        return f"Failed to cache: {e}"


# ═══════════════════════════════════════════════════════════════
# Entry point — run as standalone MCP server via stdio
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mcp.run(transport="stdio")
