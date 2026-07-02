import os
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

_chroma_client = None
_embedding_model = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        # using a small fast model for semantic caching
        _embedding_model = SentenceTransformer(os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    return _embedding_model

def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        persist_dir = os.path.abspath(settings.CHROMA_PERSIST_DIR)
        _chroma_client = chromadb.PersistentClient(path=persist_dir)
    return _chroma_client

def check_semantic_cache(query: str, threshold: float = 0.88, patient_id: str | None = None) -> str | None:
    """Check semantic cache for a similar query. Return cached response if found."""
    try:
        client = _get_chroma_client()
        collection = client.get_or_create_collection(settings.CHROMA_CACHE_COLLECTION)
        
        if collection.count() == 0:
            return None
            
        model = _get_embedding_model()
        embedding = model.encode([query], normalize_embeddings=True)[0].tolist()
        
        where = {"patient_id": patient_id} if patient_id else {"patient_id": "global"}
        
        results = collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["documents", "distances", "metadatas"],
            where=where
        )
        
        if results and results.get("distances") and results["distances"][0]:
            distance = results["distances"][0][0]
            # ChromaDB usually uses L2 with normalized embeddings: distance = 2*(1-cos_sim).
            # So a distance of < 0.24 means cos_sim > 0.88
            if distance < (2 * (1 - threshold)):
                doc = results["documents"][0][0]
                logger.info(f"Semantic Cache HIT for query: '{query}' (distance: {distance}, patient: {patient_id or 'global'})")
                return doc
                
        return None
    except Exception as e:
        logger.error(f"Semantic Cache checking error: {e}")
        return None

def save_semantic_cache(query: str, response: str, patient_id: str | None = None):
    """Save query and response to semantic cache."""
    try:
        import uuid
        client = _get_chroma_client()
        collection = client.get_or_create_collection(settings.CHROMA_CACHE_COLLECTION)
        
        model = _get_embedding_model()
        embedding = model.encode([query], normalize_embeddings=True)[0].tolist()
        
        collection.add(
            ids=[f"semcache_{uuid.uuid4()}"],
            embeddings=[embedding],
            documents=[response],
            metadatas=[{"query": query, "patient_id": patient_id or "global"}]
        )
        logger.info(f"Saved to Semantic Cache: '{query[:50]}...' (patient: {patient_id or 'global'})")
    except Exception as e:
        logger.error(f"Semantic Cache saving error: {e}")

