"""
Centralized configuration using Pydantic BaseSettings.
Reads from environment variables or .env file.
"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # ── API Info ──
    APP_TITLE: str = "ThyraX CDSS API"
    APP_VERSION: str = "4.0.0"

    # ── Groq LLM (Primary) ──
    GROQ_API_KEY: str = ""
    GROQ_API_KEY_2: str = ""
    GROQ_API_KEY_3: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    LLM_TEMPERATURE: float = 0.1

    def get_groq_keys(self) -> list[str]:
        """Returns a list of all configured Groq API keys."""
        keys = [self.GROQ_API_KEY, self.GROQ_API_KEY_2, self.GROQ_API_KEY_3]
        return [k for k in keys if k and k.strip()]

    # ── Internal Service Auth ──
    INTERNAL_SERVICE_KEY: str = ""

    # ── PostgreSQL Database ──
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/thyrax"

    @property
    def ASYNC_DATABASE_URL(self) -> str:
        """Derive the asyncpg URL from the sync DATABASE_URL."""
        url = self.DATABASE_URL
        if url.startswith("postgresql+asyncpg://"):
            return url
        if url.startswith("postgresql+psycopg2://"):
            return url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url

    # ── Supabase ──
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""

    # ── ChromaDB ──
    CHROMA_PERSIST_DIR: str = str(
        Path(__file__).resolve().parent.parent.parent / "data"
    )
    CHROMA_GUIDELINES_COLLECTION: str = "pdf_documents"
    CHROMA_CACHE_COLLECTION: str = "semantic_cache"

    # ── Embeddings ──
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    model_config = {
        "env_file": str(Path(__file__).resolve().parent.parent.parent / ".env"),
        "extra": "ignore"
    }


settings = Settings()
