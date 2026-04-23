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
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.1

    # ── Internal Service Auth ──
    INTERNAL_SERVICE_KEY: str = ""

    # ── ChromaDB ──
    CHROMA_PERSIST_DIR: str = str(
        Path(__file__).resolve().parent.parent.parent / "data"
    )
    CHROMA_GUIDELINES_COLLECTION: str = "pdf_documents"

    # ── Embeddings ──
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
