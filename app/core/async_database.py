"""
Async SQLAlchemy Engine for the ThyraX Dual-State Memory System.

Uses asyncpg as the async PostgreSQL driver.

Architecture:
  - AsyncSessionLocal: for new async code (MemoryManager, Agent).
  - The sync engine in database.py is preserved for backward
    compatibility with legacy sync callers.

Usage:
    from app.core.async_database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Patient))
"""

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings


def _build_async_url(sync_url: str) -> str:
    """
    Convert a sync PostgreSQL URL to an async one.

    postgresql://...  →  postgresql+asyncpg://...
    postgresql+psycopg2://...  →  postgresql+asyncpg://...
    """
    if sync_url.startswith("postgresql+asyncpg://"):
        return sync_url
    if sync_url.startswith("postgresql+psycopg2://"):
        return sync_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    if sync_url.startswith("postgresql://"):
        return sync_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    raise ValueError(
        f"Unsupported DATABASE_URL scheme: {sync_url!r}. "
        "Expected a postgresql:// connection string."
    )


ASYNC_DATABASE_URL = _build_async_url(settings.DATABASE_URL)

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
