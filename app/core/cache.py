"""
Lightweight in-memory TTL cache for ThyraX CDSS.

Uses cachetools (stdlib-weight, no Redis dependency) to avoid
redundant LLM calls for identical symptom queries or repeated
clinical assessments within a short window.

Memory-safe: TTL eviction + max size cap prevent unbounded growth.
"""

import hashlib
import logging
from functools import wraps
from typing import Any

from cachetools import TTLCache

logger = logging.getLogger(__name__)

# ── Global caches (lazily populated, bounded) ──────────────────
# Max 128 entries, 10-minute TTL — keeps RAM under control
_symptoms_cache: TTLCache = TTLCache(maxsize=128, ttl=600)
_ocr_cache: TTLCache = TTLCache(maxsize=64, ttl=600)


def _hash_key(*args: Any) -> str:
    """Create a deterministic cache key from arbitrary arguments."""
    raw = "|".join(str(a) for a in args)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_symptoms_cache() -> TTLCache:
    """Return the shared symptoms analysis cache."""
    return _symptoms_cache


def get_ocr_cache() -> TTLCache:
    """Return the shared OCR analysis cache."""
    return _ocr_cache


def cache_key_for_text(text: str) -> str:
    """Generate a cache key for a text input."""
    normalized = text.strip().lower()
    return _hash_key(normalized)


def cache_key_for_bytes(data: bytes) -> str:
    """Generate a cache key for binary input (images/PDFs)."""
    return hashlib.sha256(data).hexdigest()[:16]
