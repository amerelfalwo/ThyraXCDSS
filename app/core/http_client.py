import httpx
from typing import Optional

_shared_async_client: Optional[httpx.AsyncClient] = None
_shared_sync_client: Optional[httpx.Client] = None

def get_shared_async_client() -> httpx.AsyncClient:
    global _shared_async_client
    if _shared_async_client is None:
        limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
        _shared_async_client = httpx.AsyncClient(limits=limits, timeout=60.0)
    return _shared_async_client

def get_shared_sync_client() -> httpx.Client:
    global _shared_sync_client
    if _shared_sync_client is None:
        limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
        _shared_sync_client = httpx.Client(limits=limits, timeout=60.0)
    return _shared_sync_client
