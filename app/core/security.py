from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from app.core.config import settings

header_scheme = APIKeyHeader(name="X-AI-Service-Key", auto_error=False)


async def verify_internal_api_key(
    api_key_header: str = Security(header_scheme),
):
    if not settings.INTERNAL_SERVICE_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="INTERNAL_SERVICE_KEY is not configured on the server.",
        )
    if api_key_header != settings.INTERNAL_SERVICE_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Invalid or missing X-AI-Service-Key header.",
        )
    return api_key_header
