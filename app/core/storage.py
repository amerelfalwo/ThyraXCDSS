import os
import logging
import asyncio

logger = logging.getLogger(__name__)

# Lazy import — supabase SDK is optional; the app should boot without it.
try:
    from supabase import create_client, Client as _SupabaseClient
except ImportError as e:
    create_client = None  # type: ignore[assignment]
    _SupabaseClient = None  # type: ignore[assignment,misc]
    import traceback
    logger.warning(
        f"supabase package not installed or failed to import. "
        f"Exception: {e}\n{traceback.format_exc()}"
    )

from app.core.config import settings

SUPABASE_URL = settings.SUPABASE_URL
SUPABASE_KEY = settings.SUPABASE_KEY

supabase_client = None

if create_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
elif not create_client:
    pass  # already warned above
else:
    logger.warning("SUPABASE_URL or SUPABASE_KEY not found in environment. Supabase Storage client not initialized.")

async def upload_image_to_storage(file_bytes: bytes, file_name: str, folder_path: str = "") -> str:
    """
    Upload an image to the Supabase 'medical_scans' bucket.
    Returns the relative file path inside the bucket.
    """
    if not supabase_client:
        raise RuntimeError("Supabase client is not initialized. Please check SUPABASE_URL and SUPABASE_KEY.")
    
    bucket_name = "medical_scans"
    
    # Format the folder path
    if folder_path and not folder_path.endswith("/"):
        folder_path += "/"
    if folder_path.startswith("/"):
        folder_path = folder_path[1:]
        
    full_path = f"{folder_path}{file_name}"

    def _upload():
        # The supabase-py library is synchronous for storage operations
        supabase_client.storage.from_(bucket_name).upload(
            file=file_bytes,
            path=full_path,
            file_options={"content-type": "image/png"}
        )
        return full_path

    try:
        # Run the synchronous upload in a thread pool to avoid blocking the event loop
        path = await asyncio.to_thread(_upload)
        return path
    except Exception as e:
        logger.error(f"Failed to upload {file_name} to Supabase storage: {str(e)}")
        raise e

async def get_signed_url(file_path: str, expiration_in_seconds: int = 3600) -> str:
    """
    Generate a signed URL for temporary frontend access to a private bucket image.
    """
    if not supabase_client:
        raise RuntimeError("Supabase client is not initialized.")
        
    bucket_name = "medical_scans"
    
    def _get_url():
        response = supabase_client.storage.from_(bucket_name).create_signed_url(
            path=file_path, 
            expires_in=expiration_in_seconds
        )
        # response is typically a dict: {'signedURL': 'https://...'}
        if isinstance(response, dict) and "signedURL" in response:
            return response["signedURL"]
        # In some versions, it might return a string directly or an object
        if hasattr(response, "signed_url"):
            return response.signed_url
        return response

    try:
        url = await asyncio.to_thread(_get_url)
        return url
    except Exception as e:
        logger.error(f"Failed to create signed URL for {file_path}: {str(e)}")
        raise e
