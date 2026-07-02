import sys
sys.path.append('/home/hunter/ThyraXCDSS')

from app.core.config import settings
from app.services.semantic_cache import _get_chroma_client

client = _get_chroma_client()
try:
    client.delete_collection(settings.CHROMA_CACHE_COLLECTION)
    print("Chroma cache collection deleted.")
except Exception as e:
    print(f"Error: {e}")
