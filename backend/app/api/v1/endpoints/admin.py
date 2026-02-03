from fastapi import APIRouter
from app.core.config import settings
from app.core.database import get_qdrant_client

router = APIRouter()

@router.get("/status")
async def get_system_status():
    """Check connections to Qdrant and other services."""
    client = get_qdrant_client()
    try:
        # Simple check: get collection info
        client.get_collection(settings.QDRANT_COLLECTION_NAME)
        qdrant_status = "connected"
    except Exception as e:
        qdrant_status = f"error: {str(e)}"
        
    return {
        "app_name": settings.PROJECT_NAME,
        "qdrant_status": qdrant_status,
        "mode": "standard" if settings.OPENAI_API_KEY else "local_mock"
    }

@router.get("/config")
async def get_config():
    """Get current configuration (sanitized)."""
    return {
        "project_name": settings.PROJECT_NAME,
        "api_v1_str": settings.API_V1_STR,
        "qdrant_url": settings.QDRANT_URL,
        "collection_name": settings.QDRANT_COLLECTION_NAME,
        "has_openai_key": bool(settings.OPENAI_API_KEY)
    }
