from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional
from app.core.config import settings
from app.core.database import get_qdrant_client

router = APIRouter()


# ==================== Response Models ====================

class SystemStatusResponse(BaseModel):
    """Response model for system health status."""
    app_name: str = Field(..., description="Application name")
    qdrant_status: str = Field(..., description="Qdrant vector database connection status")
    llm_provider: str = Field(..., description="Current LLM provider (ollama, groq, etc.)")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "app_name": "Ngenux GenAI Knowledge Assistant",
                "qdrant_status": "connected",
                "llm_provider": "ollama"
            }
        }
    }


class ConfigResponse(BaseModel):
    """Response model for system configuration (sanitized - no secrets)."""
    project_name: str = Field(..., description="Project name")
    api_v1_str: str = Field(..., description="API version prefix")
    qdrant_url: str = Field(..., description="Qdrant server URL")
    collection_name: str = Field(..., description="Qdrant collection name")
    llm_provider: str = Field(..., description="LLM provider")
    llm_model: str = Field(..., description="LLM model name")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "project_name": "Ngenux GenAI Knowledge Assistant",
                "api_v1_str": "/api/v1",
                "qdrant_url": "http://localhost:6334",
                "collection_name": "lic_docs",
                "llm_provider": "ollama",
                "llm_model": "qwen2.5:7b"
            }
        }
    }


# ==================== API Endpoints ====================

@router.get(
    "/status",
    response_model=SystemStatusResponse,
    summary="System Health Check",
    response_description="Current system health status"
)
async def get_system_status():
    """
    Check the health status of the system and its connections.
    
    **Checks:**
    - Qdrant vector database connectivity
    - LLM provider configuration
    
    **Use this endpoint to verify the system is operational before making queries.**
    """
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
        "llm_provider": settings.LLM_PROVIDER
    }

@router.get(
    "/config",
    response_model=ConfigResponse,
    summary="View Configuration",
    response_description="Current system configuration"
)
async def get_config():
    """
    Get current system configuration (sanitized - no secrets exposed).
    
    **Useful for:**
    - Debugging configuration issues
    - Verifying environment settings
    - Confirming which collection is active
    """
    return {
        "project_name": settings.PROJECT_NAME,
        "api_v1_str": settings.API_V1_STR,
        "qdrant_url": settings.QDRANT_URL,
        "collection_name": settings.QDRANT_COLLECTION_NAME,
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL_NAME
    }

