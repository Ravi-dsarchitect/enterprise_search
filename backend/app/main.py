from fastapi import FastAPI
from app.core.config import settings

from app.api.v1.api import api_router
from app.core.database import init_qdrant

# Custom tags for better organization in Swagger UI
TAGS_METADATA = [
    {
        "name": "ingestion",
        "description": "Upload and index documents from local files or S3 buckets.",
    },
    {
        "name": "chat",
        "description": "Query your documents using natural language with RAG-powered responses.",
    },
    {
        "name": "admin",
        "description": "System health checks and configuration management.",
    },
]

app = FastAPI(
    title="Ngenux Enterprise Search",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=TAGS_METADATA,
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/", tags=["root"])
def root():
    """
    Welcome endpoint - verifies the API is running.
    
    Returns a simple welcome message to confirm the service is operational.
    """
    return {"message": "Welcome to Ngenux Enterprise GenAI Knowledge Assistant API"}

# Wrapper for startup (can be used for DB connection checks)
@app.on_event("startup")
async def startup_event():
    print("Starting up...")
    init_qdrant()

