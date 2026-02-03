from fastapi import FastAPI
from app.core.config import settings

from app.api.v1.api import api_router
from app.core.database import init_qdrant

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {"message": "Welcome to Ngenux Enterprise GenAI Knowledge Assistant API"}

# Wrapper for startup (can be used for DB connection checks)
@app.on_event("startup")
async def startup_event():
    print("Starting up...")
    init_qdrant()
