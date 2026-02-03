from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "Ngenux GenAI Knowledge Assistant"
    API_V1_STR: str = "/api/v1"
    
    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION_NAME: str = "enterprise_docs"
    
    # Embedding Configuration
    EMBEDDING_PROVIDER: str = "openai"  # Options: openai, local
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # OpenAI's fast embedding model

    # LLM Configuration
    LLM_PROVIDER: str = "openai"  # Options: openai, groq, bedrock, ollama
    
    # Provider-specific settings
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""  # Get free API key from https://console.groq.com
    
    # AWS Bedrock
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    
    # Ollama / Self-hosted
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL_NAME: str = ""  # Override default model name if needed
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
