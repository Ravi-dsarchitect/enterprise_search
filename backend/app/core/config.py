from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Ngenux GenAI Knowledge Assistant"
    API_V1_STR: str = "/api/v1"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION_NAME: str = "lic_docs"

    # Embedding Configuration
    EMBEDDING_PROVIDER: str = "local"
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"

    # Reranker Configuration
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

    # LLM Configuration
    LLM_PROVIDER: str = "ollama"
    LLM_MODEL_NAME: str = "qwen2.5:14b"
    LLM_METADATA_MODEL: str = "qwen2.5:7b"  # Smaller model for metadata extraction during ingestion
    GROQ_API_KEY: str = ""

    # Ollama / Self-hosted (optional)
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Retrieval Configuration (best config from evaluation)
    RETRIEVAL_CANDIDATE_MULTIPLIER: int = 10
    RERANK_CANDIDATES: int = 30
    RERANK_TOP_N: int = 7
    MULTI_QUERY_COUNT: int = 3

    # Chunking Configuration
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_TABLE_CHUNK: int = 2000

    class Config:
        env_file = ("../.env", ".env")
        extra = "ignore"


settings = Settings()
