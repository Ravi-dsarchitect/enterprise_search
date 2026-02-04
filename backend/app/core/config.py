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
    LLM_PROVIDER: str = "groq"
    LLM_MODEL_NAME: str = "llama-3.3-70b-versatile"
    GROQ_API_KEY: str = ""

    # Ollama / Self-hosted (optional)
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Retrieval Configuration
    RETRIEVAL_CANDIDATE_MULTIPLIER: int = 10
    RERANK_CANDIDATES: int = 30
    RERANK_TOP_N: int = 7
    MULTI_QUERY_COUNT: int = 3

    # Chunking Configuration
    # 1800 chars (~450 tokens) optimized for RAG retrieval with LIC documents
    CHUNK_SIZE: int = 1800
    CHUNK_OVERLAP: int = 200
    MAX_TABLE_CHUNK: int = 2500

    class Config:
        env_file = ("../.env", ".env")
        extra = "ignore"


settings = Settings()
