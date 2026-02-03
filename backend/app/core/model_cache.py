"""
Cached model instances to avoid reloading models on every request.
Models are loaded once and reused across requests for better performance.
"""
from functools import lru_cache
from app.services.ingestion.embedders import EmbedderFactory, Embedder
from app.services.rag.reranker import Reranker
from app.core.config import settings

# Cache the embedder instance (loaded once, reused forever)
@lru_cache(maxsize=1)
def get_cached_embedder() -> Embedder:
    """Get or create the cached embedder instance"""
    return EmbedderFactory.create_embedder(
        provider=settings.EMBEDDING_PROVIDER,
        api_key=settings.OPENAI_API_KEY,
        model_name=settings.EMBEDDING_MODEL
    )

# Cache the reranker instance
@lru_cache(maxsize=1)
def get_cached_reranker() -> Reranker:
    """Get or create the cached reranker instance"""
    return Reranker()
