"""
Cached model instances to avoid reloading models on every request.
Models are loaded once and reused across requests for better performance.
"""
from functools import lru_cache
from app.services.ingestion.embedders import EmbedderFactory, Embedder
from app.core.config import settings


@lru_cache(maxsize=1)
def get_cached_embedder() -> Embedder:
    """Get or create the cached embedder instance."""
    return EmbedderFactory.create_embedder(
        provider=settings.EMBEDDING_PROVIDER,
        model_name=settings.EMBEDDING_MODEL,
    )


@lru_cache(maxsize=1)
def get_cached_reranker():
    """Get or create the cached reranker instance."""
    from app.services.rag.reranker import Reranker

    return Reranker(model_name=settings.RERANKER_MODEL)
