from qdrant_client import QdrantClient
from app.core.config import settings

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.QDRANT_URL)

def init_qdrant():
    """Initialize Qdrant collection if it doesn't exist."""
    try:
        from app.services.ingestion.embedders import EmbedderFactory
        
        client = get_qdrant_client()
        collections = client.get_collections().collections
        exists = any(c.name == settings.QDRANT_COLLECTION_NAME for c in collections)
        
        if not exists:
            from qdrant_client.http import models
            
            # Get the embedding dimension based on provider
            dimension = EmbedderFactory.get_embedding_dimension(
                provider=settings.EMBEDDING_PROVIDER,
                model_name=settings.EMBEDDING_MODEL
            )
            
            client.create_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE
                )
            )
            print(f"✅ Created new Qdrant collection: {settings.QDRANT_COLLECTION_NAME} (dimension: {dimension})")
        else:
            print(f"✓ Qdrant collection '{settings.QDRANT_COLLECTION_NAME}' already exists (skipping creation)")
    except Exception as e:
        print(f"❌ Error connecting to Qdrant at {settings.QDRANT_URL}: {e}")
        print("⚠️  Make sure Qdrant is running (e.g., 'docker-compose up -d qdrant')")
        raise

# BM25 Index Cache (singleton pattern)
_bm25_retriever = None

def get_cached_bm25_retriever():
    """
    Get cached BM25 retriever instance (singleton).
    Builds index on first call.
    """
    global _bm25_retriever
    if _bm25_retriever is None:
        from app.services.rag.bm25_retriever import BM25Retriever
        print("🔨 Initializing BM25 retriever...")
        _bm25_retriever = BM25Retriever()
        _bm25_retriever.build_index()
    return _bm25_retriever

def rebuild_bm25_index():
    """
    Force rebuild of BM25 index.
    Call this after ingesting new documents.
    """
    global _bm25_retriever
    if _bm25_retriever is not None:
        _bm25_retriever.build_index(force_rebuild=True)
    else:
        get_cached_bm25_retriever()
