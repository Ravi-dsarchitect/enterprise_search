import threading
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import settings


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.QDRANT_URL, timeout=30)


def create_collection_with_indexes(client: QdrantClient):
    """Create the Qdrant collection with payload indexes for filtered retrieval."""
    from app.services.ingestion.embedders import EmbedderFactory

    dimension = EmbedderFactory.get_embedding_dimension(
        provider=settings.EMBEDDING_PROVIDER,
        model_name=settings.EMBEDDING_MODEL,
    )

    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=dimension,
            distance=models.Distance.COSINE,
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=0,  # Build index immediately
        ),
    )

    # Create payload indexes for efficient filtered retrieval
    keyword_fields = [
        "source",
        "plan_name",
        "plan_type",
        "plan_number",
        "uin",
        "section_type",
        "content_type",
        "category",
        "chunk_tags",
        "benefit_types",
        "riders_mentioned",
        "project_ids",  # Multi-tenant project filtering
    ]
    for field_name in keyword_fields:
        client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    # Integer indexes
    client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        field_name="page_number",
        field_schema=models.PayloadSchemaType.INTEGER,
    )

    print(
        f"Created Qdrant collection: {settings.QDRANT_COLLECTION_NAME} "
        f"(dimension: {dimension}, indexes: {len(keyword_fields) + 1})"
    )


def init_qdrant():
    """Initialize Qdrant collection if it doesn't exist."""
    try:
        client = get_qdrant_client()
        collections = client.get_collections().collections
        exists = any(c.name == settings.QDRANT_COLLECTION_NAME for c in collections)

        if not exists:
            create_collection_with_indexes(client)
        else:
            print(f"Qdrant collection '{settings.QDRANT_COLLECTION_NAME}' already exists")
    except Exception as e:
        print(f"Error connecting to Qdrant at {settings.QDRANT_URL}: {e}")
        print("Make sure Qdrant is running (docker-compose up -d qdrant)")
        raise


def recreate_collection():
    """Drop and recreate the collection. Use when re-indexing with new embeddings."""
    client = get_qdrant_client()
    try:
        client.delete_collection(settings.QDRANT_COLLECTION_NAME)
        print(f"Deleted collection: {settings.QDRANT_COLLECTION_NAME}")
    except Exception:
        pass
    create_collection_with_indexes(client)


# BM25 Index Cache (thread-safe singleton)
_bm25_retriever = None
_bm25_lock = threading.Lock()


def get_cached_bm25_retriever():
    """Get cached BM25 retriever instance (thread-safe singleton). Builds index on first call."""
    global _bm25_retriever
    if _bm25_retriever is None:
        with _bm25_lock:
            # Double-check after acquiring lock
            if _bm25_retriever is None:
                from app.services.rag.bm25_retriever import BM25Retriever

                print("Initializing BM25 retriever...")
                _bm25_retriever = BM25Retriever()
                _bm25_retriever.build_index()
    return _bm25_retriever


def rebuild_bm25_index():
    """Force rebuild of BM25 index. Call after ingesting new documents."""
    global _bm25_retriever
    with _bm25_lock:
        if _bm25_retriever is not None:
            _bm25_retriever.build_index(force_rebuild=True)
        else:
            # Create directly here instead of calling get_cached_bm25_retriever()
            # to avoid deadlock (it also tries to acquire _bm25_lock)
            from app.services.rag.bm25_retriever import BM25Retriever

            _bm25_retriever = BM25Retriever()
            _bm25_retriever.build_index()
