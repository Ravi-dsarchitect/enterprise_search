from typing import List, Dict, Any, Optional
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from app.core.database import get_qdrant_client, get_cached_bm25_retriever
from app.core.config import settings
from app.core.model_cache import get_cached_embedder, get_cached_reranker
from app.services.ingestion.interfaces import Embedder
from app.services.rag.reranker import Reranker
from app.services.rag.bm25_retriever import BM25Retriever
from app.services.rag.fusion import ReciprocalRankFusion

class Retriever:
    def __init__(self):
        self.qdrant: QdrantClient = get_qdrant_client()
        
        # Use cached models (loaded once, reused across requests)
        self.embedder: Embedder = get_cached_embedder()
        self.reranker: Reranker = get_cached_reranker()
        
        # BM25 retriever for hybrid search (lazy loaded)
        self._bm25_retriever: Optional[BM25Retriever] = None
        self.rrf_fusion = ReciprocalRankFusion(k=60)
    
    def _get_bm25_retriever(self) -> BM25Retriever:
        """Lazy load BM25 retriever."""
        if self._bm25_retriever is None:
            self._bm25_retriever = get_cached_bm25_retriever()
        return self._bm25_retriever
    
    def _build_qdrant_filter(self, metadata_filters: Dict[str, Any]) -> Optional[qdrant_models.Filter]:
        """
        Build Qdrant filter from metadata dictionary.
        
        Supports:
        - Exact match: {"source": "document.pdf"}
        - List (OR): {"document_type": ["pdf", "docx"]}
        - Range: {"date": {"gte": "2024-01-01", "lte": "2026-01-21"}}
        
        Note: Date fields are converted to Unix timestamps for numeric range comparison.
        """
        if not metadata_filters:
            return None
        
        conditions = []
        
        # Fields that contain dates and should be converted to timestamps
        date_fields = {"document_date", "created_at", "modified_at", "date"}
        
        for key, value in metadata_filters.items():
            if isinstance(value, list):
                # OR condition for list values
                conditions.append(
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchAny(any=value)
                    )
                )
            elif isinstance(value, dict):
                # Range condition - check if this is a date field
                is_date_field = key in date_fields
                range_params = {}
                
                for range_key in ["gte", "lte", "gt", "lt"]:
                    if range_key in value:
                        range_value = value[range_key]
                        
                        # Convert date strings to Unix timestamps for numeric comparison
                        if is_date_field and isinstance(range_value, str):
                            try:
                                # Parse ISO date format (YYYY-MM-DD) to timestamp
                                dt = datetime.fromisoformat(range_value)
                                range_params[range_key] = dt.timestamp()
                            except (ValueError, TypeError):
                                # If conversion fails, use the original value
                                range_params[range_key] = range_value
                        else:
                            range_params[range_key] = range_value
                
                if range_params:
                    conditions.append(
                        qdrant_models.FieldCondition(
                            key=key,
                            range=qdrant_models.Range(**range_params)
                        )
                    )
            else:
                # Exact match
                conditions.append(
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchValue(value=value)
                    )
                )
        
        if not conditions:
            return None
        
        return qdrant_models.Filter(must=conditions)

    async def search(
        self,
        query: str,
        limit: int = 5,
        use_hybrid: bool = False,
        metadata_filters: Optional[Dict[str, Any]] = None,
        is_hyde: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search with optional hybrid mode and metadata filtering.

        Args:
            query: Search query text
            limit: Number of final results to return
            use_hybrid: If True, combine vector + BM25 search using RRF
            metadata_filters: Optional filters to apply (e.g., {"source": "doc.pdf"})
            is_hyde: If True, embed query as document (no BGE prefix) for HyDE search

        Returns:
            List of ranked documents with scores
        """
        # Build Qdrant filter if metadata filters provided
        qdrant_filter = self._build_qdrant_filter(metadata_filters)

        if use_hybrid:
            print(f"Hybrid search mode: Vector + BM25")
            return await self._hybrid_search(query, limit, qdrant_filter, metadata_filters, is_hyde)
        else:
            return await self._vector_search(query, limit, qdrant_filter, is_hyde)
    
    async def _vector_search(
        self,
        query: str,
        limit: int,
        qdrant_filter: Optional[qdrant_models.Filter] = None,
        is_hyde: bool = False,
    ) -> List[Dict[str, Any]]:
        """Standard vector-only search."""
        # 1. Generate Embedding
        # HyDE: embed as document (no query prefix) since it's a hypothetical document
        if is_hyde:
            query_vector = self.embedder.embed_documents([query])[0]
        else:
            query_vector = self.embedder.embed_query(query)
        
        # 2. Semantic Search (Fetch more candidates for re-ranking)
        candidate_limit = limit * 2
        
        results = self.qdrant.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query=query_vector,
            limit=candidate_limit,
            with_payload=True,
            query_filter=qdrant_filter  # Apply metadata filters
        ).points
        
        initial_docs = [
            {
                "id": hit.id,
                "text": hit.payload.get("text"),
                "source": hit.payload.get("source"),
                "score": float(hit.score),  # Convert numpy.float32 to Python float
                "payload": hit.payload  # Keep full payload
            }
            for hit in results
        ]
        
        # 3. Rerank Results
        ranked_docs = self.reranker.rerank(query, initial_docs, top_n=limit)
        
        return ranked_docs
    
    async def _hybrid_search(
        self,
        query: str,
        limit: int,
        qdrant_filter: Optional[qdrant_models.Filter] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        is_hyde: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector and BM25 using Reciprocal Rank Fusion.
        """
        # Fetch more candidates for fusion
        candidate_limit = limit * 3

        # 1. Vector Search
        # HyDE: embed as document (no query prefix) since it's a hypothetical document
        if is_hyde:
            query_vector = self.embedder.embed_documents([query])[0]
        else:
            query_vector = self.embedder.embed_query(query)
        
        vector_results = self.qdrant.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query=query_vector,
            limit=candidate_limit,
            with_payload=True,
            query_filter=qdrant_filter
        ).points
        
        vector_docs = [
            {
                "id": str(hit.id),  # Ensure string for RRF matching
                "text": hit.payload.get("text"),
                "source": hit.payload.get("source"),
                "score": float(hit.score),
                "payload": hit.payload
            }
            for hit in vector_results
        ]
        
        # 2. BM25 Search
        bm25_retriever = self._get_bm25_retriever()
        bm25_docs = bm25_retriever.search(query, limit=candidate_limit)
        
        # Convert BM25 IDs to strings for matching
        for doc in bm25_docs:
            doc["id"] = str(doc["id"])
        
        # Apply metadata filters to BM25 results if needed
        if metadata_filters:
            bm25_docs = self._filter_bm25_results(bm25_docs, metadata_filters)
        
        print(f"  - Vector search: {len(vector_docs)} results")
        print(f"  - BM25 search: {len(bm25_docs)} results")
        
        # 3. Fuse results using RRF
        fused_docs = self.rrf_fusion.fuse(
            ranked_lists=[vector_docs, bm25_docs],
            id_key="id",
            limit=limit * 2  # Get more for reranking
        )
        
        print(f"  - Fused results: {len(fused_docs)} unique documents")
        
        # 4. Rerank fused results
        final_docs = self.reranker.rerank(query, fused_docs, top_n=limit)
        
        return final_docs
    
    
    def _filter_bm25_results(
        self, 
        bm25_docs: List[Dict[str, Any]], 
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply metadata filters to BM25 results manually."""
        filtered = []
        
        # Fields that contain dates stored as timestamps
        date_fields = {"document_date", "created_at", "modified_at", "date"}
        
        for doc in bm25_docs:
            payload = doc.get("payload", {})
            matches = True
            
            for key, value in filters.items():
                payload_value = payload.get(key)
                
                if isinstance(value, list):
                    if payload_value not in value:
                        matches = False
                        break
                elif isinstance(value, dict):
                    # Range filtering - convert date strings to timestamps
                    is_date_field = key in date_fields
                    
                    for range_key, range_value in value.items():
                        if range_key in ["gte", "lte", "gt", "lt"]:
                            # Convert date strings to timestamps for comparison
                            if is_date_field and isinstance(range_value, str):
                                try:
                                    dt = datetime.fromisoformat(range_value)
                                    range_value = dt.timestamp()
                                except (ValueError, TypeError):
                                    pass  # Use original value if conversion fails
                            
                            # Perform the comparison (skip if payload_value is None)
                            if payload_value is None:
                                matches = False
                                break
                            
                            if range_key == "gte" and payload_value < range_value:
                                matches = False
                                break
                            elif range_key == "lte" and payload_value > range_value:
                                matches = False
                                break
                            elif range_key == "gt" and payload_value <= range_value:
                                matches = False
                                break
                            elif range_key == "lt" and payload_value >= range_value:
                                matches = False
                                break
                    
                    if not matches:
                        break
                else:
                    if payload_value != value:
                        matches = False
                        break
            
            if matches:
                filtered.append(doc)
        
        return filtered
