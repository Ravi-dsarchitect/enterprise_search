from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import re
from qdrant_client import QdrantClient
from app.core.database import get_qdrant_client
from app.core.config import settings

class BM25Retriever:
    """
    Sparse retrieval using BM25 algorithm for keyword-based search.
    Complements dense vector search in hybrid retrieval.
    """
    
    def __init__(self):
        self.qdrant: QdrantClient = get_qdrant_client()
        self.corpus: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._index_built = False
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase, remove special chars, split on whitespace.
        """
        if not text:
            return []
        # Lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Split and filter empty tokens
        tokens = [token for token in text.split() if token]
        return tokens
    
    def build_index(self, force_rebuild: bool = False):
        """
        Build BM25 index from all documents in Qdrant collection.
        This should be called after ingestion or periodically.
        """
        if self._index_built and not force_rebuild:
            print("BM25 index already built. Use force_rebuild=True to rebuild.")
            return
        
        print("🔨 Building BM25 index from Qdrant collection...")
        
        # Fetch all documents from Qdrant
        # Note: For very large collections, consider pagination
        scroll_result = self.qdrant.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            limit=10000,  # Adjust based on collection size
            with_payload=True,
            with_vectors=False  # We don't need vectors for BM25
        )
        
        points = scroll_result[0]
        
        if not points:
            print("⚠️  No documents found in collection. BM25 index is empty.")
            self.corpus = []
            self.tokenized_corpus = []
            self.bm25 = None
            self._index_built = True
            return
        
        # Extract text and metadata
        self.corpus = []
        for point in points:
            self.corpus.append({
                "id": point.id,
                "text": point.payload.get("text", ""),
                "payload": point.payload
            })
        
        # Tokenize corpus
        self.tokenized_corpus = [
            self._tokenize(doc["text"]) for doc in self.corpus
        ]
        
        # Build BM25 index
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            self._index_built = True
            print(f"✅ BM25 index built with {len(self.corpus)} documents")
        else:
            print("⚠️  No valid documents to index")
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform BM25 search and return top-k results with scores.
        
        Returns:
            List of dicts with 'id', 'text', 'score', 'payload'
        """
        if not self._index_built or self.bm25 is None:
            print("⚠️  BM25 index not built. Building now...")
            self.build_index()
        
        if not self.bm25 or not self.corpus:
            return []
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        if not tokenized_query:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:limit]
        
        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return documents with positive scores
                results.append({
                    "id": self.corpus[idx]["id"],
                    "text": self.corpus[idx]["text"],
                    "score": float(scores[idx]),
                    "source": self.corpus[idx]["payload"].get("source", ""),
                    "payload": self.corpus[idx]["payload"]
                })
        
        return results
    
    def get_index_size(self) -> int:
        """Return number of documents in BM25 index."""
        return len(self.corpus) if self.corpus else 0
