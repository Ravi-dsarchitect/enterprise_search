import tempfile
import os
from typing import List, Dict, Any
from flashrank import Ranker, RerankRequest
from app.core.config import settings


class Reranker:
    def __init__(self, model_name: str = None):
        model_name = model_name or settings.RERANKER_MODEL
        cache_dir = os.path.join(tempfile.gettempdir(), "flashrank")
        self.ranker = Ranker(model_name=model_name, cache_dir=cache_dir)

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        if not docs:
            return []
            
        # Format for FlashRank
        passages = [
            {"id": str(i), "text": doc.get("text", "")} 
            for i, doc in enumerate(docs)
        ]
        
        request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(request)
        
        # Map back to original docs
        reranked_docs = []
        for res in results[:top_n]:
            try:
                original_idx = int(res["id"])
                doc = docs[original_idx]
                doc["score"] = float(res["score"])  # Convert to Python float for serialization
                reranked_docs.append(doc)
            except (ValueError, IndexError):
                continue
                
        return reranked_docs
