from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from app.core.config import settings


class Reranker:
    """Cross-encoder reranker using sentence-transformers.

    Uses BAAI/bge-reranker-v2-m3 by default — one of the top open-source
    cross-encoders. Takes (query, passage) pairs and outputs relevance scores.
    """

    def __init__(self, model_name: str = None):
        model_name = model_name or settings.RERANKER_MODEL
        print(f"Loading cross-encoder reranker: {model_name}")
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        print(f"Reranker loaded: {model_name}")

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder relevance scores.

        Args:
            query: The search query.
            docs: List of dicts with at least a "text" field.
            top_n: Number of top results to return.

        Returns:
            Top-n documents sorted by cross-encoder relevance score.
        """
        if not docs:
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, doc.get("text", "")) for doc in docs]

        # Predict relevance scores
        scores = self.model.predict(pairs)

        # Pair each doc with its score and sort descending
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: float(x[1]), reverse=True)

        # Return top_n with updated scores
        reranked = []
        for doc, score in scored_docs[:top_n]:
            doc = dict(doc)  # avoid mutating the original
            doc["score"] = float(score)
            reranked.append(doc)

        return reranked
