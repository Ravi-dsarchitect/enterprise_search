from typing import List, Dict, Any

class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) for merging ranked lists from multiple retrievers.
    RRF is effective for combining results from different ranking systems.
    
    Formula: RRF_score(d) = Σ 1 / (k + rank(d))
    where k is a constant (typically 60), and rank(d) is the rank of document d.
    """
    
    def __init__(self, k: int = 60):
        """
        Args:
            k: Ranking constant for RRF. Higher values decrease the impact of high ranks.
               Default 60 is commonly used in literature.
        """
        self.k = k
    
    def fuse(
        self, 
        ranked_lists: List[List[Dict[str, Any]]], 
        id_key: str = "id",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fuse multiple ranked lists using RRF.
        
        Args:
            ranked_lists: List of ranked result lists. Each list contains dicts with documents.
            id_key: Key to use for document identification (e.g., "id", "source")
            limit: Number of top results to return
            
        Returns:
            Merged and re-ranked list of documents with RRF scores
        """
        if not ranked_lists:
            return []
        
        # Filter out empty lists
        ranked_lists = [lst for lst in ranked_lists if lst]
        
        if not ranked_lists:
            return []
        
        # Calculate RRF scores
        rrf_scores: Dict[str, float] = {}
        doc_data: Dict[str, Dict[str, Any]] = {}
        
        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list, start=1):
                doc_id = doc.get(id_key)
                
                if doc_id is None:
                    continue
                
                # RRF formula: 1 / (k + rank)
                rrf_score = 1.0 / (self.k + rank)
                
                # Accumulate scores across different ranked lists
                if doc_id in rrf_scores:
                    rrf_scores[doc_id] += rrf_score
                else:
                    rrf_scores[doc_id] = rrf_score
                    doc_data[doc_id] = doc  # Store document data
        
        # Sort by RRF score (descending)
        sorted_doc_ids = sorted(
            rrf_scores.keys(), 
            key=lambda doc_id: rrf_scores[doc_id], 
            reverse=True
        )
        
        # Build final result list
        fused_results = []
        for doc_id in sorted_doc_ids[:limit]:
            doc = doc_data[doc_id].copy()
            # Add RRF score to document
            doc["rrf_score"] = rrf_scores[doc_id]
            # Keep original score if it exists, rename it
            if "score" in doc:
                doc["original_score"] = doc["score"]
            doc["score"] = rrf_scores[doc_id]  # Use RRF as primary score
            fused_results.append(doc)
        
        return fused_results


class WeightedFusion:
    """
    Alternative fusion strategy using weighted score combination.
    """
    
    def __init__(self, weights: List[float] = None):
        """
        Args:
            weights: List of weights for each ranked list. Must sum to 1.0
                    If None, equal weights are used.
        """
        self.weights = weights
    
    def fuse(
        self,
        ranked_lists: List[List[Dict[str, Any]]],
        id_key: str = "id",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fuse results using weighted score combination.
        """
        if not ranked_lists:
            return []
        
        ranked_lists = [lst for lst in ranked_lists if lst]
        
        if not ranked_lists:
            return []
        
        # Set equal weights if not provided
        if self.weights is None:
            self.weights = [1.0 / len(ranked_lists)] * len(ranked_lists)
        
        # Normalize weights
        total_weight = sum(self.weights)
        normalized_weights = [w / total_weight for w in self.weights]
        
        # Combine scores
        combined_scores: Dict[str, float] = {}
        doc_data: Dict[str, Dict[str, Any]] = {}
        
        for weight, ranked_list in zip(normalized_weights, ranked_lists):
            # Normalize scores within this list to [0, 1]
            if not ranked_list:
                continue
                
            # Filter out None values before computing min/max
            scores = [doc.get("score", 0) for doc in ranked_list if doc.get("score") is not None]
            if not scores:
                scores = [0]  # Default to 0 if all scores are None
            max_score = max(scores)
            min_score = min(scores)
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            for doc in ranked_list:
                doc_id = doc.get(id_key)
                if doc_id is None:
                    continue
                
                # Normalize score
                raw_score = doc.get("score", 0)
                normalized_score = (raw_score - min_score) / score_range
                weighted_score = normalized_score * weight
                
                if doc_id in combined_scores:
                    combined_scores[doc_id] += weighted_score
                else:
                    combined_scores[doc_id] = weighted_score
                    doc_data[doc_id] = doc
        
        # Sort by combined score
        sorted_doc_ids = sorted(
            combined_scores.keys(),
            key=lambda doc_id: combined_scores[doc_id],
            reverse=True
        )
        
        # Build result list
        fused_results = []
        for doc_id in sorted_doc_ids[:limit]:
            doc = doc_data[doc_id].copy()
            doc["fused_score"] = combined_scores[doc_id]
            if "score" in doc:
                doc["original_score"] = doc["score"]
            doc["score"] = combined_scores[doc_id]
            fused_results.append(doc)
        
        return fused_results
