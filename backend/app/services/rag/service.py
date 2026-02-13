import math
import time
from typing import Dict, Any, List
from app.services.rag.generator import LLMFactory, LLMGenerator, AnswerCritic
from app.services.rag.query_transformer import QueryTransformer
from app.services.rag.query_analyzer import QueryAnalyzer
from app.services.rag.retriever import Retriever
from app.core.config import settings


# Singleton instance
_rag_service_instance: "RAGService" = None


def get_rag_service() -> "RAGService":
    """Get or create the singleton RAGService instance."""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance


class RAGService:
    # Filter categories for fallback logic
    # Hard filters: explicit mentions that should always be applied
    HARD_FILTER_KEYS = {"plan_name", "plan_number", "source", "category"}
    # Soft filters: inferred from query intent, can be loosened if no results
    SOFT_FILTER_KEYS = {
        "section_type", "content_type", "chunk_tags",
        "contains_age_info", "contains_currency",
        "plan_type", "premium_type", "document_date"  # Inferred filters
    }

    def __init__(self):
        self.retriever = Retriever()
        self.generator: LLMGenerator = LLMFactory.create_generator()
        self.query_transformer = QueryTransformer()
        self.query_analyzer = QueryAnalyzer()
        self.critic = AnswerCritic() if settings.ENABLE_ANSWER_CRITIC else None

    def _split_filters(self, filters: Dict[str, Any]) -> tuple:
        """Split filters into hard (always apply) and soft (can loosen)."""
        if not filters:
            return None, None
        hard = {k: v for k, v in filters.items() if k in self.HARD_FILTER_KEYS}
        soft = {k: v for k, v in filters.items() if k in self.SOFT_FILTER_KEYS}
        return hard or None, soft or None

    async def _retrieve_with_fallback(
        self,
        search_text: str,
        limit: int,
        use_hybrid: bool,
        metadata_filters: Dict[str, Any],
        is_hyde: bool,
        min_results: int = 1,
        min_score: float = 0.4,  # Minimum acceptable top score for fallback
        hard_threshold: float = 0.5,  # Hard minimum - return empty if below this
    ) -> tuple:
        """
        Retrieve with fallback: full filters → hard filters only → no filters.

        Fallback triggers:
        1. Too few results (< min_results)
        2. Top score too low (< min_score)

        Hard threshold: If top score < hard_threshold, return empty results (no answer).

        Returns: (results, filters_used, fallback_level)
            fallback_level: 0=full, 1=hard-only, 2=no-filters
        """
        hard_filters, soft_filters = self._split_filters(metadata_filters)

        # Level 0: Try with ALL filters
        if metadata_filters:
            results = await self.retriever.search(
                search_text, limit=limit, use_hybrid=use_hybrid,
                metadata_filters=metadata_filters, is_hyde=is_hyde
            )
            top_score = results[0].get("score", 0.0) if results else 0.0

            # Hard threshold check - return empty if too low
            if results and top_score < hard_threshold:
                print(f"  ❌ Top score {top_score:.3f} below hard threshold {hard_threshold}, returning no results")
                return [], metadata_filters, 0

            # Check both quantity and quality
            if len(results) >= min_results and top_score >= min_score:
                return results, metadata_filters, 0

            # Log why we're falling back
            if len(results) < min_results:
                print(f"  ⚠️  Only {len(results)} results with full filters, trying hard-only...")
            elif top_score < min_score:
                print(f"  ⚠️  Top score {top_score:.3f} < {min_score} with full filters, trying hard-only...")

        # Level 1: Try with HARD filters only (drop soft filters)
        if hard_filters:
            results = await self.retriever.search(
                search_text, limit=limit, use_hybrid=use_hybrid,
                metadata_filters=hard_filters, is_hyde=is_hyde
            )
            top_score = results[0].get("score", 0.0) if results else 0.0

            # Hard threshold check - return empty if too low
            if results and top_score < hard_threshold:
                print(f"  ❌ Top score {top_score:.3f} below hard threshold {hard_threshold}, returning no results")
                return [], hard_filters, 1

            if len(results) >= min_results and top_score >= min_score:
                print(f"  ✅ Got {len(results)} results with hard filters only (top score: {top_score:.3f})")
                return results, hard_filters, 1

            # Log why we're falling back
            if len(results) < min_results:
                print(f"  ⚠️  Only {len(results)} results with hard filters, trying no filters...")
            elif top_score < min_score:
                print(f"  ⚠️  Top score {top_score:.3f} < {min_score} with hard filters, trying no filters...")

        # Level 2: No filters (pure semantic search)
        results = await self.retriever.search(
            search_text, limit=limit, use_hybrid=use_hybrid,
            metadata_filters=None, is_hyde=is_hyde
        )
        top_score = results[0].get("score", 0.0) if results else 0.0

        # Hard threshold check - return empty if too low
        if results and top_score < hard_threshold:
            print(f"  ❌ Top score {top_score:.3f} below hard threshold {hard_threshold}, returning no results")
            return [], None, 2

        print(f"  📊 Got {len(results)} results without filters (top score: {top_score:.3f})")
        return results, None, 2

    @staticmethod
    def _compute_confidence(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert raw reranker/retrieval scores to softmax probabilities."""
        if not docs:
            return docs
        scores = [doc.get("score", 0.0) for doc in docs]
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        for doc, exp_s in zip(docs, exp_scores):
            doc["confidence"] = exp_s / total
        return docs

    @staticmethod
    def _filter_by_confidence(docs: List[Dict[str, Any]], min_confidence: float = 0.10) -> List[Dict[str, Any]]:
        """
        Keep only documents above minimum confidence threshold.
        Ensures at least 1 document (the top result) is always kept.

        Args:
            docs: List of documents with confidence scores
            min_confidence: Minimum confidence threshold (default 0.10 = 10%)

        Returns:
            Filtered list of documents
        """
        if not docs:
            return docs

        # Always keep top result
        filtered = [docs[0]]

        # Add additional results if they meet threshold
        for doc in docs[1:]:
            confidence = doc.get("confidence", 0)
            if confidence >= min_confidence:
                filtered.append(doc)
            else:
                # Since sorted by confidence (via score), we can stop early
                print(f"  🔽 Dropping low-confidence citations (<{min_confidence*100:.0f}%): {len(docs) - len(filtered)} citations removed")
                break

        return filtered

    def _build_context_str(self, context):
        """Build formatted context string for the critic (same format as generator)."""
        parts = []
        for i, item in enumerate(context):
            payload = item.get("payload", {})
            source = item.get("source", "unknown")
            page = payload.get("page_number", "")
            section = payload.get("section_header", "") or ""
            score = item.get("score", 0)

            header = f"[{i+1}] Source: {source}"
            if page:
                header += f" | Page {page}"
            if section:
                header += f" | Section: {section}"
            header += f" (relevance: {score:.2f})"

            parts.append(f"{header}\n{item['text']}\n{'='*60}")
        return "\n\n".join(parts)

    async def answer_query(
        self, 
        query: str,
        conversation_history: list = None,
        use_hyde: bool = False, 
        use_decomposition: bool = False, 
        use_hybrid_search: bool = False,
        use_auto_filters: bool = True,
        metadata_filters: Dict[str, Any] = None,
        limit: int = 5,
        project_ids: List[str] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        print(f"\n🔍 Processing query: {query}")
        
        # Auto-extract filters if enabled and no manual filters provided
        if use_auto_filters and not metadata_filters:
            metadata_filters = self.query_analyzer.extract_filters(query)
        
        # Add project_ids to metadata filters for multi-tenant filtering
        if project_ids:
            if metadata_filters is None:
                metadata_filters = {}
            metadata_filters["project_ids"] = project_ids
        
        print(f"⚙️  Settings: HyDE={use_hyde}, Decomposition={use_decomposition}, Hybrid={use_hybrid_search}, AutoFilters={use_auto_filters}, Filters={metadata_filters is not None}, ProjectIDs={project_ids}, Limit={limit}")
        
        queries_to_run = [query]
        
        # 1. Query Transformation (Expansion)
        if use_decomposition:
            queries_to_run = self.query_transformer.decompose_query(query)
            print(f"Expanded queries: {queries_to_run}")

        all_results = []
        
        # 2. Retrieval Loop
        for q in queries_to_run:
            search_text = q
            is_hyde_query = False

            # HyDE Transformation (Per query)
            if use_hyde:
                hypo_doc = self.query_transformer.generate_hyde_doc(q)
                if hypo_doc:
                    search_text = hypo_doc
                    is_hyde_query = True

            # Retrieve with fallback (full filters → hard-only → no filters)
            retrieval_start = time.time()
            results, filters_used, fallback_level = await self._retrieve_with_fallback(
                search_text,
                limit=limit,
                use_hybrid=use_hybrid_search,
                metadata_filters=metadata_filters,
                is_hyde=is_hyde_query,
                min_results=2,  # Minimum results before fallback
            )
            all_results.extend(results)
            fallback_msg = ["full filters", "hard filters only", "no filters"][fallback_level]
            print(f"Retrieval took: {time.time() - retrieval_start:.2f}s ({fallback_msg})")
            
        # 3. Deduplication (by source and text content)
        seen = set()
        unique_results = []
        for doc in all_results:
            # Create a unique key key based on source and a snippet of text
            key = (doc['source'], doc['text'][:100]) 
            if key not in seen:
                seen.add(key)
                unique_results.append(doc)
        
        # 4. Rerank Global Results (Optional but recommended if we have many from expansion)
        # The retriever already reranks implicitly per search, but re-ranking the combined set is better.
        # For simplicity, we'll trust the individual top results or just slice.
        final_context = unique_results[:limit]  # Respect the limit parameter
        final_context = self._compute_confidence(final_context)
        final_context = self._filter_by_confidence(final_context, min_confidence=0.10)  # Dynamic K: filter low-confidence citations

        # 5. Generate Answer
        llm_start = time.time()
        answer = self.generator.generate_answer(query, final_context, conversation_history=conversation_history)
        print(f"⏱️  LLM generation took: {time.time() - llm_start:.2f}s")

        # 6. Critic: verify answer against source passages (optional)
        if self.critic:
            critic_start = time.time()
            context_str = self._build_context_str(final_context)
            verified_answer = self.critic.verify_and_fix(answer, query, context_str)
            critic_time = time.time() - critic_start
            if verified_answer != answer:
                print(f"🔄 Critic corrected the answer ({critic_time:.2f}s)")
                answer = verified_answer
            else:
                print(f"✅ Critic passed the answer ({critic_time:.2f}s)")

        total_time = time.time() - start_time
        print(f"✅ Total query time: {total_time:.2f}s\n")

        return {
            "query": query,
            "answer": answer,
            "citations": final_context,
            "generated_queries": queries_to_run if use_decomposition else []
        }
    
    async def answer_query_stream(
        self, 
        query: str,
        conversation_history: list = None,
        use_hyde: bool = False, 
        use_decomposition: bool = False, 
        use_hybrid_search: bool = False,
        use_auto_filters: bool = True,
        metadata_filters: Dict[str, Any] = None,
        limit: int = 5,
        project_ids: List[str] = None
    ):
        """
        Stream answer generation. Retrieval happens first, then answer is streamed.
        Yields dict events with 'type' and 'data' fields.
        """
        start_time = time.time()
        print(f"\n🔍 Processing query (streaming): {query}")
        
        # Auto-extract filters if enabled and no manual filters provided
        if use_auto_filters and not metadata_filters:
            metadata_filters = self.query_analyzer.extract_filters(query)
        
        # Add project_ids to metadata filters for multi-tenant filtering
        if project_ids:
            if metadata_filters is None:
                metadata_filters = {}
            metadata_filters["project_ids"] = project_ids
        
        print(f"⚙️  Settings: HyDE={use_hyde}, Decomposition={use_decomposition}, Hybrid={use_hybrid_search}, AutoFilters={use_auto_filters}, Filters={metadata_filters is not None}, ProjectIDs={project_ids}, Limit={limit}")
        
        queries_to_run = [query]
        
        # 1. Query Transformation (Expansion)
        if use_decomposition:
            queries_to_run = self.query_transformer.decompose_query(query)
            print(f"Expanded queries: {queries_to_run}")

        all_results = []
        
        # 2. Retrieval Loop
        for q in queries_to_run:
            search_text = q
            is_hyde_query = False

            # HyDE Transformation (Per query)
            if use_hyde:
                hypo_doc = self.query_transformer.generate_hyde_doc(q)
                if hypo_doc:
                    search_text = hypo_doc
                    is_hyde_query = True

            # Retrieve with fallback (full filters → hard-only → no filters)
            retrieval_start = time.time()
            results, filters_used, fallback_level = await self._retrieve_with_fallback(
                search_text,
                limit=limit,
                use_hybrid=use_hybrid_search,
                metadata_filters=metadata_filters,
                is_hyde=is_hyde_query,
                min_results=2,  # Minimum results before fallback
            )
            all_results.extend(results)
            fallback_msg = ["full filters", "hard filters only", "no filters"][fallback_level]
            print(f"Retrieval took: {time.time() - retrieval_start:.2f}s ({fallback_msg})")
            
        # 3. Deduplication
        seen = set()
        unique_results = []
        for doc in all_results:
            key = (doc['source'], doc['text'][:100])
            if key not in seen:
                seen.add(key)
                unique_results.append(doc)

        final_context = unique_results[:limit]  # Respect the limit parameter
        final_context = self._compute_confidence(final_context)
        final_context = self._filter_by_confidence(final_context, min_confidence=0.10)  # Dynamic K: filter low-confidence citations

        # Yield metadata and citations first
        yield {
            "type": "metadata",
            "data": {
                "query": query,
                "retrieval_time": time.time() - start_time,
                "num_citations": len(final_context)
            }
        }
        
        yield {
            "type": "citations",
            "data": final_context
        }
        
        # 4. Stream the answer
        llm_start = time.time()
        full_answer = ""
        
        async for token in self.generator.generate_answer_stream(query, final_context, conversation_history=conversation_history):
            full_answer += token
            yield {
                "type": "token",
                "data": token
            }
        
        print(f"⏱️  LLM generation took: {time.time() - llm_start:.2f}s")

        # 5. Critic: verify answer against source passages (optional)
        if self.critic:
            critic_start = time.time()
            context_str = self._build_context_str(final_context)
            verified_answer = self.critic.verify_and_fix(full_answer, query, context_str)
            critic_time = time.time() - critic_start
            if verified_answer != full_answer:
                print(f"🔄 Critic corrected the answer ({critic_time:.2f}s)")
                full_answer = verified_answer
                yield {
                    "type": "correction",
                    "data": full_answer
                }
            else:
                print(f"✅ Critic passed the answer ({critic_time:.2f}s)")

        print(f"✅ Total query time: {time.time() - start_time:.2f}s\n")

        # Yield completion event
        yield {
            "type": "done",
            "data": {
                "total_time": time.time() - start_time,
                "answer": full_answer
            }
        }
