import time
from typing import Dict, Any
from app.services.rag.generator import LLMFactory, LLMGenerator
from app.services.rag.query_transformer import QueryTransformer
from app.services.rag.query_analyzer import QueryAnalyzer
from app.services.rag.retriever import Retriever
from app.core.config import settings

class RAGService:
    def __init__(self):
        self.retriever = Retriever()
        self.generator: LLMGenerator = LLMFactory.create_generator()
        self.query_transformer = QueryTransformer()
        self.query_analyzer = QueryAnalyzer()

    async def answer_query(
        self, 
        query: str,
        conversation_history: list = None,
        use_hyde: bool = False, 
        use_decomposition: bool = False, 
        use_hybrid_search: bool = False,
        use_auto_filters: bool = True,
        metadata_filters: Dict[str, Any] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        start_time = time.time()
        print(f"\n🔍 Processing query: {query}")
        
        # Auto-extract filters if enabled and no manual filters provided
        if use_auto_filters and not metadata_filters:
            metadata_filters = self.query_analyzer.extract_filters(query)
        
        print(f"⚙️  Settings: HyDE={use_hyde}, Decomposition={use_decomposition}, Hybrid={use_hybrid_search}, AutoFilters={use_auto_filters}, Filters={metadata_filters is not None}, Limit={limit}")
        
        queries_to_run = [query]
        
        # 1. Query Transformation (Expansion)
        if use_decomposition:
            queries_to_run = self.query_transformer.decompose_query(query)
            print(f"Expanded queries: {queries_to_run}")

        all_results = []
        
        # 2. Retrieval Loop
        for q in queries_to_run:
            search_text = q
            
            # HyDE Transformation (Per query)
            if use_hyde:
                hypo_doc = self.query_transformer.generate_hyde_doc(q)
                if hypo_doc:
                    search_text = hypo_doc # Search using the hypothetical document vector
            
            # Retrieve
            retrieval_start = time.time()
            results = await self.retriever.search(
                search_text, 
                limit=limit,
                use_hybrid=use_hybrid_search,
                metadata_filters=metadata_filters
            )
            all_results.extend(results)
            print(f"⏱️  Retrieval took: {time.time() - retrieval_start:.2f}s")
            
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
        final_context = unique_results[:10] # Cap context size
        
        # 5. Generate Answer
        llm_start = time.time()
        answer = self.generator.generate_answer(query, final_context, conversation_history=conversation_history)
        print(f"⏱️  LLM generation took: {time.time() - llm_start:.2f}s")
        
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
        limit: int = 5
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
        
        print(f"⚙️  Settings: HyDE={use_hyde}, Decomposition={use_decomposition}, Hybrid={use_hybrid_search}, AutoFilters={use_auto_filters}, Filters={metadata_filters is not None}, Limit={limit}")
        
        queries_to_run = [query]
        
        # 1. Query Transformation (Expansion)
        if use_decomposition:
            queries_to_run = self.query_transformer.decompose_query(query)
            print(f"Expanded queries: {queries_to_run}")

        all_results = []
        
        # 2. Retrieval Loop
        for q in queries_to_run:
            search_text = q
            
            # HyDE Transformation (Per query)
            if use_hyde:
                hypo_doc = self.query_transformer.generate_hyde_doc(q)
                if hypo_doc:
                    search_text = hypo_doc
            
            # Retrieve
            retrieval_start = time.time()
            results = await self.retriever.search(
                search_text, 
                limit=limit,
                use_hybrid=use_hybrid_search,
                metadata_filters=metadata_filters
            )
            all_results.extend(results)
            print(f"⏱️  Retrieval took: {time.time() - retrieval_start:.2f}s")
            
        # 3. Deduplication
        seen = set()
        unique_results = []
        for doc in all_results:
            key = (doc['source'], doc['text'][:100]) 
            if key not in seen:
                seen.add(key)
                unique_results.append(doc)
        
        final_context = unique_results[:10]
        
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
        print(f"✅ Total query time: {time.time() - start_time:.2f}s\n")
        
        # Yield completion event
        yield {
            "type": "done",
            "data": {
                "total_time": time.time() - start_time,
                "answer": full_answer
            }
        }
