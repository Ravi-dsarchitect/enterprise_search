from typing import List
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.rag.generator import LLMFactory

class QueryTransformer:
    def __init__(self):
        self.llm_generator = LLMFactory.create_generator()

    def generate_hyde_doc(self, query: str) -> str:
        """
        Generates a hypothetical document based on the query (HyDE).
        This fictitious document is used for embedding lookup, which often
        matches the embedding space of real documents better than the raw query.
        """
        system_prompt = """You are a helpful assistant. 
        Write a short passage (1 paragraph) that answers the user's question accurately. 
        It doesn't have to be perfect factually, but it should look like a document that *would* contain the answer.
        Do not include any intro (e.g. "Here is a passage"), just text.
        """
        
        user_prompt = f"Question: {query}"
        
        # Accessing underlying LLM
        llm = self.llm_generator.llm
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            print(f"HyDE generation failed: {e}")
            return query # Fallback to original query

    def decompose_query(self, query: str) -> List[str]:
        """
        Breaks a complex query into sub-queries (Multi-Query Expansion).
        """
        system_prompt = """You are an expert at information retrieval.
        Break the following complex user question into 3-5 simple, independent search queries 
        that would help answer the original question.
        Return ONLY a list of queries, one per line.
        """
        
        user_prompt = f"Question: {query}"
        
        llm = self.llm_generator.llm
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = llm.invoke(messages)
            lines = response.content.strip().split('\n')
            # Clean up numbering (1. query -> query)
            cleaned_queries = []
            for line in lines:
                line = line.strip()
                if not line: continue
                # Remove leading numbers/dots
                if line[0].isdigit():
                    parts = line.split('.', 1)
                    if len(parts) > 1:
                        line = parts[1].strip()
                cleaned_queries.append(line)
            
            # Always include original query
            if query not in cleaned_queries:
                cleaned_queries.insert(0, query)
                
            return cleaned_queries
            
        except Exception as e:
            print(f"Query decomposition failed: {e}")
            return [query]
