from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.core.config import settings

class LLMGenerator(ABC):
    @abstractmethod
    def generate_answer(self, query: str, context: List[Dict[str, Any]], conversation_history: list = None) -> str:
        """Generate an answer given the query and retrieved context."""
        pass

class BaseLangChainGenerator(LLMGenerator):
    """Base class for LangChain-based generators to share prompt logic."""
    def __init__(self, llm):
        self.llm = llm

    def _build_messages(self, query: str, context: List[Dict[str, Any]], conversation_history: list = None):
        """Build messages for both streaming and non-streaming."""
        context_str = "\n\n".join([
            f"[{i+1}] Source: {item['source']}\n{item['text']}"
            for i, item in enumerate(context)
        ])

        system_prompt = """You are a document-grounded assistant for the Life Insurance Corporation (LIC) of India.

STRICT RULES:
- Answer ONLY using information explicitly stated in the numbered context passages below.
- DO NOT use any outside knowledge, training data, or general information about LIC or insurance.
- If the context does not contain enough information to answer, respond EXACTLY with: "I don't have sufficient information in the provided documents to answer this question."
- NEVER guess, infer beyond what is stated, or fill gaps with general knowledge.

CITATION FORMAT:
- Every factual statement MUST cite the source using [Source: filename.pdf].
- If multiple sources support a fact, cite all of them.

RESPONSE FORMAT:
- Use Markdown: headers, bullet points, and tables where appropriate.
- When comparing plans, use a table and clearly label which plan each detail belongs to.
- Keep answers focused and concise — do not add disclaimers or boilerplate beyond what the context states."""

        user_prompt = f"""Context passages:
{context_str}

Question: {query}

Answer using ONLY the context passages above. Cite sources for every fact."""
        
        messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        
        # Add current query
        messages.append(HumanMessage(content=user_prompt))
        
        return messages

    def generate_answer(self, query: str, context: List[Dict[str, Any]], conversation_history: list = None) -> str:
        messages = self._build_messages(query, context, conversation_history)
        response = self.llm.invoke(messages)
        return response.content
    
    async def generate_answer_stream(self, query: str, context: List[Dict[str, Any]], conversation_history: list = None):
        """Stream the answer generation token by token."""
        messages = self._build_messages(query, context, conversation_history)
        
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content

class OpenAIGenerator(BaseLangChainGenerator):
    def __init__(self):
        from langchain_openai import ChatOpenAI
        model = settings.LLM_MODEL_NAME or "gpt-4o"
        llm = ChatOpenAI(
            model=model,
            temperature=0.1,
            openai_api_key=settings.OPENAI_API_KEY
        )
        super().__init__(llm)

class GroqGenerator(BaseLangChainGenerator):
    """Groq inference platform - fast and free tier available"""
    def __init__(self):
        from langchain_groq import ChatGroq
        model = settings.LLM_MODEL_NAME or "llama-3.3-70b-versatile"
        llm = ChatGroq(
            model=model,
            temperature=0.1,
            groq_api_key=settings.GROQ_API_KEY
        )
        super().__init__(llm)

class OllamaGenerator(BaseLangChainGenerator):
    def __init__(self):
        from langchain_ollama import ChatOllama
        model = settings.LLM_MODEL_NAME or "llama3"
        llm = ChatOllama(
            model=model,
            temperature=0.1,
            base_url=settings.OLLAMA_BASE_URL
        )
        super().__init__(llm)

class BedrockGenerator(BaseLangChainGenerator):
    def __init__(self):
        from langchain_aws import ChatBedrock
        model = settings.LLM_MODEL_NAME or "anthropic.claude-v2"
        # Credentials are Auto-resolved from env vars AWS_ACCESS_KEY_ID etc.
        # or passed explicitly if needed.
        llm = ChatBedrock(
            model_id=model,
            model_kwargs={"temperature": 0.1},
            region_name=settings.AWS_REGION
        )
        super().__init__(llm)

class LLMFactory:
    @staticmethod
    def create_generator() -> LLMGenerator:
        provider = settings.LLM_PROVIDER.lower()
        
        if provider == "openai":
            return OpenAIGenerator()
        elif provider == "groq":
            return GroqGenerator()
        elif provider == "ollama":
            return OllamaGenerator()
        elif provider == "bedrock":
            return BedrockGenerator()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
