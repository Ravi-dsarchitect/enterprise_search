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
        context_parts = []
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

            context_parts.append(f"{header}\n{item['text']}\n{'='*60}")

        context_str = "\n\n".join(context_parts)

        system_prompt = """You are a document-grounded assistant for the Life Insurance Corporation (LIC) of India.

RULES:
- Answer using information from the numbered context passages below.
- Each passage is separated by === lines. Treat each passage as a SEPARATE piece of evidence.
- IMPORTANT: Do NOT mix or combine numerical values, amounts, or specific details from different passages. Each passage may describe a different plan, club tier, or benefit category. Only attribute data to the specific entity named in that passage.
- Passage [1] is the most relevant. Give it the highest priority.
- DO NOT use outside knowledge or training data. Only use what is in the passages.
- If the passages contain NO relevant information at all, respond with: "I don't have sufficient information in the provided documents to answer this question."
- Do NOT refuse to answer if the passages contain relevant information, even if partial.

CITATION FORMAT:
- Cite sources using [Source: filename.pdf, Page X] after each factual statement.
- If multiple passages support a fact, cite all of them.

RESPONSE FORMAT:
- Use Markdown: headers, bullet points, and tables where appropriate.
- When comparing plans, use a table and clearly label which plan each detail belongs to.
- Keep answers focused and concise."""

        user_prompt = f"""Context passages (ordered by relevance, separated by === lines):
{context_str}

Question: {query}

Answer using the context passages above. Pay attention to which passage each detail comes from — do not mix numbers from different sections. Cite sources for every fact."""
        
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

class AnswerCritic:
    """
    Lightweight critic that verifies the generator's answer against source passages.
    Uses the smaller model (qwen2.5:7b) to check for data mixing errors.
    Returns the corrected answer if errors are found, or the original if clean.
    """

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            model = settings.LLM_METADATA_MODEL or settings.LLM_MODEL_NAME
            if settings.LLM_PROVIDER.lower() == "ollama":
                from langchain_ollama import ChatOllama
                self._llm = ChatOllama(
                    model=model,
                    temperature=0.0,
                    base_url=settings.OLLAMA_BASE_URL,
                )
            else:
                from langchain_ollama import ChatOllama
                self._llm = ChatOllama(
                    model=model,
                    temperature=0.0,
                    base_url=settings.OLLAMA_BASE_URL,
                )
            print(f"  [AnswerCritic] Using model: {model}")
        return self._llm

    def verify_and_fix(self, answer: str, query: str, context_str: str) -> str:
        """
        Verify the answer against source passages and fix attribution errors.

        Returns the corrected answer, or the original if no errors found.
        """
        prompt = f"""You are a strict fact-checking assistant. You must verify the ANSWER against the source PASSAGES.

PASSAGES:
{context_str}

QUESTION: {query}

ANSWER TO VERIFY:
{answer}

VERIFICATION STEPS (do each one):

STEP 1 — ENTITY CHECK:
The question asks about a specific entity (e.g., a plan name, club tier, category). Identify it.
Then check: does the ANSWER use data that belongs to THAT entity, or does it accidentally use data from a DIFFERENT entity in another passage?
Example error: Question asks about "Corporate Club" but answer uses numbers from "CM Club" or "ZM Club".

STEP 2 — PASSAGE [1] PRIORITY:
Passage [1] is the highest-relevance result. Read it carefully.
Does the ANSWER use the key facts from [1]? If [1] directly answers the question but the ANSWER uses different data from a lower-ranked passage, that is an error.

STEP 3 — CONFLICT CHECK:
If passage [1] and another passage give DIFFERENT values for the same attribute (e.g., different loan amounts for different club tiers), and the answer picks the WRONG one for the entity in the question, that is an error.

DECISION:
If ANY step found an error → Output ONLY the corrected answer with proper citations. Do not explain.
If ALL steps pass → Output ONLY the word: PASS"""

        messages = [HumanMessage(content=prompt)]
        response = self._get_llm().invoke(messages)
        result = response.content.strip()

        if result == "PASS" or result.startswith("PASS"):
            return answer

        return result


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
