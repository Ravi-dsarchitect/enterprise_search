from typing import List
import os
from app.services.ingestion.interfaces import Embedder
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

class OpenAIEmbedder(Embedder):
    def __init__(self, api_key: str):
        self.client = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Remove newlines to avoid embedding issues
        texts = [t.replace("\n", " ") for t in texts]
        return self.client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        return self.client.embed_query(text)

class LocalEmbedder(Embedder):
    """Free local embeddings using sentence-transformers (no API key needed)"""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize local embedder with a sentence-transformers model.
        
        Popular models:
        - all-MiniLM-L6-v2: Fast, lightweight (384 dimensions)
        - all-mpnet-base-v2: Better quality (768 dimensions)
        - multi-qa-MiniLM-L6-cos-v1: Optimized for Q&A (384 dimensions)
        """
        print(f"Loading sentence-transformers model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[OK] Model loaded (dimension: {self.dimension})")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents"""
        # Remove newlines to avoid issues
        texts = [t.replace("\n", " ") for t in texts]
        # Returns numpy arrays, convert to lists
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        text = text.replace("\n", " ")
        embedding = self.model.encode([text], show_progress_bar=False)[0]
        return embedding.tolist()
    
    def get_dimension(self) -> int:
        """Return the embedding dimension size"""
        return self.dimension


class EmbedderFactory:
    """Factory to create embedders based on configuration"""
    
    @staticmethod
    def create_embedder(provider: str = "local", **kwargs) -> Embedder:
        """
        Create an embedder instance based on provider.
        
        Args:
            provider: 'openai' or 'local'
            **kwargs: Provider-specific arguments
                - For openai: api_key
                - For local: model_name (optional)
        """
        if provider == "openai":
            api_key = kwargs.get("api_key")
            if not api_key:
                raise ValueError("OpenAI API key is required for 'openai' provider")
            return OpenAIEmbedder(api_key=api_key)
        
        elif provider == "local":
            model_name = kwargs.get("model_name", "all-MiniLM-L6-v2")
            return LocalEmbedder(model_name=model_name)
        
        else:
            raise ValueError(f"Unknown embedder provider: {provider}. Use 'openai' or 'local'")
    
    @staticmethod
    def get_embedding_dimension(provider: str, **kwargs) -> int:
        """Get the embedding dimension for a provider"""
        if provider == "openai":
            return 1536  # text-embedding-3-small
        elif provider == "local":
            model_name = kwargs.get("model_name", "all-MiniLM-L6-v2")
            # Common models and their dimensions
            dimensions = {
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
                "multi-qa-MiniLM-L6-cos-v1": 384,
            }
            return dimensions.get(model_name, 384)
        else:
            raise ValueError(f"Unknown provider: {provider}")
