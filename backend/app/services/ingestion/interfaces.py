from abc import ABC, abstractmethod
from typing import List, Dict, Any

class DocumentParser(ABC):
    """Abstract base class for document parsers."""
    
    @abstractmethod
    def parse(self, file_path: str) -> str:
        """Parses a file and returns the extracted text."""
        pass

class Chunker(ABC):
    """Abstract base class for text chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Splits text into chunks."""
        pass

class Embedder(ABC):
    """Abstract base class for embedding generation strategies."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generates embedding for a single query text."""
        pass
