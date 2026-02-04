from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


# --- Parsed Document Data Classes ---

@dataclass
class ParsedSpan:
    """A span of text with formatting information from the PDF."""
    text: str
    font_size: float
    is_bold: bool
    is_italic: bool = False
    bbox: tuple = ()  # (x0, y0, x1, y1)


@dataclass
class ParsedBlock:
    """A block of content from a parsed document."""
    page_number: int
    block_type: str  # "text", "table", "image"
    content: str
    heading_level: int = 0  # 0=body, 1=h1, 2=h2, 3=h3
    spans: List[ParsedSpan] = field(default_factory=list)
    bbox: tuple = ()


@dataclass
class ParsedTable:
    """A table extracted from a document."""
    page_number: int
    rows: List[List[str]]
    headers: List[str] = field(default_factory=list)
    markdown: str = ""
    caption: str = ""
    bbox: tuple = ()


@dataclass
class ParsedDocument:
    """Complete parsed document with structural information."""
    filename: str
    total_pages: int
    blocks: List[ParsedBlock] = field(default_factory=list)
    tables: List[ParsedTable] = field(default_factory=list)
    full_text: str = ""
    headings: List[str] = field(default_factory=list)
    page_dimensions: tuple = ()       # (width, height) of first page in points
    median_font_size: float = 0.0     # Median font size across all spans


@dataclass
class StructuredChunk:
    """A chunk with structural metadata from the chunker."""
    text: str
    section_type: str = "general"
    content_type: str = "text"  # "text", "table", "list"
    heading: Optional[str] = None
    page_number: Optional[int] = None
    parent_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# --- Abstract Base Classes ---

class DocumentParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(self, file_path: str) -> str:
        """Parses a file and returns the extracted text."""
        pass

    def parse_structured(self, file_path: str) -> ParsedDocument:
        """Parses a file and returns a structured document. Override for structured parsers."""
        text = self.parse(file_path)
        return ParsedDocument(
            filename=file_path,
            total_pages=1,
            full_text=text,
            blocks=[ParsedBlock(page_number=1, block_type="text", content=text)],
        )


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
