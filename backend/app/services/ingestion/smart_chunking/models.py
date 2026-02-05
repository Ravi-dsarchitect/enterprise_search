"""
Data models for Smart Chunking system.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


class ContentStructure(Enum):
    """Detected document structure type - basis for chunker selection."""
    HIERARCHICAL = "hierarchical"   # Nested headings (reports, manuals, policies)
    TABULAR = "tabular"             # Predominantly tables (specs, data sheets)
    QA = "qa"                       # Question-Answer format (FAQs, interviews)
    SLIDE = "slide"                 # Presentation slides (sparse, visual)
    NARRATIVE = "narrative"         # Long-form prose (articles, essays)
    LIST_HEAVY = "list_heavy"       # Bullet/numbered lists
    MIXED = "mixed"                 # Combination of structures
    UNKNOWN = "unknown"             # Cannot determine


@dataclass
class StructuralMetrics:
    """Quantitative metrics about document structure."""
    # Page metrics
    total_pages: int = 1
    total_chars: int = 0
    chars_per_page: float = 0.0

    # Heading analysis
    heading_count: int = 0
    heading_depth: int = 0              # Max nesting level (h1->h2->h3 = 3)
    headings_by_level: Dict[int, int] = field(default_factory=dict)

    # Content distribution
    table_count: int = 0
    table_coverage: float = 0.0         # % of content in tables
    list_count: int = 0
    list_coverage: float = 0.0          # % of content in lists
    qa_pair_count: int = 0              # Detected Q&A patterns

    # Text metrics
    avg_paragraph_length: float = 0.0   # Average paragraph char length
    avg_section_length: float = 0.0     # Average section char length

    # Font/layout metrics (from PDF)
    median_font_size: float = 12.0
    is_landscape: bool = False
    is_sparse: bool = False             # <500 chars/page


@dataclass
class ContentAnalysis:
    """Result of content analysis - guides chunker selection."""
    primary_structure: ContentStructure
    secondary_structure: Optional[ContentStructure] = None
    metrics: StructuralMetrics = field(default_factory=StructuralMetrics)

    # Recommendations
    recommended_chunker: str = "structured"
    chunk_size_multiplier: float = 1.0
    confidence: float = 0.0             # 0.0-1.0, low = consider LLM classifier

    # Hierarchy decision
    supports_hierarchy: bool = False
    hierarchy_depth: int = 0

    # Detected sections (for hierarchical chunking)
    major_sections: List[str] = field(default_factory=list)

    # Debug/logging
    analysis_notes: List[str] = field(default_factory=list)


@dataclass
class HierarchicalChunk:
    """
    Chunk with parent-child linking for hierarchical retrieval.

    Parent chunks contain full section (broader context).
    Child chunks contain details (specific retrieval).
    At retrieval: match child -> expand to parent for complete context.
    """
    text: str
    chunk_id: str

    # Hierarchy linking
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)
    hierarchy_level: int = 0            # 0=root, 1=section, 2=subsection
    is_parent: bool = False
    section_complete: bool = False      # True if chunk contains full section

    # Standard chunk fields (compatible with StructuredChunk)
    section_type: str = "general"
    content_type: str = "text"          # "text", "table", "qa_pair"
    heading: Optional[str] = None
    page_number: Optional[int] = None
    parent_text: Optional[str] = None   # Full section text (capped at 2000)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SmartChunkingResult:
    """Result of smart chunking including analysis metadata."""
    chunks: List[Any]                   # List[StructuredChunk] or List[HierarchicalChunk]
    analysis: ContentAnalysis
    chunker_used: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            "chunk_count": len(self.chunks),
            "chunker_used": self.chunker_used,
            "structure": self.analysis.primary_structure.value,
            "confidence": self.analysis.confidence,
            "supports_hierarchy": self.analysis.supports_hierarchy,
            "notes": self.analysis.analysis_notes,
        }
