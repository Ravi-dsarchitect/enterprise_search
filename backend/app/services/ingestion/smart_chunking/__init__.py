"""
Smart Chunking System - Content-aware document chunking.

Usage:
    from app.services.ingestion.smart_chunking import SmartChunker

    chunker = SmartChunker()
    result = chunker.chunk_smart(parsed_doc, doc_metadata)

    chunks = result.chunks
    analysis = result.analysis
    print(f"Used {result.chunker_used} chunker")
"""

from app.services.ingestion.smart_chunking.models import (
    ContentStructure,
    StructuralMetrics,
    ContentAnalysis,
    HierarchicalChunk,
    SmartChunkingResult,
)
from app.services.ingestion.smart_chunking.analyzer import ContentAnalyzer
from app.services.ingestion.smart_chunking.factory import (
    ChunkingFactory,
    get_chunking_factory,
)
from app.services.ingestion.smart_chunking.smart_chunker import SmartChunker
from app.services.ingestion.smart_chunking.hierarchical_chunker import HierarchicalChunker
from app.services.ingestion.smart_chunking.qa_chunker import QAChunker
from app.services.ingestion.smart_chunking.table_aware_chunker import TableAwareChunker
from app.services.ingestion.smart_chunking.hybrid_section_chunker import HybridSectionChunker

__all__ = [
    # Main entry point
    "SmartChunker",
    # Data models
    "ContentStructure",
    "StructuralMetrics",
    "ContentAnalysis",
    "HierarchicalChunk",
    "SmartChunkingResult",
    # Components
    "ContentAnalyzer",
    "ChunkingFactory",
    "get_chunking_factory",
    # Specialized chunkers
    "HierarchicalChunker",
    "QAChunker",
    "TableAwareChunker",
    "HybridSectionChunker",
]
