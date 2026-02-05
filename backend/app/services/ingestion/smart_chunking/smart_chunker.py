"""
SmartChunker - Orchestrator that analyzes content and applies optimal chunking.

Usage:
    smart_chunker = SmartChunker()
    result = smart_chunker.chunk_smart(parsed_doc, doc_metadata)
    chunks = result.chunks
    analysis = result.analysis
"""

from typing import List, Dict, Optional

from app.services.ingestion.interfaces import (
    Chunker,
    ParsedDocument,
    ParsedBlock,
    StructuredChunk,
)
from app.services.ingestion.smart_chunking.models import (
    ContentAnalysis,
    SmartChunkingResult,
)
from app.services.ingestion.smart_chunking.analyzer import ContentAnalyzer
from app.services.ingestion.smart_chunking.factory import get_chunking_factory
from app.services.ingestion.chunkers import classify_section


class SmartChunker(Chunker):
    """
    Intelligent chunker that analyzes content first, then routes to appropriate strategy.

    Flow:
    1. ContentAnalyzer detects document structure
    2. ChunkingFactory selects appropriate chunker
    3. Chunker produces optimized chunks
    4. Result includes analysis metadata
    """

    def __init__(
        self,
        use_llm_classifier: bool = False,
        llm_confidence_threshold: float = 0.5,
        verbose: bool = True,
    ):
        self.analyzer = ContentAnalyzer()
        self.factory = get_chunking_factory()
        self.use_llm_classifier = use_llm_classifier
        self.llm_confidence_threshold = llm_confidence_threshold
        self.verbose = verbose

        self._llm_classifier = None

    def chunk(self, text: str) -> List[str]:
        """
        Basic interface - analyze and chunk plain text.

        For best results, use chunk_smart() with ParsedDocument.
        """
        # Create minimal ParsedDocument
        blocks = [ParsedBlock(page_number=1, block_type="text", content=text)]
        doc = ParsedDocument(
            filename="plain_text",
            total_pages=1,
            blocks=blocks,
            full_text=text,
        )

        result = self.chunk_smart(doc)
        return [c.text for c in result.chunks]

    def chunk_structured(
        self,
        doc: ParsedDocument,
        doc_metadata: Dict = None,
    ) -> List[StructuredChunk]:
        """Interface compatible with existing chunkers."""
        result = self.chunk_smart(doc, doc_metadata)
        return result.chunks

    def chunk_smart(
        self,
        doc: ParsedDocument,
        doc_metadata: Dict = None,
    ) -> SmartChunkingResult:
        """
        Analyze document and chunk with the most appropriate strategy.

        Args:
            doc: ParsedDocument with structural information
            doc_metadata: Optional document-level metadata

        Returns:
            SmartChunkingResult with chunks, analysis, and chunker used
        """
        doc_metadata = doc_metadata or {}

        # Step 1: Analyze content
        analysis = self.analyzer.analyze(doc)

        if self.verbose:
            self._log_analysis(analysis)

        # Step 2: Optional LLM refinement for low confidence
        if (
            self.use_llm_classifier
            and analysis.confidence < self.llm_confidence_threshold
        ):
            analysis = self._refine_with_llm(doc, analysis)

        # Step 3: Get appropriate chunker
        chunker = self.factory.get_chunker(analysis)
        chunker_name = analysis.recommended_chunker

        # Step 4: Chunk the document
        if hasattr(chunker, "chunk_structured"):
            chunks = chunker.chunk_structured(doc, doc_metadata)
        else:
            chunk_texts = chunker.chunk(doc.full_text)
            chunks = [
                StructuredChunk(
                    text=t,
                    section_type=classify_section(t[:500]),
                )
                for t in chunk_texts
            ]

        # Step 5: Add analysis metadata to chunks
        for chunk in chunks:
            if not hasattr(chunk, "metadata") or chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["content_analysis"] = {
                "primary_structure": analysis.primary_structure.value,
                "confidence": analysis.confidence,
                "chunker_used": chunker_name,
            }

        if self.verbose:
            self._log_result(chunks, analysis)

        return SmartChunkingResult(
            chunks=chunks,
            analysis=analysis,
            chunker_used=chunker_name,
        )

    def _log_analysis(self, analysis: ContentAnalysis):
        """Log analysis summary."""
        m = analysis.metrics
        print(f"  📊 Content Analysis:")
        print(f"     Structure: {analysis.primary_structure.value} (confidence: {analysis.confidence:.0%})")
        print(f"     Headings: {m.heading_count} (depth {m.heading_depth})")
        if m.table_count > 0:
            print(f"     Tables: {m.table_count} ({m.table_coverage:.0%} coverage)")
        if m.qa_pair_count > 0:
            print(f"     Q&A pairs: {m.qa_pair_count}")
        print(f"     Chunker: {analysis.recommended_chunker}")
        if analysis.supports_hierarchy:
            print(f"     Mode: parent-child hierarchical")

    def _log_result(self, chunks: List, analysis: ContentAnalysis):
        """Log chunking result."""
        # Count parent/child if hierarchical
        parent_count = sum(
            1 for c in chunks
            if hasattr(c, "metadata") and c.metadata and c.metadata.get("is_parent_chunk")
        )

        if parent_count > 0:
            child_count = len(chunks) - parent_count
            print(f"     Created {parent_count} parent + {child_count} child chunks")
        else:
            print(f"     Created {len(chunks)} chunks")

    def _refine_with_llm(
        self,
        doc: ParsedDocument,
        initial_analysis: ContentAnalysis,
    ) -> ContentAnalysis:
        """Use LLM to refine analysis when heuristics are uncertain."""
        if self._llm_classifier is None:
            try:
                from app.services.ingestion.smart_chunking.llm_classifier import (
                    LLMContentClassifier,
                )

                self._llm_classifier = LLMContentClassifier()
            except Exception as e:
                print(f"  [SmartChunker] LLM classifier unavailable: {e}")
                return initial_analysis

        try:
            refined = self._llm_classifier.classify(doc, initial_analysis)
            if self.verbose:
                print(f"  [SmartChunker] LLM refined: {refined.primary_structure.value}")
            return refined
        except Exception as e:
            print(f"  [SmartChunker] LLM classification failed: {e}")
            return initial_analysis
