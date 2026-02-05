"""
Content Analyzer - Detects document structure for chunker selection.

Uses GENERIC structural patterns (not domain-specific):
- Heading frequency and depth
- Table coverage
- Q&A patterns
- Text density metrics
- Layout characteristics
"""

import re
import statistics
from typing import List, Optional, Tuple
from collections import Counter

from app.services.ingestion.interfaces import ParsedDocument, ParsedBlock
from app.services.ingestion.smart_chunking.models import (
    ContentAnalysis,
    ContentStructure,
    StructuralMetrics,
)


class ContentAnalyzer:
    """
    Analyzes ParsedDocument to detect content structure and recommend chunking strategy.
    """

    # Generic Q&A detection patterns
    QA_PATTERNS = [
        r"^\s*(?:Q[\s.:]*\d*|Question[\s.:]*\d*)[:.\s]+.+\?\s*$",  # Q: or Question:
        r"^\s*(?:A[\s.:]*\d*|Answer[\s.:]*\d*)[:.\s]+",            # A: or Answer:
        r"^\s*\d+\.\s+.+\?\s*$",                                   # Numbered questions
        r"^\s*[-•]\s+.+\?\s*$",                                    # Bulleted questions
        r"(?i)^\s*(?:FAQ|Frequently Asked)",                       # FAQ markers
    ]

    # List detection patterns
    LIST_PATTERNS = [
        r"^\s*[-•*]\s+",
        r"^\s*\d+[.)]\s+",
        r"^\s*[a-z][.)]\s+",
        r"^\s*[ivx]+[.)]\s+",
    ]

    # Slide detection thresholds
    SLIDE_MIN_PAGES = 3
    SLIDE_MAX_CHARS_PER_PAGE = 500
    SLIDE_MIN_FONT_SIZE = 16
    SLIDE_MIN_ASPECT_RATIO = 1.0  # Landscape

    # Confidence thresholds
    LOW_CONFIDENCE_THRESHOLD = 0.5

    def analyze(self, doc: ParsedDocument) -> ContentAnalysis:
        """
        Analyze document and return content analysis with recommended chunker.

        Args:
            doc: ParsedDocument with blocks, tables, and metadata

        Returns:
            ContentAnalysis with structure type, metrics, and recommendations
        """
        # Step 1: Compute structural metrics
        metrics = self._compute_metrics(doc)

        # Step 2: Detect primary structure
        primary_structure, confidence = self._detect_primary_structure(doc, metrics)

        # Step 3: Check for secondary patterns
        secondary_structure = self._detect_secondary_structure(metrics, primary_structure)

        # Step 4: Determine hierarchy support
        supports_hierarchy = self._check_hierarchy_support(doc, metrics)
        hierarchy_depth = metrics.heading_depth if supports_hierarchy else 0

        # Step 5: Extract major sections
        major_sections = self._extract_major_sections(doc.blocks)

        # Step 6: Set recommendations
        recommended_chunker, chunk_multiplier = self._recommend_chunker(
            primary_structure, secondary_structure, metrics, supports_hierarchy
        )

        # Step 7: Generate analysis notes
        notes = self._generate_notes(metrics, primary_structure, supports_hierarchy)

        return ContentAnalysis(
            primary_structure=primary_structure,
            secondary_structure=secondary_structure,
            metrics=metrics,
            recommended_chunker=recommended_chunker,
            chunk_size_multiplier=chunk_multiplier,
            confidence=confidence,
            supports_hierarchy=supports_hierarchy,
            hierarchy_depth=hierarchy_depth,
            major_sections=major_sections,
            analysis_notes=notes,
        )

    def _compute_metrics(self, doc: ParsedDocument) -> StructuralMetrics:
        """Compute quantitative structural metrics from parsed document."""
        metrics = StructuralMetrics()

        metrics.total_pages = doc.total_pages or 1
        metrics.total_chars = len(doc.full_text)
        metrics.chars_per_page = metrics.total_chars / metrics.total_pages

        # Heading analysis
        headings_by_level = Counter()
        for block in doc.blocks:
            if block.heading_level > 0:
                headings_by_level[block.heading_level] += 1

        metrics.heading_count = sum(headings_by_level.values())
        metrics.heading_depth = max(headings_by_level.keys()) if headings_by_level else 0
        metrics.headings_by_level = dict(headings_by_level)

        # Table metrics
        metrics.table_count = len(doc.tables)
        if metrics.total_chars > 0 and doc.tables:
            table_chars = sum(len(t.markdown or "") for t in doc.tables)
            metrics.table_coverage = table_chars / metrics.total_chars

        # List detection
        list_chars = 0
        for block in doc.blocks:
            if self._is_list_block(block):
                metrics.list_count += 1
                list_chars += len(block.content)

        if metrics.total_chars > 0:
            metrics.list_coverage = list_chars / metrics.total_chars

        # Q&A detection
        metrics.qa_pair_count = self._count_qa_pairs(doc)

        # Paragraph/section stats
        paragraphs = [
            b.content for b in doc.blocks
            if b.heading_level == 0 and len(b.content) > 50
        ]
        if paragraphs:
            metrics.avg_paragraph_length = statistics.mean(len(p) for p in paragraphs)

        if metrics.heading_count > 0:
            metrics.avg_section_length = metrics.total_chars / metrics.heading_count

        # Font/layout metrics
        metrics.median_font_size = doc.median_font_size or 12.0

        if doc.page_dimensions and len(doc.page_dimensions) == 2:
            w, h = doc.page_dimensions
            metrics.is_landscape = w > h if h > 0 else False

        metrics.is_sparse = metrics.chars_per_page < self.SLIDE_MAX_CHARS_PER_PAGE

        return metrics

    def _detect_primary_structure(
        self,
        doc: ParsedDocument,
        metrics: StructuralMetrics,
    ) -> Tuple[ContentStructure, float]:
        """Detect the primary content structure type with confidence."""

        # Check for slide deck (highest priority - very distinct)
        if self._is_slide_deck(doc, metrics):
            return ContentStructure.SLIDE, 0.95

        # Check for Q&A format
        if metrics.qa_pair_count >= 3:
            confidence = min(0.95, 0.6 + metrics.qa_pair_count * 0.05)
            return ContentStructure.QA, confidence

        # Check for tabular (table-heavy documents)
        if metrics.table_coverage > 0.4:
            confidence = 0.85 + min(0.1, (metrics.table_coverage - 0.4) * 0.5)
            return ContentStructure.TABULAR, confidence

        # Check for hierarchical (heading-based structure)
        if metrics.heading_count >= 3 and metrics.heading_depth >= 2:
            confidence = min(0.95, 0.7 + metrics.heading_count * 0.02)
            return ContentStructure.HIERARCHICAL, confidence

        # Check for list-heavy
        if metrics.list_coverage > 0.3:
            return ContentStructure.LIST_HEAVY, 0.75

        # Check for narrative (long paragraphs, few headings)
        if metrics.avg_paragraph_length > 400 and metrics.heading_count < 3:
            return ContentStructure.NARRATIVE, 0.75

        # Mixed or unknown
        if metrics.heading_count > 0 or metrics.table_count > 0:
            return ContentStructure.MIXED, 0.55

        return ContentStructure.UNKNOWN, 0.3

    def _detect_secondary_structure(
        self,
        metrics: StructuralMetrics,
        primary: ContentStructure,
    ) -> Optional[ContentStructure]:
        """Detect secondary structure patterns."""
        if primary == ContentStructure.MIXED:
            if metrics.table_coverage > 0.15:
                return ContentStructure.TABULAR
            if metrics.heading_depth >= 2:
                return ContentStructure.HIERARCHICAL

        # Hierarchical doc with significant tables
        if primary == ContentStructure.HIERARCHICAL and metrics.table_coverage > 0.2:
            return ContentStructure.TABULAR

        return None

    def _is_slide_deck(self, doc: ParsedDocument, metrics: StructuralMetrics) -> bool:
        """
        Detect if document is a slide deck using 3-signal heuristic.
        Reuses logic from StructuredChunker._is_slide_pdf()
        """
        if doc.total_pages < self.SLIDE_MIN_PAGES:
            return False

        signals = 0

        # Signal 1: Landscape aspect ratio
        if metrics.is_landscape:
            signals += 1

        # Signal 2: Sparse text per page
        if metrics.is_sparse:
            signals += 1

        # Signal 3: Large median font size
        if metrics.median_font_size > self.SLIDE_MIN_FONT_SIZE:
            signals += 1

        return signals >= 2

    def _is_list_block(self, block: ParsedBlock) -> bool:
        """Check if a block contains list content."""
        content = block.content.strip()
        return any(
            re.match(pattern, content, re.MULTILINE)
            for pattern in self.LIST_PATTERNS
        )

    def _count_qa_pairs(self, doc: ParsedDocument) -> int:
        """Count question-answer pairs in document."""
        qa_count = 0
        text = doc.full_text

        for pattern in self.QA_PATTERNS:
            qa_count += len(re.findall(pattern, text, re.MULTILINE | re.IGNORECASE))

        # Divide by ~2 since we count both Q and A patterns
        return qa_count // 2

    def _check_hierarchy_support(
        self,
        doc: ParsedDocument,
        metrics: StructuralMetrics,
    ) -> bool:
        """Determine if document benefits from parent-child chunking."""
        # Need multiple heading levels
        if metrics.heading_depth < 2:
            return False

        # Need reasonable number of headings
        if metrics.heading_count < 3:
            return False

        # Sections should have enough content to warrant splitting
        if metrics.avg_section_length < 500:
            return False

        return True

    def _extract_major_sections(self, blocks: List[ParsedBlock]) -> List[str]:
        """Extract top-level section headings."""
        sections = []
        for block in blocks:
            if block.heading_level in (1, 2):
                heading = block.content.strip()[:80]
                if heading and heading not in sections:
                    sections.append(heading)
        return sections[:20]  # Limit to first 20

    def _recommend_chunker(
        self,
        primary: ContentStructure,
        secondary: Optional[ContentStructure],
        metrics: StructuralMetrics,
        supports_hierarchy: bool,
    ) -> Tuple[str, float]:
        """Recommend chunker and size multiplier based on analysis."""

        # Slide deck -> per-page (StructuredChunker handles this)
        if primary == ContentStructure.SLIDE:
            return "structured", 2.0

        # Q&A format -> specialized QA chunker
        if primary == ContentStructure.QA:
            return "qa", 1.5

        # HYBRID: Table-heavy BUT also has hierarchical structure
        # Use hybrid_section to apply different strategies per section
        if primary == ContentStructure.TABULAR and supports_hierarchy:
            return "hybrid_section", 1.0

        # HYBRID: Hierarchical with significant tables (secondary=TABULAR)
        # Different sections may need table-aware vs hierarchical chunking
        if primary == ContentStructure.HIERARCHICAL and secondary == ContentStructure.TABULAR:
            return "hybrid_section", 1.0

        # Table-heavy without hierarchy -> table-aware chunker
        if primary == ContentStructure.TABULAR:
            return "table_aware", 1.5

        # Hierarchical with deep structure -> hierarchical chunker
        if primary == ContentStructure.HIERARCHICAL and supports_hierarchy:
            return "hierarchical", 1.0

        # Hierarchical without deep structure -> structured
        if primary == ContentStructure.HIERARCHICAL:
            return "structured", 1.0

        # Narrative -> semantic chunker
        if primary == ContentStructure.NARRATIVE:
            return "semantic", 1.0

        # List-heavy -> structured
        if primary == ContentStructure.LIST_HEAVY:
            return "structured", 1.0

        # Mixed -> hybrid section for zone-based chunking
        if primary == ContentStructure.MIXED:
            return "hybrid_section", 1.0

        # Fallback
        return "structured", 1.0

    def _generate_notes(
        self,
        metrics: StructuralMetrics,
        primary: ContentStructure,
        supports_hierarchy: bool,
    ) -> List[str]:
        """Generate human-readable analysis notes."""
        notes = []

        notes.append(f"Structure: {primary.value}")
        notes.append(f"Pages: {metrics.total_pages}, Headings: {metrics.heading_count}")

        if metrics.table_count > 0:
            notes.append(f"Tables: {metrics.table_count} ({metrics.table_coverage:.0%} coverage)")

        if metrics.qa_pair_count > 0:
            notes.append(f"Q&A pairs: {metrics.qa_pair_count}")

        if metrics.heading_depth > 0:
            notes.append(f"Heading depth: {metrics.heading_depth}")

        if supports_hierarchy:
            notes.append("Supports parent-child chunking")

        return notes
