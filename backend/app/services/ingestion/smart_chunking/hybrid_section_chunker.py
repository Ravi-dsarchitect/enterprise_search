"""
HybridSectionChunker - Applies different chunking strategies per section.

Splits document into zones and applies the best chunker for each:
- Benefits sections → HierarchicalChunker (parent-child)
- Tables → TableAwareChunker (preserve structure)
- FAQ sections → QAChunker (keep Q&A pairs)
- Other sections → StructuredChunker
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from app.services.ingestion.interfaces import (
    Chunker,
    ParsedDocument,
    ParsedBlock,
    ParsedTable,
    StructuredChunk,
)
from app.services.ingestion.chunkers import classify_section, split_into_sentences
from app.core.config import settings


@dataclass
class DocumentSection:
    """A section of the document with its content and classification."""
    heading: Optional[str]
    heading_level: int
    blocks: List[ParsedBlock]
    tables: List[ParsedTable]
    start_page: int
    end_page: int

    # Classification
    section_type: str = "general"           # benefits, eligibility, premium, faq, etc.
    content_pattern: str = "narrative"       # hierarchical, tabular, qa, narrative

    @property
    def text(self) -> str:
        return "\n".join(b.content for b in self.blocks)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def has_tables(self) -> bool:
        return len(self.tables) > 0

    @property
    def table_text(self) -> str:
        return "\n\n".join(t.markdown or "" for t in self.tables)


class HybridSectionChunker(Chunker):
    """
    Chunks document by splitting into sections and applying
    the best chunker strategy to each section.

    Strategy:
    1. Split document into sections by major headings
    2. Classify each section (benefits, tables, faq, narrative)
    3. Apply appropriate chunker per section:
       - benefits → HierarchicalChunker (parent-child for complete context)
       - tables → TableAwareChunker (preserve table structure)
       - faq → QAChunker (keep Q&A pairs together)
       - other → StructuredChunker
    4. Merge all chunks with section context
    """

    # Section patterns that benefit from hierarchical chunking
    HIERARCHICAL_SECTIONS = {
        "benefits", "death_benefit", "maturity_benefit", "survival_benefit",
        "bonus", "rider", "loan", "surrender", "exclusions",
    }

    # FAQ detection patterns
    FAQ_PATTERNS = [
        r"(?i)^\s*(?:Q[\s.:]*\d*|Question)[:.\s]+",
        r"(?i)^\s*\d+\.\s+.+\?\s*$",
        r"(?i)faq|frequently\s+asked",
    ]

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        verbose: bool = True,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.verbose = verbose

        # Lazy-load specialized chunkers
        self._hierarchical_chunker = None
        self._qa_chunker = None
        self._table_chunker = None
        self._structured_chunker = None

    def _get_hierarchical_chunker(self):
        if self._hierarchical_chunker is None:
            from app.services.ingestion.smart_chunking.hierarchical_chunker import HierarchicalChunker
            self._hierarchical_chunker = HierarchicalChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        return self._hierarchical_chunker

    def _get_qa_chunker(self):
        if self._qa_chunker is None:
            from app.services.ingestion.smart_chunking.qa_chunker import QAChunker
            self._qa_chunker = QAChunker(chunk_size=self.chunk_size)
        return self._qa_chunker

    def _get_table_chunker(self):
        if self._table_chunker is None:
            from app.services.ingestion.smart_chunking.table_aware_chunker import TableAwareChunker
            self._table_chunker = TableAwareChunker(chunk_size=self.chunk_size)
        return self._table_chunker

    def _get_structured_chunker(self):
        if self._structured_chunker is None:
            from app.services.ingestion.chunkers import StructuredChunker
            self._structured_chunker = StructuredChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        return self._structured_chunker

    def chunk(self, text: str) -> List[str]:
        """Basic interface - fallback to structured chunking."""
        return self._get_structured_chunker().chunk(text)

    def chunk_structured(
        self,
        doc: ParsedDocument,
        doc_metadata: Dict = None,
    ) -> List[StructuredChunk]:
        """
        Chunk document using zone-based hybrid approach.
        """
        doc_metadata = doc_metadata or {}

        # Step 1: Split into sections
        sections = self._split_into_sections(doc)

        if self.verbose:
            print(f"  📑 Split into {len(sections)} sections")

        # Step 2: Classify each section
        for section in sections:
            self._classify_section(section)

        if self.verbose:
            self._log_section_classification(sections)

        # Step 3: Chunk each section with appropriate strategy
        all_chunks = []
        chunker_stats = defaultdict(int)

        for section in sections:
            section_chunks = self._chunk_section(section, doc_metadata)

            # Add section context to each chunk
            for chunk in section_chunks:
                chunk.metadata["zone_section"] = section.heading or section.section_type
                chunk.metadata["zone_chunker"] = section.content_pattern

            all_chunks.extend(section_chunks)
            chunker_stats[section.content_pattern] += len(section_chunks)

        if self.verbose:
            print(f"  📊 Chunks by strategy: {dict(chunker_stats)}")
            print(f"     Total: {len(all_chunks)} chunks")

        return all_chunks

    def _split_into_sections(self, doc: ParsedDocument) -> List[DocumentSection]:
        """Split document into sections based on major headings."""
        sections = []

        # Group blocks by section (using heading level 1-2 as boundaries)
        current_section = DocumentSection(
            heading=None,
            heading_level=0,
            blocks=[],
            tables=[],
            start_page=1,
            end_page=1,
        )

        for block in doc.blocks:
            if block.heading_level in (1, 2):
                # Save current section if it has content
                if current_section.blocks:
                    current_section.end_page = current_section.blocks[-1].page_number
                    sections.append(current_section)

                # Start new section
                current_section = DocumentSection(
                    heading=block.content.strip(),
                    heading_level=block.heading_level,
                    blocks=[],
                    tables=[],
                    start_page=block.page_number,
                    end_page=block.page_number,
                )
            else:
                current_section.blocks.append(block)

        # Don't forget last section
        if current_section.blocks:
            current_section.end_page = current_section.blocks[-1].page_number
            sections.append(current_section)

        # Assign tables to sections based on page numbers
        for table in doc.tables:
            for section in sections:
                if section.start_page <= table.page_number <= section.end_page:
                    section.tables.append(table)
                    break

        return sections

    def _classify_section(self, section: DocumentSection):
        """Classify section type and content pattern."""
        # Determine section type from heading
        if section.heading:
            section.section_type = classify_section(section.heading)
        else:
            # Classify from content
            section.section_type = classify_section(section.text[:500])

        # Determine content pattern (which chunker to use)
        section.content_pattern = self._detect_content_pattern(section)

    def _detect_content_pattern(self, section: DocumentSection) -> str:
        """Detect the best chunking strategy for this section."""
        text = section.text
        section_type = section.section_type

        # Check if it's a benefits-type section that needs hierarchical
        if section_type in self.HIERARCHICAL_SECTIONS:
            # Only use hierarchical if section has enough content
            if section.char_count > settings.HIERARCHICAL_MIN_SECTION_SIZE:
                return "hierarchical"

        # Check for FAQ pattern
        faq_matches = sum(
            len(re.findall(p, text, re.MULTILINE))
            for p in self.FAQ_PATTERNS
        )
        if faq_matches >= 3:
            return "qa"

        # Check if section is primarily tables
        if section.has_tables:
            table_chars = len(section.table_text)
            text_chars = section.char_count
            total = table_chars + text_chars

            if total > 0 and table_chars / total > 0.5:
                return "tabular"

        # Check for subsections (heading level 3+) → hierarchical
        subsection_count = sum(
            1 for b in section.blocks if b.heading_level >= 3
        )
        if subsection_count >= 2 and section.char_count > 800:
            return "hierarchical"

        # Default to structured
        return "narrative"

    def _chunk_section(
        self,
        section: DocumentSection,
        doc_metadata: Dict,
    ) -> List[StructuredChunk]:
        """Chunk a section using the appropriate strategy."""
        chunks = []
        pattern = section.content_pattern

        if pattern == "hierarchical":
            chunks = self._chunk_hierarchical(section, doc_metadata)
        elif pattern == "qa":
            chunks = self._chunk_qa(section, doc_metadata)
        elif pattern == "tabular":
            chunks = self._chunk_tables(section, doc_metadata)
        else:
            chunks = self._chunk_narrative(section, doc_metadata)

        return chunks

    def _chunk_hierarchical(
        self,
        section: DocumentSection,
        doc_metadata: Dict,
    ) -> List[StructuredChunk]:
        """Chunk section hierarchically with parent-child relationships."""
        from app.services.ingestion.interfaces import ParsedDocument

        # Create mini-document for this section
        mini_doc = ParsedDocument(
            filename=f"section_{section.heading or 'unnamed'}",
            total_pages=section.end_page - section.start_page + 1,
            blocks=section.blocks,
            tables=section.tables,
            full_text=section.text,
            headings=[section.heading] if section.heading else [],
        )

        # Add section heading as context
        meta = dict(doc_metadata)
        if section.heading:
            meta["section_heading"] = section.heading

        return self._get_hierarchical_chunker().chunk_structured(mini_doc, meta)

    def _chunk_qa(
        self,
        section: DocumentSection,
        doc_metadata: Dict,
    ) -> List[StructuredChunk]:
        """Chunk section preserving Q&A pairs."""
        from app.services.ingestion.interfaces import ParsedDocument

        mini_doc = ParsedDocument(
            filename=f"section_{section.heading or 'faq'}",
            total_pages=1,
            blocks=section.blocks,
            tables=[],
            full_text=section.text,
            headings=[section.heading] if section.heading else [],
        )

        chunks = self._get_qa_chunker().chunk_structured(mini_doc, doc_metadata)

        # Update section info
        for chunk in chunks:
            if section.heading:
                chunk.heading = chunk.heading or section.heading

        return chunks

    def _chunk_tables(
        self,
        section: DocumentSection,
        doc_metadata: Dict,
    ) -> List[StructuredChunk]:
        """Chunk section with table-aware strategy."""
        from app.services.ingestion.interfaces import ParsedDocument

        mini_doc = ParsedDocument(
            filename=f"section_{section.heading or 'tables'}",
            total_pages=section.end_page - section.start_page + 1,
            blocks=section.blocks,
            tables=section.tables,
            full_text=section.text + "\n\n" + section.table_text,
            headings=[section.heading] if section.heading else [],
        )

        chunks = self._get_table_chunker().chunk_structured(mini_doc, doc_metadata)

        # Add section context
        for chunk in chunks:
            if section.heading and not chunk.heading:
                chunk.heading = section.heading

        return chunks

    def _chunk_narrative(
        self,
        section: DocumentSection,
        doc_metadata: Dict,
    ) -> List[StructuredChunk]:
        """Chunk section with standard structured approach."""
        from app.services.ingestion.interfaces import ParsedDocument

        mini_doc = ParsedDocument(
            filename=f"section_{section.heading or 'content'}",
            total_pages=section.end_page - section.start_page + 1,
            blocks=section.blocks,
            tables=section.tables,
            full_text=section.text,
            headings=[section.heading] if section.heading else [],
        )

        chunks = self._get_structured_chunker().chunk_structured(mini_doc, doc_metadata)

        # Add section context
        for chunk in chunks:
            if section.heading and not chunk.heading:
                chunk.heading = section.heading

        return chunks

    def _log_section_classification(self, sections: List[DocumentSection]):
        """Log section classifications."""
        print(f"  📋 Section breakdown:")
        for i, s in enumerate(sections[:10]):  # Show first 10
            heading = s.heading[:40] if s.heading else "(no heading)"
            tables = f" + {len(s.tables)} tables" if s.tables else ""
            print(f"     {i+1}. [{s.content_pattern}] {heading} ({s.char_count} chars{tables})")
        if len(sections) > 10:
            print(f"     ... and {len(sections) - 10} more sections")
