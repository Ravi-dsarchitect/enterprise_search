"""
HierarchicalChunker - Creates parent-child chunk relationships.

Parent chunks = full sections (for broad context)
Child chunks = details within sections (for precise retrieval)

At retrieval: match child -> expand to parent for complete context.
"""

import uuid
from typing import List, Dict, Optional

from app.services.ingestion.interfaces import (
    Chunker,
    ParsedDocument,
    ParsedBlock,
    StructuredChunk,
)
from app.services.ingestion.smart_chunking.models import HierarchicalChunk
from app.services.ingestion.chunkers import split_into_sentences, classify_section
from app.core.config import settings


class HierarchicalChunker(Chunker):
    """
    Creates parent-child chunk hierarchies for nested documents.

    Strategy:
    1. Build section tree from headings
    2. Create PARENT chunks (full section, up to parent_chunk_size)
    3. Create CHILD chunks (normal size) for detailed retrieval
    4. Link parent-child relationships via IDs
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        parent_chunk_size: int = None,
        min_section_for_children: int = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        # Parent chunks are larger (capture full section context)
        self.parent_chunk_size = parent_chunk_size or getattr(
            settings, "HIERARCHICAL_PARENT_CHUNK_SIZE", self.chunk_size * 3
        )
        # Only create children if section exceeds this size
        self.min_section_for_children = min_section_for_children or getattr(
            settings, "HIERARCHICAL_MIN_SECTION_SIZE", 800
        )

    def chunk(self, text: str) -> List[str]:
        """Basic interface - returns flat list of chunk texts."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_text(text)

    def chunk_structured(
        self,
        doc: ParsedDocument,
        doc_metadata: Dict = None,
    ) -> List[StructuredChunk]:
        """
        Create hierarchical chunks from parsed document.

        Returns StructuredChunks with hierarchy info in metadata.
        """
        doc_metadata = doc_metadata or {}

        # Step 1: Build section tree
        sections = self._build_section_tree(doc.blocks)

        # Step 2: Create chunks with hierarchy
        all_chunks: List[HierarchicalChunk] = []
        chunk_counter = [0]

        def next_id():
            chunk_counter[0] += 1
            return f"hc_{chunk_counter[0]:04d}"

        for section in sections:
            section_chunks = self._chunk_section(section, doc_metadata, None, 0, next_id)
            all_chunks.extend(section_chunks)

        # Step 3: Add tables
        for table in doc.tables:
            all_chunks.append(self._create_table_chunk(table, next_id()))

        # Step 4: Convert to StructuredChunk
        return self._convert_to_structured_chunks(all_chunks, doc_metadata)

    def _build_section_tree(self, blocks: List[ParsedBlock]) -> List[Dict]:
        """Build hierarchical section tree from blocks."""
        if not blocks:
            return []

        sections = []
        current_sections_by_level = {}

        for block in blocks:
            if block.heading_level > 0:
                section = {
                    "heading": block.content.strip(),
                    "heading_level": block.heading_level,
                    "blocks": [],
                    "subsections": [],
                    "page_number": block.page_number,
                }

                level = block.heading_level

                # Find parent section
                parent_level = level - 1
                while parent_level > 0 and parent_level not in current_sections_by_level:
                    parent_level -= 1

                if parent_level > 0 and parent_level in current_sections_by_level:
                    current_sections_by_level[parent_level]["subsections"].append(section)
                else:
                    sections.append(section)

                current_sections_by_level[level] = section

                # Clear deeper levels
                for deeper in list(current_sections_by_level.keys()):
                    if deeper > level:
                        del current_sections_by_level[deeper]
            else:
                # Content block
                if current_sections_by_level:
                    deepest = max(current_sections_by_level.keys())
                    current_sections_by_level[deepest]["blocks"].append(block)
                else:
                    if not sections or sections[-1].get("heading"):
                        sections.append({
                            "heading": None,
                            "heading_level": 0,
                            "blocks": [],
                            "subsections": [],
                            "page_number": block.page_number,
                        })
                    sections[-1]["blocks"].append(block)

        return sections

    def _chunk_section(
        self,
        section: Dict,
        doc_metadata: Dict,
        parent_id: Optional[str],
        depth: int,
        next_id,
    ) -> List[HierarchicalChunk]:
        """Create chunks for a section and its subsections."""
        chunks = []

        # Collect section text (not including subsections)
        section_text = "\n".join(b.content for b in section["blocks"]).strip()
        heading = section.get("heading")
        page_number = section.get("page_number", 1)

        if not section_text and not section.get("subsections"):
            return chunks

        section_id = next_id()
        section_type = classify_section(section_text[:500] if section_text else heading or "")

        # Decide: create parent + children or single chunk
        needs_children = len(section_text) > self.min_section_for_children

        if section_text:
            if needs_children:
                # Create PARENT chunk (broader context)
                parent_text_capped = section_text[:self.parent_chunk_size]
                parent_chunk = HierarchicalChunk(
                    text=parent_text_capped,
                    chunk_id=section_id,
                    parent_chunk_id=parent_id,
                    child_chunk_ids=[],
                    hierarchy_level=depth,
                    is_parent=True,
                    section_complete=len(section_text) <= self.parent_chunk_size,
                    section_type=section_type,
                    content_type="text",
                    heading=heading,
                    page_number=page_number,
                    parent_text=section_text[:2000],
                    metadata={"full_section_chars": len(section_text)},
                )
                chunks.append(parent_chunk)

                # Create CHILD chunks
                child_chunks = self._create_child_chunks(
                    section_text, section_type, heading, page_number,
                    section_id, section_text[:2000], depth, next_id
                )
                parent_chunk.child_chunk_ids = [c.chunk_id for c in child_chunks]
                chunks.extend(child_chunks)
            else:
                # Single chunk for small section
                chunks.append(HierarchicalChunk(
                    text=section_text,
                    chunk_id=section_id,
                    parent_chunk_id=parent_id,
                    hierarchy_level=depth,
                    is_parent=False,
                    section_type=section_type,
                    content_type="text",
                    heading=heading,
                    page_number=page_number,
                    parent_text=section_text[:2000],
                ))

        # Process subsections recursively
        for subsection in section.get("subsections", []):
            sub_chunks = self._chunk_section(
                subsection, doc_metadata, section_id, depth + 1, next_id
            )
            chunks.extend(sub_chunks)

        return chunks

    def _create_child_chunks(
        self,
        text: str,
        section_type: str,
        heading: Optional[str],
        page_number: int,
        parent_id: str,
        parent_text: str,
        depth: int,
        next_id,
    ) -> List[HierarchicalChunk]:
        """Split section text into child chunks."""
        if len(text) <= self.chunk_size:
            return [HierarchicalChunk(
                text=text,
                chunk_id=next_id(),
                parent_chunk_id=parent_id,
                hierarchy_level=depth + 1,
                is_parent=False,
                section_type=section_type,
                content_type="text",
                heading=heading,
                page_number=page_number,
                parent_text=parent_text,
            )]

        sentences = split_into_sentences(text)
        children = []
        current = []
        current_size = 0

        for sentence in sentences:
            if current_size + len(sentence) > self.chunk_size and current:
                children.append(HierarchicalChunk(
                    text=" ".join(current),
                    chunk_id=next_id(),
                    parent_chunk_id=parent_id,
                    hierarchy_level=depth + 1,
                    is_parent=False,
                    section_type=section_type,
                    content_type="text",
                    heading=heading,
                    page_number=page_number,
                    parent_text=parent_text,
                ))
                # Overlap: keep last sentence
                current = current[-1:] if current else []
                current_size = len(current[0]) if current else 0

            current.append(sentence)
            current_size += len(sentence)

        if current:
            children.append(HierarchicalChunk(
                text=" ".join(current),
                chunk_id=next_id(),
                parent_chunk_id=parent_id,
                hierarchy_level=depth + 1,
                is_parent=False,
                section_type=section_type,
                content_type="text",
                heading=heading,
                page_number=page_number,
                parent_text=parent_text,
            ))

        return children

    def _create_table_chunk(self, table, chunk_id: str) -> HierarchicalChunk:
        """Create chunk from table."""
        text = table.markdown or ""
        return HierarchicalChunk(
            text=text,
            chunk_id=chunk_id,
            is_parent=False,
            section_type="general",
            content_type="table",
            heading=table.caption or "",
            page_number=table.page_number,
            parent_text=text[:2000],
        )

    def _convert_to_structured_chunks(
        self,
        hierarchical_chunks: List[HierarchicalChunk],
        doc_metadata: Dict,
    ) -> List[StructuredChunk]:
        """Convert HierarchicalChunks to StructuredChunks."""
        structured = []
        plan_name = doc_metadata.get("plan_name", "")

        for h in hierarchical_chunks:
            # Build prefix
            prefix = self._build_prefix(plan_name, h.heading, h.section_type, h.page_number)

            # Store hierarchy info in metadata
            meta = dict(h.metadata)
            meta.update({
                "chunk_id": h.chunk_id,
                "parent_chunk_id": h.parent_chunk_id,
                "child_chunk_ids": h.child_chunk_ids,
                "hierarchy_level": h.hierarchy_level,
                "is_parent_chunk": h.is_parent,
            })

            structured.append(StructuredChunk(
                text=prefix + h.text,
                section_type=h.section_type,
                content_type=h.content_type,
                heading=h.heading,
                page_number=h.page_number,
                parent_text=h.parent_text,
                metadata=meta,
            ))

        return structured

    def _build_prefix(
        self,
        plan_name: str,
        heading: Optional[str],
        section_type: str,
        page_number: int,
    ) -> str:
        """Build context prefix for chunk."""
        parts = []
        if plan_name:
            parts.append(f"Plan: {plan_name}")
        if heading:
            parts.append(f"Section: {heading}")
        elif section_type != "general":
            parts.append(f"Section: {section_type.replace('_', ' ').title()}")
        if page_number:
            parts.append(f"P{page_number}")

        return "[" + " | ".join(parts) + "] " if parts else ""
