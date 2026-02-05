"""
TableAwareChunker - Preserves table context and structure.

For table-heavy documents, keeps tables with their headers, captions,
and surrounding context.
"""

from typing import List, Dict, Optional

from app.services.ingestion.interfaces import (
    Chunker,
    ParsedDocument,
    ParsedTable,
    ParsedBlock,
    StructuredChunk,
)
from app.services.ingestion.chunkers import StructuredChunker, classify_section
from app.core.config import settings


class TableAwareChunker(Chunker):
    """
    Specialized chunker for table-heavy documents.

    Strategy:
    1. Identify tables and their surrounding context
    2. Keep tables with headers, captions, and preceding text
    3. Split large tables by rows while repeating headers
    4. Delegate non-table content to StructuredChunker
    """

    def __init__(
        self,
        chunk_size: int = None,
        max_table_chunk: int = None,
        context_lines: int = 3,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.max_table_chunk = max_table_chunk or settings.MAX_TABLE_CHUNK
        self.context_lines = context_lines

        # Delegate non-table content
        self._structured_chunker = StructuredChunker(
            chunk_size=chunk_size,
            max_table_chunk=max_table_chunk,
        )

    def chunk(self, text: str) -> List[str]:
        """Basic interface - delegate to structured chunker."""
        return self._structured_chunker.chunk(text)

    def chunk_structured(
        self,
        doc: ParsedDocument,
        doc_metadata: Dict = None,
    ) -> List[StructuredChunk]:
        """Create chunks with enhanced table handling."""
        doc_metadata = doc_metadata or {}
        chunks = []

        # Build table context map
        table_contexts = self._build_table_contexts(doc)

        # Track which blocks are table context
        used_block_indices = set()

        # Process tables first
        for table_ctx in table_contexts:
            table_chunks = self._chunk_table_with_context(table_ctx, doc_metadata)
            chunks.extend(table_chunks)
            used_block_indices.update(table_ctx.get("context_block_indices", []))

        # Process remaining blocks (excluding table contexts)
        remaining_blocks = [
            b for i, b in enumerate(doc.blocks)
            if i not in used_block_indices
        ]

        if remaining_blocks:
            # Create temporary doc with remaining blocks
            from app.services.ingestion.interfaces import ParsedDocument as PD

            remaining_doc = PD(
                filename=doc.filename,
                total_pages=doc.total_pages,
                blocks=remaining_blocks,
                tables=[],
                full_text="\n".join(b.content for b in remaining_blocks),
                headings=doc.headings,
            )

            text_chunks = self._structured_chunker.chunk_structured(
                remaining_doc, doc_metadata
            )
            chunks.extend(text_chunks)

        return chunks

    def _build_table_contexts(self, doc: ParsedDocument) -> List[Dict]:
        """Build context for each table (caption, preceding text, headings)."""
        contexts = []

        for table in doc.tables:
            context = {
                "table": table,
                "caption": table.caption or "",
                "preceding_text": "",
                "related_heading": None,
                "context_block_indices": [],
            }

            # Find blocks on same page before this table
            page_blocks = [
                (i, b) for i, b in enumerate(doc.blocks)
                if b.page_number == table.page_number
            ]

            # Get last few text blocks
            preceding = []
            for i, block in page_blocks[-self.context_lines:]:
                if block.heading_level > 0:
                    context["related_heading"] = block.content.strip()
                else:
                    preceding.append(block.content.strip())
                context["context_block_indices"].append(i)

            context["preceding_text"] = "\n".join(preceding)
            contexts.append(context)

        return contexts

    def _chunk_table_with_context(
        self,
        table_ctx: Dict,
        doc_metadata: Dict,
    ) -> List[StructuredChunk]:
        """Create chunks for a table with its context."""
        table: ParsedTable = table_ctx["table"]
        chunks = []

        # Build context prefix
        context_parts = []
        if table_ctx["related_heading"]:
            context_parts.append(f"Section: {table_ctx['related_heading']}")
        if table_ctx["caption"]:
            context_parts.append(f"Table: {table_ctx['caption']}")
        if table_ctx["preceding_text"]:
            context_parts.append(table_ctx["preceding_text"])

        context_prefix = "\n".join(context_parts)
        if context_prefix:
            context_prefix += "\n\n"

        # Get table text
        table_text = table.markdown or self._table_to_text(table)
        full_chunk_text = context_prefix + table_text

        if len(full_chunk_text) <= self.max_table_chunk:
            # Table fits in one chunk
            chunks.append(StructuredChunk(
                text=full_chunk_text,
                section_type=self._classify_table(table_text),
                content_type="table",
                heading=table_ctx["related_heading"] or table_ctx["caption"],
                page_number=table.page_number,
                parent_text=full_chunk_text[:2000],
                metadata={
                    "table_rows": len(table.rows) if table.rows else 0,
                    "table_cols": len(table.headers) if table.headers else 0,
                },
            ))
        else:
            # Split large table by rows
            split_chunks = self._split_large_table(
                table, context_prefix, table_ctx
            )
            chunks.extend(split_chunks)

        return chunks

    def _split_large_table(
        self,
        table: ParsedTable,
        context_prefix: str,
        table_ctx: Dict,
    ) -> List[StructuredChunk]:
        """Split a large table into multiple chunks, repeating headers."""
        chunks = []

        if not table.rows or len(table.rows) < 2:
            return chunks

        headers = table.rows[0]
        data_rows = table.rows[1:]

        # Build header string
        header_line = "| " + " | ".join(str(h) for h in headers) + " |"
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        header_text = f"{header_line}\n{separator}\n"

        current_rows = []
        current_size = len(context_prefix) + len(header_text)
        part_num = 1
        section_type = self._classify_table(table.markdown or "")

        for row in data_rows:
            row_line = "| " + " | ".join(str(c) for c in row) + " |"
            row_size = len(row_line) + 1

            if current_size + row_size > self.max_table_chunk and current_rows:
                # Create chunk
                table_body = "\n".join(current_rows)
                chunk_text = f"{context_prefix}{header_text}{table_body}"

                if part_num > 1:
                    chunk_text = f"(Table continued, Part {part_num})\n{chunk_text}"

                chunks.append(StructuredChunk(
                    text=chunk_text,
                    section_type=section_type,
                    content_type="table",
                    heading=table_ctx.get("related_heading"),
                    page_number=table.page_number,
                    parent_text=f"{context_prefix}{header_text}(Full table: {len(data_rows)} rows)",
                    metadata={
                        "table_part": part_num,
                        "total_rows": len(data_rows),
                    },
                ))

                current_rows = []
                current_size = len(context_prefix) + len(header_text) + 40
                part_num += 1

            current_rows.append(row_line)
            current_size += row_size

        # Final chunk
        if current_rows:
            table_body = "\n".join(current_rows)
            chunk_text = f"{context_prefix}{header_text}{table_body}"

            if part_num > 1:
                chunk_text = f"(Table continued, Part {part_num})\n{chunk_text}"

            chunks.append(StructuredChunk(
                text=chunk_text,
                section_type=section_type,
                content_type="table",
                heading=table_ctx.get("related_heading"),
                page_number=table.page_number,
                parent_text=f"{context_prefix}{header_text}(Full table: {len(data_rows)} rows)",
                metadata={
                    "table_part": part_num if part_num > 1 else None,
                    "total_rows": len(data_rows),
                },
            ))

        return chunks

    def _table_to_text(self, table: ParsedTable) -> str:
        """Convert table to plain text."""
        if not table.rows:
            return ""
        return "\n".join(" | ".join(str(c) for c in row) for row in table.rows)

    def _classify_table(self, table_text: str) -> str:
        """Classify table content type."""
        text_lower = table_text.lower()
        if "premium" in text_lower:
            return "premium"
        if "eligibility" in text_lower or "age" in text_lower:
            return "eligibility"
        if "benefit" in text_lower:
            return "benefits"
        if "charge" in text_lower or "fee" in text_lower:
            return "charges"
        return "general"
