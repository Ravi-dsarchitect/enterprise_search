import re
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.ingestion.interfaces import (
    Chunker,
    Embedder,
    ParsedDocument,
    ParsedBlock,
    ParsedTable,
    StructuredChunk,
)
from app.core.config import settings


@dataclass
class EnrichedChunk:
    """A chunk enriched with layout and semantic metadata."""
    text: str
    section_type: str = "general"
    content_type: str = "Paragraph"  # "Paragraph", "Table", "List", "Heading"
    section_header: Optional[str] = None
    page_number: Optional[int] = None
    metadata: Dict = field(default_factory=dict)


# Section classification patterns for LIC insurance documents
SECTION_PATTERNS = {
    "eligibility": r"(?i)(eligibility|entry\s*age|age\s*at\s*entry|who\s*can\s*buy|"
                   r"minimum.*age|maximum.*age|age\s*(?:limit|criteria|condition)|"
                   r"proposer\s*age|life\s*assured\s*age|underwriting|medical\s*exam)",

    "benefits": r"(?i)(benefits?|death\s*benefit|maturity\s*benefit|survival\s*benefit|"
                r"bonus|guaranteed\s*addition|loyalty\s*addition|terminal\s*bonus|"
                r"sum\s*assured\s*(?:on\s*death|on\s*maturity|payable)|"
                r"risk\s*cover|life\s*cover|money\s*back|income\s*benefit)",

    "premium": r"(?i)(premium|mode\s*of\s*payment|single\s*premium|regular\s*premium|"
               r"limited\s*pay|tabular\s*premium|premium\s*paying\s*term|ppt|"
               r"premium\s*rebate|high\s*sum\s*assured\s*rebate|hsa)",

    "sum_assured": r"(?i)(sum\s*assured|basic\s*sum|minimum\s*sum|maximum\s*sum|"
                   r"cover\s*amount|annualized\s*premium|bsa)",

    "policy_term": r"(?i)(policy\s*term|term\s*of\s*policy|duration|vesting|"
                   r"deferment\s*period|accumulation\s*period|payout\s*period)",

    "exclusions": r"(?i)(exclusion|not\s*covered|limitation|restriction|"
                  r"suicide\s*clause|waiting\s*period|section\s*45|"
                  r"non.?disclosure|mis.?statement)",

    "loan": r"(?i)(loan\s*facility|policy\s*loan|loan\s*against|"
            r"loan\s*interest|auto.?loan|loan\s*eligibility)",

    "surrender": r"(?i)(surrender|paid.?up|discontinu|lapse|revival|"
                 r"guaranteed\s*surrender|gsv|special\s*surrender|"
                 r"free\s*look|cooling.?off|cancellation)",

    "tax": r"(?i)(tax\s*benefit|section\s*80|10\s*\(\s*10\s*d\s*\)|"
           r"income\s*tax|80c|80ccc|tax\s*deduction|tax\s*free)",

    "rider": r"(?i)(rider|additional\s*benefit|accidental|critical\s*illness|"
             r"waiver|adb|atpd|ci\s*rider|wop|premium\s*waiver|"
             r"accident\s*benefit|disability|term\s*assurance\s*rider)",

    "claim": r"(?i)(claim|settlement|nominee|death\s*claim|maturity\s*claim|"
             r"claim\s*procedure|claim\s*process|documents\s*required|"
             r"claim\s*intimation|claim\s*settlement\s*ratio)",

    "contact": r"(?i)(contact|customer\s*care|helpline|grievance|ombudsman|"
               r"complaint|escalation|resolution)",

    "annuity": r"(?i)(annuity|pension|vesting|deferment|immediate\s*annuity|"
               r"deferred\s*annuity|annuitant|joint\s*life|single\s*life|"
               r"annuity\s*option|annuity\s*rate|purchase\s*price|corpus|"
               r"commutation)",

    "fund": r"(?i)(fund|nav|ulip|market\s*linked|unit\s*(?:price|value|allocation)|"
            r"fund\s*(?:value|option|switch|performance)|equity|debt|balanced)",

    "charges": r"(?i)(charge|deduction|fee|mortality\s*charge|admin\s*charge|"
               r"allocation\s*charge|fund\s*management|discontinuance\s*charge)",
}


def classify_section(text: str) -> str:
    """Classify text into a section type using regex patterns."""
    text_lower = text.lower().strip()
    if len(text_lower) < 3:
        return "general"

    for section_type, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, text_lower):
            return section_type

    return "general"


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, handling abbreviations like Rs., Dr., etc."""
    # Protect common abbreviations
    protected = text.replace("Rs.", "Rs\x00").replace("Dr.", "Dr\x00")
    protected = protected.replace("Mr.", "Mr\x00").replace("Mrs.", "Mrs\x00")
    protected = protected.replace("No.", "No\x00").replace("Sr.", "Sr\x00")
    protected = protected.replace("Jr.", "Jr\x00").replace("vs.", "vs\x00")
    protected = protected.replace("i.e.", "ie\x00").replace("e.g.", "eg\x00")
    protected = protected.replace("etc.", "etc\x00")

    # Split on sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+', protected)

    # Restore abbreviations
    sentences = [p.replace("\x00", ".") for p in parts if p.strip()]
    return sentences


class RecursiveChunker(Chunker):
    """Simple recursive chunker as fallback for non-PDF documents."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, text: str) -> List[str]:
        return self.splitter.split_text(text)


class StructuredChunker(Chunker):
    """
    Structure-aware chunker that consumes ParsedDocument from the new parser.

    Strategy:
    1. Group consecutive blocks by section (using headings as boundaries).
    2. Keep tables as standalone chunks.
    3. Split oversized sections at sentence boundaries.
    4. Prepend document context (plan name, plan type) to every chunk.
    5. Store parent section text for LLM context.
    6. Merge undersized chunks with neighbors.
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        max_table_chunk: int = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.max_table_chunk = max_table_chunk or settings.MAX_TABLE_CHUNK
        self.min_chunk_size = 200
        self.max_chunk_size = self.chunk_size * 1.5

        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, text: str) -> List[str]:
        """Fallback: chunk plain text."""
        return self.fallback_splitter.split_text(text)

    def chunk_structured(
        self, doc: ParsedDocument, doc_metadata: Dict = None
    ) -> List[StructuredChunk]:
        """
        Chunk a ParsedDocument using structural information.

        Args:
            doc: Parsed document with blocks, tables, and headings.
            doc_metadata: Document-level metadata (plan_name, plan_type, etc.)

        Returns:
            List of StructuredChunk with text, section_type, metadata, parent_text.
        """
        doc_metadata = doc_metadata or {}
        plan_name = doc_metadata.get("plan_name", "")
        plan_type = doc_metadata.get("plan_type", "")

        # Step 1: Group blocks into sections based on headings
        sections = self._group_blocks_into_sections(doc.blocks)

        # Step 2: Create chunks from sections
        raw_chunks = []

        for section in sections:
            section_type = section["section_type"]
            heading = section["heading"]
            section_text = section["text"]
            page_number = section["page_number"]

            # Build parent text (full section, capped at 2000 chars)
            parent_text = section_text[:2000]

            if len(section_text) <= self.max_chunk_size:
                # Section fits in one chunk
                raw_chunks.append(
                    StructuredChunk(
                        text=section_text,
                        section_type=section_type,
                        content_type="text",
                        heading=heading,
                        page_number=page_number,
                        parent_text=parent_text,
                    )
                )
            else:
                # Split large section at sentence boundaries
                sub_chunks = self._split_section(
                    section_text, section_type, heading, page_number, parent_text
                )
                raw_chunks.extend(sub_chunks)

        # Step 3: Create table chunks
        for table in doc.tables:
            table_chunk = self._create_table_chunk(table, doc_metadata)
            raw_chunks.append(table_chunk)

        # Step 4: Merge undersized chunks
        merged_chunks = self._merge_small_chunks(raw_chunks)

        # Step 5: Add contextual prefix to every chunk
        for chunk in merged_chunks:
            prefix = self._build_context_prefix(chunk, plan_name, plan_type)
            chunk.text = prefix + chunk.text

        return merged_chunks

    def _group_blocks_into_sections(
        self, blocks: List[ParsedBlock]
    ) -> List[Dict]:
        """Group consecutive blocks into sections using headings as boundaries."""
        if not blocks:
            return []

        sections = []
        current_heading = None
        current_section_type = "general"
        current_blocks = []
        current_page = 1

        for block in blocks:
            if block.heading_level > 0:
                # This is a heading - start a new section
                if current_blocks:
                    section_text = "\n".join(b.content for b in current_blocks).strip()
                    if section_text:
                        sections.append({
                            "heading": current_heading,
                            "section_type": current_section_type,
                            "text": section_text,
                            "page_number": current_page,
                        })

                current_heading = block.content.strip()
                current_section_type = classify_section(current_heading)
                current_blocks = []
                current_page = block.page_number
            else:
                if not current_blocks:
                    current_page = block.page_number
                current_blocks.append(block)

        # Don't forget the last section
        if current_blocks:
            section_text = "\n".join(b.content for b in current_blocks).strip()
            if section_text:
                sections.append({
                    "heading": current_heading,
                    "section_type": current_section_type,
                    "text": section_text,
                    "page_number": current_page,
                })

        return sections

    def _split_section(
        self,
        text: str,
        section_type: str,
        heading: Optional[str],
        page_number: int,
        parent_text: str,
    ) -> List[StructuredChunk]:
        """Split a large section into chunks at sentence boundaries with overlap."""
        sentences = split_into_sentences(text)
        chunks = []
        current_chunk_sentences = []
        current_size = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_size + sentence_len > self.chunk_size and current_chunk_sentences:
                # Save current chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(
                    StructuredChunk(
                        text=chunk_text,
                        section_type=section_type,
                        content_type="text",
                        heading=heading,
                        page_number=page_number,
                        parent_text=parent_text,
                    )
                )

                # Overlap: keep last 1-2 sentences
                overlap_sentences = current_chunk_sentences[-2:]
                current_chunk_sentences = overlap_sentences
                current_size = sum(len(s) for s in current_chunk_sentences)

            current_chunk_sentences.append(sentence)
            current_size += sentence_len

        # Save remaining
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(
                StructuredChunk(
                    text=chunk_text,
                    section_type=section_type,
                    content_type="text",
                    heading=heading,
                    page_number=page_number,
                    parent_text=parent_text,
                )
            )

        return chunks

    def _create_table_chunk(
        self, table: ParsedTable, doc_metadata: Dict
    ) -> StructuredChunk:
        """Create a chunk from a parsed table."""
        text = table.markdown or "\n".join(
            " | ".join(row) for row in table.rows
        )

        # If table is too large, truncate rows but keep header
        if len(text) > self.max_table_chunk and table.rows:
            header = table.rows[0]
            header_line = "| " + " | ".join(header) + " |"
            separator = "| " + " | ".join(["---"] * len(header)) + " |"
            truncated_lines = [header_line, separator]
            current_size = len(header_line) + len(separator)

            for row in table.rows[1:]:
                row_line = "| " + " | ".join(row) + " |"
                if current_size + len(row_line) > self.max_table_chunk:
                    truncated_lines.append("| ... (table truncated) |")
                    break
                truncated_lines.append(row_line)
                current_size += len(row_line)

            text = "\n".join(truncated_lines)

        caption = table.caption or ""
        if caption:
            text = f"{caption}\n{text}"

        return StructuredChunk(
            text=text,
            section_type="general",
            content_type="table",
            heading=caption or None,
            page_number=table.page_number,
            parent_text=text[:2000],
        )

    def _merge_small_chunks(
        self, chunks: List[StructuredChunk]
    ) -> List[StructuredChunk]:
        """Merge consecutive chunks that are too small."""
        if not chunks:
            return chunks

        merged = []
        current = None

        for chunk in chunks:
            if current is None:
                current = chunk
                continue

            current_len = len(current.text)
            chunk_len = len(chunk.text)
            combined_len = current_len + chunk_len

            # Merge if current is too small and combined fits
            should_merge = (
                current_len < self.min_chunk_size
                and combined_len < self.max_chunk_size
                and chunk.content_type == current.content_type
            )

            # Also merge same section type if combined fits
            if (
                not should_merge
                and chunk.section_type == current.section_type
                and combined_len < self.chunk_size
            ):
                should_merge = True

            if should_merge:
                current.text = current.text + "\n\n" + chunk.text
                if not current.heading and chunk.heading:
                    current.heading = chunk.heading
            else:
                merged.append(current)
                current = chunk

        if current is not None:
            merged.append(current)

        return merged

    def _build_context_prefix(
        self, chunk: StructuredChunk, plan_name: str, plan_type: str
    ) -> str:
        """Build contextual prefix for a chunk to improve retrieval."""
        parts = []
        if plan_name:
            parts.append(f"Plan: {plan_name}")
        if plan_type:
            parts.append(f"Type: {plan_type}")
        if chunk.section_type != "general":
            section_display = chunk.section_type.replace("_", " ").title()
            parts.append(f"Section: {section_display}")
        if chunk.page_number:
            parts.append(f"P{chunk.page_number}")

        if parts:
            return "[" + " | ".join(parts) + "] "
        return ""


class SemanticChunker(Chunker):
    """
    Semantic chunker that uses embedding similarity to find natural breakpoints.

    Strategy:
    1. Split text into sentences.
    2. Create sentence groups (windows) and embed them.
    3. Compute cosine similarity between consecutive groups.
    4. Split where similarity drops below a threshold (breakpoints).
    5. Merge resulting segments into chunks respecting size limits.
    """

    def __init__(
        self,
        embedder: "Embedder" = None,
        chunk_size: int = None,
        similarity_threshold: float = 0.5,
        window_size: int = 3,
    ):
        self.embedder = embedder
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, text: str) -> List[str]:
        """Split text into semantically coherent chunks."""
        if not text or not text.strip():
            return []

        sentences = split_into_sentences(text)
        if len(sentences) <= 1:
            return [text.strip()] if text.strip() else []

        # If no embedder or too few sentences, fall back to recursive splitting
        if self.embedder is None or len(sentences) < self.window_size + 1:
            return self.fallback_splitter.split_text(text)

        # Create sentence windows for smoother embeddings
        windows = self._create_windows(sentences)

        # Embed all windows
        try:
            embeddings = self.embedder.embed_documents(windows)
        except Exception:
            return self.fallback_splitter.split_text(text)

        # Find breakpoints using cosine similarity drops
        breakpoints = self._find_breakpoints(embeddings)

        # Build chunks from sentence groups
        chunks = self._build_chunks_from_breakpoints(sentences, breakpoints)
        return chunks

    def _create_windows(self, sentences: List[str]) -> List[str]:
        """Create overlapping windows of sentences for smoother embedding."""
        windows = []
        for i in range(len(sentences)):
            start = max(0, i - self.window_size // 2)
            end = min(len(sentences), i + self.window_size // 2 + 1)
            window_text = " ".join(sentences[start:end])
            windows.append(window_text)
        return windows

    def _find_breakpoints(self, embeddings: List[List[float]]) -> List[int]:
        """Find indices where semantic similarity drops, indicating topic shifts."""
        if len(embeddings) < 2:
            return []

        emb_array = np.array(embeddings)
        similarities = []
        for i in range(len(emb_array) - 1):
            a, b = emb_array[i], emb_array[i + 1]
            norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                similarities.append(0.0)
            else:
                similarities.append(float(np.dot(a, b) / (norm_a * norm_b)))

        # Breakpoints where similarity is below threshold
        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                breakpoints.append(i + 1)  # split after sentence i

        return breakpoints

    def _build_chunks_from_breakpoints(
        self, sentences: List[str], breakpoints: List[int]
    ) -> List[str]:
        """Build chunks by grouping sentences between breakpoints, respecting size limits."""
        if not breakpoints:
            # No breakpoints found - use fallback
            full_text = " ".join(sentences)
            if len(full_text) <= self.chunk_size * 1.5:
                return [full_text]
            return self.fallback_splitter.split_text(full_text)

        chunks = []
        prev = 0
        for bp in breakpoints:
            segment = " ".join(sentences[prev:bp]).strip()
            if segment:
                # Split oversized segments
                if len(segment) > self.chunk_size * 1.5:
                    chunks.extend(self.fallback_splitter.split_text(segment))
                else:
                    chunks.append(segment)
            prev = bp

        # Last segment
        segment = " ".join(sentences[prev:]).strip()
        if segment:
            if len(segment) > self.chunk_size * 1.5:
                chunks.extend(self.fallback_splitter.split_text(segment))
            else:
                chunks.append(segment)

        # Merge tiny chunks with neighbors
        merged = self._merge_tiny_chunks(chunks)
        return merged

    def _merge_tiny_chunks(self, chunks: List[str], min_size: int = 150) -> List[str]:
        """Merge chunks that are too small with their neighbors."""
        if not chunks:
            return chunks

        merged = []
        buffer = ""

        for chunk in chunks:
            if buffer:
                combined = buffer + " " + chunk
                if len(combined) <= self.chunk_size * 1.5:
                    buffer = combined
                else:
                    merged.append(buffer)
                    buffer = chunk
            else:
                buffer = chunk

            if len(buffer) >= min_size:
                merged.append(buffer)
                buffer = ""

        if buffer:
            if merged and len(merged[-1]) + len(buffer) < self.chunk_size * 1.5:
                merged[-1] = merged[-1] + " " + buffer
            else:
                merged.append(buffer)

        return merged


class LayoutAwareChunker(Chunker):
    """
    Layout-aware chunker that detects document structure from plain text.

    Detects headings, bullet lists, tables, and paragraph boundaries,
    then chunks at these layout boundaries. Each chunk is classified by
    section type and content type.
    """

    # Patterns for detecting structural elements in plain text
    HEADING_PATTERN = re.compile(
        r"^(?:"
        r"[A-Z][A-Z\s\d\-&/]{4,80}$|"               # ALL CAPS lines (headings)
        r"\d+\.\s+[A-Z][A-Za-z\s]{3,80}$|"            # "1. Section Title"
        r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,6}\s*:$|"   # "Title Case Heading:"
        r"#{1,4}\s+.+"                                  # Markdown headings
        r")",
        re.MULTILINE,
    )
    LIST_PATTERN = re.compile(r"^\s*(?:[-•*]|\d+[.)]\s|[a-z][.)]\s|[ivx]+[.)]\s)", re.MULTILINE)
    TABLE_PATTERN = re.compile(r"(?:.*\|.*\|)|(?:.*\t.*\t)")

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.min_chunk_size = 150
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, text: str) -> List[str]:
        """Split text into layout-aware chunks."""
        if not text or not text.strip():
            return []

        blocks = self._detect_blocks(text)
        if not blocks:
            return self.fallback_splitter.split_text(text)

        chunks = self._blocks_to_chunks(blocks)
        return [c.text for c in chunks]

    def chunk_with_metadata(self, text: str) -> List[EnrichedChunk]:
        """Split text into layout-aware chunks with metadata."""
        if not text or not text.strip():
            return []

        blocks = self._detect_blocks(text)
        if not blocks:
            # Fallback: wrap plain chunks in EnrichedChunk
            plain_chunks = self.fallback_splitter.split_text(text)
            return [
                EnrichedChunk(text=c, section_type="general", content_type="Paragraph")
                for c in plain_chunks
            ]

        return self._blocks_to_chunks(blocks)

    def _detect_blocks(self, text: str) -> List[Dict]:
        """
        Parse raw text into structural blocks.
        Returns list of dicts with keys: text, block_type, heading, page_number.
        """
        lines = text.split("\n")
        blocks = []
        current_block = {"lines": [], "block_type": "paragraph", "heading": None}
        current_heading = None
        estimated_page = 1

        for line in lines:
            stripped = line.strip()

            # Page break detection (common in PDFs converted to text)
            if stripped.startswith("\f") or re.match(r"^-+\s*Page\s*\d+\s*-+$", stripped, re.IGNORECASE):
                estimated_page += 1
                if stripped.startswith("\f"):
                    stripped = stripped[1:].strip()
                    if not stripped:
                        continue

            # Empty line = potential block boundary
            if not stripped:
                if current_block["lines"]:
                    current_block["page_number"] = estimated_page
                    blocks.append(current_block)
                    current_block = {"lines": [], "block_type": "paragraph", "heading": current_heading}
                continue

            # Detect headings
            if self._is_heading(stripped):
                # Save previous block
                if current_block["lines"]:
                    current_block["page_number"] = estimated_page
                    blocks.append(current_block)

                current_heading = stripped.lstrip("#*- ").rstrip(":").strip()
                blocks.append({
                    "lines": [stripped],
                    "block_type": "heading",
                    "heading": current_heading,
                    "page_number": estimated_page,
                })
                current_block = {"lines": [], "block_type": "paragraph", "heading": current_heading}
                continue

            # Detect tables (pipe-separated or tab-separated)
            if self.TABLE_PATTERN.match(stripped):
                if current_block["lines"] and current_block["block_type"] != "table":
                    current_block["page_number"] = estimated_page
                    blocks.append(current_block)
                    current_block = {"lines": [], "block_type": "table", "heading": current_heading}
                current_block["block_type"] = "table"
                current_block["lines"].append(line)
                continue

            # Detect lists
            if self.LIST_PATTERN.match(stripped):
                if current_block["lines"] and current_block["block_type"] not in ("list", "paragraph"):
                    current_block["page_number"] = estimated_page
                    blocks.append(current_block)
                    current_block = {"lines": [], "block_type": "list", "heading": current_heading}
                if not current_block["lines"]:
                    current_block["block_type"] = "list"
                current_block["lines"].append(line)
                continue

            # Regular paragraph text
            if current_block["block_type"] == "table":
                # End of table, start new paragraph
                current_block["page_number"] = estimated_page
                blocks.append(current_block)
                current_block = {"lines": [], "block_type": "paragraph", "heading": current_heading}

            current_block["lines"].append(line)

        # Don't forget the last block
        if current_block["lines"]:
            current_block["page_number"] = estimated_page
            blocks.append(current_block)

        return blocks

    def _is_heading(self, line: str) -> bool:
        """Heuristic heading detection."""
        stripped = line.strip()
        if not stripped or len(stripped) > 120:
            return False
        if len(stripped) < 3:
            return False

        # Markdown heading
        if re.match(r"^#{1,4}\s+", stripped):
            return True
        # ALL CAPS line (common in LIC PDFs)
        if stripped.isupper() and len(stripped) > 4 and not stripped.startswith("|"):
            return True
        # Numbered heading like "1. Section Title"
        if re.match(r"^\d+\.\s+[A-Z]", stripped) and len(stripped) < 80:
            return True
        # "Title Case:" pattern
        if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,6}\s*:\s*$", stripped):
            return True

        return False

    def _blocks_to_chunks(self, blocks: List[Dict]) -> List[EnrichedChunk]:
        """Convert detected blocks into EnrichedChunk objects, respecting size limits."""
        enriched = []
        current_header = None

        for block in blocks:
            block_text = "\n".join(block["lines"]).strip()
            if not block_text:
                continue

            block_type = block.get("block_type", "paragraph")
            heading = block.get("heading")
            page_number = block.get("page_number")

            if block_type == "heading":
                current_header = heading
                continue  # Headings get merged into the next chunk's header

            # Map block_type to content_type
            content_type_map = {
                "paragraph": "Paragraph",
                "table": "Table",
                "list": "List",
            }
            content_type = content_type_map.get(block_type, "Paragraph")

            # Classify section
            section_text = (current_header or "") + " " + block_text[:500]
            section_type = classify_section(section_text)

            if len(block_text) <= self.chunk_size * 1.5:
                enriched.append(EnrichedChunk(
                    text=block_text,
                    section_type=section_type,
                    content_type=content_type,
                    section_header=current_header,
                    page_number=page_number,
                ))
            else:
                # Split oversized blocks at sentence boundaries
                sub_texts = self.fallback_splitter.split_text(block_text)
                for sub in sub_texts:
                    enriched.append(EnrichedChunk(
                        text=sub,
                        section_type=section_type,
                        content_type=content_type,
                        section_header=current_header,
                        page_number=page_number,
                    ))

        # Merge tiny chunks
        merged = self._merge_small_enriched(enriched)
        return merged

    def _merge_small_enriched(self, chunks: List[EnrichedChunk]) -> List[EnrichedChunk]:
        """Merge enriched chunks that are too small."""
        if not chunks:
            return chunks

        merged = []
        current = None

        for chunk in chunks:
            if current is None:
                current = chunk
                continue

            current_len = len(current.text)
            chunk_len = len(chunk.text)

            should_merge = (
                current_len < self.min_chunk_size
                and current_len + chunk_len < self.chunk_size * 1.5
                and chunk.content_type == current.content_type
            )

            if should_merge:
                current.text = current.text + "\n\n" + chunk.text
                if not current.section_header and chunk.section_header:
                    current.section_header = chunk.section_header
            else:
                merged.append(current)
                current = chunk

        if current is not None:
            merged.append(current)

        return merged


class HybridLayoutSemanticChunker(Chunker):
    """
    Combines layout-aware splitting with semantic refinement.

    Strategy:
    1. Use LayoutAwareChunker to detect structure and split at layout boundaries.
    2. For oversized layout chunks, use SemanticChunker to refine splits.
    3. Return EnrichedChunk objects with full metadata.
    """

    def __init__(
        self,
        embedder: "Embedder" = None,
        chunk_size: int = None,
        similarity_threshold: float = 0.5,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.embedder = embedder
        self.layout_chunker = LayoutAwareChunker(chunk_size=self.chunk_size)
        self.semantic_chunker = SemanticChunker(
            embedder=embedder,
            chunk_size=self.chunk_size,
            similarity_threshold=similarity_threshold,
        )

    def chunk(self, text: str) -> List[str]:
        """Split text using hybrid layout + semantic approach."""
        enriched = self.chunk_with_metadata(text)
        return [c.text for c in enriched]

    def chunk_with_metadata(self, text: str) -> List[EnrichedChunk]:
        """
        Split text using layout detection, then refine large chunks semantically.
        Returns EnrichedChunk objects with section_type, content_type, etc.
        """
        if not text or not text.strip():
            return []

        # Step 1: Layout-aware chunking
        layout_chunks = self.layout_chunker.chunk_with_metadata(text)

        # Step 2: Refine oversized chunks with semantic splitting
        refined = []
        for chunk in layout_chunks:
            if len(chunk.text) > self.chunk_size * 1.5 and chunk.content_type == "Paragraph":
                # Use semantic chunker to split large paragraph chunks
                sub_texts = self.semantic_chunker.chunk(chunk.text)
                for sub in sub_texts:
                    refined.append(EnrichedChunk(
                        text=sub,
                        section_type=chunk.section_type,
                        content_type=chunk.content_type,
                        section_header=chunk.section_header,
                        page_number=chunk.page_number,
                    ))
            else:
                refined.append(chunk)

        return refined
