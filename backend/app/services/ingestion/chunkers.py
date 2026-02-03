from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass
from app.services.ingestion.interfaces import Chunker, Embedder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker


@dataclass
class EnrichedChunk:
    """A chunk with layout-aware metadata."""
    text: str
    section_type: str  # Benefits, Eligibility, Premium, etc.
    content_type: str  # Table, List, Paragraph, Header
    section_header: Optional[str] = None
    page_number: Optional[int] = None
    has_financial_data: bool = False
    has_age_criteria: bool = False


class RecursiveChunker(Chunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk(self, text: str) -> List[str]:
        return self.splitter.split_text(text)


class LayoutAwareChunker(Chunker):
    """
    Layout-aware chunker specifically designed for LIC insurance documents.

    Optimized based on research (2025 best practices):
    - Chunk size: 1500-2000 chars for brochures (analytical queries need more context)
    - Semantic boundaries: Respects section breaks
    - Table preservation: Keeps tables intact up to 3000 chars
    - Contextual headers: Prepends section context to each chunk
    - Small section merging: Combines related small sections

    Features:
    - Detects section boundaries (Benefits, Eligibility, Premium, etc.)
    - Keeps tables together
    - Preserves section context in each chunk
    - Returns enriched chunks with metadata
    """

    # Section patterns for LIC insurance documents (comprehensive - from all doc analysis)
    SECTION_PATTERNS = {
        # Eligibility & Entry Conditions
        "eligibility": r"(?i)(eligibility|entry\s*age|age\s*at\s*entry|who\s*can\s*buy|minimum.*age|maximum.*age|"
                       r"age\s*(?:limit|criteria|condition)|proposer\s*age|life\s*assured\s*age|"
                       r"policy\s*conditions|underwriting|medical\s*exam|standard\s*lives|"
                       r"sub.?standard|extra\s*premium|loading|age\s*nearer\s*birthday)",

        # Benefits - Death, Maturity, Survival, Bonus
        "benefits": r"(?i)(benefits?|death\s*benefit|maturity\s*benefit|survival\s*benefit|"
                    r"bonus|guaranteed\s*addition|loyalty\s*addition|terminal\s*bonus|"
                    r"reversionary\s*bonus|final\s*additional\s*bonus|fab|"
                    r"sum\s*assured\s*(?:on\s*death|on\s*maturity|payable)|"
                    r"risk\s*cover|life\s*cover|basic\s*benefit|"
                    r"money\s*back|survival\s*amount|periodic\s*payment|"
                    r"income\s*benefit|guaranteed\s*income|flexi\s*income)",

        # Premium Payment Details
        "premium": r"(?i)(premium|payment|installment|mode\s*of\s*payment|"
                   r"single\s*premium|regular\s*premium|limited\s*pay|flexible\s*premium|"
                   r"tabular\s*premium|modal\s*premium|premium\s*paying\s*term|ppt|"
                   r"yearly|half.?yearly|quarterly|monthly|ecs|nach|"
                   r"premium\s*rebate|discount|high\s*sum\s*assured\s*rebate|hsa)",

        # Sum Assured
        "sum_assured": r"(?i)(sum\s*assured|basic\s*sum|minimum\s*sum|maximum\s*sum|cover\s*amount|"
                       r"annualized\s*premium|annual\s*premium|multiplier|"
                       r"bsa|basic\s*sa|death\s*sa|maturity\s*sa)",

        # Policy Term & Duration
        "policy_term": r"(?i)(policy\s*term|term\s*of\s*policy|duration|maturity|vesting|"
                       r"deferment\s*period|accumulation\s*period|payout\s*period|"
                       r"benefit\s*term|annuity\s*term|cover\s*period)",

        # Exclusions & Limitations
        "exclusions": r"(?i)(exclusion|not\s*covered|limitation|restriction|exception|"
                      r"suicide\s*clause|waiting\s*period|contestability|"
                      r"section\s*45|non.?disclosure|mis.?statement|fraud|"
                      r"war|riot|hazardous|aviation|pre.?existing)",

        # Loan Facility
        "loan": r"(?i)(loan\s*facility|policy\s*loan|loan\s*against|borrowing|"
                r"loan\s*interest|auto.?loan|alf|loan\s*repayment|"
                r"loan\s*eligibility|loan\s*amount|outstanding\s*loan)",

        # Surrender & Paid-up
        "surrender": r"(?i)(surrender|paid.?up|discontinu|lapse|revival|"
                     r"special\s*surrender|guaranteed\s*surrender|gsv|"
                     r"special\s*surrender\s*value|ssv|acquired\s*paid.?up|"
                     r"reduced\s*paid.?up|rpv|reinstatement|revival\s*period|"
                     r"free\s*look|cooling.?off|cancellation)",

        # Tax Benefits
        "tax": r"(?i)(tax\s*benefit|section\s*80|10\s*\(\s*10\s*d\s*\)|income\s*tax|gst|exempt|"
               r"80c|80ccc|80ccd|tax\s*deduction|tax\s*free|"
                r"annuity\s*tax|pension\s*tax|capital\s*gains)",

        # Riders & Add-ons
        "rider": r"(?i)(rider|additional\s*benefit|accidental|critical\s*illness|waiver|"
                 r"adb|atpd|ci\s*rider|wop|premium\s*waiver|"
                 r"accident\s*benefit|disability|dismemberment|"
                 r"new\s*term\s*assurance|term\s*assurance\s*rider|"
                 r"new\s*critical\s*illness|linked\s*ci)",

        # Claim Settlement
        "claim": r"(?i)(claim|settlement|nominee|assignee|death\s*claim|maturity\s*claim|"
                 r"survival\s*claim|claim\s*procedure|claim\s*process|"
                 r"claim\s*form|documents\s*required|claim\s*intimation|"
                 r"proof\s*of\s*(?:death|age|identity)|"
                 r"claim\s*settlement\s*ratio|csr|neft|bank\s*details)",

        # Contact & Support
        "contact": r"(?i)(contact|customer\s*care|helpline|branch|grievance|ombudsman|"
                   r"lic\s*(?:office|branch|zonal|divisional|central)|"
                   r"portal|website|app|online\s*services|"
                   r"customer\s*zone|registered\s*user|"
                   r"complaint|escalation|resolution)",

        # Annuity & Pension (expanded for pension documents)
        "annuity": r"(?i)(annuity|pension|vesting|deferment|immediate\s*annuity|deferred\s*annuity|"
                   r"annuitant|joint\s*life|single\s*life|"
                   r"annuity\s*option|annuity\s*rate|annuity\s*factor|"
                   r"purchase\s*price|annuity\s*payable|corpus|"
                   r"commutation|one.?third\s*commutation|"
                   r"life\s*annuity|annuity\s*certain|"
                   r"pension\s*(?:amount|payment|option)|"
                   r"jeevan\s*(?:akshay|shanti|nidhi|dhara))",

        # Fund & ULIP (expanded for unit-linked documents)
        "fund": r"(?i)(fund|nav|unit|ulip|market\s*linked|investment|"
                r"unit\s*(?:price|value|allocation)|"
                r"fund\s*(?:value|option|switch|performance)|"
                r"equity|debt|balanced|bond|money\s*market|"
                r"growth|secure|liquid|"
                r"asset\s*allocation|portfolio|sfin|"
                r"fund\s*management\s*charge|fmc|"
                r"mortality\s*charge|policy\s*admin|"
                r"premium\s*allocation|settlement\s*fund)",

        # Charges & Deductions (new section for ULIPs and pension)
        "charges": r"(?i)(charge|deduction|fee|expense|cost|"
                   r"mortality\s*charge|admin\s*charge|allocation\s*charge|"
                   r"fund\s*management|switching\s*charge|"
                   r"discontinuance\s*charge|partial\s*withdrawal|"
                   r"miscellaneous\s*charge|service\s*tax|gst\s*applicable)",

        # Grace Period & Revival (new section)
        "grace_revival": r"(?i)(grace\s*period|days\s*(?:of\s*)?grace|revival|reinstatement|"
                         r"lapsed\s*policy|revival\s*interest|"
                         r"revival\s*(?:period|scheme|conditions)|"
                         r"late\s*fee|auto\s*cover|extended\s*term)",

        # Special Features (new section)
        "special_features": r"(?i)(special\s*feature|unique|advantage|highlight|"
                           r"settlement\s*option|extended\s*cover|"
                           r"auto.?cover|paid.?up\s*addition|"
                           r"guaranteed\s*insurability|gi\s*benefit|"
                           r"top.?up|partial\s*withdrawal)",
    }

    # Table detection patterns
    TABLE_INDICATORS = [
        r"\|\s*.*\s*\|",  # Markdown-style tables
        r"\t.*\t",  # Tab-separated
        r"\s{3,}\d+\s{3,}",  # Multiple spaces with numbers (typical in PDF tables)
        r"^\s*\d+\s+\d+\s+\d+",  # Rows of numbers
        r"(?:Rs\.?|₹)\s*[\d,]+(?:\.\d+)?",  # Currency amounts
    ]

    # Optimal chunk sizes based on research (in characters)
    # - Factoid queries: 800-1000 chars
    # - Analytical queries: 1600-2000 chars
    # - Tables: up to 3000 chars to keep integrity
    DEFAULT_CHUNK_SIZE = 1800  # ~450 tokens - good for mixed queries
    MIN_CHUNK_SIZE = 800  # Increased from 400 - aggressively merge small chunks
    MAX_TABLE_CHUNK = 3000  # Allow larger chunks for tables
    OVERLAP_RATIO = 0.15  # 15% overlap
    MIN_SECTION_CONTENT = 500  # Minimum chars before allowing section split

    def __init__(self, chunk_size: int = 1800, chunk_overlap: int = 270):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk(self, text: str) -> List[str]:
        """
        Main chunking method that returns plain text chunks.
        For enriched chunks with metadata, use chunk_with_metadata().
        """
        enriched = self.chunk_with_metadata(text)
        return [c.text for c in enriched]

    def chunk_with_metadata(self, text: str) -> List[EnrichedChunk]:
        """
        Chunk text with layout awareness and return enriched chunks.

        Strategy:
        1. Parse page markers and sections
        2. Split by section boundaries
        3. Keep tables together
        4. Apply size limits while preserving context
        """
        # Step 1: Extract page numbers and clean text
        pages = self._parse_pages(text)

        # Step 2: Detect sections within each page
        sections = self._detect_sections(pages)

        # Step 3: Create chunks respecting section boundaries
        chunks = self._create_section_aware_chunks(sections)

        # Step 4: Enrich chunks with metadata
        enriched_chunks = self._enrich_chunks(chunks)

        return enriched_chunks

    def _parse_pages(self, text: str) -> List[Tuple[int, str]]:
        """Extract page numbers and content."""
        pages = []

        # Match page markers like "## Page 1" or "Page 1" or "--- Page 1 ---"
        page_pattern = r"(?:##\s*)?[Pp]age\s*(\d+)"
        parts = re.split(page_pattern, text)

        if len(parts) == 1:
            # No page markers found, treat as single page
            return [(1, text)]

        # parts: [pre-content, page_num1, content1, page_num2, content2, ...]
        for i in range(1, len(parts), 2):
            page_num = int(parts[i])
            content = parts[i + 1] if i + 1 < len(parts) else ""
            if content.strip():
                pages.append((page_num, content))

        return pages if pages else [(1, text)]

    def _detect_sections(self, pages: List[Tuple[int, str]]) -> List[Dict]:
        """Detect section boundaries within the document.

        Uses MIN_SECTION_CONTENT to avoid splitting too frequently.
        Only creates new section if:
        1. New section type detected AND
        2. Current block has accumulated enough content (MIN_SECTION_CONTENT)
        """
        sections = []
        current_section = "general"
        current_header = None
        chars_since_last_split = 0

        for page_num, content in pages:
            lines = content.split('\n')
            current_block = []

            for line in lines:
                line_len = len(line)

                # Check if this line is a section header
                detected_section = self._classify_section(line)

                # Check if this looks like a header (short, possibly bold/caps)
                is_header = self._is_header_line(line)

                # Only split if:
                # 1. Different section type detected
                # 2. We have accumulated enough content since last split
                # 3. The line looks like a header (strong indicator)
                should_split = (
                    detected_section and
                    detected_section != current_section and
                    (chars_since_last_split >= self.MIN_SECTION_CONTENT or is_header)
                )

                if should_split:
                    # Save current block
                    if current_block:
                        sections.append({
                            "page": page_num,
                            "section_type": current_section,
                            "header": current_header,
                            "content": '\n'.join(current_block)
                        })

                    current_section = detected_section
                    current_header = line.strip() if is_header else None
                    current_block = [line] if not is_header else []
                    chars_since_last_split = 0
                else:
                    current_block.append(line)
                    chars_since_last_split += line_len

            # Save remaining block for this page
            if current_block:
                sections.append({
                    "page": page_num,
                    "section_type": current_section,
                    "header": current_header,
                    "content": '\n'.join(current_block)
                })

        return sections

    def _classify_section(self, line: str) -> Optional[str]:
        """Classify a line into a section type."""
        line_lower = line.lower().strip()

        # Skip empty or very short lines
        if len(line_lower) < 3:
            return None

        for section_type, pattern in self.SECTION_PATTERNS.items():
            if re.search(pattern, line_lower):
                return section_type

        return None

    def _is_header_line(self, line: str) -> bool:
        """Check if a line looks like a section header."""
        line = line.strip()

        if not line or len(line) > 100:
            return False

        # Markdown headers
        if line.startswith('#'):
            return True

        # All caps or title case, relatively short
        if len(line) < 60 and (line.isupper() or line.istitle()):
            return True

        # Ends with colon
        if line.endswith(':') and len(line) < 50:
            return True

        # Common header patterns
        header_patterns = [
            r"^\d+\.\s+[A-Z]",  # Numbered section "1. Benefits"
            r"^[A-Z][A-Z\s]+:?$",  # ALL CAPS
            r"^[A-Za-z\s]+:$",  # "Benefits:"
        ]

        for pattern in header_patterns:
            if re.match(pattern, line):
                return True

        return False

    def _create_section_aware_chunks(self, sections: List[Dict]) -> List[Dict]:
        """Create chunks while respecting section boundaries and merging small sections."""
        # Step 1: Create initial chunks
        initial_chunks = []

        for section in sections:
            content = section["content"]

            # Detect if this section contains a table
            is_table = self._is_table_content(content)

            if is_table:
                # Keep tables together if possible, or split carefully
                # Use larger size limit for tables
                table_chunks = self._chunk_table(content, section)
                initial_chunks.extend(table_chunks)
            elif len(content) > self.chunk_size * 1.5:
                # Section too large, need to split
                sub_chunks = self._split_large_section(content, section)
                initial_chunks.extend(sub_chunks)
            elif len(content.strip()) > 30:
                # Section is appropriately sized
                initial_chunks.append({
                    **section,
                    "content_type": self._detect_content_type(content)
                })

        # Step 2: Merge small consecutive chunks of same section type
        merged_chunks = self._merge_small_chunks(initial_chunks)

        # Step 3: Add contextual headers for better retrieval
        contextualized_chunks = self._add_contextual_headers(merged_chunks)

        return contextualized_chunks

    def _merge_small_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Aggressively merge small consecutive chunks to reach target size.

        Strategy:
        - Always merge same section type regardless of size (up to chunk_size)
        - Merge different section types if current chunk is very small (<MIN_CHUNK_SIZE)
        - Prefer keeping similar content types together (tables with tables)
        """
        if not chunks:
            return chunks

        merged = []
        current = None

        for chunk in chunks:
            content_len = len(chunk.get("content", ""))
            current_len = len(current.get("content", "")) if current else 0
            combined_len = current_len + content_len

            if current is None:
                current = chunk.copy()
                continue

            # Same section type: merge if under chunk_size limit
            same_section = chunk.get("section_type") == current.get("section_type")
            same_content_type = chunk.get("content_type") == current.get("content_type")

            # Decision logic for merging
            should_merge = False

            if combined_len < self.chunk_size:
                if same_section:
                    # Always merge same section (primary rule)
                    should_merge = True
                elif current_len < self.MIN_CHUNK_SIZE:
                    # Current chunk too small - merge even with different section
                    should_merge = True
                elif content_len < self.MIN_CHUNK_SIZE and same_content_type:
                    # Next chunk very small and same content type - absorb it
                    should_merge = True

            if should_merge:
                # Merge with current
                current["content"] = current["content"] + "\n\n" + chunk["content"]
                # Keep the first header if present, or update if new one is better
                if not current.get("header") and chunk.get("header"):
                    current["header"] = chunk["header"]
                # Keep section type of the larger portion
                if content_len > current_len:
                    current["section_type"] = chunk.get("section_type")
            else:
                # Save current and start new
                merged.append(current)
                current = chunk.copy()

        # Don't forget the last chunk
        if current is not None:
            merged.append(current)

        return merged

    def _add_contextual_headers(self, chunks: List[Dict]) -> List[Dict]:
        """Add compact contextual headers to chunks for better retrieval.

        Research shows that prepending context improves retrieval accuracy.
        Using shorter format to save space: [Benefits|P3] instead of verbose format
        """
        contextualized = []

        for chunk in chunks:
            content = chunk.get("content", "")
            section_type = chunk.get("section_type", "general")
            page = chunk.get("page")

            # Build compact context prefix
            section_short = section_type.replace("_", " ").title()

            # Compact format: [Benefits|P3]
            if page:
                context_line = f"[{section_short}|P{page}] "
            else:
                context_line = f"[{section_short}] "

            chunk["content"] = context_line + content

            contextualized.append(chunk)

        return contextualized

    def _is_table_content(self, content: str) -> bool:
        """Detect if content contains table data."""
        for pattern in self.TABLE_INDICATORS:
            if re.search(pattern, content):
                # Additional check: multiple rows with similar structure
                lines = content.strip().split('\n')
                if len(lines) >= 3:
                    return True

        # Check for aligned columns (common in PDF table extraction)
        lines = content.strip().split('\n')
        if len(lines) >= 3:
            # Count lines with multiple space-separated values
            tabular_lines = sum(1 for line in lines if len(re.findall(r'\s{2,}', line)) >= 2)
            if tabular_lines >= len(lines) * 0.5:
                return True

        return False

    def _chunk_table(self, content: str, section: Dict) -> List[Dict]:
        """Chunk table content while keeping rows together.

        Tables use larger chunk size (MAX_TABLE_CHUNK) to preserve integrity.
        Research shows tables should stay together when possible.
        """
        chunks = []
        lines = content.split('\n')

        # Find table header (first non-empty line with structure)
        header_line = None
        for i, line in enumerate(lines):
            if re.search(r'\s{2,}', line) or '\t' in line:
                header_line = line
                break

        current_chunk_lines = []
        current_size = 0

        # Use larger size limit for tables
        table_chunk_size = self.MAX_TABLE_CHUNK

        for line in lines:
            line_size = len(line) + 1  # +1 for newline

            # If adding this line would exceed limit, save current chunk
            if current_size + line_size > table_chunk_size and current_chunk_lines:
                chunk_content = '\n'.join(current_chunk_lines)
                chunks.append({
                    **section,
                    "content": chunk_content,
                    "content_type": "Table"
                })

                # Start new chunk with header for context
                if header_line and header_line not in current_chunk_lines[:2]:
                    current_chunk_lines = [f"(Continued from {section.get('header', 'table')})", header_line]
                    current_size = len(header_line) + 50
                else:
                    current_chunk_lines = []
                    current_size = 0

            current_chunk_lines.append(line)
            current_size += line_size

        # Save remaining content
        if current_chunk_lines:
            chunks.append({
                **section,
                "content": '\n'.join(current_chunk_lines),
                "content_type": "Table"
            })

        return chunks

    def _split_large_section(self, content: str, section: Dict) -> List[Dict]:
        """Split a large section while preserving context."""
        chunks = []

        # Use recursive splitter for text content
        text_chunks = self.recursive_splitter.split_text(content)

        for i, chunk_text in enumerate(text_chunks):
            # Add section context to non-first chunks
            if i > 0 and section.get("header"):
                context_prefix = f"(Continued: {section['header']})\n"
                chunk_text = context_prefix + chunk_text

            chunks.append({
                **section,
                "content": chunk_text,
                "content_type": self._detect_content_type(chunk_text)
            })

        return chunks

    def _detect_content_type(self, content: str) -> str:
        """Detect the type of content in a chunk."""
        content_stripped = content.strip()

        # Check for table
        if self._is_table_content(content):
            return "Table"

        # Check for list
        list_patterns = [
            r"^\s*[-•*]\s",  # Bullet points
            r"^\s*\d+[\.\)]\s",  # Numbered list
            r"^\s*[a-z][\.\)]\s",  # Lettered list
            r"^\s*[ivxIVX]+[\.\)]\s",  # Roman numerals
        ]

        lines = content_stripped.split('\n')
        list_lines = sum(1 for line in lines if any(re.match(p, line) for p in list_patterns))

        if list_lines >= len(lines) * 0.3 and list_lines >= 2:
            return "List"

        return "Paragraph"

    def _enrich_chunks(self, chunks: List[Dict]) -> List[EnrichedChunk]:
        """Add additional metadata to chunks."""
        enriched = []

        for chunk in chunks:
            content = chunk.get("content", "")

            # Financial data detection
            has_financial = bool(re.search(r"(?:Rs\.?|₹)\s*[\d,]+|premium|sum\s*assured", content, re.I))

            # Age criteria detection
            has_age = bool(re.search(r"\d+\s*(?:years?|yrs?)|age\s*(?:at|of)|entry\s*age|maturity\s*age", content, re.I))

            enriched.append(EnrichedChunk(
                text=content,
                section_type=chunk.get("section_type", "general"),
                content_type=chunk.get("content_type", "Paragraph"),
                section_header=chunk.get("header"),
                page_number=chunk.get("page"),
                has_financial_data=has_financial,
                has_age_criteria=has_age
            ))

        return enriched


class SemanticChunker(Chunker):
    def __init__(self, embedder: Embedder):
        # We need to ensure the embedder is compatible with LangChain
        # If it's our OpenAIEmbedder, it has a .client which is a LangChain object
        if hasattr(embedder, 'client') and hasattr(embedder.client, 'embed_documents'):
            self.embeddings = embedder.client
        else:
            # Duck typing: our Embedder interface matches LangChain's
            self.embeddings = embedder

        self.splitter = LCSemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            # Increased to 95 (default) to catch FEWER breakpoints.
            # This helps keep "Header" + "Content" together instead of splitting them.
            breakpoint_threshold_amount=95
        )

    def chunk(self, text: str) -> List[str]:
        """
        Hybrid Chunking:
        1. Semantic Search (Primary): Find topic breaks.
        2. Recursive Fallback (Secondary): Force split chunks > 2000 chars.
        """
        try:
            # 1. Primary Pass: Semantic Split
            docs = self.splitter.create_documents([text])
            initial_chunks = [d.page_content for d in docs]

            final_chunks = []
            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500, # Target size for fallback
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""]
            )

            # 2. Safety Check: Process each semantic chunk
            for chunk in initial_chunks:
                # If chunk is too big (Mega-Chunk), force split it
                if len(chunk) > 2000:
                    sub_chunks = recursive_splitter.split_text(chunk)
                    final_chunks.extend(sub_chunks)
                # If chunk is too small (Noise), skip or merge?
                # For now, let's keep it unless it's extremely small (<30 chars)
                elif len(chunk) < 30:
                    continue
                else:
                    final_chunks.append(chunk)

            return final_chunks

        except Exception as e:
            print(f"Semantic chunking failed: {e}. Falling back to recursive.")
            # Fallback
            recursive = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            return recursive.split_text(text)


class HybridLayoutSemanticChunker(Chunker):
    """
    Combines layout awareness with semantic chunking for optimal results.

    Optimized for LIC insurance documents based on 2025 RAG best practices:
    - Default chunk size: 1800 chars (~450 tokens)
    - Tables preserved up to 3000 chars
    - Small sections merged automatically
    - Contextual headers prepended for better retrieval

    Strategy:
    1. First pass: Layout-aware section detection
    2. Second pass: Merge small sections, add context headers
    3. Third pass: Semantic refinement for large paragraphs only
    """

    def __init__(self, embedder: Embedder, chunk_size: int = 1800):
        self.layout_chunker = LayoutAwareChunker(chunk_size=chunk_size)
        self.embedder = embedder
        self.chunk_size = chunk_size

        # Initialize semantic chunker
        if hasattr(embedder, 'client') and hasattr(embedder.client, 'embed_documents'):
            self.embeddings = embedder.client
        else:
            self.embeddings = embedder

    def chunk(self, text: str) -> List[str]:
        """Hybrid chunking combining layout and semantic approaches."""
        return [c.text for c in self.chunk_with_metadata(text)]

    def chunk_with_metadata(self, text: str) -> List[EnrichedChunk]:
        """
        Hybrid chunking with full metadata.

        Process:
        1. Layout-aware chunking first (section detection, table preservation)
        2. For large paragraph sections, apply semantic refinement
        """
        # Get layout-aware chunks
        layout_chunks = self.layout_chunker.chunk_with_metadata(text)

        final_chunks = []

        for chunk in layout_chunks:
            # Tables and lists: keep as-is (layout is more important)
            if chunk.content_type in ("Table", "List"):
                final_chunks.append(chunk)
            # Large paragraphs: consider semantic splitting
            elif len(chunk.text) > self.chunk_size * 1.3:
                # Apply semantic refinement
                semantic_sub = self._semantic_refine(chunk)
                final_chunks.extend(semantic_sub)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _semantic_refine(self, chunk: EnrichedChunk) -> List[EnrichedChunk]:
        """Apply semantic chunking to a large paragraph chunk."""
        try:
            semantic_splitter = LCSemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=90  # More aggressive for refinement
            )

            docs = semantic_splitter.create_documents([chunk.text])
            sub_texts = [d.page_content for d in docs]

            # Create enriched chunks preserving original metadata
            result = []
            for i, text in enumerate(sub_texts):
                if len(text.strip()) < 30:
                    continue

                # Add context for continuation
                if i > 0 and chunk.section_header:
                    text = f"(Continued: {chunk.section_header})\n{text}"

                result.append(EnrichedChunk(
                    text=text,
                    section_type=chunk.section_type,
                    content_type=chunk.content_type,
                    section_header=chunk.section_header,
                    page_number=chunk.page_number,
                    has_financial_data=chunk.has_financial_data or "₹" in text or "Rs." in text,
                    has_age_criteria=chunk.has_age_criteria or bool(re.search(r"\d+\s*years?", text, re.I))
                ))

            return result if result else [chunk]

        except Exception as e:
            print(f"Semantic refinement failed: {e}")
            return [chunk]
