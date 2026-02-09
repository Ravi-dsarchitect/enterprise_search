import os
import uuid
from typing import List
from datetime import datetime
from fastapi import UploadFile
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import settings
from app.core.database import get_qdrant_client, rebuild_bm25_index
from app.core.model_cache import get_cached_embedder
from app.services.ingestion.interfaces import DocumentParser, Chunker, Embedder
from app.services.ingestion.parsers import DocumentParserFactory
from app.services.ingestion.chunkers import (
    RecursiveChunker,
    StructuredChunker,
    EnrichedChunk,
    classify_section,
)
from app.services.ingestion.interfaces import StructuredChunk
from app.services.ingestion.metadata import MetadataExtractor, get_section_display_name, enrich_chunk_metadata
from app.services.ingestion.smart_chunking import SmartChunker


class IngestionService:
    def __init__(self, use_llm_metadata: bool = True, use_smart_chunking: bool = True):
        self.qdrant: QdrantClient = get_qdrant_client()

        # Use cached embedder (loaded once, reused across requests)
        self.embedder: Embedder = get_cached_embedder()

        # Smart chunking: analyze content first, then choose chunker
        self.use_smart_chunking = use_smart_chunking
        if use_smart_chunking:
            self.chunker = SmartChunker(use_llm_classifier=False, verbose=True)
        else:
            self.chunker = StructuredChunker()

        # LLM-based metadata extraction (optional - can be disabled for local testing)
        self.use_llm_metadata = use_llm_metadata
        self.metadata_extractor = None
        if use_llm_metadata:
            try:
                self.metadata_extractor = MetadataExtractor()
            except Exception as e:
                print(f"[WARN] LLM metadata extraction disabled: {e}")
                self.use_llm_metadata = False

    async def process_file(self, file: UploadFile, project_ids: List[str] = None):
        # 1. Save file temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        try:
            return await self.process_local_file(file_path, original_filename=file.filename, project_ids=project_ids)
        finally:
            # Cleanup temp file
            if os.path.exists(file_path):
                os.remove(file_path)

    async def process_local_file(self, file_path: str, original_filename: str = None, additional_metadata: dict = None, verbose: bool = False, project_ids: List[str] = None):
        """
        Process a file that already exists locally (e.g., from bulk ingestion).
        Uses StructuredChunker with ParsedDocument for heading-aware chunking.
        
        Args:
            project_ids: List of project IDs this document belongs to (for multi-tenant filtering)
        """
        filename = original_filename or os.path.basename(file_path)
        _p = (lambda msg: print(msg, end="", flush=True)) if verbose else (lambda msg: None)

        # 1. Parse Document (structured: blocks, tables, headings)
        _p("Parse")
        parser = DocumentParserFactory.get_parser(file_path)
        parsed_doc = parser.parse_structured(file_path)
        text = parsed_doc.full_text

        # 2. Extract Metadata (LLM-based or rule-based fallback)
        _p(" > Meta")
        tables_preview = "\n\n".join(t.markdown for t in parsed_doc.tables if t.markdown)

        if self.use_llm_metadata and self.metadata_extractor:
            metadata = self.metadata_extractor.extract_metadata(
                text, filename,
                headings=parsed_doc.headings,
                tables_preview=tables_preview or None,
            )
        else:
            metadata = self._extract_basic_metadata(text, filename)

        # Post-process: fix plan_name for generic docs (claims guides, grievance, etc.)
        metadata = self._fix_generic_doc_metadata(metadata, filename)

        # Merge additional metadata (e.g., from folder structure)
        if additional_metadata:
            metadata.update(additional_metadata)

        # 3. Chunk using StructuredChunker with ParsedDocument
        _p(" > Chunk")
        structured_chunks: List[StructuredChunk] = self.chunker.chunk_structured(
            parsed_doc, doc_metadata=metadata
        )
        chunk_texts = [c.text for c in structured_chunks]
        _p(f"({len(chunk_texts)})")

        # 4. Generate Embeddings
        _p(" > Embed")
        embeddings = self.embedder.embed_documents(chunk_texts)

        # 5. Index in Qdrant with Rich Metadata
        points = []
        chunk_ids = [str(uuid.uuid4()) for _ in chunk_texts]

        for i, (chunk, embedding) in enumerate(zip(structured_chunks, embeddings)):
            # Convert date fields to timestamps for numeric range queries
            processed_metadata = self._convert_dates_to_timestamps(metadata)

            section_type = chunk.section_type
            content_type = chunk.content_type
            section_header = chunk.heading
            page_number = chunk.page_number

            # Get additional chunk-level metadata (pattern-based, no LLM needed)
            chunk_meta = enrich_chunk_metadata(
                chunk_text=chunk.text,
                section_type=section_type,
                content_type=content_type,
                doc_metadata=metadata,
            )

            entity_hints = chunk_meta.get("entity_hints", [])
            chunk_tags = chunk_meta.get("chunk_tags", [])

            # Generate dynamic chunk title
            chunk_meta_for_title = {
                "chunk_tags": chunk_tags,
                "section_type": section_type,
                "entity_hints": entity_hints,
            }
            chunk_title = self._extract_chunk_title(
                chunk.text, section_header, i,
                chunk_meta=chunk_meta_for_title,
                doc_metadata=metadata,
            )

            # Build payload with enriched metadata
            payload = {
                "source": filename,
                "source_file": filename,
                "text": chunk.text,
                "chunk_index": i,

                # Structure-aware fields
                "section_type": section_type,
                "section_display": get_section_display_name(section_type),
                "content_type": content_type,
                "section_header": section_header,
                "page_number": page_number,

                # Chunk metadata
                "chunk_title": chunk_title,
                "chunk_char_count": len(chunk.text),
                "entity_hints": entity_hints,
                "chunk_tags": chunk_tags,

                # Boolean content flags (for filtering)
                "contains_age_info": chunk_meta.get("contains_age_info", False),
                "contains_currency": chunk_meta.get("contains_currency", False),
                "contains_numbers": chunk_meta.get("contains_numbers", False),
                "contains_date_info": chunk_meta.get("contains_date_info", False),

                # Contextual retrieval pointers
                "prev_chunk_id": chunk_ids[i-1] if i > 0 else None,
                "next_chunk_id": chunk_ids[i+1] if i < len(chunk_texts)-1 else None,

                # Multi-tenant project filtering
                "project_ids": project_ids or [],

                # Document-level metadata (flattened)
                **processed_metadata
            }

            points.append(models.PointStruct(
                id=chunk_ids[i],
                vector=embedding,
                payload=payload
            ))

        _p(" > Index")
        self.qdrant.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=points
        )

        # 6. Return summary
        return {
            "filename": filename,
            "chunks": len(chunk_texts),
            "status": "indexed",
            "extracted_metadata": metadata,
            "chunking_method": "structured",
        }

    # Filename patterns that indicate a generic/non-plan-specific document
    _GENERIC_DOC_PATTERNS = {
        r"claim": "Claim Guide",
        r"grievance|complaint|redressal": "Grievance & Support",
        r"faq|frequently": "FAQ",
        r"circular|notification": "Circular",
        r"manual|sop|procedure": "Manual",
        r"report|annual": "Report",
        r"endowment_plan_types|moneyback_plan_types|term_insurance_plan_types|whole_life_plan_type": "Plan Comparison",
    }

    def _fix_generic_doc_metadata(self, metadata: dict, filename: str) -> dict:
        """
        Post-process metadata to fix plan_name for generic documents.
        The LLM may pick up a plan name referenced in the content, but the
        document itself is not about that specific plan.
        """
        import re
        fn_lower = filename.lower()
        for pattern, category in self._GENERIC_DOC_PATTERNS.items():
            if re.search(pattern, fn_lower):
                # Override plan_name to the document's actual subject
                # Use the filename-derived title, not a plan mentioned in content
                doc_title = os.path.splitext(filename)[0]
                # Clean up underscores/hyphens for readability
                doc_title = doc_title.replace("_", " ").replace("-", " ").strip()
                metadata["plan_name"] = doc_title
                metadata["category"] = category
                break
        return metadata

    def _extract_basic_metadata(self, text: str, filename: str) -> dict:
        """
        Rule-based metadata extraction (no LLM needed).
        Extracts plan name, type, and keywords from filename and content patterns.
        """
        import re

        metadata = {
            "title": filename,
            "category": "Policy",
            "keywords": [],
            "plan_name": None,
            "plan_type": None,
        }

        text_lower = text.lower()
        filename_lower = filename.lower()

        # Extract plan name from filename patterns
        plan_patterns = [
            r"jeevan\s*(?:utsav|labh|akshay|shanti|umang|tarun|anand)",
            r"bima\s*(?:ratna|shree|kavach|bachat|jyoti)",
            r"nivesh\s*plus",
            r"amritbaal",
            r"index\s*plus",
            r"smart\s*pension",
            r"saral\s*(?:pension|jeevan)",
        ]

        for pattern in plan_patterns:
            match = re.search(pattern, filename_lower) or re.search(pattern, text_lower[:2000])
            if match:
                metadata["plan_name"] = match.group().replace("  ", " ").title()
                break

        # Detect plan type
        plan_type_patterns = {
            "Whole Life": r"whole\s*life|lifelong",
            "Endowment": r"endowment|savings\s*plan",
            "Money Back": r"money\s*back|survival\s*benefit",
            "Term Insurance": r"term\s*(?:insurance|assurance|plan)",
            "Pension/Annuity": r"pension|annuity|retirement",
            "ULIP": r"ulip|unit\s*linked|market\s*linked|nav",
            "Child Plan": r"child|amritbaal|minor",
        }

        for plan_type, pattern in plan_type_patterns.items():
            if re.search(pattern, text_lower[:5000]):
                metadata["plan_type"] = plan_type
                break

        # Extract keywords from content
        keyword_patterns = [
            "death benefit", "maturity benefit", "survival benefit",
            "guaranteed addition", "loyalty addition", "bonus",
            "premium waiver", "tax benefit", "loan facility",
            "surrender value", "paid-up", "revival",
        ]

        for kw in keyword_patterns:
            if kw in text_lower:
                metadata["keywords"].append(kw.title())

        # Limit keywords
        metadata["keywords"] = metadata["keywords"][:7]

        # Detect category from filename
        if "brochure" in filename_lower or "sales" in filename_lower:
            metadata["category"] = "Sales Brochure"
        elif "claim" in filename_lower:
            metadata["category"] = "Claim Guide"
        elif "grievance" in filename_lower:
            metadata["category"] = "Support"

        return metadata

    def _extract_chunk_title(
        self,
        chunk_text: str,
        section_header: str = None,
        index: int = 0,
        chunk_meta: dict = None,
        doc_metadata: dict = None,
    ) -> str:
        """
        Generate a dynamic, descriptive title for the chunk using available metadata.

        Priority:
        1. Section header + plan context (e.g., "Jeevan Utsav - Death Benefit")
        2. Top chunk tag + section type (e.g., "Eligibility Criteria - Age Requirements")
        3. First significant line from text
        4. Fallback generic title
        """
        plan_name = (doc_metadata or {}).get("plan_name", "")
        tags = (chunk_meta or {}).get("chunk_tags", [])
        section_type = (chunk_meta or {}).get("section_type", "general")
        entity_hints = (chunk_meta or {}).get("entity_hints", [])

        title_parts = []

        # Use section header if available
        if section_header:
            base_title = section_header.strip()[:60]
        elif tags:
            # Use the most specific tag as title basis
            base_title = tags[0]
        else:
            # Extract from first significant line of text
            lines = [l.strip() for l in chunk_text.split('\n') if l.strip() and len(l.strip()) > 3]
            if lines:
                base_title = lines[0].lstrip("#*-• ").strip()[:60]
            else:
                base_title = None

        if not base_title:
            # Final fallback
            if section_type != "general":
                return get_section_display_name(section_type)
            return f"Section {index + 1}"

        title_parts.append(base_title)

        # Add plan context if not already in title
        if plan_name and plan_name.lower() not in base_title.lower():
            title_parts.insert(0, plan_name)

        # Add entity hint qualifier for specificity
        if entity_hints and len(" - ".join(title_parts)) < 60:
            hint = entity_hints[0]
            if hint not in base_title:
                title_parts.append(f"({hint})")

        title = " - ".join(title_parts)
        if len(title) > 100:
            title = title[:97] + "..."
        return title
    
    def _convert_dates_to_timestamps(self, metadata: dict) -> dict:
        """
        Convert date string fields to Unix timestamps for numeric range queries.
        
        Args:
            metadata: Metadata dictionary potentially containing date fields
            
        Returns:
            New metadata dictionary with dates converted to timestamps
        """
        # Fields that should be converted to timestamps
        date_fields = {"document_date", "created_at", "modified_at", "date"}
        
        processed = {}
        for key, value in metadata.items():
            if key in date_fields and isinstance(value, str) and value:
                try:
                    # Parse ISO date format (YYYY-MM-DD) to timestamp
                    dt = datetime.fromisoformat(value)
                    processed[key] = dt.timestamp()
                except (ValueError, TypeError):
                    # If conversion fails, keep the original value
                    processed[key] = value
            else:
                processed[key] = value
        
        return processed
