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
    SemanticChunker,
    LayoutAwareChunker,
    HybridLayoutSemanticChunker,
    EnrichedChunk
)
from app.services.ingestion.metadata import MetadataExtractor, get_section_display_name


class IngestionService:
    def __init__(self, use_layout_aware: bool = True, use_llm_metadata: bool = True):
        self.qdrant: QdrantClient = get_qdrant_client()

        # Use cached embedder (loaded once, reused across requests)
        self.embedder: Embedder = get_cached_embedder()

        # Use Layout-Aware Chunker for LIC documents (recommended)
        # Falls back to Semantic Chunker if disabled
        self.use_layout_aware = use_layout_aware
        if use_layout_aware:
            # HybridLayoutSemanticChunker combines layout awareness with semantic refinement
            # Optimized chunk size: 1800 chars (~450 tokens) based on RAG best practices
            self.chunker = HybridLayoutSemanticChunker(embedder=self.embedder, chunk_size=1800)
        else:
            self.chunker: Chunker = SemanticChunker(embedder=self.embedder)

        # LLM-based metadata extraction (optional - can be disabled for local testing)
        self.use_llm_metadata = use_llm_metadata
        self.metadata_extractor = None
        if use_llm_metadata:
            try:
                self.metadata_extractor = MetadataExtractor()
            except Exception as e:
                print(f"[WARN] LLM metadata extraction disabled: {e}")
                self.use_llm_metadata = False

    async def process_file(self, file: UploadFile):
        # 1. Save file temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        try:
            return await self.process_local_file(file_path, original_filename=file.filename)
        finally:
            # Cleanup temp file
            if os.path.exists(file_path):
                os.remove(file_path)

    async def process_local_file(self, file_path: str, original_filename: str = None, additional_metadata: dict = None):
        """
        Process a file that already exists locally (e.g., from bulk ingestion).
        Uses layout-aware chunking for better LIC document handling.
        """
        filename = original_filename or os.path.basename(file_path)

        # 1. Parse Document
        parser = DocumentParserFactory.get_parser(file_path)
        text = parser.parse(file_path)

        # 2. Extract Metadata (LLM-based or rule-based fallback)
        if self.use_llm_metadata and self.metadata_extractor:
            metadata = self.metadata_extractor.extract_metadata(text, filename)
        else:
            # Rule-based metadata extraction (no LLM needed)
            metadata = self._extract_basic_metadata(text, filename)

        # Merge additional metadata (e.g., from folder structure)
        if additional_metadata:
            metadata.update(additional_metadata)

        # 3. Chunk Text with Layout Awareness
        # Use chunk_with_metadata for enriched chunks if available
        if self.use_layout_aware and hasattr(self.chunker, 'chunk_with_metadata'):
            enriched_chunks: List[EnrichedChunk] = self.chunker.chunk_with_metadata(text)
            chunk_texts = [c.text for c in enriched_chunks]
        else:
            chunk_texts = self.chunker.chunk(text)
            enriched_chunks = None

        # 4. Generate Embeddings
        embeddings = self.embedder.embed_documents(chunk_texts)

        # 5. Index in Qdrant with Rich Metadata
        points = []
        chunk_ids = [str(uuid.uuid4()) for _ in chunk_texts]

        for i, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings)):
            # Convert date fields to timestamps for numeric range queries
            processed_metadata = self._convert_dates_to_timestamps(metadata)

            # Get enriched chunk data if available
            if enriched_chunks:
                enriched = enriched_chunks[i]
                section_type = enriched.section_type
                content_type = enriched.content_type
                section_header = enriched.section_header
                page_number = enriched.page_number

                # Get additional chunk-level metadata
                chunk_meta = self.metadata_extractor.enrich_chunk_metadata(
                    chunk_text=chunk_text,
                    section_type=section_type,
                    content_type=content_type,
                    doc_metadata=metadata
                )

                entity_hints = chunk_meta.get("entity_hints", [])
                chunk_tags = chunk_meta.get("chunk_tags", [])
            else:
                # Fallback to basic enrichment
                section_type = "general"
                content_type = self._detect_content_type_basic(chunk_text)
                section_header = None
                page_number = None
                entity_hints = []
                chunk_tags = []

                # Basic entity detection
                if any(x in chunk_text for x in ["₹", "Rs.", "Rupees", "Premium"]):
                    entity_hints.append("Financial")
                if any(x in chunk_text for x in ["Year", "Age", "Maturity", "Deadline"]):
                    entity_hints.append("Time/Duration")

            # Generate chunk title from content
            chunk_title = self._extract_chunk_title(chunk_text, section_header, i)

            # Build payload with enriched metadata
            payload = {
                "source": filename,
                "text": chunk_text,
                "chunk_index": i,

                # Layout-aware fields
                "section_type": section_type,
                "section_display": get_section_display_name(section_type),
                "content_type": content_type,
                "section_header": section_header,
                "page_number": page_number,

                # Chunk metadata
                "chunk_title": chunk_title,
                "chunk_char_count": len(chunk_text),
                "entity_hints": entity_hints,
                "chunk_tags": chunk_tags,

                # Contextual retrieval pointers
                "prev_chunk_id": chunk_ids[i-1] if i > 0 else None,
                "next_chunk_id": chunk_ids[i+1] if i < len(chunk_texts)-1 else None,

                # Document-level metadata (flattened)
                **processed_metadata
            }

            points.append(models.PointStruct(
                id=chunk_ids[i],
                vector=embedding,
                payload=payload
            ))

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
            "chunking_method": "layout_aware" if self.use_layout_aware else "semantic"
        }

    def _detect_content_type_basic(self, text: str) -> str:
        """Basic content type detection fallback."""
        text_stripped = text.strip()
        if text_stripped.startswith(("- ", "* ", "1. ", "• ")):
            return "List"
        if "   " in text and "\n" in text:
            return "Table"
        return "Paragraph"

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

    def _extract_chunk_title(self, chunk_text: str, section_header: str = None, index: int = 0) -> str:
        """Extract a meaningful title for the chunk."""
        if section_header:
            return section_header[:80]

        # Use the first significant line as the title
        lines = [l.strip() for l in chunk_text.split('\n') if l.strip() and len(l.strip()) > 3]
        if lines:
            # Clean up markdown headers/bullets from title
            title = lines[0].lstrip("#*-• ").strip()
            if len(title) > 80:
                title = title[:77] + "..."
            return title

        return f"Section {index + 1}"
    
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
