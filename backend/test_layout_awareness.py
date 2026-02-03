import asyncio
from app.services.ingestion.parsers import DocumentParserFactory
from app.services.ingestion.chunkers import (
    SemanticChunker,
    LayoutAwareChunker,
    HybridLayoutSemanticChunker
)
from app.services.ingestion.metadata import MetadataExtractor, get_section_display_name
from app.core.model_cache import get_cached_embedder


async def verify_layout_awareness():
    # Pick a specific file to test
    file_path = r"C:\Users\NakulSaiAdapala\Downloads\enterprise_search_v0.3\docs\OneDrive_1_2-2-2026\102268- Jeevan Utsav Sales Brochure.pdf"

    print(f"{'='*60}")
    print(f"LAYOUT-AWARE CHUNKING TEST")
    print(f"{'='*60}")
    print(f"File: {file_path}")

    # 1. Parse using Layout Mode
    print("\n[1] Parsing with Layout Mode...")
    parser = DocumentParserFactory.get_parser(file_path)
    text = parser.parse(file_path)

    print(f"    Total characters extracted: {len(text)}")
    print("\n--- RAW EXTRACTED TEXT (First 800 chars) ---")
    print(text[:800])
    print("-" * 50)

    # 2. Initialize chunkers
    print("\n[2] Initializing Chunkers...")
    embedder = get_cached_embedder()

    # OLD: Semantic Chunker
    old_chunker = SemanticChunker(embedder)

    # NEW: Layout-Aware Chunker (optimized: 1800 chars)
    layout_chunker = LayoutAwareChunker(chunk_size=1800)

    # NEW: Hybrid (Layout + Semantic) - optimized for LIC docs
    hybrid_chunker = HybridLayoutSemanticChunker(embedder, chunk_size=1800)

    # 3. Compare chunking results
    print("\n[3] Comparing Chunking Strategies...")

    # Old semantic chunks
    print("\n--- OLD: Semantic Chunker ---")
    old_chunks = old_chunker.chunk(text)
    print(f"    Chunks generated: {len(old_chunks)}")
    print(f"    Avg chunk size: {sum(len(c) for c in old_chunks) // len(old_chunks)} chars")

    # New layout-aware chunks
    print("\n--- NEW: Layout-Aware Chunker ---")
    layout_chunks = layout_chunker.chunk_with_metadata(text)
    print(f"    Chunks generated: {len(layout_chunks)}")
    print(f"    Avg chunk size: {sum(len(c.text) for c in layout_chunks) // len(layout_chunks)} chars")

    # Hybrid chunks (uses semantic refinement for large paragraphs)
    print("\n--- NEW: Hybrid (Layout + Semantic) Chunker ---")
    hybrid_chunks = hybrid_chunker.chunk_with_metadata(text)
    print(f"    Chunks generated: {len(hybrid_chunks)}")
    print(f"    Avg chunk size: {sum(len(c.text) for c in hybrid_chunks) // len(hybrid_chunks)} chars")

    # Section distribution
    section_counts = {}
    for chunk in layout_chunks:
        section_counts[chunk.section_type] = section_counts.get(chunk.section_type, 0) + 1
    print(f"\n    Section Distribution:")
    for section, count in sorted(section_counts.items(), key=lambda x: -x[1]):
        print(f"      - {get_section_display_name(section)}: {count} chunks")

    # Content type distribution
    type_counts = {}
    for chunk in layout_chunks:
        type_counts[chunk.content_type] = type_counts.get(chunk.content_type, 0) + 1
    print(f"\n    Content Type Distribution:")
    for ctype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"      - {ctype}: {count} chunks")

    # 4. Show sample enriched chunks
    print("\n" + "=" * 60)
    print("SAMPLE ENRICHED CHUNKS (First 5)")
    print("=" * 60)

    metadata_extractor = MetadataExtractor()

    for i, chunk in enumerate(layout_chunks[:5]):
        # Get chunk-level metadata
        chunk_meta = metadata_extractor.enrich_chunk_metadata(
            chunk_text=chunk.text,
            section_type=chunk.section_type,
            content_type=chunk.content_type
        )

        print(f"\n{'*'*50}")
        print(f"CHUNK {i+1}")
        print(f"{'*'*50}")
        print(f"  Section:      {get_section_display_name(chunk.section_type)}")
        print(f"  Content Type: {chunk.content_type}")
        print(f"  Page:         {chunk.page_number}")
        print(f"  Header:       {chunk.section_header or '(none)'}")
        print(f"  Char Count:   {len(chunk.text)}")
        print(f"  Chunk Tags:   {chunk_meta.get('chunk_tags', [])}")
        print(f"  Entity Hints: {chunk_meta.get('entity_hints', [])}")
        print(f"  Has Currency: {chunk_meta.get('contains_currency', False)}")
        print(f"  Has Age Info: {chunk_meta.get('contains_age_info', False)}")

        # Show text preview
        preview = chunk.text[:300].replace('\n', ' ')
        if len(chunk.text) > 300:
            preview += "..."
        print(f"\n  Text Preview:\n  {'-'*40}")
        print(f"  {preview}")
        print(f"  {'-'*40}")

    # 5. Show comparison for a benefits section
    print("\n" + "=" * 60)
    print("BENEFITS SECTION CHUNKS")
    print("=" * 60)

    benefit_chunks = [c for c in layout_chunks if c.section_type == "benefits"]
    print(f"\nFound {len(benefit_chunks)} chunks in Benefits section:")

    for i, chunk in enumerate(benefit_chunks[:3]):
        print(f"\n  [{i+1}] {chunk.content_type} ({len(chunk.text)} chars)")
        preview = chunk.text[:200].replace('\n', ' ')
        print(f"      {preview}...")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(verify_layout_awareness())
