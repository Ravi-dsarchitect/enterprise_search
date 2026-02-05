"""Test script for smart chunking."""
import sys

from app.services.ingestion.parsers import PDFParser
from app.services.ingestion.smart_chunking import ContentAnalyzer, SmartChunker

# Use command line arg or default path
pdf_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ec2-user/nakul/enterprise_search/docs/OneDrive_1_2-2-2026/LIC_Jeevan Umang_Sales Brochure_4 inch x 9 inch_Eng (1).pdf"

parser = PDFParser()
doc = parser.parse_structured(pdf_path)
print(f"Parsed: {doc.filename}")
print(f"Pages: {doc.total_pages}")

analyzer = ContentAnalyzer()
analysis = analyzer.analyze(doc)
print(f"\n=== Analysis ===")
print(f"Structure: {analysis.primary_structure.value}")
print(f"Secondary: {analysis.secondary_structure}")
print(f"Supports hierarchy: {analysis.supports_hierarchy}")
print(f"Tables: {analysis.metrics.table_count}")
print(f"Headings: {analysis.metrics.heading_count} (depth {analysis.metrics.heading_depth})")
print(f"Confidence: {analysis.confidence:.0%}")
print(f"Recommended: {analysis.recommended_chunker}")

print(f"\n=== Full Chunking ===")
chunker = SmartChunker(verbose=True)
result = chunker.chunk_smart(doc)

print(f"\nChunker used: {result.chunker_used}")
print(f"Total chunks: {len(result.chunks)}")

print(f"\n=== Sample Chunks ===")
for i, c in enumerate(result.chunks[:5]):
    section = c.metadata.get("section_heading", "N/A")[:40] if c.metadata else "N/A"
    text_preview = c.text[:80].replace("\n", " ")
    print(f"{i+1}. [{section}] {text_preview}...")
