"""Debug script to inspect chunks in Qdrant."""
from qdrant_client import QdrantClient
from app.core.config import settings

client = QdrantClient(url=settings.QDRANT_URL)

# Get all chunks for Jeevan Umang
results = client.scroll(
    collection_name=settings.QDRANT_COLLECTION_NAME,
    scroll_filter={
        "must": [
            {"key": "source", "match": {"text": "LIC_Jeevan Umang"}}
        ]
    },
    limit=500,
    with_payload=True,
    with_vectors=False
)

chunks = results[0]
print(f"Found {len(chunks)} chunks for Jeevan Umang\n")

# Group by page
by_page = {}
for chunk in chunks:
    page = chunk.payload.get("page_number", "?")
    if page not in by_page:
        by_page[page] = []
    by_page[page].append(chunk)

# Show summary by page
print("=" * 80)
print("CHUNKS BY PAGE")
print("=" * 80)
for page in sorted(by_page.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
    print(f"\n📄 Page {page}: {len(by_page[page])} chunks")
    for i, chunk in enumerate(by_page[page][:3]):  # Show first 3 per page
        text = chunk.payload.get("text", "")[:100].replace("\n", " ")
        section = chunk.payload.get("section_header", "") or chunk.payload.get("section_type", "")
        meta_keys = list(chunk.payload.keys())
        print(f"   [{i+1}] {section[:30]:30} | {text}...")

# Find eligibility-related chunks
print("\n" + "=" * 80)
print("ELIGIBILITY-RELATED CHUNKS (searching text)")
print("=" * 80)
eligibility_keywords = ["eligibility", "entry age", "minimum age", "maximum age", "sum assured", "policy term"]

for chunk in chunks:
    text = chunk.payload.get("text", "").lower()
    for kw in eligibility_keywords:
        if kw in text:
            page = chunk.payload.get("page_number", "?")
            section = chunk.payload.get("section_header", "") or chunk.payload.get("section_type", "")
            text_preview = chunk.payload.get("text", "")[:200].replace("\n", " ")
            print(f"\n🎯 FOUND '{kw}' on Page {page}")
            print(f"   Section: {section}")
            print(f"   Metadata keys: {list(chunk.payload.keys())}")
            print(f"   Text: {text_preview}...")
            break

# Show metadata of first few chunks
print("\n" + "=" * 80)
print("SAMPLE CHUNK METADATA (first 5)")
print("=" * 80)
for i, chunk in enumerate(chunks[:5]):
    print(f"\n--- Chunk {i+1} ---")
    for key, value in chunk.payload.items():
        if key == "text":
            print(f"  {key}: {str(value)[:80]}...")
        else:
            print(f"  {key}: {value}")
