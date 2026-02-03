from app.core.database import get_qdrant_client
from app.core.config import settings
import json

def inspect_latest_data(limit: int = 5):
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION_NAME
    
    print(f"🔍 Analyzing top {limit} chunks from collection: {collection_name}")
    
    try:
        response, _ = client.scroll(
            collection_name=collection_name,
            limit=100, # Fetch more to get a better count estimate
            with_payload=True,
            with_vectors=False
        )
        
        if not response:
            print("❌ No data found in collection.")
            return

        # Aggregation: Count chunks per file
        file_counts = {}
        for point in response:
            source = point.payload.get('source', 'Unknown')
            file_counts[source] = file_counts.get(source, 0) + 1
            
        print("\n📊 Chunk Distribution (Sample):")
        for source, count in file_counts.items():
            print(f"   - {source}: {count} chunks found in sample")

        print("\n--- Detailed Inspection (Top 5) ---")
        for i, point in enumerate(response[:5]):
            payload = point.payload
            print(f"\n--- Document {i+1} ---")
            print(f"📄 Source: {payload.get('source', 'Unknown')}")
            print(f"🏷️  Category: {payload.get('category', 'N/A')}")
            
            # Print specific LIC metadata if present
            lic_fields = ["plan_name", "plan_number", "tags", "entry_age"]
            found_fields = {k: payload.get(k) for k in lic_fields if payload.get(k)}
            if found_fields:
                print(f"📋 Metadata: {json.dumps(found_fields, indent=2)}")
            
            # Print Chunk Analytics
            print(f"🧩 Chunk Analytics:")
            print(f"   - Title: {payload.get('chunk_title', 'N/A')}")
            print(f"   - Type: {payload.get('chunk_type', 'Unknown')}")
            print(f"   - Hints: {payload.get('entity_hints', [])}")
            print(f"   - Length: {payload.get('chunk_char_count', 0)} chars")
            print(f"   - Context: [Prev: {bool(payload.get('prev_chunk_id'))}] [Next: {bool(payload.get('next_chunk_id'))}]")
            
            print(f"\n📝 Content (first 200 chars):")
            print(f"\"{payload.get('text', '')[:200]}...\"")
            print("-" * 30)
            
    except Exception as e:
        print(f"Error inspecting data: {e}")

if __name__ == "__main__":
    inspect_latest_data()
