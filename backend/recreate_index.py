from app.core.database import get_qdrant_client, init_qdrant
from app.core.config import settings

def recreate_collection():
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION_NAME
    
    print(f"🗑️ Deleting collection: {collection_name}")
    try:
        client.delete_collection(collection_name=collection_name)
        print("✅ Collection deleted.")
    except Exception as e:
        print(f"⚠️ Failed to delete collection (maybe it didn't exist): {e}")
        
    print("🔄 Recreating collection...")
    init_qdrant()
    print("✅ Collection recreated successfully.")

if __name__ == "__main__":
    recreate_collection()
