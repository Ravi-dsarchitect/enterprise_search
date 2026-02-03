try:
    print("Checking imports...")
    from app.services.ingestion.bulk_service import BulkIngestionService
    print("✓ BulkIngestionService imported")
    from app.services.ingestion.chunkers import SemanticChunker
    print("✓ SemanticChunker imported")
    from app.services.ingestion.metadata import MetadataExtractor
    print("✓ MetadataExtractor imported")
    from app.services.rag.query_analyzer import QueryAnalyzer
    print("✓ QueryAnalyzer imported")
    from app.api.v1.endpoints.ingestion import bulk_ingest
    print("✓ API Endpoint imported")
    print("All checks passed.")
except Exception as e:
    print(f"Import failed: {e}")
