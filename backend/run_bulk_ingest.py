"""Bulk ingestion script with smart chunking."""
import asyncio
import sys

from app.services.ingestion.service import IngestionService
from app.services.ingestion.bulk_service import BulkIngestionService


async def main():
    # Default directory or use command line arg
    directory = sys.argv[1] if len(sys.argv) > 1 else "/home/ec2-user/nakul/enterprise_search/docs/OneDrive_1_2-2-2026"

    # Optional limit (for testing)
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"Starting bulk ingestion from: {directory}")
    if limit:
        print(f"Limiting to {limit} files")

    service = IngestionService()
    bulk = BulkIngestionService(service)

    results = await bulk.ingest_directory(directory, limit=limit)

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Total:      {results.get('total', 0)}")
    print(f"  Successful: {results.get('successful', 0)}")
    print(f"  Failed:     {results.get('failed', 0)}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
