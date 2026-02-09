"""Bulk ingestion script. Supports local directories and S3 buckets."""
import asyncio
import sys

from app.services.ingestion.service import IngestionService
from app.services.ingestion.bulk_service import BulkIngestionService


async def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/ec2-user/nakul/enterprise_search/docs"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    project_ids = sys.argv[3].split(",") if len(sys.argv) > 3 else None

    service = IngestionService()
    bulk = BulkIngestionService(service)

    if path.startswith("s3://"):
        results = await bulk.ingest_s3_folder(path, limit=limit, project_ids=project_ids)
    else:
        results = await bulk.ingest_directory(path, limit=limit, project_ids=project_ids)

    print(f"\nTotal: {results.get('total', 0)}, Success: {results.get('successful', 0)}, Failed: {results.get('failed', 0)}")


if __name__ == "__main__":
    asyncio.run(main())
