import os
import asyncio
import logging
import time
import tempfile
import shutil
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from app.services.ingestion.service import IngestionService
from app.services.ingestion.s3_service import S3Service

logger = logging.getLogger(__name__)


class BulkIngestionService:
    def __init__(self, ingestion_service: IngestionService, max_concurrent: int = 3):
        self.ingestion_service = ingestion_service
        self.max_concurrent = max_concurrent
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.total_files = 0
        self.processed_files = 0
        self.start_time = None

    async def ingest_directory(self, directory_path: str, limit: int = None, project_ids: List[str] = None) -> Dict[str, Any]:
        if not os.path.isdir(directory_path):
            raise ValueError(f"Invalid directory path: {directory_path}")

        files_to_process = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(('.pdf', '.docx', '.txt')):
                    files_to_process.append(os.path.join(root, file))

        seen_names = set()
        unique_files = []
        for fp in files_to_process:
            fname = os.path.basename(fp)
            if fname not in seen_names:
                seen_names.add(fname)
                unique_files.append(fp)
        files_to_process = unique_files

        if limit:
            files_to_process = files_to_process[:limit]

        self.total_files = len(files_to_process)
        self.processed_files = 0
        self.start_time = time.time()

        print(f"\n{'='*60}\nBULK INGESTION STARTED\n{'='*60}")
        print(f"  Total files: {self.total_files}\n{'='*60}\n")

        results = {"total": self.total_files, "successful": 0, "failed": 0, "details": [], "duration_seconds": 0}
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [self._process_single_file(fp, directory_path, semaphore, project_ids) for fp in files_to_process]
        processed_results = await asyncio.gather(*tasks)

        for res in processed_results:
            if res["status"] == "success":
                results["successful"] += 1
            else:
                results["failed"] += 1
            results["details"].append(res)

        results["duration_seconds"] = round(time.time() - self.start_time, 2)
        print(f"\n{'='*60}\nBULK INGESTION COMPLETE\n  Successful: {results['successful']}/{self.total_files}\n{'='*60}\n")
        return results

    async def _process_single_file(self, file_path: str, root_path: str, semaphore: asyncio.Semaphore, project_ids: List[str] = None) -> Dict[str, Any]:
        async with semaphore:
            try:
                rel_path = os.path.relpath(file_path, root_path)
                folder_name = os.path.dirname(rel_path)
                category = folder_name.replace(os.sep, " > ") if folder_name else "General"
                result = await self.ingestion_service.process_local_file(
                    file_path=file_path,
                    additional_metadata={"source_directory": root_path, "category": category},
                    project_ids=project_ids
                )
                self.processed_files += 1
                print(f"  [{self.processed_files}/{self.total_files}] OK: {os.path.basename(file_path)}")
                return {"file": rel_path, "status": "success", "chunks": result.get("chunks", 0)}
            except Exception as e:
                self.processed_files += 1
                print(f"  [{self.processed_files}/{self.total_files}] FAILED: {os.path.basename(file_path)}")
                return {"file": os.path.basename(file_path), "status": "failed", "error": str(e)}

    async def ingest_s3_folder(self, s3_uri: str, limit: int = None, project_ids: List[str] = None) -> Dict[str, Any]:
        s3_service = S3Service()
        bucket, prefix = S3Service.parse_s3_uri(s3_uri)
        print(f"\nS3 BULK INGESTION\n  Bucket: {bucket}\n  Prefix: {prefix}\n")
        
        s3_files = s3_service.list_files(bucket, prefix)
        if not s3_files:
            return {"total": 0, "successful": 0, "failed": 0, "details": []}
        if limit:
            s3_files = s3_files[:limit]
        
        temp_dir = tempfile.mkdtemp(prefix="s3_ingest_")
        try:
            local_files = []
            for i, f in enumerate(s3_files):
                local_path = s3_service.download_file(bucket, f['key'], temp_dir)
                local_files.append(local_path)
                print(f"  [{i+1}/{len(s3_files)}] Downloaded: {os.path.basename(f['key'])}")
            
            results = await self.ingest_directory(temp_dir, limit=None, project_ids=project_ids)
            results["s3_source"] = s3_uri
            return results
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
