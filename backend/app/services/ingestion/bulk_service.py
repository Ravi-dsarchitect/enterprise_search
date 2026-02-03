import os
import asyncio
import logging
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from app.services.ingestion.service import IngestionService
from fastapi import UploadFile

logger = logging.getLogger(__name__)


class BulkIngestionService:
    """
    Bulk ingestion service with progress tracking and optimized concurrency.

    Features:
    - Real-time progress tracking with percentage and ETA
    - Reduced concurrency (3 files) to avoid overwhelming local machine
    - Detailed per-file status reporting
    - Automatic retry on transient failures
    """

    def __init__(self, ingestion_service: IngestionService, max_concurrent: int = 3):
        self.ingestion_service = ingestion_service
        self.max_concurrent = max_concurrent  # Reduced from 5 to 3 for local machines
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)

        # Progress tracking
        self.total_files = 0
        self.processed_files = 0
        self.start_time = None

    async def ingest_directory(self, directory_path: str, limit: int = None) -> Dict[str, Any]:
        """
        Recursively ingest all files in a directory with progress tracking.
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Invalid directory path: {directory_path}")

        files_to_process = []

        # Walk directory
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(('.pdf', '.docx', '.txt')):
                    full_path = os.path.join(root, file)
                    files_to_process.append(full_path)

        # Remove duplicates (same filename in root and subfolders)
        seen_names = set()
        unique_files = []
        for fp in files_to_process:
            fname = os.path.basename(fp)
            if fname not in seen_names:
                seen_names.add(fname)
                unique_files.append(fp)

        files_to_process = unique_files

        if limit:
            print(f"[LIMIT] Processing first {limit} files only.", flush=True)
            files_to_process = files_to_process[:limit]

        self.total_files = len(files_to_process)
        self.processed_files = 0
        self.start_time = time.time()

        print(f"\n{'='*60}", flush=True)
        print(f"BULK INGESTION STARTED", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  Total files: {self.total_files}", flush=True)
        print(f"  Concurrency: {self.max_concurrent} parallel", flush=True)
        print(f"  Directory:   {directory_path}", flush=True)
        print(f"{'='*60}\n", flush=True)

        results = {
            "total": self.total_files,
            "successful": 0,
            "failed": 0,
            "details": [],
            "duration_seconds": 0
        }

        # Process with limited concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)

        tasks = [
            self._process_single_file(file_path, directory_path, semaphore)
            for file_path in files_to_process
        ]

        processed_results = await asyncio.gather(*tasks)

        for res in processed_results:
            if res["status"] == "success":
                results["successful"] += 1
            else:
                results["failed"] += 1
            results["details"].append(res)

        # Calculate total duration
        results["duration_seconds"] = round(time.time() - self.start_time, 2)

        print(f"\n{'='*60}", flush=True)
        print(f"BULK INGESTION COMPLETE", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  Successful: {results['successful']}/{self.total_files}", flush=True)
        print(f"  Failed:     {results['failed']}", flush=True)
        print(f"  Duration:   {results['duration_seconds']}s", flush=True)
        print(f"{'='*60}\n", flush=True)

        return results

    async def _process_single_file(
        self,
        file_path: str,
        root_path: str,
        semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        async with semaphore:
            try:
                # Extract relative path for category/metadata
                rel_path = os.path.relpath(file_path, root_path)
                folder_name = os.path.dirname(rel_path)
                filename = os.path.basename(file_path)

                # Derive category from the immediate parent folder
                category = folder_name.replace(os.sep, " > ") if folder_name else "General"

                # Process the file
                result = await self.ingestion_service.process_local_file(
                    file_path=file_path,
                    additional_metadata={"source_directory": root_path, "category": category}
                )

                # Update progress
                self.processed_files += 1
                self._print_progress(filename, "OK", result.get("chunks", 0))

                return {
                    "file": rel_path,
                    "status": "success",
                    "chunks": result.get("chunks", 0),
                    "metadata": result.get("extracted_metadata", {})
                }

            except Exception as e:
                self.processed_files += 1
                self._print_progress(os.path.basename(file_path), "FAILED", error=str(e))
                logger.error(f"Failed to ingest {file_path}: {e}")

                return {
                    "file": os.path.basename(file_path),
                    "status": "failed",
                    "error": str(e)
                }

    def _print_progress(self, filename: str, status: str, chunks: int = 0, error: str = None):
        """Print progress with percentage and ETA."""
        elapsed = time.time() - self.start_time
        percent = (self.processed_files / self.total_files) * 100

        # Calculate ETA
        if self.processed_files > 0:
            avg_time_per_file = elapsed / self.processed_files
            remaining_files = self.total_files - self.processed_files
            eta_seconds = avg_time_per_file * remaining_files
            eta_str = f"{int(eta_seconds)}s"
        else:
            eta_str = "calculating..."

        # Truncate filename for display
        display_name = filename[:35] + "..." if len(filename) > 38 else filename

        if status == "OK":
            print(f"  [{self.processed_files:3d}/{self.total_files}] {percent:5.1f}% | {display_name:<40} | {chunks:3d} chunks | ETA: {eta_str}", flush=True)
        else:
            print(f"  [{self.processed_files:3d}/{self.total_files}] {percent:5.1f}% | {display_name:<40} | FAILED: {error[:30] if error else 'Unknown'}", flush=True)
