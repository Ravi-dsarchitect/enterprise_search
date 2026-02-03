from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ingestion.service import IngestionService

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    service = IngestionService()
    try:
        result = await service.process_file(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk")
async def bulk_ingest(directory_path: str, limit: int = None):
    """
    Trigger bulk ingestion of a directory.
    - directory_path: Absolute path to the directory (e.g. C:\docs\...)
    - limit: (Optional) Max number of files to process. Useful for testing.
    """
    from app.services.ingestion.bulk_service import BulkIngestionService
    
    service = IngestionService()
    bulk_service = BulkIngestionService(service)
    
    try:
        # Running synchronously for now as per plan, but parallel inside
        results = await bulk_service.ingest_directory(directory_path, limit=limit)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
