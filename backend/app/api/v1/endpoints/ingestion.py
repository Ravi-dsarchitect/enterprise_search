from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.services.ingestion.service import IngestionService
from app.services.ingestion.s3_service import S3Service

router = APIRouter()


# ==================== Request Models ====================

class S3FileRequest(BaseModel):
    """Request model for ingesting a single document from S3."""
    s3_uri: str = Field(
        ...,
        description="S3 URI to the document file",
        json_schema_extra={"example": "s3://my-bucket/documents/policy.pdf"}
    )
    project_ids: Optional[List[str]] = Field(
        default=None,
        description="List of project IDs for multi-tenant filtering",
        json_schema_extra={"example": ["project_alpha", "project_beta"]}
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "s3_uri": "s3://my-bucket/documents/policy.pdf",
                    "project_ids": ["project_alpha"]
                }
            ]
        }
    }


class BulkIngestRequest(BaseModel):
    """Request model for bulk ingestion from local directory or S3 bucket folder."""
    path: str = Field(
        ...,
        description="Local directory path OR S3 URI (auto-detected by 's3://' prefix)",
        json_schema_extra={"example": "s3://my-bucket/documents/"}
    )
    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of files to process (useful for testing)",
        ge=1,
        json_schema_extra={"example": 10}
    )
    project_ids: Optional[List[str]] = Field(
        default=None,
        description="List of project IDs to assign to ALL ingested documents",
        json_schema_extra={"example": ["project_alpha"]}
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "path": "s3://my-bucket/documents/policies/",
                    "limit": 10,
                    "project_ids": ["project_alpha"]
                },
                {
                    "path": "/home/user/documents",
                    "limit": None,
                    "project_ids": None
                }
            ]
        }
    }


# ==================== Response Models ====================

class IngestionResponse(BaseModel):
    """Response model for single document ingestion."""
    filename: str = Field(..., description="Name of the processed file")
    chunks: int = Field(..., description="Number of chunks created from the document")
    status: str = Field(..., description="Ingestion status ('indexed' on success)")
    extracted_metadata: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Extracted document metadata (title, keywords, etc.)"
    )
    chunking_method: Optional[str] = Field(default=None, description="Method used for chunking")
    s3_source: Optional[str] = Field(default=None, description="S3 source URI if ingested from S3")

    model_config = {
        "json_schema_extra": {
            "example": {
                "filename": "policy_document.pdf",
                "chunks": 15,
                "status": "indexed",
                "extracted_metadata": {
                    "title": "Insurance Policy Terms",
                    "plan_name": "Jeevan Utsav",
                    "keywords": ["Death Benefit", "Maturity Benefit"]
                },
                "chunking_method": "structured"
            }
        }
    }


class FileResult(BaseModel):
    """Result for a single file in bulk ingestion."""
    file: str = Field(..., description="File path or name")
    status: str = Field(..., description="Processing status ('success' or 'failed')")
    chunks: Optional[int] = Field(default=None, description="Number of chunks created")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class BulkIngestionResponse(BaseModel):
    """Response model for bulk ingestion operations."""
    total: int = Field(..., description="Total number of files processed")
    successful: int = Field(..., description="Number of successfully ingested files")
    failed: int = Field(..., description="Number of failed ingestions")
    duration_seconds: float = Field(..., description="Total processing time in seconds")
    details: Optional[List[FileResult]] = Field(default=None, description="Per-file results")
    s3_source: Optional[str] = Field(default=None, description="S3 source URI if ingested from S3")
    downloaded_files: Optional[int] = Field(default=None, description="Number of files downloaded from S3")

    model_config = {
        "json_schema_extra": {
            "example": {
                "total": 10,
                "successful": 9,
                "failed": 1,
                "duration_seconds": 45.2,
                "s3_source": "s3://my-bucket/documents/"
            }
        }
    }


# ==================== API Endpoints ====================

@router.post(
    "/upload",
    response_model=IngestionResponse,
    summary="Upload & Ingest Document",
    response_description="Document successfully ingested and indexed"
)
async def upload_document(
    file: UploadFile = File(..., description="PDF, DOCX, or TXT document file"),
    project_ids: List[str] = Query(
        default=[], 
        description="List of project IDs for multi-tenant filtering"
    )
):
    """
    Upload and ingest a single document file.
    
    **Supported formats:** PDF, DOCX, TXT
    
    **Multi-tenancy:** Optionally assign the document to one or more projects 
    using the `project_ids` query parameter for data isolation.
    
    **Example:**
    ```
    POST /api/v1/ingestion/upload?project_ids=project_a&project_ids=project_b
    ```
    """
    service = IngestionService()
    try:
        result = await service.process_file(file, project_ids=project_ids if project_ids else None)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/upload-s3",
    response_model=IngestionResponse,
    summary="Ingest Document from S3",
    response_description="S3 document successfully ingested and indexed"
)
async def upload_s3_document(request: S3FileRequest):
    """
    Ingest a single document directly from an AWS S3 bucket.
    
    **S3 URI Format:** `s3://bucket-name/path/to/file.pdf`
    
    **Authentication:** Uses EC2 IAM role credentials automatically.
    
    **Note:** For ingesting entire folders, use the `/bulk` endpoint instead.
    """
    import tempfile
    import os
    
    service = IngestionService()
    s3_service = S3Service()
    
    try:
        # Parse S3 URI
        bucket, key = S3Service.parse_s3_uri(request.s3_uri)
        
        # Validate it's a single file, not a folder
        if key.endswith('/') or not key:
            raise HTTPException(
                status_code=400, 
                detail="S3 URI must point to a specific file, not a folder. Use /bulk for folder ingestion."
            )
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="s3_single_")
        
        try:
            # Download file
            local_path = s3_service.download_file(bucket, key, temp_dir)
            
            # Process the file
            result = await service.process_local_file(
                file_path=local_path,
                original_filename=os.path.basename(key),
                additional_metadata={"s3_source": request.s3_uri},
                project_ids=request.project_ids
            )
            
            result["s3_source"] = request.s3_uri
            return result
            
        finally:
            # Clean up temp files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/bulk",
    response_model=BulkIngestionResponse,
    summary="Bulk Ingest Documents",
    response_description="Bulk ingestion completed with summary"
)
async def bulk_ingest(request: BulkIngestRequest):
    """
    Bulk ingest all documents from a local directory or S3 bucket folder.
    
    **Path Types:**
    - **Local:** `/home/user/documents/`
    - **S3:** `s3://bucket-name/folder/` (auto-detected by `s3://` prefix)
    
    **Supported formats:** PDF, DOCX, TXT (recursively scans all subdirectories)
    
    **Features:**
    - Progress tracking with ETA
    - Parallel processing (3 concurrent files)
    - Automatic cleanup of temp files for S3
    
    **Tip:** Use `limit` parameter to test with a small batch first.
    """
    from app.services.ingestion.bulk_service import BulkIngestionService
    
    service = IngestionService()
    bulk_service = BulkIngestionService(service)
    
    try:
        # Auto-detect S3 vs local path
        if request.path.startswith("s3://"):
            results = await bulk_service.ingest_s3_folder(
                request.path,
                limit=request.limit,
                project_ids=request.project_ids
            )
        else:
            results = await bulk_service.ingest_directory(
                request.path, 
                limit=request.limit, 
                project_ids=request.project_ids
            )
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


