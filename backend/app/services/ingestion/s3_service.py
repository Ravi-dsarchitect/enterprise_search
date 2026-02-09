"""S3 Service for downloading documents from AWS S3 buckets."""
import os
import logging
from typing import List, Tuple
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

from app.core.config import settings

logger = logging.getLogger(__name__)


class S3Service:
    SUPPORTED_EXTENSIONS = ('.pdf', '.docx', '.txt')
    
    def __init__(self):
        self.s3_client = boto3.client('s3', region_name=settings.AWS_REGION)
    
    @staticmethod
    def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
        parsed = urlparse(s3_uri)
        if parsed.scheme != 's3':
            raise ValueError(f"Invalid S3 URI: {s3_uri}. Must start with 's3://'")
        bucket = parsed.netloc
        prefix = parsed.path.lstrip('/')
        if not bucket:
            raise ValueError(f"Invalid S3 URI: {s3_uri}. Bucket name is required.")
        return bucket, prefix
    
    def list_files(self, bucket: str, prefix: str) -> List[dict]:
        files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        try:
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key.lower().endswith(self.SUPPORTED_EXTENSIONS):
                        files.append({'key': key, 'size': obj['Size'], 'last_modified': obj['LastModified']})
            return files
        except ClientError as e:
            logger.error(f"Error listing S3 bucket: {e}")
            raise
    
    def download_file(self, bucket: str, key: str, local_dir: str) -> str:
        local_path = os.path.join(local_dir, key.replace('/', os.sep))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            self.s3_client.download_file(bucket, key, local_path)
            return local_path
        except ClientError as e:
            logger.error(f"Error downloading {key}: {e}")
            raise
