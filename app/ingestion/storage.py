"""
Object storage service (AWS S3).
Handles:
  - Raw document files
  - Rendered page/slide images
  - Multipart upload orchestration for large files
"""
import uuid
from typing import Optional
import boto3
from botocore.exceptions import ClientError
import structlog

from app.core.config import settings

log = structlog.get_logger()

_s3 = boto3.client(
    "s3",
    region_name=settings.S3_REGION,
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
)


def upload_file(
    file_bytes: bytes,
    key: str,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload bytes to S3. Returns the S3 key."""
    _s3.put_object(
        Bucket=settings.S3_BUCKET,
        Key=key,
        Body=file_bytes,
        ContentType=content_type,
    )
    log.info("s3_upload_done", key=key, size=len(file_bytes))
    return key


def upload_page_image(
    image_bytes: bytes,
    course_id: str,
    document_id: str,
    page_number: int,
) -> str:
    """Store a rendered page/slide image. Returns S3 key."""
    key = f"courses/{course_id}/documents/{document_id}/pages/page_{page_number:04d}.png"
    return upload_file(image_bytes, key, content_type="image/png")


def upload_raw_document(
    file_bytes: bytes,
    course_id: str,
    document_id: str,
    filename: str,
) -> str:
    """Store original uploaded document. Returns S3 key."""
    ext = filename.rsplit(".", 1)[-1].lower()
    key = f"courses/{course_id}/documents/{document_id}/raw/{filename}"
    content_types = {
        "pdf": "application/pdf",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "ipynb": "application/json",
    }
    content_type = content_types.get(ext, "application/octet-stream")
    return upload_file(file_bytes, key, content_type)


def get_presigned_url(key: str, expires_in: int = 3600) -> str:
    """Generate a temporary public URL for a stored object."""
    return _s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.S3_BUCKET, "Key": key},
        ExpiresIn=expires_in,
    )


def download_file(key: str) -> bytes:
    """Download object from S3 and return bytes."""
    response = _s3.get_object(Bucket=settings.S3_BUCKET, Key=key)
    return response["Body"].read()


# ---------------------------------------------------------------------------
# Multipart upload — for large files uploaded in chunks from the client
# ---------------------------------------------------------------------------

def initiate_multipart_upload(key: str, content_type: str) -> str:
    """Start a multipart upload session. Returns the S3 upload ID."""
    response = _s3.create_multipart_upload(
        Bucket=settings.S3_BUCKET,
        Key=key,
        ContentType=content_type,
    )
    return response["UploadId"]


def get_presigned_part_url(key: str, upload_id: str, part_number: int) -> str:
    """Presigned URL for client to upload one chunk directly to S3."""
    return _s3.generate_presigned_url(
        "upload_part",
        Params={
            "Bucket": settings.S3_BUCKET,
            "Key": key,
            "UploadId": upload_id,
            "PartNumber": part_number,
        },
        ExpiresIn=3600,
    )


def complete_multipart_upload(key: str, upload_id: str, parts: list[dict]) -> str:
    """
    Finalise a multipart upload after all parts are uploaded.
    parts: [{"PartNumber": 1, "ETag": "..."}, ...]
    """
    _s3.complete_multipart_upload(
        Bucket=settings.S3_BUCKET,
        Key=key,
        UploadId=upload_id,
        MultipartUpload={"Parts": parts},
    )
    log.info("multipart_upload_complete", key=key)
    return key
