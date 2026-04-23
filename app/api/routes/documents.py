"""
Document upload routes.

Two upload modes:
  1. Direct upload  — small files, single POST with file in body
  2. Chunked upload — large files, client splits into parts, uploads directly to S3

After upload, triggers async ingestion via Celery.
"""
import uuid
import mimetypes
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import (
    Course, Document, DocumentStatus, IngestionJob, JobStatus,
    PageRecord, PageStatus, UploadSession
)
from app.ingestion import storage
from app.workers.ingestion_tasks import ingest_document, resume_failed_ingestion
from app.core.config import settings

router = APIRouter(tags=["Documents"])

SUPPORTED_TYPES = {"pdf", "pptx", "ppt", "docx", "doc", "ipynb", "png", "jpg", "jpeg"}


def get_file_type(filename: str) -> str:
    return Path(filename).suffix.lstrip(".").lower()


# ---------------------------------------------------------------------------
# Direct upload (single request, files up to MAX_FILE_SIZE_MB)
# ---------------------------------------------------------------------------

@router.post("/courses/{course_id}/documents", status_code=status.HTTP_202_ACCEPTED)
async def upload_documents(
    course_id: str,
    files: list[UploadFile] = File(...),
    run_vision_pass: bool = Form(True),
    chunk_strategy: str = Form("semantic"),
    lecture_numbers: Optional[str] = Form(None),    # comma-separated e.g. "1,2,3"
    lecture_titles: Optional[str] = Form(None),     # comma-separated, same order
    db: AsyncSession = Depends(get_db),
):
    """
    Upload one or more course documents. Returns a job ID to poll for progress.
    Triggers async ingestion immediately after upload.
    """
    course = await db.get(Course, uuid.UUID(course_id))
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    # Parse lecture metadata
    lec_nums = [int(n.strip()) for n in lecture_numbers.split(",")] \
        if lecture_numbers else []
    lec_titles = [t.strip() for t in lecture_titles.split(",")] \
        if lecture_titles else []

    # Create ingestion job
    job = IngestionJob(course_id=course.id, status=JobStatus.queued)
    db.add(job)
    await db.flush()

    doc_responses = []

    for i, file in enumerate(files):
        file_type = get_file_type(file.filename)
        if file_type not in SUPPORTED_TYPES:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file type: {file_type}. Supported: {SUPPORTED_TYPES}"
            )

        file_bytes = await file.read()
        size_bytes = len(file_bytes)

        if size_bytes > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"{file.filename} exceeds {settings.MAX_FILE_SIZE_MB}MB limit. "
                       f"Use the chunked upload endpoint for large files."
            )

        # Create document record
        doc = Document(
            course_id=course.id,
            job_id=job.id,
            filename=file.filename,
            original_filename=file.filename,
            file_type=file_type,
            size_bytes=size_bytes,
            status=DocumentStatus.queued,
            lecture_number=lec_nums[i] if i < len(lec_nums) else None,
            lecture_title=lec_titles[i] if i < len(lec_titles) else None,
        )
        db.add(doc)
        await db.flush()

        # Upload raw file to S3
        s3_key = storage.upload_raw_document(
            file_bytes, course_id, str(doc.id), file.filename
        )
        doc.s3_key = s3_key

        doc_responses.append({
            "id": str(doc.id),
            "filename": file.filename,
            "file_type": file_type,
            "size_bytes": size_bytes,
            "status": "queued",
            "lecture_number": doc.lecture_number,
        })

    await db.commit()

    for doc_resp in doc_responses:
        # Trigger async ingestion
        #ingest_document.delay(str(doc.id), str(job.id))
        ingest_document.delay(doc_resp["id"], str(job.id))

    #await db.commit()

    return {
        "job_id": str(job.id),
        "course_id": course_id,
        "documents": doc_responses,
        "poll_url": f"/api/v1/courses/{course_id}/jobs/{job.id}",
        "message": "Ingestion started. Poll the job URL for progress.",
    }


# ---------------------------------------------------------------------------
# Chunked upload — for files > MAX_FILE_SIZE_MB
# ---------------------------------------------------------------------------

class InitiateUploadRequest(BaseModel):
    filename: str
    size_bytes: int
    total_chunks: int
    lecture_number: Optional[int] = None
    lecture_title: Optional[str] = None


@router.post("/courses/{course_id}/uploads/initiate")
async def initiate_chunked_upload(
    course_id: str,
    body: InitiateUploadRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Step 1 of chunked upload.
    Creates an S3 multipart upload session.
    Returns presigned URLs for each chunk — client uploads directly to S3.
    """
    course = await db.get(Course, uuid.UUID(course_id))
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    file_type = get_file_type(body.filename)
    if file_type not in SUPPORTED_TYPES:
        raise HTTPException(status_code=422, detail=f"Unsupported file type: {file_type}")

    # Create upload session record
    session = UploadSession(
        course_id=course.id,
        filename=body.filename,
        size_bytes=body.size_bytes,
        total_chunks=body.total_chunks,
        received_chunks=[],
        expires_at=datetime.utcnow() + timedelta(hours=24),
    )
    db.add(session)
    await db.flush()

    # Create S3 key and multipart upload
    s3_key = f"courses/{course_id}/uploads/{session.id}/{body.filename}"
    upload_id = storage.initiate_multipart_upload(s3_key, "application/octet-stream")
    session.s3_upload_id = upload_id
    await db.commit()

    # Generate presigned URL for each chunk
    part_urls = [
        {
            "chunk_number": i + 1,
            "upload_url": storage.get_presigned_part_url(s3_key, upload_id, i + 1),
        }
        for i in range(body.total_chunks)
    ]

    return {
        "upload_id": str(session.id),
        "s3_key": s3_key,
        "chunk_urls": part_urls,
        "expires_at": session.expires_at.isoformat(),
        "complete_url": f"/api/v1/courses/{course_id}/uploads/{session.id}/complete",
    }


class CompleteUploadRequest(BaseModel):
    parts: list[dict]       # [{"PartNumber": 1, "ETag": "..."}, ...]
    lecture_number: Optional[int] = None
    lecture_title: Optional[str] = None


@router.post("/courses/{course_id}/uploads/{upload_id}/complete",
             status_code=status.HTTP_202_ACCEPTED)
async def complete_chunked_upload(
    course_id: str,
    upload_id: str,
    body: CompleteUploadRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Step 2 of chunked upload.
    Tells S3 to assemble the parts, then triggers ingestion.
    """
    session = await db.get(UploadSession, uuid.UUID(upload_id))
    if not session:
        raise HTTPException(status_code=404, detail="Upload session not found")

    # Complete the S3 multipart upload — S3 assembles all parts into one object
    s3_key = f"courses/{course_id}/uploads/{upload_id}/{session.filename}"
    storage.complete_multipart_upload(s3_key, session.s3_upload_id, body.parts)

    # Create document record
    file_type = get_file_type(session.filename)
    doc = Document(
        course_id=session.course_id,
        filename=session.filename,
        original_filename=session.filename,
        file_type=file_type,
        size_bytes=session.size_bytes,
        s3_key=s3_key,
        status=DocumentStatus.queued,
        lecture_number=body.lecture_number,
        lecture_title=body.lecture_title,
    )
    db.add(doc)

    job = IngestionJob(course_id=session.course_id, status=JobStatus.queued)
    db.add(job)
    await db.flush()

    doc.job_id = job.id
    session.status = "complete"
    await db.commit()

    ingest_document.delay(str(doc.id), str(job.id))
    #ingest_document(str(doc.id), str(job.id))

    return {
        "job_id": str(job.id),
        "document_id": str(doc.id),
        "status": "queued",
        "poll_url": f"/api/v1/courses/{course_id}/jobs/{job.id}",
    }


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------

@router.get("/courses/{course_id}/documents")
async def list_documents(course_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Document).where(Document.course_id == uuid.UUID(course_id))
    )
    docs = result.scalars().all()
    return [
        {
            "id": str(d.id),
            "filename": d.filename,
            "file_type": d.file_type,
            "status": d.status.value,
            "page_count": d.page_count,
            "chunk_count": d.chunk_count,
            "lecture_number": d.lecture_number,
            "lecture_title": d.lecture_title,
            "created_at": d.created_at.isoformat(),
        }
        for d in docs
    ]


@router.get("/courses/{course_id}/documents/{doc_id}")
async def get_document(course_id: str, doc_id: str, db: AsyncSession = Depends(get_db)):
    doc = await db.get(Document, uuid.UUID(doc_id))
    if not doc or str(doc.course_id) != course_id:
        raise HTTPException(status_code=404, detail="Document not found")

    pages_result = await db.execute(
        select(PageRecord).where(PageRecord.document_id == doc.id)
        .order_by(PageRecord.page_number)
    )
    pages = pages_result.scalars().all()

    return {
        "id": str(doc.id),
        "filename": doc.filename,
        "file_type": doc.file_type,
        "status": doc.status.value,
        "lecture_number": doc.lecture_number,
        "lecture_title": doc.lecture_title,
        "file": {
            "size_bytes": doc.size_bytes,
            "page_count": doc.page_count,
            "slide_count": doc.slide_count,
        },
        "processing": doc.processing_stats,
        "chunk_count": doc.chunk_count,
        "pages": [
            {
                "page_number": p.page_number,
                "status": p.status.value,
                "retries": p.retries,
                "has_image": bool(p.s3_image_key),
                "has_caption": bool(p.vision_caption),
            }
            for p in pages
        ],
    }


@router.delete("/courses/{course_id}/documents/{doc_id}", status_code=204)
async def delete_document(course_id: str, doc_id: str, db: AsyncSession = Depends(get_db)):
    doc = await db.get(Document, uuid.UUID(doc_id))
    if not doc or str(doc.course_id) != course_id:
        raise HTTPException(status_code=404, detail="Document not found")
    await db.delete(doc)
    await db.commit()


@router.post("/courses/{course_id}/documents/{doc_id}/retry",
             status_code=status.HTTP_202_ACCEPTED)
async def retry_failed_document(
    course_id: str, doc_id: str, db: AsyncSession = Depends(get_db)
):
    """Retry ingestion for a document with failed pages."""
    doc = await db.get(Document, uuid.UUID(doc_id))
    if not doc or str(doc.course_id) != course_id:
        raise HTTPException(status_code=404, detail="Document not found")

    resume_failed_ingestion.delay(doc_id, str(doc.job_id))
    return {"message": "Retry queued", "document_id": doc_id}
