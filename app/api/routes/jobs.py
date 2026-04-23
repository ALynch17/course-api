import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import IngestionJob, Document

router = APIRouter(tags=["Jobs"])


@router.get("/courses/{course_id}/jobs")
async def list_jobs(course_id: str, db: AsyncSession = Depends(get_db)):
    """All ingestion jobs for a course — history + active."""
    result = await db.execute(
        select(IngestionJob)
        .where(IngestionJob.course_id == uuid.UUID(course_id))
        .order_by(IngestionJob.created_at.desc())
    )
    jobs = result.scalars().all()

    return [
        {
            "id": str(j.id),
            "status": j.status.value,
            "current_stage": j.current_stage,
            "total_pages": j.total_pages,
            "processed_pages": j.processed_pages,
            "percent": round((j.processed_pages / j.total_pages) * 100)
                       if j.total_pages else 0,
            "started_at": j.started_at.isoformat() if j.started_at else None,
            "completed_at": j.completed_at.isoformat() if j.completed_at else None,
            "created_at": j.created_at.isoformat(),
        }
        for j in jobs
    ]


@router.get("/courses/{course_id}/jobs/{job_id}")
async def get_job(course_id: str, job_id: str, db: AsyncSession = Depends(get_db)):
    """
    Granular job status — stage, progress, per-document breakdown.
    Client polls this every 3–5 seconds during ingestion.
    """
    job = await db.get(IngestionJob, uuid.UUID(job_id))
    if not job or str(job.course_id) != course_id:
        raise HTTPException(status_code=404, detail="Job not found")

    # Get all documents in this job
    docs_result = await db.execute(
        select(Document).where(Document.job_id == job.id)
    )
    docs = docs_result.scalars().all()

    total_pages = job.total_pages or 0
    processed = job.processed_pages or 0

    return {
        "id": str(job.id),
        "status": job.status.value,
        "current_stage": job.current_stage,
        "progress": {
            "total_pages": total_pages,
            "processed_pages": processed,
            "indexed_chunks": job.indexed_chunks,
            "percent": round((processed / total_pages) * 100) if total_pages else 0,
        },
        "documents": [
            {
                "id": str(d.id),
                "filename": d.filename,
                "lecture_number": d.lecture_number,
                "status": d.status.value,
                "page_count": d.page_count,
                "chunk_count": d.chunk_count,
                "error": d.error_message,
            }
            for d in docs
        ],
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error": job.error_message,
    }
