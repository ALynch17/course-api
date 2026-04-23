from typing import Optional
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.session import get_db
from app.db.models import Course, CourseStatus, Document, Chunk

router = APIRouter(prefix="/courses", tags=["Courses"])


class CourseCreate(BaseModel):
    name: str
    description: Optional[str] = None
    settings: Optional[dict] = {}


class CourseResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    status: str
    document_count: int
    created_at: str

    class Config:
        from_attributes = True


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_course(body: CourseCreate, db: AsyncSession = Depends(get_db)):
    course = Course(
        name=body.name,
        description=body.description,
        settings=body.settings or {},
    )
    db.add(course)
    await db.commit()
    await db.refresh(course)
    return {
        "id": str(course.id),
        "name": course.name,
        "description": course.description,
        "status": course.status.value,
        "document_count": 0,
        "created_at": course.created_at.isoformat(),
    }


@router.get("")
async def list_courses(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Course).where(Course.status == CourseStatus.active)
    )
    courses = result.scalars().all()
    return [{"id": str(c.id), "name": c.name, "status": c.status.value,
             "created_at": c.created_at.isoformat()} for c in courses]


@router.get("/{course_id}")
async def get_course(course_id: str, db: AsyncSession = Depends(get_db)):
    course = await db.get(Course, uuid.UUID(course_id))
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    doc_count = await db.execute(
        select(func.count()).where(Document.course_id == course.id)
    )
    chunk_count = await db.execute(
        select(func.count()).where(Chunk.course_id == course.id)
    )

    return {
        "id": str(course.id),
        "name": course.name,
        "description": course.description,
        "status": course.status.value,
        "settings": course.settings,
        "document_count": doc_count.scalar(),
        "total_chunks_indexed": chunk_count.scalar(),
        "created_at": course.created_at.isoformat(),
    }


@router.delete("/{course_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_course(course_id: str, db: AsyncSession = Depends(get_db)):
    course = await db.get(Course, uuid.UUID(course_id))
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    await db.delete(course)
    await db.commit()
