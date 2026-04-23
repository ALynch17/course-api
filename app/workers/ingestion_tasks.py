"""
Celery tasks for async, fault-tolerant document ingestion.
"""
import asyncio
import uuid
from datetime import datetime
from typing import Optional

import structlog
from celery import Celery, group, chain
from sqlalchemy import select, func

from app.core.config import settings
from app.db.models import (
    Document, DocumentStatus, PageRecord, PageStatus,
    Chunk, IngestionJob, JobStatus, ChunkType
)
from app.ingestion import storage

log = structlog.get_logger()

celery_app = Celery(
    "course_ingestion",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
)


def run_async(coro):
    """Run an async coroutine from a sync Celery task with a fresh event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def get_db_session():
    """Create a fresh async session for each task — avoids event loop conflicts."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    engine = create_async_engine(
        settings.DATABASE_URL,
        pool_size=1,
        max_overflow=1,
        pool_pre_ping=True,
        connect_args={
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
        },
    )
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# ---------------------------------------------------------------------------
# TASK 1: Orchestrator — parse document, create page records, fan out
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=2)
def ingest_document(self, document_id: str, job_id: str):
    """
    Entry point for document ingestion.
    1. Downloads raw file from S3
    2. Parses into pages
    3. Creates a PageRecord for each page
    4. Fans out one process_page task per page (all run in parallel)
    5. Chains embed_and_index after all pages done
    """
    log.info("ingest_document_start", document_id=document_id)

    async def _run():
        SessionLocal = get_db_session()
        async with SessionLocal() as db:
            doc = await db.get(Document, uuid.UUID(document_id))
            if not doc:
                raise ValueError(f"Document {document_id} not found")

            doc.status = DocumentStatus.processing
            await db.commit()

            # Download raw file from S3
            file_bytes = storage.download_file(doc.s3_key)

            # Parse into pages
            from app.ingestion.parsers import get_parser
            parser = get_parser(doc.file_type)
            pages = parser.parse(file_bytes)

            # Upload rendered images to S3 and create PageRecords
            page_records = []
            for page in pages:
                s3_image_key = None
                if page.image_bytes:
                    s3_image_key = storage.upload_page_image(
                        page.image_bytes, str(doc.course_id),
                        document_id, page.page_number
                    )

                record = PageRecord(
                    document_id=doc.id,
                    page_number=page.page_number,
                    raw_text=page.raw_text,
                    s3_image_key=s3_image_key,
                    status=PageStatus.pending,
                )
                db.add(record)
                page_records.append(record)

            doc.page_count = len(pages)
            doc.slide_count = len(pages) if doc.file_type in ("pptx", "ppt") else None

            job = await db.get(IngestionJob, uuid.UUID(job_id))
            if job:
                job.total_pages = (job.total_pages or 0) + len(pages)
                job.current_stage = "vision_pass"
                job.started_at = datetime.utcnow()
                job.status = JobStatus.processing

            await db.commit()

            doc_context = f"Lecture {doc.lecture_number}: {doc.lecture_title}" \
                if doc.lecture_number else doc.filename

            return [str(r.id) for r in page_records], doc_context, str(doc.course_id)

    page_record_ids, doc_context, course_id = run_async(_run())

    # Fan out — one task per page, all run in parallel
    page_tasks = group(
        process_page.s(page_record_id, document_id, job_id, doc_context)
        for page_record_id in page_record_ids
    )

    pipeline = chain(
        page_tasks,
        embed_and_index_document.s(document_id, job_id, course_id),
    )
    pipeline.apply_async()
    log.info("ingest_document_fanned_out", pages=len(page_record_ids))


# ---------------------------------------------------------------------------
# TASK 2: Process one page — vision pass + chunking (runs in parallel)
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    rate_limit=settings.VISION_RATE_LIMIT,
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
)
def process_page(self, page_record_id: str, document_id: str, job_id: str, doc_context: str):
    """Processes a single page — vision pass + chunking + save chunks."""
    log.info("process_page_start", page_record_id=page_record_id)

    async def _run():
        SessionLocal = get_db_session()
        async with SessionLocal() as db:
            record = await db.get(PageRecord, uuid.UUID(page_record_id))
            if not record:
                return
            if record.status == PageStatus.done:
                log.info("process_page_skip_already_done", page_record_id=page_record_id)
                return

            record.status = PageStatus.processing
            await db.commit()

            doc = await db.get(Document, record.document_id)

            try:
                from app.ingestion.parsers import ParsedPage
                from app.ingestion.vision import run_vision_pass
                from app.ingestion.chunker import chunk_page

                image_bytes = None
                if record.s3_image_key:
                    image_bytes = storage.download_file(record.s3_image_key)

                page = ParsedPage(
                    page_number=record.page_number,
                    raw_text=record.raw_text or "",
                    image_bytes=image_bytes,
                    slide_title=None,
                )

                caption = await run_vision_pass(page, doc_context)
                record.vision_caption = caption

                chunks_data = chunk_page(
                    page=page,
                    vision_caption=caption,
                    source_file=doc.filename,
                    lecture_number=doc.lecture_number,
                    lecture_title=doc.lecture_title,
                )

                for cd in chunks_data:
                    chunk = Chunk(
                        document_id=doc.id,
                        course_id=doc.course_id,
                        source_file=cd.source_file,
                        page_number=cd.page_number,
                        slide_number=cd.slide_number,
                        section_title=cd.section_title,
                        chunk_index=cd.chunk_index,
                        lecture_number=cd.lecture_number,
                        lecture_title=cd.lecture_title,
                        chunk_type=cd.chunk_type,
                        raw_text=cd.raw_text,
                        vision_caption=cd.vision_caption,
                        enriched_text=cd.enriched_text,
                        table_markdown=cd.table_markdown,
                        latex_formula=cd.latex_formula,
                        s3_image_key=record.s3_image_key if cd.has_visual else None,
                        has_visual=cd.has_visual,
                    )
                    db.add(chunk)

                record.status = PageStatus.done
                record.processed_at = datetime.utcnow()

                job = await db.get(IngestionJob, uuid.UUID(job_id))
                if job:
                    job.processed_pages = (job.processed_pages or 0) + 1

                await db.commit()
                log.info("process_page_done", page=record.page_number, chunks=len(chunks_data))

            except Exception as e:
                record.status = PageStatus.failed
                record.retries = (record.retries or 0) + 1
                record.error_message = str(e)
                await db.commit()
                raise

    run_async(_run())


# ---------------------------------------------------------------------------
# TASK 3: Embed + index all chunks (runs once after all pages done)
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=2)
def embed_and_index_document(self, _results, document_id: str, job_id: str, course_id: str):
    """Embeds all chunks for a document and marks it ready."""
    log.info("embed_and_index_start", document_id=document_id)

    async def _run():
        SessionLocal = get_db_session()
        async with SessionLocal() as db:
            job = await db.get(IngestionJob, uuid.UUID(job_id))
            if job:
                job.current_stage = "embedding"
                await db.commit()

            result = await db.execute(
                select(Chunk).where(
                    Chunk.document_id == uuid.UUID(document_id),
                    Chunk.embedding.is_(None),
                )
            )
            chunks = result.scalars().all()

            if not chunks:
                log.info("embed_skip_no_chunks", document_id=document_id)
                return

            log.info("embedding_chunks", count=len(chunks))

            from app.ingestion.embedder import embed_chunks
            from app.ingestion.chunker import ChunkData

            chunk_data_list = []
            for c in chunks:
                cd = ChunkData(
                    source_file=c.source_file,
                    page_number=c.page_number,
                    chunk_index=c.chunk_index or 0,
                    enriched_text=c.enriched_text or c.raw_text or "",
                )
                chunk_data_list.append((c, cd))

            cd_only = [cd for _, cd in chunk_data_list]
            embedded = await embed_chunks(cd_only)

            for (db_chunk, _), cd in zip(chunk_data_list, embedded):
                db_chunk.embedding = cd.embedding

            doc = await db.get(Document, uuid.UUID(document_id))
            if doc:
                doc.status = DocumentStatus.ready
                doc.chunk_count = len(chunks)

            if job:
                job.current_stage = "indexing"
                job.indexed_chunks = (job.indexed_chunks or 0) + len(chunks)
                all_docs_result = await db.execute(
                    select(func.count()).where(
                        Document.job_id == uuid.UUID(job_id),
                        Document.status != DocumentStatus.ready,
                    )
                )
                pending_count = all_docs_result.scalar()
                if pending_count == 0:
                    job.status = JobStatus.done
                    job.completed_at = datetime.utcnow()

            await db.commit()
            log.info("embed_and_index_done", document_id=document_id, chunks=len(chunks))

    run_async(_run())


# ---------------------------------------------------------------------------
# TASK 4: Resume failed ingestion
# ---------------------------------------------------------------------------

@celery_app.task
def resume_failed_ingestion(document_id: str, job_id: str):
    """Requeues only failed/pending pages."""
    async def _run():
        SessionLocal = get_db_session()
        async with SessionLocal() as db:
            result = await db.execute(
                select(PageRecord).where(
                    PageRecord.document_id == uuid.UUID(document_id),
                    PageRecord.status.in_([PageStatus.failed, PageStatus.pending]),
                )
            )
            pages = result.scalars().all()

            doc = await db.get(Document, uuid.UUID(document_id))
            doc_context = f"Lecture {doc.lecture_number}: {doc.lecture_title}" \
                if doc and doc.lecture_number else (doc.filename if doc else "")

            log.info("resume_requeuing", pages=len(pages), document_id=document_id)
            return [(str(p.id), doc_context) for p in pages], str(doc.course_id)

    page_info, course_id = run_async(_run())

    if not page_info:
        log.info("resume_nothing_to_requeue", document_id=document_id)
        return

    page_tasks = group(
        process_page.s(pid, document_id, job_id, doc_context)
        for pid, doc_context in page_info
    )
    pipeline = chain(
        page_tasks,
        embed_and_index_document.s(document_id, job_id, course_id),
    )
    pipeline.apply_async()