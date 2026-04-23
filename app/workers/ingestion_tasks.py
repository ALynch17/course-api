"""
Celery tasks for async, fault-tolerant document ingestion.

Architecture:
  ingest_document
    └── fan-out: process_page (one per page, all parallel)
          └── on all done: embed_and_index_document
                └── mark_document_ready

Each page is tracked individually in PageRecord so we can:
  - Resume from failure (only requeue failed/pending pages)
  - Report granular progress
  - Never reprocess completed pages
"""
import asyncio
import uuid
from datetime import datetime
from typing import Optional

import structlog
from celery import Celery, group, chain
from sqlalchemy import select, update, func

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.db.models import (
    Document, DocumentStatus, PageRecord, PageStatus,
    Chunk, IngestionJob, JobStatus, ChunkType
)
from app.ingestion.parsers import get_parser
from app.ingestion.vision import run_vision_pass, run_vision_passes_batch
from app.ingestion.chunker import chunk_page, ChunkData
from app.ingestion.embedder import embed_chunks
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
    task_acks_late=True,            # only ack after task completes (fault tolerance)
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,   # one task at a time per worker (prevents memory blowout)
)


def run_async(coro):
    """Run an async coroutine from a sync Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# TASK 1: Orchestrator — parse document, create page records, fan out
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=2)
def ingest_document(self, document_id: str, job_id: str):
    """
    Synchronous ingestion - processes all pages sequentially.
    No Celery fan-out needed when running without Redis.
    """
    log.info("ingest_document_start", document_id=document_id)

    async def _run():
        async with AsyncSessionLocal() as db:
            doc = await db.get(Document, uuid.UUID(document_id))
            if not doc:
                raise ValueError(f"Document {document_id} not found")

            doc.status = DocumentStatus.processing
            await db.commit()

            # Download raw file from S3
            file_bytes = storage.download_file(doc.s3_key)

            # Parse into pages
            parser = get_parser(doc.file_type)
            pages = parser.parse(file_bytes)

            # Update job
            job = await db.get(IngestionJob, uuid.UUID(job_id))
            if job:
                job.total_pages = len(pages)
                job.current_stage = "vision_pass"
                job.started_at = datetime.utcnow()
                job.status = JobStatus.processing
            await db.commit()

            # Build document context string for vision model
            doc_context = f"Lecture {doc.lecture_number}: {doc.lecture_title}" \
                if doc.lecture_number else doc.filename

            # Process each page sequentially
            all_chunks = []
            for page in pages:
                try:
                    # Upload page image to S3
                    s3_image_key = None
                    if page.image_bytes:
                        s3_image_key = storage.upload_page_image(
                            page.image_bytes, str(doc.course_id),
                            document_id, page.page_number
                        )

                    # Vision pass
                    from app.ingestion.vision import run_vision_pass
                    caption = await run_vision_pass(page, doc_context)

                    # Chunk the page
                    from app.ingestion.chunker import chunk_page
                    chunks_data = chunk_page(
                        page=page,
                        vision_caption=caption,
                        source_file=doc.filename,
                        lecture_number=doc.lecture_number,
                        lecture_title=doc.lecture_title,
                    )

                    # Save chunks to DB
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
                            s3_image_key=s3_image_key if cd.has_visual else None,
                            has_visual=cd.has_visual,
                        )
                        db.add(chunk)
                        all_chunks.append(chunk)

                    # Update progress
                    if job:
                        job.processed_pages = (job.processed_pages or 0) + 1
                    await db.commit()

                    log.info("page_processed", page=page.page_number)

                except Exception as e:
                    log.error("page_failed", page=page.page_number, error=str(e))
                    continue

            # Embed all chunks in batches
            if job:
                job.current_stage = "embedding"
            await db.commit()

            from app.ingestion.embedder import embed_chunks
            from app.ingestion.chunker import ChunkData

            # Reload chunks that need embedding
            from sqlalchemy import select
            result = await db.execute(
                select(Chunk).where(
                    Chunk.document_id == uuid.UUID(document_id),
                    Chunk.embedding.is_(None),
                )
            )
            chunks_to_embed = result.scalars().all()

            if chunks_to_embed:
                chunk_data_list = []
                for c in chunks_to_embed:
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

            # Mark document ready
            doc.status = DocumentStatus.ready
            doc.chunk_count = len(chunks_to_embed)

            if job:
                job.status = JobStatus.done
                job.current_stage = "done"
                job.completed_at = datetime.utcnow()
                job.indexed_chunks = len(chunks_to_embed)

            await db.commit()
            log.info("ingest_document_done", document_id=document_id,
                     chunks=len(chunks_to_embed))

    run_async(_run())


# ---------------------------------------------------------------------------
# TASK 2: Process one page — vision pass + chunking (runs in parallel)
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    rate_limit=settings.VISION_RATE_LIMIT,     # e.g. "50/m" — prevents vision API flooding
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
)
def process_page(self, page_record_id: str, document_id: str, job_id: str, doc_context: str):
    """
    Processes a single page:
    1. Loads the rendered image from S3
    2. Runs vision model pass to get rich caption
    3. Runs chunker to produce ChunkData objects
    4. Saves chunks to DB (without embeddings yet)
    5. Marks page as done
    """
    log.info("process_page_start", page_record_id=page_record_id)

    async def _run():
        async with AsyncSessionLocal() as db:
            record = await db.get(PageRecord, uuid.UUID(page_record_id))
            if not record:
                return
            if record.status == PageStatus.done:
                log.info("process_page_skip_already_done", page_record_id=page_record_id)
                return  # Resumability: skip if already processed

            record.status = PageStatus.processing
            await db.commit()

            doc = await db.get(Document, record.document_id)

            try:
                # Build a ParsedPage-like object from stored data
                from app.ingestion.parsers import ParsedPage
                image_bytes = None
                if record.s3_image_key:
                    image_bytes = storage.download_file(record.s3_image_key)

                page = ParsedPage(
                    page_number=record.page_number,
                    raw_text=record.raw_text or "",
                    image_bytes=image_bytes,
                    slide_title=None,  # already stored in raw_text
                )

                # Vision pass
                caption = await run_vision_pass(page, doc_context)
                record.vision_caption = caption

                # Chunk the page
                chunks_data = chunk_page(
                    page=page,
                    vision_caption=caption,
                    source_file=doc.filename,
                    lecture_number=doc.lecture_number,
                    lecture_title=doc.lecture_title,
                )

                # Save chunks (no embeddings yet — done in batch later)
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

                # Increment job progress counter
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
                raise  # Celery will retry

    run_async(_run())


# ---------------------------------------------------------------------------
# TASK 3: Embed + index all chunks for the document (runs once after all pages)
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=2)
def embed_and_index_document(self, _results, document_id: str, job_id: str, course_id: str):
    """
    After all pages are processed:
    1. Loads all unembedded chunks for this document
    2. Embeds enriched_text in batches
    3. Writes embeddings to DB
    4. Marks document as ready
    """
    log.info("embed_and_index_start", document_id=document_id)

    async def _run():
        async with AsyncSessionLocal() as db:
            job = await db.get(IngestionJob, uuid.UUID(job_id))
            if job:
                job.current_stage = "embedding"
                await db.commit()

            # Load all chunks for this document that have no embedding yet
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

            # Build ChunkData objects for the embedder
            chunk_data_list = []
            for c in chunks:
                cd = ChunkData(
                    source_file=c.source_file,
                    page_number=c.page_number,
                    chunk_index=c.chunk_index,
                    enriched_text=c.enriched_text or c.raw_text,
                )
                chunk_data_list.append((c, cd))

            # Batch embed
            cd_only = [cd for _, cd in chunk_data_list]
            embedded = await embed_chunks(cd_only)

            # Write embeddings back to DB records
            for (db_chunk, _), cd in zip(chunk_data_list, embedded):
                db_chunk.embedding = cd.embedding

            # Mark document ready
            doc = await db.get(Document, uuid.UUID(document_id))
            if doc:
                doc.status = DocumentStatus.ready
                doc.chunk_count = len(chunks)

            # Update job
            if job:
                job.current_stage = "indexing"
                job.indexed_chunks = (job.indexed_chunks or 0) + len(chunks)
                # Check if all documents in job are done
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
# TASK 4: Resume failed ingestion — requeues only failed/pending pages
# ---------------------------------------------------------------------------

@celery_app.task
def resume_failed_ingestion(document_id: str, job_id: str):
    """
    Finds all pages with status failed or pending and requeues them.
    Safe to call multiple times — already-done pages are skipped in process_page.
    """
    async def _run():
        async with AsyncSessionLocal() as db:
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
