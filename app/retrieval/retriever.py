"""
Retrieval pipeline:
  1. Embed the query
  2. Hybrid search: vector (semantic) + keyword (BM25-style full-text)
  3. Fuse results with Reciprocal Rank Fusion
  4. Cross-encoder rerank to find chunks that actually ANSWER the question
  5. Return top-K with metadata and presigned image URLs
"""
import uuid
from dataclasses import dataclass
from typing import Optional
import structlog
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import Chunk
from app.ingestion.embedder import embed_query
from app.ingestion import storage

log = structlog.get_logger()


@dataclass
class RetrievedChunk:
    """A retrieved chunk with all data needed to build the LLM prompt."""
    chunk_id: str
    course_id: str
    source_file: str
    page_number: Optional[int]
    slide_number: Optional[int]
    section_title: Optional[str]
    lecture_number: Optional[int]
    lecture_title: Optional[str]
    chunk_type: str
    raw_text: str
    vision_caption: str
    enriched_text: str
    table_markdown: Optional[str]
    has_visual: bool
    image_url: Optional[str]        # presigned S3 URL, valid 1 hour
    vector_score: float = 0.0
    rerank_score: float = 0.0


async def vector_search(
    query_embedding: list[float],
    course_id: str,
    db: AsyncSession,
    top_k: int = 30,
    lecture_filter: Optional[list[int]] = None,
) -> list[tuple[Chunk, float]]:
    """
    Semantic vector search using pgvector cosine distance.
    Optionally filters by lecture number.
    Returns (Chunk, score) pairs sorted by similarity descending.
    """
    filter_clause = "AND c.course_id = :course_id"
    params = {
        "course_id": uuid.UUID(course_id),
        "embedding": str(query_embedding),
        "top_k": top_k,
    }

    if lecture_filter:
        filter_clause += " AND c.lecture_number = ANY(:lectures)"
        params["lectures"] = lecture_filter

    sql = text(f"""
        SELECT c.id, 1 - (c.embedding <=> CAST(:embedding AS vector)) AS score
        FROM chunks c
        WHERE c.embedding IS NOT NULL
        {filter_clause}
        ORDER BY score DESC
        LIMIT :top_k
    """)

    result = await db.execute(sql, params)
    rows = result.fetchall()

    chunk_ids = [r[0] for r in rows]
    scores = {r[0]: r[1] for r in rows}

    if not chunk_ids:
        return []

    chunks_result = await db.execute(
        select(Chunk).where(Chunk.id.in_(chunk_ids))
    )
    chunks = {c.id: c for c in chunks_result.scalars().all()}

    return [(chunks[cid], scores[cid]) for cid in chunk_ids if cid in chunks]


async def keyword_search(
    query: str,
    course_id: str,
    db: AsyncSession,
    top_k: int = 30,
    lecture_filter: Optional[list[int]] = None,
) -> list[tuple[Chunk, float]]:
    """
    Full-text keyword search using PostgreSQL tsvector.
    Catches exact matches that vector search might rank low
    (specific formula names, proper nouns, defined terms).
    """
    filter_clause = "AND course_id = :course_id"
    params = {
        "course_id": uuid.UUID(course_id),
        "query": query,
        "top_k": top_k,
    }

    if lecture_filter:
        filter_clause += " AND lecture_number = ANY(:lectures)"
        params["lectures"] = lecture_filter

    sql = text(f"""
        SELECT id,
               ts_rank(
                   to_tsvector('english', COALESCE(enriched_text, '')),
                   plainto_tsquery('english', :query)
               ) AS score
        FROM chunks
        WHERE to_tsvector('english', COALESCE(enriched_text, ''))
              @@ plainto_tsquery('english', :query)
        {filter_clause}
        ORDER BY score DESC
        LIMIT :top_k
    """)

    result = await db.execute(sql, params)
    rows = result.fetchall()

    chunk_ids = [r[0] for r in rows]
    scores = {r[0]: r[1] for r in rows}

    if not chunk_ids:
        return []

    chunks_result = await db.execute(
        select(Chunk).where(Chunk.id.in_(chunk_ids))
    )
    chunks = {c.id: c for c in chunks_result.scalars().all()}

    return [(chunks[cid], float(scores[cid])) for cid in chunk_ids if cid in chunks]


def reciprocal_rank_fusion(
    result_lists: list[list[tuple[Chunk, float]]],
    k: int = 60,
) -> list[tuple[Chunk, float]]:
    """
    Merges multiple ranked result lists into a single ranking.
    RRF score = sum(1 / (k + rank)) across all lists.
    Higher score = appeared highly ranked in more lists.
    """
    scores: dict[uuid.UUID, float] = {}
    chunk_map: dict[uuid.UUID, Chunk] = {}

    for result_list in result_lists:
        for rank, (chunk, _) in enumerate(result_list):
            cid = chunk.id
            if cid not in scores:
                scores[cid] = 0.0
            scores[cid] += 1.0 / (k + rank + 1)
            chunk_map[cid] = chunk

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [(chunk_map[cid], scores[cid]) for cid in sorted_ids]


def rerank(
    query: str,
    candidates: list[tuple[Chunk, float]],
    top_k: int = 5,
) -> list[tuple[Chunk, float]]:
    """
    Cross-encoder reranking: runs a deep transformer on (query, document) pairs
    to score how well each chunk *answers* the question — not just semantic similarity.

    Uses sentence-transformers CrossEncoder (runs locally, no extra API cost).
    """
    from sentence_transformers import CrossEncoder

    # Load once and cache (module-level singleton pattern)
    if not hasattr(rerank, "_model"):
        log.info("loading_reranker", model=settings.RERANKER_MODEL)
        rerank._model = CrossEncoder(settings.RERANKER_MODEL)

    model = rerank._model

    pairs = [
        (query, chunk.enriched_text or chunk.raw_text or "")
        for chunk, _ in candidates
    ]

    scores = model.predict(pairs)

    reranked = sorted(
        zip([c for c, _ in candidates], scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    return [(chunk, float(score)) for chunk, score in reranked[:top_k]]


async def retrieve(
    query: str,
    course_id: str,
    db: AsyncSession,
    top_k: int = None,
    lecture_filter: Optional[list[int]] = None,
) -> list[RetrievedChunk]:
    """
    Full retrieval pipeline:
      embed → hybrid search → RRF fusion → cross-encoder rerank → build results
    """
    top_k = top_k or settings.RETRIEVAL_TOP_K_FINAL
    candidates_k = settings.RETRIEVAL_TOP_K_CANDIDATES

    log.info("retrieval_start", query=query[:80], course_id=course_id)

    # Step 1: embed query
    query_embedding = await embed_query(query)

    # Step 2: hybrid search
    vector_results = await vector_search(
        query_embedding, course_id, db,
        top_k=candidates_k, lecture_filter=lecture_filter
    )
    keyword_results = await keyword_search(
        query, course_id, db,
        top_k=candidates_k, lecture_filter=lecture_filter
    )

    log.info("retrieval_raw_counts",
             vector=len(vector_results), keyword=len(keyword_results))

    # Step 3: fuse
    fused = reciprocal_rank_fusion([vector_results, keyword_results])

    if not fused:
        log.info("retrieval_no_results", query=query[:80])
        return []

    # Step 4: rerank top candidates
    top_candidates = fused[:candidates_k]
    reranked = rerank(query, top_candidates, top_k=top_k)

    # Step 5: build RetrievedChunk objects with presigned image URLs
    results = []
    for chunk, score in reranked:
        image_url = None
        if chunk.has_visual and chunk.s3_image_key:
            image_url = storage.get_presigned_url(chunk.s3_image_key)

        results.append(RetrievedChunk(
            chunk_id=str(chunk.id),
            course_id=str(chunk.course_id),
            source_file=chunk.source_file,
            page_number=chunk.page_number,
            slide_number=chunk.slide_number,
            section_title=chunk.section_title,
            lecture_number=chunk.lecture_number,
            lecture_title=chunk.lecture_title,
            chunk_type=chunk.chunk_type.value if chunk.chunk_type else "text",
            raw_text=chunk.raw_text or "",
            vision_caption=chunk.vision_caption or "",
            enriched_text=chunk.enriched_text or "",
            table_markdown=chunk.table_markdown,
            has_visual=chunk.has_visual,
            image_url=image_url,
            rerank_score=score,
        ))

    log.info("retrieval_done", final_chunks=len(results))
    return results
