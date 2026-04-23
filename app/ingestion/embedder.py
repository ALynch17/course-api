"""
Embedding service.
Uses text-embedding-3-large (3072 dims) via OpenAI.
Batches calls to avoid rate limits — max 100 texts per API call.
"""
from typing import Optional
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI

from app.core.config import settings
from app.ingestion.chunker import ChunkData

log = structlog.get_logger()
_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns list of float vectors."""
    if not texts:
        return []
    response = await _client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=texts,
        dimensions=settings.EMBEDDING_DIMENSIONS,
    )
    # Sort by index to guarantee order matches input
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]


async def embed_chunks(chunks: list[ChunkData]) -> list[ChunkData]:
    """
    Embeds the enriched_text of every chunk in batches of EMBEDDING_BATCH_SIZE.
    Mutates each chunk's .embedding field in place and returns the list.
    """
    batch_size = settings.EMBEDDING_BATCH_SIZE
    log.info("embedding_start", total_chunks=len(chunks))

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        texts = [c.enriched_text for c in batch]

        embeddings = await embed_texts(texts)

        for chunk, vector in zip(batch, embeddings):
            chunk.embedding = vector

        log.info("embedding_batch_done", batch_end=i + len(batch), total=len(chunks))

    return chunks


async def embed_query(query: str) -> list[float]:
    """Embed a single query string for retrieval."""
    vectors = await embed_texts([query])
    return vectors[0]
