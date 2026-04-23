"""
Chunker: splits parsed + vision-captioned pages into retrievable chunks.

Rules:
- Tables, diagrams, and formulas are ATOMIC chunks — never split them
- Respect natural boundaries: slide = one chunk, section = chunk boundary
- Every chunk stores full provenance metadata
- Enriched text = metadata prefix + raw text + vision caption (what gets embedded)
"""
from dataclasses import dataclass, field
from typing import Optional
import re

from app.core.config import settings
from app.db.models import ChunkType
from app.ingestion.parsers import ParsedPage


@dataclass
class ChunkData:
    """In-memory representation of a chunk before it's written to the DB."""
    # Provenance
    source_file: str
    page_number: int
    chunk_index: int
    section_title: Optional[str] = None
    slide_number: Optional[int] = None
    lecture_number: Optional[int] = None
    lecture_title: Optional[str] = None

    # Content
    chunk_type: ChunkType = ChunkType.text
    raw_text: str = ""
    vision_caption: str = ""
    enriched_text: str = ""         # built by build_enriched_text()
    table_markdown: Optional[str] = None
    latex_formula: Optional[str] = None

    # Visual
    image_bytes: Optional[bytes] = None
    has_visual: bool = False

    # Filled after embedding
    embedding: Optional[list[float]] = None
    s3_image_key: Optional[str] = None


def detect_chunk_type(page: ParsedPage, vision_caption: str) -> ChunkType:
    """
    Heuristically detect what type of content this page/chunk primarily contains.
    Used to drive retrieval and LLM prompt assembly.
    """
    caption_lower = vision_caption.lower()
    raw_lower = page.raw_text.lower()

    if page.code_blocks:
        return ChunkType.code
    if page.tables:
        return ChunkType.table
    # Check vision caption for visual indicators
    visual_keywords = ["diagram", "chart", "figure", "graph", "flowchart",
                       "illustration", "plot", "schematic", "neural network",
                       "architecture", "arrow", "node"]
    if any(kw in caption_lower for kw in visual_keywords) and len(page.raw_text) < 200:
        return ChunkType.diagram
    formula_keywords = ["equation", "formula", "δ", "∂", "∑", "∫", "≈", "→", "\\frac"]
    if any(kw in raw_lower or kw in caption_lower for kw in formula_keywords):
        return ChunkType.formula
    if page.slide_title is not None:
        return ChunkType.slide
    return ChunkType.text


def build_enriched_text(
    chunk: ChunkData,
    course_name: str = "",
) -> str:
    """
    Builds the string that gets embedded.
    Metadata prefix grounds the embedding in course context so a vague question
    like 'how do we update weights?' finds a formula chunk from lecture 5.
    """
    parts = []

    # --- Context prefix (most important for embedding quality) ---
    if course_name:
        parts.append(f"Course: {course_name}")
    if chunk.lecture_number and chunk.lecture_title:
        parts.append(f"Lecture {chunk.lecture_number}: {chunk.lecture_title}")
    elif chunk.lecture_number:
        parts.append(f"Lecture {chunk.lecture_number}")
    if chunk.section_title:
        parts.append(f"Section: {chunk.section_title}")
    if chunk.slide_number:
        parts.append(f"Slide: {chunk.slide_number}")
    parts.append(f"Source: {chunk.source_file}, Page {chunk.page_number}")
    parts.append(f"Content type: {chunk.chunk_type.value}")

    # --- Actual content ---
    if chunk.raw_text:
        parts.append(chunk.raw_text)
    if chunk.vision_caption:
        parts.append(f"Visual description: {chunk.vision_caption}")
    if chunk.table_markdown:
        parts.append(f"Table:\n{chunk.table_markdown}")
    if chunk.latex_formula:
        parts.append(f"Formula: {chunk.latex_formula}")

    return "\n".join(parts)


def chunk_page(
    page: ParsedPage,
    vision_caption: str,
    source_file: str,
    lecture_number: Optional[int] = None,
    lecture_title: Optional[str] = None,
    course_name: str = "",
    chunk_size: int = None,
) -> list[ChunkData]:
    """
    Converts a single ParsedPage (+ its vision caption) into one or more ChunkData objects.

    Strategy:
    - If page is a slide (PPTX): one chunk for the whole slide
    - If page has tables: one atomic chunk per table + one chunk for remaining text
    - If page is diagram-heavy (little text, rich caption): one chunk
    - Otherwise: split raw text into overlapping windows
    """
    chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
    overlap = settings.DEFAULT_CHUNK_OVERLAP
    chunks: list[ChunkData] = []
    chunk_index = 0
    chunk_type = detect_chunk_type(page, vision_caption)

    base = dict(
        source_file=source_file,
        page_number=page.page_number,
        slide_number=page.page_number if page.slide_title is not None else None,
        section_title=page.section_heading or page.slide_title,
        lecture_number=lecture_number,
        lecture_title=lecture_title,
        has_visual=page.image_bytes is not None,
        image_bytes=page.image_bytes,
    )

    # --- Tables: atomic chunks ---
    for table_md in page.tables:
        c = ChunkData(
            **base,
            chunk_index=chunk_index,
            chunk_type=ChunkType.table,
            raw_text=f"Table from page {page.page_number}",
            vision_caption=vision_caption,
            table_markdown=table_md,
        )
        c.enriched_text = build_enriched_text(c, course_name)
        chunks.append(c)
        chunk_index += 1

    # --- Code cells: atomic chunks ---
    for code in page.code_blocks:
        c = ChunkData(
            **base,
            chunk_index=chunk_index,
            chunk_type=ChunkType.code,
            raw_text=code,
            vision_caption=vision_caption,
        )
        c.enriched_text = build_enriched_text(c, course_name)
        chunks.append(c)
        chunk_index += 1

    # --- Slides and diagram-heavy pages: one chunk ---
    if page.slide_title is not None or chunk_type in (ChunkType.diagram, ChunkType.chart):
        c = ChunkData(
            **base,
            chunk_index=chunk_index,
            chunk_type=chunk_type,
            raw_text=page.raw_text,
            vision_caption=vision_caption,
        )
        c.enriched_text = build_enriched_text(c, course_name)
        chunks.append(c)
        return chunks

    # --- Text pages: sliding window chunking ---
    words = page.raw_text.split()
    if not words:
        # Vision-only page: single chunk from caption
        c = ChunkData(
            **base,
            chunk_index=chunk_index,
            chunk_type=ChunkType.diagram,
            raw_text="",
            vision_caption=vision_caption,
        )
        c.enriched_text = build_enriched_text(c, course_name)
        chunks.append(c)
        return chunks

    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        window = words[start: start + chunk_size]
        if len(window) < 20 and start > 0:
            break  # avoid tiny trailing chunks
        raw_text = " ".join(window)
        c = ChunkData(
            **base,
            chunk_index=chunk_index,
            chunk_type=chunk_type,
            raw_text=raw_text,
            vision_caption=vision_caption if start == 0 else "",  # caption only on first
        )
        c.enriched_text = build_enriched_text(c, course_name)
        chunks.append(c)
        chunk_index += 1

    return chunks
