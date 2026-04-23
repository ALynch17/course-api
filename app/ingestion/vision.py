"""
Vision pass: runs a vision model on every rendered page/slide image.
Generates a rich educational description that becomes the primary
searchable content for visual-heavy pages (diagrams, charts, formulas).

Concurrency is controlled via an asyncio Semaphore to avoid flooding
the vision API — respects VISION_CONCURRENCY from settings.
"""
import asyncio
import base64
from typing import Optional
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import anthropic

from app.core.config import settings
from app.ingestion.parsers import ParsedPage

log = structlog.get_logger()

# One shared semaphore across all workers — caps concurrent vision API calls
_vision_semaphore = asyncio.Semaphore(settings.VISION_CONCURRENCY)

# Anthropic client (async)
_client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

VISION_PROMPT = """You are processing course lecture material for a student Q&A and quiz system.

Analyze this page or slide and produce a detailed educational description that will be stored
as searchable text. Your description must be rich enough that a student's question about
the concept taught here can be matched to this page — even if the page contains no paragraph text.

Include ALL of the following that are present:

1. CORE CONCEPT: What is the main idea or topic being taught on this page?
2. TEXT CONTENT: Every piece of text visible — labels, axis titles, legend entries, callouts, annotations
3. VISUAL ELEMENTS: Describe any diagram, chart, figure, or illustration in detail:
   - What type of visual is it? (flowchart, bar chart, neural network diagram, chemical structure, etc.)
   - What does it show and what does it mean educationally?
   - Describe relationships, flows, arrows, and connections between elements
4. FORMULAS & EQUATIONS: Write out every formula both in symbols AND in plain English description
   (e.g. "δL/δw — the partial derivative of the loss with respect to weight, used in gradient descent")
5. TABLES: Describe what the table shows, its columns, and key data patterns
6. CODE: If code is shown, describe what it does and what concept it demonstrates
7. TAKEAWAY: What should a student understand or be able to do after studying this page?

Be specific and educational. Do not just describe what you see — explain what it means.
"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((anthropic.APIError, asyncio.TimeoutError)),
)
async def run_vision_pass(
    page: ParsedPage,
    document_context: str = "",
) -> str:
    """
    Runs a vision model on a single page image.
    Returns a rich text description to be stored alongside the page.

    document_context: e.g. "Lecture 5: Backpropagation — Deep Learning 401"
    Prepending context helps the model ground its description.
    """
    if page.image_bytes is None:
        # No image available — return enriched version of raw text
        log.info("vision_pass_skipped_no_image", page=page.page_number)
        return _fallback_description(page)

    async with _vision_semaphore:
        log.info("vision_pass_start", page=page.page_number)

        image_b64 = base64.standard_b64encode(page.image_bytes).decode("utf-8")

        prompt_parts = []
        if document_context:
            prompt_parts.append(f"Document context: {document_context}\n")
        if page.slide_title:
            prompt_parts.append(f"Slide title: {page.slide_title}\n")
        if page.section_heading:
            prompt_parts.append(f"Section heading: {page.section_heading}\n")
        prompt_parts.append(VISION_PROMPT)

        full_prompt = "\n".join(prompt_parts)

        response = await _client.messages.create(
            model=settings.VISION_MODEL,
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": full_prompt},
                ],
            }],
        )

        caption = response.content[0].text
        log.info("vision_pass_done", page=page.page_number, caption_length=len(caption))
        return caption


def _fallback_description(page: ParsedPage) -> str:
    """When no image is available, build a description from structured text fields."""
    parts = []
    if page.slide_title:
        parts.append(f"Slide title: {page.slide_title}")
    if page.section_heading:
        parts.append(f"Section: {page.section_heading}")
    if page.raw_text:
        parts.append(page.raw_text)
    if page.slide_notes:
        parts.append(f"Speaker notes: {page.slide_notes}")
    if page.code_blocks:
        parts.append("Code blocks: " + " | ".join(page.code_blocks[:3]))
    return "\n".join(parts) or "No content extracted."


async def run_vision_passes_batch(
    pages: list[ParsedPage],
    document_context: str = "",
) -> list[str]:
    """
    Runs vision passes on a batch of pages concurrently.
    The semaphore inside run_vision_pass caps actual concurrency.
    """
    tasks = [
        run_vision_pass(page, document_context)
        for page in pages
    ]
    captions = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for i, caption in enumerate(captions):
        if isinstance(caption, Exception):
            log.error("vision_pass_failed", page=pages[i].page_number, error=str(caption))
            results.append(_fallback_description(pages[i]))
        else:
            results.append(caption)

    return results
