"""
Quiz Generator

Flow:
  1. Professor specifies: question count, topics/lectures, difficulty mix, format, duration
  2. We retrieve relevant chunks for the requested scope
  3. LLM generates structured questions grounded in retrieved chunks
  4. Questions are stored with full provenance (chunk IDs → slide/page references)
  5. Quality flags are auto-detected (unclear wording, multiple answers, missing assumptions)
  6. Professor can accept/reject/edit questions inline
  7. Answer key with step-by-step explanations is included
  8. Mark scheme is suggested per question
  9. Multiple quiz versions can be generated from the same chunks
"""
import json
import uuid
from typing import Optional
import structlog
import anthropic

from app.core.config import settings
from app.retrieval.retriever import RetrievedChunk, retrieve
from app.db.models import (
    Quiz, QuizQuestion, QuestionType, QuestionDifficulty, QuestionStatus
)

log = structlog.get_logger()
_client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)


QUESTION_GENERATION_PROMPT = """You are an expert academic assessment designer for university courses.
You will generate quiz questions that are STRICTLY grounded in the provided course materials.
Do NOT invent facts, examples, or content not present in the sources.

COURSE MATERIALS:
{context}

QUIZ REQUIREMENTS:
- Total questions: {question_count}
- Duration: {duration_minutes} minutes
- Difficulty distribution: {difficulty_distribution}
- Question types allowed: {question_types}
- Topics to focus on: {topic_focus}
{special_request}

INSTRUCTIONS:
For each question produce a JSON object with exactly these fields:
{{
  "question_number": <int>,
  "question_type": <"mcq"|"short_answer"|"derivation"|"diagram_based"|"table_interpretation"|"true_false">,
  "difficulty": <"easy"|"medium"|"hard">,
  "estimated_minutes": <float — realistic time to answer>,
  "marks": <int — suggested marks>,
  "question_text": <string — the full question>,
  "options": <list of {{"label": "A", "text": "..."}} for MCQ, null for others>,
  "correct_answer": <string — the correct answer or expected answer outline>,
  "answer_explanation": <string — step-by-step reasoning as would appear in an answer key>,
  "mark_scheme": <list of {{"marks": int, "criteria": "..."}} — what earns each mark>,
  "source_references": <list of {{"lecture": int, "slide_or_page": int, "section": "..."}}>,
  "flags": <list — any of: "unclear_wording", "multiple_valid_answers", "missing_assumption", "diagram_required">,
  "difficulty_reason": <string — brief explanation of why this difficulty was assigned>
}}

IMPORTANT RULES:
- Every question must be traceable to specific content from the provided sources
- MCQ distractors must be plausible but clearly wrong based on the material
- For derivation questions, specify exactly what the student needs to show
- For diagram_based questions, describe the diagram in the question text (student should recognise it)
- Flag any question where a student could reasonably argue multiple answers
- Flag any question that assumes knowledge not in the provided materials
- Estimate time honestly — a derivation takes longer than an MCQ

Respond ONLY with a valid JSON array of question objects. No preamble, no markdown.
"""


MULTI_VERSION_PROMPT = """You are generating an alternative version of an existing quiz.
Keep the same question structure, topics, and difficulty — but change numbers, figures, variable names,
or specific examples so students cannot share answers.

ORIGINAL QUIZ QUESTIONS:
{original_questions}

COURSE MATERIALS (for grounding):
{context}

Generate exactly {question_count} questions following the same JSON schema.
Respond ONLY with a valid JSON array. No preamble, no markdown.
"""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Assembles retrieved chunks into a structured context string for the LLM."""
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        ref = f"Lecture {chunk.lecture_number}" if chunk.lecture_number else chunk.source_file
        if chunk.lecture_title:
            ref += f": {chunk.lecture_title}"
        if chunk.slide_number:
            ref += f", Slide {chunk.slide_number}"
        elif chunk.page_number:
            ref += f", Page {chunk.page_number}"
        if chunk.section_title:
            ref += f" — {chunk.section_title}"

        content_parts = []
        if chunk.raw_text:
            content_parts.append(chunk.raw_text)
        if chunk.vision_caption and chunk.chunk_type in ("diagram", "chart", "slide", "formula"):
            content_parts.append(f"[Visual: {chunk.vision_caption}]")
        if chunk.table_markdown:
            content_parts.append(f"[Table:\n{chunk.table_markdown}]")

        blocks.append(f"SOURCE [{i}] — {ref}\n{chr(10).join(content_parts)}")

    return "\n\n---\n\n".join(blocks)


def detect_flags(question: dict) -> list[str]:
    """
    Auto-detect quality issues in generated questions.
    Supplements any flags the LLM itself raised.
    """
    flags = list(question.get("flags") or [])
    text = question.get("question_text", "")

    # Vague language heuristics
    vague_terms = ["discuss", "explain briefly", "comment on", "describe in general"]
    if any(t in text.lower() for t in vague_terms) and question.get("question_type") == "mcq":
        if "unclear_wording" not in flags:
            flags.append("unclear_wording")

    # MCQ without 4 options
    if question.get("question_type") == "mcq":
        options = question.get("options") or []
        if len(options) < 4:
            if "missing_assumption" not in flags:
                flags.append("missing_assumption")

    # Derivation without mark scheme
    if question.get("question_type") == "derivation":
        if not question.get("mark_scheme"):
            flags.append("missing_assumption")

    return flags


async def generate_quiz(
    course_id: str,
    db,
    question_count: int,
    duration_minutes: int,
    difficulty_distribution: dict,      # {"easy": 30, "medium": 50, "hard": 20}
    question_types: list[str],
    topic_focus: Optional[str] = None,
    lecture_filter: Optional[list[int]] = None,
    special_request: Optional[str] = None,
) -> list[dict]:
    """
    Main quiz generation function.
    Returns a list of question dicts ready to be saved as QuizQuestion records.
    """
    # Build a rich query from topic focus or lecture filter
    query = topic_focus or "key concepts, definitions, formulas, and applications"
    if lecture_filter:
        query += f" from lectures {', '.join(str(l) for l in lecture_filter)}"

    # Retrieve grounding chunks — cast a wide net for quiz generation
    chunks = await retrieve(
        query=query,
        course_id=course_id,
        db=db,
        top_k=15,                   # more chunks = richer question variety
        lecture_filter=lecture_filter,
    )

    if not chunks:
        raise ValueError("No course content found for the requested scope.")

    context = build_context_block(chunks)

    # Build difficulty string for prompt
    diff_str = ", ".join(f"{pct}% {level}" for level, pct in difficulty_distribution.items())
    types_str = ", ".join(question_types)
    special_str = f"\nSPECIAL INSTRUCTION: {special_request}" if special_request else ""

    prompt = QUESTION_GENERATION_PROMPT.format(
        context=context,
        question_count=question_count,
        duration_minutes=duration_minutes,
        difficulty_distribution=diff_str,
        question_types=types_str,
        topic_focus=topic_focus or "all covered topics",
        special_request=special_str,
    )

    log.info("quiz_generation_start", question_count=question_count, chunks=len(chunks))

    response = await _client.messages.create(
        model="claude-opus-4-6",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if model added them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    questions_raw = json.loads(raw)

    # Build chunk ID lookup for provenance
    chunk_id_by_ref = {i + 1: chunk.chunk_id for i, chunk in enumerate(chunks)}

    # Enrich questions with provenance and flag detection
    enriched = []
    for q in questions_raw:
        q["flags"] = detect_flags(q)
        # Map source references back to chunk IDs
        q["source_chunk_ids"] = [
            chunk_id_by_ref.get(ref.get("source_index", 0))
            for ref in (q.get("source_references") or [])
            if ref.get("source_index") in chunk_id_by_ref
        ]
        enriched.append(q)

    log.info("quiz_generation_done", questions=len(enriched))
    return enriched


async def generate_quiz_version(
    original_questions: list[dict],
    course_id: str,
    db,
    query: str = "all course content",
) -> list[dict]:
    """
    Generates an alternative version of an existing quiz.
    Same structure and difficulty — different specific content (numbers, examples).
    """
    chunks = await retrieve(query=query, course_id=course_id, db=db, top_k=15)
    context = build_context_block(chunks)

    original_json = json.dumps(
        [{k: v for k, v in q.items() if k not in ("source_chunk_ids",)}
         for q in original_questions],
        indent=2
    )

    prompt = MULTI_VERSION_PROMPT.format(
        original_questions=original_json,
        context=context,
        question_count=len(original_questions),
    )

    response = await _client.messages.create(
        model="claude-opus-4-6",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    questions_raw = json.loads(raw)
    for q in questions_raw:
        q["flags"] = detect_flags(q)
    return questions_raw
