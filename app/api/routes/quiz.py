"""
Quiz routes — full professor workflow:

  POST   /courses/{id}/quizzes              — generate a new quiz
  GET    /courses/{id}/quizzes              — list quizzes for course
  GET    /courses/{id}/quizzes/{qid}        — full quiz with questions + answer key
  PATCH  /courses/{id}/quizzes/{qid}/questions/{question_id}  — accept/reject/edit a question
  POST   /courses/{id}/quizzes/{qid}/versions                 — generate alternative version
  POST   /courses/{id}/quizzes/{qid}/analytics                — record student performance
  GET    /courses/{id}/quizzes/{qid}/analytics                — question analytics
"""
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import (
    Quiz, QuizQuestion, QuestionType, QuestionDifficulty, QuestionStatus, Course
)
from app.quiz.generator import generate_quiz, generate_quiz_version

router = APIRouter(tags=["Quiz"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class GenerateQuizRequest(BaseModel):
    title: Optional[str] = None
    question_count: int = 10
    duration_minutes: int = 60
    difficulty_distribution: dict = {"easy": 30, "medium": 50, "hard": 20}
    question_types: list[str] = ["mcq", "short_answer"]
    topic_focus: Optional[str] = None
    lecture_filter: Optional[list[int]] = None       # e.g. [1, 2, 3] — only these lectures
    special_request: Optional[str] = None            # e.g. "include one question on the chain rule"
    allow_multiple_versions: bool = False


class QuestionReviewRequest(BaseModel):
    professor_status: str                            # "accepted" | "rejected" | "edited"
    question_text: Optional[str] = None             # if edited
    correct_answer: Optional[str] = None
    options: Optional[list[dict]] = None
    professor_note: Optional[str] = None


class RecordAnswerRequest(BaseModel):
    """Called after students complete the quiz to record performance."""
    question_results: list[dict]    # [{"question_id": "...", "correct": true}, ...]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/courses/{course_id}/quizzes", status_code=status.HTTP_201_CREATED)
async def generate_quiz_endpoint(
    course_id: str,
    body: GenerateQuizRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a quiz from course materials.
    Returns all questions with answer key, mark scheme, source references, and flags.
    """
    course = await db.get(Course, uuid.UUID(course_id))
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    # Validate difficulty distribution sums to ~100
    total_pct = sum(body.difficulty_distribution.values())
    if not (95 <= total_pct <= 105):
        raise HTTPException(
            status_code=422,
            detail="difficulty_distribution percentages must sum to ~100"
        )

    # Validate question types
    valid_types = {t.value for t in QuestionType}
    for qt in body.question_types:
        if qt not in valid_types:
            raise HTTPException(status_code=422, detail=f"Invalid question type: {qt}")

    # Generate questions via LLM
    questions_data = await generate_quiz(
        course_id=course_id,
        db=db,
        question_count=body.question_count,
        duration_minutes=body.duration_minutes,
        difficulty_distribution=body.difficulty_distribution,
        question_types=body.question_types,
        topic_focus=body.topic_focus,
        lecture_filter=body.lecture_filter,
        special_request=body.special_request,
    )

    # Persist quiz
    quiz = Quiz(
        course_id=course.id,
        title=body.title or f"Quiz — {body.topic_focus or 'All Topics'}",
        topic_filter=body.lecture_filter or [],
        question_count_requested=body.question_count,
        duration_minutes=body.duration_minutes,
        difficulty_distribution=body.difficulty_distribution,
        question_types_requested=body.question_types,
        allow_multiple_versions=body.allow_multiple_versions,
    )
    db.add(quiz)
    await db.flush()

    saved_questions = []
    for q in questions_data:
        qq = QuizQuestion(
            quiz_id=quiz.id,
            question_number=q.get("question_number"),
            question_type=q.get("question_type"),
            difficulty=q.get("difficulty"),
            estimated_minutes=q.get("estimated_minutes"),
            marks=q.get("marks", 1),
            question_text=q.get("question_text"),
            options=q.get("options"),
            correct_answer=q.get("correct_answer"),
            answer_explanation=q.get("answer_explanation"),
            mark_scheme=q.get("mark_scheme"),
            source_chunk_ids=q.get("source_chunk_ids", []),
            source_references=q.get("source_references", []),
            flags=q.get("flags", []),
            professor_status=QuestionStatus.pending_review,
        )
        db.add(qq)
        saved_questions.append(qq)

    await db.commit()

    return _format_quiz_response(quiz, saved_questions, include_answers=True)


@router.get("/courses/{course_id}/quizzes")
async def list_quizzes(course_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Quiz).where(Quiz.course_id == uuid.UUID(course_id))
        .order_by(Quiz.created_at.desc())
    )
    quizzes = result.scalars().all()
    return [
        {
            "id": str(q.id),
            "title": q.title,
            "status": q.status,
            "question_count": q.question_count_requested,
            "duration_minutes": q.duration_minutes,
            "version_number": q.version_number,
            "generated_at": q.generated_at.isoformat(),
        }
        for q in quizzes
    ]


@router.get("/courses/{course_id}/quizzes/{quiz_id}")
async def get_quiz(
    course_id: str,
    quiz_id: str,
    include_answers: bool = True,       # set False to get student-facing version
    db: AsyncSession = Depends(get_db),
):
    """Full quiz — with or without answer key depending on include_answers flag."""
    quiz = await db.get(Quiz, uuid.UUID(quiz_id))
    if not quiz or str(quiz.course_id) != course_id:
        raise HTTPException(status_code=404, detail="Quiz not found")

    result = await db.execute(
        select(QuizQuestion)
        .where(QuizQuestion.quiz_id == quiz.id)
        .order_by(QuizQuestion.question_number)
    )
    questions = result.scalars().all()

    return _format_quiz_response(quiz, questions, include_answers=include_answers)


@router.patch("/courses/{course_id}/quizzes/{quiz_id}/questions/{question_id}")
async def review_question(
    course_id: str,
    quiz_id: str,
    question_id: str,
    body: QuestionReviewRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Professor accepts, rejects, or edits a question inline.
    Edited questions get status 'edited' and store the original alongside changes.
    """
    question = await db.get(QuizQuestion, uuid.UUID(question_id))
    if not question or str(question.quiz_id) != quiz_id:
        raise HTTPException(status_code=404, detail="Question not found")

    valid_statuses = {"accepted", "rejected", "edited"}
    if body.professor_status not in valid_statuses:
        raise HTTPException(status_code=422, detail=f"Status must be one of {valid_statuses}")

    question.professor_status = body.professor_status
    if body.question_text:
        question.question_text = body.question_text
    if body.correct_answer:
        question.correct_answer = body.correct_answer
    if body.options:
        question.options = body.options
    if body.professor_note:
        question.professor_note = body.professor_note

    await db.commit()
    return {"id": str(question.id), "professor_status": question.professor_status}


@router.post("/courses/{course_id}/quizzes/{quiz_id}/versions",
             status_code=status.HTTP_201_CREATED)
async def generate_version(
    course_id: str,
    quiz_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Generates an alternative version of an existing quiz.
    Same structure, difficulty, and topics — different specific content.
    Useful for preventing answer sharing between students.
    """
    original_quiz = await db.get(Quiz, uuid.UUID(quiz_id))
    if not original_quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")

    original_qs_result = await db.execute(
        select(QuizQuestion).where(QuizQuestion.quiz_id == original_quiz.id)
    )
    original_questions = original_qs_result.scalars().all()
    original_dicts = [
        {
            "question_number": q.question_number,
            "question_type": q.question_type,
            "difficulty": q.difficulty,
            "question_text": q.question_text,
            "options": q.options,
            "correct_answer": q.correct_answer,
            "estimated_minutes": q.estimated_minutes,
        }
        for q in original_questions
    ]

    new_questions_data = await generate_quiz_version(
        original_questions=original_dicts,
        course_id=course_id,
        db=db,
        query=original_quiz.title or "all course content",
    )

    # Create new quiz as a version of the original
    new_quiz = Quiz(
        course_id=original_quiz.course_id,
        title=f"{original_quiz.title} (Version {original_quiz.version_number + 1})",
        topic_filter=original_quiz.topic_filter,
        question_count_requested=original_quiz.question_count_requested,
        duration_minutes=original_quiz.duration_minutes,
        difficulty_distribution=original_quiz.difficulty_distribution,
        question_types_requested=original_quiz.question_types_requested,
        version_number=original_quiz.version_number + 1,
        parent_quiz_id=original_quiz.id,
    )
    db.add(new_quiz)
    await db.flush()

    saved = []
    for q in new_questions_data:
        qq = QuizQuestion(
            quiz_id=new_quiz.id,
            question_number=q.get("question_number"),
            question_type=q.get("question_type"),
            difficulty=q.get("difficulty"),
            estimated_minutes=q.get("estimated_minutes"),
            marks=q.get("marks", 1),
            question_text=q.get("question_text"),
            options=q.get("options"),
            correct_answer=q.get("correct_answer"),
            answer_explanation=q.get("answer_explanation"),
            mark_scheme=q.get("mark_scheme"),
            source_references=q.get("source_references", []),
            flags=q.get("flags", []),
        )
        db.add(qq)
        saved.append(qq)

    await db.commit()
    return _format_quiz_response(new_quiz, saved, include_answers=True)


@router.post("/courses/{course_id}/quizzes/{quiz_id}/analytics")
async def record_quiz_performance(
    course_id: str,
    quiz_id: str,
    body: RecordAnswerRequest,
    db: AsyncSession = Depends(get_db),
):
    """Record student performance after quiz completion."""
    for result in body.question_results:
        question = await db.get(QuizQuestion, uuid.UUID(result["question_id"]))
        if not question:
            continue
        question.times_answered = (question.times_answered or 0) + 1
        if result.get("correct"):
            question.times_correct = (question.times_correct or 0) + 1

    await db.commit()
    return {"message": "Performance recorded"}


@router.get("/courses/{course_id}/quizzes/{quiz_id}/analytics")
async def get_quiz_analytics(
    course_id: str,
    quiz_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Question performance analytics:
    - % correct per question
    - Most-missed questions
    - Performance by difficulty
    - Which concepts students are struggling with
    """
    result = await db.execute(
        select(QuizQuestion).where(QuizQuestion.quiz_id == uuid.UUID(quiz_id))
    )
    questions = result.scalars().all()

    question_stats = []
    difficulty_buckets = {"easy": [], "medium": [], "hard": []}

    for q in questions:
        answered = q.times_answered or 0
        correct = q.times_correct or 0
        pct = round((correct / answered) * 100) if answered else None

        stat = {
            "question_id": str(q.id),
            "question_number": q.question_number,
            "question_type": q.question_type,
            "difficulty": q.difficulty,
            "times_answered": answered,
            "times_correct": correct,
            "percent_correct": pct,
            "source_references": q.source_references,
        }
        question_stats.append(stat)

        if q.difficulty and pct is not None:
            difficulty_buckets[q.difficulty].append(pct)

    # Sort by % correct ascending (most missed first)
    answered_qs = [s for s in question_stats if s["percent_correct"] is not None]
    most_missed = sorted(answered_qs, key=lambda x: x["percent_correct"])[:5]

    def avg(lst):
        return round(sum(lst) / len(lst)) if lst else None

    return {
        "quiz_id": quiz_id,
        "questions": question_stats,
        "summary": {
            "most_missed_questions": most_missed,
            "average_score_by_difficulty": {
                "easy":   avg(difficulty_buckets["easy"]),
                "medium": avg(difficulty_buckets["medium"]),
                "hard":   avg(difficulty_buckets["hard"]),
            },
            "struggling_topics": [
                {
                    "references": q["source_references"],
                    "percent_correct": q["percent_correct"],
                }
                for q in most_missed
            ],
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_quiz_response(quiz: Quiz, questions: list[QuizQuestion], include_answers: bool):
    total_minutes = sum(q.estimated_minutes or 0 for q in questions)
    difficulty_counts = {}
    for q in questions:
        d = str(q.difficulty or "unknown")
        difficulty_counts[d] = difficulty_counts.get(d, 0) + 1

    formatted_questions = []
    for q in questions:
        qd = {
            "id": str(q.id),
            "question_number": q.question_number,
            "question_type": q.question_type,
            "difficulty": q.difficulty,
            "difficulty_reason": None,
            "estimated_minutes": q.estimated_minutes,
            "marks": q.marks,
            "question_text": q.question_text,
            "options": q.options,
            "image_url": q.image_url,
            "source_references": q.source_references,
            "flags": q.flags,
            "professor_status": q.professor_status,
            "professor_note": q.professor_note,
        }
        if include_answers:
            qd["correct_answer"] = q.correct_answer
            qd["answer_explanation"] = q.answer_explanation
            qd["mark_scheme"] = q.mark_scheme

        formatted_questions.append(qd)

    return {
        "id": str(quiz.id),
        "title": quiz.title,
        "status": quiz.status,
        "version_number": quiz.version_number,
        "parent_quiz_id": str(quiz.parent_quiz_id) if quiz.parent_quiz_id else None,
        "settings": {
            "question_count": len(questions),
            "duration_minutes": quiz.duration_minutes,
            "estimated_total_minutes": round(total_minutes, 1),
            "difficulty_distribution": difficulty_counts,
            "question_types": quiz.question_types_requested,
        },
        "questions": formatted_questions,
        "flagged_count": sum(1 for q in questions if q.flags),
        "pending_review_count": sum(
            1 for q in questions if q.professor_status == QuestionStatus.pending_review
        ),
        "generated_at": quiz.generated_at.isoformat(),
    }
