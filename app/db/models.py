"""
SQLAlchemy models for all core entities.
pgvector is used for the embedding columns.
"""
import uuid
from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Float, ForeignKey,
    Integer, JSON, String, Text, Enum as SAEnum
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship
import enum


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CourseStatus(str, enum.Enum):
    active = "active"
    archived = "archived"


class DocumentStatus(str, enum.Enum):
    uploaded = "uploaded"
    queued = "queued"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class JobStatus(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"


class PageStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    failed = "failed"


class ChunkType(str, enum.Enum):
    text = "text"
    diagram = "diagram"
    chart = "chart"
    table = "table"
    formula = "formula"
    code = "code"
    slide = "slide"
    image = "image"
    mixed = "mixed"


class QuestionType(str, enum.Enum):
    mcq = "mcq"
    short_answer = "short_answer"
    derivation = "derivation"
    diagram_based = "diagram_based"
    table_interpretation = "table_interpretation"
    true_false = "true_false"


class QuestionDifficulty(str, enum.Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class QuestionStatus(str, enum.Enum):
    pending_review = "pending_review"
    accepted = "accepted"
    rejected = "rejected"
    edited = "edited"


# ---------------------------------------------------------------------------
# Course
# ---------------------------------------------------------------------------

class Course(Base):
    __tablename__ = "courses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(SAEnum(CourseStatus), default=CourseStatus.active)
    settings = Column(JSON, default={})             # chunk_size, vision_model overrides etc
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    documents = relationship("Document", back_populates="course", cascade="all, delete-orphan")
    jobs = relationship("IngestionJob", back_populates="course", cascade="all, delete-orphan")
    quizzes = relationship("Quiz", back_populates="course", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    course_id = Column(UUID(as_uuid=True), ForeignKey("courses.id"), nullable=False)
    job_id = Column(UUID(as_uuid=True), ForeignKey("ingestion_jobs.id"))

    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500), nullable=False)
    file_type = Column(String(50))                  # pdf, pptx, docx, ipynb, png ...
    size_bytes = Column(BigInteger)
    s3_key = Column(String(1000))                   # location of raw file in S3
    status = Column(SAEnum(DocumentStatus), default=DocumentStatus.uploaded)

    # Populated after parsing
    page_count = Column(Integer)
    slide_count = Column(Integer)
    chunk_count = Column(Integer, default=0)
    lecture_number = Column(Integer)                # e.g. Lecture 3
    lecture_title = Column(String(500))

    processing_stats = Column(JSON, default={})     # vision passes run, tables found etc
    error_message = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    course = relationship("Course", back_populates="documents")
    pages = relationship("PageRecord", back_populates="document", cascade="all, delete-orphan")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# Page-level tracking — enables resumable ingestion
# ---------------------------------------------------------------------------

class PageRecord(Base):
    """
    One row per page/slide. Lets us track and resume at page granularity.
    If a worker crashes on page 847, we requeue only failed/pending pages.
    """
    __tablename__ = "page_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)

    status = Column(SAEnum(PageStatus), default=PageStatus.pending)
    retries = Column(Integer, default=0)
    error_message = Column(Text)

    # Raw outputs stored here before chunking
    raw_text = Column(Text)
    vision_caption = Column(Text)
    s3_image_key = Column(String(1000))             # rendered page image in S3

    processed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="pages")


# ---------------------------------------------------------------------------
# Chunk — the core retrieval unit
# ---------------------------------------------------------------------------

class Chunk(Base):
    """
    A single retrievable piece of content. Every chunk has:
    - Source provenance (file, page, section, heading)
    - Dual representation: raw text + vision caption
    - Embedding vector for semantic search
    - Rendered image reference for visual chunks
    """
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    course_id = Column(UUID(as_uuid=True), ForeignKey("courses.id"), nullable=False)

    # Provenance — every chunk knows exactly where it came from
    source_file = Column(String(500))
    page_number = Column(Integer)
    slide_number = Column(Integer)
    section_title = Column(String(500))
    chunk_index = Column(Integer)                   # position within the page
    lecture_number = Column(Integer)
    lecture_title = Column(String(500))

    # Content
    chunk_type = Column(SAEnum(ChunkType), default=ChunkType.text)
    raw_text = Column(Text)                         # extracted text layer
    vision_caption = Column(Text)                   # vision model description
    enriched_text = Column(Text)                    # combined: metadata + raw + caption
    table_markdown = Column(Text)                   # if chunk is a table
    latex_formula = Column(Text)                    # if chunk contains formulas

    # Visual
    s3_image_key = Column(String(1000))             # rendered image for diagram/chart chunks
    has_visual = Column(Boolean, default=False)

    # Embedding (text-embedding-3-large = 3072 dims)
    embedding = Column(Vector(1536))

    token_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")


# ---------------------------------------------------------------------------
# Ingestion Job
# ---------------------------------------------------------------------------

class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    course_id = Column(UUID(as_uuid=True), ForeignKey("courses.id"), nullable=False)

    status = Column(SAEnum(JobStatus), default=JobStatus.queued)
    current_stage = Column(String(100))             # parsing | vision_pass | chunking | embedding | indexing
    total_pages = Column(Integer, default=0)
    processed_pages = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    indexed_chunks = Column(Integer, default=0)

    error_message = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    course = relationship("Course", back_populates="jobs")
    documents = relationship("Document", back_populates=None,
                             primaryjoin="IngestionJob.id == foreign(Document.job_id)")


# ---------------------------------------------------------------------------
# Quiz
# ---------------------------------------------------------------------------

class Quiz(Base):
    __tablename__ = "quizzes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    course_id = Column(UUID(as_uuid=True), ForeignKey("courses.id"), nullable=False)

    title = Column(String(500))
    topic_filter = Column(JSON, default=[])         # lecture numbers / topic strings requested
    question_count_requested = Column(Integer)
    duration_minutes = Column(Integer)
    difficulty_distribution = Column(JSON)          # { easy: 30, medium: 50, hard: 20 } (%)
    question_types_requested = Column(JSON, default=[])
    allow_multiple_versions = Column(Boolean, default=False)
    version_number = Column(Integer, default=1)
    parent_quiz_id = Column(UUID(as_uuid=True), ForeignKey("quizzes.id"), nullable=True)

    status = Column(String(50), default="draft")    # draft | published | closed
    performance_stats = Column(JSON, default={})    # filled after quiz is taken

    generated_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    course = relationship("Course", back_populates="quizzes")
    questions = relationship("QuizQuestion", back_populates="quiz", cascade="all, delete-orphan")


class QuizQuestion(Base):
    __tablename__ = "quiz_questions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    quiz_id = Column(UUID(as_uuid=True), ForeignKey("quizzes.id"), nullable=False)

    question_number = Column(Integer)
    question_type = Column(SAEnum(QuestionType))
    difficulty = Column(SAEnum(QuestionDifficulty))
    estimated_minutes = Column(Float)               # expected time per question
    marks = Column(Integer, default=1)

    question_text = Column(Text, nullable=False)
    options = Column(JSON)                          # MCQ options: [{label: "A", text: "..."}]
    correct_answer = Column(Text)
    answer_explanation = Column(Text)               # step-by-step reasoning for answer key
    mark_scheme = Column(JSON)                      # [{marks: 1, criteria: "..."}, ...]
    image_url = Column(String(1000))                # for diagram-based questions

    # Provenance — traces back to actual course content
    source_chunk_ids = Column(JSON, default=[])
    source_references = Column(JSON, default=[])    # [{lecture, slide, section, page}]

    # Quality flags
    flags = Column(JSON, default=[])                # ["unclear_wording", "multiple_answers", "missing_assumption"]
    professor_status = Column(SAEnum(QuestionStatus), default=QuestionStatus.pending_review)
    professor_note = Column(Text)                   # inline edit note from professor

    # Analytics
    times_answered = Column(Integer, default=0)
    times_correct = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    quiz = relationship("Quiz", back_populates="questions")


# ---------------------------------------------------------------------------
# Upload Session — for chunked file uploads
# ---------------------------------------------------------------------------

class UploadSession(Base):
    __tablename__ = "upload_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    course_id = Column(UUID(as_uuid=True), ForeignKey("courses.id"), nullable=False)
    filename = Column(String(500), nullable=False)
    size_bytes = Column(BigInteger)
    total_chunks = Column(Integer)
    received_chunks = Column(JSON, default=[])      # list of chunk indices received
    s3_upload_id = Column(String(500))              # S3 multipart upload ID
    status = Column(String(50), default="in_progress")
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
