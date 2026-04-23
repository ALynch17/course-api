## Course Intelligence API

A backend API that ingests lecture materials and powers AI-driven quiz generation, with a foundation ready for an AI Tutor. It is built for a use case where a professor uploads 10-12 lecture files over a semester and needs to generate quizzes and answer cross-lecture questions during finals week.

---

## Live Demo

**API:** https://course-api-production.up.railway.app
**Documentation:** https://course-api-production.up.railway.app/docs
**Demo Video:** https://drive.google.com/file/d/1rHNQqz2_9rqFkzb7q5TLGETvDP0K_MLr/view?usp=sharing 

No setup required

---

## Table of Contents

1. [What This System Does](#what-this-system-does)
2. [Architecture Overview](#architecture-overview)
3. [Why I Made These Choices](#why-i-made-these-choices)
4. [Setup Instructions](#setup-instructions)
5. [Running the API](#running-the-api)
6. [API Reference](#api-reference)
7. [How the Ingestion Pipeline Works](#how-the-ingestion-pipeline-works)
8. [How Retrieval Works](#how-retrieval-works)
9. [How the Quiz Generator Works](#how-the-quiz-generator-works)
10. [AI Tutor - How It Would Be Built](#ai-tutor--how-it-would-be-built)
11. [What Breaks](#what-breaks)
12. [What I would Do Differently](#what-i-would-do-differently)
13. [What I Would Build Next](#what-i-would-build-next)
14. [Project Structure](#project-structure)

---

## What This System Does

A professor uploads lecture files including PDFs, PowerPoints, Word documents, Jupyter notebooks. The system reads every page, runs an AI vision model on each one to understand diagrams and charts that have no text, breaks everything into searchable chunks, and stores them in a vector database. Afterwards, a professor can generate a quiz at any time. They have the ability to specify how many questions, what difficulty, what lectures, what question formats etc. and the system produces grounded questions with answer keys, mark schemes, and citations back to specific slides and pages. Every question traces back to something actually taught.

---

## Architecture Overview

```
                        Professor / Student
                               │
                         HTTP Request
                               │
                     ┌─────────▼─────────┐
                     │     FastAPI        │  Python API server
                     │  Railway :8000     │  (deployed on Railway)
                     └─────────┬─────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                     │
   ┌──────▼──────┐    ┌────────▼────────┐   ┌───────▼───────┐
   │  Supabase   │    │    AWS S3       │   │  Upstash Redis │
   │ PostgreSQL  │    │  File Storage   │   │  Task Queue    │
   │ + pgvector  │    │                 │   │                │
   └─────────────┘    └─────────────────┘   └───────┬───────┘
                                                     │
                                            ┌────────▼────────┐
                                            │  Celery Worker  │
                                            │ (Railway Linux) │
                                            │ concurrency=4   │
                                            └────────┬────────┘
                                                     │
                                    ┌────────────────┼────────────────┐
                                    │                                  │
                           ┌────────▼────────┐              ┌─────────▼───────┐
                           │    Anthropic    │              │     OpenAI      │
                           │    (Claude)     │              │   Embeddings    │
                           │ Vision + Quiz   │              │text-embedding   │
                           └─────────────────┘              └─────────────────┘
```

Layers:

1. FastAPI - the webframework that turns python functions into HTTP endpoints. This handles all routing, request validation, and response formatting. It also auto-generates the '/docs' Swagger UI.

2. Supabase - a managed PostgreSQL database in the cloud. It stores all structured data including courses, documents, chunks, jobs, quizzes, and questions. It also runs pgvector which is a PostgreSQL extension that adds vector storage and similarity search. This addition is what makes semantic search possible without a separate vector database service.

3. AWS S3 - the cloud file storage that stores raw uploaded documents and the rendered PNG images of every page. The database stores text and metadata while S3 stores binary files. They are kept separate because databases are not designed for large binary files.

4. Upstash Redis -  the cloud message queue that enables the API to drop a task message into Redis and return immediately when a document is uploaded. The celery worker picks it up and processes in the background. This keeps the API fast regardless of document size.

5. Celery Worker - a background task processor deployed as a separate Railway service. It processes pages in parallel with '--concurrency=4' on Linux - four pages being vision-processed simultaneously.

6. Anthropic Claude - This was used for two things. Firstly, it was used to perform a vision pass on every page image during ingestion. Claude reads each page as an image and writes a rich educational description of what it contains, including diagrams, charts, and formulas that have no extractable text. Secondly, it was used for the quiz generation. Claude receives the retrieved course content and generates structured questions with answer keys.

7. OpenAI - This was used only for embeddings. Every chunk of text is converted into a list of 1536 numbers (a vector) that represents its meaning. These vectors are what enable semantic search - finding content by meaning rather than exact keywords.

---

## Why I Made These Choices

1) FastAPI was used because it is async-native. This means that it can handle multiple requests concurrently without blocking. It also auto-generates interactive API documentation (the '/docs' page) and validates all request/response shapes automatically using Python type hints. Async support matters for an AI backend that makes a lot of external API calls.

2) Supabase was chosen because it is a managed service with no installation or maintenance required. The pgvector is also already enabled and you can see your data in a UI. This made the project implementation a bit simpler as the infrastructure complexity was removed. However, for real production, a dedicated managed PostgreSQL on AWS RDS or similar would be more appropriate.

3) Pgvector was used because each chunk has rich metadata such as the lecture number, page, section, and document type, and that metadata needs to be filtered on during search. Pgvector allows for vector search and relational filtering in one query. Dedicated vector databases usually do not support complex filtering well or require the management of a separate service. This also reduced the complexity.

4) The most important design decision in the system was having dual representation (text and vision caption). This was done because there may be scenarios in lecture notes where the key concept is entirely in a diagram with no extractable text. By running Claude on every page image and storing the description alongside the raw text, every page becomes searchable no matter how visual.

5) A cross-encoder reranker is used after vector search as vector search finds chunks that are semantically close to the query but being semantically close is not the same as actually answering the question. For example, a chunk about dropout regularization is semantically close to a question about overfitting, but might not directly answer it. The cross-encoder reads the query and each candidate chunk together and scores how well the chunk answers the question. The two-step approach of retrieving broadly then reranking consistently produces better results than pure vector search alone.

6) A hybrid search (vector + keyword) was used because a pure vector search misses exact matches. For example, if a student asks about a specific formula name or a defined term from the lecture, vector search might not rank that chunk highly because the embedding captures general meaning, not exact terminology. Keyword search (BM25-Style) catches exact matches. Combining both with Reciprocal Rank Fusion gives the best of both approaches.

7) Celery and Redis were used for ingestion because without background processing, uploading a 150-page pdf would block the API for 10+ minutes. Celery lets the API return immediately with a job ID while workers process pages in parallel in the background. On Railway's Linux servers, 4 workers run simultaneously.

8) 1536 embedding dimensions were used as Supabase's hosted pgvector has a hard limit of 2000 dimensions for vector indexes. OpenAI's 'text-embedding-3-large' supports Matryoshka embeddings which allows us to request fewer dimensions without significant quality loss. 1536 dimensions gives strong retrieval quality while staying within Supabase's limit.

---

## Setup Instructions

A .env file with all credentials can be provided separately upon request. Place it in the root of the project folder before running.

The only thing you need installed is Python 3.12, which can be downloaded from python.org/downloads.



### Step 1 — Get the Code

```bash
git clone https://github.com/ALynch17/course-api.git
cd course-api
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
pip install supabase nest_asyncio psycopg2-binary asyncpg
```

## Running the API

1) Use the live Railway deployment:  https://course-api-production.up.railway.app/docs

2) Run locally with the provided information below

```bash
uvicorn app.main:app --reload --port 8000
```

Wait for:
```
INFO:     Application startup complete.
```

Open your browser at `http://localhost:8000/docs` — this is the full interactive API where you can test every endpoint.

> **Network note:** This application connects to Supabase, AWS, Anthropic, and OpenAI. Corporate or university networks often block these connections. Use a personal network or mobile hotspot.

---

## API Reference

### Courses
| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/courses` | Create a course namespace |
| GET | `/api/v1/courses` | List all courses |
| GET | `/api/v1/courses/{id}` | Course details + chunk count |
| DELETE | `/api/v1/courses/{id}` | Delete course and all associated data |

### Documents
| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/courses/{id}/documents` | Upload documents — triggers ingestion |
| GET | `/api/v1/courses/{id}/documents` | List all documents |
| GET | `/api/v1/courses/{id}/documents/{docId}` | Document details + page-level breakdown |
| DELETE | `/api/v1/courses/{id}/documents/{docId}` | Remove document |
| POST | `/api/v1/courses/{id}/documents/{docId}/retry` | Retry any failed pages |
| POST | `/api/v1/courses/{id}/uploads/initiate` | Start a chunked upload session |
| POST | `/api/v1/courses/{id}/uploads/{uploadId}/complete` | Finish chunked upload |

### Jobs
| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/courses/{id}/jobs` | All ingestion jobs for a course |
| GET | `/api/v1/courses/{id}/jobs/{jobId}` | Progress, stage, per-document status |

### Quiz
| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/courses/{id}/quizzes` | Generate quiz from course content |
| GET | `/api/v1/courses/{id}/quizzes` | List all quizzes |
| GET | `/api/v1/courses/{id}/quizzes/{qid}` | Full quiz with answer key |
| PATCH | `/api/v1/courses/{id}/quizzes/{qid}/questions/{questionId}` | Accept / reject / edit a question |
| POST | `/api/v1/courses/{id}/quizzes/{qid}/versions` | Generate an alternative version |
| POST | `/api/v1/courses/{id}/quizzes/{qid}/analytics` | Record student performance |
| GET | `/api/v1/courses/{id}/quizzes/{qid}/analytics` | Performance breakdown |

---

## How the Ingestion Pipeline Works

```
Professor uploads file(s)
        │
        ▼
POST /documents
  ├── Save raw file to AWS S3
  ├── Create Document + IngestionJob records in Supabase
  ├── Drop task into Upstash Redis queue
  └── Return 202 immediately with job_id
              │
              ▼
  Celery Worker picks up task
              │
        Parse document by type:
          PDF   → layout-aware text extraction (bounding boxes, column structure)
                  table extraction via pdfplumber
                  render each page as PNG image
          PPTX  → extract title, bullets, speaker notes per slide
                  render each slide as image via LibreOffice
          DOCX  → paragraph grouping by heading
          ipynb → separate code cells from markdown cells
              │
         Fan-out: one process_page task per page
         All pages run in parallel (concurrency=4)
              │
         Per page:
           1. Upload rendered image to S3
           2. Vision pass → Claude reads image, writes educational description
              (captures diagrams, charts, formulas with no extractable text)
           3. Chunker → combines raw text + vision caption into enriched_text
              with metadata prefix (course, lecture, section, page)
              Tables and diagrams = atomic chunks, never split
           4. Save chunks to Supabase
              │
              ▼
        After all pages done:
        Batch embed all chunks
        (OpenAI text-embedding-3-large, 1536 dims, 100 texts per API call)
              │
              ▼
        Write vectors to pgvector (ivfflat index)
              │
              ▼
        Document status → "ready"
        Job status → "done"
```

Every chunk stores: `source_file`, `page_number`, `slide_number`, `section_title`, `lecture_number`, `lecture_title`, `chunk_index`, `chunk_type`, `raw_text`, `vision_caption`, `enriched_text`.

Every page has its own `PageRecord` with status `pending | processing | done | failed`. If a worker crashes on page 847, pages 1–846 stay done and only failed pages are requeued via the retry endpoint.

---

## How Retrieval Works

```
Query arrives
      │
      ▼
Embed query (text-embedding-3-large, 1536 dims)
      │
      ├──► Vector search (pgvector ivfflat)
      │    top 30 by cosine similarity
      │                                      ──► Reciprocal Rank Fusion
      └──► Keyword search (PostgreSQL tsvector)   merge both lists
           top 30 by BM25-style rank               into one ranking
                        │
                        ▼
              Cross-encoder reranker
              (ms-marco-MiniLM-L-6-v2)
              Reads query + each chunk together
              Scores by "does this answer the question"
                        │
                   Top 5 chunks
                        │
             Build structured prompt:
             - Include chunk text
             - Include metadata (lecture, page, section)
             - Attach rendered image for diagram/chart chunks
                        │
             Pass to Claude → answer with citations
```

---

## How the Quiz Generator Works

1. Professor sends a request specifying: question count, duration, difficulty distribution, questions types (MCQ, short answer, derivation, diagram-based, table interpretation), which lectures draw from, and any special instructions.
2. The system retrieves the 15 most relevant chunks for the requested scope
3. Claude receives the retrieved content and requirements and generates structured JSON
4. Every question includes:
    - Question text and options (for MCQ)
    - Correct answer
    - Step-by-step answer explanation for the answer key
    - Mark scheme showing what earns each mark
    - Source References - which lecture, slide, and page
    - Difficulty rating and the reason it was assigned
    - Auto detected quality flags: 'unclear_wording',
    'multiple_valid_answers', 'missing assumption'
5. Professor reviews questions and accepts, rejects, or edits them inline
6. Alternative versions can be generated to prevent answer sharing but maintaining same structure and difficulty with different specific content
7. After students complete the quiz, results are recorded and analytics show which questions were most missed, performance by difficulty, and which topics students were struggling with

### Example Request

```json
POST /api/v1/courses/{id}/quizzes
{
  "title": "Finals Week Quiz",
  "question_count": 20,
  "duration_minutes": 45,
  "difficulty_distribution": { "easy": 25, "medium": 50, "hard": 25 },
  "question_types": ["mcq", "short_answer", "derivation"],
  "lecture_filter": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
  "special_request": "Include a question connecting lecture 3 and lecture 7",
  "allow_multiple_versions": true
}
```

Quizzes draw from all uploaded lectures in the course. 'lecture_filter' scopes generation to specific lectures. Leave it empty to draw from everything.

---

## AI Tutor - How It Would Be Built

The tutor shares the same ingestion pipeline, vector store, and retrieval system as the quiz generator. Nothing already built changes. The tutor is a layer on top. What makes the tutor different from the quiz generator is that the quiz generator is a single-shot operation, performing just chunk retrieval, question generation, and JSON return. The tutor is conversational, which introduces three new problems:
    - it needs memory
    - it needs to handle questions that span multiple lectures
    - it needs to be honest about what it does not know

**Session memory** - A Tutor session model stores conversation history. Every query includes the full history and this is passed to the LLM alongside the retrieved chunks. This means the tutor remembers what was already explained and understands follow-up questions like "can you go deeper on that?" or "how does that relate to what you just said?" Without this, every question would be answered in isolation and the experience would feel broken.

**Multi-lecture query decomposition** - A question can span multiple lectures. A single retrieval call with that question will not reliably surface the right content from the stated lectures because the query would be pulling in different directions. The tutor solves this by first sending the question to the LLM and asking it to break it into sub-queries per lecture. Retrieval runs separately for each sub-query, the results are merged and deduplicated, and then a cross-lecture answer is synthesized by LLM from the combined text.

**Concept graph** - This is built at ingestion time. An LLM pass extracts named concepts from every chunk and builds a graph of relationships, connecting ones that co-occur frequently. At query time, the tutor uses the graph to find related content even when the student's question doesn't use the exact terminology from the lecture. This gives the tutor global understanding of the course rather than treating each chunk in isolation.

**Confidence threshold** - This would be a simple check that only answers if the top reranked chunk exceeds a minimum relevance score. Otherwise responds: "This doesn't appear to be covered in the course materials." Without this, the LLM will try to answer from loosely related content and produce responses that sound plausible but are not grounded in what was actually taught. This prevents hallucination.

**Structured Answer Format** - Every answer follows a fixed output schema: Definition -> Explanation -> Example -> Application -> Connection to other lectures -> Sources. This is enforced through the system prompt. If a section can't be answered from the retrieved content, it says so rather than filling the gap with general knowledge. The sources section cites specific lecture numbers, slide numbers, and page numbers.

**Answer Verification** - After the answer is generated, a second LLM pass checks every claim in the answer against retrieved chunks. Fully supported claims are kept. Partially supported claims are rewritten conservatively. Unsupported claims are removed or flagged as not covered in the materials. This two-pass approach catches the cases where the LLM drifts beyond what the sources actually say.

**Adaptive Depth** - Before retrieval runs, the question is classified by complexity. A simple definitional question gets a short focused answer. A complex question gets a multi-part step-by-step explanation with formula references. This classification is prompt driven as the system prompt instructs the LLM to match response depth to question complexity, using classification as a signal.

**New Endpoints**
```
POST   /courses/{id}/tutor/sessions                 # start a session
POST   /courses/{id}/tutor/sessions/{sid}/query     # ask a question
GET    /courses/{id}/tutor/sessions/{sid}           # full conversation history
DELETE /courses/{id}/tutor/sessions/{sid}           # end session
```

---

## What Breaks

1) Supabase session pooler connection limit - large documents (100+ pages) hit the 15 connection limit during parallel ingestion. Pages fail and require repeated use of the retry endpoint until all pages are complete.

2) Upstash Redis free tier - 10000 commands/day limit could be hit with heavy ingestion.

3) Network dependency - The application requires internet access to Supabase, AWS, Anthropic, and OpenAI. The local version will not run on a corporate network that blocks outbound connections. Therefore, development must be done on a personal network or mobile hotspot, or the necessary permissions need to be granted.

4) LibreOffice dependency for PPTX - This is used for PPTX slide rendering. If LibreOffice is not installed. PowerPoint slides are processed as text only with no image rendering. This means that the vision pass falls back to text description but loses the visual content entirely.

---

## What I Would Do Differently


1) Work on a personal network or mobile hotspot for development locally - The school's network caused every major setup problem. These problems included Docker installation failure, Supabase connection blockage, and HuggingFace model download timing out. Working on a personal network or mobile hotspot from the beginning would have saved time and led to better results.

2) Test with large documents earlier - I tested with a 140-page document and that exposed the connection limit issue that a shorter document would never encounter. Testing with larger documents would have caught this issue earlier in the development.

3) Implement Authentication first - Building a multi-user system without authentication means the security model is bolted on later. JWT auth with role-based access should be the first priority.

4) Adding a confidence threshold to retrieval - The quiz generator and AI tutor would both need a minimum relevance score before the LLM is called. Without this, questions can be generated from loosely related content when the topic is not well covered in the uploaded materials.

5) Store token counts per chunk - The chunker produces chunks of roughly similar word counts but does not track exact token counts. When assembling the LLM prompt, hitting the context window limit silently truncates content. Tracking tokens per chunk would allow precise context assembly.

6) Switch to transaction pooler with proper prepared statement disabling from the start - Using the session pooler on Supabase causes a 15 connection limit that blocks parallel ingestion of large documents. Using the transaction pooler would have avoided hours of connection limit errors and repeated retries during ingestion.

---

## What I would Build Next

1) Authentication and Role System - Using JWT-based auth, only professors would be allowed to create courses, upload documents, and generate and review quizzes. Students would only be able to query the tutor and take quizzes.

2) AI tutor - The major components for the AI tutor are already built including the vector store, retrieval pipeline, vision captions, and cross-encoder reranker. The tutor would be a session management layer in addition to structured prompting on top of what exists.

3) User Interface - A clean user interface for uploading lecture files, watching ingestion progress, reviewing generated quiz questions, and seeing analytics after students complete a quiz would be implemented. Additionally, a timed quiz user interface that presents questions, accepts answers, and submits results to the analytics endpoint automatically would be implemented.

4) Notifications - When ingestion is successfully completed, the professor will be notified via email or webhook telling them how many pages were processed and how many chunks were indexed and the course is ready to query. If any pages fail during ingestion, notifications are sent immediately with which document and page failed so the professor can retry without checking manually. Additionally, notifications can be implemented on the quiz side, with professors receiving notifications that their quiz has been generated and is ready for review with a direct link to the review interface. The professor would also get a notification when the full cohort has completed a quiz with their individual scores, and summary for analytics. 

## Project Structure

```
course-api/
├── app/
│   ├── main.py                      # FastAPI app, routers, startup
│   ├── core/
│   │   └── config.py                # All settings (env-driven)
│   ├── db/
│   │   ├── models.py                # SQLAlchemy models
│   │   └── session.py               # Async DB engine, ivfflat index
│   ├── ingestion/
│   │   ├── parsers.py               # PDF, PPTX, DOCX, Notebook parsers
│   │   ├── vision.py                # Claude vision pass — captions every page
│   │   ├── chunker.py               # Semantic chunker — enriched ChunkData
│   │   ├── embedder.py              # Batched text-embedding-3-large (1536 dims)
│   │   └── storage.py               # AWS S3 upload/download
│   ├── retrieval/
│   │   └── retriever.py             # Vector + keyword search + RRF + reranker
│   ├── quiz/
│   │   └── generator.py             # LLM quiz generation, flags, versioning
│   ├── workers/
│   │   └── ingestion_tasks.py       # Celery tasks — parallel page processing
│   └── api/routes/
│       ├── courses.py
│       ├── documents.py             # Direct + chunked upload, retry
│       ├── jobs.py                  # Ingestion progress polling
│       └── quiz.py                  # Generate, review, version, analytics
├── Procfile                         # Railway process definitions
├── Dockerfile                       # Container build instructions
├── requirements.txt
```
