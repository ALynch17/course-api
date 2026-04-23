"""
Course Intelligence API — main application entry point.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from app.core.config import settings
from app.db.session import init_db, create_vector_index
from app.api.routes import courses, documents, jobs, quiz

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup/shutdown tasks."""
    log.info("startup_begin")
    await init_db()
    await create_vector_index()
    log.info("startup_complete")
    yield
    log.info("shutdown")


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="""
## Course Intelligence API

Ingests lecture materials (PDF, PPTX, DOCX, Jupyter notebooks) and powers:
- **Quiz generation** grounded in actual course content
- **Semantic retrieval** across all lectures for an AI tutor

### Key design decisions
- Every page gets a **dual representation**: raw text + vision model caption
- Ingestion is **fully async** with per-page resumability
- Retrieval uses **hybrid search** (vector + keyword) + **cross-encoder reranking**
""",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — adjust origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global error handler — consistent error envelope across all routes
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(exc),
                "path": str(request.url.path),
            }
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "VALIDATION_ERROR", "message": str(exc)}},
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

PREFIX = settings.API_PREFIX

app.include_router(courses.router,   prefix=PREFIX)
app.include_router(documents.router, prefix=PREFIX)
app.include_router(jobs.router,      prefix=PREFIX)
app.include_router(quiz.router,      prefix=PREFIX)


@app.get("/health")
async def health():
    return {"status": "ok", "service": settings.APP_NAME}


@app.get("/")
async def root():
    return {
        "service": settings.APP_NAME,
        "docs": "/docs",
        "health": "/health",
    }
