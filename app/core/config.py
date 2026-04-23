from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Course Intelligence API"
    DEBUG: bool = False
    API_PREFIX: str = "/api/v1"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/coursedb"
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10

    # Redis (optional - not needed without Celery)
    REDIS_URL: Optional[str] = None

    # Object Storage (S3)
    S3_BUCKET: str = "course-materials"
    S3_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None

    # Supabase (optional)
    SUPABASE_URL: Optional[str] = None
    SUPABASE_SERVICE_KEY: Optional[str] = None

    # AI APIs
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str

    # Embedding
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIMENSIONS: int = 1536
    EMBEDDING_BATCH_SIZE: int = 100

    # Vision
    VISION_MODEL: str = "claude-opus-4-6"
    VISION_CONCURRENCY: int = 20
    VISION_RATE_LIMIT: str = "50/m"

    # Chunking
    DEFAULT_CHUNK_SIZE: int = 512
    DEFAULT_CHUNK_OVERLAP: int = 64

    # Retrieval
    RETRIEVAL_TOP_K_CANDIDATES: int = 30
    RETRIEVAL_TOP_K_FINAL: int = 5
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Upload limits
    MAX_FILE_SIZE_MB: int = 500
    UPLOAD_CHUNK_SIZE_MB: int = 10

    # Quiz
    MAX_QUIZ_QUESTIONS: int = 100

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()