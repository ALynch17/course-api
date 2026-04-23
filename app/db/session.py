from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text
from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    echo=settings.DEBUG,
    pool_recycle=300,
    pool_pre_ping=True,
    execution_options={"compiled_cache": None},
    connect_args={
        "statement_cache_size": 0,
        "prepared_statement_cache_size": 0,
    },
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        from app.db.models import Base
        await conn.run_sync(Base.metadata.create_all)


async def create_vector_index():
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding
            ON chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_course_id
            ON chunks (course_id)
        """))