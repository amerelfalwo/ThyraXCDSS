import ssl
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.core.config import settings

connect_args = {}
# Supabase requires SSL for external connections
if "supabase" in settings.ASYNC_DATABASE_URL.lower():
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    connect_args["ssl"] = ssl_context
    # For Supabase transaction connection pooler (port 6543), prepared statements can be problematic 
    # depending on the pooler mode (transaction vs session), but usually asyncpg needs statement cache disabled
    # if using transaction pooler. We'll disable statement cache if using the pooler port.
    if ":6543" in settings.ASYNC_DATABASE_URL:
        connect_args["statement_cache_size"] = 0

engine = create_async_engine(
    settings.ASYNC_DATABASE_URL,
    pool_pre_ping=True,
    connect_args=connect_args,
    echo=False
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine, 
    class_=AsyncSession, 
    autocommit=False, 
    autoflush=False,
    expire_on_commit=False
)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as db:
        try:
            yield db
        finally:
            await db.close()
