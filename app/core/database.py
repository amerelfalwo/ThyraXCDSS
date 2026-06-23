import ssl
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.core.config import settings

connect_args = {
    "statement_cache_size": 0,
    "max_cached_statement_lifetime": 0,
}

# Dynamically append ?prepared_statement_cache_size=0 to the URL
db_url = settings.ASYNC_DATABASE_URL
if "?" in db_url:
    db_url += "&prepared_statement_cache_size=0"
else:
    db_url += "?prepared_statement_cache_size=0"

# Supabase requires SSL for external connections
if "supabase" in settings.ASYNC_DATABASE_URL.lower():
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    connect_args["ssl"] = ssl_context

engine = create_async_engine(
    db_url,
    poolclass=NullPool,
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
