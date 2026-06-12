import asyncio
from logging.config import fileConfig
import ssl

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import pool
from alembic import context

# ── Import app config & models ──
from app.core.config import settings
from app.core.database import Base

# Import ALL model modules so their tables register on Base.metadata
import app.core.db_models  # noqa: F401  — PatientSession
import app.schemas.memory_models  # noqa: F401  — Patient, Session

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = settings.ASYNC_DATABASE_URL
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()

async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    connect_args = {}
    if "supabase" in settings.ASYNC_DATABASE_URL.lower():
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        connect_args["ssl"] = ssl_context
        if ":6543" in settings.ASYNC_DATABASE_URL:
            connect_args["statement_cache_size"] = 0

    connectable = create_async_engine(
        settings.ASYNC_DATABASE_URL,
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
