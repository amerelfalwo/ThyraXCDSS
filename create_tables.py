import asyncio
from app.core.database import engine, Base
from app.schemas.memory_models import *

async def create_tables():
    print("Creating all missing tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Successfully created missing tables.")

if __name__ == "__main__":
    asyncio.run(create_tables())
