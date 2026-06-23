import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

DATABASE_URL="postgresql+asyncpg://postgres.dlvfucwwmhsdmzsdmdgh:Thyrax123*%23@aws-0-eu-west-1.pooler.supabase.com:6543/postgres"

async def migrate():
    print("Migrating DB...")
    engine = create_async_engine(DATABASE_URL)
    async with engine.begin() as conn:
        try:
            await conn.execute(text("ALTER TABLE patient_sessions ADD COLUMN doctor_id VARCHAR;"))
            await conn.execute(text("CREATE INDEX ix_patient_sessions_doctor_id ON patient_sessions (doctor_id);"))
            print("Successfully added doctor_id column.")
        except Exception as e:
            print(f"Error (maybe already exists?): {e}")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(migrate())
