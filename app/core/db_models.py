from sqlalchemy import Column, String, DateTime, JSON, Text
from datetime import datetime, timezone
from app.core.database import Base

# PatientSession has been deprecated and removed. Memory relies exclusively on app.schemas.memory_models.
