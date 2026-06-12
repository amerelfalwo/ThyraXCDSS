from sqlalchemy import Column, String, DateTime, JSON, Text
from datetime import datetime, timezone
from app.core.database import Base

class PatientSession(Base):
    __tablename__ = "patient_sessions"

    session_id = Column(String, primary_key=True, index=True)
    doctor_id = Column(String, index=True, nullable=True)
    clinical_assessment = Column(JSON, nullable=True)
    ultrasound_result = Column(JSON, nullable=True)
    fnac_result = Column(JSON, nullable=True)
    chat_history = Column(JSON, default=list)
    conversation_summary = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_updated = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
