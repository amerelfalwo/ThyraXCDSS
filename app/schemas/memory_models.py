"""
Dual-State Memory Models for the ThyraX CDSS.

Long-Term Memory (Patient):
    Persistent patient record that survives across sessions.
    Stores demographics, medical history, allergies, and a rolling
    summary of all past interactions.

Short-Term Memory (Session):
    Ephemeral per-visit context linked to a Patient.
    Stores the active conversation history and diagnostic results
    accumulated during the current clinical encounter.

Relationship:
    Patient  1 ──── ∗  Session
    A patient can have many sessions over time. Each session belongs
    to exactly one patient (or none, for anonymous sessions).
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy import LargeBinary

from app.core.database import Base


def get_doctor_filtered_query(model, doctor_id: str):
    """
    Utility function to enforce strict data isolation for doctors.
    Generates a SQLAlchemy select query that ALWAYS filters by the authenticated doctor_id.
    """
    from sqlalchemy import select
    if not hasattr(model, 'doctor_id'):
        raise ValueError(f"Model {model.__name__} does not support doctor-level isolation.")
    return select(model).where(model.doctor_id == doctor_id)


class Doctor(Base):
    """
    Doctor / Multi-tenant Entity.

    Fields:
        doctor_id: Unique doctor identifier.
        name: Name of the doctor.
        specialty: Medical specialty.
    """

    __tablename__ = "doctors"

    doctor_id = Column(String(128), primary_key=True, index=True)
    name = Column(String(255), nullable=True)
    specialty = Column(String(128), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    patients = relationship("Patient", back_populates="doctor", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Doctor(doctor_id={self.doctor_id!r})>"


class Patient(Base):
    """
    Long-Term Memory — Persistent Patient Record.

    Fields:
        patient_id:       Unique patient identifier (clinic MRN, UUID, etc.).
        demographics:     JSONB — name, age, sex, contact, etc.
        medical_history:  JSONB — structured list of conditions, surgeries,
                          medications, lab baselines, and prior diagnoses.
        allergies:        JSONB — known drug and environmental allergies.
        long_term_summary: Text — rolling natural-language summary of all
                           past interactions, periodically condensed by LLM.
        created_at:       Timestamp of first registration.
        updated_at:       Timestamp of last modification.
    """

    __tablename__ = "patients"

    patient_id = Column(String(128), primary_key=True, index=True)
    doctor_id = Column(String(128), ForeignKey("doctors.doctor_id", ondelete="CASCADE"), nullable=False, index=True)
    demographics = Column(JSONB, nullable=True, default=dict)
    medical_history = Column(JSONB, nullable=True, default=list)
    allergies = Column(JSONB, nullable=True, default=list)
    long_term_summary = Column(Text, nullable=True, default="")
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    doctor = relationship("Doctor", back_populates="patients")

    def __repr__(self) -> str:
        return f"<Patient(patient_id={self.patient_id!r})>"


class Session(Base):
    """
    Short-Term Memory — Active Clinical Encounter.

    Fields:
        session_id:            Unique session identifier (UUID).
        conversation_history:  JSONB — list of {role, content, timestamp} dicts,
                               representing the active dialogue window.
        diagnostic_context:    JSONB — accumulated results from clinical,
                               ultrasound, and FNAC nodes during this visit.
        session_summary:       Text — condensed summary of this session so far,
                               updated when the conversation window is pruned.
        is_active:             Whether the session is still in progress.
        created_at:            Timestamp of session creation.
        updated_at:            Timestamp of last activity.
    """

    __tablename__ = "sessions"

    session_id = Column(String(128), primary_key=True, index=True)
    doctor_id = Column(String(128), ForeignKey("doctors.doctor_id", ondelete="CASCADE"), nullable=False, index=True, default="test_doc_123")
    conversation_history = Column(JSONB, nullable=True, default=list)
    diagnostic_context = Column(JSONB, nullable=True, default=dict)
    session_summary = Column(Text, nullable=True, default="")
    is_active = Column(String(8), nullable=False, default="true")
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Session(session_id={self.session_id!r})>"

from sqlalchemy import Integer

class AuditLog(Base):
    """
    Audit Log for Weekly Hallucination Evaluation.
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(128), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    score = Column(Integer, nullable=False)
    reason = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    session = relationship("Session")

    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, session_id={self.session_id!r}, score={self.score})>"

class DiagnosticImage(Base):
    """
    Stores raw medical images or composited output directly in the DB.
    """
    __tablename__ = "diagnostic_images"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(128), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    image_data = Column(LargeBinary, nullable=False)
    image_type = Column(String(64), nullable=False) # e.g., 'synthesis_composite'
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    session = relationship("Session")

    def __repr__(self) -> str:
        return f"<DiagnosticImage(id={self.id}, session_id={self.session_id!r}, type={self.image_type!r})>"
