"""
Dual-State Memory Manager — Async PostgreSQL Backend.

Orchestrates the interplay between:
  - Long-Term Memory  (Patient table)  — persistent across sessions
  - Short-Term Memory (Session table)  — ephemeral per-visit context

This service is fully async (asyncpg) and designed to be called from
the FastAPI/Agent layer without blocking the event loop.

Architecture:
    ┌───────────────────────────────────────────────────┐
    │  MemoryManager                                    │
    │                                                   │
    │  load_context(session_id, patient_id?)             │
    │    → MemoryContext (long-term + short-term merged) │
    │                                                   │
    │  save_exchange(session_id, user_msg, ai_msg)       │
    │    → persists to Session.conversation_history      │
    │                                                   │
    │  save_diagnostic(session_id, node_type, data)      │
    │    → persists to Session.diagnostic_context        │
    │                                                   │
    │  summarize_and_prune(session_id)                   │
    │    → LLM-compress old messages → session_summary   │
    │    → update Patient.long_term_summary              │
    │                                                   │
    │  get_or_create_patient(patient_id, demographics?)  │
    │    → upsert into Patient table                     │
    └───────────────────────────────────────────────────┘

Usage:
    from app.services.memory_manager import memory_manager

    ctx = await memory_manager.load_context(session_id="abc", patient_id="P001")
    # ctx.chat_history, ctx.long_term_summary, ctx.diagnostic_context, ...
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app.core.config import settings

logger = logging.getLogger(__name__)


# ─── Async Engine (module-level singleton) ────────────────────

from app.core.database import AsyncSessionLocal

def _get_async_session_factory():
    return AsyncSessionLocal


# ─── Data Transfer Object ────────────────────────────────────


@dataclass
class MemoryContext:
    """
    Merged memory context injected into the agent prompt.

    Combines long-term patient data with short-term session state
    into a single, prompt-ready object.
    """

    # Short-term (Session)
    session_id: str
    chat_history: list[dict] = field(default_factory=list)
    diagnostic_context: dict = field(default_factory=dict)
    session_summary: str = ""

    # Long-term (Patient)
    patient_id: Optional[str] = None
    demographics: dict = field(default_factory=dict)
    medical_history: list = field(default_factory=list)
    allergies: list = field(default_factory=list)
    long_term_summary: str = ""

    def to_prompt_context(self) -> str:
        """
        Render the full memory context as a human-readable string
        suitable for injection into the LLM system prompt.
        """
        parts = []

        # Patient identity
        if self.patient_id:
            demo = self.demographics or {}
            name = demo.get("name", "Unknown")
            age = demo.get("age", "N/A")
            sex = demo.get("sex", "N/A")
            parts.append(
                f"**Patient:** {name} (ID: {self.patient_id}) | "
                f"Age: {age} | Sex: {sex}"
            )

        # Allergies
        if self.allergies:
            allergy_str = ", ".join(str(a) for a in self.allergies)
            parts.append(f"**Known Allergies:** {allergy_str}")

        # Medical history
        if self.medical_history:
            history_items = []
            for item in self.medical_history[:10]:  # Cap to avoid token bloat
                if isinstance(item, dict):
                    history_items.append(
                        f"  - {item.get('condition', item.get('description', str(item)))}"
                    )
                else:
                    history_items.append(f"  - {item}")
            parts.append(
                "**Medical History:**\n" + "\n".join(history_items)
            )

        # Long-term conversation summary
        if self.long_term_summary:
            parts.append(
                f"**Long-Term Summary (Prior Visits):**\n{self.long_term_summary}"
            )

        # Session summary (current visit compressed)
        if self.session_summary:
            parts.append(
                f"**Current Visit Summary:**\n{self.session_summary}"
            )

        # Diagnostic results from this session
        if self.diagnostic_context:
            diag_parts = []
            for node_type, data in self.diagnostic_context.items():
                if isinstance(data, dict):
                    details = ", ".join(
                        f"{k}: {v}" for k, v in data.items()
                        if k != "timestamp"
                    )
                    diag_parts.append(f"  - **{node_type}:** {details}")
                else:
                    diag_parts.append(f"  - **{node_type}:** {data}")
            if diag_parts:
                parts.append(
                    "**Diagnostic Results (This Visit):**\n"
                    + "\n".join(diag_parts)
                )

        if not parts:
            return "No patient context available for this session."

        return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# Memory Manager
# ═══════════════════════════════════════════════════════════════


class MemoryManager:
    """
    Async Dual-State Memory Manager.

    Handles all CRUD operations for the Patient (long-term) and
    Session (short-term) tables, and provides a unified
    MemoryContext for prompt injection.
    """



    # ── Context Loading ───────────────────────────────────────

    async def load_context(
        self,
        session_id: str,
        patient_id: Optional[str] = None,
        doctor_id: Optional[str] = None,
    ) -> MemoryContext:
        """
        Load the merged memory context for a given session.

        If patient_id is provided, also loads long-term patient data.
        If the session doesn't exist, creates one.

        Args:
            session_id: The current session identifier.
            patient_id: Optional patient identifier for long-term memory.
            doctor_id: Optional doctor identifier to link the session if it needs creation.

        Returns:
            MemoryContext with both short and long-term data merged.
        """
        from app.schemas.memory_models import Patient, Session

        factory = _get_async_session_factory()

        async with factory() as db:
            # ── Load or create Session (short-term) ──
            result = await db.execute(
                select(Session).where(Session.session_id == session_id)
            )
            session = result.scalar_one_or_none()

            if session is None:
                session = Session(
                    session_id=session_id,
                    doctor_id=doctor_id or "test_doc_123",
                    conversation_history=[],
                    diagnostic_context={},
                    session_summary="",
                    is_active="true",
                )
                db.add(session)
                await db.commit()
                await db.refresh(session)
                logger.info(f"Created new session: {session_id}")

            ctx = MemoryContext(
                session_id=session_id,
                chat_history=session.conversation_history or [],
                diagnostic_context=session.diagnostic_context or {},
                session_summary=session.session_summary or "",
            )

            # ── Load Patient (long-term) if linked ──
            effective_patient_id = patient_id
            if effective_patient_id:
                result = await db.execute(
                    select(Patient).where(
                        Patient.patient_id == effective_patient_id
                    )
                )
                patient = result.scalar_one_or_none()

                if patient:
                    ctx.patient_id = patient.patient_id
                    ctx.demographics = patient.demographics or {}
                    ctx.medical_history = patient.medical_history or []
                    ctx.allergies = patient.allergies or []
                    ctx.long_term_summary = patient.long_term_summary or ""

            return ctx

    # ── Save Chat Exchange ────────────────────────────────────

    async def save_exchange(
        self,
        session_id: str,
        user_message: str,
        ai_response: str,
        doctor_id: Optional[str] = None,
    ) -> None:
        """
        Append a user→AI exchange to the session's conversation history.

        Args:
            session_id: The session to update.
            user_message: What the user said.
            ai_response: What the agent responded.
            doctor_id: The doctor context in case the session needs to be created.
        """
        from app.schemas.memory_models import Session

        factory = _get_async_session_factory()
        now = datetime.now(timezone.utc).isoformat()

        async with factory() as db:
            result = await db.execute(
                select(Session).where(Session.session_id == session_id)
            )
            session = result.scalar_one_or_none()

            if session is None:
                logger.warning(
                    f"save_exchange: session {session_id} not found, creating"
                )
                session = Session(
                    session_id=session_id,
                    doctor_id=doctor_id or "test_doc_123",
                    conversation_history=[],
                    diagnostic_context={},
                )
                db.add(session)

            history = list(session.conversation_history or [])
            history.append({"role": "user", "content": user_message, "ts": now})
            history.append({"role": "assistant", "content": ai_response, "ts": now})

            # Force SQLAlchemy JSONB mutation detection
            session.conversation_history = history
            await db.commit()

            logger.debug(
                f"Saved exchange to session {session_id} "
                f"(total messages: {len(history)})"
            )

    # ── JSON Sanitizer for JSONB Compatibility ───────────────

    @staticmethod
    def _sanitize_for_json(obj: Any) -> Any:
        """
        Recursively convert numpy / non-native types to JSON-safe
        Python primitives before insertion into a JSONB column.

        Handles:
          - numpy.bool_       → bool
          - numpy.integer     → int
          - numpy.floating    → float
          - numpy.ndarray     → list (recursive)
          - bytes / bytearray → <stripped>  (not JSON serializable)
          - dict / list        → recurse into children
        """
        import numpy as np

        if isinstance(obj, dict):
            return {k: MemoryManager._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [MemoryManager._sanitize_for_json(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, (bytes, bytearray)):
            # Raw binary data cannot go into JSONB — skip it
            return None
        return obj

    # ── Save Diagnostic Results ───────────────────────────────

    async def save_diagnostic(
        self,
        session_id: str,
        node_type: str,
        data: Any,
        doctor_id: Optional[str] = None,
    ) -> None:
        """
        Store a diagnostic result (clinical, ultrasound, FNAC) in
        the session's diagnostic_context JSONB.

        All incoming data is sanitized to convert numpy types
        (bool_, int64, float32, ndarray) to native Python primitives
        before insertion, preventing asyncpg serialization errors.

        Args:
            session_id: The session to update.
            node_type: One of 'clinical', 'ultrasound', 'fnac'.
            data: The diagnostic payload from the respective node.
        """
        from app.schemas.memory_models import Session
        from sqlalchemy.orm.attributes import flag_modified

        factory = _get_async_session_factory()

        # ── Sanitize all numpy types to native Python before JSONB insert ──
        data = self._sanitize_for_json(data)

        async with factory() as db:
            # Use with_for_update() to acquire a row-level lock and prevent
            # race conditions when multiple diagnostics are saved concurrently
            result = await db.execute(
                select(Session).where(Session.session_id == session_id).with_for_update()
            )
            session = result.scalar_one_or_none()
            
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Ensure the dictionary can hold both keys seamlessly at the same time
            # and support list arrays or single dicts as instructed
            if isinstance(data, list):
                payload = data
            elif isinstance(data, dict):
                payload = {**data, "timestamp": timestamp}
            else:
                payload = {"value": data, "timestamp": timestamp}

            if session is None:
                session = Session(
                    session_id=session_id,
                    doctor_id=doctor_id or "test_doc_123",
                    diagnostic_context={node_type: payload},
                    conversation_history=[],
                )
                db.add(session)
            else:
                # Deep merge the incoming dictionary for the specific node_type
                diag = dict(session.diagnostic_context or {})
                diag[node_type] = payload
                session.diagnostic_context = diag
                
                # SQLAlchemy requires flag_modified to detect JSONB mutations
                flag_modified(session, "diagnostic_context")

            await db.commit()

            logger.info(
                f"Saved {node_type} diagnostic to session {session_id}"
            )

    # ── Summarize & Prune ─────────────────────────────────────

    MAX_HISTORY_MESSAGES = 10



    async def summarize_and_prune(
        self,
        session_id: str,
        keep_recent: int = 4,
    ) -> None:
        """
        Compress older messages into a summary and prune the history.

        Steps:
        1. If conversation_history ≤ 6 messages, skip.
        2. Take all messages except the last `keep_recent`.
        3. Ask the LLM to produce a condensed clinical summary.
        4. Update Session.session_summary with the new summary.
        5. Truncate conversation_history to the last `keep_recent`.
        6. If the session is linked to a Patient, also append
           the session summary to Patient.long_term_summary.

        This prevents the context window from overflowing while
        preserving important clinical information.
        """
        from app.schemas.memory_models import Patient, Session

        factory = _get_async_session_factory()

        async with factory() as db:
            result = await db.execute(
                select(Session).where(Session.session_id == session_id)
            )
            session = result.scalar_one_or_none()

            if session is None:
                return

            history = session.conversation_history or []
            if len(history) <= 6:
                return  # Not enough messages to warrant summarization

            messages_to_summarize = history[:-keep_recent]
            existing_summary = session.session_summary or ""

            # Build summarization prompt
            text_block = "\n".join(
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                for msg in messages_to_summarize
            )

            prompt = (
                "You are a medical AI assistant. Produce a concise clinical "
                "summary that preserves key diagnostic findings, patient "
                "concerns, and treatment decisions.\n\n"
                "EXISTING SUMMARY:\n"
                f"{existing_summary if existing_summary else '(none)'}\n\n"
                "NEW MESSAGES TO INCORPORATE:\n"
                f"{text_block}\n\n"
                "Write an updated clinical summary (max 300 words). "
                "Focus on diagnoses, test results, medications, and "
                "clinical decisions."
            )

            try:
                from app.core.llm_client import get_shared_llm
                llm = get_shared_llm(temperature=0.3)
                res = await llm.ainvoke(prompt)
                new_summary = res.content

                # Update session: save summary + prune history
                session.session_summary = new_summary
                session.conversation_history = list(history[-keep_recent:])
                await db.commit()

                logger.info(
                    f"Summarized and pruned session {session_id} "
                    f"({len(messages_to_summarize)} msgs → summary)"
                )

                # Propagation removed because Session no longer tracks patient_id
                pass

            except Exception as e:
                logger.error(f"Summarization failed for {session_id}: {e}")

    # ── Patient CRUD ──────────────────────────────────────────

    async def get_or_create_patient(
        self,
        patient_id: str,
        demographics: Optional[dict] = None,
        medical_history: Optional[list] = None,
        allergies: Optional[list] = None,
        doctor_id: Optional[str] = None,
    ) -> dict:
        """
        Upsert a patient record. Creates if not found, updates
        demographics/history/allergies if provided.

        Returns the patient record as a dict.
        """
        from app.schemas.memory_models import Patient

        factory = _get_async_session_factory()

        async with factory() as db:
            result = await db.execute(
                select(Patient).where(Patient.patient_id == patient_id)
            )
            patient = result.scalar_one_or_none()

            if patient is None:
                doc_id = doctor_id or "test_doc_123"
                patient = Patient(
                    patient_id=patient_id,
                    doctor_id=doc_id,
                    demographics=demographics or {},
                    medical_history=medical_history or [],
                    allergies=allergies or [],
                    long_term_summary="",
                )
                db.add(patient)
                await db.commit()
                await db.refresh(patient)
                logger.info(f"Created new patient: {patient_id}")
            else:
                # Update fields if new data provided
                changed = False
                if demographics:
                    patient.demographics = demographics
                    changed = True
                if medical_history is not None:
                    patient.medical_history = medical_history
                    changed = True
                if allergies is not None:
                    patient.allergies = allergies
                    changed = True
                if changed:
                    await db.commit()
                    logger.info(f"Updated patient: {patient_id}")

            return {
                "patient_id": patient.patient_id,
                "demographics": patient.demographics,
                "medical_history": patient.medical_history,
                "allergies": patient.allergies,
                "long_term_summary": patient.long_term_summary or "",
            }

    # ── Session Lifecycle ─────────────────────────────────────

    async def close_session(self, session_id: str) -> bool:
        """Mark a session as inactive (end of visit)."""
        from app.schemas.memory_models import Session

        factory = _get_async_session_factory()

        async with factory() as db:
            result = await db.execute(
                select(Session).where(Session.session_id == session_id)
            )
            session = result.scalar_one_or_none()

            if session is None:
                return False

            session.is_active = "false"
            await db.commit()
            logger.info(f"Session {session_id} closed")
            return True

    # ── Image Storage (Database) ──────────────────────────────
    
    async def save_image(
        self,
        session_id: str,
        image_data: bytes,
        image_type: str = "synthesis_composite",
        doctor_id: Optional[str] = None,
    ) -> int:
        """
        Saves an image directly to the database.
        Returns the ID of the saved image.
        """
        from app.schemas.memory_models import Session, DiagnosticImage
        
        factory = _get_async_session_factory()
        
        async with factory() as db:
            result = await db.execute(
                select(Session).where(Session.session_id == session_id)
            )
            session = result.scalar_one_or_none()
            if session is None:
                session = Session(
                    session_id=session_id,
                    doctor_id=doctor_id or "test_doc_123",
                    conversation_history=[],
                    diagnostic_context={},
                )
                db.add(session)
                await db.commit()
                
            diag_image = DiagnosticImage(
                session_id=session_id,
                image_data=image_data,
                image_type=image_type
            )
            db.add(diag_image)
            await db.commit()
            await db.refresh(diag_image)
            
            logger.info(f"Saved {image_type} image {diag_image.id} for session {session_id}")
            return diag_image.id

    async def get_image(self, image_id: int) -> Optional[bytes]:
        """
        Retrieves image bytes by image_id.
        """
        from app.schemas.memory_models import DiagnosticImage
        
        factory = _get_async_session_factory()
        
        async with factory() as db:
            result = await db.execute(
                select(DiagnosticImage).where(DiagnosticImage.id == image_id)
            )
            diag_image = result.scalar_one_or_none()
            if diag_image:
                return diag_image.image_data
            return None



# ── Module-level singleton ──
memory_manager = MemoryManager()
