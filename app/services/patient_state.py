"""
Patient State Manager — Lightweight In-Memory Context Store.

Provides a thread-safe, TTL-backed dictionary for tracking a patient's
evolving diagnostic journey across ThyraX nodes.

Architecture:
  - Uses cachetools.TTLCache for automatic expiration (default 2 hours).
  - Thread-safe via threading.Lock (GIL alone is NOT sufficient for
    compound operations like read-modify-write).
  - Zero external dependencies beyond cachetools (already installed).
  - Memory footprint: ~1KB per session (well within 512MB constraint).

Usage:
  from app.services.patient_state import state_manager

  state_manager.update_clinical("session-123", {...})
  context = state_manager.get_state("session-123")
"""

import threading
import logging
from datetime import datetime, timezone
from typing import Optional

from cachetools import TTLCache

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Patient State Data Structure
# ═══════════════════════════════════════════════════════════════

_DEFAULT_STATE = {
    "clinical_assessment": None,
    "ultrasound_result": None,
    "fnac_result": None,
    "created_at": None,
    "last_updated": None,
}


class PatientStateManager:
    """
    Thread-safe in-memory patient state manager with TTL expiration.

    Each session_id maps to a dict holding the patient's cumulative
    diagnostic context from all ThyraX nodes they've passed through.
    Sessions expire after `ttl_seconds` of inactivity (default 2h).
    """

    def __init__(self, max_sessions: int = 500, ttl_seconds: int = 7200):
        """
        Args:
            max_sessions: Max concurrent sessions before LRU eviction.
            ttl_seconds: Time-to-live per session entry (seconds).
        """
        self._store = TTLCache(maxsize=max_sessions, ttl=ttl_seconds)
        self._lock = threading.Lock()
        logger.info(
            f"PatientStateManager initialized "
            f"(max_sessions={max_sessions}, ttl={ttl_seconds}s)"
        )

    def _ensure_session(self, session_id: str) -> dict:
        """Get or create a session state dict (must be called inside lock)."""
        if session_id not in self._store:
            self._store[session_id] = {
                **_DEFAULT_STATE,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        return self._store[session_id]

    # ─── Node Update Methods ──────────────────────────────────

    def update_clinical(self, session_id: str, data: dict) -> None:
        """Push clinical assessment results (Node 1+2) into state."""
        with self._lock:
            state = self._ensure_session(session_id)
            state["clinical_assessment"] = {
                "functional_status": data.get("functional_status"),
                "risk_level": data.get("risk_level"),
                "probabilities": data.get("probabilities"),
                "model_confidence": data.get("model_confidence"),
                "clinical_recommendation": data.get("clinical_recommendation"),
                "next_step": data.get("next_step"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            state["last_updated"] = datetime.now(timezone.utc).isoformat()
            logger.info(f"[State] Clinical updated for session={session_id}")

    def update_ultrasound(self, session_id: str, data: dict) -> None:
        """Push ultrasound prediction results (Node 4) into state."""
        with self._lock:
            state = self._ensure_session(session_id)
            cls = data.get("classification", {})
            if isinstance(cls, dict):
                state["ultrasound_result"] = {
                    "label": cls.get("label"),
                    "confidence_pct": cls.get("confidence_pct"),
                    "risk_level": cls.get("risk_level"),
                    "acr_tirads_level": cls.get("acr_tirads_level"),
                    "clinical_recommendation": cls.get("clinical_recommendation"),
                    "bbox": data.get("bbox"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            else:
                state["ultrasound_result"] = {
                    "label": "no_nodule_detected",
                    "message": str(cls),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            state["last_updated"] = datetime.now(timezone.utc).isoformat()
            logger.info(f"[State] Ultrasound updated for session={session_id}")

    def update_fnac(self, session_id: str, data: dict) -> None:
        """Push FNAC cytopathology results into state."""
        with self._lock:
            state = self._ensure_session(session_id)
            state["fnac_result"] = {
                "bethesda_category": data.get("bethesda_category"),
                "bethesda_label": data.get("bethesda_label"),
                "malignancy_risk": data.get("malignancy_risk"),
                "recommendation": data.get("recommendation"),
                "confidence_pct": data.get("confidence_pct"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            state["last_updated"] = datetime.now(timezone.utc).isoformat()
            logger.info(f"[State] FNAC updated for session={session_id}")

    # ─── Read Methods ─────────────────────────────────────────

    def get_state(self, session_id: str) -> Optional[dict]:
        """Get the full state for a session. Returns None if expired/absent."""
        with self._lock:
            return self._store.get(session_id)

    def get_state_summary(self, session_id: str) -> str:
        """
        Generate a human-readable summary of the patient's current state.
        Used for injecting into the LLM system prompt.
        """
        with self._lock:
            state = self._store.get(session_id)

        if not state:
            return "No patient context available for this session."

        parts = []

        # Clinical
        clinical = state.get("clinical_assessment")
        if clinical:
            parts.append(
                f"**Clinical Assessment (Node 1+2):**\n"
                f"  - Functional Status: {clinical.get('functional_status', 'N/A')}\n"
                f"  - Risk Level: {clinical.get('risk_level', 'N/A')}\n"
                f"  - Model Confidence: {clinical.get('model_confidence', 'N/A')}\n"
                f"  - Recommendation: {clinical.get('clinical_recommendation', 'N/A')}\n"
                f"  - Next Step: {clinical.get('next_step', 'N/A')}"
            )

        # Ultrasound
        us = state.get("ultrasound_result")
        if us:
            parts.append(
                f"**Ultrasound Analysis (Node 4):**\n"
                f"  - Classification: {us.get('label', 'N/A')}\n"
                f"  - Confidence: {us.get('confidence_pct', 'N/A')}%\n"
                f"  - Risk Level: {us.get('risk_level', 'N/A')}\n"
                f"  - ACR TI-RADS: {us.get('acr_tirads_level', 'N/A')}\n"
                f"  - Recommendation: {us.get('clinical_recommendation', 'N/A')}"
            )

        # FNAC
        fnac = state.get("fnac_result")
        if fnac:
            parts.append(
                f"**FNAC Cytopathology:**\n"
                f"  - Bethesda Category: {fnac.get('bethesda_category', 'N/A')}\n"
                f"  - Label: {fnac.get('bethesda_label', 'N/A')}\n"
                f"  - Malignancy Risk: {fnac.get('malignancy_risk', 'N/A')}\n"
                f"  - Recommendation: {fnac.get('recommendation', 'N/A')}"
            )

        if not parts:
            return "Patient session exists but no diagnostic data has been recorded yet."

        return "\n\n".join(parts)

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        with self._lock:
            return list(self._store.keys())

    def clear_session(self, session_id: str) -> bool:
        """Clear a session's state."""
        with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                return True
            return False


# ── Module-level singleton ──
state_manager = PatientStateManager()
