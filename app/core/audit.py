"""
Audit Logging for ThyraX CDSS.

Stores every AI inference result, recommendation, and confidence score
as a JSON-lines file for traceability. This lightweight approach avoids
adding SQLAlchemy / PostgreSQL to the Free Tier deployment while still
providing a queryable audit trail.

Each entry includes:
  - timestamp (ISO 8601)
  - node (which endpoint triggered the log)
  - patient_id (if applicable)
  - action (what the AI did)
  - result (the AI's output summary)
  - confidence (model confidence, if applicable)
  - metadata (additional context)
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Audit file location ───────────────────────────────────────
_AUDIT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "audit"
_AUDIT_FILE = _AUDIT_DIR / "audit_log.jsonl"


def _ensure_audit_dir() -> None:
    """Create the audit directory if it doesn't exist."""
    _AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def log_audit_event(
    node: str,
    action: str,
    result: str,
    patient_id: Optional[int] = None,
    confidence: Optional[float] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """
    Write an audit event to the JSONL log file.

    Args:
        node: The endpoint/node that triggered this event
              (e.g. 'clinical_assess', 'image_predict', 'agent_chat').
        action: What the AI did (e.g. 'xgboost_prediction', 'onnx_classification').
        result: Summary of the AI's output.
        patient_id: Optional patient identifier.
        confidence: Optional model confidence score (0.0–1.0).
        metadata: Optional dict with extra context (probabilities, tools_used, etc.).
    """
    _ensure_audit_dir()

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": node,
        "patient_id": patient_id,
        "action": action,
        "result": result,
        "confidence": confidence,
        "metadata": metadata or {},
    }

    try:
        with open(_AUDIT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        # Audit logging must NEVER crash the main flow
        logger.error(f"Failed to write audit log: {e}")


def read_recent_audits(limit: int = 50) -> list[dict]:
    """
    Read the most recent audit entries.

    Args:
        limit: Maximum number of entries to return.

    Returns:
        List of audit event dicts, most recent first.
    """
    if not _AUDIT_FILE.exists():
        return []

    try:
        with open(_AUDIT_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()

        entries = []
        for line in reversed(lines[-limit:]):
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        return entries
    except Exception as e:
        logger.error(f"Failed to read audit log: {e}")
        return []
