"""
Schemas for shared system models (Audit Log).
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any


# ═══════════════════════════════════════════════════════════════
# Audit Log
# ═══════════════════════════════════════════════════════════════

class AuditLogEntry(BaseModel):
    """A single audit log event."""
    timestamp: str
    node: str
    patient_id: Optional[int] = None
    action: str
    result: str
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuditLogResponse(BaseModel):
    """Response for the audit log endpoint."""
    entries: List[AuditLogEntry]
    total: int
