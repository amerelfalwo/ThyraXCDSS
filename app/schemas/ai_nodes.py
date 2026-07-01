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
    action: str
    result: str
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuditLogResponse(BaseModel):
    """Response for the audit log endpoint."""
    entries: List[AuditLogEntry]
    total: int


MULTI_IMAGE_REQUEST_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {"type": "string", "format": "binary"},
                            "description": "Upload one or more ultrasound images",
                        },
                        "session_id": {
                            "type": "string",
                            "nullable": True,
                            "description": "Enter Session ID for automated Synthesis trigger",
                        },
                        "force": {
                            "type": "boolean",
                            "default": False,
                            "description": "Force prediction even if gatekeeper fails",
                        },
                    },
                    "required": ["files"],
                }
            }
        },
    }
}
