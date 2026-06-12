"""
Chat schemas for the ThyraX AI Agent (Node 5).

Supports three input modes:
  - Text only:   query provided, no image
  - Image only:  image_base64 provided, no query
  - Multimodal:  both query and image_base64 provided
"""

from pydantic import BaseModel, Field, model_validator
from typing import Optional, List


class ChatMessage(BaseModel):
    """A single message in the conversation history."""
    role: str = Field(
        ..., description="Message role: 'user' or 'assistant'"
    )
    content: str = Field(
        ..., description="Message content text"
    )


class ChatRequest(BaseModel):
    """
    Flexible multimodal chat request.

    At least one of `query` or `image_base64` must be provided.
    Both can be sent together for multimodal analysis.
    """
    query: Optional[str] = Field(
        default=None,
        description=(
            "The user's text message / medical question. "
            "Optional if an image is provided."
        ),
    )
    chat_history: Optional[List[ChatMessage]] = Field(
        default_factory=list,
        description=(
            "Previous conversation messages for context continuity. "
            "Each item has 'role' ('user' or 'assistant') and 'content'."
        ),
    )
    image_base64: Optional[str] = Field(
        default=None,
        description=(
            "Optional base64-encoded image (lab report, ultrasound frame) "
            "for the agent to visually analyze using Gemini Vision."
        ),
    )
    image_content_type: Optional[str] = Field(
        default=None,
        description="MIME type of the attached image (e.g. 'image/png', 'image/jpeg').",
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Session ID to retrieve the patient's diagnostic context. "
            "When provided, the AI agent receives the cumulative results "
            "from clinical assessment, ultrasound, and FNAC nodes."
        ),
    )

    @model_validator(mode="after")
    def _require_query_or_image(self) -> "ChatRequest":
        """Ensure the client sends at least a query or an image."""
        if not self.query and not self.image_base64:
            raise ValueError(
                "At least one of 'query' or 'image_base64' must be provided."
            )
        return self


class AgentChatRequest(BaseModel):
    """JSON payload for Phase 3 refactored agent chat."""
    patient_id: int
    session_id: str
    doctor_id: int
    user_message: str


class ChatResponse(BaseModel):
    """Response from the ThyraX AI agent."""
    status: str
    query: Optional[str] = None
    response: str
    tools_used: List[str] = Field(default_factory=list)
