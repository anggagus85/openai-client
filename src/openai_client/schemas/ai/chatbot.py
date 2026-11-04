# src/openai_client/schemas/ai/chatbot.py

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_HISTORY_PARTS = 8
MAX_PART_PAYLOAD_BYTES = 8 * 1024
MAX_MESSAGE_PAYLOAD_BYTES = 32 * 1024


class ChatPart(BaseModel):
    """A single textual part of a chat history message."""

    model_config = ConfigDict(extra="ignore")

    text: str | None = Field(
        default=None,
        description="The text content of the part",
        examples=["Hello, how are you?"],
    )


class ChatMessage(BaseModel):
    """Model for individual chat messages in conversation history."""

    model_config = ConfigDict(extra="ignore")

    role: str = Field(
        ...,
        description="The role of the message sender",
        examples=["user"],
    )
    parts: list[ChatPart] = Field(
        ...,
        min_length=1,
        max_length=MAX_HISTORY_PARTS,
        description="The message content parts",
        examples=[[ChatPart(text="Hello, how are you?")]],
    )

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate message role."""
        allowed_roles = {"user", "model", "system"}
        if v not in allowed_roles:
            msg = f"Role must be one of: {allowed_roles}"
            raise ValueError(msg)
        return v

    @field_validator("parts")
    @classmethod
    def validate_parts_payload(cls, v: list[ChatPart]) -> list[ChatPart]:
        """Bound each part and the total serialized message payload."""
        total_bytes = 0
        for index, part in enumerate(v):
            payload_bytes = len(part.model_dump_json(exclude_none=True).encode("utf-8"))
            if payload_bytes > MAX_PART_PAYLOAD_BYTES:
                msg = f"Part at index {index} exceeds the {MAX_PART_PAYLOAD_BYTES}-byte limit"
                raise ValueError(msg)
            total_bytes += payload_bytes

        if total_bytes > MAX_MESSAGE_PAYLOAD_BYTES:
            msg = f"Message parts exceed the {MAX_MESSAGE_PAYLOAD_BYTES}-byte limit"
            raise ValueError(msg)
        return v


class ChatRequest(BaseModel):
    """Model for user query requests."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The user's query",
        examples=["What are the best beaches in Bali?"],
    )
    history: list[ChatMessage] = Field(
        default=[],
        max_length=50,
        description="The chat history",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """
        Validate and normalize query input.

        Downstream consumers must apply context-specific sanitization:
        - HTML output: Use html.escape() or dedicated HTML sanitizer
        - Shell commands: Use shlex.quote() or shell escaping
        - SQL queries: Use parameterized queries
        - Other contexts: Apply appropriate escaping/sanitization
        """
        if not v or not v.strip():
            msg = "Query cannot be empty"
            raise ValueError(msg)

        return v.strip()


class ChatResponse(BaseModel):
    """Response model for query processing."""

    answer: str = Field(..., description="AI-generated response to the query")


DEFAULT_RESPONSE = "I'm sorry, I couldn't process your request. Please try again."
