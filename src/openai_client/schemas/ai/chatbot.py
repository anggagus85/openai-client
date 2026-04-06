from pydantic import BaseModel, ConfigDict, Field, field_validator


class OpenAIChat(BaseModel):
    """Model for individual chat messages in conversation history."""

    model_config = ConfigDict(extra="ignore")

    role: str = Field(
        ...,
        description="The role of the message sender",
        examples=["user"],
    )
    content: str = Field(
        ...,
        description="The message content parts",
        examples=["Hello, how are you?"],
    )

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate message role."""
        allowed_roles = {"user", "assistant", "system"}
        if v not in allowed_roles:
            msg = f"Role must be one of: {allowed_roles}"
            raise ValueError(msg)
        return v


class OpenAIChatRequest(BaseModel):
    """Model for user query requests."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The user's query",
        examples=["What are the best beaches in Bali?"],
    )
    history: list[OpenAIChat] = Field(
        default=[],
        max_length=50,  # Limit conversation history
        description="The chat history",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """
        Validate query input - neutral validation only.

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


# --- Response Models ---


class ChatResponse(BaseModel):
    """Response model for query processing."""

    answer: str = Field(..., description="AI-generated response to the query")
