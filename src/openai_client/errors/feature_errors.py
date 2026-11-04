"""Domain-specific agent errors for generation and inquiry features."""

from openai_client.errors.exceptions import AgentError


class AgentGenerationError(AgentError):
    """Raised when AI content generation fails."""

    def __init__(
        self,
        detail: str = (
            "We couldn't generate the content you requested. "
            "Please try again or rephrase your request."
        ),
    ) -> None:
        super().__init__(detail=detail, status_code=502)

    def __str__(self) -> str:
        return f"{self.detail}"


class AgentClientError(AgentError):
    """Raised when the AI client encounters transport or protocol errors."""

    def __init__(
        self,
        detail: str = "We're having trouble with our AI service. Please try again in a moment.",
    ) -> None:
        super().__init__(detail)
        self.status_code = 502


class ItineraryGenerationError(AgentError):
    """Raised when AI itinerary generation fails."""

    def __init__(
        self,
        detail: str = (
            "Sorry, we couldn't create your itinerary right now. "
            "Please try again or contact us for personalized assistance."
        ),
    ) -> None:
        super().__init__(detail)
        self.status_code = 502

    def __str__(self) -> str:
        return f"{self.detail}"


class QueryProcessingError(AgentError):
    """Raised when AI query processing fails."""

    def __init__(
        self,
        detail: str = (
            "We couldn't process your question. Please try rephrasing it or ask something else."
        ),
    ) -> None:
        super().__init__(detail)
        self.status_code = 502

    def __str__(self) -> str:
        return f"{self.detail}"


class ContactAnalysisError(AgentError):
    """Raised when AI contact analysis fails."""

    def __init__(
        self,
        detail: str = (
            "We couldn't analyze your message. Please try again or submit your inquiry directly."
        ),
    ) -> None:
        super().__init__(detail)
        self.status_code = 502

    def __str__(self) -> str:
        return f"{self.detail}"
