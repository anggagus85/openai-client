# src/openai_client/errors/errors.py

from logging import getLogger
from re import search
from typing import NoReturn

from httpx2 import ReadTimeout
from mistralai.client.errors import MistralError
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    ContentFilterFinishReasonError,
    InternalServerError,
    InvalidWebhookSignatureError,
    LengthFinishReasonError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

logger = getLogger(__name__)

API_EXCEPTIONS = (
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    NotFoundError,
    ConflictError,
    UnprocessableEntityError,
    InternalServerError,
    LengthFinishReasonError,
    ContentFilterFinishReasonError,
    InvalidWebhookSignatureError,
    ReadTimeout,
)

SYSTEM_EXCEPTIONS = (
    SystemExit,
    EOFError,
    OSError,
    PermissionError,
    ConnectionError,
    TimeoutError,
    FileNotFoundError,
)


class ApiError(Exception):
    pass


class UnknownError(Exception):
    pass


class AppSystemError(Exception):
    pass


# ---------------------------------------------------------------------------
# Production agent error hierarchy (mirrors app/errors/agent.py)
# ---------------------------------------------------------------------------


class CircuitBreakerError(Exception):
    """Raised when a circuit breaker is open."""

    def __init__(
        self,
        detail: str = "Service temporarily unavailable",
        retry_after: float = 0.0,
        circuit_name: str = "default",
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.retry_after = retry_after
        self.circuit_name = circuit_name


class AgentError(Exception):
    """Base exception for agent errors."""

    def __init__(
        self,
        detail: str = "We're having trouble with our agent. Please try again in a moment.",
        status_code: int = 500,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class AgentInvalidAPIKeyError(AgentError):
    def __init__(self, detail: str = "Invalid API key provided.") -> None:
        super().__init__(detail, status_code=400)


class AgentPermissionDeniedError(AgentError):
    def __init__(
        self,
        detail: str = "Your project has been denied access. Please contact support",
    ) -> None:
        super().__init__(detail, status_code=403)


class AgentAuthError(AgentError):
    def __init__(
        self,
        detail: str = "Our AI service is temporarily unavailable. Please try again later.",
    ) -> None:
        super().__init__(detail=detail, status_code=401)


class AgentQuotaExceededError(AgentError):
    def __init__(
        self,
        detail: str = "We've reached our AI service limit. Please try again in a few moments.",
    ) -> None:
        super().__init__(detail=detail, status_code=429)


class AgentNetworkError(AgentError):
    def __init__(
        self,
        detail: str = "We're having trouble connecting to our AI service. Please try again.",
    ) -> None:
        super().__init__(detail=detail, status_code=503)


class AgentResponseError(AgentError):
    def __init__(
        self,
        detail: str = "We received an unexpected response from our AI service. Please try again.",
    ) -> None:
        super().__init__(detail=detail, status_code=502)


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


def _handle_openai_error(e: Exception) -> None:
    """Handle OpenAI SDK errors."""
    if isinstance(e, AuthenticationError):
        raise AgentAuthError() from e
    if isinstance(e, RateLimitError):
        raise AgentQuotaExceededError() from e
    if isinstance(e, (APIConnectionError, APITimeoutError)):
        raise AgentNetworkError() from e
    if isinstance(e, InternalServerError):
        raise AgentNetworkError(
            detail="Our AI service is experiencing internal issues. Please try again later.",
        ) from e


def _handle_mistral_error(e: Exception) -> None:
    """Handle MistralAI SDK errors."""
    if isinstance(e, MistralError):
        status_code = getattr(e, "status_code", None)
        if status_code == 401:
            raise AgentAuthError() from e
        if status_code == 429:
            raise AgentQuotaExceededError() from e

        err_str = str(e).lower()
        if "401" in err_str or "unauthorized" in err_str:
            raise AgentAuthError() from e
        if any(p in err_str for p in ("429", "rate limit", "rate-limited", "too many requests")):
            raise AgentQuotaExceededError() from e

        logger.error("Mistral SDK error: %s", e)
        raise AgentError(detail="Mistral Agent service error. Please try again later.") from e


def raise_ai_error(e: Exception) -> NoReturn:
    """
    Map provider exceptions to user-facing AgentError subclasses.

    Mirrors the production dispatcher for OpenAI and Mistral. Google GenAI
    handling is intentionally excluded: this repository only mirrors the
    OpenAI agent systems.
    """
    _handle_openai_error(e)
    _handle_mistral_error(e)

    if isinstance(e, (ConnectionError, TimeoutError)):
        raise AgentNetworkError() from e

    error_msg = str(e)

    if "401" in error_msg or "unauthenticated" in error_msg.lower():
        raise AgentAuthError() from e

    if "429" in error_msg or "quota" in error_msg.lower():
        if retry_match := search(r"retryDelay': '([\d\.]+)s'", error_msg):
            seconds = retry_match.group(1)
            detail = (
                f"Our AI service is currently very busy. Please try again in {seconds} seconds."
            )
            raise AgentQuotaExceededError(detail=detail) from e
        raise AgentQuotaExceededError() from e

    if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
        raise AgentNetworkError() from e

    logger.error("Unhandled AI service error: %s", e)
    raise AgentError() from e
