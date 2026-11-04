"""Map provider exceptions to user-facing agent errors."""

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

from openai_client.errors.exceptions import (
    AgentAuthError,
    AgentError,
    AgentNetworkError,
    AgentQuotaExceededError,
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
