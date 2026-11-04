# src/openai_client/errors/__init__.py

from openai_client.errors.errors import (
    API_EXCEPTIONS,
    SYSTEM_EXCEPTIONS,
    raise_ai_error,
)
from openai_client.errors.exceptions import (
    AgentAuthError,
    AgentError,
    AgentInvalidAPIKeyError,
    AgentNetworkError,
    AgentPermissionDeniedError,
    AgentQuotaExceededError,
    AgentResponseError,
    ApiError,
    AppSystemError,
    CircuitBreakerError,
    UnknownError,
)
from openai_client.errors.feature_errors import (
    AgentClientError,
    AgentGenerationError,
    ContactAnalysisError,
    ItineraryGenerationError,
    QueryProcessingError,
)

__all__ = [
    "API_EXCEPTIONS",
    "SYSTEM_EXCEPTIONS",
    "AgentAuthError",
    "AgentClientError",
    "AgentError",
    "AgentGenerationError",
    "AgentInvalidAPIKeyError",
    "AgentNetworkError",
    "AgentPermissionDeniedError",
    "AgentQuotaExceededError",
    "AgentResponseError",
    "ApiError",
    "AppSystemError",
    "CircuitBreakerError",
    "ContactAnalysisError",
    "ItineraryGenerationError",
    "QueryProcessingError",
    "UnknownError",
    "raise_ai_error",
]
