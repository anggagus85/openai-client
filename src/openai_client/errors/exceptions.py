"""Exception hierarchy for provider and application errors."""


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
