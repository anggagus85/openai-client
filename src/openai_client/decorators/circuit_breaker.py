"""
Circuit breaker pattern implementation for external service calls.

Mirrors the BaliRoam backend circuit breaker used by the production OpenAI
client, with the Prometheus integration removed for standalone use.
"""

from asyncio import Lock
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from logging import getLogger
from time import time
from typing import ParamSpec, TypeVar, cast

from openai_client.errors import AgentError, CircuitBreakerError

logger = getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for CircuitBreaker."""

    name: str = "default"
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exceptions: type[Exception] | tuple[type[Exception], ...] = Exception
    half_open_max_calls: int = 1
    success_threshold: int = 1


class CircuitBreaker:
    """
    Async circuit breaker for external service calls.

    States:
        - CLOSED: Normal operation, all requests pass through
        - OPEN: Service is failing, requests are rejected immediately
        - HALF_OPEN: Testing recovery, limited requests allowed
    """

    __slots__ = (
        "_failure_count",
        "_half_open_successes",
        "_last_failure_time",
        "_lock",
        "_state",
        "expected_exceptions",
        "failure_threshold",
        "half_open_max_calls",
        "name",
        "recovery_timeout",
        "success_threshold",
    )

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self.name = config.name
        self.failure_threshold = config.failure_threshold
        self.recovery_timeout = config.recovery_timeout
        self.expected_exceptions = config.expected_exceptions
        self.half_open_max_calls = config.half_open_max_calls
        self.success_threshold = config.success_threshold

        self._failure_count: int = 0
        self._half_open_successes: int = 0
        self._last_failure_time: float | None = None
        self._state: CircuitState = CircuitState.CLOSED
        self._lock: Lock = Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state (read-only)."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count (read-only)."""
        return self._failure_count

    @property
    def last_failure_time(self) -> float | None:
        """Get timestamp of last failure (read-only)."""
        return self._last_failure_time

    async def call(
        self,
        func: Callable[P, Awaitable[T]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Execute an async function with circuit breaker protection."""
        exc = cast(type[Exception], self.expected_exceptions)
        async with self._lock:
            self.check_state()

        try:
            result = await func(*args, **kwargs)
            await self.on_success()
            return result
        except exc:
            await self.on_failure()
            raise

    def check_state(self) -> None:
        """Check circuit state and transition if necessary."""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._state = CircuitState.HALF_OPEN
                self._half_open_successes = 0
                logger.info("Circuit breaker '%s' entering HALF_OPEN state", self.name)
            else:
                retry_after = self._time_until_reset()
                logger.warning(
                    "Circuit breaker '%s' is OPEN. Retry in %.1fs",
                    self.name,
                    retry_after,
                )
                raise CircuitBreakerError(
                    detail=f"Service '{self.name}' temporarily unavailable",
                    retry_after=retry_after,
                    circuit_name=self.name,
                )

    def _should_attempt_reset(self) -> bool:
        if self._last_failure_time is None:
            return True
        return time() - self._last_failure_time >= self.recovery_timeout

    def _time_until_reset(self) -> float:
        if self._last_failure_time is None:
            return 0.0
        elapsed = time() - self._last_failure_time
        return max(0.0, self.recovery_timeout - elapsed)

    async def on_success(self) -> None:
        """Handle successful call - potentially close the circuit."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._half_open_successes = 0
                    logger.info("Circuit breaker '%s' recovered, now CLOSED", self.name)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def on_failure(self) -> None:
        """Handle failed call - potentially open the circuit."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._half_open_successes = 0
                logger.warning(
                    "Circuit breaker '%s' reopened after failure in HALF_OPEN",
                    self.name,
                )
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.error(
                    "Circuit breaker '%s' OPENED after %d consecutive failures",
                    self.name,
                    self._failure_count,
                )

    async def reset(self) -> None:
        """Manually reset circuit breaker to CLOSED state."""
        async with self._lock:
            self._failure_count = 0
            self._half_open_successes = 0
            self._last_failure_time = None
            self._state = CircuitState.CLOSED
        logger.info("Circuit breaker '%s' manually reset", self.name)

    def __call__(
        self,
        func: Callable[P, Awaitable[T]],
    ) -> Callable[P, Awaitable[T]]:
        """Wrap an async function with circuit breaker protection as a decorator."""

        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await self.call(func, *args, **kwargs)

        return wrapper


agent_cb = CircuitBreaker(
    config=CircuitBreakerConfig(
        name="openai_agent",
        failure_threshold=5,
        recovery_timeout=60.0,
        expected_exceptions=AgentError,
    ),
)


def circuit_breaker(
    cb: CircuitBreaker,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Wrap an async function with circuit breaker protection as a decorator."""

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await cb.call(func, *args, **kwargs)

        return wrapper

    return decorator
