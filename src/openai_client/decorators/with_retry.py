from collections.abc import Awaitable, Callable
from logging import getLogger
from typing import ParamSpec, TypeVar

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")
RETRIABLE_EXCEPTIONS = ConnectionError, TimeoutError


def _log_before_sleep(
    max_retries: int,
) -> Callable[[RetryCallState], None]:
    def before_sleep_callback(retry_state: RetryCallState) -> None:
        exception: BaseException | None = None
        if (outcome := retry_state.outcome) and outcome.failed:
            exception = outcome.exception()

        sleep_duration = 0
        func_name = retry_state.fn.__name__ if retry_state.fn else "unknown"
        if retry_state.next_action:
            sleep_duration = retry_state.next_action.sleep
            logger.warning(
                "Retry %d/%d for %s after %.2fs delay. Exception: %s",
                retry_state.attempt_number,
                max_retries,
                func_name,
                sleep_duration,
                exception,
            )

    return before_sleep_callback


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 2.0,
    exec_retry: type[Exception] | tuple[type[Exception], ...] = RETRIABLE_EXCEPTIONS,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Apply retry logic with exponential backoff to async functions using Tenacity.

    Args:
        max_attempts: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        exec_retry: Tuple of exception types to retry on.

    Returns:
        Decorated function with retry logic.
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=base_delay, min=base_delay, max=max_delay),
        retry=retry_if_exception_type(exec_retry),
        before_sleep=_log_before_sleep(max_attempts),
        reraise=True,
    )
