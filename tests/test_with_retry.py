# tests/test_with_retry.py

"""Tests for the with_retry decorator."""

from unittest.mock import MagicMock, patch

import pytest
from tenacity import RetryCallState

from openai_client.decorators.with_retry import (
    _log_before_sleep,
    with_retry,
)


class TestLogBeforeSleep:
    def test_returns_callable(self) -> None:
        callback = _log_before_sleep(max_retries=3)
        assert callable(callback)

    def test_logs_on_retry(self) -> None:
        callback = _log_before_sleep(max_retries=3)
        mock_retry_state = MagicMock(spec=RetryCallState)
        mock_retry_state.fn = MagicMock(__name__="test_func")
        mock_retry_state.attempt_number = 2
        mock_retry_state.next_action = MagicMock()
        mock_retry_state.next_action.sleep = 1.5
        mock_outcome = MagicMock()
        mock_outcome.failed = True
        mock_outcome.exception.return_value = ConnectionError("test")
        mock_retry_state.outcome = mock_outcome

        with patch("openai_client.decorators.with_retry.logger") as mock_logger:
            callback(mock_retry_state)
            mock_logger.warning.assert_called_once()


class TestWithRetry:
    def test_returns_callable(self) -> None:
        @with_retry(max_attempts=3)
        async def dummy() -> str:
            return "ok"

        assert callable(dummy)

    @pytest.mark.asyncio
    async def test_successful_call_no_retry(self) -> None:
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.01, max_delay=0.01)
        async def succeeds() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = await succeeds()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self) -> None:
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.01, max_delay=0.01)
        async def fails_twice() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("fail")
            return "success"

        result = await fails_twice()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self) -> None:
        @with_retry(max_attempts=2, base_delay=0.01, max_delay=0.01)
        async def always_fails() -> str:
            details = "always fails"
            raise ConnectionError(details)

        with pytest.raises(ConnectionError):
            await always_fails()

    @pytest.mark.asyncio
    async def test_does_not_retry_on_other_exceptions(self) -> None:
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.01, max_delay=0.01)
        async def raises_value_error() -> str:
            nonlocal call_count
            call_count += 1
            details = "not retriable"
            raise ValueError(details)

        with pytest.raises(ValueError):
            await raises_value_error()
        assert call_count == 1
