# tests/test_caller.py

"""Tests for the caller module."""

from unittest.mock import MagicMock

import pytest
from openai import (
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from openai_client.client.openai_client import OpenAIClient, _handle_api_call
from openai_client.errors import (
    SYSTEM_EXCEPTIONS,
    ApiError,
    AppSystemError,
    UnknownError,
)


class TestCleanupAnswer:
    def test_strips_whitespace(self) -> None:
        assert OpenAIClient._cleanup_answer("  hello  ") == "hello"

    def test_removes_faq_tag(self) -> None:
        assert (
            OpenAIClient._cleanup_answer("<think> reasoning </think> actual answer")
            == "actual answer"
        )

    def test_no_faq_tag(self) -> None:
        assert OpenAIClient._cleanup_answer("just plain text") == "just plain text"

    def test_empty_string(self) -> None:
        assert OpenAIClient._cleanup_answer("") == ""

    def test_multiple_faq_tags(self) -> None:
        assert (
            OpenAIClient._cleanup_answer("<think> a </think> <think> b </think> answer")
            == "<think> b </think> answer"
        )


class TestHandleApiCall:
    async def test_yields_success(self) -> None:
        async with _handle_api_call() as result:
            pass
        assert result is None

    async def test_raises_api_error_on_api_exception(self) -> None:

        standard_exceptions = (
            AuthenticationError,
            BadRequestError,
            PermissionDeniedError,
            NotFoundError,
            ConflictError,
            UnprocessableEntityError,
            InternalServerError,
        )

        for exc_type in standard_exceptions:
            with pytest.raises(ApiError):
                async with _handle_api_call():
                    raise exc_type("test", response=MagicMock(), body={"error": "test"})

    async def test_raises_api_error_on_rate_limit(self) -> None:
        with pytest.raises(ApiError):
            async with _handle_api_call():
                details = "rate limited"
                raise RateLimitError(details, response=MagicMock(), body=None)

    async def test_raises_app_system_error_on_system_exception(self) -> None:
        for exc_type in SYSTEM_EXCEPTIONS:
            with pytest.raises(AppSystemError):
                async with _handle_api_call():
                    details = "system error"
                    raise exc_type(details)

    async def test_raises_unknown_error_on_other_exception(self) -> None:
        with pytest.raises(UnknownError):
            async with _handle_api_call():
                raise RuntimeError("unexpected")
