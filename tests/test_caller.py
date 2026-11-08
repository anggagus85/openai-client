# tests/test_caller.py

"""Tests for the OpenAI client surface and AI error mapping."""

from unittest.mock import MagicMock

import pytest
from openai import (
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)

from openai_client.client.openai_client import OpenAIClient
from openai_client.configs import Providers as Prv
from openai_client.errors import (
    AgentAuthError,
    AgentNetworkError,
    AgentQuotaExceededError,
    raise_ai_error,
)


def _response() -> MagicMock:
    return MagicMock()


class TestClientInit:
    def test_sets_model_and_base_url(self) -> None:
        client = OpenAIClient(
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="test-model",
            prv=Prv.INCEPTION,
        )
        assert client.model == "test-model"
        assert client.is_mistral is False

    def test_str(self) -> None:
        client = OpenAIClient(
            api_key="test-key",
            base_url="https://api.example.com/v1",
            model="test-model",
            prv=Prv.INCEPTION,
        )
        assert "test-model" in str(client)
        assert "https://api.example.com/v1" in str(client)


class TestRaiseAIError:
    def test_authentication_error(self) -> None:
        with pytest.raises(AgentAuthError):
            raise_ai_error(AuthenticationError("bad key", response=_response(), body=None))

    def test_rate_limit_error(self) -> None:
        with pytest.raises(AgentQuotaExceededError):
            raise_ai_error(RateLimitError("limited", response=_response(), body=None))

    def test_internal_server_error(self) -> None:
        with pytest.raises(AgentNetworkError):
            raise_ai_error(InternalServerError("down", response=_response(), body=None))

    def test_connection_error(self) -> None:
        request = MagicMock()
        with pytest.raises(AgentNetworkError):
            raise_ai_error(APIConnectionError(request=request))

    def test_timeout_error(self) -> None:
        with pytest.raises(AgentNetworkError):
            raise_ai_error(TimeoutError("timed out"))

    def test_unknown_error_falls_back(self) -> None:
        with pytest.raises(Exception, match="trouble with our agent"):
            raise_ai_error(RuntimeError("unexpected"))
