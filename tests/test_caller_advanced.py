# tests/test_caller_advanced.py

"""Tests for production completion and streaming behavior."""

from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import APIConnectionError, AuthenticationError

from openai_client.client.openai_client import OpenAIClient
from openai_client.configs import Providers as Prv
from openai_client.configs import settings
from openai_client.decorators.circuit_breaker import agent_cb
from openai_client.errors import AgentAuthError, AgentNetworkError


@pytest.fixture(autouse=True)
async def _reset_breaker() -> AsyncIterator[None]:
    await agent_cb.reset()
    yield
    await agent_cb.reset()


def _make_client(prv: Prv = Prv.INCEPTION) -> OpenAIClient:
    instance = OpenAIClient(
        api_key="test-key",
        base_url="https://api.example.com/v1",
        model="test-model",
        prv=prv,
    )
    object.__setattr__(instance, "chat", AsyncMock())
    return instance


def _mock_chat(instance: OpenAIClient) -> AsyncMock:
    return cast(AsyncMock, instance.chat)


class TestGenerateNonStreaming:
    async def test_returns_content(self) -> None:
        instance = _make_client()
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = "Hello, world!"
        _mock_chat(instance).completions.create.return_value = completion

        result = await instance.generate({"messages": []})
        assert result == "Hello, world!"

    async def test_returns_none_on_empty_choices(self) -> None:
        instance = _make_client()
        completion = MagicMock()
        completion.choices = []
        _mock_chat(instance).completions.create.return_value = completion

        result = await instance.generate({"messages": []})
        assert result is None

    async def test_passes_production_reasoning_options(self) -> None:
        instance = _make_client()
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = "ok"
        create = _mock_chat(instance).completions.create
        create.return_value = completion

        await instance.generate({"messages": [], "temperature": 0.0})

        kwargs = create.call_args.kwargs
        assert kwargs["model"] == "test-model"
        assert kwargs["reasoning_effort"] == "high"
        assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}

    async def test_maps_authentication_error(self) -> None:
        instance = _make_client()
        _mock_chat(instance).completions.create.side_effect = AuthenticationError(
            "bad key",
            response=MagicMock(),
            body=None,
        )

        with pytest.raises(AgentAuthError):
            await instance.generate({"messages": []})

    async def test_does_not_retry_auth_errors(self) -> None:
        instance = _make_client()
        create = _mock_chat(instance).completions.create
        create.side_effect = AuthenticationError(
            "bad key",
            response=MagicMock(),
            body=None,
        )

        with pytest.raises(AgentAuthError):
            await instance.generate({"messages": []})

        assert create.await_count == 1

    async def test_retries_network_errors(self) -> None:
        instance = _make_client()
        create = _mock_chat(instance).completions.create
        create.side_effect = APIConnectionError(request=MagicMock())

        with pytest.raises(AgentNetworkError):
            await instance.generate({"messages": []})

        assert create.await_count == settings.AI_MAX_RETRIES


class TestGenerateStreaming:
    async def test_yields_chunks(self) -> None:
        chunks = [MagicMock(), MagicMock()]

        async def stream() -> AsyncIterator[MagicMock]:
            for chunk in chunks:
                yield chunk

        instance = _make_client()
        _mock_chat(instance).completions.create.return_value = stream()

        generator = await instance.generate({"messages": []}, stream=True)
        collected = [chunk async for chunk in generator]
        assert collected == chunks

    async def test_stream_config_is_forwarded(self) -> None:
        async def stream() -> AsyncIterator[MagicMock]:
            yield MagicMock()

        instance = _make_client()
        create = _mock_chat(instance).completions.create
        create.return_value = stream()

        await instance.generate({"messages": [], "stream": True}, stream=True)
        assert create.call_args.kwargs["stream"] is True

    async def test_does_not_mutate_caller_configs(self) -> None:
        async def stream() -> AsyncIterator[MagicMock]:
            yield MagicMock()

        instance = _make_client()
        create = _mock_chat(instance).completions.create
        create.return_value = stream()

        configs = {"messages": [], "temperature": 0.0}
        generator = await instance.generate(configs, stream=True)
        _ = [chunk async for chunk in generator]

        assert "stream" not in configs

    async def test_stream_init_does_not_retry_auth_errors(self) -> None:
        instance = _make_client()
        create = _mock_chat(instance).completions.create
        create.side_effect = AuthenticationError(
            "bad key",
            response=MagicMock(),
            body=None,
        )

        with pytest.raises(AgentAuthError):
            await instance.generate({"messages": []}, stream=True)

        assert create.await_count == 1

    async def test_stream_init_retries_network_errors(self) -> None:
        instance = _make_client()
        create = _mock_chat(instance).completions.create
        create.side_effect = APIConnectionError(request=MagicMock())

        with pytest.raises(AgentNetworkError):
            await instance.generate({"messages": []}, stream=True)

        assert create.await_count == settings.AI_MAX_RETRIES


class TestMistralOpenAICompatible:
    """Mistral rejects OpenAI-only reasoning fields with 422/400."""

    async def test_non_streaming_omits_reasoning_options(self) -> None:
        instance = _make_client(Prv.MISTRAL)
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = "ok"
        create = _mock_chat(instance).completions.create
        create.return_value = completion

        await instance.generate({"messages": [], "temperature": 0.0})

        kwargs = create.call_args.kwargs
        assert kwargs["model"] == "test-model"
        assert "reasoning_effort" not in kwargs
        assert "extra_body" not in kwargs

    async def test_streaming_omits_reasoning_options(self) -> None:
        async def stream() -> AsyncIterator[MagicMock]:
            yield MagicMock()

        instance = _make_client(Prv.MISTRAL)
        create = _mock_chat(instance).completions.create
        create.return_value = stream()

        generator = await instance.generate({"messages": []}, stream=True)
        _ = [chunk async for chunk in generator]

        kwargs = create.call_args.kwargs
        assert kwargs["stream"] is True
        assert "reasoning_effort" not in kwargs
        assert "extra_body" not in kwargs


class TestCloseCompletionStream:
    async def test_closes_iterator_and_stream(self) -> None:
        stream = MagicMock()
        stream.aclose = AsyncMock()
        iterator = MagicMock()
        iterator.aclose = AsyncMock()

        object.__setattr__(stream, "_iterator", iterator)
        await OpenAIClient._close_completion_stream(stream)  # noqa: SLF001

        iterator.aclose.assert_awaited_once()
        stream.aclose.assert_awaited_once()
