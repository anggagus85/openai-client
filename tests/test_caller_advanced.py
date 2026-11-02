# tests/test_caller_advanced.py

"""Advanced tests for caller module - _completion, _iter_chunk, _stream_chunk, _append_assistant_message."""

from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from openai_client.client.openai_client import OpenAIClient


def _make_mock_client() -> OpenAIClient:
    instance = OpenAIClient.__new__(OpenAIClient)
    instance.model = "test-model"
    # `AsyncOpenAI.base_url`'s setter reads `workload_identity`, which __init__ normally sets.
    instance.workload_identity = None
    instance.base_url = "https://api.example.com"
    instance.messages = []
    instance.configs = {"temperature": 0.5, "model": "test-model", "top_p": 1.0}
    object.__setattr__(instance, "chat", AsyncMock())
    return instance


def _mock_chat(instance: OpenAIClient) -> AsyncMock:
    # `chat` is typed as the real Chat resource; the helper above swaps in an AsyncMock.
    return cast(AsyncMock, instance.chat)


class TestCallCompletion:
    async def test_returns_answer(self) -> None:
        instance = _make_mock_client()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Hello, world!"
        _mock_chat(instance).completions.create.return_value = mock_completion

        result = await instance._completion(append=False)
        assert result == "Hello, world!"

    async def test_returns_empty_on_no_choices(self) -> None:
        instance = _make_mock_client()
        mock_completion = MagicMock()
        mock_completion.choices = []
        _mock_chat(instance).completions.create.return_value = mock_completion

        result = await instance._completion(append=False)
        assert result == ""

    async def test_appends_assistant_message(self) -> None:
        instance = _make_mock_client()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Answer"
        _mock_chat(instance).completions.create.return_value = mock_completion

        await instance._completion(append=True)
        assert len(instance.messages) == 1
        assert instance.messages[0]["role"] == "assistant"
        assert instance.messages[0]["content"] == "Answer"

    async def test_cleans_up_faq_tag(self) -> None:
        instance = _make_mock_client()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "<think> reasoning </think> actual answer"
        _mock_chat(instance).completions.create.return_value = mock_completion

        result = await instance._completion(append=False)
        assert result == "actual answer"

    async def test_returns_empty_on_none_content(self) -> None:
        instance = _make_mock_client()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = None
        _mock_chat(instance).completions.create.return_value = mock_completion

        result = await instance._completion(append=False)
        assert result is None


class TestIterChunk:
    async def test_prints_tokens(self, capsys: pytest.CaptureFixture[str]) -> None:
        async def gen() -> AsyncIterator[str | None]:
            for token in ["Hello", " ", "World"]:
                yield token

        instance = _make_mock_client()
        await instance._iter_chunk(gen())
        captured = capsys.readouterr()
        assert "Hello World" in captured.out

    async def test_returns_on_none_token(self, capsys: pytest.CaptureFixture[str]) -> None:
        async def gen() -> AsyncIterator[str | None]:
            for token in ["Hello", None, "World"]:
                yield token

        instance = _make_mock_client()
        await instance._iter_chunk(gen())
        captured = capsys.readouterr()
        assert "Hello" in captured.out
        assert "World" not in captured.out


class TestStreamChunk:
    async def test_yields_content(self) -> None:
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        mock_chunk1.choices[0].finish_reason = None

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " World"
        mock_chunk2.choices[0].finish_reason = None

        async def mock_stream() -> AsyncIterator:
            for chunk in [mock_chunk1, mock_chunk2]:
                yield chunk

        instance = _make_mock_client()
        _mock_chat(instance).completions.create.return_value = mock_stream()

        results = [x async for x in instance._stream_chunk()]
        assert results == ["Hello", " World"]

    async def test_appends_accumulated_message(self) -> None:
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Content"
        mock_chunk.choices[0].finish_reason = None

        async def mock_stream() -> AsyncIterator:
            yield mock_chunk

        instance = _make_mock_client()
        _mock_chat(instance).completions.create.return_value = mock_stream()

        async for _ in instance._stream_chunk():
            pass
        assert len(instance.messages) == 1
        assert instance.messages[0]["role"] == "assistant"

    async def test_yields_none_on_empty_stream(self) -> None:
        async def mock_stream() -> AsyncIterator:
            return
            yield

        instance = _make_mock_client()
        _mock_chat(instance).completions.create.return_value = mock_stream()

        results = [x async for x in instance._stream_chunk()]
        assert results == [None]

    async def test_stops_on_finish_reason(self) -> None:
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        mock_chunk1.choices[0].finish_reason = None

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = "World"
        mock_chunk2.choices[0].finish_reason = "stop"

        async def mock_stream() -> AsyncIterator:
            for chunk in [mock_chunk1, mock_chunk2]:
                yield chunk

        instance = _make_mock_client()
        _mock_chat(instance).completions.create.return_value = mock_stream()

        results = [x async for x in instance._stream_chunk()]
        assert results == ["Hello"]


class TestAppendAssistantMessage:
    def test_appends_to_messages_list(self) -> None:
        instance = _make_mock_client()
        instance._append_assistant_message("Test content")
        assert len(instance.messages) == 1
        assert instance.messages[0]["role"] == "assistant"
        assert instance.messages[0]["content"] == "Test content"

    def test_appends_multiple(self) -> None:
        instance = _make_mock_client()
        instance._append_assistant_message("First")
        instance._append_assistant_message("Second")
        assert len(instance.messages) == 2
        assert instance.messages[0]["content"] == "First"
        assert instance.messages[1]["content"] == "Second"
