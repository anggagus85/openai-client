# tests/test_caller_advanced.py

"""Advanced tests for caller module - _call_completion, _call_bytez, _iter_chunk, _stream_chunk, _append_assistant_message."""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from bytez.client import Response

from openai_client.errors import ApiError, AppSystemError
from openai_client.services.caller.caller import (
    _append_assistant_message,
    _call_bytez,
    _call_completion,
    _iter_chunk,
    _stream_chunk,
)


class TestCallCompletion:
    async def test_returns_answer(self) -> None:
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Hello, world!"
        mock_client.completion.return_value = mock_completion

        messages: list = []
        result = await _call_completion(mock_client, messages, append=False)
        assert result == "Hello, world!"

    async def test_returns_empty_on_no_choices(self) -> None:
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"
        mock_completion = MagicMock()
        mock_completion.choices = []
        mock_client.completion.return_value = mock_completion

        messages: list = []
        result = await _call_completion(mock_client, messages, append=False)
        assert result == ""

    async def test_appends_assistant_message(self) -> None:
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Answer"
        mock_client.completion.return_value = mock_completion

        messages: list = []
        await _call_completion(mock_client, messages, append=True)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Answer"

    async def test_cleans_up_faq_tag(self) -> None:
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "<think> reasoning </think> actual answer"
        mock_client.completion.return_value = mock_completion

        messages: list = []
        result = await _call_completion(mock_client, messages, append=False)
        assert result == "actual answer"

    async def test_returns_empty_on_none_content(self) -> None:
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = None
        mock_client.completion.return_value = mock_completion

        messages: list = []
        result = await _call_completion(mock_client, messages, append=False)
        assert result is None


class TestCallBytez:
    async def test_returns_answer(self) -> None:
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"
        mock_response = MagicMock(spec=Response)
        mock_response.error = None
        mock_response.output = {"content": "Bytez answer"}
        mock_client.bytez.return_value = mock_response

        messages: list = []
        result = await _call_bytez(mock_client, messages, append=False)
        assert result == "Bytez answer"

    async def test_invalid_response_type_raises(self) -> None:
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"
        mock_client.bytez.return_value = "not a response"

        messages: list = []
        with pytest.raises(AppSystemError, match="Invalid response type"):
            await _call_bytez(mock_client, messages, append=False)

    async def test_error_response_raises(self) -> None:

        mock_client = AsyncMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"
        mock_response = MagicMock(spec=Response)
        mock_response.error = "API error"
        mock_response.output = {}
        mock_client.bytez.return_value = mock_response

        messages: list = []
        with pytest.raises(ApiError, match="API error"):
            await _call_bytez(mock_client, messages, append=False)

    async def test_invalid_output_type_raises(self) -> None:
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"
        mock_response = MagicMock(spec=Response)
        mock_response.error = None
        mock_response.output = "not a dict"
        mock_client.bytez.return_value = mock_response

        messages: list = []
        with pytest.raises(AppSystemError, match="Invalid output type"):
            await _call_bytez(mock_client, messages, append=False)

    async def test_missing_content_key_raises(self) -> None:
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"
        mock_response = MagicMock(spec=Response)
        mock_response.error = None
        mock_response.output = {"no_content": "here"}
        mock_client.bytez.return_value = mock_response

        messages: list = []
        with pytest.raises(AppSystemError, match="missing 'content' key"):
            await _call_bytez(mock_client, messages, append=False)

    async def test_appends_assistant_message(self) -> None:
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"
        mock_response = MagicMock(spec=Response)
        mock_response.error = None
        mock_response.output = {"content": "Bytez answer"}
        mock_client.bytez.return_value = mock_response

        messages: list = []
        await _call_bytez(mock_client, messages, append=True)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"

    async def test_cleans_up_faq_tag(self) -> None:
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"
        mock_response = MagicMock(spec=Response)
        mock_response.error = None
        mock_response.output = {"content": "<think> reasoning </think> bytez answer"}
        mock_client.bytez.return_value = mock_response

        messages: list = []
        result = await _call_bytez(mock_client, messages, append=False)
        assert result == "bytez answer"


class TestIterChunk:
    async def test_prints_non_bytez_tokens(self, capsys: pytest.CaptureFixture[str]) -> None:
        async def gen() -> AsyncIterator[str | None]:
            for token in ["Hello", " ", "World"]:
                yield token

        await _iter_chunk(gen(), is_bytez=False)
        captured = capsys.readouterr()
        assert "Hello World" in captured.out

    async def test_returns_on_none_token(self, capsys: pytest.CaptureFixture[str]) -> None:
        async def gen() -> AsyncIterator[str | None]:
            for token in ["Hello", None, "World"]:
                yield token

        await _iter_chunk(gen(), is_bytez=False)
        captured = capsys.readouterr()
        assert "Hello" in captured.out
        assert "World" not in captured.out

    async def test_skips_faq_prefix_for_bytez(self, capsys: pytest.CaptureFixture[str]) -> None:
        async def gen() -> AsyncIterator[str | None]:
            yield "<think> reasoning </think> answer"

        await _iter_chunk(gen(), is_bytez=True)
        captured = capsys.readouterr()
        assert "answer" in captured.out
        assert "reasoning" not in captured.out

    async def test_skips_leading_whitespace_after_faq(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        async def gen() -> AsyncIterator[str | None]:
            for token in ["<think> reasoning </think>", "\n\n", "  ", "actual answer"]:
                yield token

        await _iter_chunk(gen(), is_bytez=True)
        captured = capsys.readouterr()
        assert "actual answer" in captured.out


class TestStreamChunk:
    async def test_yields_content(self) -> None:
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"

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

        mock_client.completion.return_value = mock_stream()

        messages: list = []
        results = [x async for x in _stream_chunk(mock_client, messages)]
        assert results == ["Hello", " World"]

    async def test_appends_accumulated_message(self) -> None:
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Content"
        mock_chunk.choices[0].finish_reason = None

        async def mock_stream() -> AsyncIterator:
            yield mock_chunk

        mock_client.completion.return_value = mock_stream()

        messages: list = []
        async for _ in _stream_chunk(mock_client, messages):
            pass
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"

    async def test_yields_none_on_empty_stream(self) -> None:
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"

        async def mock_stream() -> AsyncIterator:
            return
            yield

        mock_client.completion.return_value = mock_stream()

        messages: list = []
        results = [x async for x in _stream_chunk(mock_client, messages)]
        assert results == [None]

    async def test_stops_on_finish_reason(self) -> None:
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"

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

        mock_client.completion.return_value = mock_stream()

        messages: list = []
        results = [x async for x in _stream_chunk(mock_client, messages)]
        assert results == ["Hello"]


class TestAppendAssistantMessage:
    def test_appends_to_messages_list(self) -> None:
        messages: list = []
        _append_assistant_message(messages, "Test content")
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Test content"

    def test_appends_multiple(self) -> None:
        messages: list = []
        _append_assistant_message(messages, "First")
        _append_assistant_message(messages, "Second")
        assert len(messages) == 2
        assert messages[0]["content"] == "First"
        assert messages[1]["content"] == "Second"
