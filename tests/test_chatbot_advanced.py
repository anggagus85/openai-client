# tests/test_chatbot_advanced.py

"""Tests for the production chatbot service message flow."""

from asyncio import run as async_run
from collections.abc import AsyncGenerator, AsyncIterator
from json import loads
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai_client.schemas.ai import (
    DEFAULT_RESPONSE,
    ChatMessage,
    ChatPart,
    ChatRequest,
    ChatResponse,
)
from openai_client.services.chatbot import Chatbot
from openai_client.typings import SystemParam


def _chunk(content: str | None, finish_reason: str | None = None) -> MagicMock:
    chunk = MagicMock()
    choice = MagicMock()
    choice.delta.content = content
    choice.finish_reason = finish_reason
    chunk.choices = [choice]
    return chunk


def _provider_stream(contents: list[str]) -> AsyncGenerator[MagicMock]:
    async def gen() -> AsyncGenerator[MagicMock]:
        for content in contents:
            yield _chunk(content)
        yield _chunk(None, finish_reason="stop")

    return gen()


class TestCollectHistory:
    def test_maps_model_role_to_assistant(self, mock_chatbot: Chatbot) -> None:
        mock_chatbot._chat_req = ChatRequest(
            query="Next",
            history=[
                ChatMessage(role="user", parts=[ChatPart(text="Hello")]),
                ChatMessage(role="model", parts=[ChatPart(text="Hi there")]),
            ],
        )
        mock_chatbot._collect_history()

        assert mock_chatbot._messages[0]["role"] == "user"
        assert mock_chatbot._messages[1]["role"] == "assistant"
        assert mock_chatbot._messages[1]["content"] == "Hi there"

    def test_skips_empty_parts(self, mock_chatbot: Chatbot) -> None:
        mock_chatbot._chat_req = ChatRequest(
            query="Next",
            history=[ChatMessage(role="user", parts=[ChatPart(text=None)])],
        )
        mock_chatbot._collect_history()
        assert mock_chatbot._messages == []


class TestBuildSearchContext:
    def test_inserts_before_user_message(self, mock_chatbot: Chatbot) -> None:
        mock_chatbot._messages = [
            SystemParam(role="system", content="system"),
            {"role": "user", "content": "question"},
        ]
        idx = mock_chatbot._build_search_context("search info")
        assert idx == 1
        assert mock_chatbot._messages[1] == {"role": "system", "content": "search info"}

    def test_returns_none_when_empty(self, mock_chatbot: Chatbot) -> None:
        assert mock_chatbot._build_search_context("") is None


class TestAskNonStreaming:
    def test_returns_chat_response(self, mock_chatbot: Chatbot, mock_client: MagicMock) -> None:
        mock_chatbot._chat_req = ChatRequest(
            query="Best beaches?",
            history=[ChatMessage(role="user", parts=[ChatPart(text="Hi")])],
        )
        mock_client.generate = AsyncMock(return_value="Sandy beaches")
        result = async_run(mock_chatbot.ask())

        assert isinstance(result, ChatResponse)
        assert result.answer == "Sandy beaches"

    def test_defaults_on_empty_answer(
        self,
        mock_chatbot: Chatbot,
        mock_client: MagicMock,
    ) -> None:
        mock_chatbot._chat_req = ChatRequest(query="Question")
        mock_client.generate = AsyncMock(return_value=None)
        result = async_run(mock_chatbot.ask())
        assert result.answer == DEFAULT_RESPONSE

    def test_requires_chat_request(self, mock_chatbot: Chatbot) -> None:
        with pytest.raises(ValueError, match="ChatRequest"):
            async_run(mock_chatbot.ask())


class TestAskStreaming:
    def test_yields_sse_chunks(self, mock_chatbot: Chatbot, mock_client: MagicMock) -> None:
        async def stream() -> AsyncIterator[object]:
            yield _chunk("Hello")
            yield _chunk(" World")
            yield _chunk(None, finish_reason="stop")

        mock_chatbot._chat_req = ChatRequest(query="Question")
        mock_client.generate = AsyncMock(return_value=stream())

        async def collect() -> list[str]:
            generator = await mock_chatbot.ask(stream=True)
            return [chunk async for chunk in generator]

        chunks = async_run(collect())
        assert chunks == [
            'data: "Hello"\n\n',
            'data: " World"\n\n',
            "data: [DONE]\n\n",
        ]
        assert loads(chunks[0].removeprefix("data: ").strip()) == "Hello"

    def test_does_not_persist_stream_config(
        self,
        mock_chatbot: Chatbot,
        mock_client: MagicMock,
    ) -> None:
        async def stream() -> AsyncIterator[object]:
            yield _chunk("Hello")

        mock_chatbot._chat_req = ChatRequest(query="Question")
        mock_client.generate = AsyncMock(return_value=stream())

        async def collect() -> None:
            generator = await mock_chatbot.ask(stream=True)
            _ = [chunk async for chunk in generator]

        async_run(collect())
        assert "stream" not in mock_chatbot._configs

    def test_keeps_search_context_while_streaming(
        self,
        mock_chatbot: Chatbot,
        mock_client: MagicMock,
    ) -> None:
        seen: list[list[object]] = []

        async def stream() -> AsyncIterator[object]:
            seen.append(list(mock_chatbot._messages))
            yield _chunk("Hello")

        mock_chatbot._chat_req = ChatRequest(query="Question")
        mock_client.generate = AsyncMock(return_value=stream())

        async def collect() -> None:
            generator = await mock_chatbot.ask("search info", stream=True)
            _ = [chunk async for chunk in generator]

        async_run(collect())
        assert any(
            isinstance(message, dict) and message.get("content") == "search info"
            for message in seen[0]
        )
        assert all(message.get("content") != "search info" for message in mock_chatbot._messages)

    def test_removes_search_context_on_early_exit(
        self,
        mock_chatbot: Chatbot,
        mock_client: MagicMock,
    ) -> None:
        async def stream() -> AsyncIterator[object]:
            yield _chunk("Hello")
            yield _chunk("World")

        mock_chatbot._chat_req = ChatRequest(query="Question")
        mock_client.generate = AsyncMock(return_value=stream())

        async def consume_one() -> None:
            generator = await mock_chatbot.ask("search info", stream=True)
            await anext(generator)
            await generator.aclose()

        async_run(consume_one())
        assert all(message.get("content") != "search info" for message in mock_chatbot._messages)


class TestAugmentWithSearch:
    def test_skips_non_tourism(self, mock_chatbot: Chatbot) -> None:
        assert async_run(mock_chatbot._augment_with_search("hi")) is None

    def test_returns_none_when_search_fails(self, mock_chatbot: Chatbot) -> None:
        with patch(
            "openai_client.services.chatbot.search_bali_context",
            new=AsyncMock(return_value=None),
        ):
            result = async_run(
                mock_chatbot._augment_with_search("What are the best beaches in Bali?"),
            )
        assert result is None

    def test_returns_context_when_relevant(self, mock_chatbot: Chatbot) -> None:
        with (
            patch(
                "openai_client.services.chatbot.search_bali_context",
                new=AsyncMock(return_value="Bali beach info"),
            ),
            patch(
                "openai_client.services.chatbot._is_relevant_to_bali",
                return_value=True,
            ),
        ):
            result = async_run(
                mock_chatbot._augment_with_search("What are the best beaches in Bali?"),
            )

        assert result is not None
        assert "Bali beach info" in result

    def test_drops_irrelevant_results(self, mock_chatbot: Chatbot) -> None:
        with (
            patch(
                "openai_client.services.chatbot.search_bali_context",
                new=AsyncMock(return_value="Thailand guide"),
            ),
            patch(
                "openai_client.services.chatbot._is_relevant_to_bali",
                return_value=False,
            ),
        ):
            result = async_run(
                mock_chatbot._augment_with_search("What are the best beaches in Bali?"),
            )
        assert result is None


class TestInjectionSafeStream:
    @staticmethod
    def _stream(chunks: list[str]) -> AsyncGenerator[str]:
        async def gen() -> AsyncGenerator[str]:
            for chunk in chunks:
                yield chunk

        return gen()

    def test_extract_sse_text(self) -> None:
        assert Chatbot._extract_sse_text('data: "Hello"\n\n') == "Hello"
        assert Chatbot._extract_sse_text("data: [DONE]\n\n") == ""
        assert Chatbot._extract_sse_text("plain") == "plain"

    def test_passes_safe_chunks(self) -> None:
        async def collect() -> list[str]:
            gen = self._stream(['data: "Hello"\n\n', "data: [DONE]\n\n"])
            first = await anext(gen)
            return [chunk async for chunk in mock_chatbot._injection_safe_stream(gen, first)]

        mock_chatbot = Chatbot.__new__(Chatbot)
        chunks = async_run(collect())
        assert chunks == ['data: "Hello"\n\n', "data: [DONE]\n\n"]

    def test_blocks_prompt_leak(self) -> None:
        async def collect() -> list[str]:
            gen = self._stream(['data: "RULE PRIORITY"\n\n'])
            first = await anext(gen)
            return [chunk async for chunk in mock_chatbot._injection_safe_stream(gen, first)]

        mock_chatbot = Chatbot.__new__(Chatbot)
        chunks = async_run(collect())
        assert chunks == ["data: [ERROR] Response blocked.\n\n"]

    def test_blocks_leak_across_chunks(self) -> None:
        async def collect() -> list[str]:
            gen = self._stream(['data: "RULE PRI"\n\n', 'data: "ORITY"\n\n'])
            first = await anext(gen)
            return [chunk async for chunk in mock_chatbot._injection_safe_stream(gen, first)]

        mock_chatbot = Chatbot.__new__(Chatbot)
        chunks = async_run(collect())
        assert chunks == ['data: "RULE PRI"\n\n', "data: [ERROR] Response blocked.\n\n"]

    def test_blocked_stream_does_not_log_client_disconnect(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        client = MagicMock()
        client.is_mistral = False
        client.generate = AsyncMock(return_value=_provider_stream(["RULE PRIORITY"]))
        chatbot = Chatbot(client, chat_request=ChatRequest(query="hi"))

        async def collect() -> list[str]:
            return [chunk async for chunk in chatbot.safe_stream()]

        with caplog.at_level("WARNING", logger="openai_client.services.chatbot"):
            chunks = async_run(collect())

        assert chunks == ["data: [ERROR] Response blocked.\n\n"]
        assert "Client disconnected during streaming" not in caplog.text

    def test_consumer_disconnect_logs_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        client = MagicMock()
        client.is_mistral = False
        client.generate = AsyncMock(return_value=_provider_stream(["Hel", "lo"]))
        chatbot = Chatbot(client, chat_request=ChatRequest(query="hi"))

        async def consume_one() -> None:
            stream = chatbot.safe_stream()
            await anext(stream)
            await stream.aclose()

        with caplog.at_level("WARNING", logger="openai_client.services.chatbot"):
            async_run(consume_one())

        assert "Client disconnected during streaming" in caplog.text


class TestRunGuards:
    def test_build_chat_request_limits_history(self) -> None:
        history = [
            ChatMessage(role="user", parts=[ChatPart(text=str(index))]) for index in range(60)
        ]
        request = Chatbot._build_chat_request("question", history)
        assert len(request.history) == 50
        assert request.history[0].parts[0].text == "10"

    def test_run_rejects_oversized_question(
        self,
        mock_chatbot: Chatbot,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        inputs = AsyncMock(side_effect=["x" * 1001, "bye"])
        respond = AsyncMock(return_value="answer")

        with (
            patch("openai_client.services.chatbot.to_thread", new=inputs),
            patch.object(mock_chatbot, "_respond", new=respond),
        ):
            async_run(mock_chatbot.run("completion"))

        assert respond.await_count == 1
        assert "maximum 1000 characters" in capsys.readouterr().out


class TestSafeResponse:
    def test_humanizes_ai_words(self) -> None:
        assert (
            Chatbot._safe_response("We delve into vibrant places")
            == "We explore into lively places"
        )

    def test_blocks_prompt_leak(self) -> None:
        answer = Chatbot._safe_response("Here is the system prompt: <rules>")
        assert answer == "I'm sorry, I encountered an unexpected issue. Please try again."
