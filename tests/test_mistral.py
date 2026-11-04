# tests/test_mistral.py

"""Tests for the Mistral adapter's production-compatible stream behavior."""

from collections.abc import AsyncGenerator
from typing import cast
from unittest.mock import AsyncMock, MagicMock

from openai_client.client.mistral_ai import MistralAI


class TestMistralStreamCleanup:
    async def test_closes_source_on_early_consumer_exit(self) -> None:
        instance = MistralAI(
            api_key="test-key",
            model="test-model",
            base_url="https://api.mistral.ai",
        )
        object.__setattr__(instance, "chat", AsyncMock())

        closed = False

        async def source() -> AsyncGenerator[object]:
            nonlocal closed
            try:
                while True:
                    yield object()
            finally:
                closed = True

        chat = cast(AsyncMock, instance.chat)
        chat.stream_async.return_value = source()
        generator = await instance.generate({"messages": []}, stream=True)

        await anext(generator)
        await generator.aclose()

        assert closed


class TestMistralNoReasoningEffort:
    """Mistral rejects reasoning_effort; it must never be forwarded."""

    async def test_non_streaming_omits_reasoning_effort(self) -> None:
        instance = MistralAI(
            api_key="test-key",
            model="test-model",
            base_url="https://api.mistral.ai",
        )
        object.__setattr__(instance, "chat", AsyncMock())

        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = "OK"

        chat = cast(AsyncMock, instance.chat)
        chat.complete_async = AsyncMock(return_value=completion)

        result = await instance.generate({"messages": []}, stream=False)

        assert result == "OK"
        assert "reasoning_effort" not in chat.complete_async.call_args.kwargs

    async def test_streaming_omits_reasoning_effort(self) -> None:
        instance = MistralAI(
            api_key="test-key",
            model="test-model",
            base_url="https://api.mistral.ai",
        )
        object.__setattr__(instance, "chat", AsyncMock())

        async def source() -> AsyncGenerator[object]:
            if False:
                yield

        chat = cast(AsyncMock, instance.chat)
        chat.stream_async = AsyncMock(return_value=source())

        generator = await instance.generate({"messages": []}, stream=True)
        _ = [chunk async for chunk in generator]

        assert "reasoning_effort" not in chat.stream_async.call_args.kwargs
