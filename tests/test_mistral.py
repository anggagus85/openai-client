# tests/test_mistral.py

"""Tests for the Mistral adapter's production-compatible stream behavior."""

from collections.abc import AsyncGenerator
from typing import cast
from unittest.mock import AsyncMock

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
