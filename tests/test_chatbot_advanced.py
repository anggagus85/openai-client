# tests/test_chatbot_advanced.py

"""Advanced tests for chatbot service - _augment_with_search, _append_user_message."""

from asyncio import run as async_run
from unittest.mock import patch

from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from openai_client.services.chatbot import (
    _append_user_message,
    _augment_with_search,
)


class TestAppendUserMessage:
    def test_appends_user_message(self) -> None:
        messages: list = []
        _append_user_message("  Hello Bali  ", messages)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello Bali"

    def test_strips_whitespace(self) -> None:
        messages: list = []
        _append_user_message("   query   ", messages)
        assert messages[0]["content"] == "query"


class TestAugmentWithSearch:
    def test_returns_original_for_non_tourism_query(self) -> None:
        messages: list = [
            ChatCompletionSystemMessageParam(role="system", content="system"),
            ChatCompletionUserMessageParam(role="user", content="hi"),
        ]
        result = async_run(_augment_with_search("hi", messages))
        assert result is None
        assert len(messages) == 2

    def test_returns_original_when_search_fails(self) -> None:
        messages: list = [
            ChatCompletionSystemMessageParam(role="system", content="system"),
            ChatCompletionUserMessageParam(
                role="user",
                content="What are the best beaches in Bali?",
            ),
        ]

        with patch("openai_client.services.chatbot._search_bali_context", return_value=None):
            result = async_run(
                _augment_with_search("What are the best beaches in Bali?", messages),
            )

        assert result is None
        assert len(messages) == 2

    def test_returns_augmented_when_search_succeeds(self) -> None:
        messages: list = [
            ChatCompletionSystemMessageParam(role="system", content="system"),
            ChatCompletionUserMessageParam(
                role="user",
                content="What are the best beaches in Bali?",
            ),
        ]

        with patch(
            "openai_client.services.chatbot._search_bali_context",
            return_value="Bali beach info",
        ):
            result = async_run(
                _augment_with_search("What are the best beaches in Bali?", messages),
            )

        assert result == 1
        assert len(messages) == 3
