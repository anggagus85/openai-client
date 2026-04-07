# tests/test_chatbot_advanced.py

"""Advanced tests for chatbot service - _augment_with_search, _append_user_message."""

from asyncio import run as async_run
from unittest.mock import patch

from openai_client.services.chatbot import Chatbot
from openai_client.types import MistralUser, SystemParam, UserParam


class TestAppendUserMessage:
    def test_appends_user_message(self, mock_chatbot: Chatbot) -> None:
        chatbot = mock_chatbot

        chatbot._append_user_message("  Hello Bali  ")
        user_mssg = chatbot._messages[1]

        if isinstance(user_mssg, MistralUser):
            role = user_mssg.role
            content = user_mssg.content
        else:
            role = user_mssg.get("role")
            content = user_mssg.get("content")

        assert len(chatbot._messages) == 2
        assert role == "user"
        assert content == "Hello Bali"

    def test_strips_whitespace(self, mock_chatbot: Chatbot) -> None:
        chatbot = mock_chatbot

        chatbot._append_user_message("   query   ")

        user_mssg = chatbot._messages[1]
        if isinstance(user_mssg, MistralUser):
            content = user_mssg.content
        else:
            content = user_mssg.get("content")
        assert content == "query"


class TestAugmentWithSearch:
    def test_returns_original_for_non_tourism_query(self, mock_chatbot: Chatbot) -> None:
        chatbot = mock_chatbot
        chatbot._messages = [
            SystemParam(role="system", content="system"),
            UserParam(role="user", content="hi"),
        ]
        chatbot._question = "hi"

        result = async_run(chatbot._augment_with_search())
        assert result is None
        assert len(chatbot._messages) == 2

    def test_returns_original_when_search_fails(self, mock_chatbot: Chatbot) -> None:
        chatbot = mock_chatbot
        chatbot._messages = [
            SystemParam(role="system", content="system"),
            UserParam(
                role="user",
                content="What are the best beaches in Bali?",
            ),
        ]
        chatbot._question = "What are the best beaches in Bali?"

        with patch.object(chatbot, "_search_bali_context", return_value=None):
            result = async_run(chatbot._augment_with_search())

        assert result is None
        assert len(chatbot._messages) == 2

    def test_returns_augmented_when_search_succeeds(self, mock_chatbot: Chatbot) -> None:
        chatbot = mock_chatbot
        chatbot._messages = [
            SystemParam(role="system", content="system"),
            UserParam(
                role="user",
                content="What are the best beaches in Bali?",
            ),
        ]
        chatbot._question = "What are the best beaches in Bali?"

        with patch.object(chatbot, "_search_bali_context", return_value="Bali beach info"):
            result = async_run(chatbot._augment_with_search())

        assert result == 1
        assert len(chatbot._messages) == 3
