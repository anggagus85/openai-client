# tests/test_chatbot_helpers.py

"""Tests for chatbot service helper functions."""

from unittest.mock import MagicMock

import pytest

from openai_client.services.chatbot import (
    _BALI_KEYWORDS,
    _SKIP_PATTERNS,
    _TOURISM_KEYWORDS,
    Chatbot,
    _chat_system_message,
)


@pytest.fixture
def chatbot() -> Chatbot:
    """Create a minimally valid Chatbot instance with mocked dependencies."""
    instance = Chatbot.__new__(Chatbot)
    instance._client = MagicMock()
    instance._messages = []
    instance._question = ""
    instance._search_results = ""
    return instance


class TestIsTourismQuery:
    def test_bali_keyword_detected(self, chatbot: Chatbot) -> None:
        chatbot._question = "What are the best beaches in Bali?"
        assert chatbot._is_tourism_query() is True

    def test_tourism_keyword_detected(self, chatbot: Chatbot) -> None:
        chatbot._question = "Can you recommend a good hotel?"
        assert chatbot._is_tourism_query() is True

    def test_greeting_not_tourism(self, chatbot: Chatbot) -> None:
        chatbot._question = "hi"
        assert chatbot._is_tourism_query() is False

    def test_short_input_not_tourism(self, chatbot: Chatbot) -> None:
        chatbot._question = "hello there"
        assert chatbot._is_tourism_query() is False

    @pytest.mark.parametrize("pattern", _SKIP_PATTERNS)
    def test_skip_patterns(self, chatbot: Chatbot, pattern: str) -> None:
        chatbot._question = pattern
        assert chatbot._is_tourism_query() is False

    def test_empty_input(self, chatbot: Chatbot) -> None:
        chatbot._question = ""
        assert chatbot._is_tourism_query() is False

    def test_whitespace_only(self, chatbot: Chatbot) -> None:
        chatbot._question = "   "
        assert chatbot._is_tourism_query() is False

    def test_specific_bali_location(self, chatbot: Chatbot) -> None:
        chatbot._question = "Tell me about Ubud rice terraces"
        assert chatbot._is_tourism_query() is True

    def test_food_related(self, chatbot: Chatbot) -> None:
        chatbot._question = "Where can I try nasi goreng?"
        assert chatbot._is_tourism_query() is True


class TestBuildSearchContextMessage:
    def test_returns_system_message(self, chatbot: Chatbot) -> None:
        chatbot._search_results = "Search result about Bali"
        msg = chatbot._build_search_context_message()
        assert msg["role"] == "system"
        assert isinstance(msg["content"], str)
        assert "search_context" in msg["content"]
        assert "Search result about Bali" in msg["content"]

    def test_contains_persona_reminder(self, chatbot: Chatbot) -> None:
        chatbot._search_results = "test results"
        msg = chatbot._build_search_context_message()
        assert isinstance(msg["content"], str)
        assert "BaliBlissed Concierge" in msg["content"]


class TestSystemMessage:
    def test_returns_openai_chat_message(self) -> None:
        msg = _chat_system_message()
        assert msg.role == "system"
        assert "Made" in msg.content
        assert "BaliBlissed" in msg.content

    def test_contains_personality(self) -> None:
        msg = _chat_system_message()
        assert "personality" in msg.content.lower()

    def test_contains_boundaries(self) -> None:
        msg = _chat_system_message()
        assert "boundaries" in msg.content.lower()


class TestKeywordSets:
    def test_bali_keywords_not_empty(self) -> None:
        assert len(_BALI_KEYWORDS) > 0

    def test_tourism_keywords_not_empty(self) -> None:
        assert len(_TOURISM_KEYWORDS) > 0

    def test_bali_contains_expected_keywords(self) -> None:
        assert "bali" in _BALI_KEYWORDS
        assert "ubud" in _BALI_KEYWORDS
        assert "kuta" in _BALI_KEYWORDS

    def test_tourism_contains_expected_keywords(self) -> None:
        assert "hotel" in _TOURISM_KEYWORDS
        assert "beach" in _TOURISM_KEYWORDS
        assert "temple" in _TOURISM_KEYWORDS
