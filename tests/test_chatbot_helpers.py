# tests/test_chatbot_helpers.py

"""Tests for chatbot service helper functions."""

import pytest

from openai_client.services.chatbot import (
    _BALI_KEYWORDS,
    _SKIP_PATTERNS,
    _TOURISM_KEYWORDS,
    _build_search_context_message,
    _is_tourism_query,
    _system_message,
)


class TestIsTourismQuery:
    def test_bali_keyword_detected(self) -> None:
        assert _is_tourism_query("What are the best beaches in Bali?") is True

    def test_tourism_keyword_detected(self) -> None:
        assert _is_tourism_query("Can you recommend a good hotel?") is True

    def test_greeting_not_tourism(self) -> None:
        assert _is_tourism_query("hi") is False

    def test_short_input_not_tourism(self) -> None:
        assert _is_tourism_query("hello there") is False

    @pytest.mark.parametrize("pattern", _SKIP_PATTERNS)
    def test_skip_patterns(self, pattern: str) -> None:
        assert _is_tourism_query(pattern) is False

    def test_empty_input(self) -> None:
        assert _is_tourism_query("") is False

    def test_whitespace_only(self) -> None:
        assert _is_tourism_query("   ") is False

    def test_specific_bali_location(self) -> None:
        assert _is_tourism_query("Tell me about Ubud rice terraces") is True

    def test_food_related(self) -> None:
        assert _is_tourism_query("Where can I try nasi goreng?") is True


class TestBuildSearchContextMessage:
    def test_returns_system_message(self) -> None:
        results = "Search result about Bali"
        msg = _build_search_context_message(results)
        assert msg["role"] == "system"
        assert isinstance(msg["content"], str)
        assert "search_context" in msg["content"]
        assert results in msg["content"]

    def test_contains_persona_reminder(self) -> None:
        msg = _build_search_context_message("test results")
        assert isinstance(msg["content"], str)
        assert "BaliBlissed Concierge" in msg["content"]


class TestSystemMessage:
    def test_returns_openai_chat_message(self) -> None:
        msg = _system_message()
        assert msg.role == "system"
        assert "Made" in msg.content
        assert "BaliBlissed" in msg.content

    def test_contains_personality(self) -> None:
        msg = _system_message()
        assert "personality" in msg.content.lower()

    def test_contains_boundaries(self) -> None:
        msg = _system_message()
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
