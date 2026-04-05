# tests/test_itinerary_helpers.py

"""Tests for itinerary service helper functions."""

from datetime import datetime
from unittest.mock import patch

from openai_client.schemas.ai import ItineraryRequestMD
from openai_client.services.itinerary import (
    _clean_markdown,
    _itinerary_messages,
    _itinerary_search_messages,
    _prompt,
    _search_bali_info,
    _structure,
    _topics,
)


class TestTopics:
    def test_contains_daily_breakdown(self) -> None:
        result = _topics(5)
        assert "DAILY BREAKDOWN" in result

    def test_contains_duration_days(self) -> None:
        result = _topics(7)
        assert "7 full days" in result

    def test_contains_accommodations(self) -> None:
        result = _topics(3)
        assert "HANDPICKED ACCOMMODATIONS" in result

    def test_contains_culinary(self) -> None:
        result = _topics(5)
        assert "CULINARY EXPERIENCES" in result

    def test_contains_budget_breakdown(self) -> None:
        result = _topics(5)
        assert "COMPREHENSIVE BUDGET BREAKDOWN" in result


class TestStructure:
    def test_contains_header(self) -> None:
        result = _structure(5, "US$ 1000")
        assert "Here's your 5-Days Bali trip Itinerary" in result

    def test_contains_friendly_note(self) -> None:
        result = _structure(7, "US$ 1500")
        assert "A Friendly Note" in result

    def test_contains_sections(self) -> None:
        result = _structure(5, "US$ 1000")
        assert "DAILY BREAKDOWN" in result
        assert "CULTURAL IMMERSION" in result
        assert "AUTHENTIC EXPERIENCES" in result


class TestPrompt:
    def test_contains_traveler_profile(self) -> None:
        req = ItineraryRequestMD(
            duration=5,
            interests=["beaches", "temples"],
            budget="US$ 1000",
        )
        result = _prompt(req)
        assert "Interests: beaches, temples" in result
        assert "Duration: 5" in result
        assert "Budget: US$ 1000" in result

    def test_contains_search_context_when_provided(self) -> None:
        req = ItineraryRequestMD(
            duration=3,
            interests=["food"],
            budget="US$ 500",
        )
        result = _prompt(req, search_context="Bali food guide 2024")
        assert "Bali food guide 2024" in result

    def test_no_search_context_when_empty(self) -> None:
        req = ItineraryRequestMD(
            duration=3,
            interests=["food"],
            budget="US$ 500",
        )
        result = _prompt(req)
        assert "search_context" not in result


class TestCleanMarkdown:
    def test_returns_formatted_text(self) -> None:
        result = _clean_markdown("# Hello\n\n**World**")
        assert "# Hello" in result

    def test_fallback_on_error(self) -> None:
        with patch("openai_client.services.itinerary.mdformat_text", side_effect=Exception("fail")):
            result = _clean_markdown("raw text")
            assert result == "raw text"


class TestItineraryMessages:
    def test_returns_two_messages(self) -> None:
        req = ItineraryRequestMD(
            duration=5,
            interests=["beaches"],
            budget="US$ 1000",
        )
        messages = _itinerary_messages(req)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


class TestItinerarySearchMessages:
    def test_returns_two_messages_with_search_context(self) -> None:
        req = ItineraryRequestMD(
            duration=5,
            interests=["beaches"],
            budget="US$ 1000",
        )
        messages = _itinerary_search_messages(req, "Search results here")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert isinstance(messages[1]["content"], str)
        assert "Search results here" in messages[1]["content"]


class TestSearchBaliInfo:
    def test_contains_interests(self) -> None:
        req = ItineraryRequestMD(
            duration=7,
            interests=["temples", "food"],
            budget="US$ 1500",
        )
        result = _search_bali_info(req)
        assert "temples" in result
        assert "food" in result

    def test_contains_duration(self) -> None:
        req = ItineraryRequestMD(
            duration=10,
            interests=["surfing"],
            budget="US$ 2000",
        )
        result = _search_bali_info(req)
        assert "10 day" in result

    def test_contains_current_year(self) -> None:

        req = ItineraryRequestMD(
            duration=5,
            interests=["beaches"],
            budget="US$ 1000",
        )
        result = _search_bali_info(req)
        assert str(datetime.now().year) in result
