# tests/test_itinerary_helpers.py

"""Tests for itinerary service helper functions."""

from datetime import datetime
from unittest.mock import MagicMock, patch

from openai_client.schemas.ai import ItineraryRequestMD
from openai_client.services.itinerary import Itinerary
from openai_client.typings import SystemParam, UserParam
from openai_client.utils import clean_markdown


class TestTopics:
    def test_contains_daily_breakdown(self) -> None:
        req = ItineraryRequestMD(duration=5, interests=["beaches"], budget="US$ 1000")
        instance = Itinerary.__new__(Itinerary)
        instance._itinerary_req = req
        result = instance._topics()
        assert "DAILY BREAKDOWN" in result

    def test_contains_duration_days(self) -> None:
        req = ItineraryRequestMD(duration=7, interests=["beaches"], budget="US$ 1000")
        instance = Itinerary.__new__(Itinerary)
        instance._itinerary_req = req
        result = instance._topics()
        assert "7 full days" in result

    def test_contains_accommodations(self) -> None:
        req = ItineraryRequestMD(duration=3, interests=["beaches"], budget="US$ 1000")
        instance = Itinerary.__new__(Itinerary)
        instance._itinerary_req = req
        result = instance._topics()
        assert "HANDPICKED ACCOMMODATIONS" in result

    def test_contains_culinary(self) -> None:
        req = ItineraryRequestMD(duration=5, interests=["beaches"], budget="US$ 1000")
        instance = Itinerary.__new__(Itinerary)
        instance._itinerary_req = req
        result = instance._topics()
        assert "CULINARY EXPERIENCES" in result

    def test_contains_budget_breakdown(self) -> None:
        req = ItineraryRequestMD(duration=5, interests=["beaches"], budget="US$ 1000")
        instance = Itinerary.__new__(Itinerary)
        instance._itinerary_req = req
        result = instance._topics()
        assert "COMPREHENSIVE BUDGET BREAKDOWN" in result


class TestStructure:
    def test_contains_header(self) -> None:
        req = ItineraryRequestMD(duration=5, interests=["beaches"], budget="US$ 1000")
        instance = Itinerary.__new__(Itinerary)
        instance._itinerary_req = req
        result = instance._structure()
        assert "Here's your 5-Days Bali trip Itinerary" in result

    def test_contains_friendly_note(self) -> None:
        req = ItineraryRequestMD(duration=7, interests=["beaches"], budget="US$ 1500")
        instance = Itinerary.__new__(Itinerary)
        instance._itinerary_req = req
        result = instance._structure()
        assert "A Friendly Note" in result

    def test_contains_sections(self) -> None:
        req = ItineraryRequestMD(duration=5, interests=["beaches"], budget="US$ 1000")
        instance = Itinerary.__new__(Itinerary)
        instance._itinerary_req = req
        result = instance._structure()
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
        instance = Itinerary.__new__(Itinerary)
        instance._itinerary_req = req
        result = instance._prompt()
        assert "Interests: beaches, temples" in result
        assert "Duration: 5" in result
        assert "Budget: US$ 1000" in result

    def test_contains_search_context_when_provided(self) -> None:
        req = ItineraryRequestMD(
            duration=3,
            interests=["food"],
            budget="US$ 500",
        )
        instance = Itinerary.__new__(Itinerary)
        instance._itinerary_req = req
        result = instance._prompt(search_context="Bali food guide 2024")
        assert "Bali food guide 2024" in result

    def test_no_search_context_when_empty(self) -> None:
        req = ItineraryRequestMD(
            duration=3,
            interests=["food"],
            budget="US$ 500",
        )
        instance = Itinerary.__new__(Itinerary)
        instance._itinerary_req = req
        result = instance._prompt()
        assert "search_context" not in result


class TestCleanMarkdown:
    def test_returns_formatted_text(self) -> None:
        result = clean_markdown("# Hello\n\n**World**")
        assert "# Hello" in result

    def test_fallback_on_error(self) -> None:
        with patch("openai_client.utils.helpers.mdformat_text", side_effect=Exception("fail")):
            result = clean_markdown("raw text")
            assert result == "raw text"


class TestItineraryMessages:
    def test_returns_two_messages(self) -> None:
        req = ItineraryRequestMD(
            duration=5,
            interests=["beaches"],
            budget="US$ 1000",
        )
        mock_client = MagicMock()
        mock_client.messages = []
        instance = Itinerary.__new__(Itinerary)
        instance._itinerary_req = req
        instance._client = mock_client
        instance.messages = [SystemParam(role="system", content="system")]
        instance._create_user_message = lambda content: UserParam(role="user", content=content)
        instance.messages.append(instance._create_user_message(instance._prompt()))
        assert len(instance.messages) == 2
        assert instance.messages[0]["role"] == "system"
        assert instance.messages[1]["role"] == "user"


class TestItinerarySearchMessages:
    def test_returns_user_param_with_search_context(self) -> None:
        req = ItineraryRequestMD(
            duration=5,
            interests=["beaches"],
            budget="US$ 1000",
        )
        mock_client = MagicMock()
        mock_client.messages = []
        instance = Itinerary.__new__(Itinerary)
        instance._itinerary_req = req
        instance._client = mock_client
        user_param = instance._itinerary_search_messages("Search results here")
        assert user_param["role"] == "user"
        assert isinstance(user_param["content"], str)
        assert "Search results here" in user_param["content"]


class TestSearchBaliInfo:
    def test_contains_interests(self) -> None:
        req = ItineraryRequestMD(
            duration=7,
            interests=["temples", "food"],
            budget="US$ 1500",
        )
        instance = Itinerary.__new__(Itinerary)
        instance._itinerary_req = req
        result = instance._search_bali_info()
        assert "temples" in result
        assert "food" in result

    def test_contains_duration(self) -> None:
        req = ItineraryRequestMD(
            duration=10,
            interests=["surfing"],
            budget="US$ 2000",
        )
        instance = Itinerary.__new__(Itinerary)
        instance._itinerary_req = req
        result = instance._search_bali_info()
        assert "10 day" in result

    def test_contains_current_year(self) -> None:
        req = ItineraryRequestMD(
            duration=5,
            interests=["beaches"],
            budget="US$ 1000",
        )
        instance = Itinerary.__new__(Itinerary)
        instance._itinerary_req = req
        result = instance._search_bali_info()
        assert str(datetime.now().year) in result
