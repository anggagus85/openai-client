# tests/test_itinerary_helpers.py

"""Tests for production itinerary prompts and service wiring."""

from asyncio import run as async_run
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai_client.configs import settings
from openai_client.errors import ItineraryGenerationError
from openai_client.schemas.ai import ItineraryRequest
from openai_client.services.itinerary import Itinerary
from openai_client.services.prompts import (
    itinerary_prompt,
    search_bali_info,
    search_bali_queries,
)
from openai_client.typings import SystemParam
from openai_client.utils import clean_markdown


def _dates(count: int) -> list[str]:
    start = date(2026, 11, 10)
    return [(start + timedelta(days=i)).isoformat() for i in range(count)]


def _request(duration: int = 5, budget: str = "US$ 1000") -> ItineraryRequest:
    return ItineraryRequest(
        duration=duration,
        dates=_dates(duration),
        interests=["beaches", "temples"],
        budget=budget,
    )


def _client(answer: str | None = "# Generated Itinerary") -> MagicMock:
    client = MagicMock()
    client.is_mistral = False
    client.generate = AsyncMock(return_value=answer)
    return client


class TestItineraryPrompt:
    def test_contains_traveler_profile(self) -> None:
        result = itinerary_prompt(_request())
        assert "Interests: beaches, temples" in result
        assert "Duration: 5" in result
        assert "Budget: US$ 1000" in result

    def test_contains_dates(self) -> None:
        result = itinerary_prompt(_request())
        assert "2026-11-10" in result

    def test_contains_search_context_when_provided(self) -> None:
        result = itinerary_prompt(_request(3), search_context="Bali food guide 2026")
        assert "Bali food guide 2026" in result

    def test_contains_writing_rules(self) -> None:
        result = itinerary_prompt(_request())
        assert "<writing_rules>" in result
        assert "humans read these" in result

    def test_contains_structure(self) -> None:
        result = itinerary_prompt(_request())
        assert "Here's your 5-Days Bali trip Itinerary" in result


class TestSearchBaliInfo:
    def test_contains_interests_and_budget(self) -> None:
        result = search_bali_info(_request(7, "US$ 1500"))
        assert "beaches, temples" in result
        assert "7 day" in result
        assert "US$ 1500" in result

    def test_contains_travel_month(self) -> None:
        result = search_bali_info(_request())
        assert "11" in result


class TestSearchBaliQueries:
    def test_returns_six_queries(self) -> None:
        assert len(search_bali_queries(_request())) == 6

    def test_queries_fit_deepseek_limit(self) -> None:
        for query in search_bali_queries(_request()):
            assert len(query) <= 128


class TestItineraryGenerate:
    def test_returns_response(self) -> None:
        client = _client()
        response = async_run(
            Itinerary(client, _request(), SystemParam(role="system", content="system")).generate(
                "Day 1: Beach",
            ),
        )

        assert response.markdown
        assert response.whatsapp
        assert client.generate.await_count == 1

    def test_builds_system_and_user_messages(self) -> None:
        client = _client()
        async_run(Itinerary(client, _request()).generate("Prompt content"))

        configs = client.generate.call_args.args[0]
        messages = configs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Prompt content"

    def test_raises_on_empty_content(self) -> None:
        client = _client(None)
        with pytest.raises(ItineraryGenerationError):
            async_run(Itinerary(client, _request()).generate("Prompt content"))


class TestItineraryRun:
    def test_uses_search_when_available(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        client = _client()
        with (
            patch(
                "openai_client.services.itinerary.search_bali_context",
                new=AsyncMock(return_value="Bali search results"),
            ) as search,
            patch.object(settings, "DEBUG", new=False),
        ):
            response = async_run(Itinerary(client, _request()).run())

        assert search.await_count == 1
        assert response.markdown
        user_content = client.generate.call_args.args[0]["messages"][1]["content"]
        assert "Bali search results" in user_content

    def test_falls_back_when_search_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        client = _client()
        with (
            patch(
                "openai_client.services.itinerary.search_bali_context",
                new=AsyncMock(return_value=None),
            ),
            patch.object(settings, "DEBUG", new=False),
        ):
            response = async_run(Itinerary(client, _request()).run())

        assert response.markdown
        user_content = client.generate.call_args.args[0]["messages"][1]["content"]
        assert "latest_web_search_results" not in user_content

    def test_rejects_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            async_run(Itinerary(_client(), _request()).run("bad"))


class TestCleanMarkdown:
    def test_returns_formatted_text(self) -> None:
        assert "# Hello" in clean_markdown("# Hello\n\n**World**")

    def test_fallback_on_error(self) -> None:
        with patch("openai_client.utils.helpers.mdformat_text", side_effect=Exception("fail")):
            assert clean_markdown("raw text") == "raw text"
