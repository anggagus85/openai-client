# tests/test_itinerary_advanced.py

"""Advanced tests for itinerary service - itinerary(), _generate_and_save()."""

import tempfile
from asyncio import run as async_run
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai_client.schemas.ai import ItineraryRequestMD
from openai_client.services.itinerary import (
    _generate_and_save,
    itinerary,
)
from openai_client.types import ListParams


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.model = "test-model"
    client.base_url = "https://api.example.com"
    return client


@pytest.fixture
def itinerary_request() -> ItineraryRequestMD:
    return ItineraryRequestMD(
        duration=5,
        interests=["beaches"],
        budget="US$ 1000",
    )


@contextmanager
def mock_itinerary_caller(content: str) -> Generator[MagicMock]:
    mock_caller = AsyncMock(return_value=content)
    with patch(
        "openai_client.services.itinerary.caller",
        return_value={"completion": mock_caller},
    ) as mock:
        yield mock


@contextmanager
def mock_web_search(results: str = "") -> Generator[MagicMock]:
    mock = MagicMock()
    call_tracker = MagicMock()

    async def _side_effect(*_args: object, **_kwargs: object) -> str | None:
        call_tracker(*_args, **_kwargs)
        return results if results else None

    mock_search = AsyncMock(side_effect=_side_effect)
    mock.search = mock_search
    mock.call_tracker = call_tracker
    with patch("openai_client.services.itinerary.WebSearch", return_value=mock):
        yield mock


@contextmanager
def temp_cwd() -> Generator[Path]:
    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = Path(tmpdir)
        with patch("pathlib.Path.cwd", return_value=cwd):
            yield cwd


class TestGenerateAndSave:
    def test_saves_markdown_and_text(self, mock_client: MagicMock) -> None:
        messages: ListParams = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "Test Itinerary\n\nDay 1: Beach",
            },
        ]
        with temp_cwd(), mock_itinerary_caller("# Test Itinerary\n\nDay 1: Beach"):
            async_run(_generate_and_save(messages, mock_client, "completion"))

            md_path = Path.cwd() / "itinerary.md"
            txt_path = Path.cwd() / "itinerary.txt"
            assert md_path.exists()
            assert txt_path.exists()
            assert "# Test Itinerary" in md_path.read_text()


class TestItinerary:
    def test_uses_search_when_available(
        self,
        itinerary_request: ItineraryRequestMD,
        mock_client: MagicMock,
    ) -> None:
        with (
            mock_web_search(results="Search results") as mock_search,
            mock_itinerary_caller("# Generated Itinerary"),
            temp_cwd(),
        ):
            async_run(itinerary(itinerary_request, mock_client, "completion"))

        mock_search.call_tracker.assert_called_once()

    def test_falls_back_when_search_fails(
        self,
        itinerary_request: ItineraryRequestMD,
        mock_client: MagicMock,
    ) -> None:
        with (
            mock_web_search(results=""),
            mock_itinerary_caller("# Fallback Itinerary"),
            temp_cwd() as cwd,
        ):
            async_run(itinerary(itinerary_request, mock_client, "completion"))

            md_path = cwd / "itinerary.md"
            txt_path = cwd / "itinerary.txt"
            assert md_path.exists(), "Fallback path should have created itinerary.md"
            assert txt_path.exists(), "Fallback path should have created itinerary.txt"
            assert "# Fallback Itinerary" in md_path.read_text()

    def test_returns_early_on_empty_content(
        self,
        itinerary_request: ItineraryRequestMD,
        mock_client: MagicMock,
    ) -> None:
        with (
            mock_web_search(results="Search results"),
            mock_itinerary_caller(""),
            temp_cwd() as cwd,
        ):
            async_run(itinerary(itinerary_request, mock_client, "completion"))

            md_path = cwd / "itinerary.md"
            assert not md_path.exists()
