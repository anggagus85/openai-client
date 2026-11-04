# tests/test_itinerary_advanced.py

"""Advanced tests for production itinerary generation and file output."""

import tempfile
from asyncio import run as async_run
from collections.abc import Generator
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anyio import Path as AnyPath

from openai_client.configs import settings
from openai_client.schemas.ai import ItineraryRequest
from openai_client.services.itinerary import Itinerary


def _dates(count: int) -> list[str]:
    start = date(2026, 11, 10)
    return [(start + timedelta(days=i)).isoformat() for i in range(count)]


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.is_mistral = False
    client.generate = AsyncMock(return_value="# Test Itinerary\n\nDay 1: Beach")
    return client


@pytest.fixture
def itinerary_request() -> ItineraryRequest:
    return ItineraryRequest(
        duration=5,
        dates=_dates(5),
        interests=["beaches"],
        budget="US$ 1000",
    )


@contextmanager
def temp_cwd() -> Generator[Path]:
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("anyio.Path.cwd", new=AsyncMock(return_value=AnyPath(tmpdir))),
    ):
        yield Path(tmpdir)


class TestGenerateAndSave:
    def test_saves_markdown_and_text_in_debug(
        self,
        mock_client: MagicMock,
        itinerary_request: ItineraryRequest,
    ) -> None:
        instance = Itinerary(mock_client, itinerary_request)

        with temp_cwd() as cwd, patch.object(settings, "DEBUG", new=True):
            async_run(instance.generate("Prompt"))

            md_path = cwd / "itinerary.md"
            txt_path = cwd / "itinerary.txt"
            assert md_path.exists()
            assert txt_path.exists()
            assert "# Test Itinerary" in md_path.read_text()

    def test_does_not_save_without_debug(
        self,
        mock_client: MagicMock,
        itinerary_request: ItineraryRequest,
    ) -> None:
        instance = Itinerary(mock_client, itinerary_request)

        with temp_cwd() as cwd, patch.object(settings, "DEBUG", new=False):
            async_run(instance.generate("Prompt"))
            assert not (cwd / "itinerary.md").exists()


class TestSearchFallback:
    def test_uses_search_when_available(
        self,
        mock_client: MagicMock,
        itinerary_request: ItineraryRequest,
    ) -> None:
        instance = Itinerary(mock_client, itinerary_request)

        with (
            patch(
                "openai_client.services.itinerary.search_bali_context",
                new=AsyncMock(return_value="Search results"),
            ) as search,
            temp_cwd(),
            patch.object(settings, "DEBUG", new=False),
        ):
            async_run(instance.run())

        search.assert_awaited_once()

    def test_falls_back_when_search_fails(
        self,
        mock_client: MagicMock,
        itinerary_request: ItineraryRequest,
    ) -> None:
        instance = Itinerary(mock_client, itinerary_request)

        with (
            patch(
                "openai_client.services.itinerary.search_bali_context",
                new=AsyncMock(return_value=None),
            ),
            temp_cwd(),
            patch.object(settings, "DEBUG", new=False),
        ):
            response = async_run(instance.run())

        assert response.markdown.startswith("# Test Itinerary")
