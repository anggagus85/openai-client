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
from openai_client.services.itinerary import Itinerary
from openai_client.types import SystemParam


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.model = "test-model"
    client.base_url = "https://api.example.com"
    return client


@pytest.fixture
def system_param() -> SystemParam:
    return SystemParam(role="system", content="test system")


@pytest.fixture
def itinerary_request() -> ItineraryRequestMD:
    return ItineraryRequestMD(
        duration=5,
        interests=["beaches"],
        budget="US$ 1000",
    )


@contextmanager
def mock_itinerary_caller() -> Generator[None]:
    with patch.object(Itinerary, "_generate_and_save", new_callable=AsyncMock) as mock:
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
        instance = Itinerary.__new__(Itinerary)
        instance._client = mock_client
        instance._client.messages = []

        with temp_cwd() as cwd:
            mock_client.call.return_value = {
                "completion": AsyncMock(return_value="# Test Itinerary\n\nDay 1: Beach"),
            }
            async_run(instance._generate_and_save("completion"))

            md_path = cwd / "itinerary.md"
            txt_path = cwd / "itinerary.txt"
            assert md_path.exists()
            assert txt_path.exists()
            assert "# Test Itinerary" in md_path.read_text()


class TestItinerary:
    def test_uses_search_when_available(
        self,
        itinerary_request: ItineraryRequestMD,
        mock_client: MagicMock,
        system_param: SystemParam,
    ) -> None:
        instance = Itinerary(mock_client, itinerary_request, system_param)

        with (
            mock_web_search(results="Search results") as mock_search,
            temp_cwd(),
        ):
            mock_client.call.return_value = {
                "completion": AsyncMock(return_value="# Generated Itinerary"),
            }
            async_run(instance.generate("completion"))

        mock_search.call_tracker.assert_called_once()

    def test_falls_back_when_search_fails(
        self,
        itinerary_request: ItineraryRequestMD,
        mock_client: MagicMock,
        system_param: SystemParam,
    ) -> None:
        instance = Itinerary(mock_client, itinerary_request, system_param)

        with (
            mock_web_search(results=""),
            temp_cwd() as cwd,
        ):
            mock_client.call.return_value = {
                "completion": AsyncMock(return_value="# Fallback Itinerary"),
            }
            async_run(instance.generate("completion"))

            md_path = cwd / "itinerary.md"
            txt_path = cwd / "itinerary.txt"
            assert md_path.exists(), "Fallback path should have created itinerary.md"
            assert txt_path.exists(), "Fallback path should have created itinerary.txt"
            assert "# Fallback Itinerary" in md_path.read_text()

    def test_returns_early_on_empty_content(
        self,
        itinerary_request: ItineraryRequestMD,
        mock_client: MagicMock,
        system_param: SystemParam,
    ) -> None:
        instance = Itinerary(mock_client, itinerary_request, system_param)

        with (
            mock_web_search(results="Search results"),
            temp_cwd() as cwd,
        ):
            mock_client.call.return_value = {"completion": AsyncMock(return_value="")}
            async_run(instance.generate("completion"))

            md_path = cwd / "itinerary.md"
            assert not md_path.exists()
