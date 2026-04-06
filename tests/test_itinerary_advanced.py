# tests/test_itinerary_advanced.py

"""Advanced tests for itinerary service - itinerary(), _generate_and_save()."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from openai_client.schemas.ai import ItineraryRequestMD
from openai_client.services.itinerary import (
    _generate_and_save,
    itinerary,
)
from openai_client.types import ListParams


class TestGenerateAndSave:
    def test_saves_markdown_and_text(self) -> None:
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"
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
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("pathlib.Path.cwd", return_value=Path(tmpdir)),
            patch(
                "openai_client.services.itinerary.caller",
                return_value={"completion": lambda: "# Test Itinerary\n\nDay 1: Beach"},
            ),
        ):
            _generate_and_save(messages, mock_client, "completion")

            md_path = Path(tmpdir) / "itinerary.md"
            txt_path = Path(tmpdir) / "itinerary.txt"
            assert md_path.exists()
            assert txt_path.exists()
            assert "# Test Itinerary" in md_path.read_text()


class TestItinerary:
    def test_uses_search_when_available(self) -> None:
        req = ItineraryRequestMD(
            duration=5,
            interests=["beaches"],
            budget="US$ 1000",
        )
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"

        mock_provider = MagicMock()
        mock_provider.search.return_value = (True, "Search results")

        with (
            patch(
                "openai_client.services.itinerary.providers",
                return_value=[mock_provider],
            ),
            patch(
                "openai_client.services.itinerary.caller",
                return_value={"completion": lambda: "# Generated Itinerary"},
            ),
            tempfile.TemporaryDirectory() as tmpdir,
            patch("pathlib.Path.cwd", return_value=Path(tmpdir)),
        ):
            itinerary(req, mock_client, "completion")

        mock_provider.search.assert_called_once()

    def test_falls_back_when_search_fails(self) -> None:
        req = ItineraryRequestMD(
            duration=5,
            interests=["beaches"],
            budget="US$ 1000",
        )
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"

        mock_provider = MagicMock()
        mock_provider.search.return_value = (False, "Error")

        with (
            patch(
                "openai_client.services.itinerary.providers",
                return_value=[mock_provider],
            ),
            patch(
                "openai_client.services.itinerary.caller",
                return_value={"completion": lambda: "# Fallback Itinerary"},
            ),
            tempfile.TemporaryDirectory() as tmpdir,
            patch("pathlib.Path.cwd", return_value=Path(tmpdir)),
        ):
            itinerary(req, mock_client, "completion")

            md_path = Path(tmpdir) / "itinerary.md"
            txt_path = Path(tmpdir) / "itinerary.txt"
            assert md_path.exists(), "Fallback path should have created itinerary.md"
            assert txt_path.exists(), "Fallback path should have created itinerary.txt"
            assert "# Fallback Itinerary" in md_path.read_text()

    def test_returns_early_on_empty_content(self) -> None:
        req = ItineraryRequestMD(
            duration=5,
            interests=["beaches"],
            budget="US$ 1000",
        )
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"

        mock_provider = MagicMock()
        mock_provider.search.return_value = (True, "Search results")

        with (
            patch(
                "openai_client.services.itinerary.providers",
                return_value=[mock_provider],
            ),
            patch(
                "openai_client.services.itinerary.caller",
                return_value={"completion": lambda: ""},
            ),
            tempfile.TemporaryDirectory() as tmpdir,
            patch("pathlib.Path.cwd", return_value=Path(tmpdir)),
        ):
            itinerary(req, mock_client, "completion")

            md_path = Path(tmpdir) / "itinerary.md"
            assert not md_path.exists()
