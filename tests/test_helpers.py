# tests/test_helpers.py

"""Tests for utility helper functions."""

import tempfile
from unittest.mock import AsyncMock, patch

import pytest
from anyio import Path

from openai_client.utils.helpers import save_itinerary_to_file, save_to_file


class TestSaveToFile:
    async def test_saves_string_to_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            path = Path(f.name)
        try:
            await save_to_file("Hello, World!", path)
            assert await path.read_text(encoding="utf-8") == "Hello, World!"
        finally:
            await path.unlink(missing_ok=True)

    async def test_saves_utf8_content(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            path = Path(f.name)
        try:
            await save_to_file("Hello 🌴 Bali", path)
            assert await path.read_text(encoding="utf-8") == "Hello 🌴 Bali"
        finally:
            await path.unlink(missing_ok=True)

    async def test_overwrites_existing_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Old content")
            path = Path(f.name)
        try:
            await save_to_file("New content", path)
            assert await path.read_text(encoding="utf-8") == "New content"
        finally:
            await path.unlink(missing_ok=True)

    async def test_raises_on_invalid_path(self) -> None:
        invalid_path = Path("/nonexistent/directory/file.txt")
        with pytest.raises(FileNotFoundError):
            await save_to_file("test", invalid_path)


class TestSaveItineraryToFile:
    async def test_logs_and_propagates_write_errors(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with (
            patch(
                "openai_client.utils.helpers.save_to_file",
                new=AsyncMock(side_effect=OSError("disk full")),
            ),
            caplog.at_level("ERROR"),
            pytest.raises(OSError, match="disk full"),
        ):
            await save_itinerary_to_file("markdown", "whatsapp")

        assert "Failed to save itinerary files" in caplog.text
