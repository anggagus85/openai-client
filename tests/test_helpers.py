# tests/test_helpers.py

"""Tests for utility helper functions."""

import tempfile
from pathlib import Path

import pytest

from openai_client.utils.helpers import save_to_file


class TestSaveToFile:
    def test_saves_string_to_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            path = Path(f.name)
        try:
            save_to_file("Hello, World!", path)
            assert path.read_text() == "Hello, World!"
        finally:
            path.unlink(missing_ok=True)

    def test_saves_utf8_content(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            path = Path(f.name)
        try:
            save_to_file("Hello 🌴 Bali", path)
            assert path.read_text(encoding="utf-8") == "Hello 🌴 Bali"
        finally:
            path.unlink(missing_ok=True)

    def test_overwrites_existing_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Old content")
            path = Path(f.name)
        try:
            save_to_file("New content", path)
            assert path.read_text() == "New content"
        finally:
            path.unlink(missing_ok=True)

    def test_raises_on_invalid_path(self) -> None:
        invalid_path = Path("/nonexistent/directory/file.txt")
        with pytest.raises(FileNotFoundError):
            save_to_file("test", invalid_path)
