# tests/test_search.py

"""Tests for search service clients and helpers."""

from asyncio import run as async_run
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai_client.services.web_search import (
    SearchClientError,
    _BraveSearch,
    _ExaSearch,
    _providers,
)


class TestSearchClientError:
    def test_inherits_from_exception(self) -> None:
        assert issubclass(SearchClientError, Exception)

    def test_can_be_raised(self) -> None:
        def _client_error() -> None:
            details = "Search failed"
            raise SearchClientError(details)

        with pytest.raises(SearchClientError) as context:
            _client_error()
        assert str(context.value) == "Search failed"


class TestExaSearchClient:
    def test_init_defaults(self) -> None:
        client = _ExaSearch(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.base_url == "https://api.exa.ai/search"

    def test_init_custom_url(self) -> None:
        client = _ExaSearch(api_key="test-key")
        client.base_url = "https://custom.exa.ai"
        assert client.base_url == "https://custom.exa.ai"

    def test_str(self) -> None:
        client = _ExaSearch(api_key="test-key")
        assert "exa" in str(client)

    async def test_search_success(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": "Bali Beach", "url": "https://example.com", "text": "Beautiful beach"},
            ],
        }
        mock_response.raise_for_status.return_value = None
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)

    def test_search_failure(self) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=ConnectionError("Network error"))

        async def _run_test() -> None:
            with patch("openai_client.services.web_search.AsyncClient", return_value=mock_client):
                success, result = await _ExaSearch(api_key="key").search("Bali")
                assert success is False
                assert "Search error" in result

        async_run(_run_test())

    def test_process_results(self) -> None:
        client = _ExaSearch(api_key="key")
        data = {
            "results": [
                {"title": "Place 1", "url": "https://a.com", "text": "Details about place 1"},
                {"title": "Place 2", "url": "https://b.com", "text": "Details about place 2"},
            ],
        }
        result = client._process_results(data)
        assert "1. **Place 1**" in result
        assert "2. **Place 2**" in result
        assert "https://a.com" in result

    def test_process_results_empty(self) -> None:
        client = _ExaSearch(api_key="key")
        result = client._process_results({"results": []})
        assert result == ""


class TestBraveSearchClient:
    def test_init_defaults(self) -> None:
        client = _BraveSearch(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.base_url == "https://api.search.brave.com/res/v1/web/search"

    def test_str(self) -> None:
        client = _BraveSearch(api_key="test-key")
        assert "brave" in str(client)

    def test_search_success(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {"title": "Bali Guide", "url": "https://example.com", "description": "A guide"},
                ],
            },
        }
        mock_response.raise_for_status.return_value = None
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        async def _run_test() -> None:
            with patch("openai_client.services.web_search.AsyncClient", return_value=mock_client):
                success, result = await _BraveSearch(api_key="key").search("Bali travel")
                assert success is True
                assert "Bali Guide" in result

        async_run(_run_test())

    def test_search_failure(self) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=ConnectionError("Network error"))

        async def _run_test() -> None:
            with patch("openai_client.services.web_search.AsyncClient", return_value=mock_client):
                success, result = await _BraveSearch(api_key="key").search("Bali")
                assert success is False
                assert "Search error" in result

        async_run(_run_test())


class TestProviders:
    def test_returns_list(self) -> None:
        result = _providers()
        assert isinstance(result, list)
        assert len(result) > 0
        for provider in result:
            assert hasattr(provider, "api_key")
            assert callable(getattr(provider, "search", None))
