# tests/test_search.py

"""Tests for search service clients and helpers."""

from unittest.mock import MagicMock, patch

import pytest

from openai_client.services.search import (
    BraveSearchClient,
    ExaSearchClient,
    SearchClientError,
    providers,
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
        client = ExaSearchClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.base_url == "https://api.exa.ai/search"

    def test_init_custom_url(self) -> None:
        client = ExaSearchClient(api_key="test-key", base_url="https://custom.exa.ai")
        assert client.base_url == "https://custom.exa.ai"

    def test_str(self) -> None:
        client = ExaSearchClient(api_key="test-key")
        assert "exa" in str(client)

    def test_search_success(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": "Bali Beach", "url": "https://example.com", "text": "Beautiful beach"},
            ],
        }
        mock_response.raise_for_status.return_value = None
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch("openai_client.services.search.Client", return_value=mock_client):
            success, result = ExaSearchClient(api_key="key").search("Bali beaches")
            assert success is True
            assert "Bali Beach" in result

    def test_search_failure(self) -> None:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = ConnectionError("Network error")

        with patch("openai_client.services.search.Client", return_value=mock_client):
            success, result = ExaSearchClient(api_key="key").search("Bali")
            assert success is False
            assert "Search error" in result

    def test_process_results(self) -> None:
        client = ExaSearchClient(api_key="key")
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
        client = ExaSearchClient(api_key="key")
        result = client._process_results({"results": []})
        assert result == ""


class TestBraveSearchClient:
    def test_init_defaults(self) -> None:
        client = BraveSearchClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.base_url == "https://api.search.brave.com/res/v1/web/search"

    def test_str(self) -> None:
        client = BraveSearchClient(api_key="test-key")
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
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response

        with patch("openai_client.services.search.Client", return_value=mock_client):
            success, result = BraveSearchClient(api_key="key").search("Bali travel")
            assert success is True
            assert "Bali Guide" in result

    def test_search_failure(self) -> None:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = ConnectionError("Network error")

        with patch("openai_client.services.search.Client", return_value=mock_client):
            success, result = BraveSearchClient(api_key="key").search("Bali")
            assert success is False
            assert "Search error" in result


class TestProviders:
    def test_returns_list(self) -> None:
        result = providers()
        assert isinstance(result, list)
        assert len(result) > 0
        for provider in result:
            assert hasattr(provider, "api_key")
            assert callable(getattr(provider, "search", None))
