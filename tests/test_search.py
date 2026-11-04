# tests/test_search.py

"""Tests for the production web search providers."""

from asyncio import run as async_run
from typing import Self
from unittest.mock import AsyncMock, MagicMock, patch

from openai_client.configs import settings
from openai_client.services.web_search import (
    SearchPlan,
    WebSearch,
    _BraveSearch,
    _DeepseekSearch,
    _ExaSearch,
    _providers,
    _result_score,
    search_bali_context,
)


class MockAsyncClient:
    def __init__(self, responses: list[MagicMock]) -> None:
        self._responses = responses

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def post(self, *args: object, **kwargs: object) -> MagicMock:
        return self._responses[-1]

    async def get(self, *args: object, **kwargs: object) -> MagicMock:
        return self._responses[-1]


class TestSearchPlan:
    def test_full_query_only(self) -> None:
        provider = _ExaSearch(api_key="key")
        plan = SearchPlan(full_query="short query", focused_queries=("a", "b"))
        assert provider.queries_for(plan) == ["short query"]

    def test_focused_queries_when_too_long(self) -> None:
        provider = _DeepseekSearch(api_key="key")
        plan = SearchPlan(full_query="x" * 200, focused_queries=("a", "b"))
        assert provider.queries_for(plan) == ["a", "b"]


class TestResultScore:
    def test_rerank_score(self) -> None:
        assert _result_score({"rerank_score": "0.75"}) == 0.75

    def test_missing_score(self) -> None:
        assert _result_score({}) == 0.0

    def test_invalid_score(self) -> None:
        assert _result_score({"rerank_score": "bad"}) == 0.0


class TestDeepseekSearch:
    def test_init(self) -> None:
        client = _DeepseekSearch(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.base_url.endswith("/v1/tools/search")
        assert "deepseek" in str(client)

    def test_refines_query_to_limit(self) -> None:
        client = _DeepseekSearch(api_key="key")
        query = "Bali " * 100
        assert len(client._refine_query(query)) <= 128

    def test_merge_results_deduplicates_and_ranks(self) -> None:
        results = [
            [
                {"title": "A", "url": "https://a.com", "rerank_score": 0.2},
                {"title": "B", "url": "https://b.com", "rerank_score": 0.9},
            ],
            [
                {"title": "A duplicate", "url": "https://a.com/#section", "rerank_score": 0.4},
                {"title": "C", "url": "https://c.com", "rerank_score": 0.5},
            ],
        ]
        merged = _DeepseekSearch._merge_results(results, {})
        assert [item["title"] for item in merged] == ["B", "C", "A duplicate"]

    def test_formatted_result_contains_sources(self) -> None:
        results = [
            {
                "title": "Bali Guide",
                "url": "https://example.com",
                "content": "x" * 10000,
                "date": "2026-11-01",
            },
        ]
        formatted = _DeepseekSearch._formatted_result(results, {"max_content_chars": 10})
        assert "Latest Bali Travel Information:" in formatted
        assert "Sources & Citations:" in formatted
        assert "[1] Bali Guide: https://example.com (2026-11-01)" in formatted
        assert "x" * 10000 not in formatted

    async def test_search_success(self) -> None:
        response = MagicMock()
        response.json.return_value = {
            "results": [{"title": "Bali", "url": "https://bali.com", "content": "info"}],
        }
        response.raise_for_status.return_value = None
        client = _DeepseekSearch(api_key="key")

        with patch(
            "openai_client.services.web_search.deepseek.AsyncClient",
            return_value=MockAsyncClient([response]),
        ):
            success, result = await client.search("Bali", {})

        assert success is True
        assert "Bali" in result

    async def test_search_failure(self) -> None:
        client = _DeepseekSearch(api_key="key")
        with patch(
            "openai_client.services.web_search.deepseek.AsyncClient",
            return_value=MockAsyncClient([MagicMock()]),
        ) as mock_cls:
            mock_cls.return_value.post = AsyncMock(side_effect=ConnectionError("down"))
            success, result = await client.search("Bali", {})

        assert success is False
        assert "Search error" in result


class TestExaSearch:
    def test_init(self) -> None:
        client = _ExaSearch(api_key="test-key")
        assert client.base_url == "https://api.exa.ai/search"
        assert "exa" in str(client)

    async def test_search_success(self) -> None:
        response = MagicMock()
        response.json.return_value = {"output": {"content": "Exa result"}}
        response.raise_for_status.return_value = None
        client = _ExaSearch(api_key="key")

        with patch(
            "openai_client.services.web_search.exa.AsyncClient",
            return_value=MockAsyncClient([response]),
        ):
            success, result = await client.search("Bali", {"num_results": 3})

        assert success is True
        assert result == "Exa result"

    async def test_search_empty_output(self) -> None:
        response = MagicMock()
        response.json.return_value = {}
        response.raise_for_status.return_value = None
        client = _ExaSearch(api_key="key")

        with patch(
            "openai_client.services.web_search.exa.AsyncClient",
            return_value=MockAsyncClient([response]),
        ):
            success, result = await client.search("Bali", {})

        assert success is False
        assert result == ""

    async def test_search_failure(self) -> None:
        client = _ExaSearch(api_key="key")
        with patch(
            "openai_client.services.web_search.exa.AsyncClient",
            return_value=MockAsyncClient([MagicMock()]),
        ) as mock_cls:
            mock_cls.return_value.post = AsyncMock(side_effect=ConnectionError("down"))
            success, result = await client.search("Bali", {})

        assert success is False
        assert "Search error" in result


class TestBraveSearch:
    def test_init(self) -> None:
        client = _BraveSearch(api_key="test-key")
        assert client.base_url == "https://api.search.brave.com/res/v1/web/search"
        assert "brave" in str(client)

    async def test_search_success(self) -> None:
        response = MagicMock()
        response.json.return_value = {
            "web": {
                "results": [
                    {"title": "Bali Guide", "url": "https://example.com", "description": "A guide"},
                ],
            },
        }
        response.raise_for_status.return_value = None
        client = _BraveSearch(api_key="key")

        with patch(
            "openai_client.services.web_search.brave.AsyncClient",
            return_value=MockAsyncClient([response]),
        ):
            success, result = await client.search("Bali travel", {})

        assert success is True
        assert "Bali Guide" in result
        assert "Sources & Citations:" in result

    async def test_search_no_results(self) -> None:
        response = MagicMock()
        response.json.return_value = {}
        response.raise_for_status.return_value = None
        client = _BraveSearch(api_key="key")

        with patch(
            "openai_client.services.web_search.brave.AsyncClient",
            return_value=MockAsyncClient([response]),
        ):
            success, result = await client.search("Bali", {})

        assert success is False
        assert result == ""


class TestProviders:
    def test_returns_all_configured_providers(self) -> None:
        with (
            patch.object(settings, "DEEPSEEK_API_KEY", "deepseek"),
            patch.object(settings, "EXA_API_KEY", "exa"),
            patch.object(settings, "BRAVE_API_KEY", "brave"),
        ):
            providers = _providers()

        assert [type(provider) for provider in providers] == [
            _DeepseekSearch,
            _ExaSearch,
            _BraveSearch,
        ]


class TestWebSearch:
    def test_falls_back_to_next_provider(self) -> None:
        first = MagicMock()
        first.__str__ = lambda self: "first"  # type: ignore[method-assign]
        first.search_many = AsyncMock(return_value=(False, "nope"))
        second = MagicMock()
        second.__str__ = lambda self: "second"  # type: ignore[method-assign]
        second.search_many = AsyncMock(return_value=(True, "result"))

        with patch("openai_client.services.web_search.core._providers", return_value=[first, second]):
            result = async_run(WebSearch("test", {}).search("query"))

        assert result == "result"


class TestSearchBaliContext:
    def test_prefixes_query_with_bali_latest(self) -> None:
        mock_search = MagicMock()
        mock_search.search = AsyncMock(return_value="results")

        with patch("openai_client.services.web_search.core.WebSearch", return_value=mock_search):
            result = async_run(search_bali_context("beaches", "Agent", {}))

        assert result == "results"
        plan = mock_search.search.call_args.args[0]
        assert plan.full_query == "Bali latest beaches"
