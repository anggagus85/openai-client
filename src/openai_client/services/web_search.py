from asyncio import gather
from dataclasses import dataclass
from datetime import datetime
from logging import getLogger
from typing import Any

from httpx2 import AsyncClient

from openai_client.configs import settings
from openai_client.utils.query import fit_query

logger = getLogger(__name__)

SEARCH_SYSTEM_PROMPT = "You are Bali travel guide expert."

_DEEPSEEK_MAX_QUERY_LENGTH = 128
_DEEPSEEK_MAX_CONTENT_CHARS = 8000
_DEEPSEEK_MAX_TOTAL_CHARS = 24000
_DEEPSEEK_MAX_MERGED_RESULTS = 10
_EXA_MAX_QUERY_LENGTH = 400


@dataclass(frozen=True)
class SearchPlan:
    """A search request with a full query and optional focused sub-queries."""

    full_query: str
    focused_queries: tuple[str, ...] = ()


class _SearchProvider:
    """Base class for web search providers."""

    query_limit: int | None = None

    def queries_for(self, plan: SearchPlan) -> list[str]:
        """Choose which queries to send based on this provider's limit."""
        if (
            self.query_limit is not None
            and plan.focused_queries
            and len(plan.full_query) > self.query_limit
        ):
            return list(plan.focused_queries)

        return [plan.full_query]

    async def search(self, query: str, configs: dict[str, Any]) -> tuple[bool, str]:
        """Run a single search query."""
        raise NotImplementedError

    async def search_many(
        self,
        queries: list[str],
        configs: dict[str, Any],
    ) -> tuple[bool, str]:
        """Run one or more queries."""
        if not queries:
            return False, ""

        return await self.search(queries[0], configs)


def _result_score(result: dict[str, Any]) -> float:
    try:
        return float(result.get("rerank_score") or 0)
    except (TypeError, ValueError):
        return 0.0


class _DeepseekSearch(_SearchProvider):
    """DeepSeek in-house web search provider."""

    query_limit = _DEEPSEEK_MAX_QUERY_LENGTH

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.base_url = f"{settings.DEEPSEEK_BASE_URL.rstrip('/')}/v1/tools/search"

    @staticmethod
    def _refine_query(query: str) -> str:
        """Fit a query to DeepSeek's 128-character limit."""
        return fit_query(query, _DEEPSEEK_MAX_QUERY_LENGTH)

    async def _fetch(self, query: str, configs: dict[str, Any]) -> list[dict[str, Any]]:
        """Fetch raw DeepSeek search results for one query."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        count = configs.get("num_results") or configs.get("count") or 5
        payload: dict[str, Any] = {
            "query": self._refine_query(query),
            "count": max(1, min(int(count), 50)),
            "need_content": bool(configs.get("need_content", False)),
        }

        if ori_question := configs.get("ori_question"):
            payload["ori_question"] = self._refine_query(str(ori_question))
        if localization := configs.get("localization"):
            payload["localization"] = localization

        async with AsyncClient(timeout=30.0) as client:
            response = await client.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        results = data.get("results", [])
        if not isinstance(results, list):
            return []

        return [result for result in results if isinstance(result, dict)]

    async def search(self, query: str, configs: dict[str, Any]) -> tuple[bool, str]:
        try:
            results = await self._fetch(query, configs)
            if not results:
                return False, ""
            return True, self._formatted_result(results, configs)
        except Exception as e:
            logger.exception("DeepSeek search failed")
            return False, f"Search error: {e}"

    async def _fetch_safe(
        self,
        query: str,
        configs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            return await self._fetch(query, configs)
        except Exception:
            logger.exception("DeepSeek search failed for query %r", query)
            return []

    async def search_many(
        self,
        queries: list[str],
        configs: dict[str, Any],
    ) -> tuple[bool, str]:
        if not queries:
            return False, ""
        if len(queries) == 1:
            return await self.search(queries[0], configs)

        try:
            result_sets = await gather(
                *(self._fetch_safe(query, configs) for query in queries),
            )
            merged = self._merge_results(result_sets, configs)
            if not merged:
                return False, ""
            return True, self._formatted_result(merged, configs)
        except Exception as e:
            logger.exception("DeepSeek search failed")
            return False, f"Search error: {e}"

    @staticmethod
    def _merge_results(
        result_sets: list[list[dict[str, Any]]],
        configs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        max_results = max(
            1,
            int(configs.get("max_merged_results", _DEEPSEEK_MAX_MERGED_RESULTS)),
        )
        ranked = sorted(
            (result for results in result_sets for result in results),
            key=_result_score,
            reverse=True,
        )
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()

        for result in ranked:
            url = str(result.get("url") or "")
            key = url.split("#", 1)[0].rstrip("/").lower() or str(result.get("title") or "")
            if not key or key in seen:
                continue

            seen.add(key)
            merged.append(result)
            if len(merged) >= max_results:
                break

        return merged

    @staticmethod
    def _formatted_result(results: list[dict[str, Any]], configs: dict[str, Any]) -> str:
        max_chars = int(configs.get("max_content_chars", _DEEPSEEK_MAX_CONTENT_CHARS))
        max_total_chars = int(configs.get("max_total_chars", _DEEPSEEK_MAX_TOTAL_CHARS))
        remaining = max_total_chars
        content_parts: list[str] = []
        sources: list[str] = []

        for i, result in enumerate(results, 1):
            title = result.get("title") or "Unknown Source"
            url = result.get("url") or "#"
            raw_body = result.get("content") or result.get("summary") or result.get("snippet") or ""
            body = str(raw_body).strip()
            if body and remaining > 0:
                body = body[: min(max_chars, remaining)].rstrip()
                remaining -= len(body)
                content_parts.append(f"[{i}] {title}\n{body}")
            else:
                content_parts.append(f"[{i}] {title}")

            source = f"[{i}] {title}: {url}"
            if date := result.get("date"):
                source += f" ({date})"
            sources.append(source)

        return (
            "Latest Bali Travel Information:\n"
            f"{chr(10).join(content_parts)}\n\n"
            "Sources & Citations:\n"
            f"{chr(10).join(sources)}"
        )

    def __str__(self) -> str:
        return f"provider 'deepseek' via {self.base_url}"


class _ExaSearch(_SearchProvider):
    query_limit = _EXA_MAX_QUERY_LENGTH

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.base_url = "https://api.exa.ai/search"

    @staticmethod
    def _refine_query(query: str) -> str:
        """Fit a query to Exa's request limit."""
        return fit_query(query, _EXA_MAX_QUERY_LENGTH)

    async def search(self, query: str, configs: dict[str, Any]) -> tuple[bool, str]:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        # Default to start of the current year
        start_date = datetime(datetime.now().year, 1, 1).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        # end_date = "2026-04-23T00:00:00.000Z"
        payload = {
            "systemPrompt": SEARCH_SYSTEM_PROMPT,
            "query": self._refine_query(query),
            "numResults": configs.get("num_results", 5),
            "contents": configs.get("contents", {"highlights": {"max_characters": 8000}}),
            "type": configs.get("type", "auto"),
            "startPublishedDate": start_date,
            # "endPublishedDate": end_date,
            "outputSchema": {
                "type": "text",
                "description": "return latest Bali travel information",
            },
        }

        if category := configs.get("category"):
            payload["category"] = category

        try:
            async with AsyncClient(timeout=30.0) as client:
                response = await client.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

            if not (content := data.get("output", {}).get("content", "")):
                return False, ""

            return True, content
        except Exception as e:
            logger.exception("Exa search failed")
            return False, f"Search error: {e}"

    def __str__(self) -> str:
        return f"provider 'exa' via {self.base_url}"


class _BraveSearch(_SearchProvider):
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self._headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }

    async def search(self, query: str, configs: dict[str, Any]) -> tuple[bool, str]:

        # 1. Combine persona and query
        full_query = f"{SEARCH_SYSTEM_PROMPT} {query}"

        # 2. Strict truncation to Brave's 400-character limit
        # We use 390 to be safe and avoid edge cases with URL encoding
        refined_query = (full_query[:390] + "..") if len(full_query) > 390 else full_query

        # Value  | Description   | Best Use Case
        # ------ | ------------- | ----------------------------------------------
        # "pd"   | Past 24 Hours | For breaking news or real-time travel alerts.
        # "pw"   | Past Week     | For high-turnover topics (e.g., events this weekend).
        # "pm"   | Past Month    | Excellent balance for ""recent"" Bali restaurant/hotel reviews.
        # "py"   | Past Year     | Closest to your 2026-01-01 goal.

        params = {
            "q": refined_query,
            # Map 'num_results' (Exa style) to Brave's 'count' parameter
            "count": configs.get("num_results") or configs.get("count") or 5,
            "freshness": configs.get("freshness", "pm"),
            "text_decorations": False,
            "extra_snippets": True,
        }

        try:
            async with AsyncClient(timeout=30.0) as client:
                response = await client.get(self.base_url, params=params, headers=self._headers)
                response.raise_for_status()
                data = response.json()

            if not (web_results := data.get("web", {}).get("results", [])):
                return False, ""

            return True, self._formatted_result(web_results)
        except Exception as e:
            logger.exception("Brave search failed")
            return False, f"Search error: {str(e)}"

    @staticmethod
    def _formatted_result(web_results: list[dict[str, Any]]) -> str:
        # Structural Alignment with Exa: Narrated snippets + Citation block
        content_parts: list[str] = []
        sources: list[str] = []

        for i, result in enumerate(web_results, 1):
            title = result.get("title", "Unknown Source")
            url = result.get("url", "#")
            snippet = str(result.get("description") or "").strip()

            if snippet:
                content_parts.append(f"[{i}] {snippet}")
                sources.append(f"[{i}] {title}: {url}")

        # Formatted for the LLM's context window
        return (
            "Latest Bali Travel Information:\n"
            f"{' '.join(content_parts)}\n\n"
            "Sources & Citations:\n"
            f"{chr(10).join(sources)}"
        )

    def __str__(self) -> str:
        return f"provider 'brave' via {self.base_url}"


class WebSearch:
    def __init__(self, operation: str, configs: dict[str, Any]) -> None:
        self._providers = _providers()
        self._operation = operation
        self._configs = configs

    async def search(self, query: str | SearchPlan) -> str | None:
        plan = query if isinstance(query, SearchPlan) else SearchPlan(full_query=query)
        for provider in self._providers:
            logger.info("%s %s", self._operation, provider)
            try:
                success, results = await provider.search_many(
                    provider.queries_for(plan),
                    self._configs,
                )
            except Exception as e:
                logger.exception("Search provider %s failed", provider)
                success, results = False, f"Search error: {e}"

            if success:
                logger.info("Search succeeded with %s", provider)
                return results

            logger.warning("Search with %s failed, error: %s", provider, results)


def _providers() -> list[_SearchProvider]:
    clients: list[_SearchProvider] = []
    if settings.DEEPSEEK_API_KEY:
        clients.append(_DeepseekSearch(api_key=settings.DEEPSEEK_API_KEY))
    else:
        logger.warning("DEEPSEEK_API_KEY not available, skipping DeepSeek search provider")
    if settings.EXA_API_KEY:
        clients.append(_ExaSearch(api_key=settings.EXA_API_KEY))
    else:
        logger.warning("EXA_API_KEY not available, skipping Exa provider")
    if settings.BRAVE_API_KEY:
        clients.append(_BraveSearch(api_key=settings.BRAVE_API_KEY))
    else:
        logger.warning("BRAVE_API_KEY not available, skipping Brave provider")
    return clients


async def search_bali_context(
    query: str | SearchPlan,
    agent: str,
    configs: dict[str, Any],
) -> str | None:
    """Run a Bali-focused web search, mirroring the production agent helper."""
    web_search = WebSearch(f"{agent} is searching Bali latest information with", configs)
    search_plan = (
        query if isinstance(query, SearchPlan) else SearchPlan(full_query=f"Bali latest {query}")
    )

    if not (results := await web_search.search(search_plan)):
        return

    return results
