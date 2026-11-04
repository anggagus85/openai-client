"""DeepSeek web search provider."""

from asyncio import gather
from logging import getLogger
from typing import Any

from httpx2 import AsyncClient

from openai_client.configs import settings
from openai_client.services.web_search.base import _result_score, _SearchProvider
from openai_client.utils.query import fit_query

logger = getLogger(__name__)

_DEEPSEEK_MAX_QUERY_LENGTH = 128


_DEEPSEEK_MAX_CONTENT_CHARS = 8000


_DEEPSEEK_MAX_TOTAL_CHARS = 24000


_DEEPSEEK_MAX_MERGED_RESULTS = 10


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
