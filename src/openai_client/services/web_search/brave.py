"""Brave web search provider."""

from logging import getLogger
from typing import Any

from httpx2 import AsyncClient

from openai_client.services.web_search.base import SEARCH_SYSTEM_PROMPT, _SearchProvider

logger = getLogger(__name__)

# Brave's endpoint caps the total query; the prompt prefix is fixed, so this is
# the remaining budget available to the query itself.
_BRAVE_QUERY_LIMIT = 400 - len(SEARCH_SYSTEM_PROMPT) - 1


class _BraveSearch(_SearchProvider):
    # Available query length after the prompt prefix; queries_for() and search()
    # both use this so a selected query is not truncated differently when sent.
    query_limit = _BRAVE_QUERY_LIMIT

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self._headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }

    async def search(self, query: str, configs: dict[str, Any]) -> tuple[bool, str]:

        # Bound the query to the same limit used by queries_for() (which already
        # accounts for the prompt prefix), then prepend the persona.
        limit = self.query_limit or _BRAVE_QUERY_LIMIT
        refined_query = query[:limit]
        full_query = f"{SEARCH_SYSTEM_PROMPT} {refined_query}"

        # Value  | Description   | Best Use Case
        # ------ | ------------- | ----------------------------------------------
        # "pd"   | Past 24 Hours | For breaking news or real-time travel alerts.
        # "pw"   | Past Week     | For high-turnover topics (e.g., events this weekend).
        # "pm"   | Past Month    | Excellent balance for ""recent"" Bali restaurant/hotel reviews.
        # "py"   | Past Year     | Closest to your 2026-01-01 goal.

        count = configs.get("num_results") or configs.get("count") or 5
        params = {
            "q": full_query,
            # Map 'num_results' (Exa style) to Brave's 'count' parameter.
            # Brave caps results per request at 20.
            "count": max(1, min(int(count), 20)),
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

            formatted = self._formatted_result(web_results)
            if not formatted:
                return False, ""
            return True, formatted
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

        if not content_parts:
            return ""

        # Formatted for the LLM's context window
        return (
            "Latest Bali Travel Information:\n"
            f"{' '.join(content_parts)}\n\n"
            "Sources & Citations:\n"
            f"{chr(10).join(sources)}"
        )

    def __str__(self) -> str:
        return f"provider 'brave' via {self.base_url}"
