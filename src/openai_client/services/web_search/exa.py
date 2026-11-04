"""Exa web search provider."""

from datetime import datetime
from logging import getLogger
from typing import Any

from httpx2 import AsyncClient

from openai_client.services.web_search.base import SEARCH_SYSTEM_PROMPT, _SearchProvider
from openai_client.utils.query import fit_query

logger = getLogger(__name__)

_EXA_MAX_QUERY_LENGTH = 400


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
