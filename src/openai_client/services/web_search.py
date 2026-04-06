# src/openai_client/services/search.py

from logging import getLogger
from typing import Any

from httpx import Client

from openai_client.configs import settings

logger = getLogger(__name__)


class SearchClientError(Exception):
    pass


class _ExaSearch:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.base_url = "https://api.exa.ai/search"

    def search(self, query: str, configs: dict[str, Any] | None = None) -> tuple[bool, str]:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        num_results = (configs or {}).get("num_results", 5)
        contents = (configs or {}).get("contents", {"text": {"max_characters": 20000}})
        payload = {"query": query, "numResults": num_results, "contents": contents, "type": "auto"}

        try:
            with Client(timeout=30.0) as client:
                response = client.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

            return True, self._process_results(data)
        except Exception as e:
            logger.exception("Exa search failed")
            return False, f"Search error: {e}"

    def _process_results(self, data: dict) -> str:
        results = []
        for i, result in enumerate(data.get("results", []), 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            text = result.get("text", "")[:500]
            results.append(f"{i}. **{title}**\n   URL: {url}\n   {text}...")

        return "\n\n".join(results)

    def __str__(self) -> str:
        return f"provider 'exa' via {self.base_url}"


class _BraveSearch:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    def search(self, query: str, configs: dict[str, Any] | None = None) -> tuple[bool, str]:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        count = (configs or {}).get("count", 5)
        params = {"q": query, "count": count}

        try:
            with Client(timeout=30.0) as client:
                response = client.get(self.base_url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()

            results = []
            for i, result in enumerate(data.get("web", {}).get("results", []), 1):
                title = result.get("title", "Untitled")
                url = result.get("url", "")
                description = result.get("description", "")
                results.append(f"{i}. **{title}**\n   URL: {url}\n   {description}")

            return True, "\n\n".join(results)
        except Exception as e:
            logger.exception("Brave search failed")
            return False, f"Search error: {e}"

    def __str__(self) -> str:
        return f"provider 'brave' via {self.base_url}"


class WebSearch:
    """
    Unified search interface.

    Handles provider iteration and fallback. Iterates through configured
    search providers in order, returning the first successful result.  Logs
    failures automatically and returns ``None`` when all providers fail.
    """

    def __init__(self, operation: str, configs: dict[str, Any] | None = None) -> None:
        self._providers = _providers()
        self._operation = operation
        self._configs = configs

    def search(self, query: str) -> str | None:
        """
        Execute search across providers.

        Automatically falls back to the next provider on failure.

        Args:
            query: The search query string.
            configs: Provider-specific parameters (e.g., ``num_results``,
                ``contents`` for Exa; ``count`` for Brave).  Each provider
                ignores configs it does not use.

        Returns:
            Formatted search results string, or ``None`` if all providers fail.
        """
        for provider in self._providers:
            logger.info("%s %s", self._operation, provider)
            success, results = provider.search(query, self._configs)

            if success:
                logger.info("Search succeeded with %s", provider)
                return results

            logger.warning("Search with %s failed, error: %s", provider, results)


def _providers() -> list[_ExaSearch | _BraveSearch]:
    clients: list[_ExaSearch | _BraveSearch] = []
    if settings.EXA_API_KEY:
        clients.append(_ExaSearch(api_key=settings.EXA_API_KEY))
    else:
        logger.warning("EXA_API_KEY not available, skipping Exa provider")
    if settings.BRAVE_API_KEY:
        clients.append(_BraveSearch(api_key=settings.BRAVE_API_KEY))
    else:
        logger.warning("BRAVE_API_KEY not available, skipping Brave provider")
    return clients
