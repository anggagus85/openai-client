# src/openai_client/services/search.py

from logging import getLogger

from httpx import Client

from openai_client.configs import settings

logger = getLogger(__name__)


class SearchClientError(Exception):
    pass


class ExaSearchClient:
    def __init__(self, api_key: str, base_url: str = "https://api.exa.ai/search") -> None:
        self.api_key = api_key
        self.base_url = base_url

    def search(self, query: str, num_results: int = 5) -> tuple[bool, str]:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "query": query,
            "numResults": num_results,
            "contents": {"text": True, "highlights": True},
        }

        try:
            with Client(timeout=30.0) as client:
                response = client.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

            results = []
            for i, result in enumerate(data.get("results", []), 1):
                title = result.get("title", "Untitled")
                url = result.get("url", "")
                text = result.get("text", "")[:500]
                results.append(f"{i}. **{title}**\n   URL: {url}\n   {text}...")

            return True, "\n\n".join(results)
        except Exception as e:
            logger.exception("Exa search failed")
            return False, f"Search error: {e}"

    def __str__(self) -> str:
        return f"provider 'exa' via {self.base_url}"


class BraveSearchClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.search.brave.com/res/v1/web/search",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url

    def search(self, query: str, count: int = 5) -> tuple[bool, str]:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
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


def providers() -> list[ExaSearchClient | BraveSearchClient]:
    clients: list[ExaSearchClient | BraveSearchClient] = []
    if settings.EXA_API_KEY:
        clients.append(ExaSearchClient(api_key=settings.EXA_API_KEY))
    else:
        logger.warning("EXA_API_KEY not available, skipping Exa provider")
    if settings.BRAVE_API_KEY:
        clients.append(BraveSearchClient(api_key=settings.BRAVE_API_KEY))
    else:
        logger.warning("BRAVE_API_KEY not available, skipping Brave provider")
    return clients
