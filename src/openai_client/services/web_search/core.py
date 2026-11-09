"""Web search orchestration and public entry point."""

from logging import getLogger
from typing import Any

from openai_client.configs import settings
from openai_client.services.web_search.base import SearchPlan, _SearchProvider
from openai_client.services.web_search.brave import _BraveSearch
from openai_client.services.web_search.deepseek import _DeepseekSearch
from openai_client.services.web_search.exa import _ExaSearch

logger = getLogger(__name__)


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
    if settings.EXA_API_KEY:
        clients.append(_ExaSearch(api_key=settings.EXA_API_KEY))
    else:
        logger.warning("EXA_API_KEY not available, skipping Exa provider")
    if settings.DEEPSEEK_API_KEY:
        clients.append(_DeepseekSearch(api_key=settings.DEEPSEEK_API_KEY))
    else:
        logger.warning("DEEPSEEK_API_KEY not available, skipping DeepSeek search provider")
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
