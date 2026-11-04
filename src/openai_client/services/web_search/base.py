"""Shared web-search types, constants, and provider base class."""

from dataclasses import dataclass
from typing import Any

SEARCH_SYSTEM_PROMPT = "You are Bali travel guide expert."


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
        """Run one or more queries, combining every successful result."""
        if not queries:
            return False, ""

        if len(queries) == 1:
            return await self.search(queries[0], configs)

        parts: list[str] = []
        for query in queries:
            success, result = await self.search(query, configs)
            if success and result:
                parts.append(result)

        if not parts:
            return False, ""

        return True, "\n\n".join(parts)


def _result_score(result: dict[str, Any]) -> float:
    try:
        return float(result.get("rerank_score") or 0)
    except (TypeError, ValueError):
        return 0.0
