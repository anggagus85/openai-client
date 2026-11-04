# src/openai_client/services/itinerary.py

"""OpenAI itinerary service mirrored from the production BaliRoam implementation."""

from logging import getLogger
from typing import Any

from openai_client.client import MistralAI, OpenAIClient
from openai_client.configs import settings
from openai_client.errors import ItineraryGenerationError
from openai_client.schemas.ai import ItineraryRequest, ItineraryResponse
from openai_client.services.prompts import (
    ITINERARY_SYSTEM_PROMPT,
    append_system_message,
    append_user_message,
    itinerary_prompt,
    itinerary_search_messages,
    search_bali_info,
    search_bali_queries,
)
from openai_client.services.web_search import SearchPlan, search_bali_context
from openai_client.typings import ListParams, MistralParams, MistralSystem, SystemParam
from openai_client.utils import clean_markdown, markdown_to_whatsapp, save_itinerary_to_file

logger = getLogger(__name__)


class Itinerary:
    """Itinerary generation service using OpenAI-compatible providers."""

    def __init__(
        self,
        client: OpenAIClient | MistralAI,
        itinerary_req: ItineraryRequest,
        system_message: SystemParam | MistralSystem | None = None,
    ) -> None:
        self._client = client
        self._itinerary_req = itinerary_req
        self._system_message = system_message
        self._messages: ListParams | MistralParams = []
        self._is_mistral = bool(getattr(client, "is_mistral", False))
        self._configs: dict[str, Any] = {"temperature": 0.0, "top_p": 1.0}

    def _system_content(self) -> str:
        if self._system_message is None:
            return ITINERARY_SYSTEM_PROMPT
        if isinstance(self._system_message, MistralSystem):
            return str(self._system_message.content or "")
        return str(self._system_message["content"])

    async def generate(self, search_result: str) -> ItineraryResponse:
        """
        Generate an itinerary from the supplied search/prompt content.

        Mirrors the production service: the caller supplies the full user
        prompt, the service prepends the itinerary system prompt and performs
        the completion.
        """
        self._messages = []
        append_system_message(
            self._messages,
            self._system_content(),
            is_mistral=self._is_mistral,
        )
        append_user_message(self._messages, search_result, is_mistral=self._is_mistral)
        self._configs["messages"] = self._messages

        logger.info("Generating itinerary with %s", self._client)

        if not (content := await self._client.generate(self._configs, stream=False)):
            logger.warning("No itinerary content returned from %s", self._client)
            raise ItineraryGenerationError()

        content = clean_markdown(content)
        whatsapp_text = markdown_to_whatsapp(content)

        if settings.DEBUG:
            await save_itinerary_to_file(content, whatsapp_text)

        return ItineraryResponse(markdown=content, whatsapp=whatsapp_text)

    async def run(self, method: str = "completion") -> ItineraryResponse:
        """
        CLI harness: search Bali, build the production prompt, then generate.

        Production always performs itinerary generation without streaming, so
        ``method`` is accepted for CLI symmetry but does not alter the call.
        """
        if method not in {"completion", "stream"}:
            details = f"Unknown method: {method}"
            raise ValueError(details)

        request = self._itinerary_req
        configs: dict[str, Any] = {
            "num_results": 5,
            "contents": {"highlights": {"max_characters": 8000}},
            "type": "deep",
            "category": "personal site",
            "max_merged_results": 10,
            "max_total_chars": 24000,
        }
        query = search_bali_info(request)
        search_plan = SearchPlan(
            full_query=f"Bali latest {query}",
            focused_queries=tuple(search_bali_queries(request)),
        )

        if not (results := await search_bali_context(search_plan, "Itinerary Agent", configs)):
            logger.warning("Search returned no results or error, proceeding without search data")
            content = itinerary_prompt(request)
        else:
            content = itinerary_search_messages(results, request)

        return await self.generate(content)
