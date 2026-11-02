from asyncio import run as async_run
from collections.abc import Awaitable, Callable
from enum import StrEnum
from json import dumps, loads
from logging import ERROR, basicConfig, getLogger
from typing import Any

from anyio import Path
from rich import print as rprint
from rich.logging import RichHandler
from rich.traceback import install

from openai_client.client import MistralAI, OpenAIClient
from openai_client.configs import ProviderConfig, settings
from openai_client.configs import Providers as Prv
from openai_client.schemas.ai import EmailInquiry, ItineraryRequestMD
from openai_client.services import Chatbot, EmailService, Itinerary
from openai_client.services.prompts import (
    EMAIL_SYSTEM_MESSAGE,
    ITINERARY_SYSTEM_MESSAGE,
    system_message,
)
from openai_client.typings import MistralSystem, SystemParam
from openai_client.utils import save_to_file

basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="%X",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = getLogger("rich")
getLogger("openai").setLevel(ERROR)
getLogger("httpcore").setLevel(ERROR)
getLogger("urllib3").setLevel(ERROR)
getLogger("mdformat").setLevel(ERROR)
getLogger("markdown_it").setLevel(ERROR)
getLogger("httpx").setLevel(ERROR)
getLogger("asyncio").setLevel(ERROR)
install()


class Methods(StrEnum):
    COMPLETION = "completion"
    STREAM = "stream"


itinerary_req = ItineraryRequestMD(
    duration=6,
    interests=["wellness", "temples", "activities"],
    budget="US$ 1500",
)

inquiry = EmailInquiry(
    name="John Doe",
    subject="Travel Inquiry",
    message="I would like to book a trip to Bali for 7 days. And for a smooth and safe drive please provide a driver who can speak English.",
    email="johndoe@gmail.com",
)


async def _openai_chat(client: OpenAIClient, method: str) -> None:
    system_param = SystemParam(**system_message().model_dump())
    await Chatbot(client, system_param=system_param).run(method)


async def _openai_itinerary(client: OpenAIClient, method: str) -> None:
    client.configs["temperature"] = 0
    system_param = SystemParam(role="system", content=ITINERARY_SYSTEM_MESSAGE)
    await Itinerary(client, itinerary_req, system_param).generate(method)


async def _openai_email_inquiry(client: OpenAIClient, method: str) -> None:
    system_param = SystemParam(role="system", content=EMAIL_SYSTEM_MESSAGE)
    service = EmailService(client, inquiry, system_param)
    body = await service.analyze_contact(method)
    await save_to_file(body, (await Path.cwd()) / "email_inquiry.html")
    confirmation = await service.confirmation_message(method)
    rprint(f"[b i green]confirmation :[/b i green]\n\n{confirmation.confirmation}")


async def _openai_services(
    client: OpenAIClient,
    service: Callable[[OpenAIClient, str], Awaitable[None]],
    method: str,
) -> None:
    if service is _openai_itinerary:
        client.configs["temperature"] = 0

    await service(client, method)


# ------------
# Mistral AI
# ------------


async def _mistral_chat(mistral: MistralAI, method: str) -> None:
    system_param = MistralSystem(**system_message().model_dump())
    await Chatbot(mistral, system_param=system_param).run(method)


async def _mistral_itinerary(mistral: MistralAI, method: str) -> None:
    system_param = MistralSystem(role="system", content=ITINERARY_SYSTEM_MESSAGE)
    mistral.configs["temperature"] = 0
    await Itinerary(mistral, itinerary_req, system_param).generate(method)


async def _mistral_email_inquiry(mistral: MistralAI, method: str) -> None:
    system_param = MistralSystem(role="system", content=EMAIL_SYSTEM_MESSAGE)
    service = EmailService(mistral, inquiry, system_param)
    body = await service.analyze_contact(method)
    await save_to_file(body, (await Path.cwd()) / "email_inquiry.html")
    confirmation = await service.confirmation_message(method)
    rprint(f"[b i green]confirmation :[/b i green]\n\n{confirmation.confirmation}")


async def _mistrals_models(mistral: MistralAI) -> None:
    data: list[dict[str, Any]] = mistral.models.list().model_dump()["data"]
    models: list[dict[str, str]] = []

    for entries in data:
        if not (capabilities := entries["capabilities"])["completion_chat"]:
            continue
        if "labs" in (_id := entries["id"]):
            continue
        models.append(
            {
                "id": _id,
                "description": entries.get("description") or "",
                "max_context_length": entries["max_context_length"],
                "vision": capabilities["vision"],
            },
        )

    path = Path("models.json")
    all_models: dict[str, list[Any]] = (
        loads(await path.read_text(encoding="utf-8")) if await path.exists() else {}
    )
    all_models["mistral"] = models

    await save_to_file(dumps(all_models, indent=4), (await Path.cwd()) / "models.json")


async def _mistrals(
    client: MistralAI,
    service: Callable[[MistralAI, str], Awaitable[None]],
    method: str,
) -> None:
    if service is _mistral_itinerary:
        client.configs["temperature"] = 0

    await service(client, method)


def _require_api_key(config: ProviderConfig, provider: Prv) -> str:
    """Return the provider's API key, failing loudly when it was not configured."""
    if not (api_key := config["api_key"]):
        details = f"{provider.name}_API_KEY is not set but provider '{provider}' is selected"
        raise ValueError(details)
    return api_key


async def main() -> None:

    kwargs: dict[Prv, ProviderConfig] = {
        Prv.ATMOROUTER: {
            "api_key": settings.ATMOROUTER_API_KEY,
            "base_url": settings.ATMOROUTER_BASE_URL,
            "model": settings.ATMOROUTER_MODEL,
        },
        Prv.DEEPSEEK: {
            "api_key": settings.DEEPSEEK_API_KEY,
            "base_url": settings.DEEPSEEK_BASE_URL,
            "model": settings.DEEPSEEK_MODEL,
        },
        Prv.INCEPTION: {
            "api_key": settings.INCEPTION_API_KEY,
            "base_url": settings.INCEPTION_BASE_URL,
            "model": settings.INCEPTION_MODEL,
        },
        Prv.OPENROUTER: {
            "api_key": settings.OPENROUTER_API_KEY,
            "base_url": settings.OPENROUTER_BASE_URL,
            "model": settings.OPENROUTER_MODEL,
        },
        Prv.NVIDIA: {
            "api_key": settings.NVIDIA_API_KEY,
            "base_url": settings.NVIDIA_BASE_URL,
            "model": settings.NVIDIA_MODEL,
        },
        Prv.MISTRAL: {
            "api_key": settings.MISTRAL_API_KEY,
            "base_url": settings.MISTRAL_BASE_URL,
            "model": settings.MISTRAL_MODEL,
        },
    }

    # ------------
    # Mistral AI
    # ------------
    # mistral_config = kwargs[Prv.MISTRAL]
    # mistral = MistralAI(
    #     api_key=_require_api_key(mistral_config, Prv.MISTRAL),
    #     base_url=mistral_config["base_url"],
    #     model=mistral_config["model"],
    # )
    # # await _mistrals_models(mistral)
    # services = [_mistral_chat, _mistral_itinerary, _mistral_email_inquiry]
    # await _mistrals(mistral, services[0], Methods.STREAM)

    provider = Prv.INCEPTION
    config = kwargs[provider]
    client = OpenAIClient(
        api_key=_require_api_key(config, provider),
        base_url=config["base_url"],
        model=config["model"],
    )
    # await client.list_models(provider)
    services = [_openai_chat, _openai_itinerary, _openai_email_inquiry]
    await _openai_services(client, services[0], Methods.STREAM)


if __name__ == "__main__":
    async_run(main())
