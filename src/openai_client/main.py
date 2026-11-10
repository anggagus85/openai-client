from asyncio import run as async_run
from datetime import date, timedelta
from enum import StrEnum
from logging import ERROR, basicConfig, getLogger

from anyio import Path
from rich import print as rprint
from rich.logging import RichHandler
from rich.traceback import install

from openai_client.client import MistralAI, OpenAIClient
from openai_client.configs import ProviderConfig, settings
from openai_client.configs import Providers as Prv
from openai_client.schemas.ai import EmailInquiry, ItineraryRequest
from openai_client.services import Chatbot, EmailService, Itinerary
from openai_client.utils import save_itinerary_to_file

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


_ITINERARY_START = date(2027, 3, 3)

itinerary_req = ItineraryRequest(
    duration=6,
    dates=[(_ITINERARY_START + timedelta(days=offset)).isoformat() for offset in range(6)],
    interests=["water sports", "traditional markets", "cultural", "activities"],
    budget="US$ 2500",
)

inquiry = EmailInquiry(
    name="John Doe",
    subject="Travel Inquiry",
    message="I would like to book a trip to Bali for 7 days. And for a smooth and safe drive please provide a driver who can speak English.",
    email="johndoe@gmail.com",
)


async def _openai_chat(client: OpenAIClient, method: str) -> None:
    await Chatbot(client).run(method)


async def _openai_itinerary(client: OpenAIClient) -> None:
    response = await Itinerary(client, itinerary_req).run()
    await save_itinerary_to_file(response.markdown, response.whatsapp)
    rprint("[b i green]itinerary saved :[/b i green]\n\nitinerary.md\nitinerary.txt")


async def _openai_email_inquiry(client: OpenAIClient) -> None:
    service = EmailService(client, inquiry)
    confirmation = await service.run()
    rprint(f"[b i green]confirmation :[/b i green]\n\n{confirmation.confirmation}")


# ------------
# Mistral AI
# ------------


async def _mistral_chat(mistral: MistralAI, method: str) -> None:
    await Chatbot(mistral).run(method)


async def _mistral_itinerary(mistral: MistralAI) -> None:
    response = await Itinerary(mistral, itinerary_req).run()
    await save_itinerary_to_file(response.markdown, response.whatsapp)
    rprint("[b i green]itinerary saved :[/b i green]\n\nitinerary.md\nitinerary.txt")


async def _mistral_email_inquiry(mistral: MistralAI) -> None:
    service = EmailService(mistral, inquiry)
    confirmation = await service.run()
    rprint(f"[b i green]confirmation :[/b i green]\n\n{confirmation.confirmation}")


def _require_api_key(config: ProviderConfig, provider: Prv) -> str:
    """Return the provider's API key, failing loudly when it was not configured."""
    if not (api_key := config["api_key"]):
        details = f"{provider.name}_API_KEY is not set but provider '{provider}' is selected"
        raise ValueError(details)
    return api_key


def _collected(display_name: str, path: Path) -> str:

    return f"[b green] All [b i magenta]{display_name}[/b i magenta] models have been collected in {path}"


async def _collect_models(kwargs: dict[Prv, ProviderConfig]) -> None:

    path = Path("models.json")
    for prv, config in kwargs.items():
        display_name = prv.display_name
        try:
            api_key = _require_api_key(config, prv)
        except ValueError as e:
            logger.warning("%s", e)
            continue

        if prv == Prv.MISTRAL:
            client: MistralAI | OpenAIClient = MistralAI(
                api_key=api_key,
                model=config["model"],
                base_url=config["base_url"],
            )
        else:
            client = OpenAIClient(
                api_key=api_key,
                base_url=config["base_url"],
                model=config["model"],
                prv=prv,
            )

        try:
            await client.list_models(path)
        except Exception:
            logger.exception("Failed to collect %s models", display_name)
            continue
        finally:
            await client.close()

        rprint(_collected(display_name, path))


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

    # try:
    #     await _mistral_chat(mistral, Methods.STREAM)
    #     # await _mistral_itinerary(mistral)
    #     # await _mistral_email_inquiry(mistral)
    # finally:
    #     await mistral.close()

    # provider = Prv.ATMOROUTER
    # config = kwargs[provider]
    # client = OpenAIClient(
    #     api_key=_require_api_key(config, provider),
    #     base_url=config["base_url"],
    #     model=config["model"],
    #     prv=provider,
    # )
    # try:
    #     await _openai_chat(client, Methods.STREAM)
    #     # await _openai_itinerary(client)
    #     # await _openai_email_inquiry(client)
    # finally:
    #     await client.close()

    await _collect_models(kwargs)


if __name__ == "__main__":
    async_run(main())
