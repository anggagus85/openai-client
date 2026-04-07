from asyncio import run as async_run
from enum import StrEnum
from logging import ERROR, basicConfig, getLogger
from pathlib import Path

from rich import print as rprint
from rich.logging import RichHandler
from rich.traceback import install

from openai_client.client import MistralAI, OpenAIClient
from openai_client.configs import settings
from openai_client.schemas.ai import EmailInquiry, ItineraryRequestMD
from openai_client.services import Chatbot, EmailService, Itinerary
from openai_client.services.prompts import (
    EMAIL_SYSTEM_MESSAGE,
    ITINERARY_SYSTEM_MESSAGE,
    system_message,
)
from openai_client.types import MistralSystem, SystemParam
from openai_client.utils import save_to_file

basicConfig(
    level="NOTSET",
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
    # Notes:
    # method = "completion" or "stream" ->  all kwargs keys are valid
    # method = "bytez" ->  only kwargs["bytez"] is valid
    COMPLETION = "completion"
    BYTEZ = "bytez"
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


async def _do_chat(client: OpenAIClient, method: str) -> None:
    system_param = SystemParam(**system_message().model_dump())
    await Chatbot(client, system_param=system_param).run(method)


async def _openai_itinerary(client: OpenAIClient, method: str) -> None:
    client.configs["temperature"] = 0
    system_param = SystemParam(role="system", content=ITINERARY_SYSTEM_MESSAGE)
    await Itinerary(client, itinerary_req, system_param).generate(method)


async def _email_inquiry(client: OpenAIClient, method: str) -> None:
    system_param = SystemParam(role="system", content=EMAIL_SYSTEM_MESSAGE)
    service = EmailService(client, inquiry, system_param)
    body = await service.analyze_contact(method)
    save_to_file(body, Path.cwd() / "email_inquiry.html")
    confirmation = await service.confirmation_message(method)
    rprint(f"[b i green]confirmation :[/b i green]\n\n{confirmation.confirmation}")


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
    save_to_file(body, Path.cwd() / "email_inquiry.html")
    confirmation = await service.confirmation_message(method)
    rprint(f"[b i green]confirmation :[/b i green]\n\n{confirmation.confirmation}")


def main() -> None:

    kwargs = {
        "inception": {
            "api_key": settings.INCEPTION_API_KEY,
            "base_url": "https://api.inceptionlabs.ai/v1",
            "model": "mercury-2",
        },
        "openrouter": {
            "api_key": settings.OPENROUTER_API_KEY,
            "base_url": "https://openrouter.ai/api/v1",
            "model": "stepfun/step-3.5-flash:free",
        },
        "bytez": {
            "api_key": settings.BYTEZ_API_KEY,
            "base_url": "https://api.bytez.com/models/v2/openai/v1",
            "model": "google/gemma-4-E2B-it",
        },
        "nvidia": {
            "api_key": settings.NVIDIA_API_KEY,
            "base_url": "https://integrate.api.nvidia.com/v1",
            "model": "mistralai/mistral-large-3-675b-instruct-2512",
        },
        "mistral": {
            "api_key": settings.MISTRAL_API_KEY,
            "base_url": "https://api.mistral.ai",
            # "model": "mistral-large-latest",
            # "model": "devstral-2512",
            # "model": "mistral-medium-latest",
            "model": "mistral-small-2506",
        },
    }
    # client = OpenAIClient(**kwargs["inception"])
    # async_run(_do_chat(client, Methods.STREAM))  # break with ctrl+d
    # async_run(_openai_itinerary(client, Methods.COMPLETION))
    # async_run(_email_inquiry(client, Methods.COMPLETION))
    # client.list_models("openrouter")

    # ------------
    # Mistral AI
    # ------------

    mistral = MistralAI(**kwargs["mistral"])
    # async_run(_mistral_chat(mistral, Methods.STREAM))
    async_run(_mistral_itinerary(mistral, Methods.COMPLETION))
    # async_run(_mistral_email_inquiry(mistral, Methods.COMPLETION))
    # async_run(mistral.list_models())


if __name__ == "__main__":
    main()
