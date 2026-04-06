from enum import StrEnum
from logging import ERROR, basicConfig, getLogger
from pathlib import Path

from rich import print as rprint
from rich.logging import RichHandler
from rich.traceback import install

from openai_client.client import OpenAiClient
from openai_client.configs import settings
from openai_client.schemas.ai import EmailInquiry, ItineraryRequestMD
from openai_client.services import (
    analyze_contact,
    chatbot,
    confirmation_message,
    itinerary,
)
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
install()


class Methods(StrEnum):
    # Notes:
    # method = "completion" or "stream" ->  all kwargs keys are valid
    # method = "bytez" ->  only kwargs["bytez"] is valid
    COMPLETION = "completion"
    BYTEZ = "bytez"
    STREAM = "stream"


def _do_chat(client: OpenAiClient, method: str) -> None:
    chatbot(client, method)


def _generate_itinerary(client: OpenAiClient, method: str) -> None:
    itinerary_req = ItineraryRequestMD(
        duration=6,
        interests=["wellness", "temples", "activities"],
        budget="US$ 1500",
    )
    itinerary(itinerary_req, client, method)


def _email_inquiry(client: OpenAiClient, method: str) -> None:
    email_inquiry = EmailInquiry(
        name="John Doe",
        subject="Travel Inquiry",
        message="I would like to book a trip to Bali for 7 days. And for a smooth and safe drive please provide a driver who can speak English.",
        email="jhondoe@gmail.com",
    )
    body = analyze_contact(email_inquiry, client, method)
    save_to_file(body, Path.cwd() / "email_inquiry.html")
    confirmation = confirmation_message(email_inquiry, client, method)
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
    }
    client = OpenAiClient(**kwargs["inception"])

    # client.list_models("openrouter")

    _do_chat(client, Methods.STREAM)

    # _generate_itinerary(client, Methods.COMPLETION)

    # _email_inquiry(client, Methods.COMPLETION)


if __name__ == "__main__":
    main()
