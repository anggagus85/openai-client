from logging import ERROR, basicConfig, getLogger
from pathlib import Path

from rich import print as rprint
from rich.logging import RichHandler
from rich.traceback import install

from openai_client.client import OpenAiClient
from openai_client.schemas.ai import EmailInquiry, ItineraryRequestMD
from openai_client.services import (
    analyze_contact,
    chatbot,
    confirmation_message,
    itinerary,
)
from openai_client.utils import get_api_key, save_to_file

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
install()


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
            "api_key": get_api_key("INCEPTION_API_KEY"),
            "base_url": "https://api.inceptionlabs.ai/v1",
            "model": "mercury-2",
        },
        "openrouter": {
            "api_key": get_api_key("OPENROUTER_API_KEY"),
            "base_url": "https://openrouter.ai/api/v1",
            # "model": "stepfun/step-3.5-flash:free",
            "model": "nvidia/nemotron-nano-9b-v2:free",
        },
        "bytez": {
            "api_key": get_api_key("BYTEZ_API_KEY"),
            "base_url": "https://api.bytez.com/models/v2/openai/v1",
            "model": "Qwen/Qwen3-8B",
        },
        "nvidia": {
            "api_key": get_api_key("NVIDIA_API_KEY"),
            "base_url": "https://integrate.api.nvidia.com/v1",
            "model": "mistralai/mistral-large-3-675b-instruct-2512",
            # "model": "moonshotai/kimi-k2-thinking",
        },
    }
    client = OpenAiClient(**kwargs["inception"])

    # client.list_models("bytez")

    # Notes:
    # method = "open_ai" or "stream" ->  all kwargs keys are valid
    # method = "bytez" ->  only kwargs["bytez"] is valid
    methods = ["open_ai", "bytez", "stream"]

    # _do_chat(client, methods[2])

    _generate_itinerary(client, methods[0])

    # _email_inquiry(client, methods[0])


if __name__ == "__main__":
    main()
