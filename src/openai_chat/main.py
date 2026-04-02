from logging import ERROR, basicConfig, getLogger

from rich.logging import RichHandler
from rich.traceback import install

from openai_chat.client import OpenAiClient
from openai_chat.services import chatbot, itinerary  # noqa: F401
from openai_chat.utils import get_api_key

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
            "model": "minimax/minimax-m2.5:free",
        },
        "bytez": {
            "api_key": get_api_key("BYTEZ_API_KEY"),
            "base_url": "https://api.bytez.com/models/v2/openai/v1",
            "model": "Qwen/Qwen3-4B",
        },
    }
    client = OpenAiClient(**kwargs["bytez"])

    # client.list_models()

    # Notes:
    # - method = "open_ai" or "stream" ->  all kwargs keys are valid
    # - method = "bytez" ->  only kwargs["bytez"] is valid
    method = ["open_ai", "bytez", "stream"]

    # chatbot(client, method[0])
    itinerary(client, method[1])  # Itinerary generation prefer to use bytez


if __name__ == "__main__":
    main()
