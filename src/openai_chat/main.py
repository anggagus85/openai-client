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
install()


OPENROUTER_FREE_MODELS = [
    "qwen/qwen3.6-plus-preview:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "minimax/minimax-m2.5:free",
    "stepfun/step-3.5-flash:free",
    "arcee-ai/trinity-large-preview:free",
    "liquid/lfm-2.5-1.2b-instruct:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "arcee-ai/trinity-mini:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "z-ai/glm-4.5-air:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "google/gemma-3n-e2b-it:free",
    "google/gemma-3n-e4b-it:free",
    "google/gemma-3-4b-it:free",
    "google/gemma-3-12b-it:free",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
]

INCEPTION_MODELS = ["mercury", "mercury-2"]

KIMI_MODELS = [
    "kimi-k2-thinking",
    "kimi-k2-thinking-turbo",
    "kimi-k2-turbo-preview",
    "kimi-k2.5",
    "moonshot-v1-8k-vision-preview",
    "moonshot-v1-8k",
    "moonshot-v1-32k-vision-preview",
    "kimi-k2-0711-preview",
    "moonshot-v1-128k",
    "kimi-k2-0905-preview",
    "moonshot-v1-auto",
    "moonshot-v1-32k",
    "moonshot-v1-128k-vision-preview",
]


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
        "moonshot": {
            "api_key": get_api_key("MOONSHOT_API_KEY"),
            "base_url": "https://api.moonshot.ai/v1",
            "model": "kimi-k2-thinking",
        },
        "bytez": {
            "api_key": get_api_key("BYTEZ_API_KEY"),
            "base_url": "https://api.bytez.com/models/v2/openai/v1",
            "model": "Qwen/Qwen3-4B",
        },
    }
    client = OpenAiClient(**kwargs["inception"])
    # client.list_models()

    # method = ["ask", "bytez", "stream"]
    # chatbot(client, method[0])
    itinerary(client)


if __name__ == "__main__":
    main()
