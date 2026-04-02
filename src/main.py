from collections.abc import Generator
from contextlib import suppress
from logging import ERROR, basicConfig, getLogger
from typing import Never, cast

from bytez import Bytez
from bytez.client import Response
from dotenv import dotenv_values
from openai import (
    AuthenticationError,
    BadRequestError,
    ConflictError,
    ContentFilterFinishReasonError,
    InternalServerError,
    InvalidWebhookSignatureError,
    LengthFinishReasonError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from rich import print as rprint
from rich.logging import RichHandler
from rich.traceback import install

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


API_EXCEPTIONS = (
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    NotFoundError,
    ConflictError,
    UnprocessableEntityError,
    RateLimitError,
    InternalServerError,
    LengthFinishReasonError,
    ContentFilterFinishReasonError,
    InvalidWebhookSignatureError,
)

SYSTEM_EXCEPTIONS = (
    SystemExit,
    EOFError,
    OSError,
    PermissionError,
    ConnectionError,
    TimeoutError,
    FileNotFoundError,
)


class ApiError(Exception):
    pass


class UnknownError(Exception):
    pass


class AppSystemError(Exception):
    pass


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

_env = dotenv_values(verbose=True)


def _api_key(key: str) -> str:
    if api_key := _env.get(key):
        return api_key
    return ""


class OpenAiClient(OpenAI):
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        super().__init__(api_key=api_key, base_url=base_url)
        self._messages: list[ChatCompletionMessageParam] = []
        self._system_message()
        self._model = model
        self._bytez_model = Bytez(_api_key("BYTEZ_API_KEY")).model(model)

    def _system_message(self) -> None:

        self._messages.append(
            ChatCompletionSystemMessageParam(
                role="system",
                content="""
    <assignment>
    Your name is "Made", you are the official BaliBlissed Concierge, a friendly and expert customer service assistant dedicated to crafting the perfect Balinese getaway. 🌴🥥✨
    </assignment>

    <personality_&_style>
    - Introduction : Intruduce yourself as BaliBlissed Concierge.
    - Vibe: Be exceptionally warm, welcoming, and enthusiastic. You are the digital face of Bali hospitality.
    - Engagement: Use emojis liberally to make your content engaging, fun, and vibrant! 🏝️🍹
    - Formatting: Use bold text and bullet points to ensure your recommendations are easy to scan for travelers on the move.
    </personality_&_style>

    <knowledge_&_scope>
    - Bali Expert: You specialize in Balinese culture, luxury villas, hidden jungle waterfalls, and the best surf spots. 🗺️🌺
    - Strictly Bali: You are an expert ONLY on Bali. If a user asks about other destinations (like Thailand or Europe), politely pivot the conversation back to the wonders of Bali.
    - Company First: Always redirect to BaliBlissed website for any booking inquiries, itinerary, and detailed information.
    </knowledge_&_scope>

    <boundaries_&_safety>
    - No over-reply: Focus only the user's topic question, No suggestions or informations if the user not ask.
    - No Professional Advice: Never provide legal, medical, or financial advice.
    - Accuracy: If you aren't sure about a specific opening time or price, suggest the user check the official BaliBlissed website or contact the venue directly.
    - Prohibited: Reject any questions involving terrorism, racism, pornography, or violence.
    </boundaries_&_safety>
    """,
            ),
        )

    def _ask(self) -> str | None:
        """
        Support multi-turn conversations.

        Each time the chat function is called to converse with the large language model, it has access to the previous conversation history. In other words, the large language model has a memory.
        """
        # We engage in a conversation with the large language model, carrying the messages along
        completion = self.chat.completions.create(
            messages=self._messages,
            model=self._model,
            temperature=0.5,
            timeout=60,
            reasoning_effort=None,
        )

        # Through the API, we obtain the response message (role=assistant) from the large language model
        assistant_message = completion.choices[0].message
        assistant_param = ChatCompletionAssistantMessageParam(**assistant_message.model_dump())

        # To ensure the language model has a complete memory,
        # add the message it returns to us to the messages list
        self._messages.append(assistant_param)

        if content := assistant_message.content:
            return content

    def _bytez_chat(self) -> str | None:
        params = {"temperature": 0}

        response = cast(Response, self._bytez_model.run(self._messages, params=params))

        if error := response.error:
            raise ApiError(error)

        out_dict = cast(dict[str, str], response.output)

        assistant_param = ChatCompletionAssistantMessageParam(
            role="assistant",
            content=out_dict["content"],
        )
        self._messages.append(assistant_param)

        content = out_dict["content"]
        if "</think>" in content:
            return content.split("</think>", 1)[1].strip()
        return content.strip()

    def _stream_chat(self) -> Generator:

        completion = self.chat.completions.create(
            messages=self._messages,
            model=self._model,
            temperature=0.5,
            timeout=60,
            reasoning_effort=None,
            stream=True,
        )

        for chunk in completion:
            if (choices := chunk.choices) and (delta := choices[0].delta.content):
                yield delta

    def _append_user_message(self, question: str) -> None:
        user_param = ChatCompletionUserMessageParam(
            role="user",
            content=f"{question.strip()}",
        )
        self._messages.append(user_param)

    @staticmethod
    def _unknown_method(method: str) -> Never:
        details = f"Invalid method: {method}"
        logger.error(details)
        raise ValueError(details)

    def run_chat(self, method: str) -> None:

        with suppress(KeyboardInterrupt):
            try:
                while True:
                    rprint("\n[b i blue]You:")
                    self._append_user_message(question := input())

                    if method == "ask":
                        self._call_ask()
                    elif method == "bytez":
                        self._call_bytez()
                    elif method == "stream":
                        self._call_stream()
                    else:
                        self._unknown_method(method)

                    if "bye" in question.lower():
                        break

            except API_EXCEPTIONS as e:
                logger.exception("API error")
                raise ApiError(e) from e
            except SYSTEM_EXCEPTIONS as e:
                logger.exception("System error")
                raise AppSystemError(e) from e
            except Exception as e:
                logger.exception("Unknown error")
                raise UnknownError(e) from e

    def _call_ask(self) -> None:
        if not (reply := self._ask()):
            rprint("\n[b i red]No reply.")
            return
        rprint("\n[b i green]AI:")
        rprint(reply, end="\n", flush=True)

    def _call_bytez(self) -> None:
        if not (reply := self._bytez_chat()):
            rprint("\n[b i red]No reply..")
            return
        rprint("\n[b i green]AI:")
        rprint(reply, end="\n", flush=True)

    def _call_stream(self) -> None:
        reply = self._stream_chat()
        iterator = iter(reply)
        try:
            first_token = next(iterator)
        except StopIteration:
            rprint("\n[b i red]No reply..")
            return

        rprint("\n[b i green]AI:")
        rprint(first_token, end="", flush=True)
        for token in iterator:
            rprint(token, end="", flush=True)

    def list_models(self) -> None:
        exclude = "coder", "thinking"
        models = [
            model.id
            for model in self.models.list().data
            if not any(word in model.id for word in exclude)
            # if ":free" in model.id and not any(word in model.id for word in exclude)
        ]
        rprint(models)


def main() -> None:

    kwargs = {
        "inception": {
            "api_key": _api_key("INCEPTION_API_KEY"),
            "base_url": "https://api.inceptionlabs.ai/v1",
            "model": "mercury-2",
        },
        "openrouter": {
            "api_key": _api_key("OPENROUTER_API_KEY"),
            "base_url": "https://openrouter.ai/api/v1",
            "model": "minimax/minimax-m2.5:free",
        },
        "moonshot": {
            "api_key": _api_key("MOONSHOT_API_KEY"),
            "base_url": "https://api.moonshot.ai/v1",
            "model": "kimi-k2-thinking",
        },
        "bytez": {
            "api_key": _api_key("BYTEZ_API_KEY"),
            "base_url": "https://api.bytez.com/models/v2/openai/v1",
            "model": "Qwen/Qwen3-4B",
        },
    }
    client = OpenAiClient(**kwargs["inception"])
    # client.list_models()
    client.run_chat("ask")


if __name__ == "__main__":
    main()
