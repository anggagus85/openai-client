from contextlib import suppress
from traceback import format_exc

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
from rich import print as rprint

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


OPENROUTER_API_KEY = dotenv_values(verbose=True)["OPENROUTER_API_KEY"]
MOONSHOT_API_KEY = dotenv_values(verbose=True)["MOONSHOT_API_KEY"]

OPENROUTER_FREE_MODELS = [
    "stepfun/step-3.5-flash:free",
    "arcee-ai/trinity-large-preview:free",
    "upstage/solar-pro-3:free",
    "liquid/lfm-2.5-1.2b-thinking:free",
    "liquid/lfm-2.5-1.2b-instruct:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "arcee-ai/trinity-mini:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "z-ai/glm-4.5-air:free",
    "qwen/qwen3-coder:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "google/gemma-3n-e2b-it:free",
    "google/gemma-3n-e4b-it:free",
    "qwen/qwen3-4b:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "google/gemma-3-4b-it:free",
    "google/gemma-3-12b-it:free",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
]

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

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    # api_key=MOONSHOT_API_KEY,
    # base_url="https://api.moonshot.ai/v1",
)

messages = [
    {
        "role": "system",
        "content": "You are an travel agency expert. You provide users with safe, helpful, and accurate answers. You also reject any questions involving terrorism, racism, pornography, or violence.",
    },
]


def error_var(mssg: str, e: Exception, *, trace: bool = False) -> None:
    """Return the error message."""

    err = f"{type(e).__name__} -> {e}"
    rprint(f"\n[b i red]{mssg}{err}[/b i red]")
    if trace:
        rprint(f"\n[red]{format_exc()}[/red]\n")


def chat() -> str:
    """Supportmulti-turn conversations.

    Each time the chat function is called to converse with the Kimi large language model, it has access to the previous conversation history. In other words, the Kimi large language model has a memory.
    """

    # We engage in a conversation with the Kimi large language model, carrying the messages along
    completion = client.chat.completions.create(
        model="kimi-k2-turbo-preview",
        messages=messages,
        temperature=0.0,
    )

    # Through the API, we obtain the response message (role=assistant) from the Kimi large language model
    assistant_message = completion.choices[0].message

    # To ensure the Kimi large language model has a complete memory, we must also add the message it returns to us to the messages list
    messages.append(assistant_message)
    return assistant_message.content


def run_chat() -> None:
    with suppress(KeyboardInterrupt):
        try:
            while True:
                rprint("\n[b i blue]You:[/b i blue]")
                question = input()
                messages.append({"role": "user", "content": f"{question.strip()}"})
                if not (reply := chat()):
                    rprint("\n[b i red]No reply.[/b i red].")
                    break
                rprint("\n[b i green]AI:[/b i green]")
                rprint(reply)
                if "bye" in question.lower():
                    break

        except API_EXCEPTIONS as e:
            error_var("API error: ", e)
        except SYSTEM_EXCEPTIONS as e:
            error_var("System error: ", e)
        except Exception as e:  # noqa: BLE001
            error_var("Unknown error: ", e, trace=True)


def list_models() -> None:
    model_list = client.models.list()
    model_data = model_list.data

    # free = [model.id for model in model_data if ":free" in model.id]
    # print("Free models:", free)

    kimi_models = [model.id for model in model_data]
    print(kimi)


def main() -> None:
    # list_models()
    run_chat()


if __name__ == "__main__":
    main()
