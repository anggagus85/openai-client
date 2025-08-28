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


api_key = dotenv_values(verbose=True)["OPENROUTER_API_KEY"]


client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

messages = [
    {
        "role": "system",
        "content": "You are an artificial intelligence assistant. You provide users with safe, helpful, and accurate answers. You also reject any questions involving terrorism, racism, pornography, or violence.",
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
        model="moonshotai/kimi-k2:free",
        messages=messages,
        # temperature=0.3,
    )

    # Through the API, we obtain the response message (role=assistant) from the Kimi large language model
    assistant_message = completion.choices[0].message

    # To ensure the Kimi large language model has a complete memory, we must also add the message it returns to us to the messages list
    messages.append(assistant_message)
    return assistant_message.content


def main() -> None:
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


if __name__ == "__main__":
    main()
