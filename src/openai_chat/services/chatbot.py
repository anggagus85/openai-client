from collections.abc import Generator, Iterator
from contextlib import suppress
from logging import getLogger
from typing import cast

from bytez.client import Response
from openai import Stream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from rich import print as rprint

from openai_chat.client import OpenAiClient
from openai_chat.errors import (
    API_EXCEPTIONS,
    SYSTEM_EXCEPTIONS,
    ApiError,
    AppSystemError,
    UnknownError,
)

logger = getLogger(__name__)


def _system_message() -> list[ChatCompletionMessageParam]:
    return [
        ChatCompletionSystemMessageParam(
            role="system",
            content="""
    <assignment>
    Your name is "Made", you are the official BaliBlissed Concierge, a friendly and expert customer service assistant dedicated to crafting the perfect Balinese getaway. 🌴🥥✨
    </assignment>

    <personality_&_style>
    - Introduction : Introduce yourself as BaliBlissed Concierge.
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
    ]


def chatbot(client: OpenAiClient, method: str) -> None:
    messages = _system_message()
    _run_chat(client, method, messages)


def _append_user_message(question: str, messages: list[ChatCompletionMessageParam]) -> None:
    user_param = ChatCompletionUserMessageParam(
        role="user",
        content=question.strip(),
    )
    messages.append(user_param)


def _run_chat(
    client: OpenAiClient,
    method: str,
    messages: list[ChatCompletionMessageParam],
) -> None:

    with suppress(KeyboardInterrupt):
        try:
            while True:
                rprint("\n[b i blue]You:")
                _append_user_message(question := input(), messages)

                if method == "ask":
                    _call_open_ai(client, messages)
                elif method == "bytez":
                    _call_bytez(client, messages)
                elif method == "stream":
                    _iter_chunk(_stream_chunk(client, messages))

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


def _call_open_ai(client: OpenAiClient, messages: list[ChatCompletionMessageParam]) -> None:
    completion = client.open_ai(messages)

    # Through the API, we obtain the response message (role=assistant) from the large language model
    assistant_message = cast(ChatCompletion, completion).choices[0].message

    # To ensure the language model has a complete memory,
    # add the message it returns to us to the messages list
    assistant_param = ChatCompletionAssistantMessageParam(**assistant_message.model_dump())
    messages.append(assistant_param)

    if content := assistant_message.content:
        rprint("\n[b i green]AI:")
        rprint(content, end="\n", flush=True)
        return

    rprint("\n[b i red]No reply.")


def _call_bytez(client: OpenAiClient, messages: list[ChatCompletionMessageParam]) -> None:
    response = client.bytez(messages)

    if not isinstance(response, Response):
        rprint("\n[b i red]No reply..")
        details = f"Invalid response type: {type(response)}"
        logger.error(details)
        raise AppSystemError(details)

    if error := response.error:
        rprint("\n[b i red]No reply..")
        raise ApiError(error)

    if not isinstance(out_dict := response.output, dict):
        rprint("\n[b i red]No reply..")
        details = f"Invalid output type: {type(out_dict)}"
        logger.error(details)
        raise AppSystemError(details)

    if "content" not in out_dict:
        rprint("\n[b i red]No reply..")
        details = "Response missing 'content' key"
        logger.error(details)
        raise AppSystemError(details)

    assistant_param = ChatCompletionAssistantMessageParam(
        role="assistant",
        content=out_dict["content"],
    )
    messages.append(assistant_param)

    content = out_dict["content"]
    answer = content.split("</think>", 1)[1].strip() if "</think>" in content else content.strip()
    rprint("\n[b i green]AI:")
    rprint(answer, end="\n", flush=True)


def _stream_chunk(client: OpenAiClient, messages: list[ChatCompletionMessageParam]) -> Generator:
    completion = cast(Stream[ChatCompletionChunk], client.open_ai(messages, stream=True))

    accumulated = ""
    for chunk in completion:
        if (choices := chunk.choices) and (delta := choices[0].delta.content):
            accumulated += delta
            yield delta

    # Append assembled assistant message to conversation history
    assistant_param = ChatCompletionAssistantMessageParam(
        role="assistant",
        content=accumulated,
    )
    messages.append(assistant_param)


def _iter_chunk(chunks: Iterator[str]) -> None:
    try:
        first_token = next(chunks)
    except StopIteration:
        rprint("\n[b i red]No reply..")
        return

    rprint("\n[b i green]AI:")
    rprint(first_token, end="", flush=True)
    for token in chunks:
        rprint(token, end="", flush=True)
    print()
