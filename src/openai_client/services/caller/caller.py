from collections.abc import Callable, Generator, Iterator
from logging import getLogger
from typing import cast

from bytez.client import Response
from openai import Stream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)
from rich import print as rprint

from openai_client.client import OpenAiClient
from openai_client.errors import ApiError, AppSystemError
from openai_client.schemas.ai import OpenAIChatMessage

logger = getLogger(__name__)


def _cleanup_answer(content: str) -> str:
    """
    Cleanup the answer from the response.

    Strip and remove <think> </think> tag if exist

    Args:
        content: Response content

    Returns:
        Stripped answer
    """
    answer = content.strip()
    if "</think>" in answer:
        # bytez api response tend to add `<think> </think>` tag
        answer = answer.split("</think>", 1)[1].strip()
    return answer


def methods(
    client: OpenAiClient,
    messages: list[ChatCompletionMessageParam],
    *,
    append: bool = True,
) -> dict[str, Callable[..., str | None]]:
    return {
        "open_ai": lambda: _call_open_ai(client, messages, append=append),
        "bytez": lambda: _call_bytez(client, messages, append=append),
        "stream": lambda: _iter_chunk(_stream_chunk(client, messages)),
    }


def _call_open_ai(
    client: OpenAiClient,
    messages: list[ChatCompletionMessageParam],
    *,
    append: bool = True,
) -> str:
    completion = client.open_ai(messages)

    # Through the API, we obtain the response message (role=assistant) from the large language model
    if not (choices := cast(ChatCompletion, completion).choices):
        logger.warning("OpenAI returned empty choices")
        return ""
    assistant_message = choices[0].message

    if content := assistant_message.content:
        answer = _cleanup_answer(content)
        if append:  # for chatbot memory context
            _append_assistant_message(messages, answer)
        return answer
    return ""


def _call_bytez(
    client: OpenAiClient,
    messages: list[ChatCompletionMessageParam],
    *,
    append: bool = True,
) -> str:
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

    answer = _cleanup_answer(out_dict["content"])
    if append:  # for chatbot memory context
        _append_assistant_message(messages, answer)
    return answer


def _iter_chunk(chunks: Iterator[str]) -> None:
    try:
        first_token = next(chunks)
    except StopIteration:
        rprint("\n[b i red]No reply..")
        return

    rprint(first_token, end="", flush=True)
    for token in chunks:
        rprint(token, end="", flush=True)
    print()


def _stream_chunk(client: OpenAiClient, messages: list[ChatCompletionMessageParam]) -> Generator:
    completion = cast(Stream[ChatCompletionChunk], client.open_ai(messages, stream=True))

    accumulated = ""
    for chunk in completion:
        if (choices := chunk.choices) and (delta := choices[0].delta.content):
            accumulated += delta
            yield delta

    if accumulated:  # Only append if we received content
        _append_assistant_message(messages, accumulated)  # for chatbot memory context


def _append_assistant_message(messages: list[ChatCompletionMessageParam], content: str) -> None:
    """
    Append assistant message to conversation history for maintain the context/conversation history.

    Args:
        messages: List of chat messages
        content: Assistant message content
    """
    assistant_param = OpenAIChatMessage(
        role="assistant",
        content=content,
    )
    messages.append(ChatCompletionAssistantMessageParam(**assistant_param.model_dump()))
