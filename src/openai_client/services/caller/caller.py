from collections.abc import Callable, Generator, Iterator
from logging import ERROR, getLogger
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

getLogger("httpx").setLevel(ERROR)


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


def caller(
    client: OpenAiClient,
    messages: list[ChatCompletionMessageParam],
    *,
    append: bool = True,
    is_bytez: bool = False,
) -> dict[str, Callable[..., str | None]]:
    return {
        "open_ai": lambda: _call_open_ai(client, messages, append=append),
        "bytez": lambda: _call_bytez(client, messages, append=append),
        "stream": lambda: _iter_chunk(_stream_chunk(client, messages), is_bytez=is_bytez),
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


def _iter_chunk(chunks: Iterator[str | None], *, is_bytez: bool = False) -> None:
    """
    Iterate over streamed tokens and print them to the console.

    Filters out any content before the closing ``</think>`` tag, which is commonly
    included in Bytez API responses as part of their internal ``<think> ... </think>``
    reasoning block. Only the actual answer (content after ``</think>``) is printed.

    Args:
        chunks: An iterator yielding streamed token strings from the API.
        is_bytez: Whether the API is Bytez.
    """
    buffer = ""
    started = False
    skipping_leading = False

    for token in chunks:
        if token is None:
            return

        current_token: str = token

        if not is_bytez:
            rprint(current_token, end="", flush=True)
            continue

        if not started:
            # Accumulate tokens until the closing ``</think>`` tag is found
            buffer += token
            if "</think>" not in buffer:
                continue
            # Extract content after the last ``</think>`` occurrence
            buffer = buffer.split("</think>")[-1]
            started = True
            skipping_leading = True
            current_token = buffer  # Use remaining buffer content as current token

        if skipping_leading:
            # Skip leading whitespace/newlines immediately after ``</think>``
            if not (current_token := current_token.lstrip()):
                continue  # Still whitespace, skip to next token
            skipping_leading = False

        rprint(current_token, end="", flush=True)

    print()


def _stream_chunk(client: OpenAiClient, messages: list[ChatCompletionMessageParam]) -> Generator:
    stream = cast(Stream[ChatCompletionChunk], client.open_ai(messages, stream=True))

    accumulated = ""
    for chunk in stream:
        if not chunk.choices:
            continue
        if chunk.choices[0].finish_reason:
            break

        if (choices := chunk.choices) and (content := choices[0].delta.content):
            accumulated += content
            yield content

    if accumulated:  # Only append if we received content
        _append_assistant_message(messages, accumulated)  # for chatbot memory context
    else:
        yield None


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
