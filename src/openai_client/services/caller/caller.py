from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from logging import getLogger
from typing import cast

from bytez.client import Response
from openai import AsyncStream, RateLimitError
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
)
from rich import print as rprint

from openai_client.client import OpenAiClient
from openai_client.errors import (
    API_EXCEPTIONS,
    SYSTEM_EXCEPTIONS,
    ApiError,
    AppSystemError,
    UnknownError,
)
from openai_client.schemas.ai import OpenAIChatMessage
from openai_client.types import ListParams

logger = getLogger(__name__)


@asynccontextmanager
async def _handle_api_call() -> AsyncGenerator:
    try:
        yield
    except API_EXCEPTIONS as e:
        logger.exception("API error")
        raise ApiError(e) from e
    except RateLimitError as e:
        logger.exception("Too many requests, Try again later")
        raise ApiError(e) from e
    except SYSTEM_EXCEPTIONS as e:
        logger.exception("System error")
        raise AppSystemError(e) from e
    except Exception as e:
        logger.exception("Unknown error")
        raise UnknownError(e) from e


def _cleanup_answer(content: str) -> str:
    answer = content.strip()
    if "</think>" in answer and len(parts := answer.split("</think>", 1)) > 1:
        answer = parts[1].strip()
    return answer


def caller(
    client: OpenAiClient,
    messages: ListParams,
    *,
    append: bool = True,
    is_bytez: bool = False,
) -> dict[str, Callable]:
    return {
        "completion": lambda: _call_completion(client, messages, append=append),
        "bytez": lambda: _call_bytez(client, messages, append=append),
        "stream": lambda: _iter_chunk(_stream_chunk(client, messages), is_bytez=is_bytez),
    }


async def _call_completion(
    client: OpenAiClient,
    messages: ListParams,
    *,
    append: bool = True,
) -> str | None:
    async with _handle_api_call():
        completion = await client.completion(messages)

    if not (choices := cast(ChatCompletion, completion).choices):
        logger.warning("OpenAI returned empty choices")
        return ""
    assistant_message = choices[0].message

    if content := assistant_message.content:
        answer = _cleanup_answer(content)
        if append:
            _append_assistant_message(messages, answer)
        return answer


async def _call_bytez(client: OpenAiClient, messages: ListParams, *, append: bool = True) -> str:
    async with _handle_api_call():
        response = await client.bytez(messages)

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
    if append:
        _append_assistant_message(messages, answer)
    return answer


async def _iter_chunk(chunks: AsyncIterator[str | None], *, is_bytez: bool = False) -> None:
    buffer = ""
    started = False
    skipping_leading = False

    async for token in chunks:
        if token is None:
            return

        current_token: str = token

        if not is_bytez:
            rprint(current_token, end="", flush=True)
            continue

        if not started:
            buffer += token
            if "</think>" not in buffer:
                continue
            buffer = buffer.split("</think>")[-1]
            started = True
            skipping_leading = True
            current_token = buffer

        if skipping_leading:
            if not (current_token := current_token.lstrip()):
                continue
            skipping_leading = False

        rprint(current_token, end="", flush=True)

    print()


async def _stream_chunk(
    client: OpenAiClient,
    messages: ListParams,
) -> AsyncGenerator[str | None]:
    async with _handle_api_call():
        stream = cast(
            AsyncStream[ChatCompletionChunk],
            await client.completion(messages, stream=True),
        )

    accumulated = ""
    async for chunk in stream:
        if not chunk.choices:
            continue
        if chunk.choices[0].finish_reason:
            break

        if (choices := chunk.choices) and (content := choices[0].delta.content):
            accumulated += content
            yield content

    if accumulated:
        _append_assistant_message(messages, accumulated)
    else:
        yield None


def _append_assistant_message(messages: ListParams, content: str) -> None:
    assistant_param = OpenAIChatMessage(role="assistant", content=content)
    messages.append(ChatCompletionAssistantMessageParam(**assistant_param.model_dump()))
