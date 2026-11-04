# src/openai_client/client/mistral_ai.py

from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from logging import getLogger
from typing import Any, Literal, overload

from mistralai.client import Mistral
from mistralai.client.errors import MistralError
from rich import print as rprint

from openai_client.configs import settings
from openai_client.decorators.with_retry import with_retry
from openai_client.errors import (
    API_EXCEPTIONS,
    SYSTEM_EXCEPTIONS,
    ApiError,
    AppSystemError,
    UnknownError,
)
from openai_client.typings import MistralAssistant, MistralParams
from openai_client.utils.async_utils import close_async_stream

logger = getLogger(__name__)


@asynccontextmanager
async def _handle_api_call() -> AsyncGenerator:
    try:
        yield
    except API_EXCEPTIONS as e:
        logger.exception("API error")
        raise ApiError(e) from e
    except MistralError as e:
        logger.exception("Too many requests, Try again later")
        raise ApiError(e) from e
    except SYSTEM_EXCEPTIONS as e:
        logger.exception("System error")
        raise AppSystemError(e) from e
    except Exception as e:
        logger.exception("Unknown error")
        raise UnknownError(e) from e


class _MistralDelta:
    __slots__ = ("content",)

    def __init__(self, content: str | None) -> None:
        self.content = content


class _MistralChoice:
    __slots__ = ("delta", "finish_reason")

    def __init__(self, choice: object) -> None:
        delta = getattr(choice, "delta", None)
        self.delta = _MistralDelta(getattr(delta, "content", None))
        self.finish_reason = getattr(choice, "finish_reason", None)


class _MistralChunk:
    """Adapter exposing Mistral chunks in the OpenAI chunk shape."""

    __slots__ = ("choices",)

    def __init__(self, chunk: object) -> None:
        data = getattr(chunk, "data", chunk)
        choices = getattr(data, "choices", None) or []
        self.choices = [_MistralChoice(choice) for choice in choices]


class MistralAI(Mistral):
    is_mistral = True

    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        super().__init__(api_key=api_key, timeout_ms=300000)
        self.base_url = base_url
        self._model = model
        self.messages: MistralParams = []
        self.configs = {"temperature": 0.5, "model": model, "top_p": 1.0}

    def __str__(self) -> str:
        return f"model '{self._model}' via {self.base_url}"

    @staticmethod
    def _clean_configs(configs: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in configs.items() if key != "model"}

    @overload
    async def generate(
        self,
        configs: dict[str, Any],
        *,
        stream: Literal[False] = False,
    ) -> str | None: ...

    @overload
    async def generate(
        self,
        configs: dict[str, Any],
        *,
        stream: Literal[True],
    ) -> AsyncGenerator[Any]: ...

    async def generate(
        self,
        configs: dict[str, Any],
        *,
        stream: bool = False,
    ) -> AsyncGenerator[Any] | str | None:
        """Execute a Mistral completion with the production request shape."""
        cleaned = self._clean_configs(configs)
        self.messages = cleaned.get("messages", self.messages)
        if stream or cleaned.get("stream"):
            cleaned["stream"] = True
            return self._generate_streaming(cleaned)
        return await self._generate_non_streaming(cleaned)

    async def _generate_non_streaming(self, configs: dict[str, Any]) -> str | None:
        async with _handle_api_call():
            completion = await self.chat.complete_async(
                model=self._model,
                response_format={"type": "text"},
                **configs,
                reasoning_effort="high",
            )

        if not (choices := completion.choices):
            return
        return choices[0].message.content

    async def _generate_streaming(
        self,
        configs: dict[str, Any],
    ) -> AsyncGenerator[Any]:
        stream: object | None = None
        try:
            async with _handle_api_call():
                stream = await self.chat.stream_async(
                    model=self._model,
                    response_format={"type": "text"},
                    **configs,
                    reasoning_effort="high",
                )
                async for chunk in stream:
                    yield _MistralChunk(chunk)
        finally:
            await close_async_stream(stream)

    async def close(self) -> None:
        exit_sync = getattr(self, "__exit__", None)
        exit_async = getattr(self, "__aexit__", None)
        try:
            if callable(exit_sync):
                exit_sync(None, None, None)
        finally:
            if callable(exit_async):
                await exit_async(None, None, None)

    @with_retry(
        max_attempts=settings.AI_MAX_RETRIES,
        base_delay=settings.AI_RETRY_DELAY,
        max_delay=settings.AI_RETRY_DELAY * (settings.AI_BACKOFF_FACTOR**settings.AI_MAX_RETRIES),
        exec_retry=API_EXCEPTIONS,
    )
    async def _completion(self, *, append: bool = True) -> str | None:
        async with _handle_api_call():
            completion = await self.chat.complete_async(
                messages=self.messages,
                # response_format={"type": "text"},
                **self.configs,
            )
            content = completion.choices[0].message.content
            if not content:
                rprint("\n[b i red]No reply..")
                return
            if append:
                self._append_assistant_message(content)
            return content

    async def _iter_chunk(self, chunks: AsyncIterator[str | None]) -> None:
        async for token in chunks:
            if token is None:
                return
            rprint(token, end="", flush=True)

        print()

    def call(self, *, append: bool = True) -> dict[str, Callable]:
        return {
            "completion": lambda: self._completion(append=append),
            "stream": lambda: self._iter_chunk(self._stream_chunk()),
        }

    async def _stream_chunk(self) -> AsyncGenerator[str | None]:
        async with _handle_api_call():
            stream = await self.chat.stream_async(
                model=self._model,
                messages=self.messages,
                response_format={"type": "text"},
                stream=True,
            )

            accumulated = ""
            async for chunk in stream:
                if content := chunk.data.choices[0].delta.content:
                    accumulated += content
                    yield content

            if accumulated:
                self._append_assistant_message(accumulated)
            else:
                yield None

    def _append_assistant_message(self, content: str) -> None:
        self.messages.append(MistralAssistant(role="assistant", content=content))

    async def list_models(self) -> None:
        list_models_response = await self.models.list_async()
        # rprint(list_models_response)
        for data in list_models_response.data:
            rprint(data)
            break
