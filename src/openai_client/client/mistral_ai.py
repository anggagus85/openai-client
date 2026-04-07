# src/openai_client/client/mistral_ai.py

from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from logging import getLogger

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
from openai_client.types import MistralAssistant

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


class MistralAI(Mistral):
    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        super().__init__(api_key=api_key, timeout_ms=60000)
        self.base_url = base_url
        self._model = model
        self.messages = []
        self.configs = {"temperature": 0.5, "model": model, "top_p": 1.0}

    def __str__(self) -> str:
        return f"model '{self._model}' via {self.base_url}"

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
