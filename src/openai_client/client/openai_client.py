from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing, asynccontextmanager
from json import dumps, loads
from logging import getLogger

from anyio import Path
from openai import AsyncOpenAI, RateLimitError
from rich import print as rprint

from openai_client.configs import Providers as Prv
from openai_client.configs import settings
from openai_client.decorators.with_retry import with_retry
from openai_client.errors import (
    API_EXCEPTIONS,
    SYSTEM_EXCEPTIONS,
    ApiError,
    AppSystemError,
    UnknownError,
)
from openai_client.schemas.ai import OpenAIChat
from openai_client.typings import AssistantParam, ListParams
from openai_client.utils import save_to_file

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


class OpenAIClient(AsyncOpenAI):
    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 60.0) -> None:
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.api_key = api_key
        self.configs = {"temperature": 0.5, "model": model, "top_p": 1.0}
        self.messages: ListParams = []

    def __str__(self) -> str:
        return f"model '{self.model}' via {self.base_url}"

    async def list_models(self, provider: str) -> list[str]:
        model_list = await self.models.list()

        lists: list[str] = []
        for model in model_list.data:
            model_id = model.id
            if any(x in model_id for x in ("-coder", "-it", "embed")):
                continue
            # OpenRouter is the only provider where the free tier is worth listing;
            # every other provider keeps its full catalogue.
            if provider == Prv.OPENROUTER and ":free" not in model_id:
                continue
            lists.append(model_id)

        path = Path("models.json")
        models = loads(await path.read_text(encoding="utf-8")) if await path.exists() else {}
        models[provider] = lists
        await save_to_file(dumps(models, indent=4), path)
        return lists

    @staticmethod
    def _cleanup_answer(content: str) -> str:
        answer = content.strip()
        if "</think>" in answer and len(parts := answer.split("</think>", 1)) > 1:
            answer = parts[1].strip()
        return answer

    def call(self, *, append: bool = True) -> dict[str, Callable]:
        return {
            "completion": lambda: self._completion(append=append),
            "stream": lambda: self._iter_chunk(self._stream_chunk()),
        }

    @with_retry(
        max_attempts=settings.AI_MAX_RETRIES,
        base_delay=settings.AI_RETRY_DELAY,
        max_delay=settings.AI_RETRY_DELAY * (settings.AI_BACKOFF_FACTOR**settings.AI_MAX_RETRIES),
        exec_retry=API_EXCEPTIONS,
    )
    async def _completion(self, *, append: bool = True) -> str | None:
        async with _handle_api_call():
            completion = await self.chat.completions.create(
                messages=self.messages,
                **self.configs,
            )

        if not (choices := completion.choices):
            logger.warning("OpenAI returned empty choices")
            return ""
        assistant_message = choices[0].message

        if content := assistant_message.content:
            answer = self._cleanup_answer(content)
            if append:
                self._append_assistant_message(answer)
            return answer

    async def _iter_chunk(self, chunks: AsyncIterator[str | None]) -> str:
        output = ""

        async for token in chunks:
            if token is None:
                return output

            rprint(token, end="", flush=True)
            output += token

        print()
        return output

    async def _stream_chunk(self) -> AsyncGenerator[str | None]:
        async with _handle_api_call():
            stream = await self.chat.completions.create(
                messages=self.messages,
                **self.configs,
                stream=True,
            )

            accumulated = ""
            finished = False
            # Drain the stream after the finish chunk instead of breaking out of it:
            # abandoning the response mid-flight leaves the HTTP generator to be closed
            # reentrantly during loop shutdown ("aclose(): asynchronous generator is
            # already running") and the connection is not released.
            async with aclosing(stream):
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    if chunk.choices[0].finish_reason:
                        finished = True
                        continue
                    if finished:
                        continue

                    if content := chunk.choices[0].delta.content:
                        accumulated += content
                        yield content

            if accumulated:
                self._append_assistant_message(accumulated)
            else:
                yield None

    def _append_assistant_message(self, content: str) -> None:
        assistant_param = OpenAIChat(role="assistant", content=content)
        self.messages.append(AssistantParam(**assistant_param.model_dump()))
