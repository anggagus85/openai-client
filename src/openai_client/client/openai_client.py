from asyncio import to_thread
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from json import dumps, loads
from logging import getLogger
from pathlib import Path

from bytez import Bytez
from bytez.client import Response
from openai import AsyncOpenAI, RateLimitError
from rich import print as rprint
from rich.pretty import pprint

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

    @with_retry(
        max_attempts=settings.AI_MAX_RETRIES,
        base_delay=settings.AI_RETRY_DELAY,
        max_delay=settings.AI_RETRY_DELAY * (settings.AI_BACKOFF_FACTOR**settings.AI_MAX_RETRIES),
        exec_retry=API_EXCEPTIONS,
    )
    async def _bytez(self, messages: ListParams) -> Response:
        return await to_thread(self._sync_bytez_call, messages)

    def _sync_bytez_call(self, messages: ListParams) -> Response:
        bytez_model = Bytez(self.api_key).model(self.model)
        bytez_params = {"temperature": 0.3}
        result = bytez_model.run(messages, bytez_params)
        assert isinstance(result, Response), f"Expected Response, got {type(result)}"
        return result

    async def list_models(self, provider: str) -> None:
        if provider == "bytez":
            pprint(
                await to_thread(
                    lambda: Bytez(self.api_key).list.models(
                        options={"task": "chat", "modelId": "Qwen3-4B"},
                    ),
                ),
            )
            return

        model_list = await self.models.list()
        model_data = model_list.data

        free = [
            model.id
            for model in model_data
            if ":free" in model.id and not any(x in model.id for x in ["-coder", "-it"])
        ]
        models = loads(await to_thread(lambda: Path("models.json").read_text()))
        models[provider] = free
        await to_thread(save_to_file, dumps(models, indent=4), Path("models.json"))

    @staticmethod
    def _cleanup_answer(content: str) -> str:
        answer = content.strip()
        if "</think>" in answer and len(parts := answer.split("</think>", 1)) > 1:
            answer = parts[1].strip()
        return answer

    def call(self, *, append: bool = True, is_bytez: bool = False) -> dict[str, Callable]:
        return {
            "completion": lambda: self._completion(append=append),
            "bytez": lambda: self._call_bytez(append=append),
            "stream": lambda: self._iter_chunk(self._stream_chunk(), is_bytez=is_bytez),
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

    async def _call_bytez(self, *, append: bool = True) -> str:
        async with _handle_api_call():
            response = await self._bytez(self.messages)

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

        answer = self._cleanup_answer(out_dict["content"])
        if append:
            self._append_assistant_message(answer)
        return answer

    async def _iter_chunk(
        self,
        chunks: AsyncIterator[str | None],
        *,
        is_bytez: bool = False,
    ) -> None:
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

    async def _stream_chunk(self) -> AsyncGenerator[str | None]:
        async with _handle_api_call():
            stream = await self.chat.completions.create(
                messages=self.messages,
                **self.configs,
                stream=True,
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
                self._append_assistant_message(accumulated)
            else:
                yield None

    def _append_assistant_message(self, content: str) -> None:
        assistant_param = OpenAIChat(role="assistant", content=content)
        self.messages.append(AssistantParam(**assistant_param.model_dump()))
