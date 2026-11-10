# src/openai_client/client/openai_client.py

from collections.abc import AsyncGenerator, Callable
from json import dumps, loads
from logging import getLogger
from typing import Any, Literal, cast, overload

from anyio import Path
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk

from openai_client.configs import Providers as Prv
from openai_client.configs import settings
from openai_client.decorators.circuit_breaker import agent_cb
from openai_client.decorators.with_retry import with_retry
from openai_client.errors import AgentNetworkError, raise_ai_error
from openai_client.typings import Stream
from openai_client.utils.async_utils import close_async_stream
from openai_client.utils.files import save_to_file

logger = getLogger(__name__)


class OpenAIClient(AsyncOpenAI):
    """
    OpenAI-compatible client mirroring the production BaliRoam implementation.

    The completion and streaming paths mirror ``OpenAICompatible`` from the
    backend: high reasoning effort and thinking enabled via ``extra_body`` for
    providers that support them, circuit-breaker protection, retry with
    exponential backoff, and explicit stream cleanup.
    """

    is_mistral = False

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        prv: Prv,
        timeout: float = 300.0,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self._closed = False
        self._prv = prv
        self._prv_name = prv.__str__()

    def __str__(self) -> str:
        return f"model '{self.model}' via {self.base_url}"

    def _reasoning_options(self) -> dict[str, Any]:
        """Return reasoning extensions only for providers that accept them."""
        if self._prv == Prv.MISTRAL:
            return {}
        return {
            "reasoning_effort": "high",
            "extra_body": {"thinking": {"type": "enabled"}},
        }

    def _get_completion_func(self, configs: dict[str, Any]) -> Callable:
        request = {**configs, **self._reasoning_options()}
        return lambda: self.chat.completions.create(model=self.model, **request)

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
    ) -> Stream: ...

    async def generate(
        self,
        configs: dict[str, Any],
        *,
        stream: bool = False,
    ) -> Stream | str | None:
        """Execute AI completion with circuit breaker and retry."""
        if stream or configs.get("stream"):
            return await self._generate_streaming({**configs, "stream": True})

        return await self._generate_non_streaming(configs)

    @agent_cb
    @with_retry(
        max_attempts=settings.AI_MAX_RETRIES,
        base_delay=settings.AI_RETRY_DELAY,
        max_delay=settings.AI_RETRY_DELAY * (settings.AI_BACKOFF_FACTOR**settings.AI_MAX_RETRIES),
        exec_retry=AgentNetworkError,
    )
    async def _generate_non_streaming(self, configs: dict[str, Any]) -> str | None:
        """Handle non-streaming completion with circuit breaker and retry."""
        try:
            completion = await self._get_completion_func(configs)()
        except Exception as e:
            logger.exception("AI API call failed")
            raise_ai_error(e)

        if not (choices := completion.choices):
            logger.warning("AI provider returned empty choices for model %s", self.model)
            return

        return choices[0].message.content

    async def _generate_streaming(self, configs: dict[str, Any]) -> Stream:
        """Handle streaming completion with manual circuit breaker management."""
        stream = await agent_cb.call(self._initial_stream_call, configs)
        return cast(Stream, self._wrapped_stream_iterator(stream))

    @with_retry(
        max_attempts=settings.AI_MAX_RETRIES,
        base_delay=settings.AI_RETRY_DELAY,
        max_delay=settings.AI_RETRY_DELAY * (settings.AI_BACKOFF_FACTOR**settings.AI_MAX_RETRIES),
        exec_retry=AgentNetworkError,
    )
    async def _initial_stream_call(
        self,
        configs: dict[str, Any],
    ) -> AsyncStream[ChatCompletionChunk]:
        """Initiate the AI completion stream with retry."""
        try:
            return await self._get_completion_func(configs)()
        except Exception as e:
            logger.exception("AI API stream initiation failed")
            raise_ai_error(e)

    async def _wrapped_stream_iterator(
        self,
        stream: AsyncStream[ChatCompletionChunk],
    ) -> AsyncGenerator[Any]:
        """Wrap the stream to report mid-stream failures to the circuit breaker."""
        try:
            async for chunk in stream:
                yield chunk
            await agent_cb.on_success()
        except Exception as e:
            await agent_cb.on_failure()
            logger.exception("AI mid-stream failure")
            raise_ai_error(e)
        finally:
            await self._close_completion_stream(stream)

    @staticmethod
    async def _close_completion_stream(stream: Stream) -> None:
        """
        Close the SDK stream and the iterator it owns.

        OpenAI's ``AsyncStream.close()`` releases the HTTP response but leaves
        ``AsyncStream._iterator`` suspended. Closing the iterator explicitly
        runs that cleanup in the current task instead of leaving it to the
        async-generator finalizer.
        """
        await close_async_stream(getattr(stream, "_iterator", None))
        await close_async_stream(stream)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        try:
            await super().close()
        except Exception:
            logger.exception("Failed to close AI client")
        else:
            logger.info("AI client closed successfully")

    async def list_models(self, path: Path) -> None:
        model_list = await self.models.list()

        models: list[str] = []
        for model in model_list.data:
            model_id = model.id
            if any(x in model_id for x in ("-coder", "-it", "embed")):
                continue
            # OpenRouter is the only provider where the free tier is worth listing;
            # every other provider keeps its full catalogue.
            if self._prv == Prv.OPENROUTER and ":free" not in model_id:
                continue
            models.append(model_id)

        all_models: dict[str, list[str]] = (
            loads(await path.read_text(encoding="utf-8")) if await path.exists() else {}
        )
        all_models[self._prv_name] = models

        await save_to_file(dumps(all_models, indent=4), path)
