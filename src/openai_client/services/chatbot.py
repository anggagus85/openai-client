# src/openai_client/services/chatbot.py

"""OpenAI chatbot service mirrored from the production BaliRoam implementation."""

from asyncio import CancelledError, to_thread
from collections.abc import AsyncGenerator
from contextlib import suppress
from json import dumps as json_dumps
from json import loads as json_loads
from logging import getLogger
from typing import Any, Literal, overload

from rich import print as rprint

from openai_client.client import MistralAI, OpenAIClient
from openai_client.configs.constants import MAX_QUERY_LENGTH
from openai_client.schemas.ai import (
    DEFAULT_RESPONSE,
    ChatMessage,
    ChatPart,
    ChatRequest,
    ChatResponse,
)
from openai_client.services.prompts import (
    CHATBOT_SYSTEM_PROMPT,
    _detect_output_injection,
    _is_relevant_to_bali,
    append_system_message,
    append_user_message,
    chat_user_profile,
    humanize_response,
    is_tourism_query,
    sanitize_for_prompt,
    search_context_message,
)
from openai_client.services.web_search import search_bali_context
from openai_client.typings import (
    AssistantParam,
    ListParams,
    MistralAssistant,
    MistralParams,
    MistralSystem,
    SystemParam,
)
from openai_client.utils.async_utils import close_async_stream

logger = getLogger(__name__)

IGNORED = KeyboardInterrupt, CancelledError, InterruptedError, EOFError

_TRAILING_WINDOW = 200
_MAX_HISTORY_LENGTH = 50


class Chatbot:
    """
    Chatbot service using OpenAI-compatible providers.

    The message construction, search augmentation, history mapping, and SSE
    streaming output mirror the production OpenAI chatbot service.
    """

    def __init__(
        self,
        client: OpenAIClient | MistralAI,
        system_param: SystemParam | MistralSystem | None = None,
        *,
        chat_request: ChatRequest | None = None,
        preferred_name: str | None = None,
    ) -> None:
        self._client = client
        self._system_param = system_param
        self._chat_req = chat_request
        self._preferred_name = preferred_name
        self._messages: ListParams | MistralParams = []
        self._query = ""
        self._question = ""
        self._search_results = ""
        self._is_mistral = bool(getattr(client, "is_mistral", False))
        self._configs: dict[str, Any] = {"temperature": 0.5, "top_p": 1.0}
        # Set when the output-injection guard closes the stream itself, so the
        # resulting GeneratorExit is not misreported as a client disconnect.
        self._stream_blocked = False

    def _collect_history(self) -> None:
        """Collect and map sanitized chat history to OpenAI message format."""
        if self._chat_req is None:
            return

        for history in self._chat_req.history:
            if not (content := history.parts[0].text):
                continue

            safe_content = sanitize_for_prompt(content, max_length=2000)
            # Map the frontend's "model" role to OpenAI's "assistant" role.
            if history.role == "model":
                self._append_assistant_message(safe_content)
                continue

            append_user_message(self._messages, safe_content, is_mistral=self._is_mistral)

        self._configs["messages"] = self._messages

    def _append_assistant_message(self, content: str) -> None:
        """Append an assistant message to the message list."""
        if self._is_mistral:
            self._messages.append(MistralAssistant(role="assistant", content=content))
            return
        self._messages.append(AssistantParam(role="assistant", content=content))

    def _system_content(self) -> str:
        if self._system_param is None:
            return CHATBOT_SYSTEM_PROMPT.format(
                context="",
                user_profile=chat_user_profile(self._preferred_name),
            )

        if isinstance(self._system_param, MistralSystem):
            return str(self._system_param.content or "")
        return str(self._system_param["content"])

    @overload
    async def ask(
        self,
        search_context: str = "",
        *,
        stream: Literal[False] = False,
    ) -> ChatResponse: ...

    @overload
    async def ask(
        self,
        search_context: str = "",
        *,
        stream: Literal[True],
    ) -> AsyncGenerator[str]: ...

    async def ask(
        self,
        search_context: str = "",
        *,
        stream: bool = False,
    ) -> ChatResponse | AsyncGenerator[str]:
        """
        Execute the chatbot query.

        Args:
            search_context: The search context to use for the query.
            stream: Whether to stream the response.

        Returns:
            A ChatResponse or an AsyncGenerator of SSE chunks.
        """
        if self._chat_req is None:
            details = "ChatRequest is required before calling ask()"
            raise ValueError(details)

        self._query = sanitize_for_prompt(self._chat_req.query, max_length=MAX_QUERY_LENGTH)
        self._messages = []
        append_system_message(self._messages, self._system_content(), is_mistral=self._is_mistral)
        self._collect_history()
        append_user_message(self._messages, self._query, is_mistral=self._is_mistral)
        self._configs["messages"] = self._messages

        search_ctx_idx = self._build_search_context(search_context)

        if stream:
            logger.info("Running streaming chatbot with %s", self._client)
            return self._stream_chunk(search_ctx_idx)

        logger.info("Running chatbot with %s", self._client)
        answer = await self._client.generate(self._configs)

        self._remove_search_context(search_ctx_idx)

        return ChatResponse(answer=answer or DEFAULT_RESPONSE)

    def _build_search_context(self, search_context: str = "") -> int | None:
        """Insert the temporary search context just before the user message."""
        if not search_context:
            return None

        idx = len(self._messages) - 1
        if self._is_mistral:
            self._messages.insert(idx, MistralSystem(role="system", content=search_context))
        else:
            self._messages.insert(idx, SystemParam(role="system", content=search_context))
        return idx

    def _remove_search_context(self, idx: int | None) -> None:
        """Remove a previously inserted search-context message."""
        if idx is not None and 0 <= idx < len(self._messages):
            self._messages.pop(idx)

    async def _stream_chunk(self, search_ctx_idx: int | None = None) -> AsyncGenerator[str]:
        """Yield SSE-formatted chunks from the provider stream."""
        self._stream_blocked = False
        configs = {**self._configs, "stream": True}
        try:
            stream = await self._client.generate(configs, stream=True)
        except BaseException:
            self._remove_search_context(search_ctx_idx)
            raise

        try:
            async for chunk in stream:
                if not (choices := chunk.choices):
                    continue

                if getattr(choices[0], "finish_reason", None):
                    break

                content = None
                choice = choices[0]
                if hasattr(choice, "delta") and (delta := choice.delta):
                    content = delta.content
                elif hasattr(choice, "message") and (message := choice.message):
                    content = message.content

                if content:
                    yield f"data: {json_dumps(content)}\n\n"
        except GeneratorExit:
            if not getattr(self, "_stream_blocked", False):
                logger.warning("Client disconnected during streaming")
            raise
        finally:
            self._remove_search_context(search_ctx_idx)
            await close_async_stream(stream)

        yield "data: [DONE]\n\n"

    @staticmethod
    def _extract_sse_text(chunk: str) -> str:
        r"""Extract the raw text payload from an SSE ``data: "..."\n\n`` chunk."""
        if not chunk.startswith("data: "):
            return chunk
        json_part = chunk.removeprefix("data: ").rstrip("\n")
        if json_part == "[DONE]":
            return ""
        try:
            return json_loads(json_part)
        except (ValueError, TypeError):
            return json_part

    async def _injection_safe_stream(
        self,
        gen: AsyncGenerator[str],
        first_chunk: str | None,
    ) -> AsyncGenerator[str]:
        """
        Scan streamed SSE chunks for output injection across chunk boundaries.

        Maintains a trailing window of the last N characters of raw text to
        catch patterns that span chunk boundaries. Stops the stream
        immediately (emitting an error SSE) when a match is found.
        """
        tail = ""
        try:
            if first_chunk is not None:
                tail = self._extract_sse_text(first_chunk)
                if _detect_output_injection(tail):
                    self._stream_blocked = True
                    await close_async_stream(gen)
                    yield "data: [ERROR] Response blocked.\n\n"
                    return
                yield first_chunk

            async for chunk in gen:
                text = self._extract_sse_text(chunk)
                if not text:
                    yield chunk
                    continue

                window = (tail + text)[-_TRAILING_WINDOW:]
                if _detect_output_injection(window):
                    self._stream_blocked = True
                    await close_async_stream(gen)
                    yield "data: [ERROR] Response blocked.\n\n"
                    return

                tail = window
                yield chunk

        finally:
            await close_async_stream(gen)

    async def safe_stream(self, search_context: str = "") -> AsyncGenerator[str]:
        """Wrap the provider stream with production output-injection scanning."""
        gen = await self.ask(search_context, stream=True)
        try:
            first_chunk = await anext(gen)
        except StopAsyncIteration:
            await close_async_stream(gen)
            return
        except BaseException:
            await close_async_stream(gen)
            raise

        async for chunk in self._injection_safe_stream(gen, first_chunk):
            yield chunk

    # ------------------------------------------------------------------
    # Interactive CLI harness
    # ------------------------------------------------------------------

    async def run(self, method: str) -> None:
        """Run the interactive terminal chatbot."""
        if method not in {"completion", "stream"}:
            details = f"Unknown method: {method}"
            raise ValueError(details)

        history: list[ChatMessage] = []

        with suppress(*IGNORED):
            while True:
                rprint("\n[b i blue]You:")
                question = await to_thread(input)
                if not question.strip():
                    continue

                if len(question) > MAX_QUERY_LENGTH:
                    rprint(
                        f"[b i red]Message is too long (maximum {MAX_QUERY_LENGTH} "
                        "characters). Please shorten it.[/b i red]",
                    )
                    continue

                self._question = question
                self._chat_req = self._build_chat_request(question, history)
                self._query = sanitize_for_prompt(question, max_length=MAX_QUERY_LENGTH)
                self._search_results = await self._augment_with_search(question) or ""

                rprint("\n[b i green]AI:")
                answer = await self._respond(method)

                history.extend(
                    [
                        ChatMessage(role="user", parts=[ChatPart(text=question)]),
                        ChatMessage(role="model", parts=[ChatPart(text=answer)]),
                    ],
                )

                if "bye" in self._question.lower():
                    break

    @staticmethod
    def _build_chat_request(question: str, history: list[ChatMessage]) -> ChatRequest:
        """Build a request bounded to the most recent 50 history messages."""
        return ChatRequest(query=question, history=history[-_MAX_HISTORY_LENGTH:])

    async def _respond(self, method: str) -> str:
        if method == "stream":
            stream = self.safe_stream(self._search_results)
            output = ""
            done = False
            try:
                async for chunk in stream:
                    if not chunk.startswith("data: "):
                        continue
                    payload = chunk.removeprefix("data: ").strip()
                    if payload == "[DONE]":
                        done = True
                        break
                    if payload.startswith("[ERROR]"):
                        rprint(f"[b i red]{payload}[/b i red]")
                        return output or payload
                    try:
                        text = json_loads(payload)
                    except (ValueError, TypeError):
                        continue
                    output += text
                    rprint(text, end="", flush=True)
            finally:
                await close_async_stream(stream)
                if not done:
                    logger.warning("Client disconnected during streaming")
            print()
            return output

        response = await self.ask(self._search_results, stream=False)
        answer = self._safe_response(response.answer)
        rprint(answer, end="\n", flush=True)
        return answer

    @staticmethod
    def _safe_response(answer: str) -> str:
        """Centralized output-injection gate for non-streaming replies."""
        cleaned = humanize_response(answer)
        if _detect_output_injection(cleaned):
            return "I'm sorry, I encountered an unexpected issue. Please try again."
        return cleaned

    async def _augment_with_search(self, query: str) -> str | None:
        """Build the production search-context block for tourism queries."""
        if not is_tourism_query(query):
            return None

        rprint("[dim]Searching for latest information...[/dim]")
        configs: dict[str, Any] = {
            "num_results": 2,
            "contents": {"highlights": {"max_characters": 3000}},
        }
        if not (results := await search_bali_context(query, "Chat Bot Agent", configs)):
            return None

        if not _is_relevant_to_bali(results):
            return None

        return search_context_message(results)
