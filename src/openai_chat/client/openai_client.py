from collections.abc import Generator
from logging import getLogger
from typing import Any

from bytez import Bytez
from bytez.client import Response
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from openai_chat.utils import get_api_key

logger = getLogger(__name__)


class OpenAiClient(OpenAI):
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        super().__init__(api_key=api_key, base_url=base_url)
        self._model = model
        self._bytez_model = Bytez(get_api_key("BYTEZ_API_KEY")).model(model)
        self._bytez_params = {"temperature": 0.3}
        self._configs: dict[str, Any] = {
            "temperature": 0.5,
            "timeout": 60,
            "reasoning_effort": None,
            "model": self._model,
        }

    def open_ai(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        stream: bool = False,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """
        Support multi-turn conversations.

        Each time the chat function is called to converse with the large language model, it has access to the previous conversation history. In other words, the large language model has a memory.
        """
        return self.chat.completions.create(messages=messages, **self._configs, stream=stream)

    def bytez(self, messages: list[ChatCompletionMessageParam]) -> Generator | Response:

        return self._bytez_model.run(messages, params=self._bytez_params)
