from collections.abc import Generator
from json import dumps, loads
from logging import getLogger
from pathlib import Path

from bytez import Bytez
from bytez.client import Response
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from rich.pretty import pprint

from openai_client.configs import settings
from openai_client.decorators.with_retry import with_retry
from openai_client.errors import API_EXCEPTIONS
from openai_client.types import ListParams
from openai_client.utils import save_to_file

logger = getLogger(__name__)


class OpenAiClient(OpenAI):
    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 60.0) -> None:
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self._bytes = Bytez(api_key)
        self._bytez_model = self._bytes.model(model)
        self._bytez_params = {"temperature": 0.3}
        self._configs = {"temperature": 0.5, "model": model, "top_p": 1.0}

    @with_retry(
        max_attempts=settings.AI_MAX_RETRIES,
        base_delay=settings.AI_RETRY_DELAY,
        max_delay=settings.AI_RETRY_DELAY * (settings.AI_BACKOFF_FACTOR**settings.AI_MAX_RETRIES),
        exec_retry=API_EXCEPTIONS,
    )
    def completion(
        self,
        messages: ListParams,
        *,
        stream: bool = False,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """
        Support multi-turn conversations.

        Each time the chat function is called to converse with the large language model, it has access to the previous conversation history. In other words, the large language model has a memory.
        """
        return self.chat.completions.create(
            messages=messages,
            **self._configs,
            stream=stream,
        )

    @with_retry(
        max_attempts=settings.AI_MAX_RETRIES,
        base_delay=settings.AI_RETRY_DELAY,
        max_delay=settings.AI_RETRY_DELAY * (settings.AI_BACKOFF_FACTOR**settings.AI_MAX_RETRIES),
        exec_retry=API_EXCEPTIONS,
    )
    def bytez(self, messages: ListParams) -> Generator | Response:

        return self._bytez_model.run(messages, self._bytez_params)

    def list_models(self, provider: str) -> None:
        if provider == "bytez":
            pprint(self._bytes.list.models(options={"task": "chat", "modelId": "Qwen3-4B"}))
            return

        model_list = self.models.list()
        model_data = model_list.data

        free = [
            model.id
            for model in model_data
            if ":free" in model.id and not any(x in model.id for x in ["-coder", "-it"])
        ]
        models = loads(Path("models.json").read_text())
        models[provider] = free
        save_to_file(dumps(models, indent=4), Path("models.json"))

    def __str__(self) -> str:
        return f"model '{self.model}' via {self.base_url}"
