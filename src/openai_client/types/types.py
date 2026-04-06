# src/openai_client/types/types.py

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

AssistantParam = ChatCompletionAssistantMessageParam
ListParams = list[ChatCompletionMessageParam]
SystemParam = ChatCompletionSystemMessageParam
UserParam = ChatCompletionUserMessageParam
