# src/openai_client/typings/typings.py

from mistralai.client import models
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


MistralParams = list[models.ChatCompletionRequestMessage]
MistralAssistant = models.AssistantMessage
MistralUser = models.UserMessage
MistralSystem = models.SystemMessage
