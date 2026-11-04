# src/openai_client/schemas/ai/__init__.py

from openai_client.schemas.ai.chatbot import (
    DEFAULT_RESPONSE,
    ChatMessage,
    ChatPart,
    ChatRequest,
    ChatResponse,
)
from openai_client.schemas.ai.email import (
    DISPOSABLE_DOMAINS,
    AnalysisFormat,
    ContactAnalysisResponse,
    EmailInquiry,
    EmailResponse,
)
from openai_client.schemas.ai.itinerary import (
    ItineraryMD,
    ItineraryRequest,
    ItineraryRequestTXT,
    ItineraryResponse,
    ItineraryTXT,
)

__all__ = [
    "AnalysisFormat",
    "ChatMessage",
    "ChatPart",
    "ChatRequest",
    "ChatResponse",
    "ContactAnalysisResponse",
    "DEFAULT_RESPONSE",
    "DISPOSABLE_DOMAINS",
    "EmailInquiry",
    "EmailResponse",
    "ItineraryMD",
    "ItineraryRequest",
    "ItineraryRequestTXT",
    "ItineraryResponse",
    "ItineraryTXT",
]
