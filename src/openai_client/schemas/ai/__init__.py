# src/openai_client/schemas/ai/__init__.py

from openai_client.schemas.ai.chatbot import ChatResponse, OpenAIChat, OpenAIChatRequest
from openai_client.schemas.ai.email import AnalysisFormat, ContactAnalysisResponse, EmailInquiry
from openai_client.schemas.ai.itinerary import (
    ItineraryMD,
    ItineraryRequestMD,
    ItineraryRequestTXT,
    ItineraryTXT,
)

__all__ = [
    "AnalysisFormat",
    "ContactAnalysisResponse",
    "ChatResponse",
    "EmailInquiry",
    "ItineraryMD",
    "ItineraryRequestMD",
    "ItineraryRequestTXT",
    "ItineraryTXT",
    "OpenAIChat",
    "OpenAIChatRequest",
]
