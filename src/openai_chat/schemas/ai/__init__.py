# src/openai_chat/schemas/ai/__init__.py

from openai_chat.schemas.ai.email import AnalysisFormat, ContactAnalysisResponse, EmailInquiry
from openai_chat.schemas.ai.itinerary import (
    ItineraryMD,
    ItineraryRequestMD,
    ItineraryRequestTXT,
    ItineraryTXT,
)

__all__ = [
    "AnalysisFormat",
    "ContactAnalysisResponse",
    "EmailInquiry",
    "ItineraryMD",
    "ItineraryRequestMD",
    "ItineraryRequestTXT",
    "ItineraryTXT",
]
