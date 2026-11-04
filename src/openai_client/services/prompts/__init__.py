# src/openai_client/services/prompts/__init__.py

from openai_client.services.prompts.chat import (
    CHATBOT_SYSTEM_PROMPT,
    SKIP_PATTERNS,
    TOURISM_PATTERN,
    _detect_output_injection,
    _is_relevant_to_bali,
    is_tourism_query,
    search_context_message,
)
from openai_client.services.prompts.email import (
    EMAIL_SYSTEM_PROMPT,
    confirmation_message_prompt,
    contact_analysis_prompt,
    create_analysis_body,
    create_customer_body,
)
from openai_client.services.prompts.humanizer import humanize_response
from openai_client.services.prompts.itinerary import (
    ITINERARY_SYSTEM_PROMPT,
    itinerary_prompt,
    itinerary_search_messages,
    search_bali_info,
    search_bali_queries,
)
from openai_client.services.prompts.prompts import (
    append_system_message,
    append_user_message,
    chat_user_profile,
    resolve_preferred_name,
    sanitize_for_prompt,
)

__all__ = [
    "CHATBOT_SYSTEM_PROMPT",
    "ITINERARY_SYSTEM_PROMPT",
    "itinerary_prompt",
    "search_bali_info",
    "search_bali_queries",
    "itinerary_search_messages",
    "EMAIL_SYSTEM_PROMPT",
    "contact_analysis_prompt",
    "confirmation_message_prompt",
    "create_analysis_body",
    "create_customer_body",
    "append_system_message",
    "append_user_message",
    "chat_user_profile",
    "resolve_preferred_name",
    "TOURISM_PATTERN",
    "SKIP_PATTERNS",
    "is_tourism_query",
    "search_context_message",
    "humanize_response",
    "_detect_output_injection",
    "_is_relevant_to_bali",
    "sanitize_for_prompt",
]
