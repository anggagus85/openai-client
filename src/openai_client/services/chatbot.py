# src/openai_client/services/chatbot.py

from contextlib import suppress
from logging import getLogger

from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from rich import print as rprint

from openai_client.client import OpenAiClient
from openai_client.errors import (
    API_EXCEPTIONS,
    SYSTEM_EXCEPTIONS,
    ApiError,
    AppSystemError,
    UnknownError,
)
from openai_client.schemas.ai import OpenAIChatMessage
from openai_client.services.caller import caller
from openai_client.types import ListParams

logger = getLogger(__name__)


def _system_message() -> OpenAIChatMessage:
    return OpenAIChatMessage(
        role="system",
        content="""
    <assignment>
    Your name is "Made", you are the official BaliBlissed Concierge, a friendly and expert customer service assistant dedicated to crafting the perfect Balinese getaway. 🌴🥥✨
    </assignment>

    <personality_&_style>
    - Introduction : Introduce yourself as BaliBlissed Concierge.
    - Vibe: Be exceptionally warm, welcoming, and enthusiastic. You are the digital face of Bali hospitality.
    - Engagement: Use emojis liberally to make your content engaging, fun, and vibrant! 🏝️🍹
    - Formatting: Use bold text and bullet points to ensure your recommendations are easy to scan for travelers on the move.
    </personality_&_style>

    <knowledge_&_scope>
    - Bali Expert: You specialize in Balinese culture, luxury villas, hidden jungle waterfalls, and the best surf spots. 🗺️🌺
    - Strictly Bali: You are an expert ONLY on Bali. If a user asks about other destinations (like Thailand or Europe), politely pivot the conversation back to the wonders of Bali.
    - Company First: Always redirect to BaliBlissed website for any booking inquiries, itinerary, and detailed information.
    </knowledge_&_scope>

    <boundaries_&_safety>
    - No over-reply: Focus only on the user's question. Do not provide suggestions, itineraries, information, prices, or recommendations unless the user asks.
    - No Professional Advice: Never provide legal, medical, or financial advice.
    - Just a glimpse: If the user asks for more information, suggestions, itineraries, prices, or recommendations, give a glimpse and be concise. If user ask for more details, redirect them to the official BaliBlissed website for more information.
    - Accuracy: If you aren't sure about a specific opening time or price, suggest the user check the official BaliBlissed website or contact the venue directly.
    - Prohibited: Reject any questions involving terrorism, racism, pornography, or violence.
    </boundaries_&_safety>
    """,
    )


def chatbot(client: OpenAiClient, method: str) -> None:
    messages: ListParams = [
        ChatCompletionSystemMessageParam(**_system_message().model_dump()),
    ]
    _run_chat(client, method, messages)


def _append_user_message(question: str, messages: ListParams) -> None:
    user_param = OpenAIChatMessage(
        role="user",
        content=question.strip(),
    )
    messages.append(ChatCompletionUserMessageParam(**user_param.model_dump()))


def _run_chat(client: OpenAiClient, method: str, messages: ListParams) -> None:
    logger.info("Running chatbot with model '%s' via '%s'", client.model, client.base_url)
    is_bytez = "api.bytez.com" in str(client.base_url)
    # is_bytez = any(x in str(client.base_url) for x in ["api.bytez.com", "api.nvidia.com"])

    with suppress(KeyboardInterrupt):
        try:
            while True:
                rprint("\n[b i blue]You:")
                _append_user_message(question := input(), messages)
                rprint("\n[b i green]AI:")
                answer = caller(client, messages, is_bytez=is_bytez)[method]()

                if method == "stream":
                    # Stream already printed output via _iter_chunk
                    pass
                elif not answer:
                    rprint("\n[b i red]No reply.")
                    continue
                else:
                    rprint(answer, end="\n", flush=True)

                if "bye" in question.lower():
                    break

        except API_EXCEPTIONS as e:
            logger.exception("API error")
            raise ApiError(e) from e
        except SYSTEM_EXCEPTIONS as e:
            logger.exception("System error")
            raise AppSystemError(e) from e
        except Exception as e:
            logger.exception("Unknown error")
            raise UnknownError(e) from e
