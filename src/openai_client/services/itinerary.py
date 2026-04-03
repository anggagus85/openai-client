# src/openai_client/services/itinerary.py

from logging import getLogger
from pathlib import Path

from mdformat import text as mdformat_text
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from openai_client.client import OpenAiClient
from openai_client.schemas.ai import ItineraryRequestMD
from openai_client.services.caller import methods
from openai_client.utils import save_to_file

logger = getLogger(__name__)

WHATSAPP_NUMBER = "+6285847006743"


def _topics(duration: int) -> str:
    """
    Itinerary topics analysis.

    Args:
        duration: The duration of the itinerary.

    Returns:
        A formatted itinerary topics string.
    """

    # 🚗 **TRANSPORTATION & LOGISTICS**
    # - **1** Getting to/from airport
    # - **2** Daily transportation options (cars, scooters, taxis)
    # - **3** Estimated costs and booking tips

    return f"""
    🌅 **DAILY BREAKDOWN** (Provide {duration} full days)
    For each day, include:
    - - **Morning Activity :** (9-11 AM) with specific locations and times
    - - **Afternoon Activity :** (12-4 PM) with lunch suggestions
    - - **Evening Activity :** (5-8 PM) with dinner recommendations
    - - **Evening Wind-down :** (relaxation/beverages options)

    🏨 **HANDPICKED ACCOMMODATIONS**
    - **1** 2-3 specific hotel/villa recommendations with price ranges
    - **2** Location details and why they suit the traveler's interests
    - **3** Include both mid-range and premium options within budget

    🍜 **CULINARY EXPERIENCES**
    - **1** Daily restaurant recommendations for breakfast, lunch, dinner
    - **2** Must-try Balinese dishes with local specialties
    - **3** Food market or cooking class suggestions
    - **4** Beverage recommendations (non-alcoholic where appropriate)

    🎭 **CULTURAL IMMERSION**
    - **1** Local customs and etiquette tips
    - **2** Temple visit protocols if applicable
    - **3** Respectful photography guidelines

    ⭐ **AUTHENTIC EXPERIENCES**
    - **1** Unique local experiences not found in guidebooks
    - **2** Behind-the-scenes access opportunities
    - **3** Meet locals and community interactions
    - **4** Hidden gems based on their specific interests

    💸 **COMPREHENSIVE BUDGET BREAKDOWN**
    - **1** Daily spending estimates
    - **2** Total estimated cost vs. declared budget
    - **3** Cost-saving tips and premium upgrade options

    ⚡ **PRACTICAL INFORMATION**
    - **1** Best times to visit featured locations
    - **2** Weather considerations by season
    - **3** Emergency contacts
    """


def _structure(duration: int, budget: str) -> str:
    """
    Itinerary structure analysis.

    Returns:
        A formatted itinerary structure string.
    """

    friendly_note = f"""
    ## 🌟 A Friendly Note

    This itinerary is a great starting point, but remember that details like opening hours and prices can change.
    We recommend double-checking before you go! For the most up-to-date information and to customize this plan with one of our experts,
    please contact us on WhatsApp at {WHATSAPP_NUMBER}. We'd love to help you create the perfect Bali journey
    """

    return f"""
    # 🌴 Here's your {duration}-Days Bali trip Itinerary with {budget} budget 🌴
    [Generated Title Here]
    ### 🌅 DAILY BREAKDOWN
    ## 🏨 HANDPICKED ACCOMMODATIONS
    ## 🍜 CULINARY EXPERIENCES
    ## 🎭 CULTURAL IMMERSION
    ## ⭐ AUTHENTIC EXPERIENCES
    ## 💸 COMPREHENSIVE BUDGET BREAKDOWN
    ## ⚡ PRACTICAL INFORMATION
    {friendly_note}
    """


def _prompt(request: ItineraryRequestMD) -> str:
    """
    Create a detailed prompt for itinerary generation.

    Args:
        request: The itinerary request containing destination, duration, and interests.

    Returns:
        A formatted prompt string for the AI model.
    """

    interests = ", ".join(request.interests)
    duration = request.duration
    budget = request.budget

    return f"""
    Create a comprehensive and engaging {duration} day(s) travel itinerary for Bali.

    <traveler_profile>
    🎯 Interests: {interests}
    📅 Duration: {duration} day(s)
    💰 Budget: {budget}
    </traveler_profile>

    <title_instruction>
    Create a catchy, engaging title based on the traveler profile above.
    The title should:
    - Capture the essence of the trip in one memorable phrase
    - Include relevant emojis
    - Be concise (max 20 words)
    Example formats:
    - "## **🌴 Beach Bliss & Temple Trails: Your 5-Day Bali Escape 🌴**"
    - "## **🌊 Surf, Soul & Serenity: A Week in Paradise 🌊**"
    Place this title as a bold subtitle immediately after the main header.
    </title_instruction>

    <content_topics>
    {_topics(duration)}
    </content_topics>

    <structure>
    Follow this exact markdown structure for your output:
    {_structure(duration, budget)}
    </structure>

    <formatting_rules>
    - Follow proper markdown structure and headers usage
    - Use emojis liberally to make content engaging and easy to scan
    - Insert your creative title immediately after the main "# 🌴 Here's your..." header
    - For DAILY BREAKDOWN section, use #### (h4) for each day header (e.g., "#### **Day 1: ...**", "#### **Day 2: ...**")
    - Make sure no trailing spaces before any new line
    - Put a single newline character at the end of the document
    - Use bold text for important information (restaurant names, locations, prices)
    - Ensure each day has substantial content (at least 300-400 words each)
    - Total itinerary should be comprehensive enough for the traveler to execute
    </formatting_rules>

    Create a memorable, practical, and culturally rich Bali itinerary that exceeds expectations!
    """


def _clean_markdown(text: str) -> str:
    """
    Format raw Markdown text to be CommonMark/GFM compliant.

    Useful for standardizing AI outputs before sending to frontend.
    """
    try:
        return mdformat_text(
            text,
            extensions={"gfm"},  # 1. Enable Plugins: Explicitly list extensions that installed
            options={  # 2. Options: Customize how the text is rendered
                "wrap": "no",  # 'no' is best for Frontends (let CSS handle wrapping)
                "number": True,  # Use ordered numbering (1. 2. 3.) instead of auto (1. 1. 1.)
                "end_of_line": "lf",  # Use Unix line endings (LF)
            },
        )
    except Exception:
        # Fallback: If formatting fails (rare), return original text
        # so the user still gets their answer.
        logger.exception("Markdown formatting failed")
        return text


def _itinerary_messages(itinerary_req: ItineraryRequestMD) -> list[ChatCompletionMessageParam]:
    return [
        ChatCompletionSystemMessageParam(
            role="system",
            content="You are an expert travel planner specializing in authentic Bali experiences.",
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content=_prompt(itinerary_req),
        ),
    ]


def itinerary(itinerary_req: ItineraryRequestMD, client: OpenAiClient, method: str) -> None:
    logger.info("Generating itinerary...")
    messages = _itinerary_messages(itinerary_req)

    if not (content := methods(client, messages, append=False)[method]()):
        logger.warning("No itinerary content returned from OpenAI method %s", method)
        return

    content = _clean_markdown(content)
    save_to_file(content, Path.cwd() / "ITINERARY.md")
