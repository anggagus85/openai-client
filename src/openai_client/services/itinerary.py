# src/openai_client/services/itinerary.py

from logging import getLogger
from pathlib import Path

from mdformat import text as mdformat_text
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from openai_client.client import OpenAiClient
from openai_client.schemas.ai import ItineraryRequestMD
from openai_client.services.caller import caller
from openai_client.services.search import search_clients
from openai_client.types import ListParams
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
    For each day, use this exact format:

    #### **Day 1 – [Creative Day Title]**

    - **Morning Activity (9-11 AM):**
        - [Detailed paragraph describing the activity with specific location name, opening hours (e.g., "open 8 AM-6 PM"), and entry fees (e.g., "entry USD 6")]
    - **Afternoon Activity (12-4 PM):**
        - [Detailed paragraph with lunch at a specific restaurant (e.g., "**Warung Babi Guling Ibu Oka**"). List 2-3 specific dishes with prices (e.g., "**Babi Guling** – USD 12, **Nasi Campur** – USD 8"). Include an activity with location details.]
    - **Evening Activity (5-8 PM):**
        - [Detailed dinner recommendation with restaurant name (e.g., "**Locavore**"), opening hours, reservation notes if needed, and 2-3 signature dishes with prices (e.g., "**Bebek Betutu** – USD 22, **Sate Lilit** – USD 15").]
    - **Evening Wind-down:**
        - [Relaxation activity such as spa treatment with duration and price (e.g., "60 min, USD 45"), or a beverage option with price (e.g., "**Herbal Lemongrass Tea** – USD 2").]

    #### **Day 2 – [Creative Day Title]**

    [Follow the same detailed format as Day 1]

    Continue for all {duration} days with sequential numbering (Day 1, Day 2, Day 3, Day 4, Day 5, Day 6, Day 7, etc.)

    <requirements>
    Each day must include:
        - Creative day titles that reflect the day's theme (e.g., "Arrival & Ubud Introduction", "Temples & Terraces")
        - Specific restaurant names with exact dish recommendations and prices in bold
        - Activity locations with opening hours in parentheses and entry fees where applicable
        - At least 300-400 words per day with rich, descriptive details
    All the activities details must be in bullet points
    </requirements>

    🏨 **HANDPICKED ACCOMMODATIONS**
    - **[location]:**
        - [accommodation details for recommendation, price range, and why they suit the traveler's interests]

    <requirements>
    - 2-3 specific hotel/villa recommendations with exact price ranges per night
    - Location details and why they suit the traveler's interests
    - Include both mid-range and premium options within budget
    - Accommodations details must be in bullet points on a new line
    </requirements>

    🍜 **CULINARY EXPERIENCES**
    - **[topics]:**
        - [culinary experiences details]

    <requirements>
    - Daily restaurant recommendations for breakfast, lunch, dinner with specific dish names
    - Must-try Balinese dishes with local specialties and descriptions
    - Food market or cooking class suggestions with pricing
    - Beverage recommendations (non-alcoholic where appropriate) with prices
    - Culinary experiences details must be in bullet points on a new line
    </requirements>

    🎭 **CULTURAL IMMERSION**
    <requirements>
    - Local customs and etiquette tips specific to Bali
    - Temple visit protocols and dress code requirements
    - Respectful photography guidelines for sacred sites
    - Cultural immersion details must be in bullet points on a new line
    </requirements>

    ⭐ **AUTHENTIC EXPERIENCES**
    - **[experiences]:**
        - [authentic experiences details]

    <requirements>
    - Unique local experiences not found in guidebooks with specific locations
    - Behind-the-scenes access opportunities with booking details
    - Meet locals and community interactions with timing information
    - Hidden gems based on their specific interests with access instructions
    - Authentic experiences details must be in bullet points on a new line
    </requirements>

    💸 **COMPREHENSIVE BUDGET BREAKDOWN**
    <requirements>
    - Use a markdown table with columns: Category | Daily Avg. | 5-Day Total
    - Include: Accommodation, Meals & Drinks, Activities & Entrance Fees, Transfers & Transport, Spa & Wellness, Miscellaneous
    - Total estimated cost vs. declared budget with buffer amount
    - Cost-saving tips and premium upgrade options with specific prices
    - Comprehensive budget breakdown details must be in bullet points on a new line
    </requirements>

    ⚡ **PRACTICAL INFORMATION**
    <requirements>
    - Best times to visit featured locations with seasonal advice
    - Weather considerations by season with temperature ranges
    - Emergency contacts including local police, medical, and embassy numbers
    - Practical information details must be in bullet points on a new line
    </requirements>
    """


def _structure(duration: int, budget: str) -> str:
    """
    Itinerary structure analysis.

    Returns:
        A formatted itinerary structure string.
    """

    friendly_note = f"""
    ## 🌟 A Friendly Note

    - This itinerary is a great starting point, but remember that details like opening hours and prices can change.
    - We recommend double-checking before you go! For the most up-to-date information and to customize this plan with one of our experts, please contact us on WhatsApp at {WHATSAPP_NUMBER}.
    - We'd love to help you create the perfect Bali journey.

    <requirements>
    - Friendly note details must be in bullet points on a new line
    </requirements>
    """

    return f"""
    <section>
    # 🌴 Here's your {duration}-Days Bali trip Itinerary with {budget} budget 🌴
    ## [Generated Title Here]
    ### 🌅 DAILY BREAKDOWN
    ### 🏨 HANDPICKED ACCOMMODATIONS
    ### 🍜 CULINARY EXPERIENCES
    ### 🎭 CULTURAL IMMERSION
    ### ⭐ AUTHENTIC EXPERIENCES
    ### 💸 COMPREHENSIVE BUDGET BREAKDOWN
    ### ⚡ PRACTICAL INFORMATION
    {friendly_note}
    </section>

    <important_note>
    - Follow header section strictly
    </important_note>
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
    - Include 2 relevant emojis opening and closing for best representation of related interests
    - Be concise (max 20 words)
    Example formats:
    - ## **🌊 Beach Bliss & Temple Trails: Your 5-Day Bali Escape 🛕**
    - ## **🍛 Food, Soul & Serenity: A Week in Paradise 🧘🏼**
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
    - Follow proper markdown structure and headers usage strictly
    - Use emojis liberally to make content engaging and easy to scan
    - Insert your creative title immediately after the main "# 🌴 Here's your..." header
    - Make sure the day numbering is correct for the DAILY BREAKDOWN section
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
            # 1. Enable Plugins: Explicitly list extensions that installed
            extensions={"gfm", "footnote", "frontmatter"},
            # 2. Options: Customize how the text is rendered
            options={
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


def _itinerary_messages(itinerary_req: ItineraryRequestMD) -> ListParams:
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


def _plain_itinerary(itinerary_req: ItineraryRequestMD, client: OpenAiClient, method: str) -> None:
    logger.info("Generating itinerary with %s", client)
    messages = _itinerary_messages(itinerary_req)

    if not (content := caller(client, messages, append=False)[method]()):
        logger.warning("No itinerary content returned from OpenAI method %s", method)
        return

    _clean_and_save(content)


def _itinerary_with_search_messages(
    itinerary_req: ItineraryRequestMD,
    search_results: str,
) -> ListParams:
    interests = ", ".join(itinerary_req.interests)
    search_context = f"""
    <latest_web_search_results>
    The following is current information about Bali retrieved from web search:

    {search_results}

    Use this information to ensure your itinerary recommendations are up-to-date and relevant.
    </latest_web_search_results>
    """

    prompt = f"""
    Create a comprehensive and engaging {itinerary_req.duration} day(s) travel itinerary for Bali.

    <traveler_profile>
    🎯 Interests: {interests}
    📅 Duration: {itinerary_req.duration} day(s)
    💰 Budget: {itinerary_req.budget}
    </traveler_profile>

    {search_context}

    <title_instruction>
    Create a catchy, engaging title based on the traveler profile above.
    The title should:
    - Capture the essence of the trip in one memorable phrase
    - Include 2 relevant emojis opening and closing for best representation of related interests
    - Be concise (max 20 words)
    Example formats:
    - ## **🌊 Beach Bliss & Temple Trails: Your 5-Day Bali Escape 🛕**
    - ## **🍛 Food, Soul & Serenity: A Week in Paradise 🧘🏼**
    Place this title as a bold subtitle immediately after the main header.
    </title_instruction>

    <content_topics>
    {_topics(itinerary_req.duration)}
    </content_topics>

    <structure>
    Follow this exact markdown structure for your output:
    {_structure(itinerary_req.duration, itinerary_req.budget)}
    </structure>

    <formatting_rules>
    - Follow proper markdown structure and headers usage strictly
    - Use emojis liberally to make content engaging and easy to scan
    - Insert your creative title immediately after the main "# 🌴 Here's your..." header
    - Make sure the day numbering is correct for the DAILY BREAKDOWN section
    - For DAILY BREAKDOWN section, use #### (h4) for each day header (e.g., "#### **Day 1: ...**", "#### **Day 2: ...**")
    - Make sure no trailing spaces before any new line
    - Put a single newline character at the end of the document
    - Use bold text for important information (restaurant names, locations, prices)
    - Ensure each day has substantial content (at least 300-400 words each)
    - Total itinerary should be comprehensive enough for the traveler to execute
    - Incorporate current information from the search results where relevant
    </formatting_rules>

    Create a memorable, practical, and culturally rich Bali itinerary that exceeds expectations!
    """

    return [
        ChatCompletionSystemMessageParam(
            role="system",
            content="You are an expert travel planner specializing in authentic Bali experiences. Use the latest search results to provide current, accurate information.",
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content=prompt,
        ),
    ]


def _search_bali_info(itinerary_req: ItineraryRequestMD) -> str:
    interests = ", ".join(itinerary_req.interests)
    return (
        f"Bali 2026 travel guide: best hotels villas in Ubud Seminyak Canggu Uluwatu, "
        f"top warungs restaurants local cuisine, "
        f"hidden gems secret waterfalls lesser-known temples authentic experiences, "
        f"cooking classes traditional markets wellness retreats, "
        f"{interests} {itinerary_req.duration} day itinerary recommendations"
    )


def itinerary(
    itinerary_req: ItineraryRequestMD,
    client: OpenAiClient,
    method: str,
) -> None:
    for provider in search_clients():
        logger.info("Searching for latest Bali information with %s", provider)

        success, search_results = provider.search(_search_bali_info(itinerary_req))

        if not success:
            continue

        logger.info("Search completed successfully, generating itinerary with latest information")
        logger.info("Generating itinerary with %s", client)
        messages = _itinerary_with_search_messages(itinerary_req, search_results)

        if not (content := caller(client, messages, append=False)[method]()):
            logger.warning("No itinerary content returned from OpenAI method %s", method)
            return

        _clean_and_save(content)
        break
    else:
        logger.warning("Search returned no results or error, proceeding without search data")
        _plain_itinerary(itinerary_req, client, method)


def _clean_and_save(content: str) -> None:
    content = _clean_markdown(content)
    save_to_file(content, Path.cwd() / "itinerary.md")
