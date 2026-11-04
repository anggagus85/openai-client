"""Bali itinerary prompt builders and search-query helpers."""

from datetime import datetime
from textwrap import dedent

from openai_client.configs.constants import WHATSAPP_NUMBER
from openai_client.schemas.ai import ItineraryRequest
from openai_client.services.prompts.prompts import humanizer_rules_block
from openai_client.utils.query import fit_query

ITINERARY_SYSTEM_PROMPT = f"You are an expert travel planner specializing in authentic Bali experiences. Prioritize using the latest search results to provide current, accurate information for year {datetime.now().year}."


def _format_date_label(date_str: str) -> str:
    """
    Extract a human-readable ``Month Day`` label from an ISO 8601 date string.

    Args:
        date_str: Date in ``YYYY-MM-DD`` format (e.g., ``"2026-11-10"``).

    Returns:
        Formatted label such as ``"November 10"``, or the original string
        if parsing fails.
    """
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        return f"{dt.strftime('%B')} {dt.day}"  # e.g. "November 10"
    except ValueError:
        return date_str


def _topics(duration: int, dates: list[str]) -> str:
    """
    Itinerary topics analysis.

    Args:
        duration (int): The number of days for the itinerary.
        dates (list[str]): A list of dates for the trip.

    Returns:
        A formatted itinerary topics string.
    """
    # 🚗 **TRANSPORTATION & LOGISTICS**
    # - **1** Getting to/from airport
    # - **2** Daily transportation options (cars, scooters, taxis)
    # - **3** Estimated costs and booking tips

    # Build per-day date labels for the prompt examples
    day1_label = _format_date_label(dates[0]) if dates else "Day 1"
    day2_label = _format_date_label(dates[1]) if len(dates) > 1 else "Day 2"

    # Compact date reference list for the "continue" instruction
    date_refs = ", ".join(f"Day {i + 1} – {_format_date_label(d)}" for i, d in enumerate(dates))

    return dedent(f"""
        🌅 **DAILY BREAKDOWN** (Provide {duration} full days)
        For each day, use this exact format:

        #### **Day 1 – {day1_label} – [Creative Day Title]**
        [A brief explanation of the day's theme (max 30 words)]

        - **Morning Activity (9-11 AM):**
            - [Detailed paragraph describing the activity with specific location name, opening hours (e.g., "open 8 AM-6 PM"), and entry fees (e.g., "entry USD 6")]

        - **Afternoon Activity (12-4 PM):**
            - [Detailed paragraph with lunch at a specific restaurant (e.g., "**Warung Babi Guling Ibu Oka**"). List 2-3 specific dishes with prices (e.g., "**Babi Guling** – USD 12, **Nasi Campur** – USD 8"). Include an activity with location details.]

        - **Evening Activity (5-8 PM):**
            - [Detailed dinner recommendation with restaurant name (e.g., "**Locavore**"), opening hours, reservation notes if needed, and 2-3 signature dishes with prices (e.g., "**Bebek Betutu** – USD 22, **Sate Lilit** – USD 15").]

        - **Evening Wind-down:**
            - [Relaxation activity such as spa treatment with duration and price (e.g., "60 min, USD 45"), or a beverage option with price (e.g., "**Herbal Lemongrass Tea** – USD 2").]

        #### **Day 2 – {day2_label} – [Creative Day Title]**

        [Follow the same detailed format as Day 1]

        Continue for all {duration} days with sequential numbering and dates ({date_refs})

        <requirements>
        Add two related emojis to best portray of the Creative Day Title at the front and back (e.g., #### **Day 1 – November 10 – 🌊 [Creative Day Title] 🌿).
        Each day must include:
            - Creative day titles that reflect the day's theme (e.g., "Arrival & Ubud Introduction", "Temples & Terraces")
            - Specific restaurant names with exact dish recommendations and prices in bold
            - Activity locations with opening hours in parentheses and entry fees where applicable
            - At least 300-400 words per day with rich, descriptive details
        Must use bullet points on new line for each activity (morning, afternoon, evening, wind-down) detailed paragraphs, e.g. :
            - **Morning Activity (9-11 AM):**
                - [Detailed paragraph describing the activity with specific location name, opening hours (e.g., "open 8 AM-6 PM"), and entry fees (e.g., "entry USD 6")]

            - **Afternoon Activity (12-4 PM):**
                - [Detailed paragraph with lunch at a specific restaurant (e.g., "**Warung Babi Guling Ibu Oka**"). List 2-3 specific dishes with prices (e.g., "**Babi Guling** – USD 12, **Nasi Campur** – USD 8"). Include an activity with location details.]

            - **Evening Activity (5-8 PM):**
                - [Detailed dinner recommendation with restaurant name (e.g., "**Locavore**"), opening hours, reservation notes if needed, and 2-3 signature dishes with prices (e.g., "**Bebek Betutu** – USD 22, **Sate Lilit** – USD 15").]

            - **Evening Wind-down:**
                - [Relaxation activity such as spa treatment with duration and price (e.g., "60 min, USD 45"), or a beverage option with price (e.g., "**Herbal Lemongrass Tea** – USD 2").]

        Use exactly one horizontal rule line (______________________________________________________________________) between consecutive days.
        </requirements>

        🏨 **HANDPICKED ACCOMMODATIONS**
        - **[accommodation recommendations]:**
            - [accommodation details for recommendation, price range, and why they suit the traveler's interests]

        <requirements>
        Include 2-3 hotel/villa recommendations:
            - 2-3 specific hotel/villa recommendations with exact price ranges per night
            - Location details and why they suit the traveler's interests
            - Include both mid-range and premium options within budget
        Must use bullet points on new line for accommodation details recommendations.
        </requirements>

        🍜 **CULINARY EXPERIENCES**
        - **[topics]:**
            - [culinary experiences details]

        <requirements>
        Topics:
            - Daily restaurant recommendations for breakfast, lunch, dinner with specific dish names
            - Must-try Balinese dishes with local specialties and descriptions
            - Food market or cooking class suggestions with pricing
            - Beverage recommendations (non-alcoholic where appropriate) with prices
        Culinary experiences details must be in bullet points on a new line.
        </requirements>

        🎭 **CULTURAL IMMERSION**
        - **[immersive experiences]:**
            - [cultural immersive experiences details]

        <requirements>
        Include 4-5 immersive experiences:
            - Local customs and etiquette tips specific to Bali
            - Temple visit protocols and dress code requirements
            - Respectful photography guidelines for sacred sites
        Must use bullet points on new line for cultural immersive experiences details.
        </requirements>

        ⭐ **AUTHENTIC EXPERIENCES**
        - **[experiences]:**
            - [authentic experiences details]

        <requirements>
        Experiences:
            - Unique local experiences not found in guidebooks with specific locations
            - Behind-the-scenes access opportunities with booking details
            - Meet locals and community interactions with timing information
            - Hidden gems based on their specific interests with access instructions
        Must use bullet points on new line for authentic experiences details.
        </requirements>

        💸 **COMPREHENSIVE BUDGET BREAKDOWN**
        <requirements>
        - Use a markdown table with columns: Category | Daily Avg. | {duration}-Day Total
        - Include: Accommodation, Meals & Drinks, Activities & Entrance Fees, Transfers & Transport, Spa & Wellness, Miscellaneous
        - Total estimated cost vs. declared budget with buffer amount
        - Cost-saving tips and premium upgrade options with specific prices
        - Comprehensive budget breakdown details must be in bullet points on a new line
        </requirements>

        ⚡ **PRACTICAL INFORMATION**
        - **[informations]:**
            - [practical informations details]

        <requirements>
        Informations:
            - Best times to visit featured locations with seasonal advice
            - Weather considerations by season with temperature ranges
            - Emergency contacts including local police, medical, and embassy numbers
            - Any other practical information that the traveler may need to know, limit to 2 extra informations.
        Must use bullet points on new line for practical informations details.
        </requirements>
""").strip()


def _section_separators() -> str:

    return dedent("""
    Wrap each section headers with exactly 2 horizontal rule lines, follow pattern strictly:

        ____________________________________
        ### 🌅 DAILY BREAKDOWN
        ____________________________________

        [daily breakdown content]

        ____________________________________
        ### 🏨 HANDPICKED ACCOMMODATIONS
        ____________________________________

        [handpicked accommodations content]

        ____________________________________
        ### 🍜 CULINARY EXPERIENCES
        ____________________________________

        [culinary experiences content]

        ____________________________________
        ### 🎭 CULTURAL IMMERSION
        ____________________________________

        [cultural immersion content]

        ____________________________________
        ### ⭐ AUTHENTIC EXPERIENCES
        ____________________________________

        [authentic experiences content]

        ____________________________________
        ### 💸 COMPREHENSIVE BUDGET BREAKDOWN
        ____________________________________

        [comprehensive budget breakdown content]

        ____________________________________
        ### ⚡ PRACTICAL INFORMATION
        ____________________________________

        [practical information content]

        ____________________________________
        ## 🌟 A Friendly Note

        [friendly note content]

    """).strip()


def _friendly_notes() -> str:

    return dedent(f"""
        ## 🌟 A Friendly Note
            -[note details]

        <requirements>
        Note details:
            - This itinerary is a great starting point, but remember that details like opening hours and prices can change.
            - Daily activity times are approximate and are meant to serve as a guide. The actual duration of activities may vary depending on factors such as traffic, queue lengths, and your personal preferences. Feel free to adjust the schedule as needed to best enjoy your day.
            - We recommend double-checking before you go! For the most up-to-date information and to customize this plan with one of our experts, please contact us on WhatsApp at {WHATSAPP_NUMBER}.
            - We'd love to help you create the perfect Bali journey.
        Friendly notes details must write exactly the content above, and must be in bullet points on a new line.
        </requirements>

    """).strip()


def _structure(duration: int, budget: str) -> str:
    """
    Itinerary structure analysis.

    Args:
        duration: The duration of the itinerary in days.
        budget: The budget for the itinerary.

    Returns:
        A formatted itinerary structure string.
    """

    return "\n".join(
        [
            "<section>",
            f"# 🌴 Here's your {duration}-Days Bali trip Itinerary with {budget} budget 🌴",
            "## [Generated Title Here]",
            "### 🌅 DAILY BREAKDOWN",
            "### 🏨 HANDPICKED ACCOMMODATIONS",
            "### 🍜 CULINARY EXPERIENCES",
            "### 🎭 CULTURAL IMMERSION",
            "### ⭐ AUTHENTIC EXPERIENCES",
            "### 💸 COMPREHENSIVE BUDGET BREAKDOWN",
            "### ⚡ PRACTICAL INFORMATION",
            _friendly_notes(),
            "</section>",
            "",
            "<important_note>",
            "Follow header section strictly",
            _section_separators(),
            "</important_note>",
        ],
    )


def itinerary_prompt(req: ItineraryRequest, search_context: str = "") -> str:
    """
    Create a detailed prompt for itinerary generation.

    Args:
        req: The itinerary request containing destination, duration, and interests.
        search_context: Optional search context to inject into the prompt.

    Returns:
        A formatted prompt string for the AI model.
    """

    interests = ", ".join(req.interests)
    duration = req.duration
    dates = req.dates
    budget = req.budget

    parts = [
        "<task>",
        f"Create a comprehensive and engaging {duration} day(s) travel itinerary for Bali.",
        "</task>",
        "",
        "<traveler_profile>",
        f"🎯 Interests: {interests}",
        f"📅 Duration: {duration} day(s)",
        f"📅 Dates: {' - '.join(dates)}",
        f"💰 Budget: {budget}",
        "</traveler_profile>",
    ]

    if search_context:
        parts.extend(["", search_context.strip()])

    parts.extend(
        [
            "",
            "<title_instruction>",
            "Create a catchy, engaging title based on the traveler profile above.",
            "The title should:",
            "- Capture the essence of the trip in one memorable phrase",
            "- Include 2 relevant emojis opening and closing for best representation of related interests",
            "- Be concise (max 20 words)",
            "Example formats:",
            "- ## **🌊 Beach Bliss & Temple Trails: Your 5-Day Bali Escape 🛕**",
            "- ## **🍛 Food, Soul & Serenity: A Week in Paradise 🧘🏼**",
            "Place this title as a bold subtitle immediately after the main header.",
            "</title_instruction>",
            "",
            "<section_contents>",
            _topics(duration, dates),
            "</section_contents>",
            "",
            "<structure>",
            "Follow this exact markdown structure for your output:",
            _structure(duration, budget),
            "</structure>",
            "",
            "<formatting_rules>",
            "Follow proper markdown structure and headers usage strictly",
            "Use emojis liberally to make content engaging and easy to scan.",
            "The result will be converted and sent via WhatsApp as well, so make sure to use safe emoji that can display properly on WhatsApp.",
            'Insert your creative title immediately after the main "# 🌴 Here\'s your..." header',
            "Make sure the day numbering is correct for the DAILY BREAKDOWN section",
            'For DAILY BREAKDOWN section, use #### (h4) for each day header (e.g., "#### **Day 1: ...**", "#### **Day 2: ...**")',
            "Make sure no trailing spaces before any new line",
            "Put a single newline character at the end of the document",
            "Use bold text for important information (restaurant names, locations, prices)",
            "CRITICAL: Never escape asterisks. Do NOT write \\*\\* anywhere in the output.",
            "CRITICAL: Never open a bold span and then bold another word inside the same sentence.",
            "The following pattern is FORBIDDEN: **Lunch at Warung **Restaurant Name** ... , the correct pattern is: Lunch at **Restaurant Name** (open ...) ... , only bold the proper noun (restaurant name, dish name, location name), not the surrounding descriptive words.",
            "Ensure each day has substantial content (at least 300-400 words each)",
            "Total itinerary should be comprehensive enough for the traveler to execute",
            "Incorporate current information from the latest web search results where relevant",
            "</formatting_rules>",
            "",
            "<writing_rules>",
            humanizer_rules_block(
                ("the itinerary text",),
                extra_rules=(
                    "No forced groups of three, no hedging, no filler openers or closers.",
                    "Keep it concrete: real places, realistic times, and prices from the search results.",
                ),
                formatting_rule=None,
            ),
            "</writing_rules>",
            "",
            "<final_instruction>",
            "Write an itinerary the traveler can actually follow: accurate times, realistic travel gaps, and prices from the search results.",
            "</final_instruction>",
        ],
    )

    return "\n".join(parts)


def search_bali_info(req: ItineraryRequest) -> str:
    interests = ", ".join(req.interests)
    month_year = _trip_month_year(req.dates[0])
    dates = ", ".join(req.dates)
    budget: str = req.budget
    return (
        f"Bali {month_year} travel guide: best hotels villas restaurants, local warungs, "
        f"Bali events/attractions on {dates}, "
        f"top warungs restaurants local cuisine, "
        f"hidden gems secret waterfalls lesser-known temples authentic experiences, "
        f"cooking classes traditional markets wellness retreats, "
        f"{interests} {req.duration} day itinerary recommendations with budget {budget}, "
        f"Possibilities accommodations upgrade based on budget {budget}"
    )


def _trip_month_year(date_str: str) -> str:
    """Return a readable ``Month Year`` label for the first trip date."""
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").strftime("%B %Y")
    except ValueError:
        return f"{date_str} {datetime.now().year}"


def _compose_search_query(
    prefix: str,
    variable: str,
    suffix: str,
    limit: int = 128,
) -> str:
    """Compose a query with a bounded variable and an intact fixed suffix."""
    fixed_prefix = prefix.strip()
    fixed_suffix = suffix.strip()
    remaining = max(0, limit - len(fixed_prefix) - len(fixed_suffix) - 2)
    fitted = fit_query(variable, remaining)
    if not fitted:
        return f"{fixed_prefix} {fixed_suffix}"

    return f"{fixed_prefix} {fitted} {fixed_suffix}"


def search_bali_queries(req: ItineraryRequest) -> list[str]:
    """Build focused queries for providers with short query limits."""
    month_year = _trip_month_year(req.dates[0])
    interests = ", ".join(req.interests)
    budget = req.budget
    queries = [
        _compose_search_query(
            f"Bali {month_year} best hotels villas Ubud Canggu Uluwatu budget",
            budget,
            "lang:en -getyourguide -viator -trip.com",
        ),
        (
            f"Bali {month_year} best warungs local food restaurants Ubud Canggu "
            "lang:en -getyourguide -viator -trip.com -package"
        ),
        (
            f"Bali {month_year} temple ceremonies events calendar hidden gems "
            "lang:en -getyourguide -viator -trip.com -retreat"
        ),
        _compose_search_query(
            f"Bali {month_year} best",
            interests,
            "things to do blog lang:en -getyourguide -viator -trip.com -package",
        ),
        (
            f"Bali {month_year} hidden gems waterfalls temples off beaten path blog "
            "lang:en -getyourguide -viator -trip.com"
        ),
        (
            f"Bali {month_year} cooking class market wellness spa yoga "
            "lang:en -getyourguide -viator -trip.com -package"
        ),
    ]
    return [fit_query(query, 128) for query in queries]


def itinerary_search_messages(search_results: str, req: ItineraryRequest) -> str:
    # Keep this function focused on packaging web results; _prompt builds the shared body.
    search_context = "\n".join(  # noqa: FLY002
        [
            "<latest_web_search_results>",
            "The following is current information about Bali retrieved from web search:",
            "",
            search_results,
            "",
            "Use this information to ensure your itinerary recommendations are up-to-date and relevant.",
            "</latest_web_search_results>",
        ],
    )

    return itinerary_prompt(req=req, search_context=search_context)
