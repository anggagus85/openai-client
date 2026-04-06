# src/openai_client/services/itinerary.py

from datetime import datetime
from logging import getLogger
from pathlib import Path
from re import compile as re_compile

from mdformat import text as mdformat_text
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from openai_client.client import OpenAiClient
from openai_client.configs import WHATSAPP_NUMBER
from openai_client.schemas.ai import ItineraryRequestMD
from openai_client.services.caller import caller
from openai_client.services.search import providers
from openai_client.types import ListParams
from openai_client.utils import save_to_file

logger = getLogger(__name__)


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


def _prompt(itinerary_req: ItineraryRequestMD, search_context: str = "") -> str:
    """
    Create a detailed prompt for itinerary generation.

    Args:
        itinerary_req: The itinerary request containing destination, duration, and interests.
        search_context: Optional search context to inject into the prompt.

    Returns:
        A formatted prompt string for the AI model.
    """

    interests = ", ".join(itinerary_req.interests)
    duration = itinerary_req.duration
    budget = itinerary_req.budget

    search_block = ""
    if search_context:
        search_block = f"""
    {search_context}
    """

    search_rule = ""
    if search_context:
        search_rule = """
    - Incorporate current information from the search results where relevant"""

    return f"""
    Create a comprehensive and engaging {duration} day(s) travel itinerary for Bali.

    <traveler_profile>
    🎯 Interests: {interests}
    📅 Duration: {duration} day(s)
    💰 Budget: {budget}
    </traveler_profile>
    {search_block}
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
    - Total itinerary should be comprehensive enough for the traveler to execute{search_rule}
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


# ---------------------------------------------------------------------------
# Markdown → WhatsApp text converter
# ---------------------------------------------------------------------------

# Pre-compiled patterns for performance (module-level, compiled once).
_RE_TABLE_SEPARATOR = re_compile(r"^\|[-:\s|]+\|$")
_RE_TABLE_ROW = re_compile(r"^\|(.+)\|$")
_RE_HEADER = re_compile(r"^(#{1,6})\s+(.+)$")
_RE_BOLD = re_compile(r"\*\*(.+?)\*\*")
_RE_ITALIC = re_compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_RE_LINK = re_compile(r"\[([^\]]+)\]\(([^)]+)\)")
_RE_IMAGE = re_compile(r"!\[([^\]]*)\]\([^)]+\)")
_RE_HORIZONTAL_RULE = re_compile(r"^(-{3,}|\*{3,}|_{3,})$")
_RE_MULTIPLE_BLANK_LINES = re_compile(r"\n{3,}")


def _convert_table_to_whatsapp(table_lines: list[str]) -> list[str]:
    """
    Convert a markdown table into WhatsApp-friendly bullet-point rows.

    Expects a list of raw table lines including the header row, separator,
    and data rows.  Returns formatted lines such as:

        • *Category*: Daily Avg. $30 | 5-Day Total $150

    Args:
        table_lines: Consecutive lines that form a markdown table.

    Returns:
        Converted lines suitable for WhatsApp display.
    """
    if len(table_lines) < 2:  # noqa: PLR2004
        return table_lines

    # Extract header names from the first row.
    header_match = _RE_TABLE_ROW.match(table_lines[0])
    if not header_match:
        return table_lines
    headers = [h.strip() for h in header_match.group(1).split("|")]

    result: list[str] = []
    for line in table_lines[1:]:
        # Skip the separator line (e.g. |---|---|---|).
        if _RE_TABLE_SEPARATOR.match(line):
            continue
        row_match = _RE_TABLE_ROW.match(line)
        if not row_match:
            continue
        cells = [c.strip() for c in row_match.group(1).split("|")]

        # First cell becomes the bold label; remaining cells are paired with
        # their header names.
        if not cells:
            continue
        label = cells[0]
        # Strip any existing bold markers from the label for re-wrapping.
        label = label.replace("**", "")
        parts = []
        for i, cell in enumerate(cells[1:], start=1):
            header_name = headers[i] if i < len(headers) else ""
            parts.append(f"{header_name} {cell}".strip())
        detail = " | ".join(parts)
        result.append(f"  • **{label}**: {detail}" if detail else f"  • **{label}**")

    return result


def _markdown_to_whatsapp(md_text: str) -> str:
    """
    Convert CommonMark/GFM markdown into WhatsApp-friendly plain text.

    Transformation rules
    --------------------
    - ``# H1``       → ``*H1*`` (bold, uppercased)
    - ``## H2``–``#### H4`` → ``*H2*`` (bold)
    - ``**bold**``   → ``*bold*``  (WhatsApp bold)
    - ``*italic*``   → ``_italic_``  (WhatsApp italic)
    - ``[text](url)``→ ``text (url)``
    - ``![alt](…)``  → ``alt``  (images stripped to alt text)
    - Markdown tables → bullet-point rows
    - ``---`` / ``***`` → ``─────────────``
    - Multiple blank lines collapsed to one
    - Trailing whitespace stripped per line

    Args:
        md_text: Cleaned markdown string (ideally already passed through
                 ``_clean_markdown``).

    Returns:
        WhatsApp-compatible plain text string.
    """
    lines = md_text.split("\n")
    output_lines: list[str] = []

    # --- pass 1: collect table blocks and convert them ---
    i = 0
    while i < len(lines):
        # Detect start of a table block (line starts with |).
        if lines[i].strip().startswith("|"):
            table_block: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_block.append(lines[i].strip())
                i += 1
            output_lines.extend(_convert_table_to_whatsapp(table_block))
        else:
            output_lines.append(lines[i])
            i += 1

    # --- pass 2: line-by-line transformations ---
    converted: list[str] = []
    for line in output_lines:
        stripped = line.strip()

        # Horizontal rules → unicode line.
        if _RE_HORIZONTAL_RULE.match(stripped):
            converted.append("─────────────")
            continue

        # Headers → WhatsApp bold (H1 also uppercased).
        header_m = _RE_HEADER.match(stripped)
        if header_m:
            level = len(header_m.group(1))
            text = header_m.group(2).strip()
            # Remove inner bold markers that the AI may have added inside headers.
            text = text.replace("**", "")
            if level == 1:
                converted.append(f"*{text.upper()}*")
            else:
                converted.append(f"*{text}*")
            continue

        # Images → alt text only.
        transformed = _RE_IMAGE.sub(r"\1", line)

        # Links → text (url).
        transformed = _RE_LINK.sub(r"\1 (\2)", transformed)

        # Italic: *text* → _text_ (processed first to avoid catching bold output).
        transformed = _RE_ITALIC.sub(r"_\1_", transformed)

        # Bold: **text** → *text*.
        transformed = _RE_BOLD.sub(r"*\1*", transformed)

        # Strip trailing whitespace (preserve leading for list indentation).
        transformed = transformed.rstrip()

        converted.append(transformed)

    result = "\n".join(converted)

    # Collapse 3+ consecutive newlines into 2 (one visual blank line).
    result = _RE_MULTIPLE_BLANK_LINES.sub("\n\n", result)

    # Ensure a single trailing newline.
    return result.strip() + "\n"


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
    _generate_and_save(messages, client, method)


def _itinerary_search_messages(
    itinerary_req: ItineraryRequestMD,
    search_results: str,
) -> ListParams:
    # Keep this function focused on packaging web results; _prompt builds the shared body.
    search_context = f"""
    <latest_web_search_results>
    The following is current information about Bali retrieved from web search:

    {search_results}

    Use this information to ensure your itinerary recommendations are up-to-date and relevant.
    </latest_web_search_results>
    """
    prompt = _prompt(itinerary_req, search_context=search_context)

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
    current_year = datetime.now().year
    return (
        f"Bali {current_year} travel guide: best hotels villas in Ubud Seminyak Canggu Uluwatu, "
        f"top warungs restaurants local cuisine, "
        f"hidden gems secret waterfalls lesser-known temples authentic experiences, "
        f"cooking classes traditional markets wellness retreats, "
        f"{interests} {itinerary_req.duration} day itinerary recommendations"
    )


def itinerary(itinerary_req: ItineraryRequestMD, client: OpenAiClient, method: str) -> None:
    for provider in providers():
        logger.info("Searching for latest Bali information with %s", provider)

        success, results = provider.search(_search_bali_info(itinerary_req))

        if not success:
            logger.warning("Search with %s failed, error: %s", provider, results)
            continue

        logger.info("Search completed successfully, generating itinerary with latest information")
        logger.info("Generating itinerary with %s", client)
        messages = _itinerary_search_messages(itinerary_req, results)

        _generate_and_save(messages, client, method)
        break
    else:
        logger.warning("Search returned no results or error, proceeding without search data")
        _plain_itinerary(itinerary_req, client, method)


def _generate_and_save(messages: ListParams, client: OpenAiClient, method: str) -> None:
    if not (content := caller(client, messages, append=False)[method]()):
        logger.warning("No itinerary content returned from OpenAI method %s", method)
        return

    content = _clean_markdown(content)
    save_to_file(content, Path.cwd() / "itinerary.md")

    whatsapp_text = _markdown_to_whatsapp(content)
    save_to_file(whatsapp_text, Path.cwd() / "itinerary.txt")
