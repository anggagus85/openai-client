# src/openai_client/services/chatbot.py

from contextlib import suppress
from logging import getLogger
from re import IGNORECASE
from re import compile as re_compile
from re import escape as re_escape

from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from rich import print as rprint

from openai_client.client import OpenAiClient
from openai_client.schemas.ai import OpenAIChatMessage
from openai_client.services.caller import caller
from openai_client.services.search import BraveSearchClient, ExaSearchClient, providers
from openai_client.types import ListParams

logger = getLogger(__name__)

# ---------------------------------------------------------------------------
# Search-augmentation keyword sets
# ---------------------------------------------------------------------------

_BALI_KEYWORDS: frozenset[str] = frozenset(
    {
        "agung rai",
        "aling aling",
        "amed",
        "angel billabong",
        "arak",
        "atuh beach",
        "ayung river",
        "babi guling",
        "badung",
        "balangan",
        "bali",
        "bali belly",
        "bali bird park",
        "bali coffee",
        "bali safari",
        "bali swing",
        "bali zoo",
        "bangli",
        "banyumala",
        "barong",
        "batik",
        "batubulan",
        "batur lake",
        "bebek betutu",
        "bedugul",
        "besakih",
        "bingin",
        "bintang",
        "blanco museum",
        "blue lagoon",
        "boga wreck",
        "boreh",
        "bratan lake",
        "broken beach",
        "buleleng",
        "buyan lake",
        "cafe del mar",
        "campuhan ridge",
        "canang sari",
        "candidasa",
        "canggu",
        "celuk",
        "ceningan",
        "crystal bay",
        "denpasar",
        "deus",
        "devil tear",
        "dreamland",
        "endek",
        "fiji waterfall",
        "finns beach club",
        "gado gado",
        "galungan",
        "gamat bay",
        "gamelan",
        "gianyar",
        "gili air",
        "gili meno",
        "gili trawangan",
        "gilimanuk",
        "gitgit",
        "goa gajah",
        "goa lawah",
        "green bowl",
        "gunung kawi",
        "gunung payung",
        "ikat",
        "impossibles",
        "jagatnatha",
        "jamu",
        "janger",
        "jatiluwih",
        "jembrana",
        "jemeluk bay",
        "jimbaran",
        "jungut batu",
        "karangasem",
        "karma beach",
        "karma beach club",
        "kecak",
        "kelingking beach",
        "kerobokan",
        "kerta gosa",
        "kintamani",
        "klungkung",
        "klungkung palace",
        "kopi luwak",
        "ku de ta",
        "kubu",
        "kuningan",
        "kuta",
        "la brisa",
        "lawar",
        "legian",
        "legong",
        "leke leke",
        "lembongan",
        "liberty wreck",
        "lotus temple",
        "lovina",
        "mahagiri",
        "manta point",
        "manta ray",
        "mari beach",
        "mas",
        "melanting",
        "melasti",
        "melasti beach",
        "mertasari",
        "mie goreng",
        "mola mola",
        "motel mexicola",
        "mount agung",
        "mount batur",
        "mrs sippy",
        "munduk",
        "munduk waterfall",
        "mushroom bay",
        "nasi campur",
        "nasi goreng",
        "neka art",
        "ngaben",
        "nungnung",
        "nusa ceningan",
        "nusa dua",
        "nusa dua beach",
        "nusa lembongan",
        "nusa penida",
        "nyepi",
        "odalan",
        "old mans",
        "padang bai",
        "padang padang",
        "pandawa",
        "payangan",
        "pemuteran",
        "pendet",
        "penida",
        "pepes",
        "potato head",
        "pretty poison",
        "pura lempuyang",
        "pura luhur",
        "pura taman saraswati",
        "puri lukisan",
        "pyramids",
        "rendang",
        "rock bar",
        "rudana",
        "sacred monkey forest",
        "salak",
        "sambal matah",
        "sampalan",
        "sandbar",
        "sandy bay",
        "sanghyang",
        "sanur",
        "satay",
        "sate lilit",
        "savaya",
        "sawangan",
        "sd point",
        "seaweed farm",
        "sekumpul",
        "seminyak",
        "seniwati",
        "seraya secrets",
        "setia darma",
        "sidemen",
        "silver jewelry",
        "singaraja",
        "single fin",
        "songket",
        "subak",
        "sukawati",
        "suluban",
        "sundays beach club",
        "tabanan",
        "taman ayun",
        "taman ujung",
        "tamblingan",
        "tamblingan lake",
        "tampaksiring",
        "tanah lot",
        "tegallalang",
        "tegallalang rice terrace",
        "tegenungan",
        "the lawn",
        "tirta empul",
        "tirta gangga",
        "tirtagangga water palace",
        "topeng",
        "toyapakeh",
        "trans studio",
        "tropical temptations",
        "tukad cepung",
        "tulamben",
        "ubud",
        "ulu cliffhouse",
        "uluwatu",
        "ulun danu",
        "unesco rice terrace",
        "urap",
        "usat liberty",
        "waterbom",
        "wayang kulit",
        "woo bar",
        "yellow bridge",
    },
)

_TOURISM_KEYWORDS: frozenset[str] = frozenset(
    {
        "accommodation",
        "activity",
        "adventure",
        "airport",
        "attraction",
        "beach",
        "book",
        "budget",
        "ceremony",
        "cooking class",
        "cost",
        "culture",
        "dance",
        "distance",
        "diving",
        "driver",
        "excursion",
        "food",
        "hike",
        "hotel",
        "itinerary",
        "market",
        "mount",
        "nightlife",
        "price",
        "rafting",
        "resort",
        "restaurant",
        "rice terrace",
        "scooter",
        "season",
        "shopping",
        "snorkel",
        "spa",
        "sunrise",
        "sunset",
        "surf",
        "taxi",
        "temple",
        "tour",
        "transport",
        "travel",
        "trek",
        "trip",
        "villa",
        "visa",
        "volcano",
        "warung",
        "waterfall",
        "weather",
        "wellness",
        "yoga",
    },
)

_SKIP_PATTERNS: frozenset[str] = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "bye",
        "goodbye",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "yes",
        "no",
        "sure",
    },
)

# Single compiled regex alternating all tourism/Bali keywords with word
# boundaries.  Built once at import time so _is_tourism_query avoids
# recompiling per keyword on every call.
_TOURISM_PATTERN = re_compile(
    r"\b(?:" + "|".join(re_escape(kw) for kw in (_BALI_KEYWORDS | _TOURISM_KEYWORDS)) + r")\b",
    IGNORECASE,
)


# ---------------------------------------------------------------------------
# Search-augmentation helpers
# ---------------------------------------------------------------------------


def _is_tourism_query(question: str) -> bool:
    """
    Detect whether a user question likely relates to Bali tourism.

    Uses word-boundary regex matching against curated keyword sets.
    Short greetings and farewells are skipped so we avoid unnecessary
    search latency on trivial messages.

    Args:
        question: The raw user input.

    Returns:
        ``True`` when the question appears to be tourism-related.
    """
    text = question.strip()

    # Skip very short inputs and common non-tourism phrases
    if len(text.split()) < 3 or text.lower() in _SKIP_PATTERNS:
        return False

    return _TOURISM_PATTERN.search(text) is not None


def _search_bali_context(question: str) -> str | None:
    """
    Search the web for current Bali information relevant to *question*.

    Iterates through available search providers (Exa, Brave) and returns
    the first successful result.  Returns ``None`` when every provider
    fails or no providers are configured.

    Args:
        question: The user's tourism-related question.

    Returns:
        Formatted search results string, or ``None``.
    """
    query = f"Bali latest {question}"
    for provider in providers():
        logger.info("Chatbot searching with %s", provider)
        if isinstance(provider, ExaSearchClient):
            success, results = provider.search(query, num_results=3)
        elif isinstance(provider, BraveSearchClient):
            success, results = provider.search(query, count=3)
        else:
            continue

        if success:
            logger.info("Search returned results from %s", provider)
            return results
        logger.warning("Search with %s failed, trying next provider", provider)


def _build_search_context_message(results: str) -> ChatCompletionSystemMessageParam:
    """
    Wrap search results in a system message for LLM context injection.

    Args:
        results: Formatted search results string.

    Returns:
        A ``ChatCompletionSystemMessageParam`` containing the search context.
    """
    return ChatCompletionSystemMessageParam(
        role="system",
        content=f"""\
<search_context>
The following is current information from the web relevant to the user's question:

{results}

Use this information to provide accurate, up-to-date answers. If the search results
contradict your training data, prefer the search results for time-sensitive information
(prices, hours, availability). Always maintain your BaliBlissed Concierge persona.
</search_context>""",
    )


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


def _inject_search_context(question: str, messages: ListParams) -> int | None:
    """
    Conditionally inject web-search context into *messages*.

    When the question is tourism-related, searches the web and inserts a
    temporary system message right before the latest user message.

    Args:
        question: The raw user input.
        messages: The conversation history (mutated in-place).

    Returns:
        The index at which the context message was inserted, or ``None``
        if no search was performed / search failed.
    """
    if not _is_tourism_query(question):
        return None

    rprint("[dim]Searching for latest information...[/dim]")
    search_results = _search_bali_context(question)
    if not search_results:
        return None

    # Insert search context just before the user message
    idx = len(messages) - 1
    messages.insert(idx, _build_search_context_message(search_results))
    return idx


def _run_chat(client: OpenAiClient, method: str, messages: ListParams) -> None:
    logger.info("Running chatbot with %s", client)
    is_bytez = "api.bytez.com" in str(client.base_url)

    with suppress(KeyboardInterrupt):
        while True:
            rprint("\n[b i blue]You:")
            _append_user_message(question := input(), messages)

            search_ctx_idx = _inject_search_context(question, messages)

            rprint("\n[b i green]AI:")
            answer = caller(client, messages, is_bytez=is_bytez)[method]()

            # Remove temporary search context from history
            if search_ctx_idx is not None:
                messages.pop(search_ctx_idx)

            if method == "stream":
                pass  # Stream already printed output via _iter_chunk
            elif not answer:
                rprint("\n[b i red]No reply.")
                continue
            else:
                rprint(answer, end="\n", flush=True)

            if "bye" in question.lower():
                break
