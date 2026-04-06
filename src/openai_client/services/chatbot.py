# src/openai_client/services/chatbot.py

from asyncio import CancelledError, to_thread
from contextlib import suppress
from logging import getLogger
from re import IGNORECASE
from re import compile as re_compile
from re import escape as re_escape

from rich import print as rprint

from openai_client.client import OpenAI
from openai_client.schemas.ai import OpenAIChat
from openai_client.services.web_search import WebSearch
from openai_client.types import ListParams, SystemParam, UserParam

logger = getLogger(__name__)

IGNORED = KeyboardInterrupt, CancelledError, InterruptedError, EOFError

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

_TOURISM_PATTERN = re_compile(
    r"\b(?:" + "|".join(re_escape(kw) for kw in (_BALI_KEYWORDS | _TOURISM_KEYWORDS)) + r")\b",
    IGNORECASE,
)


def _chat_system_message() -> OpenAIChat:
    return OpenAIChat(
        role="system",
        content="""
    <assignment>
    Your name is "Made", you are the official BaliBlissed Concierge, a friendly and expert customer service assistant dedicated to crafting the perfect Balinese getaway. 🌴🥥✨
    </assignment>

    <personality&_style>
    - Introduction : Introduce yourself as BaliBlissed Concierge.
    - Vibe: Be exceptionally warm, welcoming, and enthusiastic. You are the digital face of Bali hospitality.
    - Engagement: Use emojis liberally to make your content engaging, fun, and vibrant! 🏝️🍹
    - Formatting: Use bold text and bullet points to ensure your recommendations are easy to scan for travelers on the move.
    </personality&_style>

    <knowledge&_scope>
    - Bali Expert: You specialize in Balinese culture, luxury villas, hidden jungle waterfalls, and the best surf spots. 🗺️🌺
    - Strictly Bali: You are an expert ONLY on Bali. If a user asks about other destinations (like Thailand or Europe), politely pivot the conversation back to the wonders of Bali.
    - Company First: Always redirect to BaliBlissed website for any booking inquiries, itinerary, and detailed information.
    </knowledge&_scope>

    <boundaries&_safety>
    - No over-reply: Focus only on the user's question. Do not provide suggestions, itineraries, information, prices, or recommendations unless the user asks.
    - No Professional Advice: Never provide legal, medical, or financial advice.
    - Just a glimpse: If the user asks for more information, suggestions, itineraries, prices, or recommendations, give a glimpse and be concise. If user ask for more details, redirect them to the official BaliBlissed website for more information.
    - Accuracy: If you aren't sure about a specific opening time or price, suggest the user check the official BaliBlissed website or contact the venue directly.
    - Prohibited: Reject any questions involving terrorism, racism, pornography, or violence.
    </boundaries&_safety>
    """,
    )


class Chatbot:
    def __init__(self, client: OpenAI) -> None:
        self._client = client
        self._messages: ListParams = [
            SystemParam(**_chat_system_message().model_dump()),
        ]
        self._client.messages = self._messages
        self._question: str = ""
        self._search_results: str = ""

    def _is_tourism_query(self) -> bool:
        text = self._question.strip()

        if len(text.split()) < 3 or text.lower() in _SKIP_PATTERNS:
            return False

        return _TOURISM_PATTERN.search(text) is not None

    async def run(self, method: str) -> None:
        await self._ask(method)

    def _append_user_message(self, question: str) -> None:
        user_param = OpenAIChat(
            role="user",
            content=question.strip(),
        )
        self._messages.append(UserParam(**user_param.model_dump()))
        self._client.messages = self._messages

    async def _ask(self, method: str) -> None:
        logger.info("Running chatbot with %s", self._client)
        is_bytez = "api.bytez.com" in str(self._client.base_url)

        with suppress(*IGNORED):
            while True:
                rprint("\n[b i blue]You:")
                self._append_user_message(question := await to_thread(input))
                self._question = question

                search_ctx_idx = await self._augment_with_search()

                rprint("\n[b i green]AI:")
                answer = await self._client.call(append=False, is_bytez=is_bytez)[method]()

                # Remove temporary search context from history
                if search_ctx_idx is not None:
                    self._messages.pop(search_ctx_idx)

                if method == "stream":
                    # Stream already printed output via _iter_chunk
                    pass
                elif not answer:
                    rprint("\n[b i red]No reply.")
                    continue
                else:
                    rprint(answer, end="\n", flush=True)

                if "bye" in self._question.lower():
                    break

    async def _augment_with_search(self) -> int | None:
        if not self._is_tourism_query():
            return

        rprint("[dim]Searching for latest information...[/dim]")
        if not (search_results := await self._search_bali_context()):
            return

        self._search_results = search_results
        # Insert search context just before the user message
        idx = len(self._messages) - 1
        self._messages.insert(idx, self._build_search_context_message())
        return idx

    async def _search_bali_context(self) -> str | None:
        configs = {
            "num_results": 3,
            "contents": {"highlights": {"max_characters": 8000}},
            "count": 3,
        }
        web_search = WebSearch("Chatbot searching Bali latest information with", configs)

        if not (results := await web_search.search(f"Bali latest {self._question}")):
            return

        return results

    def _build_search_context_message(self) -> SystemParam:
        return SystemParam(
            role="system",
            content=f"""\
    <search_context>
    The following is current information from the web relevant to the user's question:

    {self._search_results}

    Use this information to provide accurate, up-to-date answers. If the search results
    contradict your training data, prefer the search results for time-sensitive information
    (prices, hours, availability). Always maintain your BaliBlissed Concierge persona.
    </search_context>""",
        )
