# src/openai_client/services/itinerary.py

from datetime import datetime
from logging import getLogger
from pathlib import Path

from openai_client.client import MistralAI, OpenAIClient
from openai_client.configs import WHATSAPP_NUMBER
from openai_client.schemas.ai import ItineraryRequestMD
from openai_client.services.web_search import WebSearch
from openai_client.types import ListParams, MistralParams, MistralSystem, SystemParam, UserParam
from openai_client.utils import clean_markdown, markdown_to_whatsapp, save_to_file

logger = getLogger(__name__)


class Itinerary:
    def __init__(
        self,
        client: OpenAIClient | MistralAI,
        itinerary_req: ItineraryRequestMD,
        system_message: SystemParam | MistralSystem,
    ) -> None:
        self._client = client
        self._itinerary_req = itinerary_req
        self.messages: ListParams | MistralParams = [system_message]
        self._client.messages = self.messages

    def _topics(self) -> str:
        """
        Itinerary topics analysis.

        Returns:
            A formatted itinerary topics string.
        """
        duration = self._itinerary_req.duration
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

    def _structure(self) -> str:
        """
        Itinerary structure analysis.

        Returns:
            A formatted itinerary structure string.
        """
        duration = self._itinerary_req.duration
        budget = self._itinerary_req.budget

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

    def _prompt(self, search_context: str = "") -> str:
        """
        Create a detailed prompt for itinerary generation.

        Args:
            search_context: Optional search context to inject into the prompt.

        Returns:
            A formatted prompt string for the AI model.
        """
        req = self._itinerary_req
        interests = ", ".join(req.interests)
        duration = req.duration
        budget = req.budget

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
        {self._topics()}
        </content_topics>

        <structure>
        Follow this exact markdown structure for your output:
        {self._structure()}
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

    async def generate(self, method: str) -> None:
        web_search = WebSearch("Searching latest Bali itinerary information with")

        if not (results := await web_search.search(self._search_bali_info())):
            logger.warning("Search returned no results or error, proceeding without search data")
            await self._plain_itinerary(method)
            return

        logger.info("Generating itinerary with %s", self._client)
        user_param = self._itinerary_search_messages(results)
        self._client.messages.append(user_param)

        await self._generate_and_save(method)

    def _search_bali_info(self) -> str:
        req = self._itinerary_req
        interests = ", ".join(req.interests)
        current_year = datetime.now().year
        return (
            f"Bali {current_year} travel guide: best hotels villas in Ubud Seminyak Canggu Uluwatu, "
            f"top warungs restaurants local cuisine, "
            f"hidden gems secret waterfalls lesser-known temples authentic experiences, "
            f"cooking classes traditional markets wellness retreats, "
            f"{interests} {req.duration} day itinerary recommendations"
        )

    def _itinerary_search_messages(self, search_results: str) -> UserParam:
        # Keep this function focused on packaging web results; _prompt builds the shared body.
        search_context = f"""
        <latest_web_search_results>
        The following is current information about Bali retrieved from web search:

        {search_results}

        Use this information to ensure your itinerary recommendations are up-to-date and relevant.
        </latest_web_search_results>
        """

        return UserParam(role="user", content=self._prompt(search_context=search_context))

    def _itinerary_messages(self) -> ListParams:
        return [
            SystemParam(
                role="system",
                content="You are an expert travel planner specializing in authentic Bali experiences.",
            ),
            UserParam(
                role="user",
                content=self._prompt(),
            ),
        ]

    async def _plain_itinerary(self, method: str) -> None:
        logger.info("Generating itinerary with %s", self._client)
        messages = self._itinerary_messages()
        self._client.messages = messages
        await self._generate_and_save(method)

    async def _generate_and_save(self, method: str) -> None:
        if not (content := await self._client.call(append=False)[method]()):
            logger.warning("No itinerary content returned from OpenAI method %s", method)
            return

        content = clean_markdown(content)
        save_to_file(content, Path.cwd() / "itinerary.md")

        whatsapp_text = markdown_to_whatsapp(content)
        save_to_file(whatsapp_text, Path.cwd() / "itinerary.txt")
