# src/openai_client/services/email_inquiry.py

"""OpenAI email inquiry service mirrored from the production BaliRoam implementation."""

from logging import getLogger
from re import IGNORECASE
from re import compile as re_compile
from typing import Any

from anyio import Path
from pydantic import ValidationError

from openai_client.client import MistralAI, OpenAIClient
from openai_client.errors import AgentGenerationError
from openai_client.schemas.ai import AnalysisFormat, ContactAnalysisResponse, EmailInquiry
from openai_client.services.prompts import (
    EMAIL_SYSTEM_PROMPT,
    append_system_message,
    append_user_message,
    confirmation_message_prompt,
    contact_analysis_prompt,
    create_analysis_body,
)
from openai_client.typings import ListParams, MistralParams, MistralSystem, SystemParam
from openai_client.utils import save_to_file

logger = getLogger(__name__)

_FENCED_JSON_PATTERN = re_compile(
    r"\A\s*```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n[ \t]*```\s*\Z",
    IGNORECASE,
)


class EmailService:
    """Analyze contact inquiries and generate confirmation messages."""

    def __init__(
        self,
        client: OpenAIClient | MistralAI,
        inquiry: EmailInquiry,
        system_param: SystemParam | MistralSystem | None = None,
    ) -> None:
        self._client = client
        self._inquiry = inquiry
        self._system_param = system_param
        self._messages: ListParams | MistralParams = []
        self._is_mistral = bool(getattr(client, "is_mistral", False))
        self._configs: dict[str, Any] = {"temperature": 0.7, "top_p": 1.0}

    def _system_content(self) -> str:
        if self._system_param is None:
            return EMAIL_SYSTEM_PROMPT
        if isinstance(self._system_param, MistralSystem):
            return str(self._system_param.content or "")
        return str(self._system_param["content"])

    @staticmethod
    def _normalize_json_response(response: str) -> str:
        """Reduce fenced or prefaced model output to the JSON object body."""
        text = response.strip()

        if fenced := _FENCED_JSON_PATTERN.search(text):
            text = fenced.group(1).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return text[start : end + 1]

        return text

    @staticmethod
    def _validate_method(method: str) -> None:
        if method not in {"completion", "stream"}:
            details = f"Unknown method: {method}"
            raise ValueError(details)

    async def analyze_contact(self, method: str = "completion") -> AnalysisFormat:
        """
        Analyze a contact inquiry using the production prompt and JSON schema.

        Email analysis is always a non-streaming completion in production.
        """
        self._validate_method(method)

        logger.info("Analyzing email inquiry with %s", self._client)

        self._messages = []
        append_system_message(self._messages, self._system_content(), is_mistral=self._is_mistral)
        content = contact_analysis_prompt(
            self._inquiry.name,
            self._inquiry.message,
            self._inquiry.subject,
        )
        append_user_message(self._messages, content, is_mistral=self._is_mistral)
        self._configs["messages"] = self._messages

        if not (response := await self._client.generate(self._configs, stream=False)):
            detail = "Failed to analyze contact inquiry, please try again or contact us directly"
            raise AgentGenerationError(detail=detail)

        try:
            return AnalysisFormat.model_validate_json(response.strip())
        except ValidationError:
            pass

        try:
            return AnalysisFormat.model_validate_json(self._normalize_json_response(response))
        except ValidationError:
            logger.warning(
                "Contact inquiry analysis validation failed, using default AnalysisFormat",
                exc_info=True,
            )
            return AnalysisFormat(name=self._inquiry.name)

    async def confirmation_message(
        self,
        method: str = "completion",
        *,
        sent: bool = True,
    ) -> ContactAnalysisResponse:
        """Generate the customer-facing confirmation message."""
        self._validate_method(method)

        # Rebuild the message list so repeated calls never accumulate prompts.
        self._messages = []
        append_system_message(self._messages, self._system_content(), is_mistral=self._is_mistral)

        content = confirmation_message_prompt(self._inquiry, sent=sent)
        append_user_message(self._messages, content, is_mistral=self._is_mistral)
        self._configs["messages"] = self._messages

        logger.info("Generating confirmation message...")

        if not (response := await self._client.generate(self._configs, stream=False)):
            return ContactAnalysisResponse(
                confirmation=(
                    f"Thank you {self._inquiry.name}. Your inquiry has been received "
                    "and we will get back to you soon."
                ),
            )

        try:
            return ContactAnalysisResponse.model_validate({"confirmation": response})
        except ValidationError:
            logger.warning("Confirmation message validation failed, using default response")
            return ContactAnalysisResponse(
                confirmation=(
                    f"Thank you {self._inquiry.name}. Your inquiry has been received "
                    "and we will get back to you soon."
                ),
            )

    async def run(self, method: str = "completion") -> ContactAnalysisResponse:
        """CLI harness: analyze, write the analysis HTML, then confirm."""
        self._validate_method(method)

        analysis = await self.analyze_contact(method)
        body = create_analysis_body(self._inquiry, analysis)
        await save_to_file(body, (await Path.cwd()) / "email_inquiry.html")

        return await self.confirmation_message(method)
