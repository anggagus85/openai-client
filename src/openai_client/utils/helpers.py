# src/openai_client/utils/helpers.py

from logging import getLogger
from re import sub
from typing import Any

from anyio import Path
from mdformat import text as mdformat_text
from mistune import HTMLRenderer, Markdown, create_markdown

from openai_client.errors import SYSTEM_EXCEPTIONS, AppSystemError

logger = getLogger(__name__)


async def save_to_file(data: str, file_path: Path) -> None:
    """
    Write string data to a file.

    Args:
        data: The data string to write.
        file_path: The (anyio) path to the file to write.
    """
    try:
        await file_path.write_text(data, encoding="utf-8")
        logger.info("Data saved to '%s'", file_path)
    except SYSTEM_EXCEPTIONS as e:
        details = f"Failed to save data to file: {e}"
        logger.exception(details)
        raise
    except Exception as e:
        details = f"Failed to save data to file: {e}"
        raise AppSystemError(details) from e


async def save_itinerary_to_file(content: str, whatsapp_text: str) -> None:
    """Save the markdown and WhatsApp variants to the current directory."""
    try:
        cwd = await Path.cwd()
        await save_to_file(content, cwd / "itinerary.md")
        await save_to_file(whatsapp_text, cwd / "itinerary.txt")
    except OSError:
        logger.exception("Failed to save itinerary files")
        raise


def clean_markdown(text: str) -> str:
    """
    Format raw Markdown text to be CommonMark/GFM compliant.

    Useful for standardizing AI outputs before sending to frontend.
    """
    try:
        return mdformat_text(
            text,
            # 1. Enable Plugins: Explicitly list extensions that installed
            extensions={"gfm", "footnote", "frontmatter", "gfm_alerts"},
            # 2. Options: Customize how the text is rendered
            options={
                "wrap": "no",
                "number": True,
                "end_of_line": "lf",
            },
        )
    except Exception:
        logger.exception("Markdown formatting failed")
        return text


# ---------------------------------------------------------------------------
# Markdown → WhatsApp text converter (mistune AST renderer)
# ---------------------------------------------------------------------------

DAILY_ACTIVITIES = (
    "Morning Activity",
    "Afternoon Activity",
    "Evening Activity",
    "Evening Wind‑down",
)

_HYPHEN_TRANSLATION = str.maketrans("\u2010\u2011\u2012\u2013\u2014\u2212", "------")


def _normalize_activity(text: str) -> str:
    """Normalize hyphen variants and case for activity matching."""
    return text.translate(_HYPHEN_TRANSLATION).casefold()


_NORMALIZED_DAILY_ACTIVITIES = tuple(_normalize_activity(activity) for activity in DAILY_ACTIVITIES)


class BaliTravelRenderer(HTMLRenderer):
    """
    Renderer to convert Markdown to WhatsApp-safe text.

    Transformation rules:
    - ``# H1`` → ``*H1*`` (bold, uppercased)
    - ``## H2``–``###### H6`` → ``*H2*`` (bold)
    - ``**bold**`` → ``*bold*`` (WhatsApp bold)
    - ``*italic*`` → ``_italic_`` (WhatsApp italic)
    - ``[text](url)`` → ``text (url)``
    - images → alt text
    - tables → bullet-point rows
    - ``---`` / ``***`` → ``─────────────``
    """

    def __init__(self) -> None:
        super().__init__(escape=False)
        self._table_headers: list[str] = []
        self._current_row_cells: list[str] = []

    # --- Inline elements ---

    def text(self, text: str) -> str:
        return text

    def strong(self, text: str) -> str:
        return f"*{text}*"

    def emphasis(self, text: str) -> str:
        return f"_{text}_"

    def link(self, text: str, url: str, title: str | None = None) -> str:
        return f"{text} ({url})"

    def image(self, text: str, url: str, title: str | None = None) -> str:
        return text

    def codespan(self, text: str) -> str:
        return f"`{text}`"

    def linebreak(self) -> str:
        return "\n"

    def softbreak(self) -> str:
        return "\n"

    # --- Block elements ---

    def heading(self, text: str, level: int, **attrs: object) -> str:
        clean_text = text.replace("*", "")
        if level == 1:
            return f"\n*{clean_text.upper()}*\n"
        return f"\n*{clean_text}*\n"

    def paragraph(self, text: str) -> str:
        return f"{text}\n"

    def thematic_break(self) -> str:
        return "─────────────\n"

    def block_code(self, code: str, info: str | None = None, **attrs: object) -> str:
        return f"{code}\n"

    def block_quote(self, text: str) -> str:
        return text

    def list(self, text: str, ordered: bool, **attrs: object) -> str:  # noqa: FBT001
        return f"\n{text}"

    def list_item(self, text: str) -> str:
        normalized = _normalize_activity(text)
        if any(activity in normalized for activity in _NORMALIZED_DAILY_ACTIVITIES):
            return f"• {text.strip()}\n"
        return f"\t- {text.strip()}\n"

    # --- Tables ---

    def table(self, text: str) -> str:
        self._table_headers = []
        return f"\n{text}\n"

    def table_head(self, text: str) -> str:
        return ""

    def table_body(self, text: str) -> str:
        return text

    def table_row(self, text: str) -> str:
        return text

    def table_cell(self, text: str, align: str | None = None, *, head: bool = False) -> str:
        if head:
            self._table_headers.append(text.strip())
            return ""

        self._current_row_cells.append(text.strip())

        if len(self._current_row_cells) == len(self._table_headers):
            cells = self._current_row_cells
            self._current_row_cells = []

            label = cells[0].replace("*", "")
            parts: list[str] = []
            for i, cell in enumerate(cells[1:], start=1):
                header_name = self._table_headers[i] if i < len(self._table_headers) else ""
                parts.append(f"{header_name} {cell}".strip())

            detail = " | ".join(parts)
            return f"  • *{label}*: {detail}\n" if detail else f"  • *{label}*\n"

        return ""


def create_whatsapp_renderer() -> Markdown:
    """Create a fresh Markdown parser for WhatsApp conversion (thread-safe)."""
    return create_markdown(renderer=BaliTravelRenderer(), plugins=["table"])


def markdown_to_whatsapp(md_text: str) -> str:
    """
    Convert CommonMark/GFM markdown into WhatsApp-friendly plain text.

    Uses ``BaliTravelRenderer`` (mistune-based) for AST-aware conversion.
    """
    parser: Markdown = create_whatsapp_renderer()
    raw_output: str | list[Any] = parser(md_text)

    result: str = "".join(raw_output) if isinstance(raw_output, list) else raw_output
    result = sub(r"\n{3,}", "\n\n", result)
    return result.strip() + "\n"
