"""WhatsApp markdown renderer assembly."""

from re import IGNORECASE, sub
from re import compile as re_compile
from typing import Any

from mistune import HTMLRenderer, Markdown, create_markdown

from openai_client.utils.whatsapp.block import _BlockRendererMixin
from openai_client.utils.whatsapp.inline import _InlineRendererMixin
from openai_client.utils.whatsapp.table import _TableRendererMixin

_HTML_TAG_PATTERN = re_compile(r"<[^>]+>")
_BLOCK_TAG_PATTERN = re_compile(
    r"</?(?:p|div|section|article|header|footer|h[1-6]|li|ul|ol|blockquote|pre|table|thead|tbody|tr|td|th|hr)\b[^>]*>",
    IGNORECASE,
)


def _strip_html(html: str) -> str:
    """Convert raw HTML into WhatsApp plain text (block boundaries + ``<br>`` → newline)."""
    text = _BLOCK_TAG_PATTERN.sub("\n", html)
    text = sub(r"<br\s*/?>", "\n", text, flags=IGNORECASE)
    return _HTML_TAG_PATTERN.sub("", text)


class BaliTravelRenderer(
    _InlineRendererMixin,
    _BlockRendererMixin,
    _TableRendererMixin,
    HTMLRenderer,
):
    """
    Renderer to convert Markdown to WhatsApp-safe text.

    Transformation rules:
    - ``# H1`` → ``*H1*`` (bold, uppercased)
    - ``## H2``-``###### H6`` → ``*H2*`` (bold)
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

    def inline_html(self, html: str) -> str:
        return _strip_html(html)

    def block_html(self, html: str) -> str:
        return _strip_html(html)


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
