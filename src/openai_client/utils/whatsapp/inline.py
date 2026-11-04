"""Inline-element rendering for WhatsApp markdown."""


class _InlineRendererMixin:
    """Inline markdown rendering rules for WhatsApp text."""

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
