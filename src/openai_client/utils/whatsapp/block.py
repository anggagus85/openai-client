"""Block-element rendering for WhatsApp markdown."""

from openai_client.utils.whatsapp.data import _NORMALIZED_DAILY_ACTIVITIES, _normalize_activity


class _BlockRendererMixin:
    """Block markdown rendering rules for WhatsApp text."""

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
        if not ordered:
            return f"\n{text}"

        start_attr = attrs.get("start")
        start = 1
        if start_attr:
            try:
                start = int(str(start_attr))
            except (TypeError, ValueError):
                start = 1

        items: list[str] = []
        number = start
        pending_indent = ""
        for raw in text.split("\n"):
            entry = raw.strip()
            if not entry:
                continue
            if entry.startswith(("- ", "• ")):
                label = f"{number}. "
                items.append(label + entry[2:])
                pending_indent = " " * len(label)
                number += 1
            elif items:
                items[-1] += f"\n{pending_indent}{entry}"

        return "\n" + "\n".join(items) + "\n"

    def list_item(self, text: str) -> str:
        normalized = _normalize_activity(text)
        if any(activity in normalized for activity in _NORMALIZED_DAILY_ACTIVITIES):
            return f"• {text.strip()}\n"
        return f"\t- {text.strip()}\n"
