"""Table rendering for WhatsApp markdown."""


class _TableRendererMixin:
    """Table markdown rendering rules for WhatsApp text."""

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
