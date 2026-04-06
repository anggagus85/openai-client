# src/openai_client/utils/helpers.py

from logging import getLogger
from pathlib import Path
from re import compile as re_compile

from mdformat import text as mdformat_text

from openai_client.errors import SYSTEM_EXCEPTIONS, AppSystemError

logger = getLogger(__name__)


def save_to_file(data: str, file_path: Path) -> None:
    """
    Write string data to a file.

    Args:
        data: The data string to write.
        file_path: The path to the file to write to.
    """

    try:
        with file_path.open("w", encoding="utf-8") as f:
            f.write(data)
        logger.info("Data saved to '%s'", file_path)
    except SYSTEM_EXCEPTIONS as e:
        details = f"Failed to save data to file: {e}"
        logger.exception(details)
        raise
    except Exception as e:
        details = f"Failed to save data to file: {e}"
        raise AppSystemError(details) from e


def clean_markdown(text: str) -> str:
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
    if len(table_lines) < 2:
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


def markdown_to_whatsapp(md_text: str) -> str:
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
