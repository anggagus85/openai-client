"""HTML-to-plain-text flattening for email fallbacks."""

import re
from html import unescape

_SCRIPT_OR_STYLE_PATTERN = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)


_BLOCK_ELEMENT_PATTERN = re.compile(
    r"</?(?:p|div|tr|table|section|article|header|footer|ul|ol|li|h[1-6])\b[^>]*>",
    re.IGNORECASE,
)


_LINE_BREAK_PATTERN = re.compile(r"<br\s*/?>", re.IGNORECASE)


_ANY_TAG_PATTERN = re.compile(r"<[^>]+>")


_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")


_HORIZONTAL_SPACE_PATTERN = re.compile(r"[ \t]+")


_ANCHOR_PATTERN = re.compile(
    r'<a\b[^>]*href\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+))[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)


def _replace_anchor(match: re.Match[str]) -> str:
    """Render an anchor as 'Label: url' per the plain-text spec (rule 1)."""
    url = (match.group(1) or match.group(2) or match.group(3) or "").strip()
    label = _HORIZONTAL_SPACE_PATTERN.sub(" ", _ANY_TAG_PATTERN.sub("", match.group(4))).strip()
    return f"{label}: {url}" if label else url


def html_to_plain_text(html_content: str) -> str:
    """
    Flatten an HTML email body into a plain-text fallback.

    Used for HTML generated outside this module (e.g. LLM-produced bodies) so
    every message can still be sent as multipart/alternative. Anchors are
    rewritten as ``Label: url``; block elements are separated by blank lines.

    Args:
        html_content: The HTML body to convert.

    Returns:
        str: Plain text with tags removed, links preserved and HTML entities decoded.
    """
    text = _ANCHOR_PATTERN.sub(_replace_anchor, html_content)
    text = _SCRIPT_OR_STYLE_PATTERN.sub("", text)
    text = _BLOCK_ELEMENT_PATTERN.sub("\n\n", text)
    text = _LINE_BREAK_PATTERN.sub("\n", text)
    text = _ANY_TAG_PATTERN.sub("", text)
    text = unescape(text)
    text = _HORIZONTAL_SPACE_PATTERN.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_LINES_PATTERN.sub("\n\n", text)
    return text.strip()
