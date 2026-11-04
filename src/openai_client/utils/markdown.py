"""Markdown formatting helpers."""

from logging import getLogger

from mdformat import text as mdformat_text

logger = getLogger(__name__)

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
