"""Activity-label data used by the WhatsApp renderer."""

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
