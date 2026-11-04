"""
Post-processing to remove common AI writing patterns from generated text.

Patterns are based on the Humanizer skill (v2.7.0).  This module applies
targeted regex transforms — no second LLM call — to catch the most reliable
AI tells that slip through the system-prompt instructions.
"""

from logging import getLogger
from re import IGNORECASE, Match, Pattern
from re import compile as re_compile
from re import sub as re_sub

logger = getLogger(__name__)

# ---------------------------------------------------------------------------
# AI vocabulary — simpler replacements for words LLMs overuse
# ---------------------------------------------------------------------------

_AI_VOCAB_REPLACEMENTS: dict[str, str | None] = {
    "delve": "explore",
    "showcase": "show",
    "showcasing": "showing",
    "showcased": "showed",
    "enhance": "improve",
    "enhancing": "improving",
    "enhanced": "improved",
    "vibrant": "lively",
    "breathtaking": "beautiful",
    "must-visit": "worth visiting",
    "nestled": "located",
    "testament": "example",
    "underscore": "highlight",
    "underscores": "highlights",
    "underscored": "highlighted",
    "underscoring": "highlighting",
    "interplay": "connection",
    "pivotal": "important",
    "crucial": "important",
    "foster": "build",
    "fostering": "building",
    "fostered": "built",
    "garner": "get",
    "garnering": "getting",
    "garnered": "got",
    "enduring": "lasting",
    "tapestry": "variety",
    "testament to": None,
    "a testament": None,
    "serves as": "is",
    "serves as a": "is a",
    "stands as": "is",
    "stands as a": "is a",
    "boasts": "has",
    "boasting": "having",
    "venture": "go",
    "venturing": "going",
    "ventured": "went",
    "embark on": "start",
    "embarking on": "starting",
    "embarked on": "started",
    "immerse yourself": "enjoy",
    "immersing yourself": "enjoying",
    "immersed": "involved",
    "enchanting": "charming",
    "exquisite": "excellent",
    "pristine": "clean",
    "liminal": "in-between",
    "elevate": "improve",
    "elevating": "improving",
    "elevated": "improved",
    "curated": "selected",
    "curating": "selecting",
    "unforgettable": "memorable",
    "effortless": "easy",
    "seamless": "smooth",
    "seamlessly": "smoothly",
    "transformative": "life-changing",
    "bespoke": "custom",
    "hidden gem": "great spot",
    "hidden gems": "great spots",
}

# Pattern order matters: longer phrases first to avoid partial matches.
_AI_VOCAB_PATTERNS: list[tuple[Pattern[str], str]] = []
for phrase in sorted(_AI_VOCAB_REPLACEMENTS, key=len, reverse=True):
    replacement = _AI_VOCAB_REPLACEMENTS[phrase]
    if not isinstance(replacement, str):
        continue
    pattern = re_compile(r"(?<!\w)" + phrase + r"(?!\w)", IGNORECASE)
    _AI_VOCAB_PATTERNS.append((pattern, replacement))

# Patterns to REMOVE entirely (None replacement).
# Sorted longest-first so overlapping phrases like "testament to" and
# "a testament" are matched before their shorter counterparts.
_AI_VOCAB_REMOVE_PATTERNS: list[Pattern[str]] = []
for phrase in sorted(
    (p for p, r in _AI_VOCAB_REPLACEMENTS.items() if not isinstance(r, str)),
    key=len,
    reverse=True,
):
    _AI_VOCAB_REMOVE_PATTERNS.append(
        re_compile(r"(?<!\w)" + phrase + r"(?!\w)", IGNORECASE),
    )

# ---------------------------------------------------------------------------
# Filler phrases → concise alternatives
# ---------------------------------------------------------------------------

_FILLER_REPLACEMENTS: dict[str, str] = {
    "in order to": "to",
    "due to the fact that": "because",
    "at this point in time": "now",
    "at that point in time": "then",
    "in the event that": "if",
    "has the ability to": "can",
    "have the ability to": "can",
    "had the ability to": "could",
    "it is important to note that": "",
    "it should be noted that": "",
    "it is worth noting that": "",
    "it is worth mentioning that": "",
    "it is also worth noting": "",
    "it is also important to note": "",
    "when it comes to": "for",
    "a number of": "some",
    "the majority of": "most",
    "in order to achieve": "to achieve",
    "in order to ensure": "to ensure",
    "in the realm of": "in",
    "on a regular basis": "regularly",
    "on a daily basis": "daily",
    "on a weekly basis": "weekly",
    "on a monthly basis": "monthly",
    "on an annual basis": "annually",
    "as a means of": "to",
    "as a result of": "because of",
    "by means of": "by",
    "with the exception of": "except",
    "in close proximity to": "near",
    "prior to": "before",
    "subsequent to": "after",
    "in the vicinity of": "near",
    "in the neighborhood of": "about",
    "in the amount of": "for",
}

_FILLER_PATTERNS: list[tuple[Pattern[str], str]] = []
for phrase in sorted(_FILLER_REPLACEMENTS, key=len, reverse=True):
    replacement = _FILLER_REPLACEMENTS[phrase]
    pattern = re_compile(r"(?<!\w)" + phrase + r"(?!\w)", IGNORECASE)
    _FILLER_PATTERNS.append((pattern, replacement))

# ---------------------------------------------------------------------------
# Sycophantic/servile tone — phrases chatbots overuse
# ---------------------------------------------------------------------------

_SYCOPHANCY_PATTERNS: list[Pattern[str]] = [
    re_compile(r"(?i)\bgreat\s+question[!.]?"),
    re_compile(r"(?i)\bexcellent\s+question[!.]?"),
    re_compile(r"(?i)\byou['']?re\s+absolutely\s+right[!.]?"),
    re_compile(r"(?i)\byou['']?re\s+correct[!.]?"),
    re_compile(r"(?i)\bthat['']?s\s+a\s+great\s+point[!.]?"),
    re_compile(r"(?i)\bthat['']?s\s+an\s+excellent\s+point[!.]?"),
    re_compile(r"(?i)\bi['']?d\s+be\s+happy\s+to\b"),
    re_compile(r"(?i)\b(of\s+course|certainly)[!]\s*"),
    re_compile(r"(?i)\bi\s+hope\s+this\s+helps[!.]?"),
    re_compile(r"(?i)\bfeel\s+free\s+to\s+ask\b"),
    re_compile(r"(?i)\bdon['']?t\s+hesitate\s+to\b"),
]

# ---------------------------------------------------------------------------
# Signposting / announcements
# ---------------------------------------------------------------------------

_SIGNPOSTING_PATTERNS: list[Pattern[str]] = [
    re_compile(r"(?i)\blet['']?s\s+dive\s+in[!.]?\s*"),
    re_compile(r"(?i)\blet['']?s\s+explore\b[^.]*[.]?\s*"),
    re_compile(r"(?i)\bhere['']?s\s+what\s+you\s+need\s+to\s+know[!.]?\s*"),
    re_compile(r"(?i)\bwithout\s+further\s+ado[!.]?\s*"),
    re_compile(r"(?i)\bhere['']?s\s+a\s+(quick\s+)?(overview|breakdown|rundown)\b[^.]*[.]?\s*"),
    re_compile(r"(?i)\bnow\s+let['']?s\s+look\s+at\b[^.]*[.]?\s*"),
]

# ---------------------------------------------------------------------------
# Dashes
# ---------------------------------------------------------------------------

_EM_DASH_PATTERN: Pattern[str] = re_compile(r"\s*—\s*")
_EN_DASH_RANGE_PATTERN: Pattern[str] = re_compile(r"(?<=\d)–(?=\d)")
_EN_DASH_PATTERN: Pattern[str] = re_compile(r"\s*–\s*")

# ---------------------------------------------------------------------------
# Excessive hedging
# ---------------------------------------------------------------------------

_HEDGE_PATTERNS: list[tuple[Pattern[str], str]] = [
    (re_compile(r"(?i)\bcould\s+potentially\b"), "could"),
    (re_compile(r"(?i)\bmight\s+possibly\b"), "might"),
    (re_compile(r"(?i)\bpotentially\s+maybe\b"), "maybe"),
    (re_compile(r"(?i)\bquite\s+unique\b"), "unique"),
    (re_compile(r"(?i)\brather\s+unique\b"), "unique"),
    (re_compile(r"(?i)\bsomewhat\s+of\s+a\b"), "a"),
]

# ---------------------------------------------------------------------------
# Generic positive conclusions
# ---------------------------------------------------------------------------

_GENERIC_CLOSER_PATTERNS: list[Pattern[str]] = [
    re_compile(r"(?i)\bthe\s+future\s+looks\s+bright[!.]?\s*"),
    re_compile(r"(?i)\bexciting\s+times\s+lie\s+ahead[!.]?\s*"),
    re_compile(r"(?i)\bjourney\s+toward\s+excellence\b[^.]*[.]?\s*"),
    re_compile(r"(?i)\ba\s+major\s+step\s+in\s+the\s+right\s+direction\b[^.]*[.]?\s*"),
]

# ---------------------------------------------------------------------------
# Content protection — shields that prevent prose transforms from corrupting
# URLs, code spans, and fenced code blocks.
# ---------------------------------------------------------------------------

_PROTECT_PLACEHOLDER = "\x00"

_FENCED_CODE: Pattern[str] = re_compile(r"```[\s\S]*?```")
_INLINE_CODE: Pattern[str] = re_compile(r"`[^`\n]+`")
_MARKDOWN_DEST: Pattern[str] = re_compile(r"(\[[^\[\]]*\]\()([^()]*)(\))")
_BARE_URL: Pattern[str] = re_compile(r"https?://[^\s<>\"')]+")


def _protect_content(text: str, protected: list[str]) -> str:
    def _full(m: Match[str]) -> str:
        idx = len(protected)
        protected.append(m.group(0))
        return f"{_PROTECT_PLACEHOLDER}P{idx}{_PROTECT_PLACEHOLDER}"

    def _dest(m: Match[str]) -> str:
        idx = len(protected)
        protected.append(m.group(2))
        return f"{m.group(1)}{_PROTECT_PLACEHOLDER}P{idx}{_PROTECT_PLACEHOLDER}{m.group(3)}"

    text = _FENCED_CODE.sub(_full, text)
    text = _INLINE_CODE.sub(_full, text)
    text = _MARKDOWN_DEST.sub(_dest, text)
    text = _BARE_URL.sub(_full, text)
    return text


def _restore_content(text: str, protected: list[str]) -> str:
    for idx, original in enumerate(protected):
        text = text.replace(f"{_PROTECT_PLACEHOLDER}P{idx}{_PROTECT_PLACEHOLDER}", original)
    return text


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def humanize_response(text: str) -> str:
    """
    Remove common AI writing patterns from generated text.

    Applies targeted regex transforms — no second LLM call.  Designed to
    complement system-prompt instructions by catching what slips through.

    Parameters
    ----------
    text : str
        The AI-generated text to humanize.

    Returns
    -------
    str
        Text with AI patterns reduced.
    """
    if not text or not text.strip():
        return text

    # --- Protect structured content from prose transforms ---
    protected: list[str] = []
    text = _protect_content(text, protected)

    # 1. Numeric en-dash ranges (e.g. "10–12", "2–5pm") → " to "
    text = _EN_DASH_RANGE_PATTERN.sub(" to ", text)
    # 2. Em dashes → comma
    text = _EM_DASH_PATTERN.sub(", ", text)
    # 3. Remaining en dashes → comma
    text = _EN_DASH_PATTERN.sub(", ", text)

    # 4. Remove marked AI vocabulary entirely (longer phrases first)
    for pattern in _AI_VOCAB_REMOVE_PATTERNS:
        text = pattern.sub("", text)

    # 5. AI vocabulary → simpler words (longer phrases first)
    for pattern, replacement in _AI_VOCAB_PATTERNS:
        text = pattern.sub(replacement, text)

    # 6. Filler phrases → concise
    for pattern, replacement in _FILLER_PATTERNS:
        text = pattern.sub(replacement, text)

    # 7. Sycophantic phrases → remove
    for pattern in _SYCOPHANCY_PATTERNS:
        text = pattern.sub("", text)

    # 8. Signposting → remove
    for pattern in _SIGNPOSTING_PATTERNS:
        text = pattern.sub("", text)

    # 9. Excessive hedging → direct
    for pattern, replacement in _HEDGE_PATTERNS:
        text = pattern.sub(replacement, text)

    # 10. Generic conclusions → remove
    for pattern in _GENERIC_CLOSER_PATTERNS:
        text = pattern.sub("", text)

    # 11. Strip citation markers (e.g. [1], [1, 2]) that the
    #     AI may have adopted from search context.
    #     Done BEFORE _restore_content so protected content
    #     (code blocks, URLs) is not affected.
    #     Word-boundary assertions ((?<!\w) / (?!\w)) prevent
    #     false positives on items like ``items[1]`` in prose.
    text = re_sub(r"(?<!\w)\[\d+(?:,\s*\d+)*\](?!\w)", "", text)

    # 12. Collapse only internal horizontal whitespace (leaving indentation,
    #     newlines, and paragraph/list/heading/table breaks intact) and limit
    #     excessive blank lines. Run before restoring protected content so
    #     fenced code blocks keep their original spacing.
    text = re_sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", text)
    text = re_sub(r"\n{3,}", "\n\n", text)

    # --- Restore protected content ---
    text = _restore_content(text, protected)

    return text.strip()
