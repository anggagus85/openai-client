"""Shared prompt utilities: sanitization, humanizer rules, and message helpers."""

from collections.abc import Sequence
from hashlib import sha256
from logging import getLogger
from re import IGNORECASE, Pattern
from re import compile as re_compile
from re import escape as re_escape
from re import sub as re_sub
from typing import Literal, overload
from unicodedata import normalize

from openai_client.typings import (
    ListParams,
    MistralParams,
    MistralSystem,
    MistralUser,
    SystemParam,
    UserParam,
)

logger = getLogger(__name__)
security_logger = getLogger("openai_client.security.prompt_injection")


_LEET_MAP: dict[str, str] = {
    "a": r"[a@4]",
    "e": r"[e3]",
    "i": r"[i1!]",
    "o": r"[o0]",
    "s": r"[s5\$]",
    "t": r"[t7]",
}

_INJECTION_PHRASES: list[str] = [
    "ignore all previous instructions",
    "ignore all instructions",
    "ignore previous instructions",
    "ignore all rules",
    "ignore the above",
    "ignore previous",
    "disregard all previous instructions",
    "disregard all instructions",
    "disregard previous instructions",
    "forget all previous instructions",
    "forget all instructions",
    "forget previous instructions",
    "forget everything",
    "system prompt",
    "you are now",
    "you must obey",
    "new instructions",
    "from now on",
    "do not follow",
    "ignore all constraints",
    "forget all rules",
    "you are free",
    "no restrictions",
    "ignore system prompt",
    "ignore system instructions",
    "you can do anything",
    "you are not bound",
    "ignore safety",
    "disregard safety",
    "ignore guardrails",
    "override instructions",
]


def _build_injection_pattern() -> Pattern[str]:
    alternatives: list[str] = []
    for phrase in _INJECTION_PHRASES:
        pattern_str = ""
        for char in phrase:
            if char == " ":
                pattern_str += r"\s+"
            elif char in _LEET_MAP:
                pattern_str += _LEET_MAP[char]
            else:
                pattern_str += re_escape(char)
        alternatives.append(pattern_str)

    return re_compile(
        r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)",
        IGNORECASE,
    )


_INJECTION_PATTERN: Pattern[str] = _build_injection_pattern()


def sanitize_for_prompt(value: str, max_length: int = 2000) -> str:
    """
    Sanitize user input to mitigate prompt injection attacks.

    Strips control characters and zero-width Unicode characters, normalizes
    whitespace, removes known prompt-injection phrases (with leet-speak
    awareness) and delimiter tokens, and enforces a maximum length.
    Uses multi-pass cleaning for nested or obfuscated payloads.
    Input beyond four times ``max_length`` is truncated before normalization
    so attacker-controlled length cannot drive the sanitizer's work.
    """
    value = value[: max_length * 4]
    original_value = value
    phrase_count = 0

    # -- Pass 1: Basic normalization --
    value = normalize("NFKC", value)
    value = re_sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)

    # Strip zero-width and invisible Unicode characters
    value = re_sub(
        r"[\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064\ufeff\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180e\u2028\u2029\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u206a\u206b\u206c\u206d\u206e\u206f\uffa0\ufff9\ufffa\ufffb]",
        "",
        value,
    )

    value = value.strip()

    # -- Multi-pass: run up to 3 iterations using the pre-compiled alternation --
    for _iteration in range(3):
        previous = value
        new_value, subs = _INJECTION_PATTERN.subn(" ", value)
        phrase_count += subs
        value = new_value
        value = re_sub(r"\s+", " ", value).strip()
        if value == previous:
            break

    delimiter_tokens = [
        "<contact_inquiry_data>",
        "</contact_inquiry_data>",
        "<user_name>",
        "</user_name>",
        "<user_subject>",
        "</user_subject>",
        "<user_message>",
        "</user_message>",
        "<<USER_NAME>>",
        "<</USER_NAME>>",
        "<<USER_MESSAGE>>",
        "<</USER_MESSAGE>>",
        "<assignment>",
        "</assignment>",
        "<personality_and_style>",
        "</personality_and_style>",
        "<knowledge_and_scope>",
        "</knowledge_and_scope>",
        "<boundaries_and_safety>",
        "</boundaries_and_safety>",
        "<search_context>",
        "</search_context>",
        "<rules>",
        "</rules>",
        "<role>",
        "</role>",
        "<context>",
        "</context>",
    ]
    for token in delimiter_tokens:
        token_pattern = re_compile(re_escape(token), flags=IGNORECASE)
        value = token_pattern.sub("", value)

    if len(value) > max_length:
        value = value[:max_length]

    if value != original_value:
        orig_fp = sha256(original_value.encode("utf-8", errors="ignore")).hexdigest()[:8]
        new_fp = sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:8]
        security_logger.warning(
            "Prompt injection sanitized [fp=%s..%s | input_len=%d | output_len=%d | phrases_removed=%d]",
            orig_fp,
            new_fp,
            len(original_value),
            len(value),
            phrase_count,
        )

    return value


def resolve_preferred_name(
    first_name: str | None,
    display_name: str | None,
    username: str | None,
) -> str | None:
    """Resolve a conversational name without falling back to account handles."""
    first = normalize("NFKC", first_name or "").strip()
    if first and "@" not in first:
        return first

    display = normalize("NFKC", display_name or "").strip()
    handle = normalize("NFKC", username or "").strip().casefold()
    if display and "@" not in display and display.casefold() != handle:
        return display

    return None


def chat_user_profile(preferred_name: str | None) -> str:
    """Format the preferred name for the chatbot profile block."""
    if not preferred_name:
        return "Preferred name: not provided"

    safe_name = sanitize_for_prompt(preferred_name, max_length=80).strip()
    if not safe_name:
        return "Preferred name: not provided"

    return f"Preferred name: {safe_name}"


_HUMANIZER_RULES: tuple[str, ...] = (
    (
        "Write like a person at the company: plain, direct sentences, active\n"
        "  voice, natural word order. Keep them short."
    ),
    (
        "Cut stock AI and marketing words: vibrant, rich, testament, crucial,\n"
        "  delve, enhance, foster, showcase, furthermore, moreover, rest\n"
        "  assured, we value your, look forward to, I hope this helps."
    ),
    "No em dashes or en dashes. Use a period, a comma, or parentheses.",
)

# Formatting bullet kept separate so prompts that mandate their own formatting
# (emojis, bold, headings) can pass formatting_rule=None.
_DEFAULT_FORMATTING_RULE: str = "No forced groups of three, no bold, no emojis, no headings, no\n  exclamation marks, no hedging, no filler openers or closers."


def humanizer_rules_block(
    fields: Sequence[str],
    *,
    extra_rules: Sequence[str] = (),
    indent: str = "",
    formatting_rule: str | None = _DEFAULT_FORMATTING_RULE,
) -> str:
    """
    Build the writing-rules block that keeps generated prose human.

    Reusable across prompts that emit free-text fields (contact analysis,
    itinerary generation, chatbot replies).

    Args:
        fields: Output field names the rules apply to, e.g. ("summary",).
        extra_rules: Prompt-specific bullets appended after the core rules.
        indent: Leading whitespace for every line, matching the calling
            prompt's indentation so textwrap.dedent still applies.
        formatting_rule: The formatting bullet appended after the core rules,
            or None when the calling prompt owns its own formatting rules.

    Returns:
        The rules block as a string.
    """
    parts = [f'"{field}"' for field in fields]
    if not parts:
        listed = "the free-text fields"
    elif len(parts) == 1:
        listed = parts[0]
    else:
        listed = f"{', '.join(parts[:-1])} and {parts[-1]}"

    lines = [
        f"Writing rules for {listed}",
        "(humans read these; they must not sound AI-generated):",
    ]
    lines.extend(f"- {rule}" for rule in _HUMANIZER_RULES)
    if formatting_rule is not None:
        lines.append(f"- {formatting_rule}")
    lines.extend(f"- {rule}" for rule in extra_rules)
    text = "\n".join(lines)
    return "\n".join(indent + line for line in text.split("\n"))


@overload
def append_system_message(
    messages: MistralParams,
    content: str,
    *,
    is_mistral: Literal[True],
) -> None: ...


@overload
def append_system_message(
    messages: ListParams,
    content: str,
    *,
    is_mistral: Literal[False],
) -> None: ...


def append_system_message(
    messages: ListParams | MistralParams,
    content: str,
    *,
    is_mistral: bool,
) -> None:
    if is_mistral:
        messages.append(MistralSystem(role="system", content=content))
        return
    messages.append(SystemParam(role="system", content=content))


def append_user_message(
    messages: ListParams | MistralParams,
    content: str,
    *,
    is_mistral: bool,
) -> None:
    if is_mistral:
        messages.append(MistralUser(role="user", content=content))
        return
    messages.append(UserParam(role="user", content=content))
