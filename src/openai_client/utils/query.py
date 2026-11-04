"""Helpers for normalizing and truncating search queries."""


def fit_query(query: str, limit: int) -> str:
    """
    Collapse whitespace and truncate a query to at most ``limit`` characters.

    If the cutoff lands exactly on a word boundary (the next character is a
    space), keep the complete final word. Otherwise, drop the partial word.
    """
    compact = " ".join(query.split())
    if len(compact) <= limit:
        return compact

    if compact[limit] == " ":
        return compact[:limit]

    truncated = compact[:limit]
    return truncated.rsplit(" ", 1)[0] or truncated
