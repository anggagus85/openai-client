"""Helpers for deterministic async stream cleanup."""

from inspect import isawaitable
from logging import getLogger
from typing import Any

logger = getLogger(__name__)


async def close_async_stream(stream: object | None) -> None:
    """
    Close an async stream regardless of its closing convention.

    OpenAI ``AsyncStream`` exposes ``aclose()``, Mistral ``EventStreamAsync``
    exposes an async ``close()``, and some iterators expose neither. Cleanup
    failures are logged and swallowed so they never mask the original
    streaming outcome.
    """
    if stream is None:
        return

    close = getattr(stream, "aclose", None)
    if not callable(close):
        close = getattr(stream, "close", None)
    if not callable(close):
        return

    try:
        result: Any = close()
        if isawaitable(result):
            await result
    except Exception:  # noqa: BLE001
        logger.warning("Failed to close async stream", exc_info=True)
