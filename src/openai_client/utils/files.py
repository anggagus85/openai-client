"""File-writing helpers."""

from logging import getLogger

from anyio import Path

from openai_client.errors import SYSTEM_EXCEPTIONS, AppSystemError

logger = getLogger(__name__)

async def save_to_file(data: str, file_path: Path) -> None:
    """
    Write string data to a file.

    Args:
        data: The data string to write.
        file_path: The (anyio) path to the file to write.
    """
    try:
        await file_path.write_text(data, encoding="utf-8")
        logger.info("Data saved to '%s'", file_path)
    except SYSTEM_EXCEPTIONS as e:
        details = f"Failed to save data to file: {e}"
        logger.exception(details)
        raise
    except Exception as e:
        details = f"Failed to save data to file: {e}"
        raise AppSystemError(details) from e


async def save_itinerary_to_file(content: str, whatsapp_text: str) -> None:
    """Save the markdown and WhatsApp variants to the current directory."""
    try:
        cwd = await Path.cwd()
        await save_to_file(content, cwd / "itinerary.md")
        await save_to_file(whatsapp_text, cwd / "itinerary.txt")
    except OSError:
        logger.exception("Failed to save itinerary files")
        raise
