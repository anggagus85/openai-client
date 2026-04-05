# src/openai_client/utils/helpers.py

from logging import getLogger
from pathlib import Path

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
    except Exception:
        logger.exception("Failed to save data to file")
        raise
