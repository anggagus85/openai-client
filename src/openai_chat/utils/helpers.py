# src/openai_chat/utils/helpers.py

from logging import getLogger
from pathlib import Path

from dotenv import dotenv_values

from openai_chat.errors import SYSTEM_EXCEPTIONS

logger = getLogger(__name__)
_env = dotenv_values(verbose=True)


def get_api_key(key: str) -> str:
    if api_key := _env.get(key):
        return api_key
    return ""


def save_to_file(data: str, file_path: Path) -> None:
    """
    Write the itinerary to a file.

    Args:
        data: The data string to write.
        file_path: The path to the file to write to.
    """

    try:
        with file_path.open("w") as f:
            f.write(data)
        logger.info(f"Data saved to file: {file_path}")
    except SYSTEM_EXCEPTIONS:
        logger.exception("Failed to save data to file")
        raise
    except Exception:
        logger.exception("Failed to save data to file")
        raise
