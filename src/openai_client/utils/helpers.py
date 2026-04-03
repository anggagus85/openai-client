# src/openai_client/utils/helpers.py

from logging import getLogger
from pathlib import Path

from dotenv import dotenv_values

logger = getLogger(__name__)
_env = dotenv_values(verbose=True)


def get_api_key(key: str) -> str:
    if api_key := _env.get(key):
        return api_key
    return ""


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
        logger.info(f"Data saved to file: {file_path}")
    except Exception:
        logger.exception("Failed to save data to file")
        raise
