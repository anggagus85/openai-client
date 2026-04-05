# src/openai_client/configs/settings.py

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path.cwd() / ".env"

WHATSAPP_NUMBER = "+6285847006743"


class Settings(BaseSettings):
    """Application settings with validation and default values."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # AI Configuration
    INCEPTION_API_KEY: str
    OPENROUTER_API_KEY: str
    BYTEZ_API_KEY: str
    NVIDIA_API_KEY: str
    EXA_API_KEY: str
    BRAVE_API_KEY: str

    AI_REQUEST_TIMEOUT: int = 60  # seconds
    AI_MAX_RETRIES: int = 2
    AI_RETRY_DELAY: float = 1.0  # seconds
    AI_BACKOFF_FACTOR: float = 2.0


settings = Settings()
