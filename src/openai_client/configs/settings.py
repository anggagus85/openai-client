# src/openai_client/configs/settings.py

from enum import StrEnum
from pathlib import Path
from typing import TypedDict

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path.cwd() / ".env"
CONFIGS_FILE = Path.cwd() / ".env.configs"


class Settings(BaseSettings):
    """Application settings with validation and default values."""

    DEBUG: bool = False

    model_config = SettingsConfigDict(
        env_file=(ENV_FILE, CONFIGS_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Branding / notification configuration (production defaults)
    LOGO_URL: str = "https://assets.baliroam.com/baliroam/logo-mark-thick-mod_4096x_sgox7l.png"
    COMPANY_TARGET_EMAIL: str = "example@gmail.com"
    TZ: str = "Asia/Makassar"

    # AI Configuration
    INCEPTION_API_KEY: str
    OPENROUTER_API_KEY: str
    NVIDIA_API_KEY: str
    EXA_API_KEY: str
    BRAVE_API_KEY: str
    MISTRAL_API_KEY: str
    ATMOROUTER_API_KEY: str
    # Optional at load time; validated when Providers.DEEPSEEK is selected.
    DEEPSEEK_API_KEY: str | None = None

    # Provider endpoints (overridden by .env.configs when present)
    ATMOROUTER_BASE_URL: str = "https://atmorouter.dev/v1"
    ATMOROUTER_MODEL: str = "atmo/deepseek-v4.1-flash"

    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-flash"

    INCEPTION_BASE_URL: str = "https://api.inceptionlabs.ai/v1"
    INCEPTION_MODEL: str = "mercury-2"

    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "stepfun/step-3.5-flash:free"

    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    NVIDIA_MODEL: str = "stepfun-ai/step-3.5-flash"

    MISTRAL_BASE_URL: str = "https://api.mistral.ai"
    MISTRAL_MODEL: str = "mistral-large-latest"

    AI_REQUEST_TIMEOUT: int = 60  # seconds
    AI_MAX_RETRIES: int = 2
    AI_RETRY_DELAY: float = 1.0  # seconds
    AI_BACKOFF_FACTOR: float = 2.0


settings = Settings()


class Providers(StrEnum):
    ATMOROUTER = "atmorouter"
    DEEPSEEK = "deepseek"
    INCEPTION = "inception"
    OPENROUTER = "openrouter"
    NVIDIA = "nvidia"
    MISTRAL = "mistral"


class ProviderConfig(TypedDict):
    """Keyword arguments accepted by OpenAIClient / MistralAI."""

    # Only None for providers whose key is optional (currently DeepSeek);
    # selection-time validation rejects a missing key.
    api_key: str | None
    base_url: str
    model: str
