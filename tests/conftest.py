# tests/conftest.py

from unittest.mock import AsyncMock, MagicMock

import pytest

from openai_client.services.chatbot import Chatbot
from openai_client.typings import SystemParam


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.is_mistral = False
    client.generate = AsyncMock(return_value="Bali answer")
    return client


@pytest.fixture
def system_param() -> SystemParam:
    return SystemParam(role="system", content="test system")


@pytest.fixture
def mock_chatbot(mock_client: MagicMock, system_param: SystemParam) -> Chatbot:
    return Chatbot(mock_client, system_param)
