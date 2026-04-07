# tests/conftest.py

import pytest

from openai_client.services.chatbot import Chatbot
from openai_client.types import SystemParam


@pytest.fixture
def mock_chatbot() -> Chatbot:
    mock_client = type("MockClient", (), {"messages": []})()
    system_param = SystemParam(role="system", content="test system")
    chatbot = Chatbot.__new__(Chatbot)
    chatbot._client = mock_client
    chatbot._messages = [system_param]
    return chatbot
