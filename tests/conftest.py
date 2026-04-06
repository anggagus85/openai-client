# tests/conftest.py

import pytest

from openai_client.services.chatbot import Chatbot


@pytest.fixture
def mock_chatbot() -> Chatbot:
    mock_client = type("MockClient", (), {"messages": []})()
    chatbot = Chatbot.__new__(Chatbot)
    chatbot._client = mock_client
    chatbot._messages = []
    return chatbot
