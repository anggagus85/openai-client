# tests/test_chatbot_schemas.py

"""Tests for chatbot schemas and validation."""

import pytest
from pydantic import ValidationError

from openai_client.schemas.ai.chatbot import (
    ChatResponse,
    OpenAIChatMessage,
    OpenAIChatRequest,
)


class TestOpenAIChatMessage:
    def test_valid_user_message(self) -> None:
        msg = OpenAIChatMessage(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_valid_assistant_message(self) -> None:
        msg = OpenAIChatMessage(role="assistant", content="Hi there!")
        assert msg.role == "assistant"

    def test_valid_system_message(self) -> None:
        msg = OpenAIChatMessage(role="system", content="You are helpful.")
        assert msg.role == "system"

    def test_invalid_role_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            OpenAIChatMessage(role="admin", content="test")
        assert any(e["loc"] == ("role",) for e in exc_info.value.errors())

    def test_extra_fields_ignored(self) -> None:
        msg = OpenAIChatMessage.model_validate(
            {"role": "user", "content": "Hi", "extra_field": "ignored"},
        )
        assert not hasattr(msg, "extra_field")


class TestOpenAIChatRequest:
    def test_valid_request(self) -> None:
        req = OpenAIChatRequest(query="What are the best beaches in Bali?")
        assert req.query == "What are the best beaches in Bali?"
        assert req.history == []

    def test_request_with_history(self) -> None:
        req = OpenAIChatRequest(
            query="Tell me more",
            history=[
                OpenAIChatMessage(role="user", content="What about Bali?"),
                OpenAIChatMessage(role="assistant", content="Bali is great!"),
            ],
        )
        assert len(req.history) == 2

    def test_empty_query_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least 1 character"):
            OpenAIChatRequest(query="")

    def test_whitespace_query_raises(self) -> None:
        with pytest.raises(ValidationError, match="cannot be empty"):
            OpenAIChatRequest(query="   ")

    def test_query_stripped(self) -> None:
        req = OpenAIChatRequest(query="  Hello  ")
        assert req.query == "Hello"

    def test_query_too_long_raises(self) -> None:
        with pytest.raises(ValidationError):
            OpenAIChatRequest(query="x" * 1001)


class TestChatResponse:
    def test_valid_response(self) -> None:
        resp = ChatResponse(answer="Bali has many beautiful beaches.")
        assert resp.answer == "Bali has many beautiful beaches."
