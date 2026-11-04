# tests/test_chatbot_schemas.py

"""Tests for the production chat schemas and validation."""

import pytest
from pydantic import ValidationError

from openai_client.schemas.ai.chatbot import (
    MAX_HISTORY_PARTS,
    ChatMessage,
    ChatPart,
    ChatRequest,
    ChatResponse,
)


class TestChatPart:
    def test_default_text_is_none(self) -> None:
        assert ChatPart().text is None

    def test_text_value(self) -> None:
        assert ChatPart(text="Hello").text == "Hello"

    def test_extra_fields_ignored(self) -> None:
        part = ChatPart.model_validate({"text": "Hi", "extra": "ignored"})
        assert part.text == "Hi"
        assert not hasattr(part, "extra")


class TestChatMessage:
    def test_valid_user_message(self) -> None:
        msg = ChatMessage(role="user", parts=[ChatPart(text="Hello!")])
        assert msg.role == "user"
        assert msg.parts[0].text == "Hello!"

    def test_model_role_accepted(self) -> None:
        msg = ChatMessage(role="model", parts=[ChatPart(text="Hi there!")])
        assert msg.role == "model"

    def test_invalid_role_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(role="admin", parts=[ChatPart(text="test")])
        assert any(e["loc"] == ("role",) for e in exc_info.value.errors())

    def test_requires_at_least_one_part(self) -> None:
        with pytest.raises(ValidationError, match="at least 1 item"):
            ChatMessage(role="user", parts=[])

    def test_rejects_too_many_parts(self) -> None:
        parts = [ChatPart(text="x") for _ in range(MAX_HISTORY_PARTS + 1)]
        with pytest.raises(ValidationError, match="at most"):
            ChatMessage(role="user", parts=parts)


class TestChatRequest:
    def test_valid_request(self) -> None:
        req = ChatRequest(query="What are the best beaches in Bali?")
        assert req.query == "What are the best beaches in Bali?"
        assert req.history == []

    def test_request_with_history(self) -> None:
        req = ChatRequest(
            query="Tell me more",
            history=[
                ChatMessage(role="user", parts=[ChatPart(text="What about Bali?")]),
                ChatMessage(role="model", parts=[ChatPart(text="Bali is great!")]),
            ],
        )
        assert len(req.history) == 2
        assert req.history[1].role == "model"

    def test_empty_query_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least 1 character"):
            ChatRequest(query="")

    def test_whitespace_query_raises(self) -> None:
        with pytest.raises(ValidationError, match="Query cannot be empty"):
            ChatRequest(query="   ")

    def test_query_is_stripped(self) -> None:
        req = ChatRequest(query="  Hello  ")
        assert req.query == "Hello"

    def test_query_too_long_raises(self) -> None:
        with pytest.raises(ValidationError, match="at most 1000 characters"):
            ChatRequest(query="x" * 1001)


class TestChatResponse:
    def test_requires_answer(self) -> None:
        resp = ChatResponse(answer="Hello")
        assert resp.answer == "Hello"
