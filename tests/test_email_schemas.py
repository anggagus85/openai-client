# tests/test_email_schemas.py

"""Tests for email inquiry schemas and validation."""

import pytest
from pydantic import ValidationError

from openai_client.schemas.ai.email import (
    AnalysisFormat,
    ContactAnalysisResponse,
    EmailInquiry,
    EmailResponse,
)


class TestEmailInquiry:
    def test_valid_inquiry(self) -> None:
        inquiry = EmailInquiry(
            name="John Doe",
            subject="Travel Inquiry",
            message="I would like to book a trip to Bali for 7 days.",
            email="john@example.com",
        )
        assert inquiry.name == "John Doe"
        assert inquiry.subject == "Travel Inquiry"
        assert inquiry.email == "john@example.com"

    def test_name_sanitization(self) -> None:
        inquiry = EmailInquiry(
            name='<script>alert("xss")</script> John',
            message="I would like to book a trip to Bali for 7 days.",
            email="john@example.com",
        )
        assert "<script>" not in inquiry.name

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least 1 character"):
            EmailInquiry(
                name="",
                message="I would like to book a trip to Bali for 7 days.",
                email="john@example.com",
            )

    def test_disposable_email_raises(self) -> None:
        with pytest.raises(ValidationError, match="temporary mail service"):
            EmailInquiry(
                name="John",
                message="I would like to book a trip to Bali for 7 days.",
                email="john@mailinator.com",
            )

    def test_message_sanitization(self) -> None:
        inquiry = EmailInquiry(
            name="John",
            message="I want to visit <b>Bali</b> for 7 days with my family.",
            email="john@example.com",
        )
        assert "<b>" not in inquiry.message

    def test_short_message_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least 10 characters"):
            EmailInquiry(
                name="John",
                message="Hi",
                email="john@example.com",
            )

    def test_empty_message_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least 1 character"):
            EmailInquiry(
                name="John",
                message="",
                email="john@example.com",
            )

    def test_invalid_email_format_raises(self) -> None:
        with pytest.raises(ValidationError):
            EmailInquiry(
                name="John",
                message="I would like to book a trip to Bali for 7 days.",
                email="not-an-email",
            )


class TestEmailResponse:
    def test_default_values(self) -> None:
        resp = EmailResponse()
        assert resp.status == "success"
        assert resp.message == "Email sent successfully"

    def test_custom_values(self) -> None:
        resp = EmailResponse(status="error", message="Failed to send")
        assert resp.status == "error"
        assert resp.message == "Failed to send"


class TestAnalysisFormat:
    def test_default_values(self) -> None:
        fmt = AnalysisFormat(name="John")
        assert fmt.name == "John"
        assert fmt.summary == "Analysis of inquiry"
        assert fmt.category == "General Information"
        assert fmt.urgency == "Medium"
        assert fmt.keywords == []

    def test_custom_values(self) -> None:
        fmt = AnalysisFormat(
            name="Jane",
            summary="Custom summary",
            category="Booking",
            urgency="High",
            suggested_reply="Hello!",
            required_action="Book now",
            keywords=["bali", "beach"],
        )
        assert fmt.summary == "Custom summary"
        assert fmt.category == "Booking"
        assert fmt.keywords == ["bali", "beach"]


class TestContactAnalysisResponse:
    def test_valid_response(self) -> None:
        resp = ContactAnalysisResponse(confirmation="Your inquiry has been received.")
        assert resp.confirmation == "Your inquiry has been received."

    def test_absent_confirmation_raises(self) -> None:
        with pytest.raises(ValidationError):
            ContactAnalysisResponse()  # type: ignore

    def test_empty_string_confirmation(self) -> None:
        resp = ContactAnalysisResponse(confirmation="")
        assert resp.confirmation == ""

    def test_very_long_confirmation(self) -> None:
        long_text = "A" * 10000
        resp = ContactAnalysisResponse(confirmation=long_text)
        assert resp.confirmation == long_text
        assert len(resp.confirmation) == 10000

    def test_wrong_type_confirmation_raises(self) -> None:
        with pytest.raises(ValidationError):
            ContactAnalysisResponse(confirmation=12345)  # type: ignore
