# tests/test_email_inquiry.py

"""Tests for email inquiry service functions."""

from datetime import datetime
from unittest.mock import MagicMock, patch

from openai_client.schemas.ai import (
    AnalysisFormat,
    ContactAnalysisResponse,
    EmailInquiry,
)
from openai_client.services.email_inquiry import (
    _confirmation_message_prompt,
    _contact_analysis_prompt,
    _create_body,
    _email_messages,
    analyze_contact,
    confirmation_message,
)


class TestContactAnalysisPrompt:
    def test_contains_name_and_message(self) -> None:
        result = _contact_analysis_prompt("John", "I want to book a trip")
        assert "John" in result
        assert "I want to book a trip" in result

    def test_contains_json_format_instruction(self) -> None:
        result = _contact_analysis_prompt("Jane", "Hello")
        assert "JSON" in result
        assert '"summary"' in result
        assert '"category"' in result


class TestEmailMessages:
    def test_returns_two_messages(self) -> None:
        messages = _email_messages("Test content")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test content"


class TestCreateBody:
    def test_contains_inquiry_details(self) -> None:
        inquiry = EmailInquiry(
            name="John Doe",
            message="I want to visit Bali",
            email="john@example.com",
        )
        analysis = AnalysisFormat(name="John Doe")
        body = _create_body(inquiry, analysis)
        assert "John Doe" in body
        assert "john@example.com" in body
        assert "I want to visit Bali" in body

    def test_contains_analysis(self) -> None:
        inquiry = EmailInquiry(
            name="Jane",
            message="I need help",
            email="jane@example.com",
        )
        analysis = AnalysisFormat(
            name="Jane",
            summary="Help request",
            category="Support Request",
            urgency="High",
            keywords=["help", "urgent"],
        )
        body = _create_body(inquiry, analysis)
        assert "Help request" in body
        assert "Support Request" in body
        assert "High" in body
        assert "help, urgent" in body

    def test_contains_html_structure(self) -> None:
        inquiry = EmailInquiry(
            name="Test",
            message="Test message",
            email="test@example.com",
        )
        analysis = AnalysisFormat(name="Test")
        body = _create_body(inquiry, analysis)
        assert "<!DOCTYPE html>" in body
        assert "</html>" in body
        assert "BaliBlissed" in body

    def test_create_body_escapes_html_in_message(self) -> None:
        """Verify _create_body escapes HTML; uses a fake inquiry to bypass EmailInquiry validators."""
        fake_inquiry = EmailInquiry(
            name="Test",
            message="This is Bold message text here",
            email="test@example.com",
        )
        analysis = AnalysisFormat(name="Test")
        body = _create_body(fake_inquiry, analysis)
        assert "<b>" not in body

    def test_contains_current_year(self) -> None:
        inquiry = EmailInquiry(
            name="Test",
            message="This is a test message for validation",
            email="test@example.com",
        )
        analysis = AnalysisFormat(name="Test")
        body = _create_body(inquiry, analysis)
        assert str(datetime.now().year) in body

    def test_handles_keywords_as_string(self) -> None:
        inquiry = EmailInquiry(
            name="Test",
            message="This is a test message for validation",
            email="test@example.com",
        )
        analysis = AnalysisFormat(name="Test", keywords=["keyword1", "keyword2"])
        body = _create_body(inquiry, analysis)
        assert "keyword1, keyword2" in body


class TestConfirmationMessagePrompt:
    def test_sent_scenario(self) -> None:
        inquiry = EmailInquiry(
            name="John",
            message="I would like to inquire about your services",
            email="john@example.com",
        )
        result = _confirmation_message_prompt(inquiry, sent=True)
        assert "John" in result
        assert "sent successfully" in result

    def test_failed_scenario(self) -> None:
        inquiry = EmailInquiry(
            name="Jane",
            message="I would like to inquire about your services",
            email="jane@example.com",
        )
        result = _confirmation_message_prompt(inquiry, sent=False)
        assert "Jane" in result
        assert "failed to send" in result
        assert "technical issue" in result


class TestAnalyzeContact:
    def test_successful_analysis(self) -> None:
        inquiry = EmailInquiry(
            name="John",
            message="I want to book a trip to Bali",
            email="john@example.com",
        )
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"

        with patch(
            "openai_client.services.email_inquiry.caller",
            return_value={
                "completion": lambda: (
                    '{"name":"John","summary":"Booking request","category":"Booking Inquiry","urgency":"Medium","suggested_reply":"Hello","required_action":"Book it","keywords":["bali"]}'
                ),
            },
        ):
            result = analyze_contact(inquiry, mock_client, "completion")
            assert "John" in result
            assert "Booking request" in result
            assert "Booking Inquiry" in result

    def test_fallback_on_invalid_json(self) -> None:
        inquiry = EmailInquiry(
            name="Jane",
            message="I would like to inquire about your services",
            email="jane@example.com",
        )
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"

        with patch(
            "openai_client.services.email_inquiry.caller",
            return_value={"completion": lambda: "not valid json"},
        ):
            result = analyze_contact(inquiry, mock_client, "completion")
            assert "Jane" in result
            assert "General Information" in result

    def test_fallback_on_validation_error(self) -> None:
        inquiry = EmailInquiry(
            name="Test",
            message="I would like to inquire about your services",
            email="test@example.com",
        )
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"

        with patch(
            "openai_client.services.email_inquiry.caller",
            return_value={"completion": lambda: '{"summary":"test"}'},
        ):
            result = analyze_contact(inquiry, mock_client, "completion")
            assert "Test" in result


class TestConfirmationMessage:
    def test_successful_confirmation(self) -> None:
        inquiry = EmailInquiry(
            name="John",
            message="I would like to inquire about your services",
            email="john@example.com",
        )
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"

        with patch(
            "openai_client.services.email_inquiry.caller",
            return_value={"completion": lambda: "Thank you for your inquiry."},
        ):
            result = confirmation_message(inquiry, mock_client, "completion")
            assert isinstance(result, ContactAnalysisResponse)
            assert result.confirmation == "Thank you for your inquiry."

    def test_fallback_on_validation_error(self) -> None:
        inquiry = EmailInquiry(
            name="Jane",
            message="I would like to inquire about your services",
            email="jane@example.com",
        )
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"

        with patch(
            "openai_client.services.email_inquiry.caller",
            return_value={"completion": lambda: None},
        ):
            result = confirmation_message(inquiry, mock_client, "completion")
            assert isinstance(result, ContactAnalysisResponse)
            assert "Jane" in result.confirmation

    def test_failed_sent_flag(self) -> None:
        inquiry = EmailInquiry(
            name="Test",
            message="I would like to inquire about your services",
            email="test@example.com",
        )
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.base_url = "https://api.example.com"

        with patch(
            "openai_client.services.email_inquiry.caller",
            return_value={"completion": lambda: None},
        ):
            result = confirmation_message(inquiry, mock_client, "completion", sent=False)
            assert "failed to send" in result.confirmation
