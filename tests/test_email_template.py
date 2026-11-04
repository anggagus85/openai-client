# tests/test_email_template.py

"""Tests for the ported email template system."""

from html import escape

from openai_client.configs import settings
from openai_client.services.templates import EmailTemplate, html_to_plain_text


class TestLogo:
    def test_header_uses_configured_logo(self) -> None:
        header = EmailTemplate().build_header()
        assert settings.LOGO_URL in header
        assert 'alt="BaliRoam Logo"' in header

    def test_verification_email_contains_logo(self) -> None:
        body = EmailTemplate().build_verification_email(
            greet_name="Kusuma",
            verification_link="https://baliroam.com/verify?token=1",
            expiry_hours=24,
            year=2026,
        )
        assert settings.LOGO_URL in body
        assert "Verify Email Address" in body
        assert "Kusuma" in body


class TestButton:
    def test_preserves_percent_escape_and_fragment(self) -> None:
        url = "https://baliroam.com/verify?token=abc%2Fdef&next=/trips#section-2"
        html = EmailTemplate().build_button(url, "Verify & Go")

        assert (
            'href="https://baliroam.com/verify?token=abc%2Fdef&amp;next=/trips#section-2"' in html
        )
        assert "%2F" in html
        assert "#section-2" in html
        assert "Verify &amp; Go" in html

    def test_escapes_attribute_breaking_characters(self) -> None:
        html = EmailTemplate().build_button('https://example.com/"bad', "Click")
        assert 'href="https://example.com/&quot;bad"' in html
        assert '"bad"' not in html


class TestPlainTextFallback:
    def test_verification_plain_text(self) -> None:
        body = EmailTemplate().build_verification_email_plain(
            greet_name="Kusuma",
            verification_link="https://baliroam.com/verify?token=1",
            expiry_hours=1,
            year=2026,
        )
        assert "Welcome to BaliRoam, Kusuma!" in body
        assert "1 hour" in body
        assert "© 2026 BaliRoam" in body

    def test_html_to_plain_text_preserves_links(self) -> None:
        html = '<p>Visit <a href="https://baliroam.com">BaliRoam</a> today.</p>'
        text = html_to_plain_text(html)
        assert "BaliRoam: https://baliroam.com" in text

    def test_html_to_plain_text_decodes_entities(self) -> None:
        html = f"<p>{escape('Bali & Java')}</p>"
        assert html_to_plain_text(html) == "Bali & Java"
