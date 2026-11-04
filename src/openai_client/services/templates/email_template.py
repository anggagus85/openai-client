"""
Email template builder for HTML email generation with mobile-first responsive design.

Every HTML builder has a matching ``build_*_plain`` counterpart that renders the
same message as a ``text/plain`` alternative (see email-plain-text-spec.md), so
dispatchers can send ``multipart/alternative`` messages.
"""

import re
from html import escape, unescape
from typing import cast

from openai_client.configs import settings

# Patterns used to flatten generated HTML into a plain-text fallback body.
_SCRIPT_OR_STYLE_PATTERN = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_BLOCK_ELEMENT_PATTERN = re.compile(
    r"</?(?:p|div|tr|table|section|article|header|footer|ul|ol|li|h[1-6])\b[^>]*>",
    re.IGNORECASE,
)
_LINE_BREAK_PATTERN = re.compile(r"<br\s*/?>", re.IGNORECASE)
_ANY_TAG_PATTERN = re.compile(r"<[^>]+>")
_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")
_HORIZONTAL_SPACE_PATTERN = re.compile(r"[ \t]+")
_ANCHOR_PATTERN = re.compile(
    r'<a\b[^>]*href\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+))[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)


def _replace_anchor(match: re.Match[str]) -> str:
    """Render an anchor as 'Label: url' per the plain-text spec (rule 1)."""
    url = (match.group(1) or match.group(2) or match.group(3) or "").strip()
    label = _HORIZONTAL_SPACE_PATTERN.sub(" ", _ANY_TAG_PATTERN.sub("", match.group(4))).strip()
    return f"{label}: {url}" if label else url


def html_to_plain_text(html_content: str) -> str:
    """
    Flatten an HTML email body into a plain-text fallback.

    Used for HTML generated outside this module (e.g. LLM-produced bodies) so
    every message can still be sent as multipart/alternative. Anchors are
    rewritten as ``Label: url``; block elements are separated by blank lines.

    Args:
        html_content: The HTML body to convert.

    Returns:
        str: Plain text with tags removed, links preserved and HTML entities decoded.
    """
    text = _ANCHOR_PATTERN.sub(_replace_anchor, html_content)
    text = _SCRIPT_OR_STYLE_PATTERN.sub("", text)
    text = _BLOCK_ELEMENT_PATTERN.sub("\n\n", text)
    text = _LINE_BREAK_PATTERN.sub("\n", text)
    text = _ANY_TAG_PATTERN.sub("", text)
    text = unescape(text)
    text = _HORIZONTAL_SPACE_PATTERN.sub(" ", text)
    text = _BLANK_LINES_PATTERN.sub("\n\n", text)
    return text.strip()


class EmailTemplate:
    """
    Builder for creating professional HTML email templates.

    This class provides methods for constructing responsive, mobile-first HTML emails
    with consistent branding and styling. It supports various email components including
    headers, footers, buttons, security alerts, and info boxes.
    """

    # ==========================================================================
    # Email Template Styles - Reusable CSS Constants
    # ==========================================================================
    EMAIL_STYLES: dict[str, str | dict[str, str]] = {
        # Brand Colors
        "color_primary": "#1a2a6c",
        "color_accent": "#ce6f21",
        "color_warning_bg": "#fff3cd",
        "color_warning_border": "#ffeaa7",
        "color_warning_text": "#856404",
        "color_text_primary": "#333333",
        "color_text_secondary": "#666666",
        "color_text_muted": "#888888",
        "color_text_footer": "#999999",
        "color_bg_page": "#f4f7f9",
        "color_bg_container": "#ffffff",
        "color_bg_footer": "#f9f9f9",
        "color_border": "#eeeeee",
        "color_border_container": "#082a2b",
        # Typography
        "font_stack": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
        "font_size_base_mobile": "16px",
        "font_size_base_desktop": "16px",
        "font_size_h1_mobile": "24px",
        "font_size_h1_desktop": "24px",
        "font_size_small": "14px",
        "font_size_footer": "12px",
        "line_height_base": "1.6",
        # Spacing (Mobile First)
        "padding_container_mobile": "30px",
        "padding_container_desktop": "30px",
        "padding_header_mobile": "12px",
        "padding_header_desktop": "12px",
        "padding_footer": "20px",
        "margin_container_mobile": "40px auto",
        "margin_container_desktop": "40px auto",
        "margin_button_top": "30px",
        "margin_paragraph_top": "16px",
        # Layout
        "max_width_container": "600px",
        "border_radius_large": "12px",
        "border_radius_medium": "8px",
        "border_radius_small": "6px",
        "box_shadow": "0 4px 12px rgba(0, 0, 0, 0.1)",
        "header_background": "#082a2b",
        "logo_max_width": "200px",
        "button_min_height": "44px",  # Touch target size
        # Component-specific styles
        "button": {
            "padding": "12px 21px",
            "border_radius": "6px",
            "font_weight": "bold",
            "text_decoration": "none",
            "display": "inline-block",
            "min_height": "44px",
        },
        "security_box": {
            "padding": "16px",
            "border_radius": "8px",
            "background_color": "#fff3cd",
            "border": "1px solid #ffeaa7",
        },
        "info_box": {
            "padding": "16px",
            "border_radius": "8px",
            "background_color": "#f9f9f9",
            "border_left": "4px solid #ff6b33",
        },
    }

    # Logo URL constant
    LOGO_URL: str = settings.LOGO_URL

    def _get_logo_url(self) -> str:
        """
        Get the logo URL for email templates.

        Returns:
            str: The configured public logo URL.
        """
        return self.LOGO_URL

    def _get_base_template(self) -> str:
        """
        Get the base email template with CSS styles.

        Returns:
            str: The base email template HTML with placeholders.
        """
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>{title}</title>
    <style>
        /* Reset styles */
        body, table, td, p, a, li, blockquote {{
            -webkit-text-size-adjust: 100%;
            -ms-text-size-adjust: 100%;
        }}
        table, td {{
            mso-table-lspace: 0pt;
            mso-table-rspace: 0pt;
        }}
        img {{
            -ms-interpolation-mode: bicubic;
            border: 0;
            height: auto;
            line-height: 100%;
            outline: none;
            text-decoration: none;
        }}
        /* Mobile-first base styles */
        body {{
            font-family: {font_stack};
            background-color: {color_bg_page};
            margin: 0;
            padding: 0;
            color: {color_text_primary};
        }}
        .container {{
            width: 100%;
            max-width: {max_width};
            margin: {margin_mobile};
            background-color: {color_bg_container};
            border-radius: {border_radius};
            overflow: hidden;
            box-shadow: {box_shadow};
            border: 1px solid {color_border_container};
        }}
        .header {{
            background: {header_background};
            padding: {padding_header_mobile};
            text-align: center;
        }}
        .header img {{
            max-width: {logo_width};
            height: auto;
        }}
        .content {{
            padding: {padding_mobile};
            text-align: center;
        }}
        .content h1 {{
            color: {color_primary};
            font-size: {font_h1_mobile};
            margin: 16px 0 20px 0;
            line-height: 1.3;
        }}
        .content p {{
            line-height: {line_height};
            color: {color_text_secondary};
            font-size: {font_base_mobile};
            margin: 0 0 {para_margin} 0;
        }}
        .button-container {{
            margin-top: {button_margin};
        }}
        .button {{
            color: #ffffff !important;
            text-decoration: none;
            font-weight: bold;
            font-size: {font_base_mobile};
            padding: {button_padding};
            border-radius: {button_radius};
            background-color: {color_accent};
            display: inline-block;
            min-height: {button_min_height};
            box-sizing: border-box;
            transition: all 0.3s;
            box-shadow: -1px -1px 1px rgba(255, 255, 255, 0.46), 1px 1px 1px rgba(77, 77, 81, 0.8);
            text-shadow: 2px 1px 4px rgba(34, 33, 33, 0.5);
        }}
        .button:hover {{
            background-color: #ff6b33;
            box-shadow: inset 2px 2px 5px rgba(0, 0, 0, 0.5), inset -2px -2px 5px rgba(255, 253, 225, 0.749);
        }}
        .footer {{
            background-color: {color_bg_footer};
            padding: {padding_footer};
            text-align: center;
            font-size: {font_footer};
            color: {color_text_footer};
            border-top: 1px solid {color_border};
        }}
        .footer p {{
            margin: 5px 0;
            color: {color_text_footer};
        }}
        .security-box {{
            margin: 24px 0 0 0;
            padding: {security_padding};
            background-color: {color_warning_bg};
            border-radius: {security_radius};
            border: {security_border};
            text-align: left;
        }}
        .security-box p {{
            margin: 0;
            font-size: {font_small};
            color: {color_warning_text};
        }}
        .info-box {{
            margin: 24px 0;
            padding: {info_padding};
            background-color: {color_bg_footer};
            border-radius: {info_radius};
            border-left: {info_border_left};
            text-align: left;
        }}
        .info-box p {{
            margin: 0;
            font-size: {font_small};
            color: {color_text_secondary};
        }}
        .content p.expiry-note {{
            margin-top: 30px;
            font-size: {font_small};
            color: {color_text_muted};
        }}
        /* Desktop enhancements */
        @media screen and (min-width: 600px) {{
            .container {{
                margin: {margin_desktop};
            }}
            .header {{
                padding: {padding_header_desktop};
            }}
            .content {{
                padding: {padding_desktop};
            }}
            .content h1 {{
                font-size: {font_h1_desktop};
                margin-bottom: 20px;
            }}
            .content p {{
                font-size: {font_base_desktop};
                margin-bottom: 16px;
            }}
            .button {{
                font-size: {font_base_desktop};
            }}
        }}
        /* Dark mode support */
        @media (prefers-color-scheme: dark) {{
            body {{
                background-color: #1a1a1a;
            }}
            .container {{
                background-color: #2d2d2d;
            }}
            .content h1 {{
                color: #4a90e2;
            }}
            .content p {{
                color: #cccccc;
            }}
            .security-box p {{
                color: {color_warning_text};
            }}
            .info-box p {{
                color: {color_text_secondary};
            }}
            .content p.expiry-note {{
                color: #cccccc;
            }}
            .footer {{
                background-color: #252525;
                border-top-color: #3d3d3d;
            }}
        }}
    </style>
</head>
<body>
    {body_content}
</body>
</html>
"""

    def build_template(
        self,
        title: str,
        header_html: str,
        content_html: str,
        footer_html: str,
    ) -> str:
        """
        Build the complete email template with provided components.

        Args:
            title: The email title for the HTML head.
            header_html: The header section HTML (logo + branding).
            content_html: The main content section HTML.
            footer_html: The footer section HTML.

        Returns:
            str: The complete formatted HTML email.
        """
        body_content = f"""
    <div class="container">
        {header_html}
        {content_html}
        {footer_html}
    </div>
        """.strip()

        styles = self.EMAIL_STYLES
        button_styles = cast("dict[str, str]", styles["button"])
        security_styles = cast("dict[str, str]", styles["security_box"])
        info_styles = cast("dict[str, str]", styles["info_box"])

        return self._get_base_template().format(
            title=title,
            font_stack=styles["font_stack"],
            color_bg_page=styles["color_bg_page"],
            color_text_primary=styles["color_text_primary"],
            max_width=styles["max_width_container"],
            margin_mobile=styles["margin_container_mobile"],
            color_bg_container=styles["color_bg_container"],
            border_radius=styles["border_radius_large"],
            box_shadow=styles["box_shadow"],
            color_border_container=styles["color_border_container"],
            header_background=styles["header_background"],
            padding_header_mobile=styles["padding_header_mobile"],
            logo_width=styles["logo_max_width"],
            padding_mobile=styles["padding_container_mobile"],
            color_primary=styles["color_primary"],
            font_h1_mobile=styles["font_size_h1_mobile"],
            line_height=styles["line_height_base"],
            color_text_secondary=styles["color_text_secondary"],
            font_base_mobile=styles["font_size_base_mobile"],
            para_margin=styles["margin_paragraph_top"],
            button_margin=styles["margin_button_top"],
            button_padding=button_styles["padding"],
            button_radius=button_styles["border_radius"],
            button_min_height=button_styles["min_height"],
            color_accent=styles["color_accent"],
            color_bg_footer=styles["color_bg_footer"],
            padding_footer=styles["padding_footer"],
            font_footer=styles["font_size_footer"],
            color_text_footer=styles["color_text_footer"],
            color_border=styles["color_border"],
            color_warning_bg=styles["color_warning_bg"],
            security_padding=security_styles["padding"],
            security_radius=security_styles["border_radius"],
            security_border=security_styles["border"],
            font_small=styles["font_size_small"],
            color_warning_text=styles["color_warning_text"],
            info_padding=info_styles["padding"],
            info_radius=info_styles["border_radius"],
            info_border_left=info_styles["border_left"],
            margin_desktop=styles["margin_container_desktop"],
            padding_header_desktop=styles["padding_header_desktop"],
            padding_desktop=styles["padding_container_desktop"],
            font_h1_desktop=styles["font_size_h1_desktop"],
            font_base_desktop=styles["font_size_base_desktop"],
            color_text_muted=styles["color_text_muted"],
            body_content=body_content,
        )

    def build_header(self) -> str:
        """
        Build the email header HTML with logo.

        Returns:
            str: The header HTML section.
        """
        logo_url = self._get_logo_url()
        return f"""<div class="header">
            <img src="{logo_url}" alt="BaliRoam Logo">
        </div>"""

    def build_footer(self, year: int) -> str:
        """
        Build the email footer HTML.

        Args:
            year: The current year for copyright.

        Returns:
            str: The footer HTML section.
        """
        return f"""<div class="footer">
            <p>&copy; {year} BaliRoam. All rights reserved.</p>
            <p>Your portal to Bali's finest travel experiences.</p>
        </div>"""

    def build_button(self, url: str, text: str) -> str:
        """
        Build a styled CTA button HTML.

        Args:
            url: The button link URL.
            text: The button text.

        Returns:
            str: The button HTML.
        """
        return f"""<div class="button-container">
            <a href="{escape(url)}" class="button">{escape(text)}</a>
        </div>"""

    def build_security_box(self, message: str) -> str:
        """
        Build a security alert box HTML.

        Args:
            message: The HTML content of the security message.

        Returns:
            str: The security box HTML.
        """
        return f"""<div class="security-box">
            <p>{message}</p>
        </div>"""

    def build_info_box(self, content: str) -> str:
        """
        Build an info box HTML with accent border.

        Args:
            content: The HTML content for the info box.

        Returns:
            str: The info box HTML.
        """
        return f"""<div class="info-box">
            <p>{content}</p>
        </div>"""

    def build_content_wrapper(
        self,
        heading: str,
        paragraphs: list[str],
        button_html: str | None = None,
        expiry_note: str | None = None,
        extra_html: str | None = None,
    ) -> str:
        """
        Build the main content section with consistent structure.

        Args:
            heading: The main heading (h1) text.
            paragraphs: List of paragraph HTML strings (caller must escape user input).
            button_html: Optional button HTML to include.
            expiry_note: Optional expiry note HTML (caller must escape user input).
            extra_html: Optional additional HTML (security boxes, etc.).

        Returns:
            str: The complete content section HTML.
        """
        paragraphs_html = "\n            ".join(f"<p>{p}</p>" for p in paragraphs)

        parts = [
            '<div class="content">',
            f"            <h1>{heading}</h1>",
            f"            {paragraphs_html}",
        ]

        if button_html:
            parts.append(f"            {button_html}")

        if expiry_note:
            parts.append(f'            <p class="expiry-note">{expiry_note}</p>')

        if extra_html:
            parts.append(f"            {extra_html}")

        parts.append("        </div>")

        return "\n        ".join(parts)

    def build_verification_email(
        self,
        greet_name: str,
        verification_link: str,
        expiry_hours: int,
        year: int,
    ) -> str:
        """
        Build the complete verification email HTML.

        Args:
            greet_name: The user's greeting name.
            verification_link: The email verification link.
            expiry_hours: Hours until the link expires.
            year: The current year for copyright.

        Returns:
            str: The complete HTML email.
        """
        header_html = self.build_header()
        footer_html = self.build_footer(year)
        button_html = self.build_button(verification_link, "Verify Email Address")

        content_html = self.build_content_wrapper(
            heading=f"Welcome to BaliRoam, {escape(greet_name)}!",
            paragraphs=[
                (
                    "Thank you for embarking on your journey with us. To ensure the security "
                    "of your account and access all our travel features, please verify your "
                    "email address by clicking the button below."
                ),
            ],
            button_html=button_html,
            expiry_note=f"This link will expire in {expiry_hours} hours.<br>"
            "If you didn't create an account, you can safely ignore this email.",
        )

        return self.build_template(
            title="Verify Your BaliRoam Account",
            header_html=header_html,
            content_html=content_html,
            footer_html=footer_html,
        )

    def build_password_reset_email(
        self,
        greet_name: str,
        reset_link: str,
        expiry_hours: int,
        year: int,
    ) -> str:
        """
        Build the complete password reset email HTML.

        Args:
            greet_name: The user's greeting name.
            reset_link: The password reset link.
            expiry_hours: Hours until the link expires.
            year: The current year for copyright.

        Returns:
            str: The complete HTML email.
        """
        header_html = self.build_header()
        footer_html = self.build_footer(year)
        button_html = self.build_button(reset_link, "Reset Password")

        security_message = (
            "<strong>Security Tip:</strong><br>"
            "Never share this link with anyone. BaliRoam will never ask for your password via email."
        )
        security_box_html = self.build_security_box(security_message)

        content_html = self.build_content_wrapper(
            heading="Reset Your Password",
            paragraphs=[
                f"Hi {escape(greet_name)},",
                (
                    "We received a request to reset your BaliRoam account password. "
                    "Click the button below to create a new password:"
                ),
            ],
            button_html=button_html,
            expiry_note=f"This link will expire in {expiry_hours} hour(s).<br>"
            "If you didn't request this reset, you can safely ignore this email.",
            extra_html=security_box_html,
        )

        return self.build_template(
            title="Reset Your BaliRoam Password",
            header_html=header_html,
            content_html=content_html,
            footer_html=footer_html,
        )

    def build_password_change_email(
        self,
        greet_name: str,
        change_time: str,
        year: int,
    ) -> str:
        """
        Build the complete password change confirmation email HTML.

        Args:
            greet_name: The user's greeting name.
            change_time: The formatted change timestamp.
            year: The current year for copyright.

        Returns:
            str: The complete HTML email.
        """
        header_html = self.build_header()
        footer_html = self.build_footer(year)

        info_content = f"<strong>Change Time:</strong> {escape(change_time)}"
        info_box_html = self.build_info_box(info_content)

        escaped_email = escape(settings.COMPANY_TARGET_EMAIL)
        security_message = (
            "<strong>Didn't make this change?</strong><br>"
            "If you didn't change your password, please contact us immediately at "
            f'<a href="mailto:{escaped_email}" style="color: #1a2a6c;">'
            f"{escaped_email}</a> to secure your account."
        )
        security_box_html = self.build_security_box(security_message)

        content_html = self.build_content_wrapper(
            heading="Password Changed Successfully",
            paragraphs=[
                f"Hi {escape(greet_name)},",
                "Your BaliRoam account password was changed successfully.",
                "You will need to sign in again on all other devices with your new password.",
            ],
            extra_html=f"{info_box_html}\n            {security_box_html}",
        )

        return self.build_template(
            title="Your BaliRoam Password Was Changed",
            header_html=header_html,
            content_html=content_html,
            footer_html=footer_html,
        )

    def build_review_confirmation_email(
        self,
        guest_name: str,
        year: int,
    ) -> str:
        """
        Build the review submission confirmation email HTML.

        Sends a branded confirmation to a guest who submitted a review,
        with a disclaimer for recipients who did not initiate the review.

        Args:
            guest_name: The guest's display name (will be escaped).
            year: The current year for copyright.

        Returns:
            str: The complete HTML email.
        """
        header_html = self.build_header()
        footer_html = self.build_footer(year)

        info_message = (
            "<strong>Didn't submit a review?</strong><br>"
            "If you did not recently submit a review on BaliRoam, "
            "please disregard this email &mdash; no action is required on your part. "
            "Your email address may have been entered by mistake."
        )
        info_box_html = self.build_info_box(info_message)

        content_html = self.build_content_wrapper(
            heading=f"Thank You, {escape(guest_name)}!",
            paragraphs=[
                (
                    "We've received your review submission and truly appreciate you "
                    "taking the time to share your experience with us."
                ),
                (
                    "Our team will review your feedback shortly. Your insights help us "
                    "improve and assist other travelers in planning their perfect Bali adventure."
                ),
            ],
            extra_html=info_box_html,
        )

        return self.build_template(
            title="Thank You for Your Review — BaliRoam",
            header_html=header_html,
            content_html=content_html,
            footer_html=footer_html,
        )

    # ======================================================================
    # Plain-text alternatives (multipart/alternative fallback bodies)
    # See email-plain-text-spec.md — sections 2 and 3.
    # ======================================================================

    def build_plain_footer(self, year: int) -> str:
        """
        Build the shared plain-text footer block.

        Args:
            year: The current year for copyright.

        Returns:
            str: The plain-text footer, terminated by a newline.
        """
        return (
            f"---\n© {year} BaliRoam. All rights reserved.\n"
            "Your portal to Bali's finest travel experiences.\n"
        )

    @staticmethod
    def _expiry_phrase(expiry_hours: int) -> str:
        """Render an expiry duration with correct pluralization."""
        hour_word = "hour" if expiry_hours == 1 else "hours"
        return f"{expiry_hours} {hour_word}"

    def build_verification_email_plain(
        self,
        greet_name: str,
        verification_link: str,
        expiry_hours: int,
        year: int,
    ) -> str:
        """
        Build the plain-text alternative of the verification email.

        Args:
            greet_name: The user's greeting name.
            verification_link: The email verification link.
            expiry_hours: Hours until the link expires.
            year: The current year for copyright.

        Returns:
            str: The complete text/plain email body.
        """
        return (
            f"Welcome to BaliRoam, {greet_name}!\n\n"
            "Thank you for embarking on your journey with us. To ensure the security "
            "of your account and access all our travel features, please verify your "
            "email address by visiting the link below:\n\n"
            f"{verification_link}\n\n"
            f"Note: This link will expire in {self._expiry_phrase(expiry_hours)}. "
            "If you didn't create an account, you can safely ignore this email.\n\n"
            f"{self.build_plain_footer(year)}"
        )

    def build_password_reset_email_plain(
        self,
        greet_name: str,
        reset_link: str,
        expiry_hours: int,
        year: int,
    ) -> str:
        """
        Build the plain-text alternative of the password reset email.

        Args:
            greet_name: The user's greeting name.
            reset_link: The password reset link.
            expiry_hours: Hours until the link expires.
            year: The current year for copyright.

        Returns:
            str: The complete text/plain email body.
        """
        return (
            "Reset Your Password\n\n"
            f"Hi {greet_name},\n\n"
            "We received a request to reset your BaliRoam account password. "
            "Use the link below to create a new password:\n\n"
            f"{reset_link}\n\n"
            f"This link will expire in {self._expiry_phrase(expiry_hours)}. "
            "If you didn't request this reset, you can safely ignore this email.\n\n"
            "[SECURITY TIP]\n"
            "Never share this link with anyone. BaliRoam will never ask for your password "
            "via email.\n\n"
            f"{self.build_plain_footer(year)}"
        )

    def build_password_change_email_plain(
        self,
        greet_name: str,
        change_time: str,
        year: int,
    ) -> str:
        """
        Build the plain-text alternative of the password change confirmation email.

        Args:
            greet_name: The user's greeting name.
            change_time: The formatted change timestamp.
            year: The current year for copyright.

        Returns:
            str: The complete text/plain email body.
        """
        return (
            "Password Changed Successfully\n\n"
            f"Hi {greet_name},\n\n"
            "Your BaliRoam account password was changed successfully. "
            "You will need to sign in again on all other devices with your new password.\n\n"
            f"Change Time: {change_time}\n\n"
            "[IMPORTANT NOTICE]\n"
            "If you didn't make this change, please contact us immediately at "
            f"{settings.COMPANY_TARGET_EMAIL} to secure your account.\n\n"
            f"{self.build_plain_footer(year)}"
        )

    def build_review_confirmation_email_plain(
        self,
        guest_name: str,
        year: int,
    ) -> str:
        """
        Build the plain-text alternative of the review confirmation email.

        Args:
            guest_name: The guest's display name.
            year: The current year for copyright.

        Returns:
            str: The complete text/plain email body.
        """
        return (
            f"Thank You, {guest_name}!\n\n"
            "We've received your review submission and truly appreciate you "
            "taking the time to share your experience with us.\n\n"
            "Our team will review your feedback shortly. Your insights help us "
            "improve and assist other travelers in planning their perfect Bali adventure.\n\n"
            "[NOTE]\n"
            "If you did not recently submit a review on BaliRoam, please disregard "
            "this email — no action is required on your part.\n\n"
            f"{self.build_plain_footer(year)}"
        )
