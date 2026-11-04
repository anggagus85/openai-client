"""Reusable HTML email components and document assembly."""

from html import escape
from typing import cast

from openai_client.services.templates.base_template import BASE_TEMPLATE
from openai_client.services.templates.styles import _TemplateStyles


class _HtmlTemplateMixin(_TemplateStyles):
    """Reusable HTML email components."""

    def _get_logo_url(self) -> str:
        """
        Get the logo URL for email templates.

        Returns:
            str: The configured public logo URL.
        """
        return self.LOGO_URL

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

        return BASE_TEMPLATE.format(
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
