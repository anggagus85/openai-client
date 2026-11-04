"""Shared brand tokens for email templates."""

from openai_client.configs import settings


class _TemplateStyles:
    """Brand tokens shared by email builders."""

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

    LOGO_URL: str = settings.LOGO_URL

    @staticmethod
    def _expiry_phrase(expiry_hours: int) -> str:
        """Render an expiry duration with correct pluralization."""
        hour_word = "hour" if expiry_hours == 1 else "hours"
        return f"{expiry_hours} {hour_word}"
