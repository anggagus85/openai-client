# tests/test_whatsapp_formatter.py

"""Tests for _markdown_to_whatsapp converter in services/itinerary.py."""

from openai_client.services.itinerary import _convert_table_to_whatsapp, _markdown_to_whatsapp

# ---------------------------------------------------------------------------
# Header conversions
# ---------------------------------------------------------------------------


class TestHeaderConversions:
    """H1 → bold uppercase, H2-H6 → bold."""

    def test_h1_becomes_bold_uppercase(self) -> None:
        md = "# Welcome to Bali\n"
        result = _markdown_to_whatsapp(md)
        assert "*WELCOME TO BALI*" in result

    def test_h2_becomes_bold(self) -> None:
        md = "## Daily Breakdown\n"
        result = _markdown_to_whatsapp(md)
        assert "*Daily Breakdown*" in result

    def test_h3_becomes_bold(self) -> None:
        md = "### Culinary Experiences\n"
        result = _markdown_to_whatsapp(md)
        assert "*Culinary Experiences*" in result

    def test_h4_becomes_bold(self) -> None:
        md = "#### Day 1 – Arrival\n"
        result = _markdown_to_whatsapp(md)
        assert "*Day 1 – Arrival*" in result

    def test_h4_with_inner_bold_stripped(self) -> None:
        """Headers like #### **Day 1 – Title** should not produce nested bold."""
        md = "#### **Day 1 – Temples & Terraces**\n"
        result = _markdown_to_whatsapp(md)
        assert "*Day 1 – Temples & Terraces*" in result
        # Must NOT contain ** (double asterisks).
        assert "**" not in result

    def test_h1_with_emoji(self) -> None:
        md = "# 🌴 Here's your 5-Days Bali trip Itinerary 🌴\n"
        result = _markdown_to_whatsapp(md)
        assert "*🌴 HERE'S YOUR 5-DAYS BALI TRIP ITINERARY 🌴*" in result


# ---------------------------------------------------------------------------
# Bold conversions
# ---------------------------------------------------------------------------


class TestBoldConversion:
    """**text** → *text*."""

    def test_simple_bold(self) -> None:
        md = "Visit **Tanah Lot** temple\n"
        result = _markdown_to_whatsapp(md)
        assert "Visit *Tanah Lot* temple" in result

    def test_multiple_bold_in_line(self) -> None:
        md = "Try **Babi Guling** – USD 12, **Nasi Campur** – USD 8\n"
        result = _markdown_to_whatsapp(md)
        assert "*Babi Guling*" in result
        assert "*Nasi Campur*" in result
        assert "**" not in result

    def test_bold_at_line_start(self) -> None:
        md = "**Morning Activity (9-11 AM):**\n"
        result = _markdown_to_whatsapp(md)
        assert "*Morning Activity (9-11 AM):*" in result


# ---------------------------------------------------------------------------
# Link conversions
# ---------------------------------------------------------------------------


class TestLinkConversion:
    """[text](url) → text (url)."""

    def test_simple_link(self) -> None:
        md = "Check [Booking.com](https://booking.com) for hotels\n"
        result = _markdown_to_whatsapp(md)
        assert "Check Booking.com (https://booking.com) for hotels" in result

    def test_link_with_bold(self) -> None:
        md = "Visit [**Ubud Palace**](https://example.com)\n"
        result = _markdown_to_whatsapp(md)
        # Link should be flattened, bold converted.
        assert "**" not in result
        assert "(https://example.com)" in result


# ---------------------------------------------------------------------------
# Image conversions
# ---------------------------------------------------------------------------


class TestImageConversion:
    """![alt](url) → alt text only."""

    def test_image_stripped_to_alt(self) -> None:
        md = "![Beautiful Bali sunset](https://example.com/sunset.jpg)\n"
        result = _markdown_to_whatsapp(md)
        assert "Beautiful Bali sunset" in result
        assert "https://example.com/sunset.jpg" not in result

    def test_image_with_empty_alt(self) -> None:
        md = "![](https://example.com/photo.jpg)\n"
        result = _markdown_to_whatsapp(md)
        assert "https://example.com/photo.jpg" not in result


# ---------------------------------------------------------------------------
# Horizontal rules
# ---------------------------------------------------------------------------


class TestHorizontalRules:
    """--- / *** / ___ → unicode line."""

    def test_triple_dash(self) -> None:
        md = "---\n"
        result = _markdown_to_whatsapp(md)
        assert "─────────────" in result

    def test_triple_asterisk(self) -> None:
        md = "***\n"
        result = _markdown_to_whatsapp(md)
        assert "─────────────" in result

    def test_triple_underscore(self) -> None:
        md = "___\n"
        result = _markdown_to_whatsapp(md)
        assert "─────────────" in result


# ---------------------------------------------------------------------------
# Table conversions
# ---------------------------------------------------------------------------


class TestTableConversion:
    """Markdown tables → WhatsApp bullet-point format."""

    def test_simple_table(self) -> None:
        md = (
            "| Category | Daily Avg. | 5-Day Total |\n"
            "|----------|-----------|-------------|\n"
            "| Meals | $30 | $150 |\n"
            "| Transport | $15 | $75 |\n"
        )
        result = _markdown_to_whatsapp(md)
        assert "• *Meals*:" in result
        assert "• *Transport*:" in result
        assert "$30" in result
        assert "$150" in result
        # Separator row must not appear.
        assert "---" not in result

    def test_table_with_bold_cells(self) -> None:
        md = "| Category | Amount |\n|----------|--------|\n| **Accommodation** | $80 |\n"
        result = _markdown_to_whatsapp(md)
        # Bold markers from cells should be stripped before re-wrapping.
        assert "• *Accommodation*:" in result
        assert "**" not in result

    def test_single_column_table(self) -> None:
        md = "| Item |\n|------|\n| Sunscreen |\n"
        result = _markdown_to_whatsapp(md)
        assert "• *Sunscreen*" in result


class TestConvertTableToWhatsapp:
    """Unit tests for _convert_table_to_whatsapp helper."""

    def test_insufficient_lines(self) -> None:
        assert _convert_table_to_whatsapp(["| only one |"]) == ["| only one |"]

    def test_malformed_header(self) -> None:
        lines = ["not a table", "|---|", "| data |"]
        assert _convert_table_to_whatsapp(lines) == lines


# ---------------------------------------------------------------------------
# Whitespace handling
# ---------------------------------------------------------------------------


class TestWhitespace:
    """Trailing spaces, multiple blank lines, final newline."""

    def test_trailing_spaces_stripped(self) -> None:
        md = "Hello world   \nNext line  \n"
        result = _markdown_to_whatsapp(md)
        for line in result.split("\n"):
            if line:  # skip empty lines
                assert not line.endswith(" "), f"Trailing space found: {line!r}"

    def test_multiple_blank_lines_collapsed(self) -> None:
        md = "Line 1\n\n\n\n\nLine 2\n"
        result = _markdown_to_whatsapp(md)
        assert "\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result

    def test_single_trailing_newline(self) -> None:
        md = "Content here\n\n\n"
        result = _markdown_to_whatsapp(md)
        assert result.endswith("\n")
        assert not result.endswith("\n\n")


# ---------------------------------------------------------------------------
# Bullet list preservation
# ---------------------------------------------------------------------------


class TestBulletLists:
    """Bullet points and indentation should be preserved."""

    def test_dash_bullets_preserved(self) -> None:
        md = "- Item one\n- Item two\n"
        result = _markdown_to_whatsapp(md)
        assert "- Item one" in result
        assert "- Item two" in result

    def test_nested_bullets_indentation_preserved(self) -> None:
        md = "- Parent\n    - Child\n"
        result = _markdown_to_whatsapp(md)
        assert "    - Child" in result


# ---------------------------------------------------------------------------
# Emoji preservation
# ---------------------------------------------------------------------------


class TestEmojiPreservation:
    """Emojis must pass through untouched."""

    def test_emojis_in_text(self) -> None:
        md = "🌅 Morning Activity at 🏖️ Beach\n"
        result = _markdown_to_whatsapp(md)
        assert "🌅" in result
        assert "🏖️" in result

    def test_emojis_in_header(self) -> None:
        md = "### 🍜 CULINARY EXPERIENCES\n"
        result = _markdown_to_whatsapp(md)
        assert "🍜" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Empty input, no-op conversions, mixed content."""

    def test_empty_input(self) -> None:
        result = _markdown_to_whatsapp("")
        assert result == "\n"

    def test_plain_text_passthrough(self) -> None:
        md = "Just plain text with no markdown.\n"
        result = _markdown_to_whatsapp(md)
        assert "Just plain text with no markdown." in result

    def test_inline_code_untouched(self) -> None:
        md = "Use `npm install` to install.\n"
        result = _markdown_to_whatsapp(md)
        assert "`npm install`" in result


# ---------------------------------------------------------------------------
# Integration: realistic itinerary snippet
# ---------------------------------------------------------------------------


class TestItineraryIntegration:
    """End-to-end conversion of a realistic itinerary fragment."""

    SAMPLE_MD = """\
# 🌴 Here's your 5-Days Bali trip Itinerary with US$ 1500 budget 🌴

## **🌊 Beach Bliss & Temple Trails: Your 5-Day Bali Escape 🛕**

### 🌅 DAILY BREAKDOWN

#### **Day 1 – Arrival & Ubud Introduction**

- **Morning Activity (9-11 AM):**
    - Arrive at Ngurah Rai International Airport. Transfer to Ubud (open 24h, transfer USD 25).
- **Afternoon Activity (12-4 PM):**
    - Lunch at **Warung Babi Guling Ibu Oka** (open 11 AM-6 PM). Try **Babi Guling** – USD 12, **Nasi Campur** – USD 8.

### 💸 COMPREHENSIVE BUDGET BREAKDOWN

| Category | Daily Avg. | 5-Day Total |
|----------|-----------|-------------|
| Accommodation | $80 | $400 |
| Meals & Drinks | $30 | $150 |

---

### 🌟 A Friendly Note

- This itinerary is a great starting point.
- Contact us on WhatsApp at +6285847006743.
"""

    def test_no_double_asterisks_remain(self) -> None:
        result = _markdown_to_whatsapp(self.SAMPLE_MD)
        assert "**" not in result

    def test_h1_uppercased(self) -> None:
        result = _markdown_to_whatsapp(self.SAMPLE_MD)
        assert "*🌴 HERE'S YOUR 5-DAYS BALI TRIP ITINERARY WITH US$ 1500 BUDGET 🌴*" in result

    def test_h2_bold(self) -> None:
        result = _markdown_to_whatsapp(self.SAMPLE_MD)
        assert "*🌊 Beach Bliss & Temple Trails: Your 5-Day Bali Escape 🛕*" in result

    def test_day_header_bold(self) -> None:
        result = _markdown_to_whatsapp(self.SAMPLE_MD)
        assert "*Day 1 – Arrival & Ubud Introduction*" in result

    def test_table_converted(self) -> None:
        result = _markdown_to_whatsapp(self.SAMPLE_MD)
        assert "• *Accommodation*:" in result
        assert "$400" in result
        # No pipe-row table syntax.
        assert "|---" not in result

    def test_horizontal_rule_converted(self) -> None:
        result = _markdown_to_whatsapp(self.SAMPLE_MD)
        assert "─────────────" in result

    def test_restaurant_bold_converted(self) -> None:
        result = _markdown_to_whatsapp(self.SAMPLE_MD)
        assert "*Warung Babi Guling Ibu Oka*" in result
        assert "*Babi Guling*" in result

    def test_ends_with_single_newline(self) -> None:
        result = _markdown_to_whatsapp(self.SAMPLE_MD)
        assert result.endswith("\n")
        assert not result.endswith("\n\n")
