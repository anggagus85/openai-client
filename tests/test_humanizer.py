# tests/test_humanizer.py

"""Tests for the humanizer whitespace cleanup."""

from openai_client.services.prompts.humanizer import humanize_response


class TestWhitespaceCleanup:
    def test_collapses_internal_double_spaces(self) -> None:
        assert humanize_response("Hello   world") == "Hello world"

    def test_preserves_paragraph_and_block_breaks(self) -> None:
        text = "Heading\n\n- one\n- two\n\n| A | B |\n|---|---|\n| 1 | 2 |"
        assert humanize_response(text) == text

    def test_preserves_nested_list_indentation(self) -> None:
        text = "- parent\n    - child"
        assert humanize_response(text) == text

    def test_limits_excessive_blank_lines(self) -> None:
        assert humanize_response("One\n\n\n\nTwo") == "One\n\nTwo"

    def test_preserves_fenced_code_spacing(self) -> None:
        code = "def  foo():\n\n\n    return 1"
        text = f"Intro\n\n```\n{code}\n```\n\nOutro"
        assert code in humanize_response(text)

    def test_preserves_inline_code_spacing(self) -> None:
        text = "Use `value  =  1` exactly."
        assert "`value  =  1`" in humanize_response(text)
