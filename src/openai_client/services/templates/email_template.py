"""Email template facade composing HTML and plain-text builders."""

from openai_client.services.templates.email_builders import _EmailHtmlMixin
from openai_client.services.templates.plain_builders import _PlainTemplateMixin


class EmailTemplate(_EmailHtmlMixin, _PlainTemplateMixin):
    """
    Builder for creating professional HTML email templates.

    This class provides methods for constructing responsive, mobile-first HTML emails
    with consistent branding and styling. It supports various email components including
    headers, footers, buttons, security alerts, and info boxes.
    """
