# src/openai_client/services/templates/__init__.py

from openai_client.services.templates.email_template import EmailTemplate, html_to_plain_text

__all__ = ["EmailTemplate", "html_to_plain_text"]
