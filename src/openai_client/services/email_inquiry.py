# src/openai_client/services/email_inquiry.py

from datetime import datetime
from html import escape
from json import JSONDecodeError
from logging import getLogger

from pydantic import ValidationError

from openai_client.client import OpenAI
from openai_client.schemas.ai import AnalysisFormat, ContactAnalysisResponse, EmailInquiry
from openai_client.types import ListParams, SystemParam, UserParam

logger = getLogger(__name__)


def _email_messages(content: str) -> ListParams:
    return [
        SystemParam(
            role="system",
            content="You are a friendly, warm, welcoming, enthusiastic and an expert travel assistant for Bali travel service company.",
        ),
        UserParam(
            role="user",
            content=content,
        ),
    ]


class EmailService:
    def __init__(self, client: OpenAI, inquiry: EmailInquiry) -> None:
        self._client = client
        self._inquiry = inquiry
        self._messages: ListParams = []

    def _create_body(self, analysis: AnalysisFormat) -> str:
        """Create a body for the email with professional styling."""
        request = self._inquiry
        # Build the inquiry details section
        inquiry_details = f"""
        <div class="info-box" style="text-align: left;">
            <p style="margin: 0 0 12px 0;"><strong>Name:</strong> {escape(request.name)}</p>
            <p style="margin: 0 0 12px 0;"><strong>Email:</strong> <a href="mailto:{escape(request.email)}" style="color: #1a2a6c;">{escape(request.email)}</a></p>
        </div>
        """

        # Build the message section
        message_section = f"""
        <div class="info-box" style="text-align: left; margin-top: 24px;">
            <h3 style="color: #1a2a6c; font-size: 18px; margin: 0 0 12px 0;">Message:</h3>
            <p style="margin: 0; white-space: pre-wrap; color: #666666;">{escape(request.message)}</p>
        </div>
        """

        keywords_str = (
            ", ".join(analysis.keywords)
            if isinstance(analysis.keywords, (list, tuple))
            else str(analysis.keywords or "")
        )
        analysis_items = f"""
        <div class="info-box" style="text-align: left; margin-top: 24px;">
            <h3 style="color: #1a2a6c; font-size: 18px; margin: 0 0 16px 0;">AI Analysis</h3>
            <p style="margin: 0 0 8px 0;"><strong>Summary:</strong> {escape(analysis.summary)}</p>
            <p style="margin: 0 0 8px 0;"><strong>Category:</strong> {escape(analysis.category)}</p>
            <p style="margin: 0 0 8px 0;"><strong>Urgency:</strong> {escape(analysis.urgency)}</p>
            <p style="margin: 0 0 8px 0;"><strong>Suggested Reply:</strong> {escape(analysis.suggested_reply)}</p>
            <p style="margin: 0 0 8px 0;"><strong>Required Action:</strong> {escape(analysis.required_action)}</p>
            <p style="margin: 0;"><strong>Keywords:</strong> {escape(keywords_str)}</p>
        </div>
        """

        # Build the complete content
        content_html = f"""
        <div class="content">
            <h1 style="color: #1a2a6c; font-size: 24px; margin: 0 0 8px 0; line-height: 1.3;">New BaliBlissed Inquiry</h1>
            <p style="color: #666666; font-size: 14px; margin: 0 0 24px 0;">You have received a new contact inquiry.</p>
            {inquiry_details}
            {message_section}
            {analysis_items}
        </div>
        """

        # Build header
        logo_url = (
            "https://res.cloudinary.com/dusikjnta/image/upload/"
            "f_auto/q_auto/v1/My%20Brand/bali_blissed_simplified_dhkbvy?_a=BAMAAAhK0"
        )
        header_html = f"""
        <div class="header">
            <img src="{logo_url}" alt="BaliBlissed Logo">
        </div>
        """

        # Build footer
        year = datetime.now().year
        footer_html = f"""
        <div class="footer">
            <p>&copy; {year} BaliBlissed. All rights reserved.</p>
            <p>Your portal to Bali's finest travel experiences.</p>
        </div>
        """

        # Build the complete email with base template styles
        body_content = f"""
        <div class="container">
            {header_html}
            {content_html}
            {footer_html}
        </div>
        """.strip()

        # Use the same styling approach as EmailTemplate
        return f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <title>New BaliBlissed Inquiry</title>
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
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f4f7f9;
                margin: 0;
                padding: 0;
                color: #333333;
            }}
            .container {{
                width: 100%;
                max-width: 600px;
                margin: 20px auto;
                background-color: #ffffff;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.8);
                border: 1px solid rgba(10, 10, 10, 0.1);
            }}
            .header {{
                background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d);
                padding: 20px;
                text-align: center;
            }}
            .header img {{
                max-width: 150px;
                height: auto;
            }}
            .content {{
                padding: 16px;
                text-align: center;
            }}
            .content h1 {{
                color: #1a2a6c;
                font-size: 18px;
                margin: 0 0 16px 0;
                line-height: 1.3;
            }}
            .content p {{
                line-height: 1.6;
                color: #666666;
                font-size: 14px;
                margin: 0 0 16px 0;
            }}
            .footer {{
                background-color: #f9f9f9;
                padding: 20px;
                text-align: center;
                font-size: 12px;
                color: #999999;
                border-top: 1px solid #eeeeee;
            }}
            .footer p {{
                margin: 5px 0;
                color: #999999;
            }}
            .info-box {{
                margin: 24px 0;
                padding: 16px;
                background-color: #f9f9f9;
                border-radius: 8px;
                border-left: 4px solid #ce6f21;
                text-align: left;
            }}
            .info-box p {{
                margin: 0;
                font-size: 14px;
                color: #666666;
            }}
            .info-box h3 {{
                margin: 0 0 16px 0;
                color: #1a2a6c;
                font-size: 18px;
            }}
            /* Desktop enhancements */
            @media screen and (min-width: 600px) {{
                .container {{
                    margin: 40px auto;
                }}
                .header {{
                    padding: 30px;
                }}
                .content {{
                    padding: 40px;
                }}
                .content h1 {{
                    font-size: 24px;
                    margin-bottom: 20px;
                }}
                .content p {{
                    font-size: 16px;
                    margin-bottom: 16px;
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
                .footer {{
                    background-color: #252525;
                    border-top-color: #3d3d3d;
                }}
                .info-box {{
                    background-color: #333333;
                }}
                .info-box p {{
                    color: #cccccc;
                }}
            }}
        </style>
    </head>
    <body>
        {body_content}
    </body>
    </html>
    """

    def _contact_analysis_prompt(self) -> str:
        """
        Create a prompt for contact inquiry analysis.

        Returns: A formatted prompt string for analysis.
        """
        name = self._inquiry.name
        message = self._inquiry.message

        return f"""
                Analyze the following customer inquiry for a Bali travel service company.
                Provide a structured analysis in JSON format.

                **Customer Message:** "{message}"

                Please analyze and provide:
                1. **Name**: {name}
                2. **Summary**: A brief 1-3 sentence summary of the inquiry
                3. **Category**: Classify into one of these categories:
                - "Booking Inquiry" - Questions about reservations, availability
                - "Itinerary Planning" - Requests for travel planning assistance
                - "General Information" - Questions about Bali, services, policies
                - "Support Request" - Issues, complaints, or assistance needed
                - "Pricing Inquiry" - Questions about costs, packages, pricing
                - "Feedback" - Reviews, testimonials, suggestions

                4. **Urgency**: Rate as "High", "Medium", or "Low"
                5. **Suggested Response**: A professional, helpful response template
                6. **Required Action**: What the team should do next
                7. **Keywords**: Key topics mentioned in the {message}

                Respond ONLY with a valid JSON object in this exact format:
                {{
                    "name": "{name}",
                    "summary": "Brief summary here",
                    "category": "Category name",
                    "urgency": "Urgency level",
                    "suggested_reply": "Professional response template",
                    "required_action": "Next steps for the team",
                    "keywords": ["keyword1", "keyword2", "keyword3"]
                }}
                """

    async def analyze_contact(self, method: str) -> str:
        messages = _email_messages(self._contact_analysis_prompt())
        self._client.messages = messages

        logger.info("Analyzing email inquiry with %s", self._client)
        response = await self._client.call(append=False)[method]()
        try:
            analysis = AnalysisFormat.model_validate(response)
            return self._create_body(analysis)

        except (ValidationError, JSONDecodeError):
            logger.warning(
                "Contact inquiry analysis validation failed, using default AnalysisFormat",
            )
            return self._create_body(AnalysisFormat(name=self._inquiry.name))

    async def confirmation_message(
        self,
        method: str,
        *,
        sent: bool = True,
    ) -> ContactAnalysisResponse:
        messages = _email_messages(self._confirmation_message_prompt(sent=sent))
        self._client.messages = messages
        logger.info("Generating confirmation message...")
        response = await self._client.call(append=False)[method]()
        data = {"confirmation": response}
        req_name = self._inquiry.name
        try:
            return ContactAnalysisResponse.model_validate(data)

        except ValidationError:
            logger.warning("Confirmation message validation failed, using default response")
            status = "sent successfully" if sent else "failed to send"
            return ContactAnalysisResponse(
                confirmation=f"Thank you {req_name}. Your inquiry has been received and we will get back to you soon. Note: There was a technical issue and your message {status}.",
            )

    def _confirmation_message_prompt(self, *, sent: bool = True) -> str:
        """Generate confirmation message for contact inquiry."""
        req_name = self._inquiry.name
        if sent:
            scenario = (
                f"The email inquiry from {req_name} was sent successfully. "
                f"Generate a brief, friendly, and reassuring confirmation message. "
                f"Acknowledge the user by name ({req_name}) and mention that we will "
                f"get back to them at their provided email address within 24-48 hours. "
                f"Do not include a subject line. Do not use emojis. "
                f"Only provide the confirmation message body."
            )
        else:
            scenario = (
                f"The email inquiry from {req_name} failed to send due to a technical issue. "
                f"Politely inform the user by name ({req_name}) that there was a problem "
                f"preventing their message from being sent. Ask them to try again later. "
                f"Do not include a subject line. Do not use emojis. "
                f"Only provide the confirmation message body."
            )

        return f"""
            You are a friendly customer service assistant for a Bali travel agency called BaliBlissed.
            A user has submitted a contact inquiry. Your only job is to generate a confirmation message.

            User's Name: {req_name}

            Scenario: {scenario}

            Respond ONLY with the confirmation message text. Do not include:
            - Subject lines
            - Emojis
            - Multiple message options
            - Any explanation or meta-commentary
        """
