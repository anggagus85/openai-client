# src/openai_client/schemas/ai/email.py

from re import sub

import disposable_email_domains
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

DISPOSABLE_DOMAINS = disposable_email_domains.blocklist


class EmailInquiry(BaseModel):
    """Schema for incoming email requests, including validation checks."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="The user's name",
        examples=["John Doe"],
    )
    subject: str = Field(
        default="Travel Inquiry",
        min_length=1,
        description="Subject of the email",
        examples=["Travel Inquiry"],
    )

    message: str = Field(
        ...,
        min_length=1,
        description="Message of the email",
        examples=[
            "I would like to book a trip to Bali for 7 days for my family of 4 next month. I am interested in the family package.",
        ],
    )
    email: EmailStr = Field(
        ...,
        min_length=1,
        description="Email of the user",
        examples=["jhondoe@gmail.com"],
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate and sanitize name input."""
        if not v or not v.strip():
            msg = "Name cannot be empty"
            raise ValueError(msg)
        # Remove potentially harmful characters
        sanitized = sub(r'[<>"\']', "", v.strip())
        if len(sanitized) < 1:
            msg = "Name must contain valid characters"
            raise ValueError(msg)
        return sanitized

    # Validator runs after Pydantic's EmailStr validation passes
    @field_validator("email")
    @classmethod  # Class method decorator for class-level validation
    def validate_disposable_domain(cls, value: str) -> str:
        """
        Check if the email domain is a known temporary mail service.

        This validator runs after Pydantic's EmailStr validation.
        """
        # Split domain from email address - EmailStr ensures @ exists
        domain = value.lower().split("@")[-1]

        if domain in DISPOSABLE_DOMAINS:
            message = f"Email domain '{domain}' is a known temporary mail service."
            raise ValueError(message)

        return value

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate and sanitize message input."""
        if not v or not v.strip():
            msg = "Message cannot be empty"
            raise ValueError(msg)
        sanitized = sub(r"[<>]", "", v.strip())
        if len(sanitized) < 10:
            msg = "Message must be at least 10 characters long"
            raise ValueError(msg)
        return sanitized


class EmailResponse(BaseModel):
    """Schema for email response."""

    model_config = ConfigDict(ser_json_timedelta="iso8601", ser_json_bytes="utf8")

    status: str = Field(default="success", description="Status of the email")
    message: str = Field(default="Email sent successfully", description="Message of the email")


class AnalysisFormat(BaseModel):
    """Response model for contact inquiry analysis."""

    name: str = Field(..., description="Name of the user")

    summary: str = Field(
        default="Analysis of inquiry",
        description="Summary of the inquiry",
    )
    category: str = Field(default="General Information", description="Category of the inquiry")
    urgency: str = Field(default="Medium", description="Urgency of the inquiry")
    suggested_reply: str = Field(
        default="Thank you for your inquiry. We'll review your message and get back to you soon.",
        description="Suggested reply for the inquiry",
    )
    required_action: str = Field(
        default="Review and respond to customer inquiry",
        description="Required action for the inquiry",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords in the inquiry",
    )


class ContactAnalysisResponse(BaseModel):
    """Response model for contact inquiry analysis."""

    confirmation: str = Field(..., description="Confirmation message for the user")
