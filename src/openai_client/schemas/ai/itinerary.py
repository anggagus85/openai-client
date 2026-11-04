"""
Schemas for AI-powered itinerary generation requests and responses.

This module provides Pydantic models with comprehensive validation to ensure
realistic itinerary combinations of budget, duration, and interests.
"""

from datetime import datetime
from re import match as re_match
from re import sub
from typing import Self
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

MIN_TRIP_DURATION = 1
MAX_TRIP_DURATION = 30
MAX_INTERESTS_COUNT = 4

# --- Date Validation Constants ---
# Expected format: ISO 8601 YYYY-MM-DD (e.g., "2026-11-10")
DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"

# --- Budget Validation Constants ---
MIN_BUDGET_USD = 300
MAX_BUDGET_LENGTH = 50

# Cost calculation factors for realistic itinerary validation
BASE_DAILY_COST_USD = 50
COST_PER_INTEREST_USD = 25
BUDGET_BUFFER_PERCENTAGE = 0.20


def extract_budget_amount(budget_str: str) -> float:
    """
    Extract numeric value from budget string.

    Args:
        budget_str: Budget string in format like "US$ 1000" or "$1,500"

    Returns:
        The numeric budget value as float

    Raises:
        ValueError: If no valid numeric value can be extracted
    """
    # Bound the input before running regular expressions or building prompts.
    budget_str = budget_str[:MAX_BUDGET_LENGTH]
    cleaned = sub(r"(USD|US\$?|\$|,|\s)", "", budget_str.upper())
    cleaned = sub(r"[^\d.]", "", cleaned)

    if not cleaned:
        msg = "Could not extract budget amount from the provided value"
        raise ValueError(msg)

    return float(cleaned)


def calculate_minimum_required_budget(duration: int, interests_count: int) -> float:
    """Calculate the minimum budget required for a trip."""
    base_cost = duration * BASE_DAILY_COST_USD
    interest_cost = duration * interests_count * COST_PER_INTEREST_USD
    subtotal = base_cost + interest_cost
    buffer = subtotal * BUDGET_BUFFER_PERCENTAGE
    return subtotal + buffer


def calculate_max_duration_for_budget(budget: float, interests_count: int) -> int:
    """Calculate maximum trip duration for a given budget."""
    daily_cost_with_buffer = (BASE_DAILY_COST_USD + (interests_count * COST_PER_INTEREST_USD)) * (
        1 + BUDGET_BUFFER_PERCENTAGE
    )

    return int(budget / daily_cost_with_buffer)


def calculate_max_interests_for_budget(budget: float, duration: int) -> int:
    """Calculate maximum interests for a given budget and duration."""
    base_total = duration * BASE_DAILY_COST_USD * (1 + BUDGET_BUFFER_PERCENTAGE)
    remaining = budget - base_total
    cost_per_interest_with_buffer = (
        duration * COST_PER_INTEREST_USD * (1 + BUDGET_BUFFER_PERCENTAGE)
    )

    if remaining <= 0:
        return 0

    return int(remaining / cost_per_interest_with_buffer)


class ItineraryRequest(BaseModel):
    """
    Model for travel itinerary generation requests.

    Validation Rules:
        - Duration: 1-30 days
        - Dates: exactly one ISO 8601 date per day
        - Interests: 1-4 items
        - Budget: Minimum $300 USD
        - Cross-validation: Budget must be sufficient for duration × interests
    """

    duration: int = Field(
        ...,
        ge=MIN_TRIP_DURATION,
        le=MAX_TRIP_DURATION,
        description="The duration of the trip in days",
        examples=[7],
    )
    dates: list[str] = Field(
        ...,
        description="A list of dates for the trip in ISO 8601 format (YYYY-MM-DD)",
        examples=[
            [
                "2026-11-10",
                "2026-11-11",
                "2026-11-12",
                "2026-11-13",
                "2026-11-14",
                "2026-11-15",
                "2026-11-16",
            ],
        ],
    )
    interests: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_INTERESTS_COUNT,
        description="A list of traveler's interests",
        examples=[["activities", "wellness", "shops"]],
    )
    budget: str = Field(
        ...,
        max_length=MAX_BUDGET_LENGTH,
        description="The traveler's budget in USD",
        examples=["US$ 3000"],
    )

    @field_validator("dates")
    @classmethod
    def validate_dates(cls, v: list[str]) -> list[str]:
        """
        Validate the dates list format and content.

        Each date must follow ISO 8601 ``YYYY-MM-DD`` format and represent a
        valid calendar date.
        """
        if not v:
            msg = "At least one date must be provided"
            raise ValueError(msg)

        for idx, date_str in enumerate(v):
            if not isinstance(date_str, str) or not date_str.strip():
                msg = f"Date at index {idx} must be a non-empty string"
                raise ValueError(msg)

            cleaned = date_str.strip()

            if not re_match(DATE_PATTERN, cleaned):
                msg = (
                    f"Date '{cleaned}' at index {idx} does not match the expected "
                    "ISO 8601 format 'YYYY-MM-DD' (e.g., '2026-11-10')"
                )
                raise ValueError(msg)

            try:
                datetime.strptime(cleaned, "%Y-%m-%d")
            except ValueError:
                msg = (
                    f"Date '{cleaned}' at index {idx} is not a valid calendar date. "
                    "Ensure month is 01-12 and day is valid for the given month."
                )
                raise ValueError(msg) from None

        return [d.strip() for d in v]

    @field_validator("interests")
    @classmethod
    def validate_interests(cls, v: list[str]) -> list[str]:
        """Validate and sanitize interests list."""
        if not v:
            msg = "At least one interest must be provided"
            raise ValueError(msg)

        sanitized_interests = []
        for interest in v:
            if isinstance(interest, str) and interest.strip():
                sanitized = sub(r"[<>\"']", "", interest.strip())[:50]
                if sanitized:
                    sanitized_interests.append(sanitized)

        if not sanitized_interests:
            msg = "At least one valid interest must be provided"
            raise ValueError(msg)

        return sanitized_interests

    @field_validator("budget")
    @classmethod
    def validate_budget_format(cls, v: str) -> str:
        """Validate and normalize budget input format."""
        if not v or not v.strip():
            msg = "Budget cannot be empty"
            raise ValueError(msg)

        v = v.strip()

        if not v.upper().startswith("US$"):
            v = f"US{v}" if v.startswith("$") else f"US$ {v}"

        amount = extract_budget_amount(v)
        if amount < MIN_BUDGET_USD:
            msg = f"Budget must be at least US$ {MIN_BUDGET_USD}"
            raise ValueError(msg)

        return v

    @model_validator(mode="after")
    def validate_budget_feasibility(self) -> Self:
        """Validate dates count, then budget feasibility for the trip."""
        if len(self.dates) != self.duration:
            msg = (
                f"The number of dates provided ({len(self.dates)}) must match "
                f"the trip duration ({self.duration} days). "
                f"Please provide exactly {self.duration} date(s)."
            )
            raise ValueError(msg)

        budget_amount = extract_budget_amount(self.budget)
        interests_count = len(self.interests)
        required_budget = calculate_minimum_required_budget(self.duration, interests_count)

        if budget_amount < required_budget:
            max_duration = calculate_max_duration_for_budget(budget_amount, interests_count)
            max_interests = calculate_max_interests_for_budget(budget_amount, self.duration)

            suggestions = []
            if max_duration >= MIN_TRIP_DURATION:
                suggestions.append(f"reduce duration to {max_duration} days")
            if max_interests >= 1:
                suggestions.append(f"reduce interests to {max_interests}")
            suggestions.append(f"increase budget to at least US$ {required_budget:.0f}")

            suggestion_text = " or ".join(suggestions)

            msg = (
                f"Budget of US$ {budget_amount:.0f} is insufficient for a "
                f"{self.duration}-day trip with {interests_count} interests. "
                f"Minimum required: US$ {required_budget:.0f}. "
                f"Suggestion: {suggestion_text}."
            )
            raise ValueError(msg)

        return self


# --- Response Models ---


class ItineraryMD(BaseModel):
    """Response model for itinerary generation."""

    markdown: str = Field(..., description="Generated markdown travel itinerary")


class ItineraryResponse(BaseModel):
    """Full response model for itinerary generation."""

    markdown: str = Field(..., description="Generated markdown travel itinerary")
    whatsapp: str = Field(..., description="Generated whatsapp travel itinerary")


class ItineraryRequestTXT(BaseModel):
    user_name: str = Field(..., description="User name")
    md_id: UUID = Field(..., description="Markdown itinerary id")
    itinerary_md: str = Field(..., description="Markdown itinerary source")


class ItineraryTXT(BaseModel):
    """API Response model including WhatsApp-friendly text."""

    text_content: str = Field(..., description="WhatsApp-friendly plain text version")
