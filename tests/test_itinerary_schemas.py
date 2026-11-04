# tests/test_itinerary_schemas.py

"""Tests for production itinerary schemas and validation functions."""

from datetime import date, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from openai_client.schemas.ai.itinerary import (
    ItineraryMD,
    ItineraryRequest,
    ItineraryRequestTXT,
    ItineraryResponse,
    ItineraryTXT,
    calculate_max_duration_for_budget,
    calculate_max_interests_for_budget,
    calculate_minimum_required_budget,
    extract_budget_amount,
)


def _dates(count: int, start: date = date(2026, 11, 10)) -> list[str]:
    return [(start + timedelta(days=i)).isoformat() for i in range(count)]


class TestExtractBudgetAmount:
    def test_usd_format(self) -> None:
        assert extract_budget_amount("US$ 1000") == 1000.0

    def test_dollar_sign_format(self) -> None:
        assert extract_budget_amount("$1,500") == 1500.0

    def test_plain_number(self) -> None:
        assert extract_budget_amount("2000") == 2000.0

    def test_decimal_amount(self) -> None:
        assert extract_budget_amount("US$ 1234.56") == 1234.56

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not extract budget amount"):
            extract_budget_amount("")

    def test_no_digits_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not extract budget amount"):
            extract_budget_amount("US$")


class TestCalculateMinimumRequiredBudget:
    def test_single_day_no_interests(self) -> None:
        assert calculate_minimum_required_budget(1, 0) == 60.0

    def test_single_day_one_interest(self) -> None:
        assert calculate_minimum_required_budget(1, 1) == 90.0

    def test_seven_days_two_interests(self) -> None:
        expected = (7 * 50 + 7 * 2 * 25) * 1.2
        assert calculate_minimum_required_budget(7, 2) == expected

    def test_thirty_days_four_interests(self) -> None:
        expected = (30 * 50 + 30 * 4 * 25) * 1.2
        assert calculate_minimum_required_budget(30, 4) == expected


class TestCalculateMaxDurationForBudget:
    def test_budget_1000_no_interests(self) -> None:
        assert calculate_max_duration_for_budget(1000, 0) == int(1000 / (50 * 1.2))

    def test_budget_1000_two_interests(self) -> None:
        daily = (50 + 2 * 25) * 1.2
        assert calculate_max_duration_for_budget(1000, 2) == int(1000 / daily)

    def test_zero_budget(self) -> None:
        assert calculate_max_duration_for_budget(0, 1) == 0


class TestCalculateMaxInterestsForBudget:
    def test_budget_1000_seven_days(self) -> None:
        assert calculate_max_interests_for_budget(1000, 7) >= 0

    def test_insufficient_budget(self) -> None:
        assert calculate_max_interests_for_budget(100, 7) == 0

    def test_high_budget_allows_many_interests(self) -> None:
        assert calculate_max_interests_for_budget(5000, 3) > 0


class TestItineraryRequest:
    def test_valid_request(self) -> None:
        request = ItineraryRequest(
            duration=7,
            dates=_dates(7),
            interests=["beaches", "temples"],
            budget="US$ 1000",
        )
        assert request.duration == 7
        assert len(request.dates) == 7
        assert len(request.interests) == 2
        assert request.budget == "US$ 1000"

    def test_duration_too_low(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            ItineraryRequest.model_validate(
                {
                    "duration": 0,
                    "dates": [],
                    "interests": ["beaches"],
                    "budget": "US$ 1000",
                },
            )

    def test_duration_too_high(self) -> None:
        with pytest.raises(ValidationError, match="less than or equal to 30"):
            ItineraryRequest.model_validate(
                {
                    "duration": 31,
                    "dates": _dates(31),
                    "interests": ["beaches"],
                    "budget": "US$ 1000",
                },
            )

    def test_dates_must_match_duration(self) -> None:
        with pytest.raises(ValidationError, match="must match"):
            ItineraryRequest(
                duration=5,
                dates=_dates(4),
                interests=["beaches"],
                budget="US$ 1000",
            )

    def test_invalid_date_format(self) -> None:
        with pytest.raises(ValidationError, match="ISO 8601"):
            ItineraryRequest(
                duration=1,
                dates=["10-11-2026"],
                interests=["beaches"],
                budget="US$ 1000",
            )

    def test_invalid_calendar_date(self) -> None:
        with pytest.raises(ValidationError, match="not a valid calendar date"):
            ItineraryRequest(
                duration=1,
                dates=["2026-02-30"],
                interests=["beaches"],
                budget="US$ 1000",
            )

    def test_empty_interests(self) -> None:
        with pytest.raises(ValidationError, match="at least 1 item"):
            ItineraryRequest(duration=1, dates=_dates(1), interests=[], budget="US$ 1000")

    def test_too_many_interests(self) -> None:
        with pytest.raises(ValidationError, match="at most 4 items"):
            ItineraryRequest(
                duration=5,
                dates=_dates(5),
                interests=["a", "b", "c", "d", "e"],
                budget="US$ 1000",
            )

    def test_budget_below_minimum(self) -> None:
        with pytest.raises(ValidationError, match="at least US\\$ 300"):
            ItineraryRequest(
                duration=1,
                dates=_dates(1),
                interests=["beaches"],
                budget="US$ 100",
            )

    def test_budget_empty(self) -> None:
        with pytest.raises(ValidationError, match="cannot be empty"):
            ItineraryRequest(
                duration=1,
                dates=_dates(1),
                interests=["beaches"],
                budget="",
            )

    def test_budget_normalization(self) -> None:
        request = ItineraryRequest(
            duration=1,
            dates=_dates(1),
            interests=["beaches"],
            budget="$500",
        )
        assert request.budget.startswith("US$")

    def test_interest_sanitization(self) -> None:
        request = ItineraryRequest(
            duration=5,
            dates=_dates(5),
            interests=["<script>alert('xss')</script>", "temples"],
            budget="US$ 1000",
        )
        assert "<script>" not in request.interests[0]
        assert "alert" in request.interests[0]

    def test_budget_feasibility_insufficient(self) -> None:
        with pytest.raises(ValidationError, match="insufficient"):
            ItineraryRequest(
                duration=30,
                dates=_dates(30),
                interests=["beaches", "temples", "food", "culture"],
                budget="US$ 500",
            )

    def test_budget_feasibility_sufficient(self) -> None:
        request = ItineraryRequest(
            duration=3,
            dates=_dates(3),
            interests=["beaches"],
            budget="US$ 500",
        )
        assert request.duration == 3


class TestItineraryRequestTXT:
    def test_valid_request(self) -> None:
        req = ItineraryRequestTXT(
            user_name="John",
            md_id=uuid4(),
            itinerary_md="# Trip\n\nDay 1: Beach",
        )
        assert req.user_name == "John"
        assert req.itinerary_md == "# Trip\n\nDay 1: Beach"


class TestItineraryMD:
    def test_valid_response(self) -> None:
        resp = ItineraryMD(markdown="# Bali Trip\n\nDay 1: Beach")
        assert resp.markdown == "# Bali Trip\n\nDay 1: Beach"


class TestItineraryResponse:
    def test_valid_response(self) -> None:
        resp = ItineraryResponse(markdown="# Trip", whatsapp="*TRIP*")
        assert resp.markdown == "# Trip"
        assert resp.whatsapp == "*TRIP*"


class TestItineraryTXT:
    def test_valid_response(self) -> None:
        resp = ItineraryTXT(text_content="Bali Trip - Day 1: Beach")
        assert resp.text_content == "Bali Trip - Day 1: Beach"
