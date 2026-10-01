from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from src.agent.schemas import TripRequest, TripRequestDraft
from src.agent.validation import missing_fields, validate_draft

T = date.today()


def make_draft(**overrides) -> TripRequestDraft:
    base = dict(
        origin="Mumbai", destination="Jaipur",
        start_date=T + timedelta(days=15), end_date=T + timedelta(days=17),
        travellers=2, budget=25000, travel_priority="budget",
        interests=["history", "food"],
    )
    base.update(overrides)
    return TripRequestDraft(**base)


def test_complete_request_is_valid():
    result = validate_draft(make_draft())
    assert result.ok
    assert result.trip.num_days == 3
    assert result.trip.num_nights == 2


def test_missing_only_reports_missing():
    result = validate_draft(TripRequestDraft(destination="Jaipur"))
    assert not result.ok
    assert "Destination" not in result.missing
    assert "Starting city" in result.missing
    assert "Start and end dates" in result.missing
    assert len(result.missing) == 5


def test_missing_single_end_date():
    assert missing_fields(make_draft(end_date=None)) == ["End date"]


def test_interests_are_normalised():
    draft = make_draft(interests=[" History", "FOOD", "history", ""])
    assert draft.interests == ["history", "food"]
    assert TripRequestDraft(interests="nature, art").interests == ["nature", "art"]


def test_today_is_allowed():
    assert validate_draft(make_draft(start_date=T, end_date=T)).ok


def test_past_start_date_rejected():
    result = validate_draft(make_draft(start_date=T - timedelta(days=1)))
    assert not result.ok
    assert "past" in result.errors[0]


def test_end_before_start_rejected():
    result = validate_draft(make_draft(end_date=T + timedelta(days=10)))
    assert "end date" in result.errors[0].lower()


@pytest.mark.parametrize("days, ok", [(10, True), (11, False)])
def test_trip_length_limit(days, ok):
    start = T + timedelta(days=5)
    result = validate_draft(make_draft(start_date=start, end_date=start + timedelta(days=days - 1)))
    assert result.ok is ok


@pytest.mark.parametrize("travellers, ok", [(1, True), (10, True), (0, False), (11, False)])
def test_traveller_limits(travellers, ok):
    assert validate_draft(make_draft(travellers=travellers)).ok is ok


@pytest.mark.parametrize("budget", [0, -500])
def test_budget_must_be_positive(budget):
    result = validate_draft(make_draft(budget=budget))
    assert result.errors == ["The budget must be a positive amount."]


def test_same_city_rejected_unless_intended():
    assert not validate_draft(make_draft(destination="mumbai")).ok
    assert validate_draft(make_draft(destination="mumbai", same_city_intended=True)).ok


def test_invalid_priority_rejected_by_strict_model():
    with pytest.raises(ValidationError):
        TripRequest(**{**make_draft().model_dump(), "travel_priority": "luxury"})