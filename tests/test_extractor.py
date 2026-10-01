import os
from datetime import date

import pytest

from src.agent.extractor import extract_trip_details, extraction_to_draft
from src.agent.schemas import TripExtraction, TripRequestDraft
from src.agent.validation import missing_fields

# ---------- Offline tests (no API calls) ----------

def test_duration_fills_end_date():
    d = TripRequestDraft(start_date=date(2030, 1, 10), duration_days=3)
    assert d.end_date == date(2030, 1, 12)


def test_duration_fills_start_date():
    d = TripRequestDraft(end_date=date(2030, 1, 12), duration_days=3)
    assert d.start_date == date(2030, 1, 10)


def test_explicit_dates_set_duration():
    d = TripRequestDraft(start_date=date(2030, 1, 10), end_date=date(2030, 1, 14))
    assert d.duration_days == 5


def test_nonpositive_duration_ignored():
    assert TripRequestDraft(duration_days=0).duration_days is None


def test_bad_llm_date_becomes_none():
    d = extraction_to_draft(TripExtraction(start_date="10th Oct", end_date="2030-01-12"))
    assert d.start_date is None
    assert d.end_date == date(2030, 1, 12)


def test_duration_only_asks_for_start_date():
    missing = missing_fields(TripRequestDraft(destination="Jaipur", duration_days=3))
    assert "Start date" in missing
    assert "Start and end dates" not in missing


# ---------- Live test (calls Gemini; opt-in) ----------

@pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="Set RUN_LIVE_TESTS=1 to call Gemini")
def test_live_complete_request():
    draft = extract_trip_details(
        "Plan a Jaipur trip from Mumbai from 10 Oct to 12 Oct for 2 people with a ₹25,000 budget. "
        "We like history and food. Budget is the priority.",
        today=date(2026, 9, 25),
    )
    assert draft.origin.lower() == "mumbai"
    assert draft.destination.lower() == "jaipur"
    assert draft.start_date == date(2026, 10, 10)
    assert draft.end_date == date(2026, 10, 12)
    assert draft.travellers == 2
    assert draft.budget == 25000
    assert draft.travel_priority == "budget"
    assert set(draft.interests) >= {"history", "food"}