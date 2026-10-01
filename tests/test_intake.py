from datetime import date, timedelta

from src.agent.extractor import ExtractionError
from src.agent.formatter import format_inr
from src.agent.intake import TripIntake, is_affirmative, merge_drafts
from src.agent.schemas import TripRequestDraft

T = date.today()
S = T + timedelta(days=15)
FULL = dict(origin="Mumbai", destination="Jaipur", start_date=S, end_date=S + timedelta(days=2),
            travellers=2, budget=25000, travel_priority="budget", interests=["history", "food"])


class FakeExtractor:
    """Maps a message to a predefined extraction; records every call."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, message, history=None, today=None):
        self.calls.append(message)
        r = self.responses.get(message)
        if isinstance(r, Exception):
            raise r
        return TripRequestDraft(**(r or {}))


def test_complete_request_goes_straight_to_confirmation():
    intake = TripIntake(FakeExtractor({"full": FULL}))
    r = intake.handle("full")
    assert r.stage == "awaiting_confirmation"
    assert r.missing == []
    assert r.summary["destination"] == "Jaipur"
    assert "yes" in r.text.lower()


def test_asks_only_for_missing_then_merges():
    fx = FakeExtractor({
        "trip": {"destination": "Jaipur"},
        "answers": {"origin": "Mumbai", "start_date": S, "end_date": S + timedelta(days=2), "travellers": 2},
        "rest": {"budget": 25000, "travel_priority": "budget"},
    })
    intake = TripIntake(fx)

    r = intake.handle("trip")
    assert r.stage == "collecting"
    assert "Destination" not in r.missing and len(r.missing) == 5

    r = intake.handle("answers")
    assert r.missing == ["Total budget (₹)", "Travel priority: budget, time, or convenience"]

    r = intake.handle("rest")
    assert r.stage == "awaiting_confirmation"
    assert intake.draft.destination == "Jaipur"  # remembered from turn 1


def test_yes_confirms_without_calling_llm():
    fx = FakeExtractor({"full": FULL})
    intake = TripIntake(fx)
    intake.handle("full")
    r = intake.handle("Yes, go ahead!")
    assert r.stage == "confirmed"
    assert intake.trip is not None
    assert fx.calls == ["full"]  # 'yes' never reached the LLM


def test_yes_while_collecting_does_not_confirm():
    intake = TripIntake(FakeExtractor({"trip": {"destination": "Jaipur"}}))
    intake.handle("trip")
    assert intake.handle("yes").stage == "collecting"


def test_edit_during_confirmation_reshows_summary():
    intake = TripIntake(FakeExtractor({"full": FULL, "make it 3 people": {"travellers": 3}}))
    intake.handle("full")
    r = intake.handle("make it 3 people")
    assert r.stage == "awaiting_confirmation"
    assert r.summary["travellers"] == 3


def test_invalid_then_fixed():
    bad = {**FULL, "start_date": T - timedelta(days=5), "end_date": T - timedelta(days=3)}
    fix = {"start_date": S, "end_date": S + timedelta(days=2)}
    intake = TripIntake(FakeExtractor({"bad": bad, "fix": fix}))
    r = intake.handle("bad")
    assert r.stage == "collecting" and "past" in r.errors[0]
    assert intake.handle("fix").stage == "awaiting_confirmation"


def test_extraction_error_keeps_state():
    err = ExtractionError("I couldn't process that message right now. Please try again.")
    intake = TripIntake(FakeExtractor({"full": FULL, "boom": err}))
    intake.handle("full")
    r = intake.handle("boom")
    assert r.text == str(err)
    assert intake.stage == "awaiting_confirmation"
    assert intake.draft.destination == "Jaipur"


def test_confirm_button():
    intake = TripIntake(FakeExtractor({"full": FULL}))
    assert intake.confirm().stage == "collecting"  # nothing to confirm yet
    intake.handle("full")
    assert intake.confirm().stage == "confirmed"


def test_history_is_recorded():
    intake = TripIntake(FakeExtractor({"full": FULL}))
    intake.handle("full")
    assert len(intake.history) == 2


def test_merge_keeps_length_when_start_changes():
    cur = TripRequestDraft(start_date=S, end_date=S + timedelta(days=2))
    merged = merge_drafts(cur, TripRequestDraft(start_date=S + timedelta(days=7)))
    assert merged.end_date == S + timedelta(days=9)


def test_merge_duration_change_moves_end():
    cur = TripRequestDraft(start_date=S, end_date=S + timedelta(days=2))
    merged = merge_drafts(cur, TripRequestDraft(duration_days=5))
    assert merged.end_date == S + timedelta(days=4)


def test_merge_interests_union():
    cur = TripRequestDraft(interests=["history"])
    merged = merge_drafts(cur, TripRequestDraft(interests=["food", "history"]))
    assert merged.interests == ["history", "food"]


def test_affirmative_detection():
    assert is_affirmative("yes") and is_affirmative("Ok, go ahead!") and is_affirmative("looks good")
    assert not is_affirmative("yes but make it 3 people")
    assert not is_affirmative("no")


def test_format_inr():
    assert format_inr(25000) == "₹25,000"
    assert format_inr(150000) == "₹1,50,000"
    assert format_inr(1234567) == "₹12,34,567"