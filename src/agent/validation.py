"""Turns a TripRequestDraft into either a valid TripRequest,
a list of missing fields, or a list of user-friendly errors."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pydantic import ValidationError

from src.agent.schemas import MAX_TRAVELLERS, TripRequest, TripRequestDraft

# Friendly messages for built-in constraint / type errors, by field name.
FIELD_MESSAGES = {
    "origin": "Please give a valid starting city.",
    "destination": "Please give a valid destination.",
    "start_date": "Please give a valid start date.",
    "end_date": "Please give a valid end date.",
    "travellers": f"Number of travellers must be between 1 and {MAX_TRAVELLERS}.",
    "budget": "The budget must be a positive amount.",
    "travel_priority": "Travel priority must be budget, time, or convenience.",
    "interests": "Interests should be a list such as history, food, nature.",
}


@dataclass
class ValidationResult:
    trip: Optional[TripRequest] = None
    missing: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.trip is not None

    def describe(self) -> str:
        if self.ok:
            return f"VALID: {self.trip.summary()}"
        if self.missing:
            return "MISSING:\n  • " + "\n  • ".join(self.missing)
        return "INVALID:\n  • " + "\n  • ".join(self.errors)


def missing_fields(draft: TripRequestDraft) -> list[str]:
    """Friendly labels for mandatory fields that are still unknown."""
    missing: list[str] = []
    if not draft.origin:
        missing.append("Starting city")
    if not draft.destination:
        missing.append("Destination")
    if draft.start_date is None and draft.end_date is None:
        missing.append("Start date" if draft.duration_days else "Start and end dates")
    elif draft.start_date is None:
        missing.append("Start date")
    elif draft.end_date is None:
        missing.append("End date")
    if draft.travellers is None:
        missing.append("Number of travellers")
    if draft.budget is None:
        missing.append("Total budget (₹)")
    if draft.travel_priority is None:
        missing.append("Travel priority: budget, time, or convenience")
    return missing


def friendly_errors(exc: ValidationError) -> list[str]:
    messages: list[str] = []
    for err in exc.errors():
        loc = err.get("loc") or ()
        field_name = loc[0] if loc else None
        if err.get("type") == "value_error":
            msg = err["msg"].removeprefix("Value error, ")
        else:
            msg = FIELD_MESSAGES.get(field_name, f"Invalid value for {field_name}.")
        if msg not in messages:
            messages.append(msg)
    return messages


def validate_draft(draft: TripRequestDraft) -> ValidationResult:
    missing = missing_fields(draft)
    if missing:
        return ValidationResult(missing=missing)
    try:
        trip = TripRequest(**draft.model_dump())
    except ValidationError as exc:
        return ValidationResult(errors=friendly_errors(exc))
    return ValidationResult(trip=trip)


# ---------------------------------------------------------------------------
# Manual demo:  python -m src.agent.validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import date, timedelta

    t = date.today()
    cases = {
        "Complete request": TripRequestDraft(
            origin="Mumbai", destination="Jaipur",
            start_date=t + timedelta(days=15), end_date=t + timedelta(days=17),
            travellers=2, budget=25000, travel_priority="budget",
            interests=["History", " food ", "history"],
        ),
        "Missing details": TripRequestDraft(destination="Jaipur"),
        "Past start date": TripRequestDraft(
            origin="Mumbai", destination="Jaipur",
            start_date=t - timedelta(days=5), end_date=t - timedelta(days=3),
            travellers=2, budget=25000, travel_priority="time",
        ),
        "Trip too long": TripRequestDraft(
            origin="Mumbai", destination="Jaipur",
            start_date=t + timedelta(days=10), end_date=t + timedelta(days=25),
            travellers=2, budget=25000, travel_priority="time",
        ),
        "Same city": TripRequestDraft(
            origin="Mumbai", destination=" mumbai ",
            start_date=t + timedelta(days=5), end_date=t + timedelta(days=6),
            travellers=1, budget=5000, travel_priority="convenience",
        ),
        "Bad travellers and budget": TripRequestDraft(
            origin="Mumbai", destination="Goa",
            start_date=t + timedelta(days=5), end_date=t + timedelta(days=7),
            travellers=15, budget=-100, travel_priority="budget",
        ),
    }
    for name, draft in cases.items():
        print(f"\n=== {name} ===")
        print(validate_draft(draft).describe())