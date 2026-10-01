"""Trip intake: collects mandatory details over several turns, asks only for what's
missing, verifies the cities exist, and requires confirmation before planning."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.agent.extractor import ExtractionError, extract_trip_details
from src.agent.formatter import confirmation_text, errors_text, follow_up_text, trip_summary_card
from src.agent.schemas import GeoPlace, TripRequest, TripRequestDraft
from src.agent.validation import validate_draft
from src.tools.common import ToolError
from src.tools.geo import PlaceNotFoundError, geocode

logger = logging.getLogger("tripmind.intake")

Stage = Literal["collecting", "awaiting_confirmation", "confirmed"]
HISTORY_WINDOW = 6  # recent messages passed to the extractor for context

_AFF = (
    r"(?:yes|yeah|yep|yup|y|ok|okay|sure|confirm(?:ed)?|looks good|sounds good|go ahead|"
    r"proceed|correct|that'?s right|perfect|do it|plan it|please|let'?s go)"
)
_AFFIRMATIVE = re.compile(rf"^\s*{_AFF}(?:[\s,.!]+{_AFF})*[\s.!]*$", re.IGNORECASE)


def is_affirmative(text: str) -> bool:
    """True only for pure confirmations like 'yes', 'ok go ahead!'.
    'yes but 3 people' is NOT pure, so it is treated as an edit."""
    return bool(_AFFIRMATIVE.match(text))


def _is_empty(d: TripRequestDraft) -> bool:
    return (
        all(
            getattr(d, f) is None
            for f in ("origin", "destination", "start_date", "end_date",
                      "duration_days", "travellers", "budget", "travel_priority")
        )
        and not d.interests
        and not d.same_city_intended
    )


def merge_drafts(current: TripRequestDraft, update: TripRequestDraft) -> TripRequestDraft:
    """Apply newly extracted details on top of what we already know (pure Python)."""

    def pick(name):
        new = getattr(update, name)
        return new if new is not None else getattr(current, name)

    # Dates: keep the trip length when only the start date (or only the length) changes.
    start = update.start_date or current.start_date
    if update.end_date:
        end = update.end_date
    elif update.start_date:
        end = None if (update.duration_days or current.duration_days) else current.end_date
    elif update.duration_days:
        end = None if current.start_date else current.end_date
    else:
        end = current.end_date
    duration = update.duration_days or current.duration_days
    if update.start_date and update.end_date:
        duration = None  # recomputed from the explicit dates

    interests = list(current.interests)
    interests += [i for i in update.interests if i not in interests]

    return TripRequestDraft(
        origin=pick("origin"),
        destination=pick("destination"),
        start_date=start,
        end_date=end,
        duration_days=duration,
        travellers=pick("travellers"),
        budget=pick("budget"),
        travel_priority=pick("travel_priority"),
        interests=interests,
        same_city_intended=current.same_city_intended or update.same_city_intended,
    )


@dataclass
class IntakeReply:
    stage: Stage
    text: str
    missing: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    summary: Optional[dict] = None  # trip card data when a full trip is known


ExtractorFn = Callable[..., TripRequestDraft]
GeocoderFn = Callable[[str], GeoPlace]


class TripIntake:
    def __init__(self, extractor: Optional[ExtractorFn] = None, geocoder: Optional[GeocoderFn] = None) -> None:
        self.extractor = extractor or extract_trip_details
        self.geocoder = geocoder or geocode
        self.reset()

    def reset(self) -> None:
        self.draft = TripRequestDraft()
        self.stage: Stage = "collecting"
        self.trip: Optional[TripRequest] = None
        self.places: dict[str, GeoPlace] = {}
        self.history: list[BaseMessage] = []

    # ---------- public API ----------
    def handle(self, message: str) -> IntakeReply:
        message = (message or "").strip()
        if not message:
            return IntakeReply(self.stage, "Please tell me about the trip you have in mind.")

        if self.stage == "confirmed":
            return self._record(message, IntakeReply(
                "confirmed",
                "Your trip is confirmed. Planning will be available once the planning agent is connected.",
                summary=self._card(),
            ))

        if self.stage == "awaiting_confirmation" and is_affirmative(message):
            return self._record(message, self._confirm())

        try:
            update = self.extractor(message, history=self.history[-HISTORY_WINDOW:])
        except ExtractionError as exc:
            return self._record(message, IntakeReply(self.stage, str(exc)))

        if _is_empty(update) and self.stage == "awaiting_confirmation":
            return self._record(message, IntakeReply(
                self.stage,
                'Reply "yes" to confirm, or tell me what you\'d like to change.',
                summary=self._card(),
            ))

        self.draft = merge_drafts(self.draft, update)
        logger.info("Draft now: %s", self.draft.model_dump(exclude_none=True))
        return self._record(message, self._evaluate())

    def confirm(self) -> IntakeReply:
        """Called by the UI's Confirm button (Phase 15)."""
        if self.stage == "awaiting_confirmation":
            reply = self._confirm()
        elif self.stage == "collecting":
            reply = self._evaluate()
        else:
            reply = IntakeReply(self.stage, "Your trip is already confirmed.", summary=self._card())
        self.history.append(AIMessage(reply.text))
        return reply

    # ---------- internals ----------
    def _card(self) -> Optional[dict]:
        return trip_summary_card(self.trip, self.places) if self.trip else None

    def _evaluate(self) -> IntakeReply:
        result = validate_draft(self.draft)
        if result.missing:
            self.stage, self.trip = "collecting", None
            return IntakeReply("collecting", follow_up_text(self.draft, result.missing), missing=result.missing)
        if result.errors:
            self.stage, self.trip = "collecting", None
            return IntakeReply("collecting", errors_text(result.errors), errors=result.errors)

        problem = self._verify_places(result.trip)
        if problem:
            self.stage, self.trip = "collecting", None
            return problem

        self.trip, self.stage = result.trip, "awaiting_confirmation"
        return IntakeReply("awaiting_confirmation", confirmation_text(self.trip, self.places),
                           summary=self._card())

    def _verify_places(self, trip: TripRequest) -> Optional[IntakeReply]:
        """Geocode origin and destination. Returns an error reply, or None if both are fine."""
        not_found, fields = [], []
        for role, label in (("origin", "starting city"), ("destination", "destination")):
            try:
                self.places[role] = self.geocoder(getattr(trip, role))
            except PlaceNotFoundError as exc:
                self.places.pop(role, None)
                setattr(self.draft, role, None)  # ask for it again
                not_found.append(exc.user_message)
                fields.append(label)
            except ToolError as exc:
                logger.warning("Geocoding unavailable: %s", exc.user_message)
                return IntakeReply(
                    "collecting",
                    f"{exc.user_message}\nI need it to verify your cities — "
                    'please say "try again" in a moment.',
                    errors=[exc.user_message],
                )

        if not_found:
            text = "\n".join(not_found) + f"\n\nPlease tell me the correct {' and '.join(fields)}."
            return IntakeReply("collecting", text, errors=not_found)

        o, d = self.places["origin"], self.places["destination"]
        if not trip.same_city_intended and (o.name, o.admin1, o.country_code) == (d.name, d.admin1, d.country_code):
            msg = (f"{trip.origin} and {trip.destination} both resolve to {place_label_short(o)}. "
                   "Please check them, or tell me if you want a trip within the same city.")
            return IntakeReply("collecting", errors_text([msg]), errors=[msg])
        return None

    def _confirm(self) -> IntakeReply:
        self.stage = "confirmed"
        return IntakeReply(
            "confirmed",
            "Trip confirmed! Next I'll check the weather, compare transport, "
            "find attractions and estimate your budget.",
            summary=self._card(),
        )

    def _record(self, user_msg: str, reply: IntakeReply) -> IntakeReply:
        self.history += [HumanMessage(user_msg), AIMessage(reply.text)]
        return reply


def place_label_short(p: GeoPlace) -> str:
    return f"{p.name}, {p.country}" if p.country else p.name


# ---------------------------------------------------------------------------
# Interactive demo:  python -m src.agent.intake
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    intake = TripIntake()
    print("TripMind intake demo — type 'reset' to start over, 'quit' to exit.\n")
    while True:
        try:
            msg = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        cmd = msg.strip().lower()
        if cmd in {"quit", "exit"}:
            break
        if cmd == "reset":
            intake.reset()
            print("(conversation reset)\n")
            continue
        reply = intake.handle(msg)
        print(f"\nTripMind [{reply.stage}]:\n{reply.text}\n")