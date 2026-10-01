"""LLM-based extraction of trip details from natural language."""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from langchain_core.messages import BaseMessage

from src.agent.llm import LLMNotConfiguredError, get_llm
from src.agent.prompts import EXTRACTION_PROMPT
from src.agent.schemas import TripExtraction, TripRequestDraft

logger = logging.getLogger("tripmind.extractor")


class ExtractionError(RuntimeError):
    """Carries a user-friendly message; technical details go to the log."""


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        logger.warning("LLM returned an unparseable date: %r", value)
        return None


def extraction_to_draft(ex: TripExtraction) -> TripRequestDraft:
    """Convert raw LLM output into a typed draft (pure Python, no LLM)."""
    return TripRequestDraft(
        origin=ex.origin,
        destination=ex.destination,
        start_date=_parse_date(ex.start_date),
        end_date=_parse_date(ex.end_date),
        duration_days=ex.duration_days,
        travellers=ex.travellers,
        budget=ex.budget,
        travel_priority=ex.travel_priority,
        interests=ex.interests,
        same_city_intended=ex.same_city_intended,
    )


def extract_trip_details(
    message: str,
    history: Optional[list[BaseMessage]] = None,
    today: Optional[date] = None,
) -> TripRequestDraft:
    """Extract trip details stated in `message`. Unstated fields stay None."""
    today = today or date.today()
    try:
        llm = get_llm().with_structured_output(TripExtraction)
        chain = EXTRACTION_PROMPT | llm
        result = chain.invoke(
            {
                "message": message,
                "history": history or [],
                "today": today.isoformat(),
                "weekday": today.strftime("%A"),
            }
        )
    except LLMNotConfiguredError as exc:
        logger.error("LLM not configured: %s", exc)
        raise ExtractionError("The AI model isn't configured yet. Please add a Gemini API key.") from exc
    except Exception as exc:  # network, quota, parsing...
        logger.exception("Trip extraction failed")
        raise ExtractionError("I couldn't process that message right now. Please try again.") from exc

    if result is None:
        logger.error("LLM returned no structured output for: %r", message)
        raise ExtractionError("I couldn't understand the trip details. Could you rephrase?")
    return extraction_to_draft(result)


# ---------------------------------------------------------------------------
# Manual demo:  python -m src.agent.extractor
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    from langchain_core.messages import AIMessage, HumanMessage

    from src.agent.validation import validate_draft

    logging.basicConfig(level=logging.WARNING)

    cases = [
        ("Complete request",
         "Plan a Jaipur trip from Mumbai from 10 Oct to 12 Oct for 2 people with a ₹25,000 budget. "
         "We like history and food. Budget is the priority.", None),
        ("Missing details", "Plan a trip to Jaipur.", None),
        ("Duration but no dates",
         "3 days in Jaipur from Mumbai, two of us, ₹25k, history and food, budget trip.", None),
        ("Relative dates",
         "Me and my wife want to go from Pune to Lonavala this weekend, ₹15k total, "
         "hassle-free please, we love nature.", None),
        ("Follow-up answer (uses history)",
         "From Delhi, 5 to 8 Nov, 3 people",
         [HumanMessage("Plan a trip to Jaipur."),
          AIMessage("I can do that. I still need: starting city, dates, number of travellers, "
                    "total budget, and travel priority.")]),
    ]

    for name, msg, history in cases:
        print(f"\n=== {name} ===\nUSER: {msg}")
        try:
            draft = extract_trip_details(msg, history=history)
        except ExtractionError as e:
            print("ERROR:", e)
            continue
        print(json.dumps(draft.model_dump(mode="json", exclude_none=True), indent=2, ensure_ascii=False))
        print(validate_draft(draft).describe())