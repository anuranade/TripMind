"""Deterministic formatting for TripMind replies (no LLM involved)."""
from __future__ import annotations

from datetime import date
from typing import Optional

from src.agent.schemas import TripRequest, TripRequestDraft
from src.tools.geo import place_label


def format_inr(amount: float) -> str:
    """Indian digit grouping: 150000 -> ₹1,50,000."""
    n = int(round(amount))
    sign = "-" if n < 0 else ""
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups) + "," + tail
    return f"{sign}₹{s}"


def dates_label(start: date, end: date) -> str:
    if start.year == end.year:
        return f"{start.day} {start:%b} – {end.day} {end:%b %Y}"
    return f"{start.day} {start:%b %Y} – {end.day} {end:%b %Y}"


def known_details(d: TripRequestDraft) -> list[str]:
    """Short list of what we already know, for acknowledging the user."""
    parts: list[str] = []
    if d.origin and d.destination:
        parts.append(f"{d.origin} → {d.destination}")
    elif d.destination:
        parts.append(f"to {d.destination}")
    elif d.origin:
        parts.append(f"from {d.origin}")

    if d.start_date and d.end_date:
        parts.append(dates_label(d.start_date, d.end_date))
    elif d.duration_days:
        parts.append(f"{d.duration_days} days")
    elif d.start_date:
        parts.append(f"from {d.start_date.day} {d.start_date:%b}")
    elif d.end_date:
        parts.append(f"until {d.end_date.day} {d.end_date:%b}")

    if d.travellers is not None:
        parts.append(f"{d.travellers} traveller{'s' if d.travellers != 1 else ''}")
    if d.budget is not None:
        parts.append(format_inr(d.budget))
    if d.travel_priority:
        parts.append(f"{d.travel_priority} priority")
    if d.interests:
        parts.append(", ".join(d.interests))
    return parts


def follow_up_text(draft: TripRequestDraft, missing: list[str]) -> str:
    known = known_details(draft)
    lines = ["I can do that." if known else "Happy to help you plan a trip!"]
    if known:
        lines.append("So far: " + " · ".join(known))
    lines.append("")
    lines.append("I still need:" if known else "To get started, please tell me:")
    lines += [f"• {m}" for m in missing]
    return "\n".join(lines)


def errors_text(errors: list[str]) -> str:
    lines = ["Before I can plan this, a few things need fixing:"]
    lines += [f"• {e}" for e in errors]
    return "\n".join(lines)


def trip_summary_card(trip: TripRequest, places: Optional[dict] = None) -> dict:
    """Structured summary the web UI will render as a card (Phase 15)."""
    places = places or {}
    o, d = places.get("origin"), places.get("destination")
    return {
        "origin": trip.origin,
        "destination": trip.destination,
        "origin_label": place_label(o) if o else trip.origin,
        "destination_label": place_label(d) if d else trip.destination,
        "origin_coords": [o.latitude, o.longitude] if o else None,
        "destination_coords": [d.latitude, d.longitude] if d else None,
        "start_date": trip.start_date.isoformat(),
        "end_date": trip.end_date.isoformat(),
        "dates_label": dates_label(trip.start_date, trip.end_date),
        "days": trip.num_days,
        "travellers": trip.travellers,
        "budget": trip.budget,
        "budget_label": format_inr(trip.budget),
        "priority": trip.travel_priority,
        "interests": trip.interests,
    }


def confirmation_text(trip: TripRequest, places: Optional[dict] = None) -> str:
    card = trip_summary_card(trip, places)
    interests = ", ".join(card["interests"]) if card["interests"] else "none specified"
    lines = ["Here's your trip — please check it:",
             f"• Route: {card['origin']} → {card['destination']}"]
    if (card["origin_label"], card["destination_label"]) != (card["origin"], card["destination"]):
        lines.append(f"  ({card['origin_label']} → {card['destination_label']})")
    lines += [
        f"• Dates: {card['dates_label']} ({card['days']} days)",
        f"• Travellers: {card['travellers']}",
        f"• Budget: {card['budget_label']}",
        f"• Priority: {card['priority']}",
        f"• Interests: {interests}",
        "",
        'Shall I start planning? Reply "yes" to confirm, or tell me what to change.',
    ]
    return "\n".join(lines)