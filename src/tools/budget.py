"""Deterministic trip budget estimation (all numbers from local data, never the LLM)."""
from __future__ import annotations

import logging
import math
from datetime import date
from functools import lru_cache
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from src.agent.formatter import format_inr
from src.agent.schemas import MAX_TRAVELLERS, MAX_TRIP_DAYS, BudgetBreakdown, TransportOption
from src.tools.common import ToolError
from src.tools.transport import WEIGHTS, compare, get_route, same_country
from src.utils.config import BASE_DIR

logger = logging.getLogger("tripmind.tools.budget")

DEST_CSV = BASE_DIR / "data" / "destinations.csv"
COUNTRY_CSV = BASE_DIR / "data" / "country_cost_levels.csv"
BUFFER_RATE = 0.10
ROOM_CAPACITY = 2
METRO_POP, CITY_POP = 4_000_000, 300_000
TIER_LABEL = {"_metro": "major metro", "_city": "city", "_town": "smaller town"}
STYLE_ORDER = ["comfort", "mid", "budget"]                       # most to least expensive
PREFERRED_STYLE = {"budget": "budget", "time": "mid", "convenience": "comfort"}
BUDGET_ERROR = "The budget could not be estimated right now."
DISCLAIMER = "All amounts are estimates from TripMind's reference data — not quotes or live prices."
MODE_ALIASES = {"plane": "flight", "fly": "flight", "air": "flight", "taxi": "cab",
                "car": "self drive", "drive": "self drive", "rail": "train"}
REQUIRED_COLUMNS = {"city", "aliases", "attractions_per_person_day"} | {
    f"{kind}_{style}" for kind in ("stay", "food", "local") for style in ("budget", "mid", "comfort")
}


# ---------------------------------------------------------------------------
# Cost data
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_destinations() -> pd.DataFrame:
    df = pd.read_csv(DEST_CSV)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"destinations.csv is missing columns: {sorted(missing)}")
    if "_default" not in set(df["city"]):
        raise ValueError("destinations.csv needs a '_default' row")
    return df


@lru_cache(maxsize=1)
def load_country_levels() -> dict[str, str]:
    try:
        df = pd.read_csv(COUNTRY_CSV)
        return {str(c).upper(): str(level).strip().lower() for c, level in zip(df["country_code"], df["level"])}
    except FileNotFoundError:
        logger.warning("country_cost_levels.csv not found; all countries treated as mid-cost")
        return {}


def _special_row(key: str) -> dict:
    df = load_destinations()
    rows = df[df["city"] == key]
    if len(rows) == 0:
        rows = df[df["city"] == "_default"]
    return rows.iloc[0].to_dict()


def find_destination(name: str) -> tuple[dict, bool]:
    """(cost row, found?) — a listed city or alias, otherwise the _default row."""
    key = (name or "").strip().casefold()
    df = load_destinations()
    for row in df.to_dict("records"):
        if str(row["city"]).startswith("_"):
            continue
        names = [row["city"]] + (row["aliases"].split("|") if isinstance(row["aliases"], str) else [])
        if key in {n.strip().casefold() for n in names}:
            return row, True
    return _special_row("_default"), False


def cost_profile(destination: str, country_code: Optional[str] = None,
                 population: Optional[int] = None) -> tuple[dict, Optional[str]]:
    """Cost row for any destination: listed city → Indian size tier → international level → default."""
    row, found = find_destination(destination)
    if found:
        return row, None
    cc = (country_code or "").upper()
    if not cc:
        return row, f"No city-specific data for {destination}; national average estimates used."
    if cc == "IN":
        pop = population or 0
        key = "_metro" if pop >= METRO_POP else "_city" if pop >= CITY_POP else "_town"
        return _special_row(key), (f"No city-specific data for {destination}; typical costs for an "
                                   f"Indian {TIER_LABEL[key]} used.")
    level = load_country_levels().get(cc, "mid")
    if level not in ("premium", "mid", "budget"):
        level = "mid"
    return _special_row(f"_intl_{level}"), (f"{destination} is outside India; typical {level}-cost "
                                            "international estimates (in ₹) used.")


def _r10(x: float) -> float:
    return float(round(x, -1))


# ---------------------------------------------------------------------------
# Core calculation (pure Python)
# ---------------------------------------------------------------------------
def calculate_budget(*, destination: str, num_days: int, travellers: int, budget: float, style: str,
                     transport_cost: float, transport_label: str,
                     attraction_fees_per_person: Optional[list[float]] = None,
                     country_code: Optional[str] = None, population: Optional[int] = None
                     ) -> tuple[BudgetBreakdown, list[str]]:
    row, profile_note = cost_profile(destination, country_code, population)
    nights = max(num_days - 1, 0)
    rooms = math.ceil(travellers / ROOM_CAPACITY)

    transport = _r10(transport_cost)
    stay = _r10(rooms * nights * row[f"stay_{style}"])
    food = _r10(travellers * num_days * row[f"food_{style}"])
    local = _r10(num_days * row[f"local_{style}"])
    if attraction_fees_per_person is not None:
        fees = [max(float(f), 0.0) for f in attraction_fees_per_person]
        attractions = _r10(sum(fees) * travellers)
        attr_note = f"Entry fees for {len(fees)} planned attraction(s) × {travellers} traveller(s)"
    else:
        attractions = _r10(row["attractions_per_person_day"] * travellers * num_days)
        attr_note = (f"Typical entry fees ~{format_inr(row['attractions_per_person_day'])} "
                     f"per person per day")

    subtotal = transport + stay + food + local + attractions
    buffer = _r10(subtotal * BUFFER_RATE)
    total = subtotal + buffer

    breakdown = BudgetBreakdown(
        transport=transport, stay=stay, food=food, local_transport=local, attractions=attractions,
        buffer=buffer, total=total, budget=float(budget),
        within_budget=total <= budget, remaining=float(budget - total),
    )

    assumptions = [
        f"Transport: {transport_label}, round trip for {travellers}",
        (f"Stay: {rooms} room(s) × {nights} night(s) at ~{format_inr(row[f'stay_{style}'])}/night ({style} hotels)"
         if nights else "Stay: none (day trip)"),
        f"Food: ~{format_inr(row[f'food_{style}'])} per person per day ({style})",
        f"Local transport: ~{format_inr(row[f'local_{style}'])} per day for the group ({style})",
        attr_note,
        f"Buffer: {int(BUFFER_RATE * 100)}% for unexpected costs",
    ]
    if profile_note:
        assumptions.append(profile_note)
    return breakdown, assumptions


def status_message(b: BudgetBreakdown) -> str:
    if b.within_budget:
        return (f"Estimated total {format_inr(b.total)} is within your {format_inr(b.budget)} budget — "
                f"{format_inr(b.remaining)} to spare.")
    return (f"Estimated total {format_inr(b.total)} is OVER your {format_inr(b.budget)} budget by "
            f"{format_inr(-b.remaining)}.")


def suggestions(b: BudgetBreakdown, chosen: TransportOption, options: list[TransportOption],
                style: str, num_days: int, row: dict, travellers: int) -> list[str]:
    """Simple deterministic hints when over budget (no automatic optimisation)."""
    if b.within_budget:
        return []
    tips: list[str] = []
    cheapest = min(options, key=lambda o: o.estimated_cost)
    if cheapest.estimated_cost < chosen.estimated_cost:
        tips.append(f"Switching to {cheapest.mode} would save about "
                    f"{format_inr(chosen.estimated_cost - cheapest.estimated_cost)} on transport.")
    if num_days > 1:
        per_day = (math.ceil(travellers / ROOM_CAPACITY) * row[f"stay_{style}"]
                   + travellers * row[f"food_{style}"] + row[f"local_{style}"])
        tips.append(f"Shortening the trip by one day would save about {format_inr(_r10(per_day * 1.1))}.")
    if not tips:
        tips.append("This trip may need a larger budget.")
    return tips[:3]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_dates(start_date: str, end_date: str) -> int:
    try:
        s, e = date.fromisoformat(start_date.strip()), date.fromisoformat(end_date.strip())
    except (ValueError, AttributeError):
        raise ToolError("Dates must be in YYYY-MM-DD format.")
    days = (e - s).days + 1
    if not 1 <= days <= MAX_TRIP_DAYS:
        raise ToolError(f"Trip length must be 1–{MAX_TRIP_DAYS} days.")
    return days


def match_mode(options: list[TransportOption], query: str) -> Optional[TransportOption]:
    """Find the option the user means: 'bus', 'Train (3AC)', 'plane', 'self-drive'..."""
    def norm(t: str) -> str:
        return t.casefold().replace("-", " ").replace("_", " ").strip()

    q = norm(query or "")
    q = MODE_ALIASES.get(q, q)
    if not q:
        return None
    for o in options:
        if q in norm(o.mode):
            return o
    return None


@tool
def estimate_budget(destination: str, origin_latitude: float, origin_longitude: float,
                    destination_latitude: float, destination_longitude: float,
                    start_date: str, end_date: str, travellers: int, budget: float,
                    travel_priority: str, transport_mode: Optional[str] = None,
                    attraction_fees_per_person: Optional[list[float]] = None,
                    origin_country_code: Optional[str] = None,
                    destination_country_code: Optional[str] = None,
                    destination_population: Optional[int] = None) -> dict:
    """Estimate the full trip cost and compare it with the user's budget:
    round-trip inter-city transport + accommodation + food + local transport + attraction entry fees
    + 10% buffer. Pass origin_country_code, destination_country_code and destination_population
    from the trip state so costs match the destination. Transport defaults to the mode recommended
    for the priority; pass transport_mode (e.g. 'bus', 'train', 'flight') to use a specific one.
    Hotels and food are estimated at the most comfortable level that fits the budget.
    All amounts are ESTIMATES in INR."""
    try:
        if travel_priority not in WEIGHTS:
            raise ToolError("travel_priority must be 'budget', 'time' or 'convenience'.")
        travellers = int(travellers)
        if not 1 <= travellers <= MAX_TRAVELLERS:
            raise ToolError(f"travellers must be between 1 and {MAX_TRAVELLERS}.")
        if budget <= 0:
            raise ToolError("budget must be a positive amount.")
        num_days = _parse_dates(start_date, end_date)

        route = get_route(origin_latitude, origin_longitude, destination_latitude, destination_longitude)
        options = compare(route, travellers, travel_priority,
                          same_country(origin_country_code, destination_country_code))
        chosen = options[0]
        if transport_mode:
            chosen = match_mode(options, transport_mode)
            if chosen is None:
                raise ToolError(f"'{transport_mode}' isn't a feasible option for this route. "
                                f"Options: {', '.join(o.mode for o in options)}.")

        # Most comfortable level allowed by the priority that still fits the budget.
        preferred = PREFERRED_STYLE[travel_priority]
        for style in STYLE_ORDER[STYLE_ORDER.index(preferred):]:
            breakdown, assumptions = calculate_budget(
                destination=destination, num_days=num_days, travellers=travellers, budget=budget,
                style=style, transport_cost=chosen.estimated_cost, transport_label=chosen.mode,
                attraction_fees_per_person=attraction_fees_per_person,
                country_code=destination_country_code, population=destination_population,
            )
            if breakdown.within_budget:
                break
        if style != preferred:
            assumptions.append(f"Hotels and food estimated at {style} level to fit your budget.")

        row, _ = cost_profile(destination, destination_country_code, destination_population)
        return {
            "ok": True,
            "breakdown": breakdown.model_dump(),
            "per_person": _r10(breakdown.total / travellers),
            "within_budget": breakdown.within_budget,
            "status_message": status_message(breakdown),
            "transport_mode": chosen.mode,
            "spending_style": style,
            "days": num_days,
            "nights": max(num_days - 1, 0),
            "assumptions": assumptions,
            "suggestions": suggestions(breakdown, chosen, options, style, num_days, row, travellers),
            "disclaimer": DISCLAIMER,
        }
    except ToolError as exc:
        return {"ok": False, "error": exc.user_message}
    except Exception:
        logger.exception("Unexpected budget estimation error")
        return {"ok": False, "error": BUDGET_ERROR}


# ---------------------------------------------------------------------------
# Manual demo:  python -m src.tools.budget [origin] [destination] [days] [travellers] [budget] [priority]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from datetime import timedelta

    from src.tools.geo import geocode

    logging.basicConfig(level=logging.WARNING)
    a = sys.argv[1:] + [None] * 6
    origin, dest = a[0] or "Mumbai", a[1] or "Jaipur"
    days, n = int(a[2] or 3), int(a[3] or 2)
    budget, priority = float(a[4] or 25000), a[5] or "budget"

    try:
        o, d = geocode(origin), geocode(dest)
    except ToolError as exc:
        print(exc.user_message)
        sys.exit(1)

    start = date.today() + timedelta(days=15)
    out = estimate_budget.invoke({
        "destination": d.name, "origin_latitude": o.latitude, "origin_longitude": o.longitude,
        "destination_latitude": d.latitude, "destination_longitude": d.longitude,
        "start_date": start.isoformat(), "end_date": (start + timedelta(days=days - 1)).isoformat(),
        "travellers": n, "budget": budget, "travel_priority": priority,
        "origin_country_code": o.country_code, "destination_country_code": d.country_code,
        "destination_population": d.population})
    if not out["ok"]:
        print(out["error"])
        sys.exit(1)
    b = out["breakdown"]
    print(f"\n{o.name} → {d.name} ({d.country_code}, pop {d.population or 'unknown'}) · {days} days · "
          f"{n} travellers · {out['transport_mode']} · {out['spending_style']} level")
    for key in ("transport", "stay", "food", "local_transport", "attractions", "buffer", "total"):
        print(f"  {key.replace('_', ' ').capitalize():<16} {format_inr(b[key]):>10}")
    print(f"  {out['status_message']}")
    for line in out["assumptions"]:
        print(f"   · {line}")