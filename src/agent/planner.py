"""Builds and refines the Itinerary: the LLM organises the days, Python validates names,
adds Plan Bs, and takes every fact (weather, fees, transport, budget) from the tools."""
from __future__ import annotations

import difflib
import json
import logging
from datetime import timedelta
from typing import Optional

from src.agent.agent import ResearchResult, ToolCallRecord, run_research, run_tool, trip_state
from src.agent.followup import FollowUpIntent
from src.agent.formatter import dates_label, format_inr
from src.agent.llm import get_llm
from src.agent.prompts import PLANNER_PROMPT, REFINE_PROMPT
from src.agent.schemas import (Activity, BudgetBreakdown, DayPlan, Itinerary, PlanDraft,
                               PlannedActivity, PlannedDay, TransportOption, TransportPlan,
                               TripRequest, WeatherDay)
from src.tools.budget import estimate_budget, match_mode
from src.tools.transport import hours_label

logger = logging.getLogger("tripmind.planner")

SLOTS = ("morning", "afternoon", "evening")
RISKY = {"CAUTION", "HAZARD", "UNKNOWN"}
FREE = "FREE"
FREE_NAME = "Free time"
_INVALID = object()


# ---------------------------------------------------------------------------
# Facts for the planner
# ---------------------------------------------------------------------------
def weather_days(research: ResearchResult, trip: TripRequest) -> list[WeatherDay]:
    """One WeatherDay per trip day; UNKNOWN if the hazard tool gave nothing for that date."""
    hz = research.latest("check_weather_hazards")
    by_date = {}
    for d in (hz or {}).get("days", []):
        wd = WeatherDay(**d)
        by_date[wd.date] = wd
    days = []
    for i in range(trip.num_days):
        day = trip.start_date + timedelta(days=i)
        days.append(by_date.get(day) or WeatherDay(
            date=day, forecast_available=False, status="UNKNOWN",
            reason="Weather information could not be retrieved right now."))
    return days


def build_facts(trip: TripRequest, research: ResearchResult, weather: list[WeatherDay]) -> dict:
    tr = research.latest("compare_transport_modes")
    return {
        "trip": {"origin": trip.origin, "destination": trip.destination, "num_days": trip.num_days,
                 "travellers": trip.travellers, "travel_priority": trip.travel_priority,
                 "interests": trip.interests},
        "arrival_transport": ({"mode": tr["recommended"]["mode"],
                               "one_way_hours": tr["recommended"]["travel_time_hours"]} if tr else None),
        "days": [{"day_number": i + 1, "date": w.date.isoformat(), "weekday": w.date.strftime("%A"),
                  "weather_status": w.status, "weather_reason": w.reason, "condition": w.condition}
                 for i, w in enumerate(weather)],
        "attractions": [{k: a.get(k) for k in ("name", "category", "setting", "interest", "distance_km",
                                                "notable", "est_entry_fee_per_person")}
                        for a in research.attractions()],
    }


def plan_brief(it: Itinerary) -> dict:
    """Compact view of the current plan for the router, Q&A and refinement prompts."""
    t, tp, b = it.trip_summary, it.transport_plan, it.budget_breakdown
    return {
        "trip": {"origin": t.origin, "destination": t.destination, "start_date": t.start_date.isoformat(),
                 "end_date": t.end_date.isoformat(), "num_days": t.num_days, "travellers": t.travellers,
                 "budget": t.budget, "travel_priority": t.travel_priority, "interests": t.interests},
        "transport": ({"mode": tp.recommended.mode, "estimated_round_trip_cost": tp.recommended.estimated_cost,
                       "one_way_hours": tp.recommended.travel_time_hours} if tp else None),
        "weather": [{"date": w.date.isoformat(), "status": w.status, "condition": w.condition,
                     "temp_min_c": w.temp_min_c, "temp_max_c": w.temp_max_c} for w in it.weather],
        "days": [{"day_number": d.day_number, "date": d.date.isoformat(), "theme": d.theme,
                  "activities": [{"time_of_day": a.time_of_day,
                                  "name": FREE if a.name == FREE_NAME else a.name,
                                  "setting": a.setting, "plan_b": a.plan_b} for a in d.activities]}
                 for d in it.daily_plan],
        "budget": ({"estimated_total": b.total, "budget": b.budget, "within_budget": b.within_budget,
                    "remaining": b.remaining} if b else None),
    }


# ---------------------------------------------------------------------------
# Day-plan drafts
# ---------------------------------------------------------------------------
def draft_plan(facts: dict, prompt=PLANNER_PROMPT) -> Optional[PlanDraft]:
    try:
        chain = prompt | get_llm().with_structured_output(PlanDraft)
        return chain.invoke({"facts": json.dumps(facts, ensure_ascii=False, indent=1)})
    except Exception:
        logger.exception("Planner LLM failed")
        return None


def fallback_plan(facts: dict) -> PlanDraft:
    """Simple deterministic plan used if the planner LLM is unavailable."""
    atts = facts["attractions"]
    food = [a for a in atts if a["interest"] == "food"]
    others = [a for a in atts if a["interest"] != "food"]
    used: set[str] = set()

    def take(pool, indoor_only=False):
        for a in pool:
            if a["name"] not in used and (not indoor_only or a["setting"] == "indoor"):
                used.add(a["name"])
                return a["name"]
        return FREE

    days = []
    for day in facts["days"]:
        hazard = day["weather_status"] == "HAZARD"
        days.append(PlannedDay(day_number=day["day_number"], activities=[
            PlannedActivity(time_of_day="morning", attraction=take(others, hazard)),
            PlannedActivity(time_of_day="afternoon", attraction=take(others, hazard)),
            PlannedActivity(time_of_day="evening", attraction=take(food or others)),
        ]))
    return PlanDraft(intro="", days=days, tips=[])


def draft_from_itinerary(it: Itinerary) -> PlanDraft:
    """Turn an existing itinerary back into a draft (used when only transport changes)."""
    return PlanDraft(intro="", tips=[], days=[
        PlannedDay(day_number=d.day_number, theme=d.theme, activities=[
            PlannedActivity(time_of_day=a.time_of_day, attraction=FREE if a.name == FREE_NAME else a.name,
                            plan_b=a.plan_b, note=a.notes)
            for a in d.activities])
        for d in it.daily_plan])


# ---------------------------------------------------------------------------
# Assembly: draft + tool facts -> validated Itinerary
# ---------------------------------------------------------------------------
def assemble(trip: TripRequest, places: dict, research: ResearchResult, weather: list[WeatherDay],
             draft: PlanDraft, transport_mode: Optional[str] = None) -> tuple[Itinerary, dict]:
    atts = research.attractions()
    by_name = {a["name"].casefold(): a for a in atts}
    names = [a["name"] for a in atts]

    def resolve(name):
        if not name or name.strip().upper() == FREE:
            return None
        key = name.strip().casefold()
        if key in by_name:
            return by_name[key]
        close = difflib.get_close_matches(name.strip(), names, n=1, cutoff=0.85)
        return by_name[close[0].casefold()] if close else _INVALID

    def pick_indoor(exclude: set[str]):
        for a in atts:
            if a.get("setting") == "indoor" and a["name"] not in exclude:
                return a["name"]
        return None

    drafts = {d.day_number: d for d in draft.days}
    if set(drafts) != set(range(1, len(weather) + 1)):
        drafts = {i + 1: d for i, d in enumerate(draft.days)}

    used: set[str] = set()
    scheduled: list[dict] = []
    removed = no_plan_b = 0
    daily: list[DayPlan] = []

    for i, wd in enumerate(weather):
        day_draft = drafts.get(i + 1)
        by_slot: dict[str, PlannedActivity] = {}
        for pa in (day_draft.activities if day_draft else []):
            by_slot.setdefault(pa.time_of_day, pa)

        activities: list[Activity] = []
        for slot in SLOTS:
            pa = by_slot.get(slot)
            att = resolve(pa.attraction) if pa else None
            if att is _INVALID:
                removed += 1
                att = None
            if att is not None and att["name"] in used:
                att = None
            if att is None:
                activities.append(Activity(time_of_day=slot, name=FREE_NAME, category=FREE_NAME,
                                           setting="unknown", approx_cost=0,
                                           notes=(pa.note if pa and pa.note else "Rest, explore nearby or travel.")))
                continue

            used.add(att["name"])
            scheduled.append(att)
            plan_b = None
            if att.get("setting") in ("outdoor", "mixed") and wd.status in RISKY:
                pb = resolve(pa.plan_b) if pa.plan_b else None
                if isinstance(pb, dict) and pb.get("setting") == "indoor" and pb["name"] != att["name"]:
                    plan_b = pb["name"]
                else:
                    plan_b = pick_indoor(used)
                if plan_b is None:
                    no_plan_b += 1
            activities.append(Activity(
                time_of_day=slot, name=att["name"], category=att.get("category"),
                setting=att.get("setting", "unknown"),
                approx_cost=float(att.get("est_entry_fee_per_person") or 0) * trip.travellers,
                plan_b=plan_b, notes=pa.note))

        daily.append(DayPlan(day_number=i + 1, date=wd.date, weather_status=wd.status,
                             theme=day_draft.theme if day_draft else "", activities=activities))

    # Transport (from the tool; the user may ask for a specific feasible mode)
    tr = research.latest("compare_transport_modes")
    transport_plan, mode_note = None, None
    if tr:
        options = [TransportOption(**o) for o in (tr.get("all_options") or [tr["recommended"]])]
        chosen, reason = options[0], tr["reason"]
        if transport_mode:
            picked = match_mode(options, transport_mode)
            if picked is None:
                mode_note = f"'{transport_mode}' isn't a feasible option for this route, so {chosen.mode} is kept."
            elif picked.mode != chosen.mode:
                reason = (f"{picked.mode}, as you requested. For your {trip.travel_priority} priority, "
                          f"TripMind's top pick was {chosen.mode}.")
                chosen = picked
        transport_plan = TransportPlan(recommended=chosen, reason=reason,
                                       alternatives=[o for o in options if o.mode != chosen.mode][:2])

    # Budget: recalculated in Python with the scheduled attractions and the chosen transport
    state = trip_state(trip, places)
    b = estimate_budget.invoke({
        "destination": trip.destination,
        "origin_latitude": state["origin"]["latitude"], "origin_longitude": state["origin"]["longitude"],
        "destination_latitude": state["destination"]["latitude"],
        "destination_longitude": state["destination"]["longitude"],
        "start_date": state["start_date"], "end_date": state["end_date"],
        "travellers": trip.travellers, "budget": trip.budget, "travel_priority": trip.travel_priority,
        "origin_country_code": state["origin"]["country_code"],
        "destination_country_code": state["destination"]["country_code"],
        "destination_population": state["destination"]["population"],
        "transport_mode": transport_plan.recommended.mode if transport_plan else None,
        "attraction_fees_per_person": [float(a.get("est_entry_fee_per_person") or 0) for a in scheduled],
    })
    budget = BudgetBreakdown(**b["breakdown"]) if b.get("ok") else None

    # Notes: estimates and limitations, stated clearly
    notes: list[str] = []
    if b.get("ok"):
        notes.append(b["status_message"])
        notes += b.get("suggestions", [])
        notes += [a for a in b.get("assumptions", []) if "level to fit" in a]
    else:
        notes.append(f"Budget: {b.get('error', 'could not be estimated right now.')}")
    if mode_note:
        notes.append(mode_note)
    if tr:
        notes.append(tr["disclaimer"])
        if tr["route"]["is_estimate"]:
            notes.append("Road distance is a straight-line estimate because routing was unavailable.")
    else:
        notes.append(f"Transport: {research.error('compare_transport_modes') or 'comparison unavailable right now.'}")
    unknown = [w for w in weather if w.status == "UNKNOWN"]
    if not research.latest("check_weather_hazards"):
        notes.append("Weather information could not be retrieved right now — outdoor activities have a Plan B.")
    elif unknown:
        notes.append(f"No forecast yet for {', '.join(f'{w.date.day} {w.date:%b}' for w in unknown)} "
                     "(Open-Meteo forecasts about 16 days ahead) — outdoor activities on those days have a Plan B.")
    if atts:
        notes.append("Places come from OpenStreetMap via Geoapify; entry fees are rough estimates.")
    else:
        notes.append(f"Attractions: {research.error('search_attractions') or 'none found nearby'} "
                     "— the plan shows free time instead.")
    if removed:
        notes.append(f"{removed} suggested place(s) were left out because they could not be verified.")
    if no_plan_b:
        notes.append("No indoor alternative was found nearby for some outdoor activities on risky days.")
    notes += [t for t in draft.tips[:3] if t.strip()]

    itinerary = Itinerary(trip_summary=trip, transport_plan=transport_plan, weather=weather,
                          daily_plan=daily, budget_breakdown=budget, notes=notes)
    extras = {
        "intro": draft.intro.strip() or f"Here's your {trip.num_days}-day {trip.destination} plan.",
        "budget_status": b.get("status_message"),
        "per_person": b.get("per_person"),
        "spending_style": b.get("spending_style"),
        "transport_mode": transport_plan.recommended.mode if transport_plan else None,
        "route": tr["route"] if tr else None,
    }
    return itinerary, extras


def build_steps(research: ResearchResult, itinerary: Itinerary) -> list[dict]:
    def st(ok: bool) -> str:
        return "done" if ok else "failed"
    return [
        {"key": "validated", "label": "Trip details validated", "status": "done"},
        {"key": "located", "label": "Cities located", "status": "done"},
        {"key": "weather", "label": "Weather checked", "status": st(research.latest("check_weather_hazards") is not None)},
        {"key": "transport", "label": "Transport compared", "status": st(itinerary.transport_plan is not None)},
        {"key": "attractions", "label": "Attractions found", "status": st(bool(research.attractions()))},
        {"key": "budget", "label": "Budget calculated", "status": st(itinerary.budget_breakdown is not None)},
        {"key": "itinerary", "label": "Itinerary created", "status": "done"},
    ]


def _tools_used(research: ResearchResult) -> list[dict]:
    return [{"name": c.name, "source": c.source, "ok": bool(c.result.get("ok"))} for c in research.calls]


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def plan_trip(trip: TripRequest, places: dict) -> dict:
    research = run_research(trip, places)
    weather = weather_days(research, trip)
    facts = build_facts(trip, research, weather)

    draft, planner_used = None, "fallback"
    if facts["attractions"]:
        draft = draft_plan(facts)
        if draft and draft.days:
            planner_used = "llm"
    if planner_used == "fallback":
        draft = fallback_plan(facts)

    itinerary, extras = assemble(trip, places, research, weather, draft)
    extras.update(planner=planner_used, agent_summary=research.agent_summary, tools_used=_tools_used(research))
    return {"itinerary": itinerary, "extras": extras, "steps": build_steps(research, itinerary),
            "research": research, "weather": weather}


def refine_plan(trip: TripRequest, places: dict, stored: dict, intent: FollowUpIntent) -> Optional[dict]:
    """Change an existing plan using the stored facts (no new weather/transport calls)."""
    old = stored["research"]
    research = ResearchResult(calls=list(old.calls), agent_summary=old.agent_summary)  # don't alter older versions
    weather = stored["weather"]

    if intent.add_interests:  # the only refinement that needs new data
        d = places["destination"]
        args = {"destination": trip.destination, "latitude": d.latitude, "longitude": d.longitude,
                                "interests": intent.add_interests, "num_days": trip.num_days}
        research.calls.append(ToolCallRecord("search_attractions", args, run_tool("search_attractions", args), "refine"))

    if intent.instruction or intent.add_interests:
        facts = build_facts(trip, research, weather)
        facts["current_plan"] = plan_brief(stored["itinerary"])["days"]
        facts["user_request"] = intent.instruction or f"Add places for: {', '.join(intent.add_interests)}."
        draft = draft_plan(facts, REFINE_PROMPT)
        if draft is None or not draft.days:
            return None
    else:  # transport-only change: keep the days exactly as they are
        draft = draft_from_itinerary(stored["itinerary"])

    mode = intent.transport_mode or (stored.get("extras") or {}).get("transport_mode")
    itinerary, extras = assemble(trip, places, research, weather, draft, transport_mode=mode)

    if intent.transport_mode and itinerary.transport_plan:
        if match_mode([itinerary.transport_plan.recommended], intent.transport_mode):
            switched = (f"I've switched your transport to {itinerary.transport_plan.recommended.mode} "
                        "and updated the budget.")
        else:
            switched = f"'{intent.transport_mode}' isn't available for this route, so your transport is unchanged."
        extras["intro"] = f"{draft.intro.strip()} {switched}".strip() if intent.instruction else switched

    extras.update(planner="refine", agent_summary=(stored.get("extras") or {}).get("agent_summary", ""),
                  tools_used=_tools_used(research))
    return {"itinerary": itinerary, "extras": extras, "steps": None, "research": research, "weather": weather}


def itinerary_text(it: Itinerary, extras: dict) -> str:
    """Plain-text itinerary for the terminal demo."""
    t = it.trip_summary
    out = ["", "=" * 64, f"TRIP SUMMARY — {t.origin} → {t.destination}",
           f"{dates_label(t.start_date, t.end_date)} · {t.num_days} days · {t.travellers} travellers · "
           f"{format_inr(t.budget)} · {t.travel_priority} priority",
           f"Interests: {', '.join(t.interests) or 'none'}"]
    if extras.get("intro"):
        out += ["", extras["intro"]]
    if it.transport_plan:
        tp, r = it.transport_plan, it.transport_plan.recommended
        out += ["", "TRANSPORT (estimates)",
                f"  {r.mode}: ~{format_inr(r.estimated_cost)} round trip, ~{hours_label(r.travel_time_hours)} each way",
                f"  {tp.reason}"]
        out += [f"  Alternative: {a.mode} — ~{format_inr(a.estimated_cost)}, ~{hours_label(a.travel_time_hours)}"
                for a in tp.alternatives]
    out += ["", "WEATHER"]
    for w in it.weather:
        temps = (f"{w.temp_min_c:.0f}–{w.temp_max_c:.0f}°C"
                 if w.temp_min_c is not None and w.temp_max_c is not None else "—")
        out.append(f"  {w.date:%d %b}  {temps:<9} {(w.condition or 'No forecast yet'):<22} [{w.status}] {w.reason}")
    out += ["", "DAY-BY-DAY"]
    for d in it.daily_plan:
        out.append(f"  Day {d.day_number} · {d.date:%a %d %b} [{d.weather_status}]"
                   f"{' — ' + d.theme if d.theme else ''}")
        for a in d.activities:
            cost = f" · {format_inr(a.approx_cost)}" if a.approx_cost else ""
            setting = f" ({a.setting})" if a.setting != "unknown" else ""
            out.append(f"    {a.time_of_day.capitalize():<10} {a.name}{setting}{cost}")
            if a.plan_b:
                out.append(f"    {'':<10} Plan B: {a.plan_b}")
    if it.budget_breakdown:
        b = it.budget_breakdown
        rows = [("Transport", b.transport), ("Stay", b.stay), ("Food", b.food),
                ("Local transport", b.local_transport), ("Attractions", b.attractions),
                ("Buffer (10%)", b.buffer), ("TOTAL", b.total)]
        out += ["", "BUDGET (estimates)"] + [f"  {label:<16} {format_inr(v):>10}" for label, v in rows]
    if it.notes:
        out += ["", "NOTES"] + [f"  • {n}" for n in it.notes]
    out.append("=" * 64)
    return "\n".join(out)