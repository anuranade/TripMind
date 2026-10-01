from datetime import date, timedelta

import pytest

from src.agent.agent import ResearchResult, ToolCallRecord, ensure_coverage, trip_state
from src.agent.intake import TripIntake
from src.agent.planner import assemble, build_facts, fallback_plan, weather_days
from src.agent.schemas import (GeoPlace, Itinerary, PlanDraft, PlannedActivity, PlannedDay,
                               TripRequest)
from src.agent.session import TripMindSession
from tests.test_intake import FULL, FakeExtractor
from tests.test_transport import LONG

S = date.today() + timedelta(days=5)
TRIP = TripRequest(origin="Mumbai", destination="Jaipur", start_date=S, end_date=S + timedelta(days=1),
                   travellers=2, budget=25000, travel_priority="budget", interests=["history", "food"])
PLACES = {"origin": GeoPlace(name="Mumbai", latitude=19.076, longitude=72.878, country="India", country_code="IN"),
          "destination": GeoPlace(name="Jaipur", latitude=26.912, longitude=75.787, country="India", country_code="IN")}


def att(name, setting, fee=100, interest="history", score=5):
    return {"name": name, "category": "Sight", "setting": setting, "interest": interest,
            "est_entry_fee_per_person": fee, "distance_km": 3.0, "notable": False, "score": score,
            "latitude": 26.9, "longitude": 75.8}


ATTRACTIONS = [att("Amber Fort", "outdoor", 200, score=9), att("Albert Hall Museum", "indoor", 100, score=8),
               att("Hawa Mahal", "mixed", 50, score=7), att("City Palace", "mixed", 200, score=6),
               att("Anokhi Museum", "indoor", 80, score=5), att("Chokhi Dhani", "outdoor", 0, "food", 4)]


def wx(d, status):
    return {"date": d.isoformat(), "forecast_available": status != "UNKNOWN", "temp_min_c": 24,
            "temp_max_c": 32, "precipitation_mm": 0, "precipitation_probability": 10, "weather_code": 1,
            "condition": "Clear", "wind_gusts_kmh": 10, "status": status, "reason": "test"}


def research(statuses=("SAFE", "CAUTION"), with_weather=True):
    r = ResearchResult()
    if with_weather:
        r.calls.append(ToolCallRecord("check_weather_hazards", {}, {
            "ok": True, "days": [wx(S + timedelta(days=i), s) for i, s in enumerate(statuses)]}, "agent"))
    r.calls.append(ToolCallRecord("compare_transport_modes", {}, {
        "ok": True, "priority": "budget", "route": LONG.model_dump(), "reason": "Train is best.",
        "recommended": {"mode": "Train (3AC)", "estimated_cost": 7500, "travel_time_hours": 20.2,
                        "convenience_score": 7, "score": 75, "is_estimate": True, "note": ""},
        "alternatives": [], "all_options": [], "disclaimer": "Estimates."}, "agent"))
    r.calls.append(ToolCallRecord("search_attractions", {}, {"ok": True, "attractions": ATTRACTIONS}, "agent"))
    return r


@pytest.fixture(autouse=True)
def fixed_route(monkeypatch):
    monkeypatch.setattr("src.tools.budget.get_route", lambda *a, **k: LONG)


def draft(days):
    return PlanDraft(intro="Enjoy!", tips=[], days=[
        PlannedDay(day_number=i + 1, activities=[
            PlannedActivity(time_of_day=t, attraction=n, plan_b=pb) for t, n, pb in acts])
        for i, acts in enumerate(days)])


GOOD = [
    [("morning", "Amber Fort", None), ("afternoon", "Albert Hall Museum", None), ("evening", "Chokhi Dhani", None)],
    [("morning", "Hawa Mahal", None), ("afternoon", "City Palace", None), ("evening", "FREE", None)],
]


def build(days=GOOD, **kw):
    r = research(**kw)
    return assemble(TRIP, PLACES, r, weather_days(r, TRIP), draft(days))


def test_facts_come_from_tools_not_llm():
    it, extras = build()
    amber = it.daily_plan[0].activities[0]
    assert (amber.name, amber.setting, amber.approx_cost) == ("Amber Fort", "outdoor", 400)
    assert amber.plan_b is None  # SAFE day
    assert it.transport_plan.recommended.mode == "Train (3AC)"
    assert extras["intro"] == "Enjoy!"


def test_risky_day_gets_indoor_plan_b():
    it, _ = build()
    hawa, palace, evening = it.daily_plan[1].activities
    assert it.daily_plan[1].weather_status == "CAUTION"
    assert hawa.plan_b == "Anokhi Museum" and palace.plan_b == "Anokhi Museum"
    assert evening.name == "Free time"


def test_invented_place_is_removed():
    days = [[("morning", "Taj Mahal", None)] + GOOD[0][1:], GOOD[1]]
    it, _ = build(days)
    assert it.daily_plan[0].activities[0].name == "Free time"
    assert any("could not be verified" in n for n in it.notes)


def test_repeated_place_is_dropped():
    days = [GOOD[0], [("morning", "Amber Fort", None)] + GOOD[1][1:]]
    it, _ = build(days)
    assert it.daily_plan[1].activities[0].name == "Free time"


def test_outdoor_plan_b_from_llm_is_replaced():
    days = [GOOD[0], [("morning", "Hawa Mahal", "Amber Fort")] + GOOD[1][1:]]
    it, _ = build(days)
    assert it.daily_plan[1].activities[0].plan_b == "Anokhi Museum"


def test_budget_uses_scheduled_fees():
    it, _ = build()
    assert it.budget_breakdown.attractions == (200 + 100 + 0 + 50 + 200) * 2


def test_missing_weather_means_plan_b_everywhere():
    it, _ = build(with_weather=False)
    assert all(d.weather_status == "UNKNOWN" for d in it.daily_plan)
    assert it.daily_plan[0].activities[0].plan_b is not None
    assert any("could not be retrieved" in n for n in it.notes)


def test_fallback_plan_is_indoor_on_hazard_days():
    r = research(("HAZARD", "SAFE"))
    facts = build_facts(TRIP, r, weather_days(r, TRIP))
    day1 = fallback_plan(facts).days[0].activities
    indoor = {a["name"] for a in ATTRACTIONS if a["setting"] == "indoor"}
    assert day1[0].attraction in indoor and day1[1].attraction in indoor


def test_safety_net_calls_missing_tools(monkeypatch):
    called = []

    def fake_run_tool(name, args):
        called.append(name)
        return {"ok": True, "days": [], "priority": "budget", "attractions": []}

    monkeypatch.setattr("src.agent.agent.run_tool", fake_run_tool)
    r = ResearchResult()
    ensure_coverage(r, trip_state(TRIP, PLACES))
    assert set(called) == {"check_weather_hazards", "compare_transport_modes", "search_attractions"}
    assert all(c.source == "auto" for c in r.calls)


def test_safety_net_trusts_complete_agent_research(monkeypatch):
    monkeypatch.setattr("src.agent.agent.run_tool", lambda *a: pytest.fail("should not be called"))
    ensure_coverage(research(), trip_state(TRIP, PLACES))


# ---------- Session flow (fake planner, no LLM) ----------
def ok_planner(trip, places):
    return {"itinerary": Itinerary(trip_summary=trip), "extras": {"intro": "Ready!"}, "steps": []}


def test_session_confirm_then_plan():
    s = TripMindSession(intake=TripIntake(FakeExtractor({"full": FULL})), planner=ok_planner)
    assert s.handle("full")["stage"] == "awaiting_confirmation"
    r = s.handle("yes")
    assert r["stage"] == "planned"
    assert r["itinerary"]["trip_summary"]["destination"] == "Jaipur"
    


def test_session_planning_failure_can_retry():
    state = {"fail": True}

    def planner(trip, places):
        if state["fail"]:
            raise RuntimeError("boom")
        return ok_planner(trip, places)

    s = TripMindSession(intake=TripIntake(FakeExtractor({"full": FULL})), planner=planner)
    s.handle("full")
    r = s.confirm()
    assert r["stage"] == "confirmed" and r["errors"]
    state["fail"] = False
    assert s.confirm()["stage"] == "planned"    