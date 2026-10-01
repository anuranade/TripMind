import pytest

from src.agent.followup import FollowUpIntent, is_negative
from src.agent.intake import TripIntake
from src.agent.planner import assemble, draft_from_itinerary, refine_plan, weather_days
from src.agent.schemas import Itinerary
from src.agent.session import KEPT_MSG, REFINE_ERROR, TripMindSession
from tests.test_intake import FULL, FakeExtractor
from tests.test_planner import GOOD, PLACES, TRIP, draft, research
from tests.test_transport import LONG

BUS = {"mode": "Bus (AC sleeper)", "estimated_cost": 6840, "travel_time_hours": 23.5,
       "convenience_score": 5, "score": 70, "is_estimate": True, "note": ""}


@pytest.fixture(autouse=True)
def fixed_route(monkeypatch):
    monkeypatch.setattr("src.tools.budget.get_route", lambda *a, **k: LONG)


def stored_plan():
    r = research()
    tr = r.calls[1].result
    tr["all_options"] = [tr["recommended"], BUS]
    w = weather_days(r, TRIP)
    it, extras = assemble(TRIP, PLACES, r, w, draft(GOOD))
    return {"itinerary": it, "extras": extras, "research": r, "weather": w}


def names(it):
    return [[a.name for a in d.activities] for d in it.daily_plan]


# ---------- Planner-level refinement (no LLM) ----------
def test_user_can_choose_a_feasible_mode():
    s = stored_plan()
    it, extras = assemble(TRIP, PLACES, s["research"], s["weather"], draft(GOOD), transport_mode="bus")
    assert it.transport_plan.recommended.mode == "Bus (AC sleeper)"
    assert "as you requested" in it.transport_plan.reason
    assert it.budget_breakdown.transport == 6840
    assert extras["transport_mode"] == "Bus (AC sleeper)"


def test_infeasible_mode_keeps_recommendation():
    s = stored_plan()
    it, _ = assemble(TRIP, PLACES, s["research"], s["weather"], draft(GOOD), transport_mode="helicopter")
    assert it.transport_plan.recommended.mode == "Train (3AC)"
    assert any("isn't a feasible option" in n for n in it.notes)


def test_draft_round_trip_keeps_days():
    s = stored_plan()
    it, _ = assemble(TRIP, PLACES, s["research"], s["weather"], draft_from_itinerary(s["itinerary"]))
    assert names(it) == names(s["itinerary"])


def test_transport_only_refine_skips_llm(monkeypatch):
    monkeypatch.setattr("src.agent.planner.draft_plan", lambda *a, **k: pytest.fail("LLM should not run"))
    s = stored_plan()
    out = refine_plan(TRIP, PLACES, s, FollowUpIntent(intent="refine_plan", transport_mode="bus"))
    assert out["itinerary"].transport_plan.recommended.mode == "Bus (AC sleeper)"
    assert "switched" in out["extras"]["intro"]
    assert names(out["itinerary"]) == names(s["itinerary"])


def test_refine_with_instruction_uses_planner(monkeypatch):
    seen = {}
    relaxed = draft([GOOD[0], [("morning", "Hawa Mahal", None), ("afternoon", "FREE", None),
                               ("evening", "FREE", None)]])
    relaxed.intro = "Day 2 is now relaxed."

    def fake_draft_plan(facts, prompt=None):
        seen["facts"] = facts
        return relaxed

    monkeypatch.setattr("src.agent.planner.draft_plan", fake_draft_plan)
    out = refine_plan(TRIP, PLACES, stored_plan(),
                      FollowUpIntent(intent="refine_plan", instruction="Make day 2 more relaxed"))
    assert seen["facts"]["user_request"] == "Make day 2 more relaxed"
    assert "current_plan" in seen["facts"]
    assert [a.name for a in out["itinerary"].daily_plan[1].activities].count("Free time") == 2
    assert out["extras"]["intro"] == "Day 2 is now relaxed."


# ---------- Session follow-ups (fake router / refiner / answerer) ----------
class Router:
    def __init__(self, intent):
        self.intent = intent

    def __call__(self, message, brief, history):
        if isinstance(self.intent, Exception):
            raise self.intent
        return self.intent


def make_session(intent, refiner=None, responses=None):
    calls = {"planner": 0}

    def planner(trip, places):
        calls["planner"] += 1
        return {"itinerary": Itinerary(trip_summary=trip), "extras": {"intro": "Ready!"}, "steps": []}

    s = TripMindSession(intake=TripIntake(FakeExtractor(responses or {"full": FULL})), planner=planner,
                        router=Router(intent), refiner=refiner,
                        answerer=lambda m, b, h: "Hi! Ask me for any changes.")
    s.handle("full")
    s.confirm()
    return s, calls


def test_hello_after_plan_gets_an_answer():
    s, _ = make_session(FollowUpIntent(intent="question"))
    r = s.handle("hello")
    assert r["stage"] == "planned" and r["text"].startswith("Hi!")
    assert r["itinerary"] is None


def test_refine_updates_the_plan():
    def refiner(trip, places, stored, intent):
        return {"itinerary": Itinerary(trip_summary=trip), "extras": {"intro": "Day 2 is now relaxed."}}

    s, _ = make_session(FollowUpIntent(intent="refine_plan", instruction="Make day 2 relaxed"), refiner)
    r = s.handle("make day 2 more relaxed")
    assert r["text"] == "Day 2 is now relaxed." and r["itinerary"] is not None


def test_refine_failure_keeps_the_plan():
    s, _ = make_session(FollowUpIntent(intent="refine_plan", instruction="x"), lambda *a: None)
    before = s.plan
    r = s.handle("change something")
    assert r["text"] == REFINE_ERROR and s.plan is before


def test_change_trip_reconfirms_then_replans():
    s, calls = make_session(FollowUpIntent(intent="change_trip"),
                            responses={"full": FULL, "make it 3 people": {"travellers": 3}})
    r = s.handle("make it 3 people")
    assert r["stage"] == "awaiting_confirmation" and r["summary"]["travellers"] == 3
    assert r["text"].startswith("Here's your updated trip")
    assert s.handle("yes")["stage"] == "planned"
    assert calls["planner"] == 2


def test_pending_change_can_be_cancelled():
    s, calls = make_session(FollowUpIntent(intent="change_trip"),
                            responses={"full": FULL, "make it 3 people": {"travellers": 3}})
    s.handle("make it 3 people")
    r = s.handle("keep the current plan")
    assert r["text"] == KEPT_MSG and s.intake.trip.travellers == 2
    assert calls["planner"] == 1


def test_new_trip_clears_results():
    s, _ = make_session(FollowUpIntent(intent="new_trip"),
                        responses={"full": FULL, "Plan a trip to Goa": {"destination": "Goa"}})
    r = s.handle("Plan a trip to Goa")
    assert r["clear_results"] and r["stage"] == "collecting"
    assert s.plan is None and s.intake.draft.destination == "Goa"


def test_router_failure_is_friendly():
    s, _ = make_session(RuntimeError("boom"))
    r = s.handle("hello")
    assert r["stage"] == "planned" and "couldn't process" in r["text"]


@pytest.mark.parametrize("text, expected", [
    ("no", True), ("Keep the current plan", True), ("never mind!", True),
    ("no, make it 4 days", False), ("make day 2 relaxed", False),
])
def test_is_negative(text, expected):
    assert is_negative(text) is expected