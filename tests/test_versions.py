from src.agent.followup import FollowUpIntent
from src.agent.intake import TripIntake
from src.agent.schemas import Itinerary
from src.agent.session import TripMindSession
from tests.test_intake import FULL, FakeExtractor


class QueueRouter:
    def __init__(self, *intents):
        self.intents = list(intents)

    def __call__(self, message, brief, history):
        return self.intents.pop(0)


def planner(trip, places):
    return {"itinerary": Itinerary(trip_summary=trip), "extras": {"intro": "Ready!"}, "steps": []}


def make_session(*intents, responses=None):
    bases = []  # which version each refinement was built on

    def refiner(trip, places, stored, intent):
        bases.append(stored["label"])
        return {"itinerary": Itinerary(trip_summary=trip, notes=[intent.instruction]),
                "extras": {"intro": f"Done: {intent.instruction}"}}

    s = TripMindSession(intake=TripIntake(FakeExtractor(responses or {"full": FULL})), planner=planner,
                        router=QueueRouter(*intents), refiner=refiner, answerer=lambda *a: "Hi")
    s.handle("full")
    first = s.confirm()
    return s, first, bases


def refine(text):
    return FollowUpIntent(intent="refine_plan", instruction=text)


def test_first_plan_is_the_original():
    _, r, _ = make_session()
    assert r["versions"] == [{"number": 1, "label": "Original plan", "active": True}]


def test_refinement_keeps_the_original_intact():
    s, _, _ = make_session(refine("Make day 2 relaxed"))
    original = s.versions[0]["itinerary"]
    r = s.handle("make day 2 relaxed")
    assert len(s.versions) == 2 and s.versions[0]["itinerary"] is original
    assert [v["label"] for v in r["versions"]] == ["Original plan", "Make day 2 relaxed"]
    assert r["versions"][1]["active"] and r["itinerary"]["notes"] == ["Make day 2 relaxed"]


def test_switch_back_and_build_on_the_original():
    s, _, bases = make_session(refine("A"), refine("B"))
    s.handle("a")
    r = s.select_version(1)
    assert r["stage"] == "planned" and r["versions"][0]["active"]
    assert r["itinerary"]["notes"] == []
    s.handle("b")
    assert bases == ["Original plan", "Original plan"]
    assert len(s.versions) == 3


def test_trip_change_adds_a_version_and_switching_restores_trip_details():
    s, _, _ = make_session(FollowUpIntent(intent="change_trip"),
                           responses={"full": FULL, "make it 3 people": {"travellers": 3}})
    s.handle("make it 3 people")
    r = s.handle("yes")
    assert [v["label"] for v in r["versions"]] == ["Original plan", "Updated: 3 travellers"]
    s.select_version(1)
    assert s.intake.trip.travellers == 2
    s.select_version(2)
    assert s.intake.trip.travellers == 3


def test_invalid_version_is_friendly():
    s, _, _ = make_session()
    r = s.select_version(5)
    assert r["errors"] and "no plan version 5" in r["text"]


def test_new_trip_clears_versions():
    s, _, _ = make_session(FollowUpIntent(intent="new_trip"),
                           responses={"full": FULL, "goa": {"destination": "Goa"}})
    s.handle("goa")
    assert s.versions == [] and s.plan is None