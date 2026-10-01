import pytest
from fastapi.testclient import TestClient

import app as app_module
from src.agent.intake import TripIntake
from src.agent.schemas import Itinerary
from src.agent.session import TripMindSession
from tests.test_intake import FULL, FakeExtractor

client = TestClient(app_module.app)
SID = "test-session-123"


def fake_session():
    return TripMindSession(
        intake=TripIntake(FakeExtractor({"full": FULL})),
        planner=lambda trip, places: {"itinerary": Itinerary(trip_summary=trip),
                                      "extras": {"intro": "Ready!"}, "steps": []},
    )


@pytest.fixture(autouse=True)
def fake_sessions(monkeypatch):
    monkeypatch.setattr(app_module, "new_session", fake_session)
    app_module.SESSIONS.clear()


def test_health_ok():
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and "services" in body


def test_index_loads():
    res = client.get("/")
    assert res.status_code == 200 and "Where are you headed?" in res.text


def test_chat_confirm_flow():
    r = client.post("/api/chat", json={"session_id": SID, "message": "full"}).json()
    assert r["stage"] == "awaiting_confirmation" and r["summary"]["destination"] == "Jaipur"
    r = client.post("/api/confirm", json={"session_id": SID}).json()
    assert r["stage"] == "planned" and r["itinerary"]["trip_summary"]["origin"] == "Mumbai"


def test_reset_starts_fresh():
    client.post("/api/chat", json={"session_id": SID, "message": "full"})
    client.post("/api/reset", json={"session_id": SID})
    r = client.post("/api/chat", json={"session_id": SID, "message": "hello"}).json()
    assert r["stage"] == "collecting"


def test_invalid_request_is_rejected():
    assert client.post("/api/chat", json={"session_id": "x", "message": ""}).status_code == 422