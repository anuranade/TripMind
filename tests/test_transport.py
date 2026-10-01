import pytest
import requests

from src.tools import transport
from src.tools.transport import (DISCLAIMER, RouteInfo, compare_transport_modes, get_route,
                                 haversine_km)
from src.utils.config import settings
from tests.test_tools import FakeResponse

LONG = RouteInfo(distance_km=1150, duration_hours=20, straight_line_km=920, source="test", is_estimate=False)
SHORT = RouteInfo(distance_km=65, duration_hours=1.3, straight_line_km=55, source="test", is_estimate=False)


def args(n=2, p="budget"):
    return {"origin_latitude": 19.076, "origin_longitude": 72.878,
            "destination_latitude": 26.912, "destination_longitude": 75.787,
            "travellers": n, "travel_priority": p}


@pytest.fixture
def fixed_route(monkeypatch):
    holder = {"route": LONG}
    monkeypatch.setattr(transport, "get_route", lambda *a, **k: holder["route"])
    return holder


def test_haversine_mumbai_jaipur():
    assert 880 < haversine_km(19.076, 72.878, 26.912, 75.787) < 960


def top(p, n=2):
    return compare_transport_modes.invoke(args(n, p))["recommended"]["mode"]


def test_long_route_ranking_changes_with_priority(fixed_route):
    assert top("budget") == "Train (3AC)"
    assert top("time") == "Flight"


def test_short_route_ranking_changes_with_priority(fixed_route):
    fixed_route["route"] = SHORT
    assert top("budget") == "Bus (AC sleeper)"
    assert top("time") == "Outstation cab"
    assert top("convenience") == "Outstation cab"


def test_flight_not_offered_for_short_trips(fixed_route):
    fixed_route["route"] = SHORT
    out = compare_transport_modes.invoke(args())
    assert all(o["mode"] != "Flight" for o in out["all_options"])


def test_train_cost_is_round_trip_for_group(fixed_route):
    out = compare_transport_modes.invoke(args(2, "budget"))
    train = next(o for o in out["all_options"] if o["mode"] == "Train (3AC)")
    assert train["estimated_cost"] == 7500  # 2 people x 2 ways x (150 + 1.5 x 1150)


def test_cab_uses_enough_vehicles(fixed_route):
    fixed_route["route"] = SHORT
    out = compare_transport_modes.invoke(args(5, "convenience"))
    cab = next(o for o in out["all_options"] if o["mode"] == "Outstation cab")
    assert cab["estimated_cost"] == 5640  # 2 cars x 2 ways x (500 + 14 x 65)
    assert "2 cars" in cab["note"]


def test_output_is_labelled_as_estimate(fixed_route):
    out = compare_transport_modes.invoke(args())
    assert out["disclaimer"] == DISCLAIMER
    assert len(out["alternatives"]) <= 2
    assert all(o["is_estimate"] for o in out["all_options"])
    assert "priority" in out["reason"]


def test_invalid_priority(fixed_route):
    out = compare_transport_modes.invoke(args(2, "luxury"))
    assert out["ok"] is False


def test_no_feasible_mode(fixed_route):
    fixed_route["route"] = RouteInfo(distance_km=20000, duration_hours=300, straight_line_km=16000,
                                     source="test", is_estimate=False)
    out = compare_transport_modes.invoke(args())
    assert out["ok"] is False and "No suitable" in out["error"]


# ---------- Routing providers (mocked HTTP) ----------
@pytest.fixture
def mock_http(monkeypatch):
    transport._route_cached.cache_clear()
    monkeypatch.setattr(settings, "ors_api_key", "")
    state = {"payload": {"code": "Ok", "routes": [{"distance": 1_150_000, "duration": 72_000}]},
             "exc": None, "urls": []}

    def fake_get(url, params=None, headers=None, timeout=None):
        state["urls"].append(url)
        if state["exc"]:
            raise state["exc"]
        return FakeResponse(state["payload"])

    monkeypatch.setattr("src.tools.common.requests.get", fake_get)
    yield state
    transport._route_cached.cache_clear()


def test_osrm_route(mock_http):
    r = get_route(19.076, 72.878, 26.912, 75.787)
    assert (r.distance_km, r.duration_hours, r.is_estimate) == (1150.0, 20.0, False)
    assert "OSRM" in r.source


def test_route_falls_back_to_estimate(mock_http):
    mock_http["exc"] = requests.ConnectionError()
    r = get_route(19.076, 72.878, 26.912, 75.787)
    assert r.is_estimate and "estimate" in r.source
    assert r.distance_km == pytest.approx(r.straight_line_km * 1.3, rel=0.01)


def test_no_road_route_falls_back(mock_http):
    mock_http["payload"] = {"code": "NoRoute", "routes": []}
    assert get_route(1, 1, 2, 2).is_estimate


def test_ors_used_first_when_configured(mock_http, monkeypatch):
    monkeypatch.setattr(settings, "ors_api_key", "real-looking-key")
    get_route(19.076, 72.878, 26.912, 75.787)  # ORS payload shape is wrong here -> falls to OSRM
    assert "openrouteservice" in mock_http["urls"][0]
    assert "project-osrm" in mock_http["urls"][1]