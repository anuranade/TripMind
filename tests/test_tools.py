import os

import pytest
import requests

from src.agent.intake import TripIntake
from src.tools import geo
from src.tools.common import ServiceUnavailableError
from src.tools.geo import PlaceNotFoundError, geocode, geocode_place, place_label
from tests.test_intake import FULL, FakeExtractor


# ---------------------------------------------------------------------------
# Fake HTTP layer
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self.payload


@pytest.fixture
def mock_http(monkeypatch):
    geo._search.cache_clear()
    state = {"payload": {}, "status": 200, "exc": None, "calls": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        state["calls"] += 1
        if state["exc"]:
            raise state["exc"]
        return FakeResponse(state["payload"], state["status"])

    monkeypatch.setattr("src.tools.common.requests.get", fake_get)
    yield state
    geo._search.cache_clear()


def r(name, cc, pop=None, admin1=None, country="India", lat=10.0, lon=20.0):
    return {"name": name, "latitude": lat, "longitude": lon, "country": country,
            "country_code": cc, "admin1": admin1, "population": pop}


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------
def test_geocode_returns_place(mock_http):
    mock_http["payload"] = {"results": [r("Jaipur", "IN", 3_000_000, "Rajasthan", lat=26.92, lon=75.79)]}
    p = geocode("jaipur")
    assert (p.name, p.country_code, p.latitude) == ("Jaipur", "IN", 26.92)
    assert place_label(p) == "Jaipur, Rajasthan, India"


def test_unknown_place_raises_and_tool_returns_error(mock_http):
    mock_http["payload"] = {}  # Open-Meteo omits "results" when nothing matches
    with pytest.raises(PlaceNotFoundError):
        geocode("Xyzabad")
    out = geocode_place.invoke({"place": "Xyzabad"})
    assert out["ok"] is False and "couldn't find" in out["error"]


def test_prefers_reasonably_large_indian_match(mock_http):
    mock_http["payload"] = {"results": [r("Hyderabad", "PK", 1_700_000, country="Pakistan"),
                                        r("Hyderabad", "IN", 6_800_000, "Telangana")]}
    assert geocode("Hyderabad").country_code == "IN"


def test_does_not_prefer_tiny_indian_village(mock_http):
    mock_http["payload"] = {"results": [r("Paris", "FR", 2_100_000, country="France"),
                                        r("Paris", "IN", 900)]}
    assert geocode("Paris").country_code == "FR"


def test_hint_after_comma_filters(mock_http):
    mock_http["payload"] = {"results": [r("Aurangabad", "IN", 1_200_000, "Maharashtra"),
                                        r("Aurangabad", "IN", 100_000, "Bihar")]}
    assert geocode("Aurangabad, Bihar").admin1 == "Bihar"


def test_short_input_rejected_without_calling_api(mock_http):
    with pytest.raises(PlaceNotFoundError):
        geocode(" ")
    assert mock_http["calls"] == 0


def test_results_are_cached(mock_http):
    mock_http["payload"] = {"results": [r("Jaipur", "IN", 3_000_000)]}
    geocode("Jaipur")
    geocode("jaipur")
    assert mock_http["calls"] == 1


def test_timeout_is_friendly(mock_http):
    mock_http["exc"] = requests.Timeout()
    with pytest.raises(ServiceUnavailableError) as e:
        geocode("Jaipur")
    assert "too long" in e.value.user_message
    assert geocode_place.invoke({"place": "Jaipur"})["ok"] is False


def test_http_error_is_friendly(mock_http):
    mock_http["status"] = 500
    with pytest.raises(ServiceUnavailableError) as e:
        geocode("Jaipur")
    assert "unavailable" in e.value.user_message


# ---------------------------------------------------------------------------
# Intake integration (uses the fake geocoder from conftest.py)
# ---------------------------------------------------------------------------
def test_intake_invalid_city_asks_for_correction():
    fx = FakeExtractor({"bad": {**FULL, "destination": "Xyzabad"}, "fix": {"destination": "Jaipur"}})
    intake = TripIntake(fx)
    reply = intake.handle("bad")
    assert reply.stage == "collecting"
    assert "Xyzabad" in reply.text and "correct destination" in reply.text
    assert intake.draft.destination is None and intake.draft.origin == "Mumbai"

    reply = intake.handle("fix")
    assert reply.stage == "awaiting_confirmation"
    assert "India" in reply.summary["destination_label"]
    assert reply.summary["destination_coords"] == [20.0, 75.0]


def test_intake_location_service_down_stays_alive():
    intake = TripIntake(FakeExtractor({"down": {**FULL, "origin": "Offline"}}))
    reply = intake.handle("down")
    assert reply.stage == "collecting"
    assert "unavailable" in reply.text
    assert intake.draft.origin == "Offline"  # kept, so a retry works


# ---------------------------------------------------------------------------
# Live test (real Open-Meteo; opt-in)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="Set RUN_LIVE_TESTS=1 to call Open-Meteo")
def test_live_geocoding():
    geo._search.cache_clear()
    p = geocode("Jaipur")
    assert p.country_code == "IN" and 26 < p.latitude < 27.5
    with pytest.raises(PlaceNotFoundError):
        geocode("Xyzabadqwerty")