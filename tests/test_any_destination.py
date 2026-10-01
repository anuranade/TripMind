import pytest

from src.agent.agent import ensure_coverage, trip_state
from src.agent.schemas import GeoPlace
from src.tools import places, transport
from src.tools.budget import calculate_budget, cost_profile
from src.tools.places import resolve_interests, search_attractions
from src.tools.transport import compare, compare_transport_modes, get_route
from src.utils.config import settings
from tests.test_planner import PLACES, TRIP, research
from tests.test_tools import FakeResponse
from tests.test_transport import LONG


# ---------- Costs for any destination ----------
@pytest.mark.parametrize("dest, cc, pop, expected", [
    ("Jaipur", "IN", 3_000_000, "Jaipur"),          # listed city wins
    ("Somepur", "IN", 5_000_000, "_metro"),
    ("Somepur", "IN", 500_000, "_city"),
    ("Somepur", "IN", 40_000, "_town"),
    ("Somepur", "IN", None, "_town"),
    ("Paris", "FR", 2_100_000, "_intl_premium"),
    ("Bangkok", "TH", 8_000_000, "_intl_mid"),
    ("Kathmandu", "NP", 1_000_000, "_intl_budget"),
    ("Atlantis", "ZZ", None, "_intl_mid"),          # unlisted country -> mid
    ("Somepur", None, None, "_default"),            # nothing known
])
def test_cost_profile(dest, cc, pop, expected):
    assert cost_profile(dest, cc, pop)[0]["city"] == expected


def total(cc, pop):
    b, _ = calculate_budget(destination="Somewhere", num_days=3, travellers=2, budget=1, style="mid",
                            transport_cost=0, transport_label="x", country_code=cc, population=pop)
    return b.total


def test_costs_scale_with_place():
    assert total("FR", 2_000_000) > total("IN", 5_000_000) > total("IN", 50_000)


def test_international_note():
    _, notes = calculate_budget(destination="Paris", num_days=3, travellers=2, budget=200000, style="mid",
                                transport_cost=60000, transport_label="Flight", country_code="FR")
    assert any("outside India" in n for n in notes)


# ---------- Realistic transport ----------
def modes(options):
    return {o.mode for o in options}


def test_international_trip_has_no_train():
    found = modes(compare(LONG, 2, "budget", is_same_country=False))
    assert "Flight" in found and "Train (3AC)" not in found


def test_no_road_means_flight_only():
    island = LONG.model_copy(update={"no_road": True, "is_estimate": True})
    assert modes(compare(island, 2, "budget")) == {"Flight"}


def test_routing_outage_keeps_ground_modes_for_domestic_trips():
    outage = LONG.model_copy(update={"is_estimate": True})
    assert "Train (3AC)" in modes(compare(outage, 2, "budget", is_same_country=True))


def test_tool_explains_flight_only(monkeypatch):
    island = LONG.model_copy(update={"no_road": True, "is_estimate": True})
    monkeypatch.setattr(transport, "get_route", lambda *a, **k: island)
    out = compare_transport_modes.invoke({
        "origin_latitude": 13.08, "origin_longitude": 80.27, "destination_latitude": 11.62,
        "destination_longitude": 92.73, "travellers": 2, "travel_priority": "budget",
        "origin_country_code": "IN", "destination_country_code": "IN"})
    assert out["recommended"]["mode"] == "Flight" and "no road route" in out["note"]


def test_osrm_no_route_sets_flag(monkeypatch):
    transport._route_cached.cache_clear()
    monkeypatch.setattr(settings, "ors_api_key", "")
    monkeypatch.setattr("src.tools.common.requests.get",
                        lambda *a, **k: FakeResponse({"code": "NoRoute", "routes": []}))
    r = get_route(13.08, 80.27, 11.62, 92.73)
    assert r.no_road and r.is_estimate
    transport._route_cached.cache_clear()


# ---------- Free-form interests ----------
def test_free_form_interests_are_understood():
    groups, unmatched = resolve_interests(["Trekking", "waterfalls", "cafes", "temple"])
    assert {"nature", "food", "spirituality"} <= set(groups)
    assert unmatched == []


# ---------- Adaptive attraction search ----------
@pytest.fixture
def geo_mock(monkeypatch):
    places._fetch_group.cache_clear()
    monkeypatch.setattr(settings, "geoapify_api_key", "test-key")

    def fake_get(url, params=None, headers=None, timeout=None):
        radius_m = int(params["filter"].rsplit(",", 1)[1])
        n = 1 if radius_m <= 15000 else 6
        tag = params["categories"][:12]
        feats = [{"properties": {"name": f"{tag} place {i}", "categories": ["tourism.sights"],
                                 "lat": 10.0, "lon": 77.0, "datasource": {"raw": {}}}} for i in range(n)]
        return FakeResponse({"features": feats})

    monkeypatch.setattr("src.tools.common.requests.get", fake_get)
    yield
    places._fetch_group.cache_clear()


def test_search_widens_for_small_towns(geo_mock):
    out = search_attractions.invoke({"destination": "Munnar", "latitude": 10.09, "longitude": 77.06,
                                     "interests": ["nature"], "num_days": 3})
    assert out["radius_km"] == 35 and out["count"] >= 9


# ---------- Safety net for international trips ----------
def test_safety_net_redoes_transport_with_country_codes(monkeypatch):
    called = []
    monkeypatch.setattr("src.agent.agent.run_tool", lambda name, args: called.append((name, args)) or {"ok": True})
    paris = {"origin": PLACES["origin"],
             "destination": GeoPlace(name="Paris", latitude=48.85, longitude=2.35,
                                     country="France", country_code="FR")}
    ensure_coverage(research(), trip_state(TRIP, paris))
    assert [n for n, _ in called] == ["compare_transport_modes"]
    assert called[0][1]["destination_country_code"] == "FR"