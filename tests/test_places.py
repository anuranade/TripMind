import pytest

from src.tools import places
from src.tools.places import NOT_CONFIGURED, SEARCH_ERROR, load_interest_config, search_attractions
from src.utils.config import settings
from tests.test_tools import FakeResponse

CFG = load_interest_config()
HIST = ",".join(CFG["interests"]["history"])
FOOD = ",".join(CFG["interests"]["food"])
DEFAULT = ",".join(CFG["default"])

FORT_CATS = ["building", "building.historic", "heritage", "tourism", "tourism.attraction",
             "tourism.sights", "tourism.sights.fort"]


def feat(name, cats, lat=26.95, lon=75.85, raw=None):
    return {"type": "Feature",
            "properties": {"name": name, "categories": cats, "lat": lat, "lon": lon,
                           "formatted": f"{name}, Jaipur", "datasource": {"raw": raw or {}}},
            "geometry": {"type": "Point", "coordinates": [lon, lat]}}


FEATURES = {
    HIST: [
        feat("Amber Fort", FORT_CATS, raw={"wikidata": "Q1", "wikipedia": "en:Amber Fort"}),
        feat("Albert Hall Museum", ["entertainment", "entertainment.museum", "building.historic"]),
        feat("Old Stepwell", ["tourism", "tourism.sights"]),
        feat("", ["tourism.sights"]),  # unnamed -> skipped
    ],
    FOOD: [
        feat("Laxmi Mishthan Bhandar", ["catering", "catering.restaurant", "catering.restaurant.indian"]),
        feat("Domino's", ["catering", "catering.restaurant"], raw={"brand": "Domino's"}),  # chain -> skipped
    ],
    DEFAULT: [
        feat("Amber Fort", FORT_CATS, raw={"wikidata": "Q1"}),  # duplicate -> deduped
        feat("Hawa Mahal", ["tourism", "tourism.sights", "building.historic"], raw={"wikidata": "Q2"}),
    ],
}


@pytest.fixture
def mock_geoapify(monkeypatch):
    places._fetch_group.cache_clear()
    monkeypatch.setattr(places, "RADIUS_STEPS_KM", (15,))
    monkeypatch.setattr(settings, "geoapify_api_key", "test-key")
    state = {"fail": set(), "calls": []}

    def fake_get(url, params=None, headers=None, timeout=None):
        cats = params["categories"]
        state["calls"].append(cats)
        if cats in state["fail"]:
            return FakeResponse({}, 400)
        return FakeResponse({"type": "FeatureCollection", "features": FEATURES.get(cats, [])})

    monkeypatch.setattr("src.tools.common.requests.get", fake_get)
    yield state
    places._fetch_group.cache_clear()


def run(interests=("history", "food")):
    return search_attractions.invoke({"destination": "Jaipur", "latitude": 26.91,
                                      "longitude": 75.79, "interests": list(interests)})


def test_missing_key(monkeypatch):
    monkeypatch.setattr(settings, "geoapify_api_key", "")
    out = run()
    assert out == {"ok": False, "error": NOT_CONFIGURED}


def test_results_are_tagged_and_filtered(mock_geoapify):
    out = run()
    by_name = {a["name"]: a for a in out["attractions"]}
    assert set(by_name) == {"Amber Fort", "Albert Hall Museum", "Old Stepwell",
                            "Laxmi Mishthan Bhandar", "Hawa Mahal"}
    assert by_name["Amber Fort"]["setting"] == "outdoor"
    assert by_name["Amber Fort"]["category"] == "Fort"
    assert by_name["Amber Fort"]["est_entry_fee_per_person"] == 200
    assert by_name["Amber Fort"]["interest"] == "history"
    assert by_name["Albert Hall Museum"]["setting"] == "indoor"
    assert by_name["Albert Hall Museum"]["est_entry_fee_per_person"] == 100
    assert by_name["Laxmi Mishthan Bhandar"]["setting"] == "indoor"
    assert by_name["Laxmi Mishthan Bhandar"]["category"] == "Restaurant"


def test_notable_places_rank_higher(mock_geoapify):
    names = [a["name"] for a in run()["attractions"]]
    assert names.index("Amber Fort") < names.index("Old Stepwell")


def test_aliases_and_unmatched_interests(mock_geoapify):
    out = run(["heritage", "quantum physics"])
    assert HIST in mock_geoapify["calls"]
    assert out["unmatched_interests"] == ["quantum physics"]


def test_one_group_failing_still_returns_others(mock_geoapify):
    mock_geoapify["fail"] = {FOOD}
    out = run()
    assert out["ok"] and "food" in out["partial"]
    assert "Laxmi Mishthan Bhandar" not in [a["name"] for a in out["attractions"]]


def test_all_groups_failing_is_friendly(mock_geoapify):
    mock_geoapify["fail"] = {HIST, FOOD, DEFAULT}
    assert run() == {"ok": False, "error": SEARCH_ERROR}


def test_results_are_cached(mock_geoapify):
    run()
    first = len(mock_geoapify["calls"])
    run()
    assert len(mock_geoapify["calls"]) == first == 3