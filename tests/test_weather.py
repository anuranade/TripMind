from datetime import date, timedelta

import pytest
import requests

from src.agent.schemas import WeatherDay
from src.tools import weather
from src.tools.weather import (WEATHER_ERROR, assess_day, check_weather_hazards,
                               fetch_forecast, get_weather_forecast, worst_status)
from tests.test_tools import FakeResponse

T = date.today()


def day(**kw) -> WeatherDay:
    base = dict(date=T, temp_min_c=22, temp_max_c=30, precipitation_mm=0,
                precipitation_probability=10, weather_code=1, wind_gusts_kmh=15)
    base.update(kw)
    return WeatherDay(**base)


# ---------------------------------------------------------------------------
# Hazard rules (pure Python)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kw, status, keyword", [
    ({}, "SAFE", "No significant"),
    ({"precipitation_mm": 10}, "CAUTION", "Heavy rain"),
    ({"precipitation_mm": 30}, "CAUTION", "Heavy rain"),
    ({"precipitation_mm": 35}, "HAZARD", "Very heavy rain"),
    ({"precipitation_probability": 80, "precipitation_mm": 2}, "CAUTION", "probability"),
    ({"weather_code": 95}, "HAZARD", "thunderstorm"),
    ({"weather_code": 45}, "CAUTION", "Fog"),
    ({"temp_max_c": 38}, "CAUTION", "High heat"),
    ({"temp_max_c": 42}, "CAUTION", "High heat"),
    ({"temp_max_c": 43}, "HAZARD", "Extreme heat"),
    ({"wind_gusts_kmh": 40}, "CAUTION", "Strong wind"),
    ({"wind_gusts_kmh": 60}, "CAUTION", "Strong wind"),
    ({"wind_gusts_kmh": 65}, "HAZARD", "Dangerous wind"),
])
def test_hazard_rules(kw, status, keyword):
    d = assess_day(day(**kw))
    assert d.status == status
    assert keyword.lower() in d.reason.lower()


def test_multiple_reasons_worst_wins():
    d = assess_day(day(precipitation_mm=15, wind_gusts_kmh=65))
    assert d.status == "HAZARD"
    assert "rain" in d.reason and "wind" in d.reason


def test_missing_values_are_safe():
    d = assess_day(WeatherDay(date=T))
    assert d.status == "SAFE"


def test_unavailable_day_is_unknown():
    d = assess_day(WeatherDay(date=T, forecast_available=False))
    assert d.status == "UNKNOWN" and "16 days" in d.reason


def test_worst_status():
    days = [assess_day(day()), assess_day(day(temp_max_c=39)), WeatherDay(date=T, status="UNKNOWN")]
    assert worst_status(days) == "CAUTION"
    assert worst_status([WeatherDay(date=T, status="UNKNOWN")]) == "UNKNOWN"


# ---------------------------------------------------------------------------
# Forecast fetching (mocked HTTP)
# ---------------------------------------------------------------------------
def payload(n=16, **overrides):
    times = [(T + timedelta(days=i)).isoformat() for i in range(n)]
    daily = {"time": times,
             "temperature_2m_max": [32.0] * n, "temperature_2m_min": [24.0] * n,
             "precipitation_sum": [0.0] * n, "precipitation_probability_max": [10] * n,
             "weather_code": [1] * n, "wind_gusts_10m_max": [20.0] * n}
    daily.update(overrides)
    return {"timezone": "Asia/Kolkata", "daily": daily}


@pytest.fixture
def mock_http(monkeypatch):
    weather._fetch_daily.cache_clear()
    state = {"payload": payload(), "status": 200, "exc": None, "calls": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        state["calls"] += 1
        if state["exc"]:
            raise state["exc"]
        return FakeResponse(state["payload"], state["status"])

    monkeypatch.setattr("src.tools.common.requests.get", fake_get)
    yield state
    weather._fetch_daily.cache_clear()


def test_every_trip_day_returned(mock_http):
    days = fetch_forecast(26.9, 75.8, T + timedelta(days=1), T + timedelta(days=3))
    assert [d.date for d in days] == [T + timedelta(days=i) for i in (1, 2, 3)]
    assert days[0].temp_max_c == 32.0 and days[0].condition == "Mainly clear"


def test_days_beyond_forecast_marked_unavailable(mock_http):
    days = fetch_forecast(26.9, 75.8, T + timedelta(days=14), T + timedelta(days=17))
    assert [d.forecast_available for d in days] == [True, True, False, False]


def test_tools_share_one_request(mock_http):
    args = {"latitude": 26.9, "longitude": 75.8,
            "start_date": (T + timedelta(days=1)).isoformat(), "end_date": (T + timedelta(days=2)).isoformat()}
    assert get_weather_forecast.invoke(args)["ok"]
    out = check_weather_hazards.invoke(args)
    assert out["ok"] and out["overall"] == "SAFE" and out["counts"]["SAFE"] == 2
    assert mock_http["calls"] == 1


def test_hazard_tool_flags_storm(mock_http):
    mock_http["payload"] = payload(weather_code=[95] * 16)
    out = check_weather_hazards.invoke({"latitude": 1, "longitude": 1,
                                        "start_date": T.isoformat(), "end_date": T.isoformat()})
    assert out["overall"] == "HAZARD"


def test_service_down_is_friendly(mock_http):
    mock_http["exc"] = requests.ConnectionError()
    out = get_weather_forecast.invoke({"latitude": 1, "longitude": 1,
                                       "start_date": T.isoformat(), "end_date": T.isoformat()})
    assert out == {"ok": False, "error": WEATHER_ERROR}


def test_empty_response_is_friendly(mock_http):
    mock_http["payload"] = {}
    out = check_weather_hazards.invoke({"latitude": 1, "longitude": 1,
                                        "start_date": T.isoformat(), "end_date": T.isoformat()})
    assert out["ok"] is False


@pytest.mark.parametrize("start, end", [("10 Oct", "2030-01-01"), ("2030-01-05", "2030-01-01"),
                                        ("2030-01-01", "2030-01-20")])
def test_bad_dates_rejected(mock_http, start, end):
    out = get_weather_forecast.invoke({"latitude": 1, "longitude": 1, "start_date": start, "end_date": end})
    assert out["ok"] is False
    assert mock_http["calls"] == 0