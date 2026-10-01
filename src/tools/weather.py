"""Weather forecast (Open-Meteo) and deterministic weather-hazard assessment."""
from __future__ import annotations

import logging
from datetime import date, timedelta
from functools import lru_cache

from langchain_core.tools import tool

from src.agent.schemas import MAX_TRIP_DAYS, WeatherDay
from src.tools.common import ServiceUnavailableError, ToolError, http_get_json

logger = logging.getLogger("tripmind.tools.weather")

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_DAYS = 16
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_probability_max",
    "weather_code",
    "wind_gusts_10m_max",
]
WEATHER_ERROR = "Weather information could not be retrieved right now."
NOT_AVAILABLE = ("No forecast yet — Open-Meteo forecasts up to 16 days ahead. "
                 "Check again closer to the trip.")

# ---------------------------------------------------------------------------
# WMO weather codes -> readable condition
# ---------------------------------------------------------------------------
WMO_CONDITIONS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Freezing fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Dense drizzle",
    56: "Freezing drizzle", 57: "Dense freezing drizzle",
    61: "Light rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Freezing rain", 67: "Heavy freezing rain",
    71: "Light snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Light rain showers", 81: "Rain showers", 82: "Violent rain showers",
    85: "Snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Severe thunderstorm with hail",
}

# ---------------------------------------------------------------------------
# Hazard thresholds (deterministic — no LLM)
# ---------------------------------------------------------------------------
RAIN_CAUTION_MM, RAIN_HAZARD_MM = 10.0, 30.0
RAIN_PROB_CAUTION = 70.0
HEAT_CAUTION_C, HEAT_HAZARD_C = 38.0, 42.0
GUST_CAUTION_KMH, GUST_HAZARD_KMH = 40.0, 60.0
HAZARD_CODES = {82, 86, 75, 95, 96, 99}
CAUTION_CODES = {45, 48, 57, 65, 67, 73}
SEVERITY = {"SAFE": 0, "CAUTION": 1, "HAZARD": 2}


def assess_day(day: WeatherDay) -> WeatherDay:
    """Return a copy of `day` with status (SAFE/CAUTION/HAZARD) and reason filled in."""
    if not day.forecast_available:
        return day.model_copy(update={"status": "UNKNOWN", "reason": day.reason or NOT_AVAILABLE})

    findings: list[tuple[str, str]] = []

    rain = day.precipitation_mm
    if rain is not None:
        if rain > RAIN_HAZARD_MM:
            findings.append(("HAZARD", f"Very heavy rain expected ({rain:.0f} mm)"))
        elif rain >= RAIN_CAUTION_MM:
            findings.append(("CAUTION", f"Heavy rain expected ({rain:.0f} mm)"))

    prob = day.precipitation_probability
    if prob is not None and prob >= RAIN_PROB_CAUTION and not (rain is not None and rain >= RAIN_CAUTION_MM):
        findings.append(("CAUTION", f"High probability of rainfall ({prob:.0f}%)"))

    code = day.weather_code
    if code in HAZARD_CODES:
        findings.append(("HAZARD", f"Severe weather forecast: {WMO_CONDITIONS[code].lower()}"))
    elif code in CAUTION_CODES:
        findings.append(("CAUTION", f"{WMO_CONDITIONS[code]} forecast"))

    tmax = day.temp_max_c
    if tmax is not None:
        if tmax > HEAT_HAZARD_C:
            findings.append(("HAZARD", f"Extreme heat (max {tmax:.0f}°C)"))
        elif tmax >= HEAT_CAUTION_C:
            findings.append(("CAUTION", f"High heat (max {tmax:.0f}°C)"))

    gust = day.wind_gusts_kmh
    if gust is not None:
        if gust > GUST_HAZARD_KMH:
            findings.append(("HAZARD", f"Dangerous wind gusts up to {gust:.0f} km/h"))
        elif gust >= GUST_CAUTION_KMH:
            findings.append(("CAUTION", f"Strong wind gusts up to {gust:.0f} km/h"))

    if not findings:
        return day.model_copy(update={"status": "SAFE", "reason": "No significant weather risks"})
    status = max((s for s, _ in findings), key=SEVERITY.get)
    return day.model_copy(update={"status": status, "reason": "; ".join(r for _, r in findings)})


def assess_days(days: list[WeatherDay]) -> list[WeatherDay]:
    return [assess_day(d) for d in days]


def worst_status(days: list[WeatherDay]) -> str:
    known = [d.status for d in days if d.status in SEVERITY]
    return max(known, key=SEVERITY.get) if known else "UNKNOWN"


# ---------------------------------------------------------------------------
# Open-Meteo forecast
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _fetch_daily(lat: float, lon: float, day_key: str) -> dict:
    """One request per location per day (day_key refreshes the cache daily)."""
    data = http_get_json(
        FORECAST_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "daily": ",".join(DAILY_VARS),
            "timezone": "auto",
            "forecast_days": FORECAST_DAYS,
        },
        service="weather service",
    )
    daily = data.get("daily") or {}
    if not daily.get("time"):
        logger.warning("Open-Meteo returned no daily data")
        raise ServiceUnavailableError(WEATHER_ERROR)
    return daily


def _parse_daily(daily: dict) -> dict[date, WeatherDay]:
    def val(key: str, i: int):
        col = daily.get(key) or []
        return col[i] if i < len(col) else None

    out: dict[date, WeatherDay] = {}
    for i, t in enumerate(daily["time"]):
        code = val("weather_code", i)
        code = int(code) if code is not None else None
        out[date.fromisoformat(t)] = WeatherDay(
            date=date.fromisoformat(t),
            temp_min_c=val("temperature_2m_min", i),
            temp_max_c=val("temperature_2m_max", i),
            precipitation_mm=val("precipitation_sum", i),
            precipitation_probability=val("precipitation_probability_max", i),
            weather_code=code,
            condition=WMO_CONDITIONS.get(code, "Unknown") if code is not None else None,
            wind_gusts_kmh=val("wind_gusts_10m_max", i),
        )
    return out


def fetch_forecast(lat: float, lon: float, start: date, end: date) -> list[WeatherDay]:
    """One WeatherDay per trip day; days outside the forecast window are marked unavailable."""
    by_date = _parse_daily(_fetch_daily(round(lat, 3), round(lon, 3), date.today().isoformat()))
    days, d = [], start
    while d <= end:
        days.append(by_date.get(d) or WeatherDay(date=d, forecast_available=False,
                                                 status="UNKNOWN", reason=NOT_AVAILABLE))
        d += timedelta(days=1)
    return days


def _parse_range(start_date: str, end_date: str) -> tuple[date, date]:
    try:
        s, e = date.fromisoformat(start_date.strip()), date.fromisoformat(end_date.strip())
    except (ValueError, AttributeError):
        raise ToolError("Dates must be in YYYY-MM-DD format.")
    if e < s:
        raise ToolError("end_date must be on or after start_date.")
    if (e - s).days + 1 > MAX_TRIP_DAYS:
        raise ToolError(f"Weather can be checked for at most {MAX_TRIP_DAYS} days at a time.")
    return s, e


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------
@tool
def get_weather_forecast(latitude: float, longitude: float, start_date: str, end_date: str) -> dict:
    """Get the real daily weather forecast (Open-Meteo) for EVERY day from start_date to end_date
    (YYYY-MM-DD) at the given coordinates: min/max temperature (°C), precipitation (mm),
    precipitation probability (%), weather condition and wind gusts (km/h).
    Days beyond the ~16-day forecast range have forecast_available=false. Never guess weather."""
    try:
        s, e = _parse_range(start_date, end_date)
        days = fetch_forecast(latitude, longitude, s, e)
        return {"ok": True, "source": "Open-Meteo forecast",
                "days": [d.model_dump(mode="json", exclude={"status", "reason"}) for d in days]}
    except ServiceUnavailableError:
        return {"ok": False, "error": WEATHER_ERROR}
    except ToolError as exc:
        return {"ok": False, "error": exc.user_message}
    except Exception:
        logger.exception("Unexpected weather error")
        return {"ok": False, "error": WEATHER_ERROR}


@tool
def check_weather_hazards(latitude: float, longitude: float, start_date: str, end_date: str) -> dict:
    """Assess weather risk for EVERY day from start_date to end_date (YYYY-MM-DD) at the given
    coordinates using fixed rules for rain, thunderstorms, heat and wind. Each day gets
    SAFE, CAUTION or HAZARD with a reason (UNKNOWN if no forecast exists yet).
    Use this to decide indoor vs outdoor activities and when a Plan B is needed."""
    try:
        s, e = _parse_range(start_date, end_date)
        days = assess_days(fetch_forecast(latitude, longitude, s, e))
        counts = {k: sum(d.status == k for d in days) for k in ("SAFE", "CAUTION", "HAZARD", "UNKNOWN")}
        return {
            "ok": True,
            "source": "Open-Meteo forecast + TripMind hazard rules",
            "overall": worst_status(days),
            "counts": counts,
            "days": [d.model_dump(mode="json") for d in days],
        }
    except ServiceUnavailableError:
        return {"ok": False, "error": WEATHER_ERROR}
    except ToolError as exc:
        return {"ok": False, "error": exc.user_message}
    except Exception:
        logger.exception("Unexpected hazard-check error")
        return {"ok": False, "error": WEATHER_ERROR}


# ---------------------------------------------------------------------------
# Manual demo:  python -m src.tools.weather [city] [num_days] [days_from_today]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    from src.tools.geo import geocode, place_label

    logging.basicConfig(level=logging.WARNING)
    city = sys.argv[1] if len(sys.argv) > 1 else "Jaipur"
    n_days = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    offset = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    try:
        place = geocode(city)
    except ToolError as exc:
        print(exc.user_message)
        sys.exit(1)

    start = date.today() + timedelta(days=offset)
    end = start + timedelta(days=n_days - 1)
    print(f"\n{place_label(place)}  ({place.latitude:.2f}, {place.longitude:.2f})  {start} → {end}\n")

    out = check_weather_hazards.invoke({"latitude": place.latitude, "longitude": place.longitude,
                                        "start_date": start.isoformat(), "end_date": end.isoformat()})
    if not out["ok"]:
        print(out["error"])
        sys.exit(1)

    def f(v, spec="{:.0f}"):
        return "–" if v is None else spec.format(v)

    for d in out["days"]:
        print(f"{d['date']}  {f(d['temp_min_c']):>3}–{f(d['temp_max_c']):<3}°C  "
              f"rain {f(d['precipitation_mm'], '{:.1f}'):>5} mm ({f(d['precipitation_probability']):>3}%)  "
              f"gusts {f(d['wind_gusts_kmh']):>3} km/h  {(d['condition'] or '–'):<24} "
              f"{d['status']:<8} {d['reason']}")
    print(f"\nOverall: {out['overall']}   Counts: {out['counts']}")