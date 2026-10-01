"""Geocoding via the Open-Meteo Geocoding API (free, no key)."""
from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.tools import tool

from src.agent.schemas import GeoPlace
from src.tools.common import ToolError, http_get_json
from src.utils.config import settings

logger = logging.getLogger("tripmind.tools.geo")

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


class PlaceNotFoundError(ToolError):
    def __init__(self, place: str) -> None:
        super().__init__(
            f'I couldn\'t find "{place}". Please check the spelling or try a nearby larger city.'
        )
        self.place = place


def place_label(p: GeoPlace) -> str:
    """'Jaipur, Rajasthan, India'"""
    parts = [p.name]
    if p.admin1 and p.admin1 != p.name:
        parts.append(p.admin1)
    if p.country:
        parts.append(p.country)
    return ", ".join(parts)


@lru_cache(maxsize=256)
def _search(name: str) -> tuple:
    data = http_get_json(
        GEOCODING_URL,
        {"name": name, "count": 10, "language": "en", "format": "json"},
        service="location service",
    )
    return tuple(data.get("results") or [])


def _matches_hints(result: dict, hints: list[str]) -> bool:
    text = " ".join(str(result.get(k) or "") for k in ("admin1", "admin2", "country", "country_code")).casefold()
    return all(h in text for h in hints)


def _pick(results: list[dict], hints: list[str], prefer_country: str) -> dict | None:
    if hints:
        results = [r for r in results if _matches_hints(r, hints)]
    if not results:
        return None
    top = results[0]
    if prefer_country and (top.get("country_code") or "").upper() != prefer_country:
        top_pop = top.get("population") or 0
        for r in results:
            if (r.get("country_code") or "").upper() == prefer_country and (r.get("population") or 0) >= 0.2 * top_pop:
                return r
    return top


def geocode(place: str) -> GeoPlace:
    """Resolve a place name. Raises PlaceNotFoundError or ServiceUnavailableError."""
    query = " ".join((place or "").split())
    parts = [p.strip() for p in query.split(",") if p.strip()]
    if not parts or len(parts[0]) < 2:
        raise PlaceNotFoundError(query or "(empty)")
    name, hints = parts[0], [h.casefold() for h in parts[1:]]

    best = _pick(list(_search(name.casefold())), hints, settings.default_country_code)
    if best is None:
        raise PlaceNotFoundError(query)
    return GeoPlace(
        name=best["name"],
        latitude=best["latitude"],
        longitude=best["longitude"],
        country=best.get("country"),
        country_code=best.get("country_code"),
        admin1=best.get("admin1"),
        population=best.get("population"),
    )


@tool
def geocode_place(place: str) -> dict:
    """Look up a city or place and return its official name, latitude, longitude, state and country.
    Use it to verify that a city exists and to get coordinates for weather, routes and attractions.
    Returns {"ok": false, "error": ...} if the place cannot be found or the service is down."""
    try:
        p = geocode(place)
        return {"ok": True, **p.model_dump(), "label": place_label(p)}
    except ToolError as exc:
        return {"ok": False, "error": exc.user_message}
    except Exception:  # never crash the agent
        logger.exception("Unexpected geocoding error for %r", place)
        return {"ok": False, "error": "Location lookup failed unexpectedly. Please try again."}


# Manual demo:  python -m src.tools.geo
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.WARNING)
    for q in ["Jaipur", "Mumbai", "Hyderabad", "Aurangabad, Bihar", "Paris", "Xyzabad", ""]:
        print(f"\n>>> {q!r}")
        print(json.dumps(geocode_place.invoke({"place": q}), indent=2, ensure_ascii=False))