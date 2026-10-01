"""Attraction search via Geoapify Places (OpenStreetMap data)."""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel

from src.agent.schemas import Setting
from src.tools.common import ServiceUnavailableError, ToolError, http_get_json
from src.tools.transport import haversine_km
from src.utils.config import BASE_DIR, settings

logger = logging.getLogger("tripmind.tools.places")

PLACES_URL = "https://api.geoapify.com/v2/places"
INTEREST_FILE = BASE_DIR / "data" / "interest_categories.json"
RADIUS_STEPS_KM = (15, 35, 70)   # widen the search for small towns and hill stations
MAX_RESULTS = 25
PER_GROUP_LIMIT = 30
NOT_CONFIGURED = "Attraction search isn't configured — add a GEOAPIFY_API_KEY to .env."
SEARCH_ERROR = "Attraction information could not be retrieved right now."
NICE_LABELS = {
    "ruines": "Ruins", "place_of_worship": "Place of worship", "historic": "Historic building",
    "unesco": "UNESCO heritage site", "sights": "Sightseeing", "attraction": "Attraction",
    "fast_food": "Fast food", "shopping_mall": "Shopping mall", "theme_park": "Theme park",
    "archaeological_site": "Archaeological site", "city_gate": "City gate",
    "national_park": "National park", "marketplace": "Market", "heritage": "Heritage site",
}


class Attraction(BaseModel):
    name: str
    category: str
    interest: str
    latitude: float
    longitude: float
    address: Optional[str] = None
    setting: Setting = "unknown"
    est_entry_fee_per_person: float = 0.0
    distance_km: float
    notable: bool = False
    score: float


# ---------------------------------------------------------------------------
# Config & category helpers
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_interest_config() -> dict:
    with open(INTEREST_FILE, encoding="utf-8") as f:
        return json.load(f)


def _interest_key(raw: str, cfg: dict) -> Optional[str]:
    """Match 'Trekking', 'temple', 'cafes'... to a supported interest group."""
    key = " ".join(raw.strip().lower().split())
    for candidate in (key, key + "s", key.rstrip("s")):
        candidate = cfg["aliases"].get(candidate, candidate)
        if candidate in cfg["interests"]:
            return candidate
    return None


def resolve_interests(interests: Optional[list[str]]) -> tuple[dict[str, list[str]], list[str]]:
    """Interest groups to query (always including general sightseeing) and unrecognised interests."""
    cfg = load_interest_config()
    groups: dict[str, list[str]] = {}
    unmatched: list[str] = []
    for raw in interests or []:
        if not raw or not raw.strip():
            continue
        key = _interest_key(raw, cfg)
        if key:
            groups.setdefault(key, cfg["interests"][key])
        else:
            unmatched.append(raw.strip())
    groups["sightseeing"] = cfg["default"]
    return groups, unmatched


def targets(num_days: int) -> tuple[int, int]:
    """(places wanted before we stop widening the search, max places returned)."""
    days = max(1, int(num_days or 3))
    return max(6, 3 * days), min(40, max(MAX_RESULTS, 3 * days + 10))


def _specificity(cat: str) -> tuple[int, int]:
    return cat.count("."), len(cat)


def _lookup(categories: list[str], table: dict):
    """Value of the most specific table key that matches any of the place's categories."""
    best_key = None
    for c in categories:
        for key in table:
            if (c == key or c.startswith(key + ".")) and (best_key is None or _specificity(key) > _specificity(best_key)):
                best_key = key
    return table[best_key] if best_key else None


def _category_label(cat: str) -> str:
    parts = cat.split(".")
    last = parts[1] if parts[0] == "catering" and len(parts) > 1 else parts[-1]
    return NICE_LABELS.get(last, last.replace("_", " ").capitalize())


# ---------------------------------------------------------------------------
# Geoapify
# ---------------------------------------------------------------------------
@lru_cache(maxsize=256)
def _fetch_group(lat: float, lon: float, radius_m: int, categories: str, limit: int) -> tuple:
    data = http_get_json(
        PLACES_URL,
        {"categories": categories, "filter": f"circle:{lon},{lat},{radius_m}",
         "bias": f"proximity:{lon},{lat}", "limit": limit, "lang": "en",
         "apiKey": settings.geoapify_api_key},
        service="attraction search service",
    )
    return tuple(data.get("features") or [])


def _to_attraction(feature: dict, group: str, query_cats: list[str],
                   center: tuple[float, float], cfg: dict) -> Optional[Attraction]:
    p = feature.get("properties") or {}
    name = (p.get("name") or "").strip()
    if not name:
        return None
    raw = (p.get("datasource") or {}).get("raw") or {}
    if group == "food" and raw.get("brand"):
        return None  # skip chains for food recommendations
    lat, lon = p.get("lat"), p.get("lon")
    if lat is None or lon is None:
        coords = (feature.get("geometry") or {}).get("coordinates") or [None, None]
        lon, lat = coords[0], coords[1]
    if lat is None or lon is None:
        return None

    cats = p.get("categories") or []
    matched = [c for c in cats if any(c == q or c.startswith(q + ".") for q in query_cats)] or cats
    primary = max(matched, key=_specificity) if matched else "place"
    fee = _lookup(cats, cfg["entry_fees_per_person"])
    dist = haversine_km(center[0], center[1], lat, lon)

    score = (2.0 if group != "sightseeing" else 1.0)
    score += 2.0 if raw.get("wikidata") else 0.0
    score += 1.0 if raw.get("wikipedia") else 0.0
    score += 1.0 if raw.get("heritage") or any(c.startswith("heritage") for c in cats) else 0.0
    score += 0.5 if (raw.get("website") or p.get("website")) else 0.0
    score -= min(dist, 60) / 20

    return Attraction(
        name=name, category=_category_label(primary), interest=group,
        latitude=round(lat, 5), longitude=round(lon, 5),
        address=p.get("address_line2") or p.get("formatted"),
        setting=_lookup(cats, cfg["settings"]) or "unknown",
        est_entry_fee_per_person=float(fee or 0),
        distance_km=round(dist, 1),
        notable=bool(raw.get("wikidata") or raw.get("wikipedia")),
        score=round(score, 2),
    )


def _search_at(lat: float, lon: float, radius_km: float, groups: dict, cfg: dict):
    results: dict[str, list[Attraction]] = {}
    failed: list[str] = []
    for group, cats in groups.items():
        try:
            feats = _fetch_group(round(lat, 4), round(lon, 4), int(radius_km * 1000), ",".join(cats), PER_GROUP_LIMIT)
        except ToolError as exc:
            logger.warning("Attraction group %r failed: %s", group, exc.user_message)
            failed.append(group)
            continue
        results[group] = [a for f in feats if (a := _to_attraction(f, group, cats, (lat, lon), cfg))]
    return results, failed


def search_places(lat: float, lon: float, interests: Optional[list[str]], num_days: int = 3,
                  radius_km: Optional[float] = None):
    """Returns (attractions, unmatched_interests, failed_groups, radius_used_km)."""
    if not settings.key_status()["geoapify"]:
        raise ToolError(NOT_CONFIGURED)
    cfg = load_interest_config()
    groups, unmatched = resolve_interests(interests)
    want, limit = targets(num_days)
    steps = (radius_km,) if radius_km else RADIUS_STEPS_KM

    best = None
    for r in steps:
        results, failed = _search_at(lat, lon, r, groups, cfg)
        if failed and not results:
            if best:
                break
            raise ServiceUnavailableError(SEARCH_ERROR)
        best = (results, failed, r)
        unique = {a.name.casefold() for items in results.values() for a in items}
        if len(unique) >= want:
            break
        logger.info("Only %d places within %s km; widening the search", len(unique), r)
    results, failed, used_radius = best

    # Balanced selection: best few from each group, no duplicate names.
    per_group = max(4, limit // len(groups))
    selected, seen = [], set()
    for items in results.values():
        taken = 0
        for a in sorted(items, key=lambda x: x.score, reverse=True):
            key = a.name.casefold()
            if key in seen:
                continue
            seen.add(key)
            selected.append(a)
            taken += 1
            if taken >= per_group:
                break
    selected.sort(key=lambda x: x.score, reverse=True)
    return selected[:limit], unmatched, failed, used_radius


@tool
def search_attractions(destination: str, latitude: float, longitude: float,
                       interests: Optional[list[str]] = None, num_days: int = 3,
                       radius_km: Optional[float] = None) -> dict:
    """Find REAL attractions, restaurants and markets near the destination (Geoapify / OpenStreetMap)
    that match the user's interests (any wording, e.g. history, food, trekking, temples, cafes).
    Pass num_days so enough places are found; the search widens automatically for small towns.
    Each result has name, category, coordinates, indoor/outdoor setting, estimated entry fee per
    person and a suitability score. Only use attraction names returned by this tool in itineraries."""
    try:
        attractions, unmatched, failed, used_radius = search_places(latitude, longitude, interests,
                                                                    num_days, radius_km)
        out = {
            "ok": True,
            "destination": destination,
            "count": len(attractions),
            "radius_km": used_radius,
            "attractions": [a.model_dump() for a in attractions],
            "unmatched_interests": unmatched,
            "note": "Places from OpenStreetMap via Geoapify. Entry fees are rough estimates.",
        }
        if failed:
            out["partial"] = f"Some categories could not be searched: {', '.join(failed)}."
        if not attractions:
            out["note"] = "No attractions found nearby. Try different interests."
        return out
    except ToolError as exc:
        return {"ok": False, "error": exc.user_message}
    except Exception:
        logger.exception("Unexpected attraction search error")
        return {"ok": False, "error": SEARCH_ERROR}


# ---------------------------------------------------------------------------
# Manual demo:  python -m src.tools.places [city] [interest1,interest2] [days]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    from src.agent.formatter import format_inr
    from src.tools.geo import geocode, place_label

    logging.basicConfig(level=logging.INFO, format="   · %(message)s")
    city = sys.argv[1] if len(sys.argv) > 1 else "Jaipur"
    interests = sys.argv[2].split(",") if len(sys.argv) > 2 else ["history", "food"]
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    try:
        place = geocode(city)
    except ToolError as exc:
        print(exc.user_message)
        sys.exit(1)

    out = search_attractions.invoke({"destination": place.name, "latitude": place.latitude,
                                     "longitude": place.longitude, "interests": interests, "num_days": days})
    if not out["ok"]:
        print(out["error"])
        sys.exit(1)

    print(f"\n{place_label(place)} — interests: {', '.join(interests)} — "
          f"{out['count']} places within {out['radius_km']} km\n")
    for a in out["attractions"]:
        print(f"{a['score']:>5.1f}  {a['name'][:36]:<36} {a['category'][:20]:<20} {a['setting']:<8} "
              f"{a['interest']:<12} {format_inr(a['est_entry_fee_per_person']):>6}  "
              f"{a['distance_km']:>5.1f} km{'  ★' if a['notable'] else ''}")
    if out["unmatched_interests"]:
        print("\nUnrecognised interests:", out["unmatched_interests"])
    if out.get("partial"):
        print("\n" + out["partial"])