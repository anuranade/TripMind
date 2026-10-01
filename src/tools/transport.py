"""Route information (ORS / OSRM / straight-line fallback) and transport-mode comparison."""
from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Optional

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel

from src.agent.formatter import format_inr
from src.agent.schemas import TransportOption
from src.tools.common import ToolError, http_get_json
from src.utils.config import BASE_DIR, settings

logger = logging.getLogger("tripmind.tools.transport")

OSRM_URL = "https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
ORS_URL = "https://api.openrouteservice.org/v2/directions/driving-car"
TRANSPORT_CSV = BASE_DIR / "data" / "transport_modes.csv"
ROAD_FACTOR = 1.3
FALLBACK_SPEED_KMH = 50.0
ROUTE_ERROR = "Route information could not be retrieved right now."
TRANSPORT_ERROR = "Transport comparison is unavailable right now."
DISCLAIMER = "All fares and travel times are estimates from TripMind's reference data — not live prices or schedules."

WEIGHTS = {
    "budget": {"cost": 0.6, "time": 0.2, "convenience": 0.2},
    "time": {"cost": 0.2, "time": 0.6, "convenience": 0.2},
    "convenience": {"cost": 0.2, "time": 0.2, "convenience": 0.6},
}
REQUIRED_COLUMNS = {"mode", "label", "distance_basis", "min_km", "max_km", "pricing", "base_cost",
                    "cost_per_km", "speed_kmh", "overhead_hours", "convenience", "capacity",
                    "use_route_time", "notes"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def hours_label(h: float) -> str:
    total = round(h * 60)
    hh, mm = divmod(total, 60)
    return f"{hh} h {mm:02d} min" if hh else f"{mm} min"


def same_country(origin_cc: Optional[str], dest_cc: Optional[str]) -> Optional[bool]:
    if not origin_cc or not dest_cc:
        return None
    return origin_cc.upper() == dest_cc.upper()


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
class NoRoadRoute(ToolError):
    """The routing service confirmed there is no road between the places (e.g. islands)."""


class RouteInfo(BaseModel):
    distance_km: float
    duration_hours: float
    straight_line_km: float
    source: str
    is_estimate: bool
    no_road: bool = False


def _ors(lat1, lon1, lat2, lon2) -> tuple[float, float]:
    data = http_get_json(ORS_URL, {"api_key": settings.ors_api_key,
                                   "start": f"{lon1},{lat1}", "end": f"{lon2},{lat2}"},
                         service="routing service")
    s = data["features"][0]["properties"]["summary"]
    return s["distance"] / 1000, s["duration"] / 3600


def _osrm(lat1, lon1, lat2, lon2) -> tuple[float, float]:
    data = http_get_json(OSRM_URL.format(lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2),
                         {"overview": "false"}, service="routing service")
    if data.get("code") == "NoRoute":
        raise NoRoadRoute("There is no road route between these places.")
    if data.get("code") != "Ok" or not data.get("routes"):
        raise ToolError("No road route was found between these places.")
    r = data["routes"][0]
    return r["distance"] / 1000, r["duration"] / 3600


@lru_cache(maxsize=128)
def _route_cached(lat1: float, lon1: float, lat2: float, lon2: float) -> RouteInfo:
    air = haversine_km(lat1, lon1, lat2, lon2)
    providers = []
    if settings.key_status()["openrouteservice"]:
        providers.append(("OpenRouteService", _ors))
    providers.append(("OSRM (OpenStreetMap)", _osrm))

    no_road = False
    for name, fn in providers:
        try:
            km, hrs = fn(lat1, lon1, lat2, lon2)
            return RouteInfo(distance_km=round(km, 1), duration_hours=round(hrs, 2),
                             straight_line_km=round(air, 1), source=name, is_estimate=False)
        except NoRoadRoute:
            logger.info("%s found no road route", name)
            no_road = True
        except (ToolError, KeyError, IndexError, TypeError) as exc:
            logger.warning("%s routing failed (%s); trying next option", name, type(exc).__name__)

    km = air * ROAD_FACTOR
    source = ("No road route — straight-line distance shown" if no_road
              else f"Straight-line distance × {ROAD_FACTOR} (estimate)")
    return RouteInfo(distance_km=round(km, 1), duration_hours=round(km / FALLBACK_SPEED_KMH, 2),
                     straight_line_km=round(air, 1), source=source, is_estimate=True, no_road=no_road)


def get_route(lat1: float, lon1: float, lat2: float, lon2: float) -> RouteInfo:
    return _route_cached(round(lat1, 4), round(lon1, 4), round(lat2, 4), round(lon2, 4))


# ---------------------------------------------------------------------------
# Transport comparison (pure Python, data from CSV)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_modes() -> pd.DataFrame:
    df = pd.read_csv(TRANSPORT_CSV)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"transport_modes.csv is missing columns: {sorted(missing)}")
    return df


def compare(route: RouteInfo, travellers: int, priority: str,
            is_same_country: Optional[bool] = None) -> list[TransportOption]:
    """All feasible modes, scored 0–100 for the priority, best first."""
    w = WEIGHTS[priority]
    # Ground travel needs a real road; for international trips we also need a confirmed route.
    ground_ok = not route.no_road and not (is_same_country is False and route.is_estimate)
    rows = []
    for m in load_modes().to_dict("records"):
        if m["distance_basis"] == "road" and not ground_ok:
            continue
        if is_same_country is False and m["mode"] == "train":
            continue
        dist = route.straight_line_km if m["distance_basis"] == "air" else route.distance_km
        if not (m["min_km"] <= dist <= m["max_km"]):
            continue
        per_vehicle = m["pricing"] == "per_vehicle"
        units = math.ceil(travellers / m["capacity"]) if per_vehicle else travellers
        one_way = (m["base_cost"] + m["cost_per_km"] * dist) * units
        if bool(m["use_route_time"]) and not route.is_estimate and m["distance_basis"] == "road":
            hours = route.duration_hours + m["overhead_hours"]
        else:
            hours = dist / m["speed_kmh"] + m["overhead_hours"]
        unit_note = (f"{units} car{'s' if units > 1 else ''}" if per_vehicle
                     else f"{travellers} ticket{'s' if travellers > 1 else ''} each way")
        rows.append({"mode": m["label"], "cost": round(2 * one_way, -1), "hours": round(hours, 1),
                     "conv": float(m["convenience"]), "note": f"{m['notes']} ({unit_note})"})

    if not rows:
        raise ToolError("No suitable transport options were found for this distance.")

    costs, times = [r["cost"] for r in rows], [r["hours"] for r in rows]

    def norm(v, vals):  # 1 = best (lowest), 0 = worst
        lo, hi = min(vals), max(vals)
        return 1.0 if hi == lo else (hi - v) / (hi - lo)

    options = [
        TransportOption(
            mode=r["mode"], estimated_cost=r["cost"], travel_time_hours=r["hours"],
            convenience_score=r["conv"], is_estimate=True, note=r["note"],
            score=round(100 * (w["cost"] * norm(r["cost"], costs) + w["time"] * norm(r["hours"], times)
                               + w["convenience"] * r["conv"] / 10), 1),
        )
        for r in rows
    ]
    return sorted(options, key=lambda o: o.score, reverse=True)


def build_reason(options: list[TransportOption], priority: str, travellers: int) -> str:
    best = options[0]
    if len(options) == 1:
        return (f"{best.mode} is the only practical option for this trip. Estimated "
                f"{format_inr(best.estimated_cost)} round trip for {travellers} "
                f"{'person' if travellers == 1 else 'people'}, about {hours_label(best.travel_time_hours)} each way.")
    cheapest = min(options, key=lambda o: o.estimated_cost)
    tags = []
    if best.estimated_cost == cheapest.estimated_cost:
        tags.append("cheapest")
    if best.travel_time_hours == min(o.travel_time_hours for o in options):
        tags.append("fastest")
    if best.convenience_score == max(o.convenience_score for o in options):
        tags.append("most convenient")

    text = f"{best.mode} is the best fit for your {priority} priority"
    if tags:
        text += f" — the {' and '.join(tags)} option"
    text += (f". Estimated {format_inr(best.estimated_cost)} round trip for {travellers} "
             f"{'person' if travellers == 1 else 'people'}, about {hours_label(best.travel_time_hours)} each way.")
    if priority == "budget" and best is not cheapest:
        extra = best.estimated_cost - cheapest.estimated_cost
        saved = cheapest.travel_time_hours - best.travel_time_hours
        if saved > 0:
            text += (f" It costs {format_inr(extra)} more than {cheapest.mode} but saves about "
                     f"{hours_label(saved)} each way and is more comfortable.")
    return text


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------
@tool
def get_route_info(origin_latitude: float, origin_longitude: float,
                   destination_latitude: float, destination_longitude: float) -> dict:
    """Road distance (km) and driving time between two coordinates. Uses OpenRouteService if
    configured, otherwise OSRM, otherwise a straight-line estimate (flagged is_estimate=true).
    no_road=true means there is no road connection (e.g. an island). Never guess distances."""
    try:
        r = get_route(origin_latitude, origin_longitude, destination_latitude, destination_longitude)
        return {"ok": True, **r.model_dump(), "duration_label": hours_label(r.duration_hours)}
    except Exception:
        logger.exception("Unexpected routing error")
        return {"ok": False, "error": ROUTE_ERROR}


@tool
def compare_transport_modes(origin_latitude: float, origin_longitude: float,
                            destination_latitude: float, destination_longitude: float,
                            travellers: int, travel_priority: str,
                            origin_country_code: Optional[str] = None,
                            destination_country_code: Optional[str] = None) -> dict:
    """Compare feasible inter-city transport (flight, train, bus, cab, self-drive) between two
    coordinates and rank them for the travel priority ('budget', 'time' or 'convenience').
    Always pass origin_country_code and destination_country_code from the trip state so that
    international trips are handled correctly. Returns the recommended mode with a reason and up
    to two alternatives. Costs are ESTIMATED round-trip totals for the whole group, not live fares."""
    try:
        if travel_priority not in WEIGHTS:
            raise ToolError("travel_priority must be 'budget', 'time' or 'convenience'.")
        if not 1 <= int(travellers) <= 10:
            raise ToolError("travellers must be between 1 and 10.")
        route = get_route(origin_latitude, origin_longitude, destination_latitude, destination_longitude)
        same = same_country(origin_country_code, destination_country_code)
        options = compare(route, int(travellers), travel_priority, same)

        def dump(o: TransportOption) -> dict:
            return {**o.model_dump(), "travel_time_label": hours_label(o.travel_time_hours)}

        out = {
            "ok": True,
            "route": route.model_dump(),
            "priority": travel_priority,
            "same_country": same,
            "weights": WEIGHTS[travel_priority],
            "recommended": dump(options[0]),
            "reason": build_reason(options, travel_priority, int(travellers)),
            "alternatives": [dump(o) for o in options[1:3]],
            "all_options": [dump(o) for o in options],
            "disclaimer": DISCLAIMER,
        }
        if all(o.mode == "Flight" for o in options):
            out["note"] = ("Only flights are shown because there is no road route between these places."
                           if route.no_road else "Only flights are shown for this international trip.")
        return out
    except ToolError as exc:
        return {"ok": False, "error": exc.user_message}
    except Exception:
        logger.exception("Unexpected transport comparison error")
        return {"ok": False, "error": TRANSPORT_ERROR}


# ---------------------------------------------------------------------------
# Manual demo:  python -m src.tools.transport [origin] [destination] [travellers]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    from src.tools.geo import geocode

    logging.basicConfig(level=logging.WARNING)
    origin = sys.argv[1] if len(sys.argv) > 1 else "Mumbai"
    dest = sys.argv[2] if len(sys.argv) > 2 else "Jaipur"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    try:
        o, d = geocode(origin), geocode(dest)
    except ToolError as exc:
        print(exc.user_message)
        sys.exit(1)

    coords = {"origin_latitude": o.latitude, "origin_longitude": o.longitude,
              "destination_latitude": d.latitude, "destination_longitude": d.longitude}
    r = get_route_info.invoke(coords)
    if not r["ok"]:
        print(r["error"])
        sys.exit(1)
    print(f"\n{o.name} ({o.country_code}) → {d.name} ({d.country_code}): {r['distance_km']} km, "
          f"{r['duration_label']} — source: {r['source']}")

    for p in ("budget", "time", "convenience"):
        out = compare_transport_modes.invoke({**coords, "travellers": n, "travel_priority": p,
                                              "origin_country_code": o.country_code,
                                              "destination_country_code": d.country_code})
        print(f"\n--- {p.upper()} priority ---")
        if not out["ok"]:
            print(out["error"])
            continue
        for i, opt in enumerate([out["recommended"]] + out["alternatives"], 1):
            print(f"{i}. {opt['mode']:<18} {format_inr(opt['estimated_cost']):>10} round trip   "
                  f"~{opt['travel_time_label']:<13} each way   score {opt['score']}")
        print("Why:", out["reason"])
        if out.get("note"):
            print("Note:", out["note"])
    print(f"\n{DISCLAIMER}")