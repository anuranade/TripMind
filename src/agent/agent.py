"""TripMind research agent.

A LangChain tool-calling loop: Gemini sees the confirmed trip state and DECIDES which tools
to call. Python executes each tool, feeds the result back as a ToolMessage, and finally runs
a safety net so essential facts (weather, transport, attractions) are never missing."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import ToolMessage

from src.agent.llm import get_llm, message_text
from src.agent.prompts import AGENT_PROMPT
from src.agent.schemas import GeoPlace, TripRequest
from src.tools.budget import estimate_budget
from src.tools.geo import geocode_place
from src.tools.places import search_attractions
from src.tools.transport import compare_transport_modes, get_route_info, same_country
from src.tools.weather import check_weather_hazards, get_weather_forecast

logger = logging.getLogger("tripmind.agent")

TOOLS = [geocode_place, get_weather_forecast, check_weather_hazards, get_route_info,
         compare_transport_modes, search_attractions, estimate_budget]
TOOL_MAP = {t.name: t for t in TOOLS}
MAX_ROUNDS = 6


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    result: dict
    source: str  # "agent" = chosen by Gemini, "auto" = safety net, "refine" = added during refinement


@dataclass
class ResearchResult:
    calls: list[ToolCallRecord] = field(default_factory=list)
    agent_summary: str = ""

    def latest(self, name: str) -> Optional[dict]:
        for c in reversed(self.calls):
            if c.name == name and c.result.get("ok"):
                return c.result
        return None

    def error(self, name: str) -> Optional[str]:
        for c in reversed(self.calls):
            if c.name == name and not c.result.get("ok"):
                return c.result.get("error")
        return None

    def attractions(self) -> list[dict]:
        merged: dict[str, dict] = {}
        for c in self.calls:
            if c.name == "search_attractions" and c.result.get("ok"):
                for a in c.result.get("attractions", []):
                    merged.setdefault(a["name"].casefold(), a)
        return sorted(merged.values(), key=lambda a: a.get("score", 0), reverse=True)


def trip_state(trip: TripRequest, places: dict[str, GeoPlace]) -> dict:
    o, d = places["origin"], places["destination"]
    return {
        "origin": {"name": trip.origin, "latitude": o.latitude, "longitude": o.longitude,
                   "country_code": o.country_code},
        "destination": {"name": trip.destination, "latitude": d.latitude, "longitude": d.longitude,
                        "country_code": d.country_code, "population": d.population},
        "same_country": same_country(o.country_code, d.country_code),
        "start_date": trip.start_date.isoformat(),
        "end_date": trip.end_date.isoformat(),
        "num_days": trip.num_days,
        "travellers": trip.travellers,
        "budget": trip.budget,
        "travel_priority": trip.travel_priority,
        "interests": trip.interests,
    }


def run_tool(name: str, args: dict) -> dict:
    tool = TOOL_MAP.get(name)
    if tool is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    try:
        return tool.invoke(args)
    except Exception:
        logger.exception("Tool %s failed", name)
        return {"ok": False, "error": "The tool failed unexpectedly."}


def run_research(trip: TripRequest, places: dict[str, GeoPlace]) -> ResearchResult:
    state = trip_state(trip, places)
    result = ResearchResult()
    request = ("Trip confirmed. Gather the facts needed to plan it.\nTRIP STATE:\n"
               + json.dumps(state, indent=2, ensure_ascii=False))
    try:
        chain = AGENT_PROMPT | get_llm().bind_tools(TOOLS)
        scratchpad = []
        for _ in range(MAX_ROUNDS):
            ai = chain.invoke({"input": request, "agent_scratchpad": scratchpad})
            scratchpad.append(ai)
            if not ai.tool_calls:
                result.agent_summary = message_text(ai).strip()
                break
            for call in ai.tool_calls:
                logger.info("Agent chose tool → %s", call["name"])
                out = run_tool(call["name"], call["args"])
                result.calls.append(ToolCallRecord(call["name"], call["args"], out, "agent"))
                scratchpad.append(ToolMessage(content=json.dumps(out, ensure_ascii=False, default=str),
                                              tool_call_id=call["id"], name=call["name"]))
    except Exception:
        logger.exception("Agent loop failed; the safety net will gather the facts directly")
    ensure_coverage(result, state)
    return result


def _auto(result: ResearchResult, name: str, args: dict) -> None:
    logger.info("Safety net → %s (agent skipped it or used different inputs)", name)
    result.calls.append(ToolCallRecord(name, args, run_tool(name, args), "auto"))


def ensure_coverage(result: ResearchResult, state: dict) -> None:
    """Deterministic check that the essential facts exist and match the confirmed trip."""
    o, d = state["origin"], state["destination"]
    coords = {"origin_latitude": o["latitude"], "origin_longitude": o["longitude"],
              "destination_latitude": d["latitude"], "destination_longitude": d["longitude"]}
    codes = {"origin_country_code": o.get("country_code"), "destination_country_code": d.get("country_code")}

    hz = result.latest("check_weather_hazards")
    days = (hz or {}).get("days") or []
    if len(days) != state["num_days"] or days[0]["date"] != state["start_date"]:
        _auto(result, "check_weather_hazards", {"latitude": d["latitude"], "longitude": d["longitude"],
                                                "start_date": state["start_date"], "end_date": state["end_date"]})

    tr = result.latest("compare_transport_modes")
    international = state.get("same_country") is False
    if (not tr or tr.get("priority") != state["travel_priority"]
            or (international and tr.get("same_country") is not False)):
        _auto(result, "compare_transport_modes",
              {**coords, **codes, "travellers": state["travellers"], "travel_priority": state["travel_priority"]})

    if len(result.attractions()) < 2 * state["num_days"]:
        _auto(result, "search_attractions", {"destination": d["name"], "latitude": d["latitude"],
                                             "longitude": d["longitude"], "interests": state["interests"],
                                             "num_days": state["num_days"]})