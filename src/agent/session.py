"""One TripMind conversation: intake → planning → follow-ups, keeping every plan version.
Sessions live in memory; SQLite persistence is a future objective."""
from __future__ import annotations

import copy
import logging
from typing import Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.followup import answer_question, classify_followup, is_negative
from src.agent.formatter import dates_label, format_inr
from src.agent.intake import IntakeReply, TripIntake
from src.agent.planner import plan_brief, plan_trip, refine_plan

logger = logging.getLogger("tripmind.session")

MAX_VERSIONS = 10
PLAN_ERROR = "Sorry — I couldn't finish planning right now. Please press Confirm again in a moment."
READY_MSG = ('Your itinerary is ready. Ask me for changes — e.g. "make day 2 more relaxed", '
             '"use the bus instead" or "add more food places".')
FOLLOWUP_ERROR = "Sorry — I couldn't process that right now. Please try again."
REFINE_ERROR = "Sorry — I couldn't update the plan right now. Your current plan is unchanged."
CHANGE_PREFIX = 'Here\'s your updated trip — confirm to re-plan it, or say "keep the current plan".'
KEPT_MSG = "No problem — I've kept your current plan."
VERSION_MSG = "Showing version {number} ({label}). Changes you ask for now will build on this version."


def _response(stage: str, text: str, **kw) -> dict:
    base = {"stage": stage, "text": text, "missing": [], "errors": [], "summary": None,
            "itinerary": None, "extras": None, "steps": None, "versions": None, "clear_results": False}
    base.update(kw)
    return base


def _short(text: str, n: int = 48) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def change_label(old, new) -> str:
    """'Updated: 5 days, 3 travellers' — what changed between two confirmed trips."""
    if old is None:
        return "Updated trip"
    parts = []
    if (old.origin, old.destination) != (new.origin, new.destination):
        parts.append(f"{new.origin} → {new.destination}")
    if (old.start_date, old.end_date) != (new.start_date, new.end_date):
        parts.append(f"{dates_label(new.start_date, new.end_date)}")
    if old.travellers != new.travellers:
        parts.append(f"{new.travellers} travellers")
    if old.budget != new.budget:
        parts.append(format_inr(new.budget))
    if old.travel_priority != new.travel_priority:
        parts.append(f"{new.travel_priority} priority")
    if old.interests != new.interests:
        parts.append(", ".join(new.interests) or "no interests")
    return "Updated: " + ", ".join(parts) if parts else "Updated trip"


class TripMindSession:
    def __init__(self, intake: Optional[TripIntake] = None, planner: Optional[Callable] = None,
                 router: Optional[Callable] = None, refiner: Optional[Callable] = None,
                 answerer: Optional[Callable] = None) -> None:
        self.intake = intake or TripIntake()
        self.planner = planner or plan_trip
        self.router = router or classify_followup
        self.refiner = refiner or refine_plan
        self.answerer = answerer or answer_question
        self.versions: list[dict] = []   # every plan made in this trip; versions[0] is the original
        self.active = -1                 # index of the version being viewed / built on
        self._backup = None              # trip state saved while a change waits for confirmation

    @property
    def plan(self) -> Optional[dict]:
        return self.versions[self.active] if self.versions else None

    def reset(self) -> None:
        self.intake.reset()
        self.versions, self.active, self._backup = [], -1, None

    # ---------- public API ----------
    def handle(self, message: str) -> dict:
        message = (message or "").strip()
        if self.plan is not None and self.intake.stage == "confirmed":
            return self._follow_up(message)
        if self.plan is not None and self._backup is not None and is_negative(message):
            return self._restore(message)
        reply = self.intake.handle(message)
        return self._plan() if reply.stage == "confirmed" else self._from_intake(reply)

    def confirm(self) -> dict:
        if self.intake.stage == "confirmed":
            return self._planned(READY_MSG) if self.plan is not None else self._plan()
        reply = self.intake.confirm()
        return self._plan() if reply.stage == "confirmed" else self._from_intake(reply)

    def select_version(self, number: int) -> dict:
        """Show an earlier (or later) version; further changes build on it."""
        if not 1 <= number <= len(self.versions):
            msg = f"There is no plan version {number}."
            return _response(self.intake.stage, msg, errors=[msg], versions=self._versions_meta())
        self.active = number - 1
        v = self.plan
        self.intake.trip = copy.deepcopy(v["trip"])
        self.intake.places = dict(v["places"])
        self.intake.draft = copy.deepcopy(v["draft"])
        self.intake.stage = "confirmed"
        self._backup = None
        text = VERSION_MSG.format(number=number, label=v["label"])
        self.intake.history.append(AIMessage(text))
        return self._planned_full(text)

    # ---------- after an itinerary exists ----------
    def _follow_up(self, message: str) -> dict:
        if not message:
            return self._planned(READY_MSG)
        history = self.intake.history[-8:]
        brief = plan_brief(self.plan["itinerary"])
        try:
            intent = self.router(message, brief, history)
        except Exception:
            logger.exception("Follow-up routing failed")
            return self._record(message, self._planned(FOLLOWUP_ERROR))
        logger.info("Follow-up intent: %s", intent.model_dump())

        if intent.intent == "refine_plan" and (intent.instruction or intent.transport_mode or intent.add_interests):
            return self._refine(message, intent)
        if intent.intent == "change_trip":
            return self._change(message)
        if intent.intent == "new_trip":
            return self._new_trip(message)

        try:
            text = self.answerer(message, brief, history)
        except Exception:
            logger.exception("Answering failed")
            text = READY_MSG
        return self._record(message, self._planned(text))

    def _refine(self, message: str, intent) -> dict:
        try:
            result = self.refiner(self.intake.trip, self.intake.places, self.plan, intent)
        except Exception:
            logger.exception("Refinement failed")
            result = None
        if not result:
            return self._record(message, self._planned(REFINE_ERROR))
        label = (intent.instruction
                 or (f"Transport: {intent.transport_mode}" if intent.transport_mode else "")
                 or ("Added " + ", ".join(intent.add_interests)))
        self._store(result, label)
        text = (result.get("extras") or {}).get("intro") or "I've updated your plan."
        return self._record(message, self._planned_full(text))

    def _change(self, message: str) -> dict:
        self._backup = (copy.deepcopy(self.intake.draft), copy.deepcopy(self.intake.trip),
                        dict(self.intake.places), "confirmed")
        self.intake.stage = "awaiting_confirmation"  # reopen the confirmed trip for edits
        reply = self.intake.handle(message)
        if reply.stage == "confirmed":
            return self._plan()
        resp = self._from_intake(reply)
        if reply.stage == "awaiting_confirmation":
            resp["text"] = f"{CHANGE_PREFIX}\n{reply.text}"
        return resp

    def _new_trip(self, message: str) -> dict:
        self.reset()
        reply = self.intake.handle(message)
        resp = self._plan() if reply.stage == "confirmed" else self._from_intake(reply)
        resp["clear_results"] = True
        return resp

    def _restore(self, message: str) -> dict:
        draft, trip, places, stage = self._backup
        self.intake.draft, self.intake.trip, self.intake.places, self.intake.stage = draft, trip, places, stage
        self._backup = None
        return self._record(message, self._planned(KEPT_MSG))

    # ---------- versions ----------
    def _store(self, result: dict, label: str) -> None:
        version = {**result, "label": _short(label),
                   "trip": copy.deepcopy(self.intake.trip),
                   "places": dict(self.intake.places),
                   "draft": copy.deepcopy(self.intake.draft)}
        self.versions.append(version)
        if len(self.versions) > MAX_VERSIONS:
            self.versions.pop(1)  # never drop the original
        self.active = len(self.versions) - 1

    def _versions_meta(self) -> list[dict]:
        return [{"number": i + 1, "label": v["label"], "active": i == self.active}
                for i, v in enumerate(self.versions)]

    # ---------- helpers ----------
    def _planned(self, text: str, **kw) -> dict:
        return _response("planned", text, summary=self.intake._card(), versions=self._versions_meta(), **kw)

    def _planned_full(self, text: str) -> dict:
        return self._planned(text, itinerary=self.plan["itinerary"].model_dump(mode="json"),
                             extras=self.plan.get("extras"))

    def _record(self, message: str, resp: dict) -> dict:
        self.intake.history += [HumanMessage(message), AIMessage(resp["text"])]
        return resp

    def _from_intake(self, r: IntakeReply) -> dict:
        return _response(r.stage, r.text, missing=r.missing, errors=r.errors, summary=r.summary)

    def _plan(self) -> dict:
        try:
            result = self.planner(self.intake.trip, self.intake.places)
        except Exception:
            logger.exception("Planning failed")
            return _response("confirmed", PLAN_ERROR, summary=self.intake._card(), errors=[PLAN_ERROR])
        label = change_label(self._backup[1] if self._backup else None, self.intake.trip) \
            if self.versions else "Original plan"
        self._store(result, label)
        self._backup = None
        intro = (result.get("extras") or {}).get("intro") or "Here's your trip plan."
        resp = self._planned_full(f"{intro} Your full itinerary is below.")
        resp["steps"] = result.get("steps")
        return resp


# ---------------------------------------------------------------------------
# Terminal demo:  python -m src.agent.session
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.agent.planner import itinerary_text

    logging.basicConfig(level=logging.WARNING, format="   · %(message)s")
    logging.getLogger("tripmind.agent").setLevel(logging.INFO)
    logging.getLogger("tripmind.session").setLevel(logging.INFO)

    session = TripMindSession()
    print("TripMind — type your trip. 'reset' starts over, 'v1'/'v2'... switches version, 'quit' exits.\n")
    while True:
        try:
            msg = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        cmd = msg.strip().lower()
        if cmd in {"quit", "exit"}:
            break
        if cmd == "reset":
            session.reset()
            print("(new trip)\n")
            continue
        reply = session.select_version(int(cmd[1:])) if cmd[:1] == "v" and cmd[1:].isdigit() else session.handle(msg)
        print(f"\nTripMind [{reply['stage']}]:\n{reply['text']}\n")
        if reply["versions"]:
            print("Versions: " + " | ".join(f"{'*' if v['active'] else ''}v{v['number']} {v['label']}"
                                            for v in reply["versions"]))
        if reply["itinerary"]:
            print(itinerary_text(session.plan["itinerary"], session.plan["extras"]))