"""Understanding messages after an itinerary exists: refine, change, new trip, or question."""
from __future__ import annotations

import json
import logging
import re
from typing import Literal, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from src.agent.llm import get_llm, message_text
from src.agent.prompts import ANSWER_PROMPT, FOLLOWUP_PROMPT

logger = logging.getLogger("tripmind.followup")


class FollowUpIntent(BaseModel):
    intent: Literal["refine_plan", "change_trip", "new_trip", "question"]
    instruction: str = Field("", description="For refine_plan: the requested itinerary change in one sentence.")
    transport_mode: Optional[str] = Field(None, description="Transport mode explicitly requested, else null.")
    add_interests: list[str] = Field(default_factory=list, description="New interests to add, else empty.")


_NEGATIVE = re.compile(
    r"^\s*(no|nope|nah|cancel|never ?mind|forget it|keep (it|the (current|old|original|same) plan)|"
    r"don'?t change( it)?)\s*[.!]*\s*$",
    re.IGNORECASE,
)


def is_negative(text: str) -> bool:
    """True only for a pure 'no / keep the current plan' (used to undo a pending trip change)."""
    return bool(_NEGATIVE.match(text or ""))


def clean_history(history: list[BaseMessage]) -> list[BaseMessage]:
    """Gemini expects the conversation to start with a user message."""
    msgs = list(history or [])
    while msgs and not isinstance(msgs[0], HumanMessage):
        msgs.pop(0)
    return msgs


def classify_followup(message: str, brief: dict, history: list[BaseMessage]) -> FollowUpIntent:
    chain = FOLLOWUP_PROMPT | get_llm().with_structured_output(FollowUpIntent)
    result = chain.invoke({"plan": json.dumps(brief, ensure_ascii=False, default=str),
                           "history": clean_history(history), "message": message})
    if result is None:
        raise RuntimeError("The router returned no result")
    return result


def answer_question(message: str, brief: dict, history: list[BaseMessage]) -> str:
    chain = ANSWER_PROMPT | get_llm(temperature=0.3)
    reply = chain.invoke({"plan": json.dumps(brief, ensure_ascii=False, default=str),
                          "history": clean_history(history), "message": message})
    return message_text(reply).strip() or "Happy to help — ask me to change anything in your plan."