"""Pydantic schemas for TripMind.

TripRequestDraft -> loose, all-optional; filled by the LLM from conversation.
TripRequest      -> strict, validated; the only trip object tools may use.
The remaining models describe tool outputs and the final itinerary.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

TravelPriority = Literal["budget", "time", "convenience"]
SafetyStatus = Literal["SAFE", "CAUTION", "HAZARD", "UNKNOWN"]
Setting = Literal["indoor", "outdoor", "mixed", "unknown"]
TimeOfDay = Literal["morning", "afternoon", "evening"]

MAX_TRIP_DAYS = 10
MAX_TRAVELLERS = 10


def today() -> date:
    """Wrapped in a function so tests can override 'today'."""
    return date.today()


def _normalise_interests(value) -> list[str]:
    """Accept a list or comma-separated string; lowercase, trim, de-duplicate."""
    if value is None:
        return []
    if isinstance(value, str):
        value = value.split(",")
    seen: list[str] = []
    for item in value:
        item = " ".join(str(item).split()).lower()
        if item and item not in seen:
            seen.append(item)
    return seen


def _clean_text(value):
    if value is None:
        return None
    return " ".join(str(value).split())


# ---------------------------------------------------------------------------
# Trip request
# ---------------------------------------------------------------------------
class TripExtraction(BaseModel):
    """What the LLM returns. Simple JSON types only; Python converts and validates afterwards."""

    origin: Optional[str] = Field(None, description="Starting city stated in the latest message, else null.")
    destination: Optional[str] = Field(None, description="Destination city stated in the latest message, else null.")
    start_date: Optional[str] = Field(None, description="Start date as YYYY-MM-DD, only if stated.")
    end_date: Optional[str] = Field(None, description="End date as YYYY-MM-DD, only if explicitly stated.")
    duration_days: Optional[int] = Field(
        None, description="Trip length in days if stated ('3 days' -> 3, '3 nights' -> 4, 'a week' -> 7, 'weekend' -> 2)."
    )
    travellers: Optional[int] = Field(None, description="Number of people travelling, only if stated.")
    budget: Optional[float] = Field(None, description="Total budget in INR for the whole group, only if stated.")
    travel_priority: Optional[TravelPriority] = Field(
        None, description="'budget', 'time' or 'convenience', only if the user indicates it."
    )
    interests: list[str] = Field(default_factory=list, description="Short lowercase interest keywords.")
    same_city_intended: bool = Field(
        False, description="True only if the user explicitly wants a trip within the same city."
    )


class TripRequestDraft(BaseModel):
    """Partially known trip details. Anything the user has not said stays null."""

    origin: Optional[str] = Field(None, description="Starting city as stated by the user. null if not mentioned.")
    destination: Optional[str] = Field(None, description="Destination city as stated by the user. null if not mentioned.")
    start_date: Optional[date] = Field(None, description="Trip start date (YYYY-MM-DD). null if not mentioned.")
    end_date: Optional[date] = Field(None, description="Trip end date (YYYY-MM-DD). null if not mentioned.")
    duration_days: Optional[int] = Field(None, description="Trip length in days, if known.")
    travellers: Optional[int] = Field(None, description="Number of people travelling. null if not mentioned.")
    budget: Optional[float] = Field(None, description="Total budget in INR for the whole group. null if not mentioned.")
    travel_priority: Optional[TravelPriority] = Field(
        None, description="One of 'budget', 'time', 'convenience'. null if not mentioned."
    )
    interests: list[str] = Field(default_factory=list, description="Interests such as history, food, nature.")
    same_city_intended: bool = Field(
        False, description="True only if the user explicitly wants origin and destination to be the same place."
    )

    @field_validator("origin", "destination", mode="before")
    @classmethod
    def _clean_city(cls, v):
        return _clean_text(v) or None

    @field_validator("interests", mode="before")
    @classmethod
    def _clean_interests(cls, v):
        return _normalise_interests(v)

    @model_validator(mode="after")
    def _resolve_dates(self) -> "TripRequestDraft":
        """Deterministically fill a missing date from the trip length (Python, not the LLM)."""
        if self.duration_days is not None and self.duration_days <= 0:
            self.duration_days = None
        d = self.duration_days
        if self.start_date and self.end_date:
            if self.end_date >= self.start_date:
                self.duration_days = (self.end_date - self.start_date).days + 1
        elif self.start_date and d:
            self.end_date = self.start_date + timedelta(days=d - 1)
        elif self.end_date and d:
            self.start_date = self.end_date - timedelta(days=d - 1)
        return self

class TripRequest(BaseModel):
    """A complete, validated trip. Created only when every rule passes."""

    origin: str = Field(..., min_length=2)
    destination: str = Field(..., min_length=2)
    start_date: date
    end_date: date
    travellers: int = Field(..., ge=1, le=MAX_TRAVELLERS)
    budget: float = Field(..., gt=0)
    travel_priority: TravelPriority
    interests: list[str] = Field(default_factory=list)
    same_city_intended: bool = False

    @field_validator("origin", "destination", mode="before")
    @classmethod
    def _clean_city(cls, v):
        return _clean_text(v)

    @field_validator("interests", mode="before")
    @classmethod
    def _clean_interests(cls, v):
        return _normalise_interests(v)

    @field_validator("start_date")
    @classmethod
    def _start_not_in_past(cls, v: date) -> date:
        if v < today():
            raise ValueError(f"The start date ({v:%d %b %Y}) is in the past. Please choose a future date.")
        return v

    @model_validator(mode="after")
    def _check_trip(self) -> "TripRequest":
        if self.end_date < self.start_date:
            raise ValueError("The end date must be on or after the start date.")
        if self.num_days > MAX_TRIP_DAYS:
            raise ValueError(
                f"Trips can be 1–{MAX_TRIP_DAYS} days long; this one is {self.num_days} days. "
                "Please shorten the dates."
            )
        if self.origin.casefold() == self.destination.casefold() and not self.same_city_intended:
            raise ValueError(
                "The starting city and destination are the same. "
                "Please check them, or tell me if you want a trip within the same city."
            )
        return self

    @property
    def num_days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    @property
    def num_nights(self) -> int:
        return max(self.num_days - 1, 0)

    def summary(self) -> str:
        """One-line human summary, used for the confirmation step."""
        dates = f"{self.start_date:%d %b} – {self.end_date:%d %b %Y}"
        interests = ", ".join(self.interests) if self.interests else "none specified"
        return (
            f"{self.origin} → {self.destination} | {dates} ({self.num_days} days) | "
            f"{self.travellers} traveller{'s' if self.travellers > 1 else ''} | "
            f"₹{self.budget:,.0f} | priority: {self.travel_priority} | interests: {interests}"
        )


# ---------------------------------------------------------------------------
# Tool outputs and final itinerary (filled in later phases)
# ---------------------------------------------------------------------------
class GeoPlace(BaseModel):
    name: str
    latitude: float
    longitude: float
    country: Optional[str] = None
    country_code: Optional[str] = None
    admin1: Optional[str] = None  # state / region
    population: Optional[int] = None


class WeatherDay(BaseModel):
    date: date
    forecast_available: bool = True
    temp_min_c: Optional[float] = None
    temp_max_c: Optional[float] = None
    precipitation_mm: Optional[float] = None
    precipitation_probability: Optional[float] = None
    weather_code: Optional[int] = None
    condition: Optional[str] = None
    wind_gusts_kmh: Optional[float] = None
    status: SafetyStatus = "UNKNOWN"
    reason: str = ""


class TransportOption(BaseModel):
    mode: str
    estimated_cost: float = Field(..., description="Estimated round-trip cost for the whole group (INR).")
    travel_time_hours: float = Field(..., description="Estimated one-way travel time.")
    convenience_score: float = Field(..., ge=0, le=10)
    score: Optional[float] = Field(None, description="Weighted ranking score; higher is better.")
    is_estimate: bool = True
    note: str = ""


class TransportPlan(BaseModel):
    recommended: TransportOption
    reason: str
    alternatives: list[TransportOption] = Field(default_factory=list, max_length=2)


class Activity(BaseModel):
    time_of_day: TimeOfDay
    name: str
    category: Optional[str] = None
    setting: Setting = "unknown"
    approx_cost: Optional[float] = Field(None, description="Estimated cost for the group (INR).")
    plan_b: Optional[str] = Field(None, description="Indoor alternative for risky-weather days.")
    notes: str = ""


class DayPlan(BaseModel):
    day_number: int = Field(..., ge=1)
    date: date
    weather_status: SafetyStatus = "SAFE"
    theme: str = ""
    activities: list[Activity] = Field(default_factory=list)


class BudgetBreakdown(BaseModel):
    transport: float
    stay: float
    food: float
    local_transport: float
    attractions: float
    buffer: float
    total: float
    budget: float
    within_budget: bool
    remaining: float = Field(..., description="Positive = left over, negative = over budget.")


class Itinerary(BaseModel):
    trip_summary: TripRequest
    transport_plan: Optional[TransportPlan] = None
    weather: list[WeatherDay] = Field(default_factory=list)
    daily_plan: list[DayPlan] = Field(default_factory=list)
    budget_breakdown: Optional[BudgetBreakdown] = None
    notes: list[str] = Field(default_factory=list)

# ---------------------------------------------------------------------------
# What the planner LLM returns (validated and completed by Python afterwards)
# ---------------------------------------------------------------------------
class PlannedActivity(BaseModel):
    time_of_day: TimeOfDay
    attraction: str = Field(..., description="EXACT name from the attractions list, or 'FREE' for free time, rest or travel.")
    plan_b: Optional[str] = Field(
        None, description="EXACT name of an INDOOR attraction from the list. Required for outdoor/mixed "
                          "activities on CAUTION, HAZARD or UNKNOWN days; otherwise null.")
    note: str = Field("", description="One short practical tip.")


class PlannedDay(BaseModel):
    day_number: int
    theme: str = Field("", description="Short title for the day, e.g. 'Forts and bazaars'.")
    activities: list[PlannedActivity]


class PlanDraft(BaseModel):
    intro: str = Field("", description="One or two friendly sentences introducing the plan.")
    days: list[PlannedDay]
    tips: list[str] = Field(default_factory=list, description="Up to 3 short tips based only on the provided facts.")