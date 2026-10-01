from datetime import date, timedelta

import pytest

from src.tools.budget import calculate_budget, estimate_budget, find_destination
from tests.test_transport import LONG


def calc(**kw):
    base = dict(destination="Jaipur", num_days=3, travellers=2, budget=25000, style="budget",
                transport_cost=7500, transport_label="Train (3AC)")
    base.update(kw)
    return calculate_budget(**base)


# ---------- Pure calculation ----------
def test_test1_breakdown():
    b, _ = calc()
    assert (b.transport, b.stay, b.food, b.local_transport, b.attractions) == (7500, 3000, 3600, 1500, 2100)
    assert b.buffer == 1770
    assert b.total == 19470
    assert b.within_budget and b.remaining == 5530


def test_total_is_parts_plus_ten_percent_buffer():
    b, _ = calc(travellers=3, style="mid", destination="Goa")
    parts = b.transport + b.stay + b.food + b.local_transport + b.attractions
    assert b.total == parts + b.buffer
    assert abs(b.buffer - parts * 0.10) <= 5


def test_over_budget():
    b, _ = calc(budget=5000)
    assert not b.within_budget and b.remaining == -14470


def test_day_trip_has_no_stay():
    b, notes = calc(num_days=1)
    assert b.stay == 0 and any("day trip" in n for n in notes)


def test_rooms_round_up():
    b, _ = calc(travellers=3)
    assert b.stay == 2 * 2 * 1500  # 2 rooms x 2 nights


def test_comfort_costs_more_than_budget():
    assert calc(style="comfort")[0].total > calc(style="budget")[0].total


def test_planned_attraction_fees_used():
    b, notes = calc(attraction_fees_per_person=[200, 100, 0])
    assert b.attractions == 600 and any("3 planned" in n for n in notes)


def test_unknown_city_uses_default_and_says_so():
    row, found = find_destination("Xyzabad")
    assert not found and row["city"] == "_default"
    _, notes = calc(destination="Xyzabad")
    assert any("national average" in n for n in notes)


def test_alias_lookup():
    row, found = find_destination("Bangalore")
    assert found and row["city"] == "Bengaluru"


# ---------- Tool (route fixed, no network) ----------
S = date.today() + timedelta(days=15)


@pytest.fixture
def fixed_route(monkeypatch):
    monkeypatch.setattr("src.tools.budget.get_route", lambda *a, **k: LONG)


def run(**kw):
    base = {"destination": "Jaipur", "origin_latitude": 19.076, "origin_longitude": 72.878,
            "destination_latitude": 26.912, "destination_longitude": 75.787,
            "start_date": S.isoformat(), "end_date": (S + timedelta(days=2)).isoformat(),
            "travellers": 2, "budget": 25000, "travel_priority": "budget"}
    base.update(kw)
    return estimate_budget.invoke(base)


def test_tool_uses_recommended_transport(fixed_route):
    out = run()
    assert out["ok"] and out["transport_mode"] == "Train (3AC)"
    assert out["breakdown"]["total"] == 19470
    assert out["within_budget"] and "within" in out["status_message"]
    assert out["suggestions"] == []


def test_tool_respects_chosen_mode(fixed_route):
    out = run(transport_mode="bus")
    assert out["transport_mode"] == "Bus (AC sleeper)"
    assert out["breakdown"]["transport"] == 6840


def test_tool_rejects_infeasible_mode(fixed_route):
    out = run(transport_mode="helicopter")
    assert out["ok"] is False and "Options:" in out["error"]


def test_tool_over_budget_explains_and_suggests(fixed_route):
    out = run(budget=5000)
    assert out["within_budget"] is False
    assert "OVER" in out["status_message"] and "₹14,470" in out["status_message"]
    assert any("Bus" in s for s in out["suggestions"])


def test_convenience_steps_down_to_fit_budget(fixed_route):
    out = run(travel_priority="convenience")  # ₹25k can't afford comfort hotels plus a flight
    assert out["transport_mode"] == "Flight" and out["spending_style"] == "budget"


def test_convenience_uses_comfort_when_affordable(fixed_route):
    out = run(travel_priority="convenience", budget=200000)
    assert out["spending_style"] == "comfort" and out["within_budget"]

@pytest.mark.parametrize("kw", [{"start_date": "10 Oct"}, {"travel_priority": "luxury"},
                                {"budget": 0}, {"travellers": 12}])
def test_tool_rejects_bad_input(fixed_route, kw):
    assert run(**kw)["ok"] is False