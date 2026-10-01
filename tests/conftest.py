"""Shared test setup: intake tests use a fake geocoder so they never hit the network."""
import pytest

from src.agent.schemas import GeoPlace
from src.tools.common import ServiceUnavailableError
from src.tools.geo import PlaceNotFoundError


def fake_geocode(name: str) -> GeoPlace:
    if name.lower().startswith("xyz"):
        raise PlaceNotFoundError(name)
    if name.lower() == "offline":
        raise ServiceUnavailableError("The location service is unavailable right now. Please try again shortly.")
    return GeoPlace(name=name.title(), latitude=20.0, longitude=75.0,
                    country="India", country_code="IN", admin1="Test State")


@pytest.fixture(autouse=True)
def _offline_geocoder(monkeypatch):
    monkeypatch.setattr("src.agent.intake.geocode", fake_geocode)