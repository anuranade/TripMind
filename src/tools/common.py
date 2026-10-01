"""Shared helpers for tools: HTTP with timeouts and user-safe errors."""
from __future__ import annotations

import logging

import requests

from src.utils.config import settings

logger = logging.getLogger("tripmind.tools")

HEADERS = {"User-Agent": "TripMind/0.1 (university project)"}


class ToolError(Exception):
    """An error whose message is safe to show to the user."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


class ServiceUnavailableError(ToolError):
    """External API timed out, failed, or returned garbage."""


def _describe(exc: Exception) -> str:
    """Short technical description for logs — never includes URLs (which may contain API keys)."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


def http_get_json(url: str, params: dict | None = None, *, service: str = "external service",
                  timeout: float | None = None) -> dict:
    try:
        resp = requests.get(url, params=params, headers=HEADERS,
                            timeout=timeout or settings.request_timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.Timeout as exc:
        logger.warning("%s request timed out (%s)", service, _describe(exc))
        raise ServiceUnavailableError(f"The {service} took too long to respond. Please try again.") from exc
    except (requests.RequestException, ValueError) as exc:
        logger.warning("%s request failed (%s)", service, _describe(exc))
        raise ServiceUnavailableError(f"The {service} is unavailable right now. Please try again shortly.") from exc