"""Central configuration loaded from .env. No secrets are hardcoded here."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = two levels above src/utils/
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


def _get(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _is_real_key(value: str) -> bool:
    """Treat empty values and the .env.example placeholders as 'not configured'."""
    return bool(value) and not value.lower().startswith(("your_", "paste_"))


class Settings:
    def __init__(self) -> None:
        # LLM
        self.google_api_key = _get("GOOGLE_API_KEY")
        self.gemini_model = _get("GEMINI_MODEL", "gemini-2.5-flash")
        self.llm_timeout = float(_get("LLM_TIMEOUT", "30"))

        # External APIs
        self.ors_api_key = _get("ORS_API_KEY")
        self.geoapify_api_key = _get("GEOAPIFY_API_KEY")

        # App
        self.host = _get("APP_HOST", "127.0.0.1")
        self.port = int(_get("APP_PORT", "8000"))
        self.log_level = _get("LOG_LEVEL", "INFO").upper()
        self.sqlite_path = BASE_DIR / _get("SQLITE_PATH", "storage/tripmind.sqlite")
        self.request_timeout = float(_get("REQUEST_TIMEOUT", "10"))
        self.default_country_code = _get("DEFAULT_COUNTRY_CODE", "IN").upper()

    def key_status(self) -> dict:
        """Which services are configured. Never returns the keys themselves."""
        return {
            "gemini": _is_real_key(self.google_api_key),
            "openrouteservice": _is_real_key(self.ors_api_key),
            "geoapify": _is_real_key(self.geoapify_api_key),
            "open_meteo": True,  # no key required
        }


settings = Settings()