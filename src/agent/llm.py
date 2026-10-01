"""Gemini chat model factory."""
import logging
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils.config import settings

logger = logging.getLogger("tripmind.llm")


class LLMNotConfiguredError(RuntimeError):
    """Raised when no usable Gemini key is configured."""


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    if not settings.key_status()["gemini"]:
        raise LLMNotConfiguredError("GOOGLE_API_KEY is missing or still a placeholder in .env")
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
        timeout=settings.llm_timeout,
        max_retries=2,
    )


def message_text(msg) -> str:
    """Plain text from an AI message (content may be a string or a list of parts)."""
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        parts.append(part.get("text", "") if isinstance(part, dict) else str(part))
    return "".join(parts)


# Smoke test:  python -m src.agent.llm
if __name__ == "__main__":
    try:
        reply = get_llm().invoke("Reply with exactly: TripMind is online.")
        print("Model:", settings.gemini_model)
        print("Reply:", message_text(reply))
    except LLMNotConfiguredError as e:
        print("Not configured:", e)
    except Exception as e:  # noqa: BLE001
        print(f"LLM call failed: {type(e).__name__}: {e}")