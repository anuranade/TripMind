"""TripMind — FastAPI backend."""
import logging

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.agent.session import TripMindSession
from src.utils.config import BASE_DIR, settings

logging.basicConfig(level=settings.log_level,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
logger = logging.getLogger("tripmind")

app = FastAPI(title="TripMind", version="0.2.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# In-memory sessions (SQLite persistence is a future objective)
SESSIONS: dict[str, TripMindSession] = {}
MAX_SESSIONS = 100


def new_session() -> TripMindSession:
    return TripMindSession()


def get_session(session_id: str) -> TripMindSession:
    session = SESSIONS.get(session_id)
    if session is None:
        if len(SESSIONS) >= MAX_SESSIONS:
            SESSIONS.pop(next(iter(SESSIONS)))  # drop the oldest
        session = SESSIONS[session_id] = new_session()
    return session


class SessionRequest(BaseModel):
    session_id: str = Field(..., min_length=8, max_length=64)


class ChatRequest(SessionRequest):
    message: str = Field(..., min_length=1, max_length=2000)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"app_name": "TripMind"})


@app.get("/api/health")
async def health():
    return {"status": "ok", "app": "TripMind", "version": app.version,
            "services": settings.key_status(), "active_sessions": len(SESSIONS)}


# Plain `def` (not async): planning is slow and blocking, so FastAPI runs it in a worker thread.
@app.post("/api/chat")
def chat(req: ChatRequest):
    return {"session_id": req.session_id, **get_session(req.session_id).handle(req.message)}


@app.post("/api/confirm")
def confirm(req: SessionRequest):
    return {"session_id": req.session_id, **get_session(req.session_id).confirm()}


@app.post("/api/reset")
def reset(req: SessionRequest):
    SESSIONS.pop(req.session_id, None)
    return {"ok": True}
class VersionRequest(SessionRequest):
    version: int = Field(..., ge=1, le=50)


@app.post("/api/version")
def select_version(req: VersionRequest):
    return {"session_id": req.session_id, **get_session(req.session_id).select_version(req.version)}


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500,
                        content={"error": "Something went wrong on our side. Please try again."})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=settings.host, port=settings.port, reload=True)