# TripMind — Agentic AI Travel Planner

TripMind is an AI travel consultant built with **LangChain** and **Google Gemini**. Describe a trip in plain English, and TripMind asks only for what's missing, confirms the details, then uses an AI agent with real tools to build a day-by-day itinerary with weather, transport, attractions and a budget.

> *"Plan a Jaipur trip from Mumbai from 10 Oct to 12 Oct for 2 people with a ₹25,000 budget. We like history and food. Budget is the priority."*

**Demo video:** [add link here]

---

## Features

- **Understands natural language** — extracts origin, destination, dates, travellers, budget, priority and interests into a structured trip request.
- **Asks only for missing details** — then shows a summary and waits for confirmation before any expensive tool calls.
- **Validates input** — checks dates, traveller count and budget, and verifies that both cities exist.
- **Agentic research** — Gemini decides which of 7 tools to call: geocoding, weather forecast, weather hazards, routing, transport comparison, attraction search and budget estimation.
- **Weather-aware planning** — every day is rated SAFE / CAUTION / HAZARD by fixed rules, and outdoor activities on risky days get an indoor Plan B.
- **Transport ranking by priority** — flight, train, bus, cab and self-drive ranked for budget, time or convenience.
- **Full budget breakdown** — transport, stay, food, local transport, entry fees and a 10% buffer, compared against the user's budget.
- **Refinement with plan versions** — ask "make day 2 more relaxed" or "use the bus instead"; every version is kept, and the original can be restored with one click.
- **No hallucinated facts** — weather, distances, places and costs come only from tools and data files; estimates are clearly labelled.

## How it works

TripMind uses a **hybrid design**: the LLM understands, decides and organises; Python validates, calculates and enforces rules.

```
Browser (HTML/CSS/JS)  →  FastAPI backend  →  TripMind session
   1. Intake      Gemini structured output → TripRequest → validation + geocoding → confirm
   2. Agent       Gemini + 7 tools (bind_tools); the model chooses the tools; a safety net fills gaps
   3. Planner     Gemini arranges the days; Python verifies places, adds Plan Bs, recalculates the budget
   4. Follow-ups  Router LLM: refine plan / change trip / new trip / question; each plan saved as a version
```

## LangChain components used

| Component | Where | Role |
|---|---|---|
| `ChatGoogleGenerativeAI` | `src/agent/llm.py` | Gemini chat model behind every LLM step |
| `ChatPromptTemplate` | `src/agent/prompts.py` | 6 prompts: extraction, agent, planner, refine, router, answer |
| `MessagesPlaceholder` | `src/agent/prompts.py` | Injects chat history and the agent's tool scratchpad |
| `with_structured_output()` | extractor, planner, followup | Pydantic outputs: `TripExtraction`, `PlanDraft`, `FollowUpIntent` |
| `@tool` | `src/tools/` | 7 Python functions exposed to the agent |
| `bind_tools()` + `tool_calls` | `src/agent/agent.py` | Gemini decides which tools to call, and with what inputs |
| `ToolMessage` | `src/agent/agent.py` | Feeds each tool result back to Gemini in the agent loop |
| LCEL pipe (`prompt \| llm`) | all chains | Composes prompts and models into runnable chains |
| `HumanMessage` / `AIMessage` | intake, session | Conversation memory within a session |

## Tech stack

Python 3.12 · LangChain · Google Gemini · Pydantic · FastAPI · pandas · requests · python-dotenv · HTML/CSS/JavaScript · pytest

**External APIs:** Open-Meteo (geocoding and weather, no key) · OSRM (routing, no key) · Geoapify Places (attractions, free key)

## Project structure

```
tripmind/
├── app.py                  FastAPI app and API endpoints
├── requirements.txt
├── .env.example            Settings template (copy to .env)
├── src/
│   ├── agent/              Schemas, prompts, extraction, intake, agent, planner, follow-ups, session
│   ├── tools/              geo, weather, transport, places, budget (LangChain tools)
│   └── utils/config.py     Settings loaded from .env
├── data/                   transport_modes.csv, destinations.csv, country_cost_levels.csv, interest_categories.json
├── templates/index.html    Page layout
├── static/                 css/style.css, js/app.js
└── tests/                  Offline test suite (no API calls)
```

## Setup (Windows PowerShell)

Requires **Python 3.11 or 3.12**.

```powershell
git clone https://github.com/anuranade/TripMind.git
cd TripMind
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and add your own keys:

| Setting | Where to get it |
|---|---|
| `GOOGLE_API_KEY` | https://aistudio.google.com/apikey |
| `GEMINI_MODEL` | A model your key can use, e.g. `gemini-3.5-flash-lite` |
| `GEOAPIFY_API_KEY` | https://myprojects.geoapify.com/ (free tier) |

`ORS_API_KEY` is optional; without it, routing uses OSRM.

## Run

```powershell
python app.py
```

Open **http://127.0.0.1:8000**.

Run the tests (offline, no API calls):

```powershell
pytest -v
```

## Try these

| Test | Message |
|---|---|
| Complete request | `Plan a Jaipur trip from Mumbai from 10 Oct to 12 Oct for 2 people with a ₹25,000 budget. We like history and food. Budget is the priority.` |
| Missing details | `Plan a trip to Udaipur.` |
| Invalid city | `Plan a trip from Mumbai to Xyzabad next weekend for 2 people, ₹20,000, budget priority.` |
| Refinement | After a plan: `make day 2 more relaxed` or `use the bus instead` |
| Over budget | Any trip with a budget of `₹3,000` |

Use dates within about two weeks to see real weather forecasts.

## Limitations

- Fares, hotel prices and entry fees are **estimates** from reference data, not live prices.
- Weather forecasts cover about **16 days** ahead; later days show "no forecast yet" with a Plan B.
- Sessions are kept **in memory** and are lost when the server restarts.
- The Gemini **free tier has a small daily request limit**; one full plan uses several requests.
- Attraction coverage depends on OpenStreetMap data for each place.

## Future work

- Persistent memory with LangGraph's SQLite checkpointer
- Live fare and hotel price APIs
- Live progress updates streamed as each tool finishes
