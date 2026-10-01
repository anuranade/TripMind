// TripMind frontend — chat, trip confirmation, planning progress and itinerary cards.
const chatEl = document.getElementById("chat");
const resultsEl = document.getElementById("results");
const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const statusPill = document.getElementById("status-pill");

const AFFIRM = /^(yes|yeah|yep|y|ok|okay|sure|confirm(ed)?|go ahead|looks good|sounds good|proceed|let'?s go|plan it|do it)\b/i;
const STEPS = [
  ["validated", "Trip details validated"],
  ["located", "Cities located"],
  ["weather", "Checking weather"],
  ["transport", "Comparing transport"],
  ["attractions", "Finding attractions"],
  ["budget", "Calculating budget"],
  ["itinerary", "Building your itinerary"],
];

let sessionId = newId();
let stage = "collecting";
let busy = false;

// ---------- Helpers ----------
function newId() {
  return window.crypto && crypto.randomUUID
    ? crypto.randomUUID()
    : `s-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node; // text is always inserted as text, never as HTML
}

const inrFmt = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });
const inr = (n) => inrFmt.format(n || 0);
const toDate = (iso) => new Date(`${iso}T00:00:00`);
const fmtDay = (iso) => toDate(iso).toLocaleDateString("en-IN", { day: "numeric", month: "short" });
const fmtWeekday = (iso) => toDate(iso).toLocaleDateString("en-IN", { weekday: "short" });
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : "");
const deg = (v) => (v == null ? "–" : `${Math.round(v)}°`);
function hoursLabel(hrs) {
  const total = Math.round(hrs * 60);
  const hh = Math.floor(total / 60), mm = total % 60;
  return hh ? `${hh} h ${String(mm).padStart(2, "0")} min` : `${mm} min`;
}
const scrollTo = (node) => node.scrollIntoView({ behavior: "smooth", block: "end" });

// ---------- API ----------
async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, ...body }),
  });
  let data = {};
  try { data = await res.json(); } catch { /* ignore */ }
  if (!res.ok) throw new Error(data.error || "Something went wrong. Please try again.");
  return data;
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error();
    statusPill.textContent = "Online";
    statusPill.className = "status-pill status-ok";
  } catch {
    statusPill.textContent = "Offline";
    statusPill.className = "status-pill status-error";
  }
}

// ---------- Chat UI ----------
function addBubble(role, text, ...extra) {
  const b = h("div", { class: `bubble ${role}` }, text ? h("div", { class: "bubble-text" }, text) : null, ...extra);
  chatEl.append(b);
  scrollTo(b);
  return b;
}

function showTyping() {
  const b = h("div", { class: "bubble assistant typing" }, h("span"), h("span"), h("span"));
  chatEl.append(b);
  scrollTo(b);
  return { finish() { b.remove(); } };
}

function showProgress() {
  const items = STEPS.map(([key, label]) =>
    h("li", { class: "step", "data-key": key }, h("span", { class: "step-icon" }), h("span", { class: "step-label" }, label)));
  const title = h("div", { class: "progress-title" }, "Planning your trip…");
  const b = h("div", { class: "bubble assistant card-bubble" }, title, h("ul", { class: "steps" }, items));
  chatEl.append(b);
  scrollTo(b);

  let i = 0;
  items[0].classList.add("running");
  const timer = setInterval(() => {
    if (i >= items.length - 1) return;
    items[i].className = "step done";
    i += 1;
    items[i].classList.add("running");
  }, 2500);

  return {
    finish(steps, failed) {
      clearInterval(timer);
      if (steps && steps.length) {
        const byKey = Object.fromEntries(steps.map((s) => [s.key, s]));
        items.forEach((li) => {
          const s = byKey[li.dataset.key];
          li.className = `step ${s ? s.status : "done"}`;
          if (s) li.querySelector(".step-label").textContent = s.label;
        });
        title.textContent = "Plan ready";
      } else if (failed) {
        items[i].className = "step failed";
        title.textContent = "Planning stopped";
      } else {
        b.remove(); // no planning happened after all
      }
    },
  };
}

async function run(call, planning) {
  busy = true;
  sendBtn.disabled = true;
  const waiting = planning ? showProgress() : showTyping();
  try {
    const data = await call();
    waiting.finish(data.steps);
    handle(data);
  } catch (err) {
    waiting.finish(null, true);
    addBubble("assistant error", err.message);
  } finally {
    busy = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

async function send(message) {
  if (busy) return;
  document.body.classList.add("started");
  addBubble("user", message);
  const planning = stage === "awaiting_confirmation" && AFFIRM.test(message.trim());
  await run(() => api("/api/chat", { message }), planning);
}

async function confirmTrip(buttons) {
  if (busy) return;
  buttons.forEach((b) => (b.disabled = true));
  addBubble("user", "Confirm trip");
  await run(() => api("/api/confirm", {}), true);
}

function handle(data) {
  stage = data.stage || stage;
  if (data.clear_results) {
    resultsEl.replaceChildren();
    resultsEl.hidden = true;
  }
  if (data.itinerary) {
    addBubble("assistant", data.text);
    renderItinerary(data);
    input.placeholder = "Ask for changes — e.g. make day 2 more relaxed, or use the bus instead";
    return;
  }
  if (data.summary && data.stage === "awaiting_confirmation") {
    addBubble("assistant card-bubble", data.text.split("\n")[0], summaryCard(data.summary));
    return;
  }
  const warn = (data.errors || []).length ? " warn" : "";
  addBubble(`assistant${warn}`, data.text);
}

// ---------- Summary card ----------
const fact = (label, value) =>
  h("div", {}, h("div", { class: "fact-label" }, label), h("div", { class: "fact-value" }, String(value)));

function summaryCard(s) {
  const confirmBtn = h("button", { class: "btn-primary", type: "button" }, "Confirm trip");
  const editBtn = h("button", { class: "btn-ghost", type: "button" }, "Edit details");
  confirmBtn.addEventListener("click", () => confirmTrip([confirmBtn, editBtn]));
  editBtn.addEventListener("click", () => {
    input.placeholder = "Tell me what to change, e.g. make it 3 people";
    input.focus();
  });
  return h("div", { class: "summary-card" },
    h("div", { class: "route" }, s.origin, h("span", { class: "arrow" }, "→"), s.destination),
    h("div", { class: "route-sub" }, `${s.origin_label} → ${s.destination_label}`),
    h("div", { class: "facts" },
      fact("Dates", `${s.dates_label} · ${s.days} days`),
      fact("Travellers", s.travellers),
      fact("Budget", s.budget_label),
      fact("Priority", cap(s.priority))),
    s.interests.length ? h("div", { class: "tags" }, s.interests.map((i) => h("span", { class: "tag" }, i))) : null,
    h("div", { class: "actions" }, confirmBtn, editBtn));
}

// ---------- Itinerary cards ----------
function card(title, tag, ...children) {
  return h("section", { class: "card" },
    h("div", { class: "card-head" }, h("h2", { class: "card-title" }, title), tag ? h("span", { class: "card-tag" }, tag) : null),
    ...children);
}

const badge = (status) =>
  h("span", { class: `badge ${status.toLowerCase()}` }, status === "UNKNOWN" ? "NO FORECAST" : status);

const metric = (label, value) =>
  h("div", {}, h("div", { class: "metric-label" }, label), h("div", { class: "metric-value" }, value));

function overviewCard(s, x) {
  const agentTools = (x.tools_used || []).filter((t) => t.source === "agent").map((t) => t.name);
  const autoTools = (x.tools_used || []).filter((t) => t.source === "auto").map((t) => t.name);
  let toolsLine = agentTools.length ? `Tools chosen by the agent: ${[...new Set(agentTools)].join(", ")}.` : "";
  if (autoTools.length) toolsLine += ` Auto-completed: ${[...new Set(autoTools)].join(", ")}.`;
  return h("section", { class: "card overview" },
    h("div", { class: "card-head" }, h("h2", { class: "card-title" }, "Trip overview")),
    h("div", { class: "route" }, s.origin, h("span", { class: "arrow" }, "→"), s.destination),
    h("div", { class: "route-sub" }, `${s.dates_label} · ${s.days} days`),
    h("div", { class: "facts" },
      fact("Travellers", s.travellers), fact("Budget", s.budget_label),
      fact("Priority", cap(s.priority)), fact("Interests", s.interests.join(", ") || "—")),
    x.intro ? h("p", { class: "intro" }, x.intro) : null,
    toolsLine ? h("p", { class: "fine" }, toolsLine) : null);
}

function transportCard(tp, x) {
  const r = tp.recommended;
  return card("Transport", "Estimated fares",
    h("div", { class: "mode" }, r.mode),
    h("div", { class: "metrics" },
      metric("Est. cost (round trip)", inr(r.estimated_cost)),
      metric("Travel time", `~${hoursLabel(r.travel_time_hours)} each way`),
      metric("Convenience", `${r.convenience_score}/10`)),
    h("p", { class: "reason" }, tp.reason),
    tp.alternatives.length
      ? h("div", { class: "alts" }, h("div", { class: "alts-title" }, "Alternatives"),
          tp.alternatives.map((a) => h("div", { class: "alt" }, h("span", {}, a.mode),
            h("span", { class: "muted" }, `${inr(a.estimated_cost)} · ~${hoursLabel(a.travel_time_hours)}`))))
      : null,
    x.route ? h("p", { class: "fine" }, `Road distance ${Math.round(x.route.distance_km)} km · source: ${x.route.source}`) : null);
}

function weatherCard(days) {
  return card("Weather", "Open-Meteo forecast",
    h("div", { class: "wx-grid" }, days.map((d) =>
      h("div", { class: "wx" },
        h("div", { class: "wx-date" }, fmtDay(d.date).toUpperCase()),
        h("div", { class: "wx-temp" }, d.forecast_available ? deg(d.temp_max_c) : "—"),
        d.forecast_available ? h("div", { class: "muted" }, `Low ${deg(d.temp_min_c)}`) : null,
        h("div", { class: "wx-cond" }, d.condition || "No forecast yet"),
        badge(d.status),
        h("div", { class: "wx-reason" }, d.reason)))));
}

function slotRow(a) {
  const free = a.name === "Free time";
  return h("div", { class: `slot${free ? " free" : ""}` },
    h("div", { class: "slot-time" }, cap(a.time_of_day)),
    h("div", {},
      h("div", { class: "slot-name" }, a.name),
      free ? null : h("div", { class: "slot-meta" },
        a.category ? h("span", {}, a.category) : null,
        a.setting !== "unknown" ? h("span", { class: `pill ${a.setting}` }, a.setting) : null,
        a.approx_cost ? h("span", {}, `~${inr(a.approx_cost)}`) : null),
      a.notes ? h("div", { class: "slot-note muted" }, a.notes) : null,
      a.plan_b ? h("div", { class: "planb" }, h("strong", {}, "Plan B: "), a.plan_b) : null));
}

function dayCard(d) {
  return h("article", { class: "card" },
    h("div", { class: "day-head" },
      h("div", {}, h("div", { class: "day-num" }, `Day ${d.day_number}`),
        h("div", { class: "day-date muted" }, `${fmtWeekday(d.date)}, ${fmtDay(d.date)}`)),
      badge(d.weather_status)),
    d.theme ? h("div", { class: "day-theme" }, d.theme) : null,
    h("div", { class: "slots" }, d.activities.map(slotRow)));
}

function budgetCard(b, x) {
  const ok = b.within_budget;
  const pct = Math.min(100, (b.total / b.budget) * 100);
  const rows = [["Transport", b.transport], ["Stay", b.stay], ["Food", b.food],
    ["Local transport", b.local_transport], ["Attractions", b.attractions], ["Buffer (10%)", b.buffer]];
  return card("Budget", "Estimates",
    h("div", { class: "budget-top" }, h("div", { class: "budget-total" }, inr(b.total)),
      h("div", { class: "muted" }, `of ${inr(b.budget)}`)),
    h("div", { class: `bar ${ok ? "ok" : "over"}` }, h("span", { style: `width:${pct}%` })),
    h("div", { class: `budget-status ${ok ? "ok" : "over"}` },
      ok ? `Within budget · ${inr(b.remaining)} remaining` : `Over budget by ${inr(-b.remaining)}`),
    h("table", { class: "budget-table" },
      rows.map(([label, v]) => h("tr", {}, h("td", {}, label), h("td", {}, inr(v)))),
      h("tr", { class: "total" }, h("td", {}, "Total"), h("td", {}, inr(b.total)))),
    x.per_person ? h("p", { class: "fine" }, `About ${inr(x.per_person)} per person`) : null);
}

function versionBar(versions) {
  if (!versions || versions.length < 2) return null;
  return h("div", { class: "versions" },
    h("span", { class: "versions-label" }, "Plan versions"),
    versions.map((v) => h("button", {
      class: `version${v.active ? " active" : ""}`,
      type: "button",
      title: v.label,
      onclick: () => { if (!v.active) selectVersion(v.number); },
    }, `v${v.number} · ${v.label}`)));
}

async function selectVersion(number) {
  if (busy) return;
  await run(() => api("/api/version", { version: number }), false);
}

function renderItinerary(data) {
  const it = data.itinerary, x = data.extras || {}, s = data.summary;
  const cards = [
    versionBar(data.versions),
    s ? overviewCard(s, x) : null,
    it.transport_plan ? transportCard(it.transport_plan, x) : null,
    it.weather.length ? weatherCard(it.weather) : null,
    h("div", { class: "days" }, it.daily_plan.map(dayCard)),
    it.budget_breakdown ? budgetCard(it.budget_breakdown, x) : null,
    it.notes.length ? card("Notes", "Estimates & limitations", h("ul", { class: "notes" }, it.notes.map((n) => h("li", {}, n)))) : null,
  ].filter(Boolean);
  resultsEl.replaceChildren(...cards);
  resultsEl.hidden = false;
  setTimeout(() => resultsEl.scrollIntoView({ behavior: "smooth", block: "start" }), 150);
}

// ---------- Events ----------
form.addEventListener("submit", (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message || busy) return;
  input.value = "";
  input.style.height = "auto";
  send(message);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
});

document.querySelectorAll(".chip").forEach((chip) =>
  chip.addEventListener("click", () => {
    input.value = chip.dataset.prompt;
    input.dispatchEvent(new Event("input"));
    input.focus();
  }));

document.getElementById("new-trip").addEventListener("click", async () => {
  if (busy) return;
  try { await api("/api/reset", {}); } catch { /* ignore */ }
  sessionId = newId();
  stage = "collecting";
  chatEl.replaceChildren();
  resultsEl.replaceChildren();
  resultsEl.hidden = true;
  document.body.classList.remove("started");
  input.value = "";
  input.placeholder = "Plan a 3-day Jaipur trip from Mumbai for 2 people under ₹25,000…";
  input.focus();
});

checkHealth();