"""The agent inspector: what each swarm agent is working on, what it was asked, what it did.

The swarm runner is a separate process, so nothing here can look inside an agent's turn by
itself. Instead every Worker keeps a record of its current iteration -- the assignment
(objective, mode, parent, idea), the full system and iteration prompts it was sent, every
tool call with its arguments and a result snippet, each chat request's token counts, and what
it submitted -- and posts it here, fire-and-forget, whenever it changes. This module keeps the
last KEEP_RECORDS records per agent in memory, flushed every FLUSH_S seconds to a small sqlite
file so a restart of the control plane does not blank the inspector.

Forecasters are the other half. Every forecast goes through ``TsManager.forecast``, which
notes each call here (``note_forecast``). A feature build is dozens of batched calls from one
HTTP request, so calls are grouped by the request that caused them; ``CallerMiddleware`` tags
each request with who made it (the runner sends ``X-FreeSwarm-Agent``) and, for the forecast
endpoints, the recipe it asked for (columns, covariates, horizon...), which the forecaster
itself never sees -- it only receives arrays of numbers.

Nothing here may slow down or fail the work it observes: every hook swallows its own errors.
"""

from __future__ import annotations

import asyncio
import collections
import contextvars
import itertools
import json
import logging
import sqlite3
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger("freetoken.agent_activity")
router = APIRouter(tags=["agents"])

DB_PATH = Path(__file__).resolve().parent.parent / "agent_activity.sqlite3"
KEEP_RECORDS = 5
FLUSH_S = 5.0
# An agent not heard from in this long is dropped when the store is loaded from disk.
FORGET_AFTER_S = 7 * 86400.0
# A record is the runner's own truncated copy; this only stops a runaway post.
MAX_RECORD_BYTES = 2_000_000
FORECAST_RING = 120
# Calls from one request more than this far apart start a new row (a long feature build
# keeps one row; a later request that happens to reuse nothing gets its own).
GROUP_GAP_S = 120.0

_lock = threading.Lock()
# "<project_id>|<agent>" -> {agent, model, role, slot, project_id, updated_at, records: [...]}
_agents: dict[str, dict] = {}
_dirty: set[str] = set()
_loaded = False
_last_flush = 0.0
_conn: sqlite3.Connection | None = None


# =======================================================================================
# Persistence
# =======================================================================================
def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
        _conn.execute("CREATE TABLE IF NOT EXISTS agents (key TEXT PRIMARY KEY, doc TEXT NOT NULL, "
                      "updated_at REAL NOT NULL)")
        _conn.commit()
    return _conn


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        _loaded = True
        try:
            rows = _db().execute("SELECT key, doc FROM agents WHERE updated_at > ?",
                                 (time.time() - FORGET_AFTER_S,)).fetchall()
        except sqlite3.Error as exc:
            logger.warning("agent activity: could not load %s: %s", DB_PATH, exc)
            return
        for key, doc in rows:
            try:
                _agents.setdefault(key, json.loads(doc))
            except ValueError:
                continue


def flush() -> None:
    """Write the agents that changed since the last flush. Cheap: a few rows per flush."""
    global _last_flush
    with _lock:
        docs = [(k, json.dumps(_agents[k], default=str), _agents[k].get("updated_at") or time.time())
                for k in _dirty if k in _agents]
        _dirty.clear()
        _last_flush = time.time()
    if not docs:
        return
    try:
        conn = _db()
        conn.executemany("INSERT INTO agents (key, doc, updated_at) VALUES (?, ?, ?) "
                         "ON CONFLICT(key) DO UPDATE SET doc=excluded.doc, updated_at=excluded.updated_at", docs)
        conn.commit()
    except sqlite3.Error as exc:  # the inspector is a view; losing a flush loses nothing else
        logger.warning("agent activity: flush failed: %s", exc)


# =======================================================================================
# The store
# =======================================================================================
def record(post: dict) -> dict:
    """Upsert one agent's iteration record (matched by record id); keep the newest few."""
    _ensure_loaded()
    rec = dict(post.get("record") or {})
    if not rec.get("id"):
        raise ValueError("record.id is required")
    key = f"{post.get('project_id') or ''}|{post['agent']}"
    now = time.time()
    rec["received_at"] = now
    with _lock:
        a = _agents.get(key)
        if a is None:
            a = _agents[key] = {"records": []}
        a.update(agent=post["agent"], model=post.get("model") or post["agent"], role=post.get("role") or "search",
                 slot=post.get("slot") or 0, project_id=post.get("project_id"), updated_at=now)
        recs = [r for r in a["records"] if r.get("id") != rec["id"]] + [rec]
        recs.sort(key=lambda r: r.get("started_at") or 0)
        a["records"] = recs[-KEEP_RECORDS:]
        _dirty.add(key)
    return {"ok": True, "records": len(a["records"])}


def _summary(a: dict, now: float) -> dict:
    cur = a["records"][-1] if a["records"] else None
    out = {k: a.get(k) for k in ("agent", "model", "role", "slot", "project_id", "updated_at")}
    out["seconds_since_update"] = round(now - (a.get("updated_at") or now), 1)
    if cur:
        tools = [e for e in cur.get("timeline") or [] if e.get("kind") == "tool"]
        out["current"] = {
            "id": cur.get("id"), "mode": cur.get("mode"), "status": cur.get("status"),
            "objective": cur.get("objective"), "parent": cur.get("parent"), "idea": cur.get("idea"),
            "started_at": cur.get("started_at"), "ended_at": cur.get("ended_at"),
            "pending": cur.get("pending"), "tool_calls": len(tools),
            "last_tool": tools[-1].get("name") if tools else None, "tokens": cur.get("tokens"),
            "submissions": cur.get("submissions") or [],
        }
    return out


def agents(project_id: str | None = None) -> list[dict]:
    _ensure_loaded()
    now = time.time()
    with _lock:
        docs = [a for a in _agents.values() if not project_id or a.get("project_id") == project_id]
        return sorted((_summary(a, now) for a in docs), key=lambda s: (s["model"] or "", s["slot"] or 0))


def for_model(model: str, project_id: str | None = None) -> dict:
    """Everything the inspector shows for one model row: its agents (search slots and
    mentor) with full records, other agents' chats that used this model (audits and judging
    go to a DIFFERENT model on purpose), and -- for a forecaster -- its recent requests."""
    _ensure_loaded()
    now = time.time()
    with _lock:
        mine, peers = [], []
        for a in _agents.values():
            if project_id and a.get("project_id") != project_id:
                continue
            if a.get("model") == model:
                mine.append(json.loads(json.dumps(a, default=str)))
                continue
            for r in a.get("records") or []:
                used = [c for c in r.get("chats") or [] if c.get("model") == model]
                if used:
                    peers.append({"agent": a.get("agent"), "record_id": r.get("id"), "mode": r.get("mode"),
                                  "objective": r.get("objective"), "started_at": r.get("started_at"),
                                  "chats": len(used),
                                  "prompt_tokens": sum(int(c.get("prompt_tokens") or 0) for c in used),
                                  "completion_tokens": sum(int(c.get("completion_tokens") or 0) for c in used),
                                  "asked": [x for x in r.get("asked") or [] if x.get("model") == model][:3]})
    for a in mine:
        a["seconds_since_update"] = round(now - (a.get("updated_at") or now), 1)
    mine.sort(key=lambda a: (a.get("role") != "search", a.get("slot") or 0))
    peers.sort(key=lambda p: p.get("started_at") or 0, reverse=True)
    return {"model": model, "now": now, "agents": mine, "peer_calls": peers[:10],
            "forecasts": forecasts(model)}


def reset() -> None:
    """Forget everything in memory (tests)."""
    global _loaded
    with _lock:
        _agents.clear()
        _dirty.clear()
        _forecasts.clear()
        _pending_inputs.clear()
        _values.clear()
        _loaded = False


# =======================================================================================
# Forecast requests: who asked which forecaster for what
# =======================================================================================
_forecasts: collections.deque[dict] = collections.deque(maxlen=FORECAST_RING)
_rids = itertools.count(1)
# Set per HTTP request by CallerMiddleware; copied into tasks the request spawns.
_caller: contextvars.ContextVar[dict | None] = contextvars.ContextVar("forecast_caller", default=None)

# Request bodies worth reading for the recipe: small JSON that names columns.
_RECIPE_KEYS = ("column", "columns", "covariates", "calendar", "dataset", "horizon", "every", "context",
                "model", "bar", "name", "samples", "inputs", "target", "targets")


def _via(path: str) -> str:
    if path.endswith("/features"):
        return "forecast_feature"
    if path.endswith("/forecast") and "/objectives/" in path:
        return "forecast tool"
    if path.endswith("/python"):
        return "run_python (ft.forecast)"
    if "/candidates" in path:
        return "candidate scoring (ft.forecast)"
    if "/tslab" in path:
        return "Forecast Lab"
    if path.endswith("/ts/forecast"):
        return "forecast API"
    return path


def _objective_of(path: str) -> str | None:
    parts = path.strip("/").split("/")
    if "objectives" in parts:
        i = parts.index("objectives")
        if i + 1 < len(parts):
            return urllib.parse.unquote(parts[i + 1])
    return None


def _shape(payload: dict) -> dict:
    """What a forecast payload asked for, without its numbers."""
    out: dict[str, Any] = {"horizon": payload.get("horizon")}
    if payload.get("quantiles"):
        out["quantiles"] = len(payload["quantiles"])
    inputs = payload.get("inputs")
    if isinstance(inputs, list) and inputs:
        first = inputs[0] if isinstance(inputs[0], dict) else {}
        tgt = first.get("target")
        multi = isinstance(tgt, list) and bool(tgt) and isinstance(tgt[0], list)
        out.update(kind="covariates", anchors=len(inputs), targets=len(tgt) if multi else 1,
                   context=len(tgt[0] if multi else tgt or []),
                   past_covariates=sorted((first.get("past_covariates") or {}).keys()),
                   future_covariates=sorted((first.get("future_covariates") or {}).keys()))
        return out
    candles = payload.get("candles")
    if isinstance(candles, list) and candles:
        out.update(kind="candles", anchors=len(candles), context=len(candles[0] or []),
                   samples=payload.get("samples"), freq_seconds=payload.get("freq_seconds"))
        return out
    series = payload.get("series")
    if isinstance(series, list) and series:
        batch = isinstance(series[0], list)
        out.update(kind="series", anchors=len(series) if batch else 1,
                   context=len(series[0]) if batch else len(series))
    return out


def note_forecast(model: str, payload: dict, seconds: float | None, response: Any = None) -> None:
    """Called by TsManager.forecast after every call. Never raises."""
    try:
        caller = _caller.get() or {}
        shape = _shape(payload or {})
        status = getattr(response, "status_code", None)
        error = None
        if response is None:
            error = "no response (transport error)"
        elif status is not None and status >= 400:
            error = (getattr(response, "text", "") or "")[:300]
        now = time.time()
        rid = caller.get("rid")
        with _lock:
            # Input streams the builder described (note_inputs) before this call went out.
            described = _pending_inputs.pop((rid, model), None)
            if rid is not None:
                for e in reversed(_forecasts):
                    if e["rid"] == rid and e["model"] == model and now - e["last_at"] < GROUP_GAP_S:
                        if described:
                            e["inputs"] = (e.get("inputs") or []) + described
                            e["inputs"] = e["inputs"][-MAX_INPUT_DETAILS:]
                        e["calls"] += 1
                        e["anchors"] += int(shape.get("anchors") or 0)
                        e["seconds"] = round(e["seconds"] + (seconds or 0.0), 3)
                        e["last_at"] = now
                        e["in_flight"] = False
                        if error:
                            e["errors"] += 1
                            e["error"] = error
                        return
            _forecasts.append({
                "rid": rid if rid is not None else f"x{next(_rids)}", "model": model,
                "agent": caller.get("agent"), "via": _via(caller.get("path") or ""),
                "path": caller.get("path"), "objective_id": _objective_of(caller.get("path") or ""),
                "request": caller.get("recipe"), "shape": shape, "calls": 1,
                "anchors": int(shape.get("anchors") or 0), "seconds": round(seconds or 0.0, 3),
                "first_at": now - (seconds or 0.0), "last_at": now, "errors": 1 if error else 0, "error": error,
                "inputs": (described or [])[-MAX_INPUT_DETAILS:],
            })
    except Exception:  # noqa: BLE001 -- observing a forecast must never fail it
        logger.debug("note_forecast failed", exc_info=True)


# The forecaster only ever sees arrays of numbers; which column each array was, and which
# dates it spans, is known only to the code that built the request. The builders
# (objectives._build_*feature, forecast_by_name, the Forecast Lab) describe it here and it is
# attached to the request's row: as-of time(s), each input stream's role, source and date range.
MAX_INPUT_DETAILS = 4
_pending_inputs: collections.OrderedDict[tuple, list[dict]] = collections.OrderedDict()


def note_inputs(model: str, detail: dict) -> None:
    """Describe the input streams of the forecast(s) this request is about to make. Attached
    to the request's forecast row -- now if the row exists, else at its first call. Never raises."""
    try:
        caller = _caller.get() or {}
        rid = caller.get("rid")
        now = time.time()
        with _lock:
            if rid is not None:
                for e in reversed(_forecasts):
                    if e["rid"] == rid and e["model"] == model and now - e["last_at"] < GROUP_GAP_S:
                        e["inputs"] = ((e.get("inputs") or []) + [detail])[-MAX_INPUT_DETAILS:]
                        return
            _pending_inputs.setdefault((rid, model), []).append(detail)
            _pending_inputs.move_to_end((rid, model))
            while len(_pending_inputs) > 64:   # a request that never forecast leaves nothing behind for long
                _pending_inputs.popitem(last=False)
    except Exception:  # noqa: BLE001
        logger.debug("note_inputs failed", exc_info=True)


# The VALUES behind a request's sample anchors (forecast_values.ValueCapture): the context each
# stream sent, the forecast returned, and what followed (in-sample only). Up to MAX_VALUES_BYTES
# each, kept apart from the rows so the inspector's 2-second poll stays light: a row's input
# detail carries only a `values` summary ({id, anchors, bytes}); the numbers are fetched on click.
MAX_VALUES_BYTES = 200_000
VALUES_KEEP = 96
_values: collections.OrderedDict[str, dict] = collections.OrderedDict()


def note_values(model: str, detail: dict, payload: dict) -> str | None:
    """Attach captured values to the input detail they belong to (found by identity in the
    forecast rows or the not-yet-attached details). Returns the values id. Never raises."""
    try:
        import uuid

        size = len(json.dumps(payload, separators=(",", ":"), default=str))
        if size > MAX_VALUES_BYTES:
            logger.warning("forecast values for %s dropped: %d bytes > %d", model, size, MAX_VALUES_BYTES)
            return None
        vid = uuid.uuid4().hex[:16]
        summary = {"id": vid, "bytes": size,
                   "anchors": [{"sample": a.get("sample"), "as_of": a.get("as_of"), "note": a.get("note")}
                               for a in payload.get("anchors") or []]}
        with _lock:
            found = False
            lists = [e for e in _forecasts if e.get("model") == model]
            for e in lists:
                ins = e.get("inputs") or []
                if any(d is detail for d in ins):
                    # copy-on-write: readers hold shallow copies of the row
                    e["inputs"] = [{**d, "values": summary} if d is detail else d for d in ins]
                    found = True
            for key, ins in list(_pending_inputs.items()):
                if key[1] == model and any(d is detail for d in ins):
                    _pending_inputs[key] = [{**d, "values": summary} if d is detail else d for d in ins]
                    found = True
            if not found:
                return None
            _values[vid] = payload
            while len(_values) > VALUES_KEEP:
                _values.popitem(last=False)
        return vid
    except Exception:  # noqa: BLE001
        logger.debug("note_values failed", exc_info=True)
        return None


def values(vid: str) -> dict | None:
    with _lock:
        return _values.get(vid)


def forecasts(model: str | None = None, limit: int = 40) -> list[dict]:
    with _lock:
        rows = [dict(e) for e in _forecasts if model is None or e["model"] == model]
    rows.reverse()
    return rows[:limit]


class CallerMiddleware:
    """Tag each /api request with who made it, so a forecast can say who asked for it.

    Pure ASGI (no BaseHTTPMiddleware): it only sets a context variable and, for the small
    JSON bodies of the forecast endpoints, reads the recipe and replays the body untouched.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or not str(scope.get("path", "")).startswith("/api/"):
            return await self.app(scope, receive, send)
        path = scope["path"]
        agent = None
        for k, v in scope.get("headers") or []:
            if k == b"x-freeswarm-agent":
                agent = urllib.parse.unquote(v.decode("latin-1"))[:200]
                break
        caller: dict[str, Any] = {"rid": next(_rids), "agent": agent, "path": path}
        if scope.get("method") == "POST" and (path.endswith("/features") or
                                              (path.endswith("/forecast") and "/objectives/" in path)):
            messages, chunks, more = [], [], True
            while more:
                msg = await receive()
                messages.append(msg)
                if msg.get("type") != "http.request":
                    break
                chunks.append(msg.get("body", b""))
                more = msg.get("more_body", False)
            body = b"".join(chunks)
            if len(body) < 64_000:
                try:
                    doc = json.loads(body or b"{}")
                    caller["recipe"] = {k: doc[k] for k in _RECIPE_KEYS if doc.get(k) not in (None, "", [])}
                except ValueError:
                    pass
            pending = iter(messages)

            async def replay():
                try:
                    return next(pending)
                except StopIteration:
                    return await receive()

            receive = replay
        token = _caller.set(caller)
        try:
            return await self.app(scope, receive, send)
        finally:
            _caller.reset(token)


# =======================================================================================
# Routes (mounted under /api, so they carry the console's auth)
# =======================================================================================
class ActivityPost(BaseModel):
    agent: str = Field(..., max_length=300)
    model: str = Field(..., max_length=300)
    role: str = Field("search", max_length=40)
    slot: int = 0
    project_id: str | None = Field(None, max_length=200)
    record: dict[str, Any]


@router.post("/agents/activity")
async def post_activity(req: ActivityPost) -> dict:
    """From the swarm runner: one agent's current (or just finished) iteration record."""
    if len(json.dumps(req.record, default=str)) > MAX_RECORD_BYTES:
        raise HTTPException(status_code=413, detail="activity record too large")
    try:
        out = record(req.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    if time.time() - _last_flush > FLUSH_S:
        await asyncio.to_thread(flush)
    return out


@router.get("/agents/activity")
async def list_activity(project_id: str | None = Query(None)) -> dict:
    """One line per agent: what it is doing now. Prompts are left out; see /agents/activity/{model}."""
    return {"now": time.time(), "agents": agents(project_id)}


@router.get("/agents/forecasts")
async def list_forecasts(model: str | None = Query(None), limit: int = Query(40, ge=1, le=FORECAST_RING)) -> dict:
    return {"now": time.time(), "forecasts": forecasts(model, limit)}


@router.get("/agents/forecast-values/{vid}")
async def forecast_values(vid: str) -> dict:
    """The numbers behind one forecast request's sample anchors (see note_values)."""
    doc = values(vid)
    if doc is None:
        raise HTTPException(status_code=404, detail="those values are no longer kept (the inspector keeps the latest few)")
    return doc


@router.get("/agents/activity/{model:path}")
async def model_activity(model: str, project_id: str | None = Query(None)) -> dict:
    """The inspector for one model row: agents, their last iterations in full, peer use,
    and (for a forecaster) recent forecast requests."""
    return for_model(model, project_id)
