"""Steering: messages that reach an agent INSIDE its running iteration.

Before this, everything the team learned reached an agent only through its next iteration's
prompt -- 15 to 45 minutes later. A teammate's new champion, the operator's "stop tweaking
numbers", the mentor's pointer: all of it waited while the agent kept building on what it had
read at the start. Now the swarm runner asks for its agent's steering messages at the top of
every tool round (``POST /api/steer/take``) and hands them to the model as the next user
message, the way a person would interrupt a colleague between two steps.

Three kinds:

- ``steer``   -- advice; the agent reads it and carries on.
- ``wrap_up`` -- finish now: the next round is the iteration's last (submit what you have).
- ``abort``   -- end the iteration at once, nothing submitted.

A message is for one agent (by its board name, e.g. ``Muse-Glimmer #2``) or for ``*``, every
agent running on the project (optionally only those on one objective). A broadcast reaches
only iterations that were already running when it was sent -- one that starts later read the
same news in its prompt. Each agent receives a message once; unread messages expire.

The harness broadcasts one itself: when a candidate takes the title, every other agent on that
objective hears about it mid-iteration (``champion``), so it can build on it instead of
finishing a search the team has already overtaken.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from . import events

logger = logging.getLogger("freetoken.steering")
router = APIRouter(tags=["steering"])

DB_PATH = Path(__file__).resolve().parent.parent / "steering.sqlite3"
# An unread message is dropped after this long: advice about a search from hours ago misleads.
DEFAULT_TTL_S = 2 * 3600.0
# The harness's own broadcast when a candidate takes the title (off with FREESWARM_STEER_ON_CHAMPION=0).
STEER_ON_CHAMPION = os.environ.get("FREESWARM_STEER_ON_CHAMPION", "1").strip().lower() not in ("0", "false", "no", "off")
TEXT_MAX = 4000
Kind = Literal["steer", "wrap_up", "abort"]

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS steers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                objective_id TEXT,
                agent TEXT NOT NULL,            -- an agent's board name, or '*' for every agent
                kind TEXT NOT NULL,
                text TEXT NOT NULL,
                source TEXT NOT NULL,           -- operator | mentor | harness
                exclude_model TEXT,             -- a broadcast its subject's own model need not hear
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                cancelled_at REAL
            );
            CREATE INDEX IF NOT EXISTS steers_project ON steers (project_id, created_at);
            CREATE TABLE IF NOT EXISTS deliveries (
                steer_id INTEGER NOT NULL,
                agent TEXT NOT NULL,
                record_id TEXT,                 -- the iteration it reached (the agent inspector's record)
                round INTEGER,
                delivered_at REAL NOT NULL,
                PRIMARY KEY (steer_id, agent)
            );
            """
        )
        _conn.commit()
    return _conn


def _row(r: sqlite3.Row, deliveries: list[dict] | None = None) -> dict:
    d = dict(r)
    if deliveries is not None:
        d["deliveries"] = deliveries
    return d


def add(project_id: str, agent: str, text: str, kind: str = "steer", source: str = "operator",
        objective_id: str | None = None, ttl_s: float = DEFAULT_TTL_S, exclude_model: str | None = None) -> dict:
    """Queue a message for `agent` ('*' for every agent on the project). Returns it."""
    text = (text or "").strip()
    if kind == "steer" and not text:
        raise ValueError("a steering message needs text")
    if kind not in ("steer", "wrap_up", "abort"):
        raise ValueError(f"unknown kind {kind!r}")
    now = time.time()
    with _lock:
        cur = db().execute(
            "INSERT INTO steers (project_id, objective_id, agent, kind, text, source, exclude_model, created_at, "
            "expires_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (project_id, objective_id, agent.strip() or "*", kind, text[:TEXT_MAX], source, exclude_model, now,
             now + max(60.0, ttl_s)))
        db().commit()
        row = db().execute("SELECT * FROM steers WHERE id=?", (cur.lastrowid,)).fetchone()
    out = _row(row, [])
    events.publish(project_id, "steer", {"steer": out})
    return out


def take(project_id: str, agent: str, model: str | None = None, objective_id: str | None = None,
         since: float | None = None, record_id: str | None = None, round_: int | None = None) -> list[dict]:
    """The messages `agent` has not received yet, marked received. `since` is when its current
    iteration started: a broadcast sent before then is skipped (that iteration's prompt already
    carried the news) -- and marked, so a later iteration does not get it either."""
    now = time.time()
    out: list[dict] = []
    with _lock:
        con = db()
        rows = con.execute(
            "SELECT s.* FROM steers s LEFT JOIN deliveries d ON d.steer_id = s.id AND d.agent = ? "
            "WHERE s.project_id = ? AND d.steer_id IS NULL AND s.cancelled_at IS NULL AND s.expires_at > ? "
            "AND (s.agent = ? OR s.agent = '*') AND (s.objective_id IS NULL OR ? IS NULL OR s.objective_id = ?) "
            "ORDER BY s.id", (agent, project_id, now, agent, objective_id, objective_id)).fetchall()
        for r in rows:
            skip = (r["agent"] == "*" and since is not None and r["created_at"] < since) \
                or (r["exclude_model"] and model and r["exclude_model"] == model)
            con.execute("INSERT OR IGNORE INTO deliveries (steer_id, agent, record_id, round, delivered_at) "
                        "VALUES (?,?,?,?,?)", (r["id"], agent, None if skip else record_id,
                                               None if skip else round_, now))
            if not skip:
                out.append(_row(r))
        if rows:
            con.commit()
    for s in out:
        events.publish(project_id, "steer_delivered", {"steer_id": s["id"], "agent": agent, "kind": s["kind"],
                                                       "record_id": record_id, "round": round_})
    return out


def recent(project_id: str, limit: int = 30) -> list[dict]:
    """The project's latest messages, newest first, each with the agents it reached."""
    with _lock:
        con = db()
        rows = con.execute("SELECT * FROM steers WHERE project_id=? ORDER BY id DESC LIMIT ?",
                           (project_id, max(1, min(limit, 200)))).fetchall()
        ids = [r["id"] for r in rows]
        dels: dict[int, list[dict]] = {i: [] for i in ids}
        if ids:
            for d in con.execute(f"SELECT * FROM deliveries WHERE steer_id IN ({','.join('?' * len(ids))}) "
                                 "AND record_id IS NOT NULL ORDER BY delivered_at", ids).fetchall():
                dels[d["steer_id"]].append({k: d[k] for k in ("agent", "record_id", "round", "delivered_at")})
    return [_row(r, dels.get(r["id"], [])) for r in rows]


def cancel(steer_id: int) -> dict:
    """Withdraw a message; agents that already received it keep it."""
    with _lock:
        con = db()
        row = con.execute("SELECT * FROM steers WHERE id=?", (steer_id,)).fetchone()
        if row is None:
            raise KeyError(steer_id)
        if row["cancelled_at"] is None:
            con.execute("UPDATE steers SET cancelled_at=? WHERE id=?", (time.time(), steer_id))
            con.commit()
            row = con.execute("SELECT * FROM steers WHERE id=?", (steer_id,)).fetchone()
    out = _row(row)
    events.publish(out["project_id"], "steer", {"steer": out})
    return out


def prune(older_than_s: float = 14 * 86400.0) -> None:
    cutoff = time.time() - older_than_s
    with _lock:
        con = db()
        con.execute("DELETE FROM deliveries WHERE steer_id IN (SELECT id FROM steers WHERE created_at < ?)", (cutoff,))
        con.execute("DELETE FROM steers WHERE created_at < ?", (cutoff,))
        con.commit()


def champion_message(obj: dict, cand: dict) -> str:
    """What the agents still working on `obj` hear when `cand` takes the title."""
    score = cand.get("score")
    shown = f"{score:.4g}" if isinstance(score, (int, float)) else "?"
    why = " ".join(str(cand.get("rationale") or "").split())[:400]
    return (f"News from the team: candidate #{cand.get('seq')} by {cand.get('model') or 'a teammate'} just became the "
            f"best on \"{obj.get('title') or obj.get('id')}\" (score {shown})."
            + (f" Its idea: {why}" if why else "")
            + f" If what you are doing overlaps with it, build on #{cand.get('seq')} (cite it as your parent) or make "
              "yours clearly different -- a parameter tweak of an older candidate will not beat it. If your work is "
              "unrelated, carry on; no reply is needed.")


def broadcast_champion(obj: dict, cand: dict) -> None:
    """Tell every agent on the objective, mid-iteration, that `cand` took the title. Never raises."""
    if not STEER_ON_CHAMPION:
        return
    try:
        add(obj["project_id"], "*", champion_message(obj, cand), kind="steer", source="harness",
            objective_id=obj["id"], ttl_s=3600.0, exclude_model=cand.get("model"))
    except Exception as exc:  # noqa: BLE001 -- news must never fail the crowning it reports
        logger.warning("steering: champion broadcast failed: %s", exc)


# =======================================================================================
# Routes (mounted under /api)
# =======================================================================================
class SteerReq(BaseModel):
    agent: str = Field("*", min_length=1, max_length=300)
    text: str = Field("", max_length=TEXT_MAX)
    kind: Kind = "steer"
    objective_id: str | None = Field(None, max_length=200)


@router.post("/projects/{project_id}/steer")
async def post_steer(project_id: str, req: SteerReq) -> dict:
    """From the console: a message for one agent's running iteration (or '*' for all)."""
    try:
        return await asyncio.to_thread(add, project_id, req.agent, req.text, req.kind, "operator", req.objective_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@router.get("/projects/{project_id}/steer")
async def list_steers(project_id: str, limit: int = Query(30, ge=1, le=200)) -> dict:
    return {"now": time.time(), "steers": await asyncio.to_thread(recent, project_id, limit)}


@router.delete("/steer/{steer_id}")
async def delete_steer(steer_id: int) -> dict:
    try:
        return await asyncio.to_thread(cancel, steer_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such steering message") from None


class TakeReq(BaseModel):
    project_id: str = Field(..., max_length=200)
    agent: str = Field(..., max_length=300)
    model: str | None = Field(None, max_length=300)
    objective_id: str | None = Field(None, max_length=200)
    since: float | None = None
    record_id: str | None = Field(None, max_length=200)
    round: int | None = None


@router.post("/steer/take")
async def take_steers(req: TakeReq) -> dict:
    """From the swarm runner, at the top of each tool round: this agent's unread messages."""
    got = await asyncio.to_thread(take, req.project_id, req.agent, req.model, req.objective_id, req.since,
                                  req.record_id, req.round)
    return {"steers": got}
