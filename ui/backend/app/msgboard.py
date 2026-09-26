"""Agent coordination bus -- a standalone FastAPI service run alongside the control plane.

This is the shared substrate an agent swarm needs in order to be a swarm rather than N
independent processes:

* **Messages** -- an append-only log per channel with a monotonic `seq`. Agents read with
  `since=<cursor>` and can long-poll, so coordination costs one idle connection instead of
  a polling storm.
* **Tasks** -- a work queue with *atomic claim*. Exactly one agent wins a claim, enforced by
  a conditional UPDATE rather than a read-then-write, so two agents racing the same task
  cannot both start it. Claims carry a **lease**; a crashed agent's task returns to the
  queue when its lease expires instead of being lost forever.
* **Blackboard** -- a shared key/value space for facts agents accumulate (a repo layout, a
  build command that worked, a rejected approach), with compare-and-set so concurrent
  writers do not silently clobber each other.

Storage is SQLite in WAL mode: durable across restarts, safe for the concurrent readers and
writers a swarm produces, and no extra dependency. `check_same_thread=False` plus a module
lock because FastAPI serves these handlers from a thread pool.

Auth is deliberately the *same* token issuer as the control plane (`auth.py`), so one login
works across both services -- but note the scope split: a swarm agent should hold a token
that cannot reach the control plane's engine-start endpoint. See `require_agent` below.

Run it with:
    .venv\\Scripts\\python -m uvicorn app.msgboard:app --port 8100 --host 127.0.0.1
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import auth
from .config import settings

logger = logging.getLogger("freetoken.msgboard")

DB_PATH = Path(__file__).resolve().parent.parent / "msgboard.sqlite3"

# Default lease: long enough that a model generating a long answer does not lose its task,
# short enough that a crashed agent frees the work within a coffee break.
DEFAULT_LEASE_S = 300
AGENT_STALE_S = 90

_db_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
        _conn.row_factory = sqlite3.Row
        # WAL lets readers proceed while a writer holds the table -- the normal state of a
        # swarm, where several agents poll while one posts.
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.execute("PRAGMA foreign_keys=ON")
        _init_schema(_conn)
    return _conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS agents (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            role        TEXT NOT NULL DEFAULT '',
            model       TEXT NOT NULL DEFAULT '',
            capabilities TEXT NOT NULL DEFAULT '[]',
            registered_at REAL NOT NULL,
            last_seen   REAL NOT NULL,
            status      TEXT NOT NULL DEFAULT 'idle'
        );

        CREATE TABLE IF NOT EXISTS messages (
            seq        INTEGER PRIMARY KEY AUTOINCREMENT,
            channel    TEXT NOT NULL,
            author     TEXT NOT NULL,
            author_id  TEXT,
            kind       TEXT NOT NULL DEFAULT 'chat',
            content    TEXT NOT NULL,
            meta       TEXT NOT NULL DEFAULT '{}',
            reply_to   INTEGER,
            ts         REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_messages_channel_seq ON messages(channel, seq);

        CREATE TABLE IF NOT EXISTS tasks (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            tier        TEXT NOT NULL DEFAULT 'auto',
            status      TEXT NOT NULL DEFAULT 'open',
            parent_id   TEXT,
            claimed_by  TEXT,
            lease_until REAL,
            result      TEXT,
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL,
            attempts    INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);

        CREATE TABLE IF NOT EXISTS blackboard (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            version    INTEGER NOT NULL DEFAULT 1,
            updated_by TEXT NOT NULL DEFAULT '',
            updated_at REAL NOT NULL
        );

        -- A session groups everything that happened during one run of the swarm, so the
        -- board can be reviewed after the fact instead of being one undifferentiated log.
        -- Exactly one session has ended_at IS NULL at a time: that is "current".
        CREATE TABLE IF NOT EXISTS sessions (
            id         TEXT PRIMARY KEY,
            title      TEXT NOT NULL DEFAULT '',
            note       TEXT NOT NULL DEFAULT '',
            started_at REAL NOT NULL,
            ended_at   REAL
        );

        -- Blackboard keys are per-project: two projects may both track "repo_layout"
        -- without overwriting each other. The old single-key PRIMARY KEY is migrated
        -- below.
        CREATE TABLE IF NOT EXISTS blackboard2 (
            project_id TEXT NOT NULL DEFAULT '',
            key        TEXT NOT NULL,
            value      TEXT NOT NULL,
            version    INTEGER NOT NULL DEFAULT 1,
            updated_by TEXT NOT NULL DEFAULT '',
            updated_at REAL NOT NULL,
            PRIMARY KEY (project_id, key)
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
        """
    )

    # Added after the first release, so existing boards need the columns backfilled.
    # SQLite has no "ADD COLUMN IF NOT EXISTS"; checking PRAGMA is the portable way.
    for table in ("messages", "tasks"):
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if "session_id" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN session_id TEXT")
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_session ON {table}(session_id)"
            )

    # Projects landed later still. Every scoped table gains project_id; existing rows keep
    # '' and are adopted by the first project created, so an upgrade never loses history.
    for table in ("sessions", "messages", "tasks", "agents"):
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if "project_id" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN project_id TEXT NOT NULL DEFAULT ''")
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_project ON {table}(project_id)"
            )

    # Migrate the old single-scope blackboard into the project-scoped table once.
    have = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if "blackboard" in have:
        rows = conn.execute("SELECT * FROM blackboard").fetchall()
        for r in rows:
            conn.execute(
                "INSERT OR IGNORE INTO blackboard2 "
                "(project_id,key,value,version,updated_by,updated_at) VALUES ('',?,?,?,?,?)",
                (r["key"], r["value"], r["version"], r["updated_by"], r["updated_at"]),
            )
        conn.execute("DROP TABLE blackboard")
    conn.commit()


def adopt_orphans(conn: sqlite3.Connection, project_id: str) -> int:
    """Hand pre-projects rows to `project_id`.

    Called when the first project is created on an upgraded install: without it, the
    existing history would be invisible because every query is project-scoped.
    """
    n = 0
    for table in ("sessions", "messages", "tasks", "agents"):
        cur = conn.execute(
            f"UPDATE {table} SET project_id=? WHERE project_id=''", (project_id,)
        )
        n += cur.rowcount
    cur = conn.execute(
        "UPDATE blackboard2 SET project_id=? WHERE project_id=''", (project_id,)
    )
    n += cur.rowcount
    conn.commit()
    return n


def resolve_project(explicit: str | None = None) -> str:
    """The project a request operates on.

    An explicit id wins (an agent can pin itself to one); otherwise the console's active
    project is used. A default project is created on demand so no caller has to handle the
    empty case, and on an upgraded install it adopts the pre-projects history.
    """
    from . import projects

    if explicit:
        if projects.get(explicit) is None:
            raise HTTPException(status_code=404, detail=f"no project {explicit!r}")
        return explicit

    current = projects.active_id()
    if current:
        return current

    created = projects.ensure_default()
    with _db_lock:
        adopted = adopt_orphans(db(), created["id"])
    if adopted:
        logger.info("adopted %d pre-projects rows into %s", adopted, created["name"])
    return created["id"]


def current_session_id(conn: sqlite3.Connection, project_id: str) -> str:
    """The open session, created on demand.

    Auto-creating means a caller never has to think about sessions to use the board -- the
    first message or task of a run opens one -- while anyone who does care can close it and
    start a fresh one to separate runs.
    """
    row = conn.execute(
        "SELECT id FROM sessions WHERE ended_at IS NULL AND project_id=? "
        "ORDER BY started_at DESC LIMIT 1",
        (project_id,),
    ).fetchone()
    if row is not None:
        return str(row["id"])
    now = time.time()
    session_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO sessions (id,title,started_at,project_id) VALUES (?,?,?,?)",
        (session_id, time.strftime("%Y-%m-%d %H:%M", time.localtime(now)), now, project_id),
    )
    conn.commit()
    return session_id


def _expire_leases(conn: sqlite3.Connection) -> int:
    """Return timed-out claims to the queue.

    Called on every task read/claim rather than from a background timer: it is a cheap
    indexed UPDATE, and doing it inline means correctness does not depend on a sweeper task
    still being alive.

    Always commits, even when nothing expired: Python's sqlite3 opens a write transaction at
    the UPDATE whether or not a row matches, and leaving it open kept the board's write lock
    held between dashboard polls -- every other writer (agents posting, the console's board
    notes) then waited out its busy timeout, and the open transaction pinned the WAL so it
    could never be checkpointed (it grew past 1 GB).
    """
    now = time.time()
    cur = conn.execute(
        "UPDATE tasks SET status='open', claimed_by=NULL, lease_until=NULL, updated_at=? "
        "WHERE status='claimed' AND lease_until IS NOT NULL AND lease_until < ?",
        (now, now),
    )
    conn.commit()
    return cur.rowcount


from .version import APP_VERSION  # noqa: E402

app = FastAPI(title="FreeSwarm Agent Message Board", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

# Same issuer as the control plane, so one login covers both services.
require_agent = auth.require_user


# =======================================================================================
# Agents
# =======================================================================================
class AgentRegister(BaseModel):
    project_id: str | None = None
    name: str = Field(..., max_length=120)
    role: str = Field("", max_length=200)
    model: str = Field("", max_length=200)
    capabilities: list[str] = Field(default_factory=list)


@app.post("/mb/agents/register")
async def register_agent(req: AgentRegister, _: str | None = Depends(require_agent)) -> dict:
    """Register (or re-register) an agent. Re-using a name returns the same id, so an agent
    that restarts keeps its identity instead of orphaning its history."""
    now = time.time()
    scope = resolve_project(req.project_id)
    with _db_lock:
        conn = db()
        # Identity is per-project: the same agent name in two projects is two agents.
        row = conn.execute(
            "SELECT id FROM agents WHERE name=? AND project_id=?", (req.name, scope)
        ).fetchone()
        agent_id = row["id"] if row else uuid.uuid4().hex
        conn.execute(
            "INSERT INTO agents "
            "(id,name,role,model,capabilities,registered_at,last_seen,status,project_id) "
            "VALUES (?,?,?,?,?,?,?, 'idle',?) "
            "ON CONFLICT(id) DO UPDATE SET role=excluded.role, model=excluded.model, "
            "capabilities=excluded.capabilities, last_seen=excluded.last_seen",
            (agent_id, req.name, req.role, req.model, json.dumps(req.capabilities),
             now, now, scope),
        )
        conn.commit()
    return {"agent_id": agent_id, "name": req.name, "project_id": scope}


class Heartbeat(BaseModel):
    status: Literal["idle", "working", "blocked", "done"] = "idle"


@app.post("/mb/agents/{agent_id}/heartbeat")
async def heartbeat(agent_id: str, req: Heartbeat, _: str | None = Depends(require_agent)) -> dict:
    with _db_lock:
        conn = db()
        cur = conn.execute(
            "UPDATE agents SET last_seen=?, status=? WHERE id=?",
            (time.time(), req.status, agent_id),
        )
        conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="unknown agent")
    return {"ok": True}


@app.get("/mb/agents")
async def list_agents(
    project_id: str | None = Query(None), _: str | None = Depends(require_agent)
) -> dict:
    now = time.time()
    scope = resolve_project(project_id)
    with _db_lock:
        rows = db().execute(
            "SELECT * FROM agents WHERE project_id=? ORDER BY name", (scope,)
        ).fetchall()
    return {
        "agents": [
            {
                **dict(r),
                "capabilities": json.loads(r["capabilities"] or "[]"),
                # An agent that stopped heartbeating is reported offline rather than
                # whatever status it last claimed -- a crashed 'working' agent would
                # otherwise look busy forever.
                "online": (now - r["last_seen"]) < AGENT_STALE_S,
            }
            for r in rows
        ]
    }


# =======================================================================================
# Messages
# =======================================================================================
class PostMessage(BaseModel):
    project_id: str | None = None
    channel: str = Field("general", max_length=80)
    author: str = Field(..., max_length=120)
    author_id: str | None = None
    kind: Literal["chat", "directive", "result", "error", "thought", "system"] = "chat"
    content: str = Field(..., max_length=200_000)
    meta: dict[str, Any] = Field(default_factory=dict)
    reply_to: int | None = None


@app.post("/mb/messages")
async def post_message(req: PostMessage, _: str | None = Depends(require_agent)) -> dict:
    with _db_lock:
        conn = db()
        project_id = resolve_project(req.project_id)
        session_id = current_session_id(conn, project_id)
        cur = conn.execute(
            "INSERT INTO messages "
            "(channel,author,author_id,kind,content,meta,reply_to,ts,session_id,project_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                req.channel, req.author, req.author_id, req.kind, req.content,
                json.dumps(req.meta), req.reply_to, time.time(), session_id, project_id,
            ),
        )
        conn.commit()
        seq = cur.lastrowid
    return {"seq": seq, "session_id": session_id, "project_id": project_id}


def _read_messages(
    channel: str | None,
    since: int,
    limit: int,
    session_id: str | None = None,
    project_id: str | None = None,
    tail: int = 0,
) -> list[dict]:
    # Built as a filtered SELECT rather than three hand-written queries so adding the
    # session dimension does not multiply the branches.
    clauses = ["seq > ?"]
    params: list[Any] = [since]
    if channel:
        clauses.append("channel = ?")
        params.append(channel)
    if session_id:
        clauses.append("session_id = ?")
        params.append(session_id)
    if project_id:
        clauses.append("project_id = ?")
        params.append(project_id)
    params.append(tail or limit)
    # tail=N: the LAST N matching messages (an agent catching up on what the team just did),
    # returned oldest-first like every other read.
    order = "DESC" if tail else "ASC"
    sql = f"SELECT * FROM messages WHERE {' AND '.join(clauses)} ORDER BY seq {order} LIMIT ?"
    with _db_lock:
        rows = db().execute(sql, params).fetchall()
    if tail:
        rows = list(reversed(rows))
    return [{**dict(r), "meta": json.loads(r["meta"] or "{}")} for r in rows]


@app.get("/mb/messages")
async def get_messages(
    channel: str | None = None,
    since: int = 0,
    limit: int = Query(200, ge=1, le=1000),
    wait: float = Query(0.0, ge=0.0, le=60.0, description="Long-poll seconds"),
    session_id: str | None = Query(None, description="Restrict to one session's history"),
    project_id: str | None = Query(None, description="Defaults to the active project"),
    tail: int = Query(0, ge=0, le=500, description="Return the last N messages instead of those after `since`"),
    _: str | None = Depends(require_agent),
) -> dict:
    """Read messages after `since`. With `wait`, block until something arrives.

    Long-polling matters here: a swarm of agents each polling at 1 Hz generates constant
    load for nothing. `wait=30` turns that into one idle connection per agent that returns
    the instant a message lands.
    """
    scope = resolve_project(project_id)
    entries = _read_messages(channel, since, limit, session_id, scope, tail)
    if tail:
        return {"entries": entries, "next_cursor": entries[-1]["seq"] if entries else since}
    if not entries and wait > 0:
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            await asyncio.sleep(0.4)
            entries = _read_messages(channel, since, limit, session_id, scope)
            if entries:
                break
    cursor = entries[-1]["seq"] if entries else since
    return {"entries": entries, "next_cursor": cursor}


def _rows(sql: str, params: list[Any]) -> list[dict]:
    with _db_lock:
        rows = db().execute(sql, params).fetchall()
    return [{**dict(r), "meta": json.loads(r["meta"] or "{}")} for r in rows]


@app.get("/mb/team/threads")
def team_threads(
    agent: str = Query(..., description="Model name as it appears in the collaboration records"),
    project_id: str | None = Query(None, description="Defaults to the active project"),
    through: int | None = Query(None, description="Newest #team seq the caller counted; the window ends there"),
    _: str | None = Depends(require_agent),
) -> dict:
    """The messages behind the Team panel's sent / answered / unanswered counts for one
    model, over the same window the panel totals (the last 300 #team messages)."""
    from . import team_threads as tt

    scope = resolve_project(project_id)
    cap = [through] if through else []
    team = _rows(
        f"SELECT * FROM messages WHERE channel='team' AND project_id=? {'AND seq<=?' if through else ''} "
        "ORDER BY seq DESC LIMIT ?", [scope, *cap, tt.TEAM_WINDOW])[::-1]
    empty = {"agent": agent, "counts": tt.counts([]), "sent": [], "answered": [], "unanswered": [],
             "unlocated": {"sent": 0, "answered": 0, "unanswered": 0}, "through": through}
    recs = tt.collab_records(team, agent)
    if not recs:
        return empty
    # The agent's "Iteration on" / "Mentoring" notes mark each inbox read (all history: the
    # read before the first counted iteration may be long ago).
    markers = _rows(
        "SELECT * FROM messages WHERE channel='general' AND project_id=? AND author=? AND seq<=? "
        "AND (content LIKE 'Iteration on%' OR content LIKE 'Mentoring: reading%') ORDER BY seq",
        [scope, agent, recs[-1]["seq"]])
    first = tt.first_start_seq(team, agent, markers) or recs[0]["seq"]
    lo = _rows("SELECT seq, '{}' AS meta FROM messages WHERE project_id=? AND seq<? ORDER BY seq DESC LIMIT 1 OFFSET ?",
               [scope, first, tt.INBOX_TAIL - 1])
    board = _rows("SELECT * FROM messages WHERE project_id=? AND seq>=? ORDER BY seq",
                  [scope, lo[0]["seq"] if lo else 0])
    return {**tt.threads(team, board, agent, markers), "through": team[-1]["seq"] if team else through}


@app.get("/mb/channels")
async def channels(_: str | None = Depends(require_agent)) -> dict:
    with _db_lock:
        rows = db().execute(
            "SELECT channel, COUNT(*) n, MAX(ts) last_ts FROM messages GROUP BY channel "
            "ORDER BY last_ts DESC"
        ).fetchall()
    return {"channels": [dict(r) for r in rows]}


# =======================================================================================
# Tasks
# =======================================================================================
class CreateTask(BaseModel):
    project_id: str | None = None
    title: str = Field(..., max_length=300)
    description: str = Field("", max_length=100_000)
    # Which capability tier the router should satisfy this with. 'auto' lets the router
    # decide; the explicit tiers map onto the small/mid/hard model split.
    tier: Literal["auto", "small", "mid", "hard"] = "auto"
    parent_id: str | None = None


@app.post("/mb/tasks")
async def create_task(req: CreateTask, _: str | None = Depends(require_agent)) -> dict:
    now = time.time()
    task_id = uuid.uuid4().hex
    with _db_lock:
        conn = db()
        project_id = resolve_project(req.project_id)
        session_id = current_session_id(conn, project_id)
        conn.execute(
            "INSERT INTO tasks "
            "(id,title,description,tier,status,parent_id,created_at,updated_at,session_id,project_id) "
            "VALUES (?,?,?,?, 'open', ?,?,?,?,?)",
            (task_id, req.title, req.description, req.tier, req.parent_id, now, now,
             session_id, project_id),
        )
        conn.commit()
    return {"task_id": task_id, "status": "open", "session_id": session_id,
            "project_id": project_id}


class ClaimTask(BaseModel):
    project_id: str | None = None
    agent_id: str
    lease_s: int = Field(DEFAULT_LEASE_S, ge=10, le=7200)
    tier: Literal["auto", "small", "mid", "hard"] | None = None


@app.post("/mb/tasks/claim")
async def claim_next(req: ClaimTask, _: str | None = Depends(require_agent)) -> dict:
    """Atomically claim the oldest open task, optionally filtered by tier.

    The claim is a single conditional UPDATE guarded by `status='open'`. Two agents racing
    for the same row cannot both succeed: SQLite serialises the writes, and the loser's
    UPDATE matches zero rows because the status already changed.
    """
    now = time.time()
    with _db_lock:
        conn = db()
        _expire_leases(conn)
        scope = resolve_project(req.project_id)
        if req.tier and req.tier != "auto":
            row = conn.execute(
                "SELECT id FROM tasks WHERE status='open' AND project_id=? "
                "AND tier IN (?, 'auto') ORDER BY created_at LIMIT 1",
                (scope, req.tier),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id FROM tasks WHERE status='open' AND project_id=? "
                "ORDER BY created_at LIMIT 1",
                (scope,),
            ).fetchone()
        if row is None:
            return {"task": None}

        cur = conn.execute(
            "UPDATE tasks SET status='claimed', claimed_by=?, lease_until=?, updated_at=?, "
            "attempts=attempts+1 WHERE id=? AND status='open'",
            (req.agent_id, now + req.lease_s, now, row["id"]),
        )
        conn.commit()
        if cur.rowcount == 0:
            # Lost the race; the caller retries rather than getting someone else's task.
            return {"task": None}
        task = conn.execute("SELECT * FROM tasks WHERE id=?", (row["id"],)).fetchone()
    return {"task": dict(task)}


class CompleteTask(BaseModel):
    agent_id: str
    status: Literal["done", "failed"] = "done"
    result: str = Field("", max_length=500_000)


@app.post("/mb/tasks/{task_id}/complete")
async def complete_task(
    task_id: str, req: CompleteTask, _: str | None = Depends(require_agent)
) -> dict:
    with _db_lock:
        conn = db()
        # Guarded by claimed_by so a stale agent whose lease already expired (and whose task
        # another agent has since claimed) cannot overwrite the new owner's result.
        cur = conn.execute(
            "UPDATE tasks SET status=?, result=?, lease_until=NULL, updated_at=? "
            "WHERE id=? AND claimed_by=?",
            (req.status, req.result, time.time(), task_id, req.agent_id),
        )
        conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(
            status_code=409,
            detail="task is not claimed by this agent (lease may have expired)",
        )
    return {"ok": True, "status": req.status}


@app.post("/mb/tasks/{task_id}/extend")
async def extend_lease(
    task_id: str, req: ClaimTask, _: str | None = Depends(require_agent)
) -> dict:
    """Push a lease out. A long-running agent calls this rather than letting work be
    reclaimed underneath it."""
    with _db_lock:
        conn = db()
        cur = conn.execute(
            "UPDATE tasks SET lease_until=?, updated_at=? WHERE id=? AND claimed_by=? "
            "AND status='claimed'",
            (time.time() + req.lease_s, time.time(), task_id, req.agent_id),
        )
        conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(status_code=409, detail="task not claimed by this agent")
    return {"ok": True}


@app.get("/mb/tasks")
async def list_tasks(
    status: str | None = None,
    limit: int = Query(200, ge=1, le=1000),
    project_id: str | None = Query(None),
    _: str | None = Depends(require_agent),
) -> dict:
    scope = resolve_project(project_id)
    with _db_lock:
        conn = db()
        _expire_leases(conn)
        if status:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE project_id=? AND status=? "
                "ORDER BY created_at DESC LIMIT ?",
                (scope, status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE project_id=? ORDER BY created_at DESC LIMIT ?",
                (scope, limit),
            ).fetchall()
    return {"tasks": [dict(r) for r in rows]}


# =======================================================================================
# Blackboard
# =======================================================================================
class SetValue(BaseModel):
    project_id: str | None = None
    value: Any
    updated_by: str = ""
    # Compare-and-set. When supplied, the write only lands if the stored version matches,
    # so two agents updating the same fact cannot silently lose one of the updates.
    expect_version: int | None = None


@app.put("/mb/state/{key}")
async def set_state(key: str, req: SetValue, _: str | None = Depends(require_agent)) -> dict:
    payload = json.dumps(req.value)
    now = time.time()
    scope = resolve_project(req.project_id)
    with _db_lock:
        conn = db()
        row = conn.execute(
            "SELECT version FROM blackboard2 WHERE project_id=? AND key=?", (scope, key)
        ).fetchone()
        if req.expect_version is not None:
            current = row["version"] if row else 0
            if current != req.expect_version:
                raise HTTPException(
                    status_code=409,
                    detail=f"version conflict: stored {current}, expected {req.expect_version}",
                )
        version = (row["version"] + 1) if row else 1
        conn.execute(
            "INSERT INTO blackboard2 (project_id,key,value,version,updated_by,updated_at) "
            "VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(project_id,key) DO UPDATE SET value=excluded.value, "
            "version=excluded.version, updated_by=excluded.updated_by, "
            "updated_at=excluded.updated_at",
            (scope, key, payload, version, req.updated_by, now),
        )
        conn.commit()
    return {"key": key, "version": version, "project_id": scope}


@app.get("/mb/state/{key}")
async def get_state(
    key: str, project_id: str | None = Query(None), _: str | None = Depends(require_agent)
) -> dict:
    scope = resolve_project(project_id)
    with _db_lock:
        row = db().execute(
            "SELECT * FROM blackboard2 WHERE project_id=? AND key=?", (scope, key)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="no such key")
    return {**dict(row), "value": json.loads(row["value"])}


@app.get("/mb/state")
async def list_state(
    project_id: str | None = Query(None), _: str | None = Depends(require_agent)
) -> dict:
    scope = resolve_project(project_id)
    with _db_lock:
        rows = db().execute(
            "SELECT * FROM blackboard2 WHERE project_id=? ORDER BY key", (scope,)
        ).fetchall()
    return {"entries": [{**dict(r), "value": json.loads(r["value"])} for r in rows]}


# =======================================================================================
# Sessions -- reviewable history
# =======================================================================================
class NewSession(BaseModel):
    project_id: str | None = None
    title: str = Field("", max_length=200)
    note: str = Field("", max_length=10_000)


@app.post("/mb/sessions")
async def start_session(req: NewSession, _: str | None = Depends(require_agent)) -> dict:
    """Close the open session and start a fresh one.

    Closing first keeps the "exactly one open session" invariant that
    `current_session_id` relies on.
    """
    now = time.time()
    session_id = uuid.uuid4().hex
    title = req.title.strip() or time.strftime("%Y-%m-%d %H:%M", time.localtime(now))
    scope = resolve_project(req.project_id)
    with _db_lock:
        conn = db()
        # Only this project's open session closes: another project's run must keep going.
        conn.execute(
            "UPDATE sessions SET ended_at=? WHERE ended_at IS NULL AND project_id=?",
            (now, scope),
        )
        conn.execute(
            "INSERT INTO sessions (id,title,note,started_at,project_id) VALUES (?,?,?,?,?)",
            (session_id, title, req.note, now, scope),
        )
        conn.commit()
    return {"session_id": session_id, "title": title, "started_at": now,
            "project_id": scope}


@app.post("/mb/sessions/{session_id}/close")
async def close_session(session_id: str, _: str | None = Depends(require_agent)) -> dict:
    with _db_lock:
        conn = db()
        cur = conn.execute(
            "UPDATE sessions SET ended_at=? WHERE id=? AND ended_at IS NULL",
            (time.time(), session_id),
        )
        conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="no such open session")
    return {"ok": True}


@app.get("/mb/sessions")
async def list_sessions(
    limit: int = Query(100, ge=1, le=1000),
    project_id: str | None = Query(None),
    _: str | None = Depends(require_agent),
) -> dict:
    """Every session, newest first, with its message and task counts.

    LEFT JOINs so a session with no activity still appears -- otherwise a freshly started
    session would vanish from the picker until someone posted to it.
    """
    scope = resolve_project(project_id)
    with _db_lock:
        rows = db().execute(
            """
            SELECT s.*,
                   (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count,
                   (SELECT COUNT(*) FROM tasks t WHERE t.session_id = s.id) AS task_count,
                   (SELECT COUNT(*) FROM tasks t WHERE t.session_id = s.id
                                                  AND t.status = 'done') AS tasks_done
            FROM sessions s
            WHERE s.project_id = ?
            ORDER BY s.started_at DESC
            LIMIT ?
            """,
            (scope, limit),
        ).fetchall()
    return {
        "sessions": [
            {**dict(r), "active": r["ended_at"] is None} for r in rows
        ]
    }


@app.get("/mb/sessions/{session_id}")
async def get_session(session_id: str, _: str | None = Depends(require_agent)) -> dict:
    """Full transcript of one session: its messages and its tasks."""
    with _db_lock:
        conn = db()
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="no such session")
        messages = conn.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY seq", (session_id,)
        ).fetchall()
        tasks = conn.execute(
            "SELECT * FROM tasks WHERE session_id=? ORDER BY created_at", (session_id,)
        ).fetchall()
    return {
        "session": {**dict(row), "active": row["ended_at"] is None},
        "messages": [{**dict(m), "meta": json.loads(m["meta"] or "{}")} for m in messages],
        "tasks": [dict(t) for t in tasks],
    }


@app.post("/mb/checkpoint")
async def checkpoint(_: str | None = Depends(require_agent)) -> dict:
    """Fold the WAL back into the main database file.

    In WAL mode most recent writes live in `msgboard.sqlite3-wal`, not the .sqlite3 file,
    so copying the .sqlite3 alone for a backup silently loses them. Truncating the WAL
    makes the single file self-contained.
    """
    with _db_lock:
        db().execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return {"ok": True, "db": str(DB_PATH)}


@app.get("/mb/health")
async def health() -> dict:
    """Unauthenticated liveness probe."""
    return {"status": "ok", "db": str(DB_PATH)}


@app.get("/mb/summary")
async def summary(
    project_id: str | None = Query(None), _: str | None = Depends(require_agent)
) -> dict:
    """One call powering the swarm dashboard, scoped to one project."""
    now = time.time()
    scope = resolve_project(project_id)
    with _db_lock:
        conn = db()
        _expire_leases(conn)
        agents = conn.execute(
            "SELECT * FROM agents WHERE project_id=? ORDER BY name", (scope,)
        ).fetchall()
        counts = conn.execute(
            "SELECT status, COUNT(*) n FROM tasks WHERE project_id=? GROUP BY status",
            (scope,),
        ).fetchall()
        msg_total = conn.execute(
            "SELECT COUNT(*) n FROM messages WHERE project_id=?", (scope,)
        ).fetchone()["n"]
        session = conn.execute(
            "SELECT * FROM sessions WHERE ended_at IS NULL AND project_id=? "
            "ORDER BY started_at DESC LIMIT 1",
            (scope,),
        ).fetchone()
    return {
        "agents": [
            {**dict(a), "capabilities": json.loads(a["capabilities"] or "[]"),
             "online": (now - a["last_seen"]) < AGENT_STALE_S}
            for a in agents
        ],
        "task_counts": {r["status"]: r["n"] for r in counts},
        "message_total": msg_total,
        "project_id": scope,
        "session": dict(session) if session is not None else None,
        "ts": now,
    }
