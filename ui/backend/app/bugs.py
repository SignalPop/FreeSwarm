"""The bug list: problems the monitoring agent (app/monitor.py) found in the agents' logs, plus
any an operator files by hand.

A bug is one PROBLEM, not one occurrence: findings carry a fingerprint (what went wrong, with
ids and numbers stripped), and every new occurrence of a known fingerprint is a *sighting* of the
existing bug -- its count goes up, its evidence and script become the latest ones. A sighting has
its own key (the record and event it came from), so rescanning the same logs never counts twice.

Some findings only matter when they recur (an agent's own SyntaxError is noise; the same error in
forty runs is a prompt or docs gap): a bug stays hidden until it reaches its `min_occurrences`.
A closed bug seen again after it was closed is reopened.

Statuses: open (needs looking at), pending (acknowledged / fix in progress), closed.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(tags=["bugs"])

DB_PATH = Path(__file__).resolve().parent.parent / "bugs.sqlite3"
STATUSES = ("open", "pending", "closed")
SEVERITIES = ("critical", "high", "medium", "low")
PRIORITIES = ("P1", "P2", "P3", "P4")
MAX_SIGHTINGS = 50          # kept per bug (the count keeps going)
MAX_TEXT = 20_000           # script / evidence stored per bug

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None


def db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.executescript("""
            CREATE TABLE IF NOT EXISTS bugs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                category TEXT NOT NULL DEFAULT 'error',
                severity TEXT NOT NULL DEFAULT 'medium',
                priority TEXT NOT NULL DEFAULT 'P3',
                status TEXT NOT NULL DEFAULT 'open',
                source TEXT NOT NULL DEFAULT 'monitor',
                visible INTEGER NOT NULL DEFAULT 1,
                min_occurrences INTEGER NOT NULL DEFAULT 1,
                project_id TEXT, agent TEXT, model TEXT, objective_id TEXT, objective_title TEXT, tool TEXT,
                script TEXT NOT NULL DEFAULT '',
                evidence TEXT NOT NULL DEFAULT '',
                context TEXT NOT NULL DEFAULT '{}',
                suggestion TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                occurrences INTEGER NOT NULL DEFAULT 0,
                first_seen REAL, last_seen REAL,
                created_at REAL NOT NULL, updated_at REAL NOT NULL, closed_at REAL,
                triaged INTEGER NOT NULL DEFAULT 0,
                edited INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS bugs_status ON bugs(status, visible, last_seen);
            CREATE TABLE IF NOT EXISTS bug_sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bug_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                at REAL NOT NULL,
                agent TEXT, model TEXT, project_id TEXT, objective_id TEXT, record_id TEXT,
                evidence TEXT NOT NULL DEFAULT '',
                UNIQUE(bug_id, key)
            );
            CREATE INDEX IF NOT EXISTS bug_sightings_bug ON bug_sightings(bug_id, at);
        """)
        cols = {r[1] for r in _conn.execute("PRAGMA table_info(bugs)").fetchall()}
        if "closed_by" not in cols:          # 'monitor' (seen fixed) | 'operator'
            _conn.execute("ALTER TABLE bugs ADD COLUMN closed_by TEXT")
        _conn.commit()
    return _conn


def reset() -> None:
    """Close the connection (tests point DB_PATH elsewhere first)."""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
        _conn = None


# ---------------------------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------------------------
_HEX = re.compile(r"\b[0-9a-f]{8,}\b")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_WS = re.compile(r"\s+")


_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")


def normalize(text: str, limit: int = 160, quoted: bool = True) -> str:
    """What stays the same between two occurrences of one problem: ids, numbers and spacing go
    (and, with quoted=False, quoted names: "no dataset 'x'" and "no dataset 'y'" are one problem)."""
    t = str(text or "").lower()
    if not quoted:
        t = _QUOTED.sub("<s>", t)
    t = _HEX.sub("<id>", t)
    t = _NUM.sub("<n>", t)
    return _WS.sub(" ", t).strip()[:limit]


# ---------------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------------
def _noted(notes: str | None, line: str, at: float | None = None) -> str:
    """`notes` with one more timestamped line: the bug's history (filed, reopened, closed,
    written up, rechecked, status changes) is kept where the operator writes theirs."""
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(at or time.time()))
    return ((notes.rstrip() + "\n") if notes and notes.strip() else "") + f"[{stamp}] {line}"


def add_note(bid: int, line: str) -> None:
    with _lock:
        row = db().execute("SELECT notes FROM bugs WHERE id=?", (bid,)).fetchone()
        if row is None:
            return
        db().execute("UPDATE bugs SET notes=?, updated_at=? WHERE id=?", (_noted(row["notes"], line), time.time(), bid))
        db().commit()


def _clip(v: Any, limit: int = MAX_TEXT) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str, indent=1) if v not in (None, "") else ""
    return s if len(s) <= limit else s[: limit // 2] + "\n…\n" + s[-limit // 2:]


def sight(f: dict) -> dict:
    """Record one finding. `f`: fingerprint, key (unique per occurrence), title, description,
    category, severity, priority, min_occurrences, at, and where it was seen (project_id, agent,
    model, objective_id, objective_title, record_id, tool), script, evidence, context.

    Returns {"id", "new": created now, "counted": a new sighting, "reopened", "visible"}."""
    now = time.time()
    at = float(f.get("at") or now)
    with _lock:
        con = db()
        row = con.execute("SELECT * FROM bugs WHERE fingerprint=?", (f["fingerprint"],)).fetchone()
        new = row is None
        if new:
            cur = con.execute(
                "INSERT INTO bugs (fingerprint, title, description, category, severity, priority, status, source, visible, "
                "min_occurrences, project_id, agent, model, objective_id, objective_title, tool, script, evidence, context, "
                "occurrences, first_seen, last_seen, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'open', ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)",
                (f["fingerprint"], f["title"][:300], f.get("description") or "", f.get("category") or "error",
                 f.get("severity") or "medium", f.get("priority") or "P3", f.get("source") or "monitor",
                 int(f.get("min_occurrences") or 1), f.get("project_id"), f.get("agent"), f.get("model"),
                 f.get("objective_id"), f.get("objective_title"), f.get("tool"), _clip(f.get("script")),
                 _clip(f.get("evidence")), json.dumps(f.get("context") or {}, default=str), at, at, now, now))
            row = con.execute("SELECT * FROM bugs WHERE id=?", (cur.lastrowid,)).fetchone()
        bid = row["id"]
        # The detector's severity and priority are the calibrated ones. An earlier build let the
        # triage model overwrite them (a small model marked nearly everything "critical"); every
        # scan puts them back unless the operator set them.
        if not new and not row["edited"] and row["source"] == "monitor" and f.get("severity") and f.get("priority") \
                and (row["severity"], row["priority"]) != (f["severity"], f["priority"]):
            con.execute("UPDATE bugs SET severity=?, priority=? WHERE id=?", (f["severity"], f["priority"], bid))
        try:
            con.execute("INSERT INTO bug_sightings (bug_id, key, at, agent, model, project_id, objective_id, record_id, "
                        "evidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (bid, str(f.get("key") or at), at, f.get("agent"), f.get("model"), f.get("project_id"),
                         f.get("objective_id"), f.get("record_id"), _clip(f.get("evidence"), 2000)))
        except sqlite3.IntegrityError:           # this occurrence is already counted
            con.commit()
            return {"id": bid, "new": new, "counted": False, "reopened": False, "visible": bool(row["visible"])}
        occ = row["occurrences"] + 1
        visible = 1 if occ >= row["min_occurrences"] else 0
        reopened = row["status"] == "closed" and row["closed_at"] is not None and at > row["closed_at"]
        sets = {"occurrences": occ, "visible": visible, "updated_at": now,
                "first_seen": min(row["first_seen"] or at, at), "last_seen": max(row["last_seen"] or at, at)}
        if not new and at >= (row["last_seen"] or 0):
            # The latest occurrence is the most useful one to look at.
            for k in ("agent", "model", "project_id", "objective_id", "objective_title", "tool"):
                if f.get(k):
                    sets[k] = f[k]
            if f.get("script"):
                sets["script"] = _clip(f["script"])
            if f.get("evidence"):
                sets["evidence"] = _clip(f["evidence"])
            if f.get("context"):
                sets["context"] = json.dumps(f["context"], default=str)
        notes = row["notes"]
        if visible and not row["visible"]:
            where = " / ".join(str(f[k]) for k in ("agent", "tool") if f.get(k)) or "the logs"
            if (f.get("source") or "monitor") == "manual":
                notes = _noted(notes, "filed by hand")
            else:
                notes = _noted(notes, f"filed by the monitor: {'first seen' if occ == 1 else f'seen {occ} times, latest'} "
                                      f"in {where}" + (f" (model {f['model']})" if f.get("model")
                                                       and f.get("model") != f.get("agent") else ""))
        if reopened:
            who = f" in {f['agent']}" if f.get("agent") else ""
            notes = _noted(notes, f"reopened by the monitor: seen again{who} after it was closed", at)
            sets.update(status="open", closed_at=None, closed_by=None)
        if notes != row["notes"]:
            sets["notes"] = notes
        con.execute(f"UPDATE bugs SET {', '.join(f'{k}=?' for k in sets)} WHERE id=?", (*sets.values(), bid))
        # Keep the newest MAX_SIGHTINGS sightings; the count above is the total.
        con.execute("DELETE FROM bug_sightings WHERE bug_id=? AND id NOT IN (SELECT id FROM bug_sightings WHERE bug_id=? "
                    "ORDER BY at DESC LIMIT ?)", (bid, bid, MAX_SIGHTINGS))
        con.commit()
    return {"id": bid, "new": new, "counted": True, "reopened": reopened, "visible": bool(visible)}


def enrich(bid: int, fields: dict, model: str | None = None) -> None:
    """The monitor's model triage: better title/description, a suggested fix, severity and
    priority. What an operator already edited by hand is left alone."""
    allowed = {"title", "description", "suggestion", "severity", "priority"}
    with _lock:
        row = db().execute("SELECT edited, notes FROM bugs WHERE id=?", (bid,)).fetchone()
        if row is None:
            return
        sets: dict[str, Any] = {"triaged": 1, "updated_at": time.time(),
                                "notes": _noted(row["notes"], f"written up by {model or 'the monitor model'}"
                                                              + (" (suggested fix only: you edited the rest)"
                                                                 if row["edited"] else ""))}
        if not row["edited"]:
            for k, v in fields.items():
                if k not in allowed or not isinstance(v, str) or not v.strip():
                    continue
                if (k == "severity" and v not in SEVERITIES) or (k == "priority" and v not in PRIORITIES):
                    continue
                sets[k] = v.strip()[:300] if k == "title" else v.strip()[:8000]
        elif fields.get("suggestion"):
            sets["suggestion"] = str(fields["suggestion"])[:8000]
        db().execute(f"UPDATE bugs SET {', '.join(f'{k}=?' for k in sets)} WHERE id=?", (*sets.values(), bid))
        db().commit()


def watched() -> list[dict]:
    """Open and pending bugs the monitor filed: the ones it checks for a fix."""
    with _lock:
        rows = db().execute("SELECT id, fingerprint, title, tool, model, project_id, last_seen, status FROM bugs "
                            "WHERE visible=1 AND source='monitor' AND status IN ('open', 'pending')").fetchall()
    return [dict(r) for r in rows]


def close_fixed(bid: int, note: str) -> None:
    """The monitor saw the problem stop: closed, with the evidence in the notes. A new
    occurrence reopens it (sight())."""
    now = time.time()
    with _lock:
        row = db().execute("SELECT notes, status FROM bugs WHERE id=?", (bid,)).fetchone()
        if row is None or row["status"] == "closed":
            return
        notes = _noted(row["notes"], f"closed by the monitor as fixed: {note}", now)
        db().execute("UPDATE bugs SET status='closed', closed_at=?, closed_by='monitor', notes=?, updated_at=? "
                     "WHERE id=?", (now, notes, now, bid))
        db().commit()


def mark_triaged(bid: int) -> None:
    with _lock:
        db().execute("UPDATE bugs SET triaged=1 WHERE id=?", (bid,))
        db().commit()


# ---------------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------------
_LIST_COLS = ("id", "title", "category", "severity", "priority", "status", "source", "project_id", "agent", "model",
              "objective_id", "objective_title", "tool", "occurrences", "first_seen", "last_seen", "created_at",
              "updated_at", "closed_at", "closed_by", "triaged")
_SEV_ORDER = "CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END"


def list_bugs(status: str = "open", q: str = "", project_id: str | None = None, limit: int = 500) -> list[dict]:
    where, args = ["visible=1"], []
    if status in STATUSES:
        where.append("status=?")
        args.append(status)
    if project_id:
        where.append("(project_id=? OR project_id IS NULL)")
        args.append(project_id)
    if q.strip():
        where.append("(title LIKE ? OR description LIKE ? OR evidence LIKE ? OR model LIKE ? OR tool LIKE ?)")
        args += [f"%{q.strip()}%"] * 5
    with _lock:
        rows = db().execute(f"SELECT {', '.join(_LIST_COLS)} FROM bugs WHERE {' AND '.join(where)} "
                            f"ORDER BY priority, {_SEV_ORDER}, last_seen DESC LIMIT ?", (*args, limit)).fetchall()
    return [dict(r) for r in rows]


def counts(project_id: str | None = None) -> dict:
    where, args = "visible=1", []
    if project_id:
        where += " AND (project_id=? OR project_id IS NULL)"
        args.append(project_id)
    with _lock:
        rows = db().execute(f"SELECT status, COUNT(*) FROM bugs WHERE {where} GROUP BY status", args).fetchall()
    out = {s: 0 for s in STATUSES}
    out.update({r[0]: r[1] for r in rows})
    return out


def get_bug(bid: int) -> dict:
    with _lock:
        row = db().execute("SELECT * FROM bugs WHERE id=?", (bid,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="no such bug")
        sightings = db().execute("SELECT at, agent, model, project_id, objective_id, record_id, evidence FROM bug_sightings "
                                 "WHERE bug_id=? ORDER BY at DESC", (bid,)).fetchall()
    out = dict(row)
    try:
        out["context"] = json.loads(out.get("context") or "{}")
    except ValueError:
        out["context"] = {}
    out["sightings"] = [dict(s) for s in sightings]
    return out


def untriaged(limit: int) -> list[dict]:
    with _lock:
        rows = db().execute("SELECT id FROM bugs WHERE visible=1 AND triaged=0 AND source='monitor' AND status!='closed' "
                            "ORDER BY priority, last_seen DESC LIMIT ?", (limit,)).fetchall()
    return [get_bug(r["id"]) for r in rows]


def open_titles(limit: int = 40) -> list[str]:
    with _lock:
        rows = db().execute("SELECT id, title FROM bugs WHERE visible=1 AND status!='closed' ORDER BY last_seen DESC "
                            "LIMIT ?", (limit,)).fetchall()
    return [f"#{r['id']} {r['title']}" for r in rows]


# ---------------------------------------------------------------------------------------------
# Routes (under /api)
# ---------------------------------------------------------------------------------------------
class BugPatch(BaseModel):
    status: Literal["open", "pending", "closed"] | None = None
    severity: Literal["critical", "high", "medium", "low"] | None = None
    priority: Literal["P1", "P2", "P3", "P4"] | None = None
    title: str | None = Field(None, max_length=300)
    description: str | None = Field(None, max_length=20_000)
    notes: str | None = Field(None, max_length=20_000)


class BugNew(BaseModel):
    title: str = Field(..., min_length=3, max_length=300)
    description: str = Field("", max_length=20_000)
    severity: Literal["critical", "high", "medium", "low"] = "medium"
    priority: Literal["P1", "P2", "P3", "P4"] = "P3"
    category: str = Field("error", max_length=40)
    project_id: str | None = Field(None, max_length=200)
    script: str = Field("", max_length=MAX_TEXT)
    evidence: str = Field("", max_length=MAX_TEXT)


def patch_bug(bid: int, req: BugPatch) -> dict:
    fields = req.model_dump(exclude_none=True)
    if not fields:
        return get_bug(bid)
    now = time.time()
    with _lock:
        row = db().execute("SELECT status, notes FROM bugs WHERE id=?", (bid,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="no such bug")
        if "status" in fields and fields["status"] != row["status"]:
            fields["closed_at"] = now if fields["status"] == "closed" else None
            fields["closed_by"] = "operator" if fields["status"] == "closed" else None
            # Appended to what the operator's own edit (if any) left in the notes.
            fields["notes"] = _noted(fields.get("notes", row["notes"]),
                                     f"status {row['status']} -> {fields['status']} (by the operator)", now)
        if {"title", "description", "severity", "priority"} & fields.keys():
            fields["edited"] = 1                 # the model's triage no longer overwrites these
        fields["updated_at"] = now
        db().execute(f"UPDATE bugs SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?", (*fields.values(), bid))
        db().commit()
    return get_bug(bid)


@router.get("/bugs")
async def bugs_list(status: str = Query("open"), q: str = Query("", max_length=200),
                    project_id: str | None = Query(None)) -> dict:
    """Visible bugs, most urgent first. status: open | pending | closed | all."""
    import asyncio

    rows = await asyncio.to_thread(list_bugs, status, q, project_id)
    return {"bugs": rows, "counts": await asyncio.to_thread(counts, project_id)}


@router.get("/bugs/counts")
async def bugs_counts(project_id: str | None = Query(None)) -> dict:
    return counts(project_id)


@router.post("/bugs")
async def bugs_create(req: BugNew) -> dict:
    """A bug filed by hand (never merged with another: its fingerprint is unique)."""
    now = time.time()
    out = sight({"fingerprint": f"manual:{now:.6f}", "key": "manual", "title": req.title, "description": req.description,
                 "severity": req.severity, "priority": req.priority, "category": req.category, "source": "manual",
                 "project_id": req.project_id, "script": req.script, "evidence": req.evidence, "at": now})
    with _lock:
        db().execute("UPDATE bugs SET triaged=1, edited=1 WHERE id=?", (out["id"],))
        db().commit()
    return get_bug(out["id"])


@router.get("/bugs/{bid}")
async def bugs_get(bid: int) -> dict:
    return get_bug(bid)


@router.patch("/bugs/{bid}")
async def bugs_patch(bid: int, req: BugPatch) -> dict:
    return patch_bug(bid, req)


@router.delete("/bugs/{bid}")
async def bugs_delete(bid: int) -> dict:
    """Forget a bug. If the monitor sees the problem again it is filed afresh; close it instead
    to keep it closed until it recurs."""
    with _lock:
        cur = db().execute("DELETE FROM bugs WHERE id=?", (bid,))
        db().execute("DELETE FROM bug_sightings WHERE bug_id=?", (bid,))
        db().commit()
    if not cur.rowcount:
        raise HTTPException(status_code=404, detail="no such bug")
    return {"ok": True}
