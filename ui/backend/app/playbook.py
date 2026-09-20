"""The swarm's playbook: the instructions every agent reads at the start of every iteration.

Two parts, with different owners:

* **Charter** -- the operator's standing instructions: how the team works together, what to
  review before building, what "good work" means here. Edited on the Swarm page; a sensible
  default ships so a new project starts with one.
* **Team practices** -- written BY THE AGENTS. Every PRACTICES_EVERY candidates one agent
  gets the chore of rewriting it from the evidence: the lessons, which candidates improved
  the champion and how they were made, which library modules earned "works", what kept
  failing. It is about HOW to work (process, sequencing, what to check first), where lessons
  are about WHAT worked -- so the agents improve their own instructions, and every agent
  reads the improved version on its next iteration. Versions are kept; the operator can read,
  edit or roll back any of them.
* **Pitfalls** -- the operator's standing "never do this again" list, grown by demoting a
  candidate (see objectives.demote). It is separate from the practices ON PURPOSE: the
  practices are rewritten wholesale by an agent every PRACTICES_EVERY candidates, and lessons
  are periodically consolidated down to fifteen, so a finding recorded in either can be
  summarised away. A leak the operator had to catch by hand -- look-ahead bias an automated
  check missed -- is exactly the knowledge that must not evaporate, so it lives here, is
  never rewritten by an agent, and is read on every iteration of every objective in the
  project.

Stored in objectives.sqlite3 beside the objectives and the library.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import projects

router = APIRouter(tags=["playbook"])

PRACTICES_EVERY = 10  # candidates between rewrites of the team practices
PRACTICES_LEASE_S = 900

DEFAULT_CHARTER = """\
# Swarm charter

You are one of several agents working on the same objective at the same time. You share one
message board, one code library, one set of lessons and one leaderboard. The team gets
better only if each of you builds on what the others found -- so work as a team, not alone.

## Before you build anything
1. Read the brief: the objective, how it is scored, the operator's steering, the lessons,
   the leaderboard, recent attempts and the team practices below. Steering from the
   operator overrides everything else.
2. Read the board with team_board: what teammates just tried, what failed and why, and what
   they announced in #planning. Do not repeat an attempt that already failed unless you
   can say what you are changing and why it should now work.
3. Review the code library with library_list / library_get. Prefer building on modules with
   `works` verdicts and good evidence (used by ranked candidates, no look-ahead failures).
   Read the `broken` comments before using a module -- they are someone's lost iteration.
4. Check the regime map and the forecast features' measured skill before relying on them.

## Coordinate
5. Post your plan to #planning with team_post BEFORE the expensive work: one or two lines
   -- the hypothesis, the timeframe, the regime and the library modules you will use. If a
   teammate is already on the same idea, pick a different one or explicitly build on theirs.
6. Split the search: when the team is exploiting one idea, explore; when everything is
   exploration, improve the leader. Different timeframes, regimes and signal families are
   all useful directions.

## Build reusable, tested pieces
7. Put reusable parts (regime detectors, signals, filters, sizing and risk rules) in the
   library with library_save, with a test, instead of burying them in one script. Improve
   an existing module (a new version) rather than writing a near-duplicate.
8. Keep everything causal: a decision at bar t uses rows up to and including t only.

## Leave the team smarter than you found it
9. After your evaluation, comment on every library module you used: `works` or `broken`,
   with the candidate number and the numbers as evidence.
10. Write a specific lesson (KEEP / AVOID / TRY): features, parameters, timeframe, regime,
    costs, what the numbers showed. "It did not work" teaches nobody anything.
11. Post notable findings to #results or #planning so teammates see them now, not only in
    the next brief.
"""


def _db():
    from .objectives import _lock, db

    conn = db()
    with _lock:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS playbooks (
                project_id TEXT PRIMARY KEY, charter TEXT NOT NULL, practices TEXT NOT NULL DEFAULT '',
                practices_version INTEGER NOT NULL DEFAULT 0, practices_at REAL NOT NULL DEFAULT 0,
                practices_candidates INTEGER NOT NULL DEFAULT 0, practices_lock_until REAL NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS playbook_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL, part TEXT NOT NULL,
                version INTEGER NOT NULL, text TEXT NOT NULL, author TEXT, ts REAL NOT NULL, note TEXT NOT NULL DEFAULT ''
            );
            """
        )
        # Added after the first release: widen an existing database in place.
        if "pitfalls" not in {r[1] for r in conn.execute("PRAGMA table_info(playbooks)")}:
            conn.execute("ALTER TABLE playbooks ADD COLUMN pitfalls TEXT NOT NULL DEFAULT ''")
            conn.commit()
    return conn


def _lock():
    from .objectives import _lock as lock

    return lock


def get(project_id: str) -> dict:
    conn = _db()
    with _lock():
        r = conn.execute("SELECT * FROM playbooks WHERE project_id=?", (project_id,)).fetchone()
        if r is None:
            now = time.time()
            conn.execute("INSERT INTO playbooks (project_id, charter, updated_at) VALUES (?,?,?)",
                         (project_id, DEFAULT_CHARTER, now))
            conn.execute("INSERT INTO playbook_history (project_id, part, version, text, author, ts, note) "
                         "VALUES (?,?,?,?,?,?,?)", (project_id, "charter", 1, DEFAULT_CHARTER, "default", now, "default charter"))
            conn.commit()
            r = conn.execute("SELECT * FROM playbooks WHERE project_id=?", (project_id,)).fetchone()
    return dict(r)


def project_candidates(project_id: str) -> int:
    conn = _db()
    with _lock():
        return conn.execute("SELECT count(*) FROM candidates c JOIN objectives o ON o.id=c.objective_id "
                            "WHERE o.project_id=?", (project_id,)).fetchone()[0]


def practices_due(project_id: str) -> bool:
    """Hand out the practices rewrite as a leased chore once enough new evidence exists."""
    pb = get(project_id)
    n = project_candidates(project_id)
    now = time.time()
    if n - pb["practices_candidates"] < PRACTICES_EVERY or pb["practices_lock_until"] > now:
        return False
    conn = _db()
    with _lock():
        conn.execute("UPDATE playbooks SET practices_lock_until=? WHERE project_id=?", (now + PRACTICES_LEASE_S, project_id))
        conn.commit()
    return True


def add_pitfall(project_id: str, text: str, author: str, note: str = "") -> dict:
    """Append one entry to the operator's pitfalls list, newest last, and version it.

    Appended rather than replaced: each demotion is separate evidence, and the operator is
    the only writer, so there is nothing to merge."""
    pb = get(project_id)
    entry = f"- [{time.strftime('%Y-%m-%d')}] {text.strip()}"
    current = (pb.get("pitfalls") or "").strip()
    return _save(project_id, "pitfalls", f"{current}\n{entry}".strip() if current else entry,
                 author, note or "pitfall recorded on demotion")


def _save(project_id: str, part: str, text: str, author: str, note: str) -> dict:
    pb = get(project_id)
    conn = _db()
    now = time.time()
    with _lock():
        ver = (conn.execute("SELECT max(version) FROM playbook_history WHERE project_id=? AND part=?",
                            (project_id, part)).fetchone()[0] or 0) + 1
        conn.execute("INSERT INTO playbook_history (project_id, part, version, text, author, ts, note) VALUES (?,?,?,?,?,?,?)",
                     (project_id, part, ver, text, author, now, note))
        if part == "charter":
            conn.execute("UPDATE playbooks SET charter=?, updated_at=? WHERE project_id=?", (text, now, project_id))
        elif part == "pitfalls":
            conn.execute("UPDATE playbooks SET pitfalls=?, updated_at=? WHERE project_id=?", (text, now, project_id))
        else:
            conn.execute("UPDATE playbooks SET practices=?, practices_version=?, practices_at=?, practices_candidates=?, "
                         "practices_lock_until=0, updated_at=? WHERE project_id=?",
                         (text, ver, now, project_candidates(project_id) if author != "operator" else pb["practices_candidates"],
                          now, project_id))
        conn.commit()
    return get(project_id)


def _project(project_id: str) -> None:
    if projects.get(project_id) is None:
        raise HTTPException(status_code=404, detail=f"no project {project_id!r}")


@router.get("/projects/{project_id}/playbook")
async def read(project_id: str) -> dict:
    _project(project_id)
    pb = get(project_id)
    conn = _db()
    with _lock():
        hist = [dict(r) for r in conn.execute(
            "SELECT id, part, version, author, ts, note, length(text) AS chars FROM playbook_history "
            "WHERE project_id=? ORDER BY ts DESC LIMIT 60", (project_id,)).fetchall()]
    return {**pb, "history": hist, "practices_every": PRACTICES_EVERY,
            "candidates_since_practices": project_candidates(project_id) - pb["practices_candidates"]}


@router.get("/projects/{project_id}/playbook/history/{hid}")
async def history_item(project_id: str, hid: int) -> dict:
    conn = _db()
    with _lock():
        r = conn.execute("SELECT * FROM playbook_history WHERE project_id=? AND id=?", (project_id, hid)).fetchone()
    if r is None:
        raise HTTPException(status_code=404, detail="no such version")
    return dict(r)


class Update(BaseModel):
    part: str = Field(..., pattern="^(charter|practices)$")
    text: str = Field(..., max_length=40_000)
    author: str = Field("operator", max_length=200)
    note: str = Field("", max_length=2000)


@router.put("/projects/{project_id}/playbook")
async def update(project_id: str, req: Update) -> dict:
    """The operator edits the charter (or the practices); agents rewrite the practices."""
    _project(project_id)
    if req.part == "charter" and req.author != "operator":
        raise HTTPException(status_code=403, detail="the charter is the operator's; agents rewrite the practices")
    return _save(project_id, req.part, req.text.strip(), req.author, req.note)
