"""A project's code library: reusable modules the swarm writes, tests, uses and reviews.

Candidates are whole scripts, so without a library every good idea -- an indicator, a regime
filter, a position-sizing rule -- lives only inside the candidate that had it, and the next
agent re-derives it (often worse). The library makes those pieces first-class:

* **Modules** are plain Python files, ``from lib import vwap_bands``. Saving one runs a smoke
  test in the sandbox (import it, plus any test code the agent gives, against in-sample data)
  and keeps every version, so a bad edit never loses the working one.
* **Evidence is recorded, not claimed.** Every evaluated candidate that imports a module is
  linked to it with the version used and its outcome, so a module's page shows how many
  candidates used it, how they scored, and whether any failed the look-ahead test.
* **Comments** -- ``works`` / ``broken`` / ``note`` -- from agents and from the operator, each
  optionally tied to a candidate as its proof. Agents read them before building on a module.
* **Kinds and regimes.** A module is a ``regime`` detector (``detect(df) -> labels``), a
  ``signal`` (``signal(df) -> positions``), a ``risk`` rule or a ``util``. ``regime_map`` measures
  every signal module inside every regime a detector finds -- in the sandbox, on in-sample data
  only -- and stores the table, so a strategy can route each regime to the functions that
  actually work there (``ft.route``) and the next agent starts from what was learned.

Modules only ever execute inside the sandbox (as part of a candidate or a smoke test); the
control plane stores them as text. Storage shares objectives.sqlite3.
"""

from __future__ import annotations

import re
import time
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import projects

router = APIRouter(tags=["library"])

NAME_RE = re.compile(r"^[a-z_][a-z0-9_]{0,47}$")
# The parenthesised form may span lines; the bare form must not, or `\s` walks past the
# newline and swallows the first word of the next statement as if it were an imported name.
IMPORT_RE = re.compile(
    r"from\s+lib\s+import\s+\(([^)]*)\)"
    r"|from\s+lib\s+import\s+([^\n(#]+)"
    r"|from\s+lib\.(\w+)\s+import"
    r"|import\s+lib\.(\w+)"
)
MAX_MODULE_CHARS = 100_000


_ready = False


def _db():
    global _ready
    from .objectives import _lock, db

    conn = db()
    if _ready:
        return conn
    _ready = True
    with _lock:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS lib_modules (
                project_id TEXT NOT NULL, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
                version INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'active',
                kind TEXT NOT NULL DEFAULT 'util',
                author TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                PRIMARY KEY (project_id, name)
            );
            CREATE TABLE IF NOT EXISTS lib_versions (
                project_id TEXT NOT NULL, name TEXT NOT NULL, version INTEGER NOT NULL,
                code TEXT NOT NULL, author TEXT, ts REAL NOT NULL, note TEXT NOT NULL DEFAULT '',
                test_ok INTEGER, test_output TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (project_id, name, version)
            );
            CREATE TABLE IF NOT EXISTS lib_comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL, name TEXT NOT NULL,
                version INTEGER, ts REAL NOT NULL, author TEXT NOT NULL,
                verdict TEXT NOT NULL, text TEXT NOT NULL, candidate_id TEXT
            );
            CREATE TABLE IF NOT EXISTS regime_maps (
                id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL, objective_id TEXT,
                regime TEXT NOT NULL, regime_version INTEGER NOT NULL, ts REAL NOT NULL,
                author TEXT, result TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS field_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL, objective_id TEXT,
                ts REAL NOT NULL, author TEXT, params TEXT NOT NULL, result TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS lib_usage (
                candidate_id TEXT NOT NULL, project_id TEXT NOT NULL, name TEXT NOT NULL,
                version INTEGER NOT NULL, PRIMARY KEY (candidate_id, name)
            );
            """
        )
        # Added after the first release: widen an existing database in place. `warning` holds
        # why a module was quarantined, so the reason travels with the module the way a
        # demotion's finding travels with the candidate.
        if "warning" not in {r[1] for r in conn.execute("PRAGMA table_info(lib_modules)")}:
            conn.execute("ALTER TABLE lib_modules ADD COLUMN warning TEXT NOT NULL DEFAULT ''")
            conn.commit()
    return conn


def _lock():
    from .objectives import _lock as lock

    return lock


def imported_modules(code: str) -> list[str]:
    """Library module names a script imports."""
    names: set[str] = set()
    for m in IMPORT_RE.finditer(code or ""):
        for clause in (m.group(1) or m.group(2) or "").split(","):
            # `from lib import x as y` imports x; the alias is a local name, not a module.
            name = clause.strip().split(" as ")[0].strip()
            if NAME_RE.match(name):
                names.add(name)
        for g in (m.group(3), m.group(4)):
            if g:
                names.add(g)
    return sorted(names)


def quarantine_banner(name: str, warning: str) -> str:
    """The header stamped onto a quarantined module's source."""
    return (
        '"""!!! QUARANTINED -- DO NOT USE THIS MODULE !!!\n\n'
        f"{name} produced a result that was disqualified on review. Importing it reproduces\n"
        "the same defect and your candidate will be disqualified too. Fix the cause in a NEW\n"
        "module and save that instead; do not copy this code forward unchanged.\n\n"
        f"Why it was quarantined:\n{warning.strip()}\n"
        '"""\n\n'
    )


def module_files(project_id: str) -> dict[str, str]:
    """{relative path: source} of every LOADABLE module, for a sandbox run (/work/.ft/lib/...).

    A quarantined module still ships, so a run that imports one fails on its merits rather
    than on an ImportError nobody can read -- but its source is stamped with the warning, so
    an agent that opens it cannot miss why it must not be built on. Retired modules are gone.
    """
    conn = _db()
    with _lock():
        rows = conn.execute(
            "SELECT m.name, v.code, m.status, m.warning FROM lib_modules m "
            "JOIN lib_versions v ON v.project_id=m.project_id AND v.name=m.name AND v.version=m.version "
            "WHERE m.project_id=? AND m.status IN ('active','quarantined')",
            (project_id,)).fetchall()
    if not rows:
        return {}
    files = {".ft/lib/__init__.py": '"""The project code library (read-only copy for this run)."""\n'}
    for name, code, status, warning in rows:
        files[f".ft/lib/{name}.py"] = (
            quarantine_banner(name, warning) + code if status == "quarantined" else code)
    return files


def module_sources(project_id: str) -> dict[str, tuple[str, str]]:
    """{name: (source, status)} for EVERY module, whatever its status.

    Distinct from `module_files`, which decides what a sandbox run may import. Auditing a
    result has the opposite requirement: it must see exactly what that result was built on,
    including a module since retired. Reviewing a candidate against a library that silently
    omits its main signal produced reviews complaining the code was "unsupplied".
    """
    conn = _db()
    with _lock():
        rows = conn.execute(
            "SELECT m.name, v.code, m.status FROM lib_modules m "
            "JOIN lib_versions v ON v.project_id=m.project_id AND v.name=m.name AND v.version=m.version "
            "WHERE m.project_id=?", (project_id,)).fetchall()
    return {r["name"]: (r["code"], r["status"]) for r in rows}


def reachable_modules(project_id: str, code: str) -> dict[str, str]:
    """{name: source} of every project module `code` can reach, directly or through another.

    The swarm composes modules -- a candidate imports a signal that imports a regime filter --
    so the set that produced a result is the closure, not just the first hop. Used both to
    show a reviewer everything that ran and to quarantine everything a bad result rests on.
    Retired and quarantined modules are included: what the result used is what must be read.
    """
    wanted = set(imported_modules(code or ""))
    if not wanted:
        return {}
    by_name = {name: src for name, (src, _status) in module_sources(project_id).items()}
    out: dict[str, str] = {}
    while wanted:
        name = wanted.pop()
        src = by_name.get(name)
        if src is None or name in out:
            continue
        out[name] = src
        wanted |= {d for d in imported_modules(src) if d in by_name and d not in out}
    return out


def quarantine(project_id: str, names: list[str], warning: str, reviewer: str,
               candidate_id: str | None = None, seq: int | None = None) -> list[str]:
    """Mark the modules a disqualified result was built on, so the team stops building on them.

    The module is not deleted and not retired: it stays visible, with the reason attached, so
    an agent reading the library learns what went wrong instead of finding a hole where a
    module used to be. Returns the names that were actually quarantined.
    """
    if not names:
        return []
    conn = _db()
    hit = []
    with _lock():
        for name in names:
            row = conn.execute("SELECT version, status FROM lib_modules WHERE project_id=? AND name=?",
                               (project_id, name)).fetchone()
            if row is None:
                continue
            conn.execute("UPDATE lib_modules SET status='quarantined', warning=?, updated_at=? "
                         "WHERE project_id=? AND name=?",
                         (warning[:4000], time.time(), project_id, name))
            conn.execute("INSERT INTO lib_comments (project_id, name, version, ts, author, verdict, text, "
                         "candidate_id) VALUES (?,?,?,?,?,?,?,?)",
                         (project_id, name, row["version"], time.time(), reviewer, "quarantined",
                          (f"Quarantined after #{seq} was disqualified. " if seq else "Quarantined. ")
                          + warning[:4000], candidate_id))
            hit.append(name)
        conn.commit()
    return hit


def record_usage(project_id: str, candidate_id: str, code: str) -> list[dict]:
    """Link a candidate to the library versions it imported. Returns [{name, version}]."""
    names = imported_modules(code)
    if not names:
        return []
    conn = _db()
    used = []
    with _lock():
        for name in names:
            r = conn.execute("SELECT version FROM lib_modules WHERE project_id=? AND name=?",
                             (project_id, name)).fetchone()
            if r is None:
                continue
            conn.execute("INSERT OR REPLACE INTO lib_usage (candidate_id, project_id, name, version) VALUES (?,?,?,?)",
                         (candidate_id, project_id, name, r[0]))
            used.append({"name": name, "version": r[0]})
        conn.commit()
    return used


def _evidence(project_id: str, name: str) -> dict:
    """What the candidates that used this module actually achieved."""
    conn = _db()
    with _lock():
        rows = conn.execute(
            "SELECT c.id, c.seq, c.objective_id, c.status, c.score, c.is_score, c.lookahead, c.audit, "
            "c.champion_at, u.version FROM lib_usage u JOIN candidates c ON c.id=u.candidate_id "
            "WHERE u.project_id=? AND u.name=? ORDER BY c.created_at DESC",
            (project_id, name)).fetchall()
    uses = [dict(r) for r in rows]
    ok = [u for u in uses if u["status"] == "ok" and u["lookahead"] != "fail"]
    best_is = max((u["is_score"] for u in ok if u["is_score"] is not None), default=None)
    best = max((u["score"] for u in ok if u["score"] is not None), default=None)
    return {
        "uses": len(uses),
        "ok": len(ok),
        "errors": sum(1 for u in uses if u["status"] == "error"),
        "lookahead_fails": sum(1 for u in uses if u["lookahead"] == "fail"),
        "champions": sum(1 for u in uses if u["champion_at"]),
        "best_in_sample": best_is,
        "best_holdout": best,
        "recent": uses[:15],
    }


def _comment_counts(project_id: str, name: str) -> dict:
    conn = _db()
    with _lock():
        rows = conn.execute("SELECT verdict, count(*) FROM lib_comments WHERE project_id=? AND name=? GROUP BY verdict",
                            (project_id, name)).fetchall()
    return {r[0]: r[1] for r in rows}


def list_modules(project_id: str, include_retired: bool = True) -> list[dict]:
    conn = _db()
    with _lock():
        rows = conn.execute(
            "SELECT m.*, v.test_ok FROM lib_modules m JOIN lib_versions v ON v.project_id=m.project_id "
            "AND v.name=m.name AND v.version=m.version WHERE m.project_id=? ORDER BY m.updated_at DESC",
            (project_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if not include_retired and d["status"] != "active":
            continue
        ev = _evidence(project_id, d["name"])
        ev.pop("recent")
        d["evidence"] = ev
        d["comments"] = _comment_counts(project_id, d["name"])
        out.append(d)
    return out


def get_module(project_id: str, name: str, version: int | None = None) -> dict:
    conn = _db()
    with _lock():
        m = conn.execute("SELECT * FROM lib_modules WHERE project_id=? AND name=?", (project_id, name)).fetchone()
        if m is None:
            raise HTTPException(status_code=404, detail=f"no library module {name!r}")
        v = conn.execute("SELECT * FROM lib_versions WHERE project_id=? AND name=? AND version=?",
                         (project_id, name, version or m["version"])).fetchone()
        versions = [dict(r) for r in conn.execute(
            "SELECT version, author, ts, note, test_ok FROM lib_versions WHERE project_id=? AND name=? ORDER BY version DESC",
            (project_id, name)).fetchall()]
        comments = [dict(r) for r in conn.execute(
            "SELECT * FROM lib_comments WHERE project_id=? AND name=? ORDER BY ts DESC LIMIT 100",
            (project_id, name)).fetchall()]
    if v is None:
        raise HTTPException(status_code=404, detail=f"{name} has no version {version}")
    return {**dict(m), "shown_version": v["version"], "code": v["code"], "test_ok": v["test_ok"],
            "test_output": v["test_output"], "version_note": v["note"], "versions": versions,
            "comments": comments, "evidence": _evidence(project_id, name),
            "regime_map": latest_regime_map(project_id, name) if m["kind"] == "regime" else None}


# =======================================================================================
# Routes
# =======================================================================================
def _project(project_id: str) -> dict:
    p = projects.get(project_id)
    if p is None:
        raise HTTPException(status_code=404, detail=f"no project {project_id!r}")
    return p


@router.get("/projects/{project_id}/library")
async def library(project_id: str) -> dict:
    _project(project_id)
    return {"modules": list_modules(project_id)}


@router.get("/projects/{project_id}/library/{name}")
async def module(project_id: str, name: str, version: int | None = None) -> dict:
    _project(project_id)
    return get_module(project_id, name, version)


class SaveModule(BaseModel):
    name: str = Field(..., max_length=48)
    code: str = Field(..., min_length=1, max_length=MAX_MODULE_CHARS)
    description: str = Field("", max_length=2000)
    kind: Literal["regime", "signal", "risk", "util"] = "util"
    note: str = Field("", max_length=2000, description="what changed in this version")
    test_code: str = Field("", max_length=50_000)
    author: str = Field("", max_length=200)
    objective_id: str | None = None


@router.post("/projects/{project_id}/library")
async def save_module(project_id: str, req: SaveModule) -> dict:
    """Add a module or a new version of one. The smoke test runs first; a module that does
    not even import is refused, so the library never holds code that cannot run."""
    project = _project(project_id)
    if not NAME_RE.match(req.name) or req.name in ("ft", "lib"):
        raise HTTPException(status_code=400, detail="name must be a lowercase python identifier (a-z, 0-9, _)")
    contract = {"regime": "def detect(", "signal": "def signal("}.get(req.kind)
    if contract and contract not in req.code:
        raise HTTPException(status_code=400, detail=(
            f"a {req.kind} module must define {contract[4:-1]}(df, ...) -- "
            + ("returning one regime label per row" if req.kind == "regime" else "returning one position per row")))
    ok, output = await smoke_test(project, req)
    if not ok:
        return {"saved": False, "test_ok": False, "test_output": output[-4000:],
                "error": "the smoke test failed -- fix the module and save again"}
    conn = _db()
    now = time.time()
    with _lock():
        cur = conn.execute("SELECT version FROM lib_modules WHERE project_id=? AND name=?", (project_id, req.name)).fetchone()
        version = (cur[0] + 1) if cur else 1
        conn.execute("INSERT INTO lib_versions (project_id, name, version, code, author, ts, note, test_ok, test_output) "
                     "VALUES (?,?,?,?,?,?,?,?,?)",
                     (project_id, req.name, version, req.code, req.author, now, req.note, 1, output[-8000:]))
        if cur:
            conn.execute("UPDATE lib_modules SET version=?, updated_at=?, status='active', kind=?"
                         + (", description=?" if req.description else "") + " WHERE project_id=? AND name=?",
                         (version, now, req.kind, *([req.description] if req.description else []), project_id, req.name))
        else:
            conn.execute("INSERT INTO lib_modules (project_id, name, description, kind, version, author, created_at, updated_at) "
                         "VALUES (?,?,?,?,?,?,?,?)", (project_id, req.name, req.description, req.kind, version, req.author, now, now))
        conn.commit()
    return {"saved": True, "name": req.name, "version": version, "test_ok": True, "test_output": output[-2000:],
            "import_as": f"from lib import {req.name}"}


async def smoke_test(project: dict, req: SaveModule) -> tuple[bool, str]:
    """Import the module (with the rest of the library beside it) and run the test code, in
    the sandbox, on in-sample data when an objective is given."""
    from . import datasource
    from .objectives import _run, build_mirror, get_objective

    files = module_files(project["id"])
    files = {k: v for k, v in files.items() if k != f".ft/lib/{req.name}.py"}
    files.setdefault(".ft/lib/__init__.py", "")
    obj = get_objective(req.objective_id) if req.objective_id else None
    catalog = datasource.catalog(project["data_dir"])
    mirror = build_mirror(obj, project["data_dir"]) if obj and obj.get("split_date") else None
    script = (f"from lib import {req.name}\nprint('[lib] imported {req.name}')\n" + (req.test_code or ""))
    extra = {**files, f".ft/lib/{req.name}.py": req.code}
    rep = await _run(script, project["data_dir"], catalog, mirror, 180, obj,
                     obj.get("split_date") if obj else None, extra_files=extra)
    output = (rep["stdout"] + ("\n" + rep["stderr"] if rep["stderr"] else "")).strip()
    return rep["ok"], output


def add_comment(project_id: str, name: str, verdict: str, text: str, author: str,
                version: int | None = None, candidate_id: str | None = None) -> int:
    """Attach a verdict to a module from server code (the HTTP route is for the console)."""
    conn = _db()
    with _lock():
        cur = conn.execute(
            "INSERT INTO lib_comments (project_id, name, version, ts, author, verdict, text, candidate_id) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (project_id, name, version or 0, time.time(), author, verdict, text, candidate_id))
        conn.commit()
    return int(cur.lastrowid or 0)


class Comment(BaseModel):
    verdict: Literal["works", "broken", "note"] = "note"
    text: str = Field(..., min_length=2, max_length=8000)
    author: str = Field("operator", max_length=200)
    candidate_id: str | None = None
    version: int | None = None


@router.post("/projects/{project_id}/library/{name}/comments")
async def comment(project_id: str, name: str, req: Comment) -> dict:
    m = get_module(project_id, name)
    conn = _db()
    with _lock():
        cur = conn.execute(
            "INSERT INTO lib_comments (project_id, name, version, ts, author, verdict, text, candidate_id) VALUES (?,?,?,?,?,?,?,?)",
            (project_id, name, req.version or m["version"], time.time(), req.author, req.verdict, req.text, req.candidate_id))
        conn.commit()
    return {"id": cur.lastrowid}


class Patch(BaseModel):
    status: Literal["active", "retired", "quarantined"] | None = None
    description: str | None = Field(None, max_length=2000)
    restore_version: int | None = None


@router.patch("/projects/{project_id}/library/{name}")
async def patch(project_id: str, name: str, req: Patch) -> dict:
    """Operator controls: retire a module (candidates can no longer import it), edit its
    description, or restore an older version as current."""
    get_module(project_id, name)
    conn = _db()
    with _lock():
        if req.status:
            # Returning a module to active clears the quarantine reason with it: a stale
            # "do not use" on a module the operator has cleared is worse than none.
            conn.execute("UPDATE lib_modules SET status=?, updated_at=?"
                         + (", warning=''" if req.status == "active" else "")
                         + " WHERE project_id=? AND name=?",
                         (req.status, time.time(), project_id, name))
        if req.description is not None:
            conn.execute("UPDATE lib_modules SET description=? WHERE project_id=? AND name=?",
                         (req.description, project_id, name))
        if req.restore_version:
            old = conn.execute("SELECT code FROM lib_versions WHERE project_id=? AND name=? AND version=?",
                               (project_id, name, req.restore_version)).fetchone()
            if old is None:
                raise HTTPException(status_code=404, detail="no such version")
            top = conn.execute("SELECT max(version) FROM lib_versions WHERE project_id=? AND name=?",
                               (project_id, name)).fetchone()[0]
            conn.execute("INSERT INTO lib_versions (project_id, name, version, code, author, ts, note, test_ok) "
                         "VALUES (?,?,?,?,?,?,?,1)", (project_id, name, top + 1, old[0], "operator", time.time(),
                                                      f"restored from v{req.restore_version}"))
            conn.execute("UPDATE lib_modules SET version=?, updated_at=? WHERE project_id=? AND name=?",
                         (top + 1, time.time(), project_id, name))
        conn.commit()
    return get_module(project_id, name)


def brief(project_id: str, limit: int = 15) -> list[dict]:
    """The library as an agent sees it in its iteration brief: active modules, most useful
    first, with the evidence and the latest comments."""
    mods = list_modules(project_id, include_retired=False)
    mods.sort(key=lambda m: (m["evidence"]["champions"], m["evidence"]["ok"], m["updated_at"]), reverse=True)
    conn = _db()
    out = []
    for m in mods[:limit]:
        with _lock():
            latest = [dict(r) for r in conn.execute(
                "SELECT verdict, author, text FROM lib_comments WHERE project_id=? AND name=? ORDER BY ts DESC LIMIT 3",
                (project_id, m["name"])).fetchall()]
        ev = m["evidence"]
        out.append({"name": m["name"], "kind": m["kind"], "description": m["description"], "version": m["version"],
                    "used_by": ev["uses"], "ok": ev["ok"], "errors": ev["errors"],
                    "lookahead_fails": ev["lookahead_fails"], "champions": ev["champions"],
                    "best_in_sample": ev["best_in_sample"], "comments": latest})
    return out


def usage_note(used: list[dict]) -> str:
    return ", ".join(f"{u['name']} v{u['version']}" for u in used)


# =======================================================================================
# Regime maps: which signal works in which regime
# =======================================================================================
REGIME_HARNESS = r"""
import importlib, json, math, sys
import numpy as np, pandas as pd
import ft

CFG = json.loads(open("/work/.ft/regime_cfg.json", encoding="utf-8").read())
df = ft.load(CFG["dataset"])
tc, pc = CFG["time_column"], CFG["price_column"]
df = df.sort_values(tc).reset_index(drop=True)
regime = importlib.import_module("lib." + CFG["regime"]).detect(df)
regime = pd.Series(np.asarray(regime), index=df.index).astype(str)
# The forward one-bar return earned by a position held from the close of bar t: the harness
# measures with it; strategies never see it.
fwd = pd.to_numeric(df[pc], errors="coerce").pct_change().shift(-1).fillna(0.0)
day = pd.to_datetime(df[tc]).dt.normalize()
bars_per_year = 252.0 * max(1.0, float(df.groupby(day).size().median()))

COST = float(CFG.get("cost_bps") or 0.0) / 1e4
days_total = max(1, int(day.nunique()))

def sharpe(x):
    sd = float(x.std())
    return float(x.mean() / sd * math.sqrt(bars_per_year)) if sd > 0 else None

def stats(pos, mask):
    # Net of the objective's trading cost: |position change| x cost_bps, charged on the bar
    # the trade happens -- the same rule the scoring harness uses.
    gross = (pos * fwd)[mask]
    trades = pos.diff().abs().fillna(pos.abs())
    net = gross - (trades * COST)[mask]
    n = int(mask.sum())
    if n < 30:
        return {"bars": n}
    active = gross[pos[mask] != 0]
    return {"bars": n, "sharpe": sharpe(net), "sharpe_gross": sharpe(gross),
            "mean_bps": float(net.mean() * 1e4),
            "trades_per_day": float(trades[mask].sum() / max(1, day[mask].nunique())),
            "hit_rate": float((active > 0).mean()) if len(active) else None,
            "active": float(len(active) / n)}

signals = {"buy_and_hold": pd.Series(1.0, index=df.index)}
errors = {}
for name in CFG["signals"]:
    try:
        s = importlib.import_module("lib." + name).signal(df)
        signals[name] = pd.Series(np.asarray(s, dtype=float), index=df.index).fillna(0.0).clip(-1, 1)
    except Exception as exc:
        errors[name] = f"{type(exc).__name__}: {exc}"[:300]

table = {}
share = regime.value_counts(normalize=True)
for label in share.index:
    mask = regime == label
    table[label] = {"share": float(share[label]),
                    "signals": {k: stats(v, mask) for k, v in signals.items()}}
ft._merge({"regime_map": {"regimes": table, "errors": errors, "bars": int(len(df)),
                          "bars_per_year": bars_per_year, "from": str(df[tc].iloc[0]), "to": str(df[tc].iloc[-1])}})
print("regimes:", {k: round(v["share"], 3) for k, v in table.items()})
"""


def latest_regime_map(project_id: str, regime: str) -> dict | None:
    import json

    conn = _db()
    with _lock():
        r = conn.execute("SELECT * FROM regime_maps WHERE project_id=? AND regime=? ORDER BY ts DESC LIMIT 1",
                         (project_id, regime)).fetchone()
    if r is None:
        return None
    d = dict(r)
    d["result"] = json.loads(d["result"])
    return d


class RegimeMapReq(BaseModel):
    regime: str = Field(..., max_length=48)
    signals: list[str] | None = None
    author: str = Field("", max_length=200)


@router.post("/objectives/{oid}/regime-map")
async def regime_map(oid: str, req: RegimeMapReq) -> dict:
    """Measure every signal module inside every regime of a detector, on IN-SAMPLE data."""
    import json

    from . import datasource
    from .objectives import _run, build_mirror, get_objective

    obj = get_objective(oid)
    project = _project(obj["project_id"])
    pc = obj["metric"].get("price_column")
    if not (obj.get("dataset") and obj.get("time_column") and pc):
        raise HTTPException(status_code=400, detail="regime maps need an objective with a dataset, time column and price column")
    mods = {m["name"]: m for m in list_modules(project["id"], include_retired=False)}
    if mods.get(req.regime, {}).get("kind") != "regime":
        raise HTTPException(status_code=400, detail=f"{req.regime!r} is not an active regime module")
    signals = req.signals or [n for n, m in mods.items() if m["kind"] == "signal"]
    signals = [s for s in signals if mods.get(s, {}).get("kind") == "signal"][:20]
    cfg = {"dataset": obj["dataset"], "time_column": obj["time_column"], "price_column": pc,
           "regime": req.regime, "signals": signals, "cost_bps": obj["metric"].get("cost_bps") or 0}
    mirror = build_mirror(obj, project["data_dir"]) if obj.get("split_date") else None
    rep = await _run(REGIME_HARNESS, project["data_dir"], datasource.catalog(project["data_dir"]), mirror, 300,
                     obj, obj.get("split_date"), extra_files={".ft/regime_cfg.json": json.dumps(cfg)})
    result = (rep.get("result") or {}).get("regime_map")
    if not rep["ok"] or not result:
        return {"ok": False, "error": "regime map failed", "stderr": rep["stderr"][-3000:], "stdout": rep["stdout"][-1000:]}
    result["signals_tested"] = signals
    result["in_sample_only"] = bool(obj.get("split_date"))
    conn = _db()
    with _lock():
        conn.execute("INSERT INTO regime_maps (project_id, objective_id, regime, regime_version, ts, author, result) "
                     "VALUES (?,?,?,?,?,?,?)", (project["id"], oid, req.regime, mods[req.regime]["version"],
                                                time.time(), req.author, json.dumps(result)))
        conn.commit()
    return {"ok": True, "regime": req.regime, **compact_map(result)}


def compact_map(result: dict) -> dict:
    """The regime table trimmed for a model's context: per regime, its share and each
    signal's Sharpe / mean / hit rate, best first."""
    out = {}
    for label, row in (result.get("regimes") or {}).items():
        sigs = sorted(((k, v) for k, v in row["signals"].items() if v.get("sharpe") is not None),
                      key=lambda kv: kv[1]["sharpe"], reverse=True)
        out[label] = {"share": round(row["share"], 3),
                      "by_signal": {k: {"sharpe": round(v["sharpe"], 2),
                                        "gross": round(v["sharpe_gross"], 2) if v.get("sharpe_gross") is not None else None,
                                        "trades_per_day": round(v.get("trades_per_day") or 0, 1),
                                        "mean_bps": round(v["mean_bps"], 3),
                                        "hit": round(v["hit_rate"], 3) if v.get("hit_rate") is not None else None}
                                    for k, v in sigs}}
    return {"regimes": out, "errors": result.get("errors") or {}, "period": f"{result.get('from')} .. {result.get('to')}",
            "note": "in-sample only; Sharpe annualised per bar, NET of the objective's cost (gross shown too)"}


def regime_brief(project_id: str) -> list[dict]:
    """Latest regime map per regime module, compacted, for the iteration brief."""
    out = []
    for m in list_modules(project_id, include_retired=False):
        if m["kind"] != "regime":
            continue
        rm = latest_regime_map(project_id, m["name"])
        if rm:
            out.append({"regime_module": m["name"], "version": rm["regime_version"], **compact_map(rm["result"])})
    return out


# =======================================================================================
# Field scan: which of the dataset's fields carry information about future returns
# =======================================================================================
FIELD_SCAN_HARNESS = r"""
import importlib, json, math
import numpy as np, pandas as pd
import ft

CFG = json.loads(open("/work/.ft/scan_cfg.json", encoding="utf-8").read())
tc, pc, h, step = CFG["time_column"], CFG["price_column"], int(CFG["horizon"]), int(CFG["every"])
df = ft.load(CFG["dataset"]).sort_values(tc).reset_index(drop=True)
price = pd.to_numeric(df[pc], errors="coerce")
fwd = price.shift(-h) / price - 1.0          # measured by the harness, never a feature
num = [c for c in df.columns if c not in (tc, pc) and pd.api.types.is_numeric_dtype(df[c])]
if CFG.get("columns"):
    num = [c for c in num if c in CFG["columns"]]
regime = None
if CFG.get("regime"):
    regime = pd.Series(np.asarray(importlib.import_module("lib." + CFG["regime"]).detect(df)), index=df.index).astype(str)
idx = np.arange(0, len(df) - h, step)           # sparse sample: overlapping horizons are near-duplicates
y = fwd.iloc[idx]

def ic(x, mask=None):
    xs, ys = x.iloc[idx], y
    if mask is not None:
        m = mask.iloc[idx].to_numpy()
        xs, ys = xs[m], ys[m]
    ok = xs.notna() & ys.notna() & np.isfinite(xs) & np.isfinite(ys)
    n = int(ok.sum())
    if n < 200 or xs[ok].nunique() < 3:
        return None, n
    r = xs[ok].rank().corr(ys[ok].rank())
    return (None if r is None or not math.isfinite(r) else float(r)), n

out = []
for c in num:
    x = pd.to_numeric(df[c], errors="coerce")
    lvl, n = ic(x)
    chg, _ = ic(x - x.shift(h))
    row = {"field": c, "ic_level": lvl, "ic_change": chg, "n": n,
           "t_level": (lvl * math.sqrt(max(n - 2, 1)) / math.sqrt(max(1e-12, 1 - lvl * lvl))) if lvl is not None else None,
           "coverage": float(x.notna().mean())}
    if regime is not None:
        row["by_regime"] = {}
        for lab in regime.value_counts().index[:6]:
            v, m = ic(x, regime == lab)
            row["by_regime"][lab] = {"ic_level": v, "n": m}
    out.append(row)
key = lambda r: max(abs(r["ic_level"] or 0), abs(r["ic_change"] or 0))
out.sort(key=key, reverse=True)
ft._merge({"field_scan": {"fields": out, "horizon": h, "every": step, "rows": int(len(idx)),
                          "from": str(df[tc].iloc[0]), "to": str(df[tc].iloc[-1]), "regime": CFG.get("regime")}})
print("scanned", len(out), "fields; top:", [(r["field"], round(key(r), 4)) for r in out[:5]])
"""


class FieldScanReq(BaseModel):
    horizon: int = Field(30, ge=1, le=5000, description="bars ahead for the forward return")
    every: int = Field(6, ge=1, le=10_000, description="sample every N bars")
    regime: str | None = Field(None, max_length=48)
    columns: list[str] | None = None
    author: str = Field("", max_length=200)


@router.post("/objectives/{oid}/field-scan")
async def field_scan(oid: str, req: FieldScanReq) -> dict:
    """Rank every numeric field by its in-sample rank correlation (IC) with the forward
    return -- as a level and as its change over the horizon, optionally within each regime."""
    import json

    from . import datasource
    from .objectives import _run, build_mirror, get_objective

    obj = get_objective(oid)
    project = _project(obj["project_id"])
    pc = obj["metric"].get("price_column")
    if not (obj.get("dataset") and obj.get("time_column") and pc):
        raise HTTPException(status_code=400, detail="a field scan needs an objective with a dataset, time column and price column")
    cfg = {"dataset": obj["dataset"], "time_column": obj["time_column"], "price_column": pc,
           "horizon": req.horizon, "every": req.every, "regime": req.regime, "columns": req.columns}
    mirror = build_mirror(obj, project["data_dir"]) if obj.get("split_date") else None
    rep = await _run(FIELD_SCAN_HARNESS, project["data_dir"], datasource.catalog(project["data_dir"]), mirror, 400,
                     obj, obj.get("split_date"), extra_files={".ft/scan_cfg.json": json.dumps(cfg)})
    result = (rep.get("result") or {}).get("field_scan")
    if not rep["ok"] or not result:
        return {"ok": False, "error": "field scan failed", "stderr": rep["stderr"][-3000:]}
    conn = _db()
    with _lock():
        conn.execute("INSERT INTO field_scans (project_id, objective_id, ts, author, params, result) VALUES (?,?,?,?,?,?)",
                     (project["id"], oid, time.time(), req.author, json.dumps(cfg), json.dumps(result)))
        conn.commit()
    return {"ok": True, **compact_scan(result, 25)}


def compact_scan(result: dict, top: int = 15) -> dict:
    def r4(v):
        return None if v is None else round(v, 4)

    rows = []
    for f in result.get("fields", [])[:top]:
        row = {"field": f["field"], "ic_level": r4(f.get("ic_level")), "ic_change": r4(f.get("ic_change"))}
        if f.get("by_regime"):
            row["ic_by_regime"] = {k: r4(v.get("ic_level")) for k, v in f["by_regime"].items()}
        rows.append(row)
    return {"horizon_bars": result.get("horizon"), "fields_scanned": len(result.get("fields", [])),
            "period": f"{result.get('from')} .. {result.get('to')}", "regime": result.get("regime"), "top": rows,
            "note": ("in-sample rank correlation with the forward return over the horizon; |IC| 0.02-0.05 is "
                     "typical for real signals at this frequency. Positive = higher value, higher future return.")}


def latest_field_scan(project_id: str) -> dict | None:
    import json

    conn = _db()
    with _lock():
        r = conn.execute("SELECT * FROM field_scans WHERE project_id=? ORDER BY ts DESC LIMIT 1", (project_id,)).fetchone()
    if r is None:
        return None
    return {"ts": r["ts"], "author": r["author"], "params": json.loads(r["params"]), "result": json.loads(r["result"])}


@router.get("/projects/{project_id}/field-scan")
async def get_field_scan(project_id: str) -> dict:
    _project(project_id)
    return {"scan": latest_field_scan(project_id)}
