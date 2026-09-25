"""Standing objectives: a goal the swarm works on continuously, and the yardstick for "better".

A task is answered once. An objective -- "create the best trading strategy possible" -- is
never finished: agents keep producing CANDIDATE solutions, each is scored mechanically, and
the best one so far is the objective's champion until something beats it. The operator's
messages steer the search (notes) instead of queuing unrelated one-off tasks.

**The score is computed here, never by the agent.** A candidate is a Python script that runs
in the sandbox (``--network none``, read-only data) and reports raw output. For a trading
objective that is its POSITIONS per bar; this module computes the returns itself -- from the
dataset's own prices, holding each position to the next bar, charging costs on every change --
and then the metric (Sharpe, Sortino, ...), with code the candidate cannot see or change. A
model asked "what is your Sharpe?" will happily say 3.1; a model whose trades are marked to
market by someone else cannot. (Self-reported returns are accepted when no price column is
configured, but they are weaker on both counts -- see the look-ahead note below.)

Optimising a number relentlessly is also the fastest way to fool yourself, so three guards
sit between "scored well" and "champion":

* **Hidden holdout.** Data is split at a date. Ranking uses returns AFTER the split; agents
  are shown only their in-sample numbers, and their exploratory data access (run_python,
  query_data in objective mode) sees only data BEFORE the split.
* **Look-ahead test.** Every candidate runs again with every row after a cut removed (a
  truncated copy laid over the data mount). A causal strategy decides the same positions before
  a cut whether or not the rows after it exist. If removing the future changes a past DECISION,
  the candidate is rejected -- mechanically, not by opinion. Cuts: the split, the mid
  in-sample bar, and ``LOOKAHEAD_ACTIVE_CUTS`` more placed just AFTER the candidate's own
  trades, spread over the in-sample period, at offsets that sit on no bar grid (7 s .. 1.5 h).
  A fixed cut only catches a leak if the strategy happens to be trading there (#889 peeked 15
  minutes ahead and passed two fixed cuts); a cut seconds-to-minutes after a trade removes
  exactly the future that trade would have peeked at. Only a clean pass can be crowned. This has to compare positions, not
  returns: a strategy that trades on the next bar's move and books that same bar's return is
  perfectly self-consistent in its return stream (measured: a one-bar peek scoring Sharpe 40
  passed a returns-only comparison), but its position at the last bar before the cut changes.
* **Audit.** A would-be champion's code is reviewed by a model (the runner asks one) against a
  checklist -- costs, fabricated returns, survivorship -- before it takes the title.

**Forecast features.** Candidates run with no network, so a strategy cannot call a
time-series model itself. Instead an agent asks for a FEATURE (``forecast_feature``): the
control plane runs the forecaster over a dataset column causally -- the forecast stored at
bar t was made from bars up to and including t only -- and writes it as a dataset candidates
load with ``ft.load("fc_<name>")``. Features are truncated at each cut like any other data,
so the look-ahead test and the hidden holdout cover them too.

Storage: SQLite beside the control plane (objectives.sqlite3).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
import re
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import duckdb
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import datasource, projects
from .config import settings
from .sandbox import execute

logger = logging.getLogger("freetoken.objectives")

DB_PATH = Path(__file__).resolve().parent.parent / "objectives.sqlite3"
WORK_ROOT = Path(settings.repo_root) / "ui" / "sandbox" / ".objectives"
FT_HELPER = Path(settings.repo_root) / "ui" / "sandbox" / "ft.py"

# Two candidates evaluate at once (each is two container runs, 2 CPUs / 4 GiB apiece):
# enough to keep two agents busy without the evaluations starving the machine.
_EVAL_SLOTS = asyncio.Semaphore(2)
# The dense look-ahead test (see the module doc): cuts per candidate placed after its own
# trades, how many truncated runs go at once, and how far after a trade each cut lands --
# seconds that are a multiple of no bar size, from a few seconds to 1.5 hours.
LOOKAHEAD_ACTIVE_CUTS = 8
LOOKAHEAD_CONCURRENCY = 2
# Look-ahead file work (truncated copies, comparisons) runs on its own two threads. On the
# shared default pool it filled every worker, and every other request that needs a thread
# (the console's pages) queued behind it -- the control plane looked hung.
_LOOKAHEAD_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="lookahead")
# A leaderboard re-test is background work: one candidate at a time, never in the slots that
# live submissions use.
_RETEST_SLOT = asyncio.Semaphore(1)
RETEST_CONCURRENCY = 5  # truncated runs at once for the candidate being re-tested


async def _off(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(_LOOKAHEAD_POOL, lambda: fn(*args))
_CUT_OFFSETS_S = (7, 43, 173, 437, 881, 1333, 2711, 5413)
DEFAULT_EVAL_TIMEOUT_S = 300
# Lessons past this count get consolidated by an agent into a shorter list.
LESSONS_CONSOLIDATE_AT = 40
EXPLORE_PROBABILITY = 0.3
# A single dataset larger than this is not truncated for the look-ahead test (reported).
MAX_TRUNCATE_BYTES = 20 << 30
TIME_NAMES = ("timestamp", "datetime", "date", "time", "ts", "trade_date", "bar_time", "dt")

MetricKind = Literal["sharpe", "sortino", "calmar", "total_return", "cagr", "max_drawdown",
                     "reported", "judge"]
RETURN_METRICS = {"sharpe", "sortino", "calmar", "total_return", "cagr", "max_drawdown"}
METRIC_LABEL = {
    "sharpe": "Sharpe ratio", "sortino": "Sortino ratio", "calmar": "Calmar ratio",
    "total_return": "total return", "cagr": "CAGR", "max_drawdown": "max drawdown",
    "reported": "reported score", "judge": "judge score (0-10)",
}

router = APIRouter(tags=["objectives"])

# =======================================================================================
# Storage
# =======================================================================================
_lock = threading.RLock()
_conn: sqlite3.Connection | None = None


def db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS objectives (
                id TEXT PRIMARY KEY, project_id TEXT NOT NULL, title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '', metric TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running', dataset TEXT, time_column TEXT,
                split_date TEXT, lookahead_check INTEGER NOT NULL DEFAULT 1,
                require_audit INTEGER NOT NULL DEFAULT 1, eval_timeout_s INTEGER NOT NULL DEFAULT 300,
                cooldown_s INTEGER NOT NULL DEFAULT 5, best_id TEXT,
                consolidating_until REAL NOT NULL DEFAULT 0,
                created_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS candidates (
                id TEXT PRIMARY KEY, objective_id TEXT NOT NULL, seq INTEGER NOT NULL,
                created_at REAL NOT NULL, model TEXT, mode TEXT, parent_id TEXT,
                rationale TEXT NOT NULL DEFAULT '', code TEXT NOT NULL DEFAULT '',
                answer TEXT NOT NULL DEFAULT '', status TEXT NOT NULL,
                score REAL, is_score REAL, score_note TEXT NOT NULL DEFAULT '',
                metrics TEXT NOT NULL DEFAULT '{}', returns TEXT NOT NULL DEFAULT '[]',
                lookahead TEXT NOT NULL DEFAULT 'skipped', lookahead_detail TEXT NOT NULL DEFAULT '',
                audit TEXT NOT NULL DEFAULT 'none', audit_notes TEXT NOT NULL DEFAULT '',
                audit_started REAL NOT NULL DEFAULT 0,
                stdout TEXT NOT NULL DEFAULT '', stderr TEXT NOT NULL DEFAULT '',
                eval_seconds REAL, run_id TEXT, champion_at REAL
            );
            CREATE INDEX IF NOT EXISTS cand_obj ON candidates(objective_id, seq);
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT, objective_id TEXT NOT NULL,
                ts REAL NOT NULL, author TEXT NOT NULL, text TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT, objective_id TEXT NOT NULL,
                ts REAL NOT NULL, model TEXT, candidate_id TEXT, text TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1
            );
            -- The highest candidate number ever issued per objective. Candidates can be deleted
            -- (operator clean-up); numbering from MAX(seq) alone would then re-issue a deleted
            -- number, and lessons and board posts that cite "#N" would point at a new candidate.
            CREATE TABLE IF NOT EXISTS seq_hwm (
                objective_id TEXT PRIMARY KEY, seq INTEGER NOT NULL
            );
            """
        )
        # Which mentor idea a candidate tested (escalation.ideas.id) -- the ideas scoreboard.
        if "idea_id" not in {r[1] for r in _conn.execute("PRAGMA table_info(candidates)").fetchall()}:
            _conn.execute("ALTER TABLE candidates ADD COLUMN idea_id INTEGER")
        # Anything still "evaluating" belongs to a previous control-plane process that died
        # mid-run; it will never finish, so say so instead of showing it as in progress.
        _conn.execute("UPDATE candidates SET status='error', score_note='evaluation interrupted "
                      "(control plane restarted)' WHERE status='evaluating'")
        # Seed the high-water mark for candidates numbered before it existed, so deleting the
        # newest ones right away cannot hand their numbers out again.
        _conn.execute("INSERT OR IGNORE INTO seq_hwm (objective_id, seq) "
                      "SELECT objective_id, MAX(seq) FROM candidates GROUP BY objective_id")
        _conn.commit()
    return _conn


def _obj_row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["metric"] = json.loads(d["metric"])
    d["lookahead_check"] = bool(d["lookahead_check"])
    d["require_audit"] = bool(d["require_audit"])
    return d


_LIGHT = ("id, objective_id, seq, created_at, model, mode, parent_id, rationale, status, score, "
          "is_score, score_note, metrics, lookahead, lookahead_detail, audit, audit_notes, "
          "eval_seconds, champion_at, idea_id")


def _cand_row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["metrics"] = json.loads(d.get("metrics") or "{}")
    if "returns" in d:
        d["returns"] = json.loads(d["returns"] or "[]")
    return d


def get_objective(oid: str) -> dict:
    with _lock:
        r = db().execute("SELECT * FROM objectives WHERE id=?", (oid,)).fetchone()
    if r is None:
        raise HTTPException(status_code=404, detail=f"no objective {oid!r}")
    return _obj_row(r)


def _better(a: float | None, b: float | None, higher: bool) -> bool:
    """Is score a strictly better than score b?"""
    if a is None:
        return False
    if b is None:
        return True
    return a > b if higher else a < b


def _higher(obj: dict) -> bool:
    return bool(obj["metric"].get("higher_is_better", True))


def _cascade_quarantine(obj: dict, modules: list[str], reason: str, reviewer: str,
                        origin_cid: str | None = None) -> list[int]:
    """Disqualify every candidate built on a module that was just quarantined.

    A leak lives in the module, so every result that imported it inherited the leak and its
    score is not real either. Removing only the one candidate that was reviewed leaves its
    siblings -- often including the next-best, which is about to be crowned -- standing on
    the same defect, and the swarm spends the night refining an infected family. This takes
    the whole family off the board in one pass, then re-crowns from what is actually left.

    Returns the seqs disqualified. Candidates already failed are left alone.
    """
    if not modules:
        return []
    from . import library

    tainted = set(modules)
    hit: list[int] = []
    with _lock:
        rows = db().execute(
            "SELECT id, seq, code, audit FROM candidates WHERE objective_id=? AND status='ok'",
            (obj["id"],)).fetchall()
    for r in rows:
        if r["audit"] == "fail" or r["id"] == origin_cid:
            continue
        used = library.reachable_modules(obj["project_id"], r["code"] or "")
        if not (tainted & set(used)):
            continue
        note = (f"[auto] Disqualified with the family: this result imports "
                f"{', '.join(sorted(tainted & set(used)))}, which was quarantined. {reason}")[:20_000]
        _update_candidate(r["id"], {"audit": "fail", "audit_notes": note})
        hit.append(r["seq"])
    if not hit:
        return []
    logger.info("quarantine cascade disqualified %s", hit)

    # The champion may have just been disqualified along with the rest of the family.
    best = _best(get_objective(obj["id"]))
    if best is None or best.get("audit") == "fail":
        now = time.time()
        with _lock:
            db().execute("UPDATE objectives SET best_id=NULL, updated_at=? WHERE id=?", (now, obj["id"]))
            db().commit()
        nxt = _ranked(obj["id"], _higher(obj), limit=1)
        if nxt:
            _crown(obj["id"], nxt[0]["id"])

    _board_post(
        obj["project_id"], "results", reviewer,
        f"DISQUALIFIED WITH THE FAMILY: #{', #'.join(str(s) for s in sorted(hit))}.\n\n"
        f"These results import {', '.join(sorted(tainted))}, which was quarantined, so they "
        f"inherit the same defect and their scores are not real. {reason}\n\n"
        f"Do not tune, scale, filter or re-submit any of them. The leak is in the module, not "
        f"in the wrapper -- start a NEW module that fixes the cause.",
        {"objective_id": obj["id"], "cascade": True, "seqs": sorted(hit), "modules": sorted(tainted)})
    return sorted(hit)


def retire_modules(project_id: str, modules: list[str], reason: str, reviewer: str) -> dict:
    """Quarantine modules and disqualify every result in the project that uses them.

    The entry point for retiring a signal on its own merits -- from a review of the module
    itself rather than of one candidate that happened to import it. A module is shared
    across a project's objectives, so the sweep runs over all of them.
    """
    from . import library

    hit = library.quarantine(project_id, modules, reason, reviewer)
    if not hit:
        return {"quarantined": [], "disqualified": {}}
    with _lock:
        rows = db().execute("SELECT * FROM objectives WHERE project_id=?", (project_id,)).fetchall()
    out: dict[str, list[int]] = {}
    for row in rows:
        obj = _obj_row(row)
        seqs = _cascade_quarantine(obj, hit, reason, reviewer)
        if seqs:
            out[obj["id"]] = seqs
    return {"quarantined": hit, "disqualified": out}


def _auto_quarantine(obj: dict, cid: str, seq: int | None, code: str, why: str) -> list[str]:
    """Retire the library modules a disqualified result was built on, if the operator allows it.

    Controlled by the `auto_quarantine` preference (on by default). Never raises: a failure
    to retire a module must not turn a scored candidate into an error.
    """
    try:
        from . import library, prefs

        if not prefs.get_auto_quarantine():
            return []
        names = sorted(library.reachable_modules(obj["project_id"], code or ""))
        hit = library.quarantine(obj["project_id"], names, why, "harness", candidate_id=cid, seq=seq)
        if hit:
            logger.info("auto-quarantined %s after #%s", ", ".join(hit), seq)
            _cascade_quarantine(obj, hit, why, "harness", origin_cid=cid)
            _board_post(
                obj["project_id"], "errors", "harness",
                f"QUARANTINED {', '.join(hit)} -- do not import these.\n\n{why}\n\n"
                f"A result built on them was disqualified. Fix the cause in a NEW module; saving "
                f"another version of one of these repeats the same defect.",
                {"objective_id": obj["id"], "candidate_id": cid, "seq": seq, "quarantined": hit})
        return hit
    except Exception:  # noqa: BLE001 -- scoring must survive a library problem
        logger.exception("could not auto-quarantine after #%s", seq)
        return []


def _disqualified(oid: str, higher: bool, limit: int = 50) -> list[dict]:
    """Candidates that scored but were disqualified -- demoted, or caught looking ahead.

    `_ranked` drops them so they can never be crowned, which also made them vanish from the
    console: a result the operator had just demoted still looked like the leader. They are
    listed separately instead, so a disqualified score is visibly disqualified rather than
    absent, and nobody builds on it believing it stands.
    """
    order = "DESC" if higher else "ASC"
    with _lock:
        rows = db().execute(
            f"SELECT {_LIGHT} FROM candidates WHERE objective_id=? AND status='ok' AND score IS NOT NULL "
            f"AND (audit='fail' OR lookahead='fail') ORDER BY score {order}, seq ASC LIMIT ?",
            (oid, limit),
        ).fetchall()
    return [_cand_row(r) for r in rows]


def _ranked(oid: str, higher: bool, limit: int = 1000) -> list[dict]:
    """Eligible candidates, best first: scored, not caught looking ahead, not failed audit."""
    order = "DESC" if higher else "ASC"
    with _lock:
        rows = db().execute(
            f"SELECT {_LIGHT} FROM candidates WHERE objective_id=? AND status='ok' AND score IS NOT NULL "
            f"AND lookahead NOT IN ('fail', 'error') AND audit NOT IN ('fail') ORDER BY score {order}, seq ASC LIMIT ?",
            (oid, limit),
        ).fetchall()
    return [_cand_row(r) for r in rows]


# =======================================================================================
# Metrics (trusted: computed here from what the candidate reported)
# =======================================================================================
def _stats(rets: list[float], ppy: float) -> dict:
    n = len(rets)
    out: dict[str, Any] = {"days": n, "active_days": sum(1 for r in rets if r != 0.0)}
    if n < 2:
        return out
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    sd = math.sqrt(var)
    down = math.sqrt(sum(min(r, 0.0) ** 2 for r in rets) / n)
    eq = peak = 1.0
    mdd = 0.0
    for r in rets:
        eq *= 1.0 + r
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1.0)
    years = n / ppy
    cagr = (eq ** (1.0 / years) - 1.0) if eq > 0 and years > 0 else None
    wins = [r for r in rets if r > 0]
    nonzero = [r for r in rets if r != 0]
    out.update({
        "sharpe": (mean / sd * math.sqrt(ppy)) if sd > 1e-12 else None,
        "sortino": (mean / down * math.sqrt(ppy)) if down > 1e-12 else None,
        "total_return": eq - 1.0,
        "cagr": cagr,
        "max_drawdown": mdd,
        "calmar": (cagr / abs(mdd)) if (cagr is not None and mdd < -1e-9) else None,
        "volatility": sd * math.sqrt(ppy),
        "win_rate": (len(wins) / len(nonzero)) if nonzero else None,
    })
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}


def _score_returns(obj: dict, returns: list[list]) -> tuple[float | None, float | None, str, dict]:
    """(holdout score, in-sample score, note, metrics) for a trading-style objective."""
    m = obj["metric"]
    kind = m["kind"]
    ppy = float(m.get("periods_per_year") or 252)
    min_active = int(m.get("min_active_days") or 20)
    split = obj.get("split_date")
    ins = [r for d, r in returns if not split or d < split]
    oos = [r for d, r in returns if split and d >= split]
    metrics = {"in_sample": _stats(ins, ppy)}
    if split:
        metrics["holdout"] = _stats(oos, ppy)
    metrics["full"] = _stats([r for _, r in returns], ppy)
    big = sum(1 for _, r in returns if abs(r) > 0.5)
    if big:
        metrics["warning"] = f"{big} day(s) with a return beyond +/-50% -- check position sizing and units"

    def pick(seg: dict) -> tuple[float | None, str]:
        if seg.get("active_days", 0) < min_active:
            return None, f"only {seg.get('active_days', 0)} active days (need {min_active})"
        v = seg.get(kind)
        return (v, "") if v is not None else (None, f"{kind} undefined (no variance or no drawdown)")

    is_score, _ = pick(metrics["in_sample"])
    if split:
        score, note = pick(metrics["holdout"])
        if score is None and note:
            note = f"holdout: {note}"
    else:
        score, note = pick(metrics["full"])
    return score, is_score, note, metrics


def _costs(obj: dict, returns: list[list], gross: list[list], inverted: list[list], changes: int) -> dict:
    """The candidate's metric before costs and with every position flipped, per segment.

    The swarm only ever saw the score after costs, so a signal with a real edge that traded
    every bar looked exactly like one with no edge at all (#532: Sharpe -3.97 after costs,
    -0.88 before), and a signal pointing the wrong way looked like noise."""
    m = obj["metric"]
    kind = m["kind"]
    ppy = float(m.get("periods_per_year") or 252)
    split = obj.get("split_date")

    def seg(series: list[list], part: str) -> list[float]:
        if part == "in_sample":
            return [r for d, r in series if not split or d < split]
        return [r for d, r in series if split and d >= split]

    out: dict[str, Any] = {"metric": kind, "cost_bps": m.get("cost_bps"),
                           "changes_per_day": round(changes / max(1, len(returns)), 1)}
    for part in ("in_sample", "holdout") if split else ("in_sample",):
        net, g, inv = _stats(seg(returns, part), ppy), _stats(seg(gross, part), ppy), _stats(seg(inverted, part), ppy)
        out[part] = {"net": net.get(kind), "gross": g.get(kind), "inverted": inv.get(kind),
                     "return_net": net.get("total_return"), "return_gross": g.get("total_return"),
                     "return_inverted": inv.get("total_return")}
    out["verdict"] = _cost_verdict(out)
    return out


def _cost_verdict(costs: dict) -> str | None:
    """One plain line from the IN-SAMPLE numbers only (agents never see the holdout).

    Flipping a strategy flips its gross result (near enough) but not its costs, so there are
    four cases: an edge that costs give away (trade less), a signal pointing the wrong way
    that survives costs flipped (flip it), one that points the wrong way but whose flipped edge
    costs still erase (flip AND trade less), and no clear edge either way (change the idea)."""
    s = costs.get("in_sample") or {}
    net, gross, inv = s.get("net"), s.get("gross"), s.get("inverted")
    if net is None or gross is None:
        return None
    kind = costs["metric"]
    label = METRIC_LABEL.get(kind, kind)
    noise = 0.5 if kind in ("sharpe", "sortino", "calmar") else 0.0
    rn, rg = s.get("return_net"), s.get("return_gross")
    paid = f"costs took {abs(rg - rn):.2%} of return" if rn is not None and rg is not None else "costs"
    cpd = costs.get("changes_per_day") or 0
    head = f"In-sample {label}: {net:.2f} after costs, {gross:.2f} before ({paid}; {cpd:g} position changes/day)."
    less = ("Trade LESS: hold positions longer, act only on strong signals, and do not rescale the size every "
            "bar (every size change is a trade that pays costs).")
    if abs(gross) <= noise:
        return (head + " No clear edge before costs in either direction: the idea itself does not work here, "
                "not just its costs -- change the signal, not its thresholds.")
    if gross > 0:
        return head + (" The signal has an edge before costs; trading it this often gives it away. " + less
                       if net < 0 or gross - net > gross / 3 else "")
    if inv is not None and inv > 0:
        return (head + f" FLIPPED -- every position's sign reversed, same costs -- it scores {inv:.2f}: the "
                "signal points the wrong way. Try the opposite direction (and check it is not in-sample luck).")
    return (head + f" It points the wrong way: flipped it would make about {-gross:.2f} before costs, but costs "
            f"still sink the flipped version ({inv:.2f})" if inv is not None else head) + ". Flip it AND " + less[0].lower() + less[1:]


# =======================================================================================
# Data: time columns, the split date, and the truncated mirror for the look-ahead test
# =======================================================================================
def _reader(item: dict) -> str:
    return {"parquet": "read_parquet", "csv": "read_csv_auto", "tsv": "read_csv_auto"}.get(
        item["format"], "read_json_auto")


def _abs(data_dir: str, item: dict) -> str:
    return (Path(data_dir) / item["path"]).as_posix().replace("'", "''")


def detect_time_column(data_dir: str, item: dict, preferred: str | None = None) -> str | None:
    con = duckdb.connect(":memory:")
    try:
        cols = con.execute(f"DESCRIBE SELECT * FROM {_reader(item)}('{_abs(data_dir, item)}')").fetchall()
    except duckdb.Error:
        return None
    finally:
        con.close()
    names = {c[0]: str(c[1]).upper() for c in cols}
    if preferred and preferred in names:
        return preferred
    for name, typ in names.items():
        if typ.startswith("TIMESTAMP") or typ == "DATE":
            return name
    for want in TIME_NAMES:
        for name in names:
            if name.lower() == want:
                return name
    return None


def probe_dataset(data_dir: str, view: str, holdout_fraction: float, time_column: str | None) -> dict:
    """Time column, date range and the split date that leaves `holdout_fraction` of the days
    after it -- what the New objective form proposes."""
    item = next((i for i in datasource.catalog(data_dir) if view in (i["view"], i["path"])), None)
    if item is None:
        raise HTTPException(status_code=400, detail=f"no dataset {view!r} in the project data folder")
    con = duckdb.connect(":memory:")
    try:
        cols = con.execute(f"DESCRIBE SELECT * FROM {_reader(item)}('{_abs(data_dir, item)}')").fetchall()
    finally:
        con.close()
    columns = [{"name": c[0], "type": str(c[1])} for c in cols]
    numeric = [c["name"] for c in columns if any(t in c["type"].upper() for t in
               ("DOUBLE", "FLOAT", "DECIMAL", "REAL", "INT", "NUMERIC"))]
    price = next((n for want in ("close", "price", "last", "spot", "mid", "adj_close", "px", "underlying_price")
                  for n in numeric if n.lower() == want), None)
    if price is None:
        price = next((n for n in numeric if any(w in n.lower() for w in ("close", "price", "spot"))), None)
    tc = detect_time_column(data_dir, item, time_column)
    base = {"dataset": item["view"], "columns": columns, "numeric_columns": numeric, "price_column": price}
    if tc is None:
        return {**base, "time_column": None,
                "note": "no date/time column found -- the holdout and look-ahead test need one"}
    con = duckdb.connect(":memory:")
    try:
        con.execute("SET TimeZone = 'UTC'")
        q = (f'SELECT min(d), max(d), count(*), quantile_disc(d, {1.0 - holdout_fraction}) FROM '
             f'(SELECT DISTINCT CAST(TRY_CAST("{tc}" AS TIMESTAMP) AS DATE) AS d FROM '
             f"{_reader(item)}('{_abs(data_dir, item)}')) WHERE d IS NOT NULL")
        lo, hi, days, split = con.execute(q).fetchone()
        mid = None
        if split:
            # An intraday bar half-way through the in-sample period: a second cut for the
            # look-ahead test, mid-session so a same-day peek is caught too.
            mid = con.execute(
                f'SELECT quantile_disc(t, 0.5) FROM (SELECT TRY_CAST("{tc}" AS TIMESTAMP) AS t FROM '
                f"{_reader(item)}('{_abs(data_dir, item)}')) WHERE t < TIMESTAMP '{split}'").fetchone()[0]
    finally:
        con.close()
    return {**base, "time_column": tc, "first_date": str(lo), "last_date": str(hi),
            "days": days, "split_date": str(split) if split else None,
            "mid_cut": mid.strftime("%Y-%m-%d %H:%M:%S") if mid else None}


_mirror_locks: dict[str, threading.Lock] = {}


def cuts(obj: dict) -> list[str]:
    """Where the look-ahead test cuts the data: the split, then the mid in-sample bar."""
    out = [obj["split_date"]] if obj.get("split_date") else []
    mid = obj["metric"].get("mid_cut")
    if mid and out and mid < out[0]:
        out.append(mid)
    return out


def build_mirror(obj: dict, data_dir: str, cut: str | None = None, only: set[str] | None = None) -> dict:
    """Copies of every time-indexed dataset with the rows AT OR AFTER `cut` removed
    (default: the split -- the in-sample view agents explore). With `only`, just those
    datasets (by view name) are copied -- the ones a candidate is known to read.

    Returns {"root", "items": [{view, path, mount_rel, kind, time_column, rows_kept}], "skipped": [..]}.
    Cached per objective and cut, rebuilt when the data folder's files change.
    """
    split = cut or obj["split_date"]
    tag = "".join(ch for ch in split if ch.isdigit())
    if only is not None:
        tag += "-" + hashlib.sha1("|".join(sorted(only)).encode()).hexdigest()[:8]
    root = WORK_ROOT / obj["id"] / f"mirror-{tag}"
    manifest_path = root / "manifest.json"
    catalog = datasource.catalog(data_dir)
    signature = []
    for item in catalog:
        p = Path(data_dir) / item["path"].replace("/*.parquet", "")
        try:
            st = p.stat()
            signature.append([item["path"], int(st.st_mtime), item.get("bytes", 0)])
        except OSError:
            pass
    lock = _mirror_locks.setdefault(str(root), threading.Lock())
    with lock:
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
            if cached.get("split") == split and cached.get("signature") == signature:
                return cached
        except (OSError, ValueError):
            pass
        import shutil
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        items, skipped = [], []
        for item in catalog:
            if only is not None and item["view"] not in only:
                continue  # the candidate does not read it: nothing to hide, nothing to copy
            if item["format"] not in ("parquet", "csv", "tsv"):
                skipped.append({"view": item["view"], "reason": f"{item['format']} is not truncated"})
                continue
            if (item.get("bytes") or 0) > MAX_TRUNCATE_BYTES:
                skipped.append({"view": item["view"], "reason": "larger than 20 GiB"})
                continue
            tc = detect_time_column(data_dir, item, obj.get("time_column"))
            if tc is None:
                continue  # not time-indexed: identical in both runs, nothing to hide
            dataset = item.get("dataset") or item["path"].endswith("/*.parquet")
            rel = item["path"][: -len("/*.parquet")] if dataset else item["path"]
            if dataset:
                target = root / rel / "part-0.parquet"
                fmt = "(FORMAT parquet)"
            elif item["format"] == "parquet":
                target = root / rel
                fmt = "(FORMAT parquet)"
            else:
                target = root / rel
                fmt = "(FORMAT csv, HEADER true" + (", DELIMITER '\t')" if item["format"] == "tsv" else ")")
            target.parent.mkdir(parents=True, exist_ok=True)
            con = duckdb.connect(":memory:")
            try:
                con.execute("SET TimeZone = 'UTC'")
                con.execute(
                    f"COPY (SELECT * FROM {_reader(item)}('{_abs(data_dir, item)}') "
                    f"WHERE TRY_CAST(\"{tc}\" AS TIMESTAMP) < TIMESTAMP '{split}') "
                    f"TO '{target.as_posix()}' {fmt}"
                )
                kept = con.execute(f"SELECT count(*) FROM {_reader(item)}('{target.as_posix()}')").fetchone()[0]
            except duckdb.Error as exc:
                skipped.append({"view": item["view"], "reason": str(exc).splitlines()[0][:200]})
                continue
            finally:
                con.close()
            items.append({"view": item["view"], "path": item["path"], "mount_rel": rel,
                          "kind": "dir" if dataset else "file", "time_column": tc, "rows_kept": kept})
        doc = {"root": str(root), "split": split, "signature": signature, "items": items,
               "skipped": skipped, "built_at": time.time()}
        manifest_path.write_text(json.dumps(doc), encoding="utf-8")
        return doc


def _mounts(data_dir: str, mirror: dict | None) -> list[tuple[str, str]]:
    mounts = [(data_dir, "/data")]
    for it in (mirror or {}).get("items", []):
        mounts.append((str(Path(mirror["root"]) / it["mount_rel"]), f"/data/{it['mount_rel']}"))
    return mounts


# =======================================================================================
# Forecast features (causal forecaster output as a dataset)
# =======================================================================================
MAX_FEATURE_ANCHORS = 60_000
FEATURE_QUANTILES = [0.1, 0.5, 0.9]


def _features_root(oid: str) -> Path:
    return WORK_ROOT / oid / "features"


def list_features(oid: str) -> list[dict]:
    out = []
    root = _features_root(oid)
    for meta in sorted(root.glob("*.json")) if root.is_dir() else []:
        try:
            out.append(json.loads(meta.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def _feature_catalog(oid: str) -> list[dict]:
    return [{"view": f["view"], "path": f["file"], "format": "parquet", "root": "/features"}
            for f in list_features(oid)]


def features_dir(obj: dict, cut: str | None) -> str | None:
    """The features folder a run mounts at /features: all of it, or copies truncated at `cut`
    (rows at or after the cut removed), rebuilt when a feature is added or recomputed."""
    root = _features_root(obj["id"])
    feats = list_features(obj["id"])
    if not feats:
        return None
    if cut is None:
        return str(root)
    tag = "".join(ch for ch in cut if ch.isdigit())
    out = WORK_ROOT / obj["id"] / f"features-cut-{tag}"
    out.mkdir(parents=True, exist_ok=True)
    for f in feats:
        src, dst = root / f["file"], out / f["file"]
        if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        con = duckdb.connect(":memory:")
        try:
            con.execute(f"COPY (SELECT * FROM read_parquet('{src.as_posix()}') WHERE t < TIMESTAMP '{cut}') "
                        f"TO '{dst.as_posix()}' (FORMAT parquet)")
        finally:
            con.close()
    return str(out)


_EXPR_NODES = None


def series_expression(expr: str, columns: list[str]) -> str:
    """Validate a series expression -- a column, or arithmetic over columns -- and return it
    as DuckDB SQL. Only columns of the dataset, numbers, + - * /, parentheses and a few pure
    math functions are allowed: the expression is spliced into a query, so nothing else may
    get through (no subqueries, no file readers, no statements)."""
    import sqlglot
    from sqlglot import exp

    global _EXPR_NODES
    if _EXPR_NODES is None:
        _EXPR_NODES = (exp.Column, exp.Identifier, exp.Literal, exp.Paren, exp.Add, exp.Sub, exp.Mul,
                       exp.Div, exp.Neg, exp.Abs, exp.Ln, exp.Sqrt, exp.Exp, exp.Pow, exp.Greatest,
                       exp.Least, exp.Sign if hasattr(exp, "Sign") else exp.Abs, exp.Nullif, exp.Coalesce)
    names = {c.lower(): c for c in columns}
    if expr in columns:
        return f'"{expr}"'
    try:
        tree = sqlglot.parse_one(expr, read="duckdb")
    except sqlglot.errors.ParseError as exc:
        raise HTTPException(status_code=400, detail=f"could not parse the expression: {exc}") from None
    for node in tree.walk():
        if not isinstance(node, _EXPR_NODES):
            raise HTTPException(status_code=400, detail=(
                f"{type(node).__name__} is not allowed in a series expression -- use columns, numbers, "
                "+ - * /, parentheses, abs, ln, sqrt, exp, power, greatest, least, nullif, coalesce"))
        if isinstance(node, exp.Column):
            if node.table or node.name.lower() not in names:
                raise HTTPException(status_code=400, detail=f"unknown column {node.sql()!r}")
            node.replace(exp.column(names[node.name.lower()], quoted=True))
    return tree.sql(dialect="duckdb")


def _load_series(data_dir: str, obj: dict, dataset: str, column: str, end: str | None = None,
                 last: int | None = None) -> tuple[list, list[float], str]:
    """(timestamps, values, time column) of one numeric column -- or an expression over
    columns -- oldest first, one value per distinct timestamp. `end` excludes rows at or
    after it; `last` keeps only the final N."""
    item = next((i for i in datasource.catalog(data_dir) if dataset in (i["view"], i["path"])), None)
    if item is None:
        raise HTTPException(status_code=400, detail=f"no dataset {dataset!r}")
    tc = detect_time_column(data_dir, item, obj.get("time_column") if dataset == obj.get("dataset") else None)
    if tc is None:
        raise HTTPException(status_code=400, detail=f"{dataset} has no time column")
    con = duckdb.connect(":memory:")
    try:
        cols = [c[0] for c in con.execute(f"DESCRIBE SELECT * FROM {_reader(item)}('{_abs(data_dir, item)}')").fetchall()]
    finally:
        con.close()
    value_sql = series_expression(column, cols)
    where = f"WHERE t < TIMESTAMP '{end}'" if end else ""
    con = duckdb.connect(":memory:")
    try:
        con.execute("SET TimeZone = 'UTC'")
        rows = con.execute(
            f'SELECT t, v FROM (SELECT TRY_CAST("{tc}" AS TIMESTAMP) AS t, avg(TRY_CAST(({value_sql}) AS DOUBLE)) AS v '
            f"FROM {_reader(item)}('{_abs(data_dir, item)}') GROUP BY 1) {where} "
            f"{'AND' if where else 'WHERE'} t IS NOT NULL AND v IS NOT NULL AND isfinite(v) ORDER BY t"
        ).fetchall()
    except duckdb.Error as exc:
        raise HTTPException(status_code=400, detail=f"could not read {dataset}.{column}: {str(exc).splitlines()[0]}") from None
    finally:
        con.close()
    if last:
        rows = rows[-last:]
    return [r[0] for r in rows], [float(r[1]) for r in rows], tc


def _forecaster(model: str | None) -> tuple[Any, dict]:
    from .tsfm import ts_manager

    running = ts_manager.running()
    if not running:
        raise HTTPException(status_code=409, detail="no time-series model is loaded -- load one on the Models page")
    inst = ts_manager.for_model(model) if model else running[0]
    if inst is None:
        raise HTTPException(status_code=400, detail=f"{model} is not loaded; loaded: {', '.join(i.model_id for i in running)}")
    status = next((x for x in ts_manager.statuses() if x["model_id"] == inst.model_id), {})
    return ts_manager, {"model": inst.model_id, **((status.get("health") or {}))}


class FeatureReq(BaseModel):
    column: str | None = Field(None, max_length=500, description="a column, or an expression over columns")
    columns: list[str] | None = Field(None, max_length=12, description="several series, each forecast on its own")
    dataset: str | None = Field(None, max_length=300)
    horizon: int = Field(12, ge=1, le=256)
    every: int = Field(0, ge=0, le=100_000, description="bars between forecasts; 0 = automatic")
    context: int = Field(512, ge=16, le=8192)
    model: str | None = None
    name: str | None = Field(None, max_length=60)
    # Candle models (Kronos): the bar size candles are built at ("1min", "5min", "30s", ...).
    bar: str | None = Field(None, max_length=10, pattern=r"^\d+(s|min|h)$")
    samples: int | None = Field(None, ge=1, le=64)
    # Covariate models (Chronos-2): other columns/expressions the forecast READS as inputs
    # (past values only -- never future ones, which would leak), and calendar features
    # (time of day, weekday), which are known ahead and so are also given for the horizon.
    covariates: list[str] | None = Field(None, max_length=40)
    calendar: bool = False


def _slug(text: str, n: int = 40) -> str:
    import re

    return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")[:n] or "series"


def _skill(values: list[float], anchors: list[int], med: list[float], q10: list[float], q90: list[float],
           horizon: int, times: list, split: str | None) -> dict:
    """How good the forecasts were, on IN-SAMPLE anchors only (the holdout stays unseen).

    skill = 1 - MAE(forecast) / MAE(no-change forecast): > 0 beats "it stays where it is".
    direction = share of anchors where the forecast called the direction of the change right.
    coverage = share of outcomes inside the 10-90% band (0.8 is calibrated).
    """
    import datetime as _dt

    cut = _dt.datetime.fromisoformat(split) if split else None
    e_fc = e_nv = 0.0
    n = hits = dir_n = inside = 0
    for k, i in enumerate(anchors):
        j = i + horizon
        if j >= len(values) or (cut is not None and times[j] >= cut):
            continue
        real, last = values[j], values[i]
        e_fc += abs(med[k] - real)
        e_nv += abs(last - real)
        n += 1
        if real != last and med[k] != last:
            dir_n += 1
            hits += (med[k] > last) == (real > last)
        inside += q10[k] <= real <= q90[k]
    if n < 20 or e_nv == 0:
        return {"anchors_scored": n}
    return {"anchors_scored": n, "skill_vs_no_change": round(1 - e_fc / e_nv, 4),
            "direction_accuracy": round(hits / dir_n, 4) if dir_n else None,
            "band_coverage_10_90": round(inside / n, 4)}


MAX_KRONOS_ANCHORS = 3000
KRONOS_BATCH = 64


def _load_candles(data_dir: str, obj: dict, dataset: str, bar: str) -> tuple[list, "list[list[float]]"]:
    """(bar timestamps, [[open, high, low, close, volume]]) resampled to `bar`, each bar
    stamped at its LAST underlying row -- the moment it is complete -- so a forecast made from
    it is causal as stamped."""
    item = next((i for i in datasource.catalog(data_dir) if dataset in (i["view"], i["path"])), None)
    if item is None:
        raise HTTPException(status_code=400, detail=f"no dataset {dataset!r}")
    tc = detect_time_column(data_dir, item, obj.get("time_column") if dataset == obj.get("dataset") else None)
    con = duckdb.connect(":memory:")
    try:
        cols = {c[0].lower(): c[0] for c in con.execute(
            f"DESCRIBE SELECT * FROM {_reader(item)}('{_abs(data_dir, item)}')").fetchall()}
    finally:
        con.close()
    need = [cols.get(k) for k in ("open", "high", "low", "close")]
    if tc is None or not all(need):
        raise HTTPException(status_code=400, detail=f"{dataset} needs a time column and open/high/low/close columns for a candle model")
    vol = cols.get("volume")
    n, unit = int(bar[:-3] if bar.endswith("min") else bar[:-1]), ("min" if bar.endswith("min") else bar[-1])
    secs = n * {"s": 1, "min": 60, "h": 3600}[unit]
    con = duckdb.connect(":memory:")
    try:
        con.execute("SET TimeZone = 'UTC'")
        rows = con.execute(f"""
            WITH b AS (
              SELECT TRY_CAST("{tc}" AS TIMESTAMP) AS t, CAST("{need[0]}" AS DOUBLE) o, CAST("{need[1]}" AS DOUBLE) h,
                     CAST("{need[2]}" AS DOUBLE) l, CAST("{need[3]}" AS DOUBLE) c,
                     {f'CAST("{vol}" AS DOUBLE)' if vol else '0.0'} v
              FROM {_reader(item)}('{_abs(data_dir, item)}')
            )
            SELECT max(t) AS stamp, arg_min(o, t), max(h), min(l), arg_max(c, t), sum(v)
            FROM b WHERE t IS NOT NULL AND c IS NOT NULL AND c > 0
            GROUP BY time_bucket(INTERVAL '{secs} seconds', t) ORDER BY stamp""").fetchall()
    finally:
        con.close()
    return [r[0] for r in rows], [[float(x or 0.0) for x in r[1:]] for r in rows]


async def _build_kronos_feature(obj: dict, req: FeatureReq, mgr, info: dict, project: dict) -> dict:
    """forecast_feature for a candle model: causal candle forecasts every `every` bars."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset = req.dataset or obj.get("dataset")
    bar = req.bar or "1min"
    horizon = min(req.horizon, 60)
    samples = req.samples or 4
    context = min(req.context, int(info.get("context_length") or 512))
    times, candles = await asyncio.to_thread(_load_candles, project["data_dir"], obj, dataset, bar)
    n = len(candles)
    if n <= context + horizon:
        raise HTTPException(status_code=400, detail=f"only {n} {bar} bars; need more than {context + horizon}")
    floor_every = -(-(n - context) // MAX_KRONOS_ANCHORS)
    every = max(req.every or 0, floor_every, 1)
    anchors = list(range(context - 1, n, every))
    name = _slug(req.name or f"kronos_{bar}_h{horizon}_e{every}", 50)
    root = _features_root(obj["id"])
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / f"{name}.json"
    secs = int((times[-1] - times[-2]).total_seconds()) if n > 1 else 60
    params = {"dataset": dataset, "series": [f"candles@{bar}"], "horizon": horizon, "every": every,
              "context": context, "model": info["model"], "bar": bar, "samples": samples}
    try:
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("params") == params and (root / old["file"]).is_file():
            return {**old, "cached": True}
    except (OSError, ValueError, KeyError):
        pass
    t0 = time.time()
    med, q10, q90, hi, lo, mean_path = [], [], [], [], [], []
    for b in range(0, len(anchors), KRONOS_BATCH):
        chunk = anchors[b:b + KRONOS_BATCH]
        try:
            res = await mgr.forecast(info["model"], {
                "candles": [candles[i - context + 1:i + 1] for i in chunk],
                "timestamps": [[t.isoformat() for t in times[i - context + 1:i + 1]] for i in chunk],
                "freq_seconds": max(1, secs), "horizon": horizon, "samples": samples,
                "quantiles": FEATURE_QUANTILES})
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Kronos failed at batch {b // KRONOS_BATCH}: {exc}") from None
        for fc in res.get("forecasts") or []:
            qs = fc.get("quantiles") or {}
            med.append(fc["median"][-1])
            mean_path.append(sum(fc["median"]) / len(fc["median"]))
            q10.append((qs.get("0.1") or fc["median"])[-1])
            q90.append((qs.get("0.9") or fc["median"])[-1])
            hi.append(max(fc.get("high_q90") or fc["median"]))
            lo.append(min(fc.get("low_q10") or fc["median"]))
    closes = [c[3] for c in candles]
    last = [closes[i] for i in anchors]
    table = pa.table({
        "t": pa.array([times[i] for i in anchors], type=pa.timestamp("us")),
        "last": last, "fc_median": med, "fc_q10": q10, "fc_q90": q90, "fc_path_mean": mean_path,
        "fc_change": [m - v for m, v in zip(med, last)], "fc_high_q90": hi, "fc_low_q10": lo,
    })
    file = f"{name}.parquet"
    pq.write_table(table, root / file)
    skill = _skill(closes, anchors, med, q10, q90, horizon, times, obj.get("split_date"))
    meta = {
        "view": f"fc_{name}", "file": file, "params": params, "rows": len(anchors),
        "seconds": round(time.time() - t0, 1), "created_at": time.time(),
        "adjusted": (f"`every` set to {every} {bar} bars: at most {MAX_KRONOS_ANCHORS} Kronos forecasts per feature "
                     f"(it generates candle by candle, ~0.4-4 s each)" if every != (req.every or every) or not req.every else None),
        "columns": {
            "t": f"timestamp of the {bar} bar the forecast was made AT (bar complete; data up to and including it)",
            "last": f"close of that {bar} bar",
            "fc_median": f"median forecast close {horizon} bars ({bar}) ahead",
            "fc_q10": "10% quantile of that close", "fc_q90": "90% quantile of that close",
            "fc_path_mean": "mean of the median close path over the horizon",
            "fc_change": "fc_median - last",
            "fc_high_q90": "highest 90%-quantile high over the horizon (upside range)",
            "fc_low_q10": "lowest 10%-quantile low over the horizon (downside range)",
        },
        "skill": {f"candles@{bar}": skill},
        "skill_note": ("in-sample only. skill_vs_no_change > 0 beats 'it stays where it is'; direction 0.5 is a "
                       "coin flip; band_coverage_10_90 should be ~0.8"),
        "usage": (f'f = ft.load("fc_{name}"); df = pd.merge_asof(df.sort_values(TIME), f.sort_values("t"), '
                  f'left_on=TIME, right_on="t", direction="backward")'),
    }
    meta_path.write_text(json.dumps(meta, default=str), encoding="utf-8")
    return meta


def _calendar(t: np.ndarray, horizon: int) -> tuple[dict, dict]:
    """Time-of-day and weekday for the history AND the horizon (future bar times follow the
    grid's typical spacing). Known in advance, so they are the only future covariates."""
    import numpy as _np

    ts = t.astype("datetime64[s]").astype("int64")
    step = int(_np.median(_np.diff(ts[-50:]))) if len(ts) > 2 else 60
    fut = ts[-1] + step * _np.arange(1, horizon + 1)

    def feats(x):
        mins = (x // 60) % 1440
        dow = ((x // 86400) + 3) % 7   # 1970-01-01 was a Thursday
        return {"minute_of_day": (mins / 1440.0).astype(_np.float32), "weekday": (dow / 6.0).astype(_np.float32)}

    return feats(ts), feats(fut)


async def _build_covariate_feature(obj: dict, req: FeatureReq, mgr, info: dict, project: dict) -> dict:
    """forecast_feature with inputs: forecast the target(s) while reading other columns.

    Anchored causally like every feature (the forecast at t reads rows up to t). Stored with
    its in-sample skill AND the same forecast without the inputs, so the lift the inputs give
    is on record rather than assumed.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .tslab import load_frame, score

    dataset = req.dataset or obj.get("dataset")
    targets = [c for c in (req.columns or ([req.column] if req.column else [])) if c and c.strip()]
    if not targets:
        raise HTTPException(status_code=400, detail="give the target `column` to forecast")
    covs = [c for c in (req.covariates or []) if c not in targets]
    horizon = req.horizon
    context = min(req.context, int(info.get("context_length") or req.context))
    t, cols = await asyncio.to_thread(load_frame, project, obj, dataset, [*targets, *covs], req.bar, None)
    n = len(t)
    if n <= context + horizon:
        raise HTTPException(status_code=400, detail=f"only {n} rows; need more than {context + horizon}")
    floor_every = -(-(n - context) // MAX_FEATURE_ANCHORS)
    every = max(req.every or max(1, (n - context) // 20_000), floor_every, 1)
    anchors = list(range(context - 1, n, every))
    tag = "_x_" + "_".join(_slug(c, 10) for c in covs[:4]) if covs else ""
    name = _slug(req.name or (f"{'_'.join(_slug(c, 16) for c in targets)}{tag}{'_cal' if req.calendar else ''}_h{horizon}_e{every}"), 50)
    root = _features_root(obj["id"])
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / f"{name}.json"
    params = {"dataset": dataset, "series": targets, "covariates": covs, "calendar": req.calendar, "horizon": horizon,
              "every": every, "context": context, "model": info["model"], "bar": req.bar}
    try:
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("params") == params and (root / old["file"]).is_file():
            return {**old, "cached": True}
    except (OSError, ValueError, KeyError):
        pass

    t0 = time.time()
    per_item = context * (len(targets) + len(covs) + (2 if req.calendar else 0))
    batch = max(1, min(512, 1_500_000 // max(1, per_item)))

    async def forecast_all(with_inputs: bool, which: list[int]) -> dict[str, dict[str, list]]:
        out = {c: {"med": [], "q10": [], "q90": [], "path": []} for c in targets}
        for b in range(0, len(which), batch):
            chunk = which[b:b + batch]
            items = []
            for a in chunk:
                s0 = a - context + 1
                tgt = [cols[c][s0:a + 1].tolist() for c in targets]
                it: dict[str, Any] = {"target": tgt if len(tgt) > 1 else tgt[0]}
                past: dict[str, list] = {}
                fut: dict[str, list] = {}
                if with_inputs:
                    past = {c: cols[c][s0:a + 1].tolist() for c in covs}
                    if req.calendar:
                        ph, pf = _calendar(t[s0:a + 1], horizon)
                        past.update({k: v.tolist() for k, v in ph.items()})
                        fut = {k: v.tolist() for k, v in pf.items()}
                if past:
                    it["past_covariates"] = past
                if fut:
                    it["future_covariates"] = fut
                items.append(it)
            try:
                res = await mgr.forecast(info["model"], {"inputs": items, "horizon": horizon, "quantiles": FEATURE_QUANTILES})
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=502, detail=f"forecaster failed: {exc}") from None
            for fc in res.get("forecasts") or []:
                for c, v in zip(targets, fc.get("variates") or [fc]):
                    q = v.get("quantiles") or {}
                    out[c]["med"].append(v["median"][-1])
                    out[c]["path"].append(sum(v["median"]) / len(v["median"]))
                    out[c]["q10"].append((q.get("0.1") or v["median"])[-1])
                    out[c]["q90"].append((q.get("0.9") or v["median"])[-1])
        return out

    main = await forecast_all(True, anchors)
    # The lift of the inputs, measured on in-sample anchors whose outcome is known.
    split = obj.get("split_date")
    in_idx = [k for k, a in enumerate(anchors) if a + horizon < n and (not split or str(t[a + horizon])[:10] < split)]
    base = await forecast_all(False, [anchors[k] for k in in_idx]) if (covs or req.calendar) and in_idx else None
    cols_out: dict[str, Any] = {"t": pa.array([t[a] for a in anchors], type=pa.timestamp("us"))}
    described = {"t": "timestamp the forecast was made AT (data up to and including t)"}
    skills = {}
    single = len(targets) == 1
    for c in targets:
        pre = "" if single else f"{_slug(c, 24)}_"
        last = [float(cols[c][a]) for a in anchors]
        cols_out[f"{pre}last"] = last
        cols_out[f"{pre}fc_median"] = main[c]["med"]
        cols_out[f"{pre}fc_q10"] = main[c]["q10"]
        cols_out[f"{pre}fc_q90"] = main[c]["q90"]
        cols_out[f"{pre}fc_path_mean"] = main[c]["path"]
        cols_out[f"{pre}fc_change"] = [m - v for m, v in zip(main[c]["med"], last)]
        described.update({f"{pre}last": f"{c} at t", f"{pre}fc_median": f"median forecast of {c}, {horizon} rows ahead"
                          + (f", reading {', '.join(covs)}" if covs else "") + (" + calendar" if req.calendar else ""),
                          f"{pre}fc_q10": "10% quantile", f"{pre}fc_q90": "90% quantile",
                          f"{pre}fc_path_mean": "mean of the median path", f"{pre}fc_change": "fc_median - last"})
        target = cols[c]
        ins = [anchors[k] + 1 for k in in_idx]  # score() expects the index AFTER the last input row
        with_inputs = score(target, ins, horizon, np.array([main[c]["med"][k] for k in in_idx]),
                            np.array([main[c]["q10"][k] for k in in_idx]), np.array([main[c]["q90"][k] for k in in_idx])) if in_idx else {}
        entry = {"with_inputs": {k: v for k, v in with_inputs.items() if not k.startswith("_")}}
        if base:
            without = score(target, ins, horizon, np.array(base[c]["med"]), np.array(base[c]["q10"]), np.array(base[c]["q90"]))
            from .tslab import paired

            entry["without_inputs"] = {k: v for k, v in without.items() if not k.startswith("_")}
            entry["lift"] = paired(without, with_inputs)
            if with_inputs.get("skill") is not None and without.get("skill") is not None:
                entry["lift_skill"] = round(with_inputs["skill"] - without["skill"], 5)
            if with_inputs.get("direction") is not None and without.get("direction") is not None:
                entry["lift_direction"] = round(with_inputs["direction"] - without["direction"], 4)
        skills[c] = entry
    file = f"{name}.parquet"
    pq.write_table(pa.table(cols_out), root / file)
    meta = {"view": f"fc_{name}", "file": file, "params": params, "rows": len(anchors),
            "seconds": round(time.time() - t0, 1), "created_at": time.time(), "columns": described,
            "skill": skills,
            "skill_note": ("in-sample only; with_inputs vs without_inputs is the same model at the same points -- "
                           "lift_skill > 0 means the inputs made the forecast better"),
            "usage": (f'f = ft.load("fc_{name}"); df = pd.merge_asof(df.sort_values(TIME), f.sort_values("t"), '
                      f'left_on=TIME, right_on="t", direction="backward")')}
    meta_path.write_text(json.dumps(meta, default=str), encoding="utf-8")
    return meta


async def build_feature(obj: dict, req: FeatureReq, requested_by: str | None = None) -> dict:
    """build_feature, plus the record needed to reproduce it: the full request, the model that
    actually ran and its reported settings, the quantiles, who asked, and a checksum of the
    stored forecasts. Cached results keep the record they were built with."""
    meta = await _build_feature(obj, req)
    if meta.get("cached") or meta.get("recipe"):
        return meta
    import hashlib

    root = _features_root(obj["id"])
    try:
        digest = hashlib.sha256((root / meta["file"]).read_bytes()).hexdigest()
    except OSError:
        digest = None
    try:
        _, info = _forecaster(req.model)
    except HTTPException:
        info = {}
    meta["recipe"] = {
        "request": req.model_dump(), "resolved": meta.get("params"),
        "model": {k: info.get(k) for k in ("model", "family", "context_length", "native_horizon",
                                            "supports_covariates", "revision", "version") if info.get(k) is not None},
        "quantiles": FEATURE_QUANTILES, "requested_by": requested_by, "sha256": digest,
        "causal": "the forecast stamped t read rows up to and including t only",
    }
    (root / f"{meta['file'].rsplit('.', 1)[0]}.json").write_text(json.dumps(meta, default=str), encoding="utf-8")
    return meta


async def _build_feature(obj: dict, req: FeatureReq) -> dict:
    """Run the forecaster over one or more series causally and store the result as a dataset.

    A series is a column or an arithmetic expression over columns (``GEX / Pinning_TotalAbsGex``).
    The forecaster is univariate: several series are forecast independently and stored side by
    side. Anchors are every `every`-th bar; the forecast at anchor t is made from the `context`
    values ending AT t (inclusive) -- never later ones -- so it is usable by a position decided
    at the close of bar t. Each series' in-sample forecast skill is measured and stored with it.
    """
    project = projects.get(obj["project_id"])
    if project is None:
        raise HTTPException(status_code=404, detail="no such project")
    mgr, info = _forecaster(req.model)
    if req.covariates or req.calendar:
        if not info.get("supports_covariates"):
            raise HTTPException(status_code=400, detail=(
                f"{info['model']} forecasts from one series only; load amazon/chronos-2 to use input columns"))
        return await _build_covariate_feature(obj, req, mgr, info, project)
    if info.get("family") == "kronos":
        # A candle model reads whole OHLCV bars, not one column.
        return await _build_kronos_feature(obj, req, mgr, info, project)
    series_list = [c for c in (req.columns or ([req.column] if req.column else [])) if c and c.strip()]
    if not series_list:
        raise HTTPException(status_code=400, detail="give `column` (or `columns`) to forecast")
    ctx_max = int(info.get("context_length") or req.context)
    context = min(req.context, ctx_max)
    dataset = req.dataset or obj.get("dataset")
    if not dataset:
        raise HTTPException(status_code=400, detail="say which dataset the column is in")

    loaded = []
    for expr in series_list:
        times, values, _ = await asyncio.to_thread(_load_series, project["data_dir"], obj, dataset, expr)
        loaded.append((expr, times, values))
    # One shared anchor grid: the timestamps every series has.
    base_times = loaded[0][1]
    for _, t, _ in loaded[1:]:
        if t != base_times:
            common = set(base_times).intersection(t)
            base_times = [x for x in base_times if x in common]
    aligned = []
    for expr, t, v in loaded:
        pos = {x: k for k, x in enumerate(t)}
        aligned.append((expr, [v[pos[x]] for x in base_times]))
    times = base_times
    n = len(times)
    if n <= context:
        raise HTTPException(status_code=400, detail=f"only {n} points; need more than the {context}-point context")
    # Too dense a grid is ADJUSTED, not refused. A model asked for `every=1` (713K forecasts),
    # got a 400, and never managed to recover from it -- so the operator's "run a forecast"
    # produced zero forecaster calls across a dozen iterations. Clamp and say so instead.
    floor_every = -(-(n - context) * len(aligned) // MAX_FEATURE_ANCHORS)  # ceil
    requested = req.every
    every = max(req.every or max(1, (n - context) // 20_000), floor_every, 1)
    adjusted = requested and every != requested
    anchors = list(range(context - 1, n, every))
    name = _slug(req.name or ("_".join(_slug(e, 16) for e in series_list) + f"_h{req.horizon}_e{every}"), 50)
    root = _features_root(obj["id"])
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / f"{name}.json"
    params = {"dataset": dataset, "series": series_list, "horizon": req.horizon, "every": every,
              "context": context, "model": info["model"]}
    try:
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("params") == params and (root / old["file"]).is_file():
            return {**old, "cached": True}
    except (OSError, ValueError, KeyError):
        pass

    import pyarrow as pa
    import pyarrow.parquet as pq

    t0 = time.time()
    batch = max(1, min(256, 90_000 // context))
    cols: dict[str, Any] = {"t": pa.array([times[i] for i in anchors], type=pa.timestamp("us"))}
    described: dict[str, str] = {"t": "bar timestamp the forecasts were made AT (data up to and including t)"}
    skills: dict[str, dict] = {}
    single = len(aligned) == 1
    for expr, values in aligned:
        med, q10, q90, mean_path = [], [], [], []
        for b in range(0, len(anchors), batch):
            chunk = anchors[b:b + batch]
            try:
                res = await mgr.forecast(info["model"], {
                    "series": [values[i - context + 1:i + 1] for i in chunk],
                    "horizon": req.horizon, "quantiles": FEATURE_QUANTILES})
            except Exception as exc:  # noqa: BLE001 -- TsError and transport errors alike
                raise HTTPException(status_code=502, detail=f"forecaster failed on {expr}: {exc}") from None
            for fc in res.get("forecasts") or []:
                qs = fc.get("quantiles") or {}
                med.append(fc["median"][-1])
                mean_path.append(sum(fc["median"]) / len(fc["median"]))
                q10.append((qs.get("0.1") or fc["median"])[-1])
                q90.append((qs.get("0.9") or fc["median"])[-1])
        last = [values[i] for i in anchors]
        pre = "" if single else f"{_slug(expr, 24)}_"
        cols[f"{pre}last"] = last
        cols[f"{pre}fc_median"] = med
        cols[f"{pre}fc_q10"] = q10
        cols[f"{pre}fc_q90"] = q90
        cols[f"{pre}fc_path_mean"] = mean_path
        cols[f"{pre}fc_change"] = [m - v for m, v in zip(med, last)]
        described.update({
            f"{pre}last": f"{expr} at t",
            f"{pre}fc_median": f"median forecast of {expr}, {req.horizon} bars after t",
            f"{pre}fc_q10": "10% quantile of that forecast", f"{pre}fc_q90": "90% quantile of that forecast",
            f"{pre}fc_path_mean": f"mean of the median path over the next {req.horizon} bars",
            f"{pre}fc_change": "fc_median - last (the forecast move; divide by last for a return if it is a price)",
        })
        skills[expr] = _skill(values, anchors, med, q10, q90, req.horizon, times, obj.get("split_date"))
    file = f"{name}.parquet"
    pq.write_table(pa.table(cols), root / file)
    meta = {
        "view": f"fc_{name}", "file": file, "params": params, "rows": len(anchors),
        "adjusted": (f"`every` raised from {requested} to {every}: at most {MAX_FEATURE_ANCHORS} forecasts per "
                     f"feature. Positions still update every bar -- join with merge_asof(direction='backward')."
                     if adjusted else None),
        "seconds": round(time.time() - t0, 1), "created_at": time.time(), "columns": described,
        "skill": skills,
        "skill_note": ("in-sample only. skill_vs_no_change > 0 means the forecast beat 'it stays where it "
                       "is'; direction_accuracy 0.5 is a coin flip; band_coverage_10_90 should be ~0.8"),
        "usage": (f'f = ft.load("fc_{name}"); df = pd.merge_asof(df.sort_values(TIME), f.sort_values("t"), '
                  f'left_on=TIME, right_on="t", direction="backward")  # never "forward" or "nearest"'),
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta


# =======================================================================================
# Running a candidate
# =======================================================================================
def _failure_note(stderr: str) -> str:
    """The failure line agents read, with the fix for mistakes the team keeps repeating."""
    note = "the script failed -- see stderr"
    m = re.search(r"KeyError: '(fc_\w+)'", stderr or "")
    if m:
        note += (f". {m.group(1)!r} is missing: every forecast feature has the same column names (t, fc_median, "
                 "fc_q10, ...), so after merging two of them pandas renames them fc_median_x / fc_median_y. "
                 'Load each with a prefix -- ft.load("fc_x", prefix="x_") gives x_fc_median -- and merge those')
    return note


HARNESS = """import sys
sys.path.insert(0, "/work/.ft")
_src = open("/work/.ft/candidate.py", encoding="utf-8").read()
exec(compile(_src, "candidate.py", "exec"), {"__name__": "__main__", "__file__": "candidate.py"})
"""


async def _run(code: str, data_dir: str, catalog: list[dict], mirror: dict | None, timeout_s: int,
               obj: dict | None = None, cut: str | None = None, extra_files: dict[str, str] | None = None) -> dict:
    """`cut` also truncates the objective's forecast features (None = full). The project's
    code library is copied in as /work/.ft/lib (``from lib import x``)."""
    entries = [{k: c[k] for k in ("view", "path", "format")} for c in catalog]
    mounts = _mounts(data_dir, mirror)
    if obj is not None:
        fdir = await asyncio.to_thread(features_dir, obj, cut)
        if fdir:
            entries += _feature_catalog(obj["id"])
            mounts.append((fdir, "/features"))
    from .library import module_files

    files = {
        ".ft/ft.py": FT_HELPER.read_text(encoding="utf-8"),
        ".ft/catalog.json": json.dumps(entries),
        ".ft/candidate.py": code,
        **(module_files(obj["project_id"]) if obj is not None else {}),
        **(extra_files or {}),
    }
    report = await execute(HARNESS, timeout_s=timeout_s, files=files, mounts=mounts)
    result = {}
    try:
        result = json.loads((Path(report["run_dir"]) / ".ft" / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    report["result"] = result if isinstance(result, dict) else {}
    try:
        asked = json.loads((Path(report["run_dir"]) / ".ft" / "forecast_requests.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        asked = []
    report["forecast_requests"] = [a for a in asked if isinstance(a, dict) and a.get("name")] if isinstance(asked, list) else []
    try:
        used = json.loads((Path(report["run_dir"]) / ".ft" / "used.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        used = []
    # Forecast features the script actually loaded -- the evidence for the forecast scoreboard.
    if not isinstance(used, list):
        used = []
    report["features_used"] = sorted({u for u in used if isinstance(u, str) and u.startswith("fc_")})
    return report


MAX_AUTO_FORECASTS = 3  # new forecasts one run may cause to be built


async def _run_forecasting(code: str, data_dir: str, catalog: list[dict], mirror: dict | None, timeout_s: int,
                           obj: dict, cut: str | None = None, requested_by: str | None = None) -> dict:
    """_run, building any forecast the script asked for with ft.forecast() and running it again.

    A script that calls ft.forecast(...) for a recipe not built yet stops with ForecastPending;
    the recipe is built here -- causally, by the loaded forecaster, named by the hash of the
    recipe so the same call finds it next time -- and the script is re-run. At most
    MAX_AUTO_FORECASTS new forecasts per call; what could not be built is said in stderr."""
    built = 0
    while True:
        rep = await _run(code, data_dir, catalog, mirror, timeout_s, obj, cut)
        asked = [a for a in rep.get("forecast_requests") or []
                 if not (_features_root(obj["id"]) / f"{a['name']}.parquet").is_file()]
        if rep["ok"] or not asked:
            return rep
        problems = []
        for a in asked:
            if built >= MAX_AUTO_FORECASTS:
                problems.append(f"{a['name']}: not built -- at most {MAX_AUTO_FORECASTS} new forecasts per run")
                continue
            recipe = {k: v for k, v in (a.get("recipe") or {}).items() if v is not None}
            try:
                await build_feature(obj, FeatureReq(**recipe, name=a["name"]), requested_by=requested_by)
                built += 1
            except HTTPException as exc:
                problems.append(f"{a['name']}: {exc.detail}")
            except Exception as exc:  # noqa: BLE001 -- a bad recipe is the script's error, not ours
                problems.append(f"{a['name']}: {type(exc).__name__}: {exc}")
        if problems:
            rep["stderr"] = (rep.get("stderr") or "") + "\n[ft] ft.forecast could not be built:\n" + "\n".join(problems)
            return rep


def _positions_file(report: dict) -> Path | None:
    p = Path(report["run_dir"]) / ".ft" / "positions.parquet"
    return p if p.is_file() else None


def _mark_to_market(obj: dict, data_dir: str, positions: Path) -> tuple[list[list], dict]:
    """Daily returns of the reported positions, from the dataset's own prices.

    On the dataset's bar grid (every distinct timestamp with a price), the position in force
    at bar t is the latest one reported at or before t (an as-of join). The return realised at
    bar t is pos[t-1] * (price[t] / price[t-1] - 1), less cost_bps on |pos[t-1] - pos[t-2]| --
    the trade made at the previous bar. Returns are compounded per calendar day.

    Two more daily series ride along in ``info`` (popped by the caller, never stored there):
    ``_gross`` -- the same positions without costs -- and ``_inverted`` -- every position's sign
    flipped, same costs. From them evaluate() says whether a loser lacks an edge, pays too much
    to trade it, or points the wrong way.
    """
    m = obj["metric"]
    item = next((i for i in datasource.catalog(data_dir) if obj["dataset"] in (i["view"], i["path"])), None)
    if item is None:
        raise HTTPException(status_code=400, detail=f"dataset {obj['dataset']!r} is gone from the data folder")
    tc, pc = obj["time_column"], m["price_column"]
    lev = float(m.get("max_leverage") or 1.0)
    cost = float(m.get("cost_bps") or 0.0) / 10_000.0
    con = duckdb.connect(":memory:")
    try:
        con.execute("SET TimeZone = 'UTC'")
        con.execute(f"""
            CREATE TEMP TABLE px AS
            SELECT t, avg(p) AS p FROM (
                SELECT TRY_CAST("{tc}" AS TIMESTAMP) AS t, TRY_CAST("{pc}" AS DOUBLE) AS p
                FROM {_reader(item)}('{_abs(data_dir, item)}')
            ) WHERE t IS NOT NULL AND p IS NOT NULL AND p > 0 GROUP BY t""")
        con.execute(f"""
            CREATE TEMP TABLE pos AS
            SELECT CAST(t AS TIMESTAMP) AS t, greatest(-{lev}, least({lev}, coalesce(CAST(pos AS DOUBLE), 0))) AS pos
            FROM read_parquet('{positions.as_posix()}')""")
        n_pos, n_changes = con.execute(
            "SELECT count(*), count(*) FILTER (WHERE pos IS DISTINCT FROM prev) FROM "
            "(SELECT pos, lag(pos) OVER (ORDER BY t) AS prev FROM pos)").fetchone()
        rows = con.execute(f"""
            WITH g AS (
                SELECT px.t, px.p, coalesce(pos.pos, 0) AS pos
                FROM px ASOF LEFT JOIN pos ON px.t >= pos.t
            ), l AS (
                SELECT t, p, lag(p) OVER w AS pp, lag(pos) OVER w AS p1, lag(pos, 2) OVER w AS p2
                FROM g WINDOW w AS (ORDER BY t)
            )
            SELECT strftime(CAST(t AS DATE), '%Y-%m-%d') AS d,
                   product(1 + coalesce(p1 * (p / pp - 1), 0)
                             - {cost} * abs(coalesce(p1, 0) - coalesce(p2, 0))) - 1 AS r,
                   product(1 + coalesce(p1 * (p / pp - 1), 0)) - 1 AS rg,
                   product(1 - coalesce(p1 * (p / pp - 1), 0)
                             - {cost} * abs(coalesce(p1, 0) - coalesce(p2, 0))) - 1 AS ri
            FROM l GROUP BY d ORDER BY d""").fetchall()
        bars = con.execute("SELECT count(*), min(t), max(t) FROM px").fetchone()
        # How many positions fall inside the priced period at all. Positions indexed by row
        # number arrive as 1970 timestamps; the as-of join then carries the last of them over
        # every bar -- a constant position that scores as if it were a strategy. evaluate()
        # refuses a result whose positions miss the data instead of scoring that.
        span = con.execute(
            "SELECT count(*) FILTER (WHERE pos.t BETWEEN b.lo AND b.hi), min(pos.t), max(pos.t) "
            "FROM pos, (SELECT min(t) lo, max(t) hi FROM px) b").fetchone()
    except duckdb.Error as exc:
        raise HTTPException(status_code=400, detail=f"could not mark positions to market: {str(exc).splitlines()[0]}") from None
    finally:
        con.close()
    info = {"positions": n_pos, "position_changes": n_changes, "bars": bars[0],
            "cost_bps": m.get("cost_bps"), "max_leverage": lev, "price_column": pc,
            "positions_in_data_range": span[0],
            "positions_from": str(span[1]) if span[1] is not None else None,
            "positions_to": str(span[2]) if span[2] is not None else None,
            "data_from": str(bars[1]) if bars[1] is not None else None,
            "data_to": str(bars[2]) if bars[2] is not None else None}
    ok = [x for x in rows if all(v is not None and math.isfinite(v) for v in x[1:])]
    info["_gross"] = [[d, float(g)] for d, _, g, _ in ok]
    info["_inverted"] = [[d, float(i)] for d, _, _, i in ok]
    return [[d, float(r)] for d, r, _, _ in ok], info


def _positions_off_data(mtm: dict) -> str | None:
    """Why marked-to-market positions cannot be scored, if they (almost) all miss the data.

    Fewer than 1% of the positions inside the priced period means they are not dated by the
    bar timestamps -- typically row numbers read as nanoseconds after 1970-01-01."""
    n, inside = int(mtm.get("positions") or 0), int(mtm.get("positions_in_data_range") or 0)
    if n == 0 or inside >= max(1, 0.01 * n):
        return None
    return (f"positions are dated {mtm.get('positions_from')} .. {mtm.get('positions_to')} but the data "
            f"runs {mtm.get('data_from')} .. {mtm.get('data_to')} ({inside} of {n} positions inside it) "
            f"-- report positions indexed by the bar timestamp, e.g. pd.Series(pos.values, "
            f"index=df[time_col]); a RangeIndex (after reset_index()) turns row numbers into 1970 dates")


def _positions_lookahead(full: Path, trunc: Path, cut: str) -> tuple[str, str]:
    """Positions dated before `cut` must not change when the data from `cut` on is removed.

    Every position the FULL run made before the cut must reappear, unchanged, in the truncated
    run. Positions only the truncated run has are ignored: a strategy on resampled bars sees a
    partial bar at the cut and may decide on it there -- an artefact of cutting, not a leak.

    Both sides are compared per microsecond (DuckDB's TIMESTAMP), one position per instant --
    the latest one reported there, as ft.report_positions keeps the last duplicate. Positions
    stamped in nanoseconds can collapse onto one microsecond (row numbers read as 1970 dates
    do, 1000 to one), and the detail lookup used to be a scalar subquery that then returned
    many rows and threw "More than one row returned by a subquery" -- failing the whole
    evaluation with a message that named nothing. A comparison problem is a verdict of
    `error` here, never an exception."""
    side = ("SELECT CAST(t AS TIMESTAMP) t, arg_max(CAST(pos AS DOUBLE), t) pos FROM read_parquet('{path}') "
            "WHERE CAST(t AS TIMESTAMP) < TIMESTAMP '{cut}' GROUP BY 1")
    ctes = (f"WITH a AS ({side.format(path=full.as_posix(), cut=cut)}), "
            f"b AS ({side.format(path=trunc.as_posix(), cut=cut)}) ")
    differs = "b.pos IS NULL OR abs(a.pos - b.pos) > 1e-9 + 1e-6 * abs(a.pos)"
    con = duckdb.connect(":memory:")
    try:
        n, bad, first = con.execute(
            ctes + f"SELECT count(*), count(*) FILTER (WHERE {differs}), min(a.t) FILTER (WHERE {differs}) "
                   "FROM a LEFT JOIN b USING (t)").fetchone()
        detail = ""
        if bad:
            try:  # only decoration for the message: it must never decide the verdict
                row = con.execute(ctes + f"SELECT a.pos, b.pos FROM a LEFT JOIN b USING (t) "
                                         f"WHERE a.t = TIMESTAMP '{first}' LIMIT 1").fetchone()
                if row:
                    detail = f" (first at {first}: position {row[0]} with future data, {row[1]} without)"
            except duckdb.Error:
                detail = f" (first at {first})"
    except duckdb.Error as exc:
        return "error", f"could not compare positions on the cut at {cut}: {str(exc).splitlines()[0][:300]}"
    finally:
        con.close()
    if n < 5:
        return "error", f"too few positions before {cut} to compare"
    if not bad:
        return "pass", f"{n} positions before {cut} identical with later data removed"
    return "fail", (f"{bad} of {n} positions before {cut} changed when data from {cut} on was removed{detail} "
                    f"-- decisions depend on future data")


def _active_cuts(obj: dict, full_pos: Path, k: int = LOOKAHEAD_ACTIVE_CUTS, seed: Any = None) -> list[str]:
    """Cuts placed just after the candidate's own position changes, spread over in-sample.

    One trade is drawn from each of `k` equal slices of the in-sample trades, and the cut goes
    a few seconds to 1.5 hours after it (a different offset each, none on a bar grid). If that
    trade peeked at anything in between, it is gone in the truncated run and the trade changes.
    """
    split = obj["split_date"]
    con = duckdb.connect(":memory:")
    try:
        rows = con.execute(f"""
            SELECT t FROM (
                SELECT CAST(t AS TIMESTAMP) t, CAST(pos AS DOUBLE) pos,
                       lag(CAST(pos AS DOUBLE)) OVER (ORDER BY CAST(t AS TIMESTAMP)) prev
                FROM read_parquet('{full_pos.as_posix()}'))
            WHERE prev IS NOT NULL AND abs(pos - prev) > 1e-12 AND t < TIMESTAMP '{split}'
            ORDER BY t""").fetchall()
    finally:
        con.close()
    changes = [r[0] for r in rows]
    if not changes:
        return []
    from datetime import timedelta

    rng = random.Random(seed)
    n, k = len(changes), min(k, len(changes))
    offsets = list(_CUT_OFFSETS_S)
    rng.shuffle(offsets)
    out = set()
    for i in range(k):
        lo, hi = i * n // k, max(i * n // k + 1, (i + 1) * n // k)
        t = changes[rng.randrange(lo, hi)]
        cut = (t + timedelta(seconds=offsets[i % len(offsets)])).strftime("%Y-%m-%d %H:%M:%S")
        if cut < split:
            out.add(cut)
    return sorted(out)


def _discard_cut(obj: dict, cut: str, mirror_root: str | None) -> None:
    """Remove the truncated copies made for a one-off cut (data mirror and features)."""
    import shutil

    tag = "".join(ch for ch in cut if ch.isdigit())
    if mirror_root:
        shutil.rmtree(mirror_root, ignore_errors=True)
    shutil.rmtree(WORK_ROOT / obj["id"] / f"features-cut-{tag}", ignore_errors=True)


_LOAD_RX = re.compile(r"""ft\.(?:load|path)\(\s*(?:name\s*=\s*)?(['"])([^'"]+)\1""")
# Ways to reach a dataset other than ft.load("<literal>"): with any of these in play, every
# dataset is copied, since which ones are read cannot be told from the source.
_OPAQUE_READS = ("/data", "ft.datasets(", "catalog.json", "glob", "listdir", "scandir", "walk(",
                 "read_parquet(", "read_csv(", "read_json(", "open(", "duckdb", "pyarrow")


def _datasets_used(obj: dict, code: str) -> set[str] | None:
    """Views the candidate (and the library modules it imports) load, or None = can't tell."""
    try:
        from .library import reachable_modules

        sources = [code] + list(reachable_modules(obj["project_id"], code).values())
    except Exception:  # noqa: BLE001
        return None
    names: set[str] = set()
    for src in sources:
        if any(tok in src for tok in _OPAQUE_READS):
            return None
        literal = _LOAD_RX.findall(src)
        if len(literal) != src.count("ft.load(") + src.count("ft.path("):
            return None  # a load with a computed name
        names.update(n for _, n in literal)
    return names or None


async def _lookahead(obj: dict, code: str, data_dir: str, catalog: list[dict], positions_mode: bool,
                     full_pos: Path | None, returns: list[list], seed: Any = None,
                     concurrency: int = LOOKAHEAD_CONCURRENCY, progress: dict | None = None) -> tuple[str, str]:
    """Run the look-ahead test; (verdict, detail). Verdict: pass | fail | error.

    `progress`, if given, is kept current as {"cuts_done", "cuts_total"} for a progress bar."""
    fixed = cuts(obj)
    active = (await _off(_active_cuts, obj, full_pos, LOOKAHEAD_ACTIVE_CUTS, seed)
              if positions_mode and full_pos is not None else [])
    # The one-off cuts copy only what the candidate reads; the fixed cuts are full and cached.
    used = await _off(_datasets_used, obj, code) if active else None
    plan = [(c, False) for c in fixed] + [(c, True) for c in active if c not in fixed]
    rows: list[dict] = []
    if progress is not None:
        progress.update(cuts_done=0, cuts_total=len(plan))
        rows = [{"label": f"cut at {c}", "kind": "after a trade" if t else "fixed", "state": "queued"}
                for c, t in plan]
        progress.setdefault("runs", []).extend(rows)
    sem = asyncio.Semaphore(concurrency)
    failed = asyncio.Event()

    async def one(i: int, cut: str, temporary: bool) -> tuple[str, str] | None:
        row = rows[i] if rows else {}
        async with sem:
            if failed.is_set():
                row["state"] = "skipped"  # one proven leak is enough
                return None
            row.update(state="running", started=time.time())
            mirror = None
            v: tuple[str, str] | None = None
            try:
                mirror = await _off(build_mirror, obj, data_dir, cut, used if temporary else None)
                if not mirror["items"]:
                    return "error", "no time-indexed dataset could be truncated"
                trunc = await _run(code, data_dir, catalog, mirror, obj["eval_timeout_s"], obj, cut)
                if not trunc["ok"]:
                    return "error", f"the script failed on data cut at {cut}: " + trunc["stderr"][-600:]
                if positions_mode:
                    t_pos = _positions_file(trunc)
                    v = (await _off(_positions_lookahead, full_pos, t_pos, cut)) if t_pos \
                        else ("error", f"no positions reported on data cut at {cut}")
                else:
                    t_returns, t_problem = _clean_returns(trunc["result"].get("returns"))
                    v = ("error", t_problem) if t_problem else _lookahead_verdict(returns, t_returns, cut[:10])
                if v[0] == "fail":
                    failed.set()
                return v
            finally:
                if row:
                    row.update(state=v[0] if v else "error", seconds=round(time.time() - row["started"], 1))
                if progress is not None:
                    progress["cuts_done"] = progress.get("cuts_done", 0) + 1
                if temporary:
                    await _off(_discard_cut, obj, cut, mirror["root"] if mirror else None)

    verdicts = [v for v in await asyncio.gather(*(one(i, c, t) for i, (c, t) in enumerate(plan))) if v]
    if not verdicts:
        return "error", "no look-ahead cut could be run"
    worst = "fail" if any(v == "fail" for v, _ in verdicts) else \
        "error" if any(v == "error" for v, _ in verdicts) else "pass"
    if worst == "pass":
        detail = (f"{len(verdicts)} cuts ({len(active)} placed just after the candidate's own trades, "
                  f"{len(verdicts) - len(active)} fixed): every earlier position identical with later data removed")
        if not positions_mode:
            detail += " | returns-level check only: configure a price column for the stronger positions test"
    else:
        detail = " | ".join(d for v, d in verdicts if v == worst)
    return worst, detail


def _clean_returns(raw: Any) -> tuple[list[list], str | None]:
    if not isinstance(raw, list) or not raw:
        return [], "no returns reported -- call ft.report_returns(series)"
    seen: dict[str, float] = {}
    for row in raw[:200_000]:
        try:
            d, r = str(row[0])[:10], float(row[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(r) and r > -1.0:
            seen[d] = r
    if len(seen) < 10:
        return [], f"only {len(seen)} usable daily returns"
    return [[d, seen[d]] for d in sorted(seen)], None


def _lookahead_verdict(full: list[list], trunc: list[list], split: str) -> tuple[str, str]:
    """Compare the returns before the split. The last pre-split day is excluded: a return
    dated on the final bar may legitimately need the next bar's price to be realised."""
    a = {d: r for d, r in full if d < split}
    b = {d: r for d, r in trunc if d < split}
    days = sorted(a)
    if len(days) < 5:
        return "error", "too few pre-split days to compare"
    check = days[:-1]
    bad = [d for d in check if d not in b or abs(a[d] - b[d]) > 1e-9 + 1e-6 * abs(a[d])]
    if not bad:
        return "pass", f"returns on {len(check)} pre-split days identical with post-split data removed"
    d = bad[0]
    other = f"{b[d]:.6g}" if d in b else "missing"
    return "fail", (f"{len(bad)} of {len(check)} pre-split daily returns changed when data after "
                    f"{split} was removed (first: {d}: {a[d]:.6g} with future data vs {other} without) "
                    f"-- the strategy uses information from the future")


class Submit(BaseModel):
    code: str = Field("", max_length=400_000)
    answer: str = Field("", max_length=200_000)
    rationale: str = Field("", max_length=8_000)
    parent_id: str | None = None
    model: str = Field("", max_length=200)
    mode: str = Field("improve", max_length=20)
    # The mentor idea this candidate tests (ideas.id), if any.
    idea_id: int | None = None


def _change_kind(parent_code: str | None, code: str) -> str | None:
    """How a candidate differs from its parent: "identical", "parameters only" (the same
    program with different numbers -- thresholds, windows, multipliers) or "logic".

    Parameter tweaks are how a search turns into brute force: they overfit the in-sample
    period and teach the team nothing. Compared on the syntax tree with every number blanked,
    so comments, formatting and renumbered thresholds do not count as a new idea."""
    import ast

    if not parent_code or not code:
        return None
    if parent_code.strip() == code.strip():
        return "identical"

    def shape(src: str) -> str | None:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return None
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                node.value = 0
        return ast.dump(tree, annotate_fields=False, include_attributes=False)

    a, b = shape(parent_code), shape(code)
    if a is None or b is None:
        return None
    return "parameters only" if a == b else "logic"


async def evaluate(obj: dict, req: Submit) -> dict:
    project = projects.get(obj["project_id"])
    if project is None:
        raise HTTPException(status_code=404, detail="the objective's project no longer exists")
    kind = obj["metric"]["kind"]
    if kind != "judge" and not req.code.strip():
        raise HTTPException(status_code=400, detail="submit a complete Python script in `code`")

    cid = uuid.uuid4().hex[:10]
    now = time.time()
    with _lock:
        seq = max(db().execute("SELECT COALESCE(MAX(seq),0) FROM candidates WHERE objective_id=?",
                               (obj["id"],)).fetchone()[0] or 0,
                  db().execute("SELECT COALESCE(MAX(seq),0) FROM seq_hwm WHERE objective_id=?",
                               (obj["id"],)).fetchone()[0] or 0) + 1
        db().execute("INSERT INTO seq_hwm (objective_id, seq) VALUES (?,?) "
                     "ON CONFLICT(objective_id) DO UPDATE SET seq=excluded.seq", (obj["id"], seq))
        db().execute(
            "INSERT INTO candidates (id, objective_id, seq, created_at, model, mode, parent_id, rationale, "
            "code, answer, status, idea_id) VALUES (?,?,?,?,?,?,?,?,?,?, 'evaluating', ?)",
            (cid, obj["id"], seq, now, req.model, req.mode, req.parent_id, req.rationale, req.code, req.answer,
             req.idea_id),
        )
        db().commit()

    data_dir = project["data_dir"]
    catalog = await asyncio.to_thread(datasource.catalog, data_dir)
    from .library import record_usage

    # Which library versions this candidate runs on -- the evidence behind each module.
    record_usage(obj["project_id"], cid, req.code)
    fields: dict[str, Any] = {}
    t0 = time.time()
    try:
        async with _EVAL_SLOTS:
            if req.code.strip():
                full = await _run_forecasting(req.code, data_dir, catalog, None, obj["eval_timeout_s"], obj, None,
                                              requested_by=f"candidate #{seq}")
                fields.update(stdout=full["stdout"][-20_000:], stderr=full["stderr"][-20_000:], run_id=full["run_id"])
                res = full["result"]
                if not full["ok"]:
                    fields.update(status="error", score_note=_failure_note(full["stderr"]))
                elif kind in RETURN_METRICS:
                    positions_mode = bool(obj["metric"].get("price_column") and obj.get("dataset")
                                          and obj.get("time_column"))
                    full_pos = _positions_file(full)
                    returns, problem, mtm = [], None, None
                    if positions_mode:
                        if full_pos is None:
                            problem = "no positions reported -- call ft.report_positions(series)"
                        else:
                            returns, mtm = await asyncio.to_thread(_mark_to_market, obj, data_dir, full_pos)
                            problem = _positions_off_data(mtm)
                            if problem is None and len(returns) < 10:
                                problem = f"positions produced only {len(returns)} days of returns"
                    else:
                        returns, problem = _clean_returns(res.get("returns"))
                    if problem:
                        fields.update(status="error", score_note=problem)
                    else:
                        score, is_score, note, metrics = _score_returns(obj, returns)
                        if res.get("extra"):
                            metrics["extra"] = res["extra"]
                        if full.get("features_used"):
                            metrics["features_used"] = full["features_used"]
                        if req.parent_id:
                            try:
                                parent = get_candidate(req.parent_id)
                                kind_of_change = _change_kind(parent.get("code"), req.code)
                                if kind_of_change:
                                    metrics["change"] = {"kind": kind_of_change, "parent_seq": parent["seq"]}
                            except HTTPException:
                                pass
                        if mtm:
                            gross, inverted = mtm.pop("_gross", []), mtm.pop("_inverted", [])
                            metrics["execution"] = mtm
                            try:
                                metrics["costs"] = _costs(obj, returns, gross, inverted,
                                                          int(mtm.get("position_changes") or 0))
                            except Exception:  # noqa: BLE001 -- a diagnostic must never cost the score
                                logger.exception("cost breakdown failed for %s", cid)
                        metrics["source"] = "positions (marked to market by the harness)" if positions_mode \
                            else "self-reported returns"
                        fields.update(status="ok", score=score, is_score=is_score, score_note=note,
                                      metrics=json.dumps(metrics), returns=json.dumps(returns))
                        if obj["lookahead_check"] and cuts(obj):
                            # A crash inside the look-ahead test is the harness's problem, not proof
                            # of a leak: record it as an `error` verdict (which keeps the candidate off
                            # the title) instead of discarding the score and output already earned.
                            try:
                                worst, detail = await _lookahead(obj, req.code, data_dir, catalog, positions_mode,
                                                                 full_pos, returns)
                            except HTTPException as exc:
                                worst, detail = "error", f"the look-ahead test could not run: {exc.detail}"
                            except Exception as exc:  # noqa: BLE001
                                logger.exception("look-ahead test crashed for %s", cid)
                                worst, detail = "error", f"the look-ahead test could not run: {type(exc).__name__}: {exc}"
                            fields.update(lookahead=worst, lookahead_detail=detail[:2000])
                elif kind == "reported":
                    v = res.get("score")
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        fields.update(status="ok", score=float(v), is_score=float(v),
                                      metrics=json.dumps({"extra": res.get("extra") or {}}))
                    else:
                        fields.update(status="error", score_note="no score reported -- call ft.report_score(x)")
                else:  # judge: the run's output is what gets judged
                    fields.update(status="ok", score_note="awaiting judge")
            else:
                fields.update(status="ok", score_note="awaiting judge")
    except Exception as exc:
        # Keep what the run already produced. submit() only rewrites score_note on the way
        # out, so without this the agent got "evaluation failed: <harness internals>" and none
        # of its own stdout/stderr -- nothing to learn from (and an HTTPException, e.g. prices
        # that cannot be marked, left the candidate stuck at 'evaluating').
        why = exc.detail if isinstance(exc, HTTPException) else f"evaluation failed: {exc}"
        try:
            _update_candidate(cid, {**{k: fields[k] for k in ("stdout", "stderr", "run_id") if k in fields},
                                    "status": "error", "score_note": str(why)[:500],
                                    "eval_seconds": round(time.time() - t0, 1)})
        except Exception:  # noqa: BLE001 -- never mask the original failure
            logger.exception("could not record the failed evaluation of %s", cid)
        raise
    fields["eval_seconds"] = round(time.time() - t0, 1)

    # Contender for the title? Then it needs an audit first (if the objective asks for one).
    higher = _higher(obj)
    best = _best(obj)
    contender = (fields.get("status") == "ok" and fields.get("score") is not None
                 and (fields.get("lookahead") == "pass" if obj["lookahead_check"] and cuts(obj)
                      else fields.get("lookahead") != "fail")
                 and _better(fields["score"], best["score"] if best else None, higher))
    if contender:
        if obj["require_audit"]:
            fields["audit"] = "pending"
        else:
            fields["champion_at"] = time.time()
    _update_candidate(cid, fields)
    # A proven leak retires the modules it came from, not just the candidate. The harness
    # catches the same defect again on the next submission otherwise, because the module
    # carrying it is still `active` and every agent keeps importing it.
    if fields.get("lookahead") == "fail":
        _auto_quarantine(obj, cid, seq, req.code,
                         f"Look-ahead detected by the harness on #{seq}: "
                         f"{(fields.get('lookahead_detail') or '')[:1500]}")
    if contender and not obj["require_audit"]:
        _crown(obj["id"], cid)
        _auto_review(obj["id"], cid)
    return agent_view(obj, get_candidate(cid))


def _update_candidate(cid: str, fields: dict) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    with _lock:
        db().execute(f"UPDATE candidates SET {cols} WHERE id=?", (*fields.values(), cid))
        db().commit()


def _crown(oid: str, cid: str) -> None:
    with _lock:
        db().execute("UPDATE candidates SET champion_at=? WHERE id=?", (time.time(), cid))
        db().execute("UPDATE objectives SET best_id=?, updated_at=? WHERE id=?", (cid, time.time(), oid))
        db().commit()


def _auto_review(oid: str, cid: str) -> None:
    """Fire an external review of a candidate that just took the title, if the operator asked
    for that. Detached and best-effort: it costs money and takes a minute, so it must never
    delay or fail the crowning that triggered it. Deliberately NOT called when a demotion
    re-crowns a runner-up -- a review that demotes would crown the next one and review again.
    """
    try:
        from . import review as R

        if not (R.available() and R.config()["auto_review_champions"]):
            return
        task = asyncio.create_task(R.review_candidate(oid, cid, R.ReviewReq()))
        _bg.add(task)                      # a bare task can be collected mid-flight

        def _finished(t: asyncio.Task) -> None:
            _bg.discard(t)
            if not t.cancelled() and t.exception() is not None:
                # Retrieve it, or it surfaces later as an unhandled task exception with no
                # indication of which candidate it belonged to.
                logger.warning("automatic review of %s failed: %s", cid, t.exception())

        task.add_done_callback(_finished)
    except RuntimeError:
        pass                               # no running loop (a sync caller); skip rather than crash
    except Exception:  # noqa: BLE001
        logger.exception("could not start the automatic review")


_bg: set[asyncio.Task] = set()


def _best(obj: dict) -> dict | None:
    """The current champion (audited, if the objective requires it)."""
    if not obj.get("best_id"):
        return None
    try:
        return get_candidate(obj["best_id"], light=True)
    except HTTPException:
        return None


def get_candidate(cid: str, light: bool = False) -> dict:
    cols = _LIGHT if light else "*"
    with _lock:
        r = db().execute(f"SELECT {cols} FROM candidates WHERE id=?", (cid,)).fetchone()
    if r is None:
        raise HTTPException(status_code=404, detail=f"no candidate {cid!r}")
    return _cand_row(r)


def agent_view(obj: dict, c: dict) -> dict:
    """What the submitting agent is told. Holdout numbers are withheld on purpose."""
    ranked = _ranked(obj["id"], _higher(obj))
    rank = next((i + 1 for i, x in enumerate(ranked) if x["id"] == c["id"]), None)
    view: dict[str, Any] = {
        "candidate_id": c["id"], "seq": c["seq"], "status": c["status"],
        "eval_seconds": c.get("eval_seconds"),
    }
    if c["status"] == "error":
        view["error"] = c.get("score_note")
        view["stderr_tail"] = (c.get("stderr") or "")[-2500:]
        view["stdout_tail"] = (c.get("stdout") or "")[-800:]
        return view
    m = c.get("metrics") or {}
    if "in_sample" in m:
        view["in_sample"] = m["in_sample"]
        view["in_sample_score"] = c.get("is_score")
    if m.get("warning"):
        view["warning"] = m["warning"]
    costs = m.get("costs") or {}
    if costs.get("in_sample"):
        # In-sample only, like everything else the agent is shown.
        view["costs_in_sample"] = {"after_costs": costs["in_sample"]["net"], "before_costs": costs["in_sample"]["gross"],
                                   "flipped": costs["in_sample"]["inverted"],
                                   "position_changes_per_day": costs.get("changes_per_day")}
        if costs.get("verdict"):
            view["diagnosis"] = costs["verdict"]
    if m.get("extra"):
        view["extra"] = m["extra"]
    change = m.get("change") or {}
    if change.get("kind") in ("parameters only", "identical"):
        view["change"] = (f"This is #{change.get('parent_seq')} with only its numbers changed (thresholds, windows, "
                          "multipliers). Tuning numbers overfits the in-sample period and teaches the team nothing; "
                          "next time change the LOGIC -- a new signal, filter, regime condition or exit rule -- and "
                          "say why it should work.")
    view["lookahead"] = c.get("lookahead")
    if c.get("lookahead") in ("fail", "error"):
        view["lookahead_detail"] = c.get("lookahead_detail")
    if c.get("score") is None and c.get("score_note"):
        view["not_ranked"] = c["score_note"]
    view["rank"] = f"{rank} of {len(ranked)}" if rank else "unranked"
    view["contender_for_best"] = c.get("audit") == "pending" or bool(c.get("champion_at"))
    if obj.get("split_date"):
        view["note"] = ("Ranking uses the hidden holdout period (after the split); in-sample "
                        "numbers are shown so you can debug, not to be maximised.")
    view["stdout_tail"] = (c.get("stdout") or "")[-800:]
    return view


# =======================================================================================
# Routes
# =======================================================================================
class MetricSpec(BaseModel):
    kind: MetricKind = "sharpe"
    higher_is_better: bool = True
    periods_per_year: float = Field(252, gt=0, le=100_000)
    min_active_days: int = Field(20, ge=0, le=100_000)
    rubric: str = Field("", max_length=8_000)
    # Positions mode: the harness marks positions to market with this price column of the
    # objective's dataset. Without it, candidates report returns themselves (weaker).
    price_column: str | None = Field(None, max_length=200)
    cost_bps: float = Field(1.0, ge=0, le=1000)
    max_leverage: float = Field(1.0, gt=0, le=100)


class CreateObjective(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    description: str = Field("", max_length=50_000)
    metric: MetricSpec = Field(default_factory=MetricSpec)
    dataset: str | None = None
    time_column: str | None = None
    split_date: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    lookahead_check: bool = True
    require_audit: bool = True
    eval_timeout_s: int = Field(DEFAULT_EVAL_TIMEOUT_S, ge=30, le=600)
    cooldown_s: int = Field(0, ge=0, le=3600)


def _summary(obj: dict) -> dict:
    with _lock:
        counts = dict(db().execute(
            "SELECT status, count(*) FROM candidates WHERE objective_id=? GROUP BY status", (obj["id"],)
        ).fetchall())
        champs = db().execute(
            "SELECT count(*), max(champion_at) FROM candidates WHERE objective_id=? AND champion_at IS NOT NULL",
            (obj["id"],)).fetchone()
        last = db().execute("SELECT max(created_at) FROM candidates WHERE objective_id=?", (obj["id"],)).fetchone()[0]
    best = _best(obj)
    kind = obj["metric"]["kind"]
    return {
        **obj,
        "metric_label": METRIC_LABEL.get(kind, kind),
        "candidates": sum(counts.values()),
        "candidates_ok": counts.get("ok", 0),
        "candidates_error": counts.get("error", 0),
        "evaluating": counts.get("evaluating", 0),
        "improvements": champs[0] or 0,
        "last_improvement_at": champs[1],
        "last_candidate_at": last,
        "best": best,
    }


@router.get("/projects/{project_id}/objectives")
async def list_objectives(project_id: str, status: str | None = None) -> dict:
    with _lock:
        rows = db().execute("SELECT * FROM objectives WHERE project_id=? ORDER BY created_at DESC",
                            (project_id,)).fetchall()
    objs = [_obj_row(r) for r in rows]
    if status:
        objs = [o for o in objs if o["status"] == status]
    return {"objectives": [_summary(o) for o in objs]}


@router.get("/projects/{project_id}/objectives/probe")
async def probe(project_id: str, dataset: str, holdout: float = 0.3, time_column: str | None = None) -> dict:
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="no such project")
    if not 0.05 <= holdout <= 0.7:
        raise HTTPException(status_code=400, detail="holdout must be between 5% and 70%")
    return await asyncio.to_thread(probe_dataset, project["data_dir"], dataset, holdout, time_column)


@router.post("/projects/{project_id}/objectives")
async def create_objective(project_id: str, req: CreateObjective) -> dict:
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="no such project")
    metric = req.metric.model_dump()
    if metric["kind"] in RETURN_METRICS or metric["kind"] == "judge":
        metric["higher_is_better"] = True  # drawdown is negative: closer to zero is higher
    split = req.split_date
    tc = req.time_column
    if req.dataset and metric["kind"] in RETURN_METRICS:
        info = await asyncio.to_thread(probe_dataset, project["data_dir"], req.dataset, 0.3, tc)
        tc = tc or info.get("time_column")
        split = split or info.get("split_date")
        # None = not specified: use the probe's guess. "" = the operator chose "no price column".
        if metric.get("price_column") is None:
            metric["price_column"] = info.get("price_column")
        metric["price_column"] = metric["price_column"] or None
        if split and tc:
            # The mid in-sample cut must sit before the (possibly operator-chosen) split.
            if info.get("mid_cut") and info["mid_cut"] < split:
                metric["mid_cut"] = info["mid_cut"]
    oid = uuid.uuid4().hex[:10]
    now = time.time()
    with _lock:
        db().execute(
            "INSERT INTO objectives (id, project_id, title, description, metric, status, dataset, time_column, "
            "split_date, lookahead_check, require_audit, eval_timeout_s, cooldown_s, created_at, updated_at) "
            "VALUES (?,?,?,?,?, 'running', ?,?,?,?,?,?,?,?,?)",
            (oid, project_id, req.title, req.description, json.dumps(metric), req.dataset, tc, split,
             int(req.lookahead_check and bool(split)), int(req.require_audit), req.eval_timeout_s,
             req.cooldown_s, now, now),
        )
        db().commit()
    return _summary(get_objective(oid))


class PatchObjective(BaseModel):
    status: Literal["running", "paused", "stopped"] | None = None
    title: str | None = Field(None, max_length=300)
    description: str | None = Field(None, max_length=50_000)
    cooldown_s: int | None = Field(None, ge=0, le=3600)


@router.get("/objectives/{oid}")
async def objective(oid: str) -> dict:
    obj = get_objective(oid)
    with _lock:
        notes = [dict(r) for r in db().execute(
            "SELECT * FROM notes WHERE objective_id=? ORDER BY ts DESC LIMIT 50", (oid,)).fetchall()]
        lessons = [dict(r) for r in db().execute(
            "SELECT * FROM lessons WHERE objective_id=? AND active=1 ORDER BY ts DESC LIMIT 100", (oid,)).fetchall()]
        champions = [dict(r) for r in db().execute(
            "SELECT id, seq, score, is_score, model, champion_at FROM candidates WHERE objective_id=? "
            "AND champion_at IS NOT NULL ORDER BY champion_at", (oid,)).fetchall()]
        points = [dict(r) for r in db().execute(
            "SELECT id, seq, created_at, status, score, lookahead, audit, model, champion_at FROM candidates "
            "WHERE objective_id=? ORDER BY seq", (oid,)).fetchall()]
    return {**_summary(obj), "notes": notes, "lessons": lessons, "champions": champions, "points": points}


@router.patch("/objectives/{oid}")
async def patch_objective(oid: str, req: PatchObjective) -> dict:
    get_objective(oid)
    fields = {k: v for k, v in req.model_dump().items() if v is not None}
    if fields:
        fields["updated_at"] = time.time()
        with _lock:
            db().execute(f"UPDATE objectives SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?",
                         (*fields.values(), oid))
            db().commit()
    return _summary(get_objective(oid))


@router.delete("/objectives/{oid}")
async def delete_objective(oid: str) -> dict:
    get_objective(oid)
    with _lock:
        for table in ("candidates", "notes", "lessons"):
            db().execute(f"DELETE FROM {table} WHERE objective_id=?", (oid,))
        db().execute("DELETE FROM objectives WHERE id=?", (oid,))
        db().commit()
    import shutil
    shutil.rmtree(WORK_ROOT / oid, ignore_errors=True)
    return {"ok": True}


class Note(BaseModel):
    text: str = Field(..., min_length=1, max_length=20_000)
    author: str = Field("operator", max_length=120)


@router.post("/objectives/{oid}/notes")
async def add_note(oid: str, req: Note) -> dict:
    get_objective(oid)
    with _lock:
        cur = db().execute("INSERT INTO notes (objective_id, ts, author, text) VALUES (?,?,?,?)",
                           (oid, time.time(), req.author, req.text))
        db().commit()
    return {"id": cur.lastrowid}


@router.get("/objectives/{oid}/candidates")
async def list_candidates(oid: str, order: Literal["rank", "recent"] = "rank", limit: int = 50) -> dict:
    obj = get_objective(oid)
    limit = max(1, min(limit, 500))
    if order == "rank":
        return {"candidates": _ranked(oid, _higher(obj), limit),
                "disqualified": _disqualified(oid, _higher(obj))}
    with _lock:
        rows = db().execute(f"SELECT {_LIGHT} FROM candidates WHERE objective_id=? ORDER BY seq DESC LIMIT ?",
                            (oid, limit)).fetchall()
    return {"candidates": [_cand_row(r) for r in rows]}


@router.get("/objectives/{oid}/candidates/{cid}")
async def candidate(oid: str, cid: str) -> dict:
    c = get_candidate(cid)
    if c["objective_id"] != oid:
        raise HTTPException(status_code=404, detail="candidate belongs to another objective")
    return c


@router.post("/objectives/{oid}/candidates/{cid}/run")
async def run_candidate(oid: str, cid: str) -> dict:
    """Re-run a stored candidate exactly as the swarm scored it -- ``import ft``, the project
    data, forecast features and code library, full data -- and return its output. For the
    operator reading a candidate: the chat sandbox has none of that, so the same script fails
    there with ``No module named 'ft'``. Nothing is scored or recorded."""
    c = get_candidate(cid)
    if c["objective_id"] != oid:
        raise HTTPException(status_code=404, detail="candidate belongs to another objective")
    if not c.get("code"):
        raise HTTPException(status_code=400, detail="this candidate has no code")
    obj = get_objective(oid)
    project = projects.get(obj["project_id"])
    if project is None:
        raise HTTPException(status_code=404, detail="no such project")
    catalog = await asyncio.to_thread(datasource.catalog, project["data_dir"])
    async with _EVAL_SLOTS:
        rep = await _run_forecasting(c["code"], project["data_dir"], catalog, None, obj["eval_timeout_s"], obj, None,
                                     requested_by=f"operator re-run of #{c['seq']}")
    return {"ok": rep["ok"], "stdout": rep["stdout"][-12_000:], "stderr": rep["stderr"][-6_000:],
            "duration_s": rep["duration_s"]}


@router.post("/objectives/{oid}/candidates")
async def submit(oid: str, req: Submit) -> dict:
    obj = get_objective(oid)
    if obj["status"] != "running":
        raise HTTPException(status_code=409, detail=f"objective is {obj['status']}")
    try:
        return await evaluate(obj, req)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 -- record it; the agent needs to hear why
        with _lock:
            db().execute("UPDATE candidates SET status='error', score_note=? WHERE objective_id=? "
                         "AND status='evaluating' AND model=? AND created_at > ?",
                         (f"evaluation failed: {exc}"[:500], oid, req.model, time.time() - 3600))
            db().commit()
        raise HTTPException(status_code=500, detail=f"evaluation failed: {exc}") from None


class Audit(BaseModel):
    passed: bool
    notes: str = Field("", max_length=20_000)
    model: str = Field("", max_length=200)


@router.post("/objectives/{oid}/candidates/{cid}/audit")
async def audit(oid: str, cid: str, req: Audit) -> dict:
    obj = get_objective(oid)
    c = get_candidate(cid, light=True)
    if c["objective_id"] != oid:
        raise HTTPException(status_code=404, detail="candidate belongs to another objective")
    note = f"[{req.model or 'auditor'}] {req.notes}".strip()
    crowned = False
    with _lock:
        if req.passed:
            # Re-check against the CURRENT champion: another candidate may have been crowned
            # while this one was being audited.
            best = _best(get_objective(oid))
            crowned = _better(c["score"], best["score"] if best else None, _higher(obj))
        _update_candidate(cid, {"audit": "pass" if req.passed else "fail", "audit_notes": note})
        if crowned:
            _crown(oid, cid)
    if crowned:
        _auto_review(oid, cid)
    quarantined: list[str] = []
    if not req.passed:
        # A failed audit disqualifies the result the same way a proven leak does, so the
        # modules it was built on go with it.
        quarantined = _auto_quarantine(obj, cid, c.get("seq"), get_candidate(cid).get("code") or "",
                                       f"Audit failed on #{c.get('seq')}: {note[:1500]}")
    return {"audit": "pass" if req.passed else "fail", "champion": crowned, "quarantined": quarantined}


def _descendants(oid: str, cid: str) -> list[dict]:
    """Candidates built on this one, transitively. A leak usually lives in a pattern that was
    copied forward, so demoting one candidate raises the question of its whole line."""
    with _lock:
        rows = db().execute(
            f"SELECT {_LIGHT} FROM candidates WHERE objective_id=? AND parent_id IS NOT NULL", (oid,)).fetchall()
    kids: dict[str, list[dict]] = {}
    for r in rows:
        c = _cand_row(r)
        kids.setdefault(c["parent_id"], []).append(c)
    out, seen, queue = [], {cid}, list(kids.get(cid, []))
    while queue:
        c = queue.pop(0)
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        out.append(c)
        queue.extend(kids.get(c["id"], []))
    return sorted(out, key=lambda c: c["seq"])


# One writer thread for board posts, so they keep their order and never run on the event loop.
_BOARD_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="board-post")


def _board_post(project_id: str, channel: str, author: str, content: str, meta: dict) -> None:
    """Put a finding on the swarm's message board, so the team sees it in the run it happens in.

    Queued, not written inline: most callers are request handlers on the event loop, and a
    board write can wait out SQLite's 30 s busy timeout (five times, with the retries below)
    while another process holds the lock -- which froze the whole console, every poll
    included, for as long as it took. The caller gets on with its response; the post follows.
    """
    _BOARD_POOL.submit(_board_post_now, project_id, channel, author, content, meta)


def _board_post_now(project_id: str, channel: str, author: str, content: str, meta: dict) -> None:
    from . import msgboard

    # The board database is written by three processes (this one, the board service, the
    # swarm runner), so a write can lose the lock to one of them. Retry before giving up:
    # dropping the post is not cosmetic -- a demotion the agents never read is a demotion
    # that does not stop them rebuilding the thing it disqualified.
    for attempt in range(5):
        try:
            with msgboard._db_lock:  # noqa: SLF001 -- same process, same connection discipline
                conn = msgboard.db()
                conn.execute(
                    "INSERT INTO messages (channel,author,author_id,kind,content,meta,reply_to,ts,session_id,project_id) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (channel, author, "operator", "result", content, json.dumps(meta), None, time.time(),
                     msgboard.current_session_id(conn, project_id), project_id))
                conn.commit()
            return
        except sqlite3.OperationalError as exc:
            try:
                msgboard.db().rollback()  # never leave the shared connection holding the write lock
            except sqlite3.Error:
                pass
            if "locked" not in str(exc) and "busy" not in str(exc):
                logger.exception("could not post to the message board")
                return
            time.sleep(0.25 * (attempt + 1))
        except Exception:  # noqa: BLE001 -- the caller must not fail because the board did
            logger.exception("could not post to the message board")
            return
    logger.error("gave up posting to the message board after 5 attempts (database locked): %.80s", content)


class Demote(BaseModel):
    """An operator (or an external reviewer) disqualifying a result the automated checks passed."""
    finding: str = Field(..., min_length=1, max_length=40_000)   # the full critique, pasted
    lesson: str = Field("", max_length=2000)                     # one line for TEAM LESSONS
    reviewer: str = Field("operator", max_length=200)
    to_playbook: bool = True
    # Quarantine the library modules the result was built on. Off only when the operator
    # knows the modules are sound and the candidate's own script was at fault.
    quarantine_modules: bool = True


@router.post("/objectives/{oid}/candidates/{cid}/demote")
async def demote(oid: str, cid: str, req: Demote) -> dict:
    """Disqualify a candidate by hand and teach the team why.

    The automated look-ahead check compares positions with and without future rows, which
    catches a candidate that READS the future but not one whose positions are correctly
    computed and then mis-aligned onto earlier bars. That is a human (or external-model)
    finding, so this is the door for it -- and it is worth nothing unless the reason travels
    with it, which is why the finding is required and lands in three places at once.
    """
    obj = get_objective(oid)
    c = get_candidate(cid, light=True)
    if c["objective_id"] != oid:
        raise HTTPException(status_code=404, detail="candidate belongs to another objective")

    lesson = (req.lesson.strip() or req.finding.strip().split("\n")[0])[:2000]
    note = f"[demoted by {req.reviewer}] {req.finding.strip()}"[:20_000]
    now = time.time()
    was_champion = obj.get("best_id") == cid

    with _lock:
        # audit='fail' is what _ranked already filters on, so this drops it off the
        # leaderboard and keeps it from ever being crowned again.
        _update_candidate(cid, {"audit": "fail", "audit_notes": note})
        db().execute("INSERT INTO lessons (objective_id, ts, model, candidate_id, text) VALUES (?,?,?,?,?)",
                     (oid, now, req.reviewer, cid, lesson))
        db().commit()

    # A demoted champion must not keep the crown: promote the best still-eligible candidate.
    recrowned = None
    if was_champion:
        with _lock:
            db().execute("UPDATE objectives SET best_id=NULL, updated_at=? WHERE id=?", (now, oid))
            db().commit()
        nxt = _ranked(oid, _higher(obj), limit=1)
        if nxt:
            _crown(oid, nxt[0]["id"])
            recrowned = nxt[0]

    # The leak is usually in the module, not the script that called it -- and the swarm is
    # already saving v2s of that module. Take the modules down with the candidate, reason
    # attached, or the next agent rebuilds the same defect from the same source.
    quarantined: list[str] = []
    cascaded: list[int] = []
    if req.quarantine_modules:
        try:
            from . import library

            code = get_candidate(cid).get("code") or ""
            used = sorted(library.reachable_modules(obj["project_id"], code))
            why = (f"{lesson}\n\nFrom the demotion of #{c['seq']} on \"{obj['title']}\":\n"
                   f"{req.finding.strip()[:3000]}")
            quarantined = library.quarantine(
                obj["project_id"], used, why, req.reviewer, candidate_id=cid, seq=c["seq"])
            # Everything else built on those modules inherits the same defect. Take the whole
            # family off the board now, or the next crowning promotes a sibling of the result
            # just disqualified and the swarm keeps refining the leak.
            cascaded = _cascade_quarantine(obj, quarantined, lesson, req.reviewer, origin_cid=cid)
        except Exception:  # noqa: BLE001 -- never lose the demotion over the library write
            logger.exception("could not quarantine the library modules")

    if req.to_playbook:
        try:
            from .playbook import add_pitfall

            add_pitfall(obj["project_id"], lesson, req.reviewer, f"demoted #{c['seq']} on '{obj['title']}'")
        except Exception:  # noqa: BLE001 -- never lose the demotion over the playbook write
            logger.exception("could not record the pitfall")

    _board_post(
        obj["project_id"], "results", req.reviewer,
        f"DEMOTED #{c['seq']} on \"{obj['title']}\" -- disqualified by {req.reviewer}, not by the harness.\n\n"
        f"{req.finding.strip()[:4000]}\n\n"
        f"This is now a team lesson and a project pitfall. Do not reproduce this pattern; if you built on "
        f"#{c['seq']}, re-check your own alignment before submitting again."
        + (f"\n\nQUARANTINED library modules -- do not import these, they carry the defect: "
           f"{', '.join(quarantined)}. Fix the cause in a new module instead of saving another "
           f"version of one of these." if quarantined else ""),
        {"objective_id": oid, "candidate_id": cid, "seq": c["seq"], "demoted": True,
         "quarantined": quarantined})

    kids = _descendants(oid, cid)
    return {"demoted": c["seq"], "audit": "fail", "lesson": lesson,
            "was_champion": was_champion, "quarantined": quarantined, "cascaded": cascaded,
            "recrowned": {"id": recrowned["id"], "seq": recrowned["seq"]} if recrowned else None,
            "descendants": [{"id": k["id"], "seq": k["seq"], "model": k["model"], "score": k["score"],
                             "audit": k["audit"], "rationale": k["rationale"][:200]} for k in kids]}


# =======================================================================================
# Re-testing candidates scored under an older, weaker look-ahead test
# =======================================================================================
_retests: dict[str, dict] = {}
_retest_tasks: dict[str, asyncio.Task] = {}


async def _retest(obj: dict, ids: list[str]) -> None:
    st = _retests[obj["id"]]
    project = projects.get(obj["project_id"]) or {}
    data_dir = project.get("data_dir", "")
    catalog = await asyncio.to_thread(datasource.catalog, data_dir)
    positions_mode = bool(obj["metric"].get("price_column") and obj.get("dataset") and obj.get("time_column"))
    for rank, cid in enumerate(ids, 1):
        if st.get("cancel"):
            st["cancelled"] = True
            break  # the candidate in flight has finished; stop before the next one
        c = get_candidate(cid)
        st["current"], st["current_rank"], st["current_started"] = c["seq"], rank, time.time()
        prog = st["progress"][str(c["seq"])] = {"phase": "full run", "cuts_done": 0, "cuts_total": 0,
                                                "runs": [{"label": "full run", "kind": "positions to compare against",
                                                          "state": "running", "started": time.time()}]}
        try:
            async with _RETEST_SLOT:
                full = await _run_forecasting(c["code"], data_dir, catalog, None, obj["eval_timeout_s"], obj, None,
                                              requested_by=f"look-ahead re-test of #{c['seq']}")
                prog["runs"][0].update(state="pass" if full["ok"] else "error",
                                       seconds=round(time.time() - prog["runs"][0]["started"], 1))
                if not full["ok"]:
                    st["results"].append({"seq": c["seq"], "verdict": "error", "detail": "the script no longer runs"})
                    prog["phase"] = "done"
                    continue
                full_pos = _positions_file(full)
                returns = json.loads(c.get("returns") or "[]") if isinstance(c.get("returns"), str) else (c.get("returns") or [])
                prog["phase"] = "cuts"
                verdict, detail = await _lookahead(obj, c["code"], data_dir, catalog, positions_mode, full_pos,
                                                   returns, concurrency=RETEST_CONCURRENCY, progress=prog)
        except asyncio.CancelledError:
            prog["phase"] = "cancelled"
            st.update(cancelled=True, done_at=time.time(), current=None, current_rank=None)
            raise
        except Exception as exc:  # noqa: BLE001 -- one candidate must not stop the re-test
            logger.exception("re-test of %s failed", cid)
            st["results"].append({"seq": c["seq"], "verdict": "error", "detail": str(exc)[:300]})
            continue
        st["results"].append({"seq": c["seq"], "verdict": verdict, "detail": detail[:600]})
        prog["phase"] = "done"
        if verdict == "pass":
            _update_candidate(cid, {"lookahead": "pass", "lookahead_detail": detail[:2000]})
            continue
        if verdict == "error":
            continue  # untestable now (e.g. data changed): leave the old verdict, report it
        # A proven leak: off the leaderboard, crown handed on, modules retired, team told.
        now = time.time()
        fresh = get_objective(obj["id"])
        _update_candidate(cid, {"lookahead": "fail", "lookahead_detail": ("re-test: " + detail)[:2000]})
        if fresh.get("best_id") == cid:
            with _lock:
                db().execute("UPDATE objectives SET best_id=NULL, updated_at=? WHERE id=?", (now, obj["id"]))
                db().commit()
            nxt = _ranked(obj["id"], _higher(obj), limit=1)
            if nxt:
                _crown(obj["id"], nxt[0]["id"])
            st["recrowned"] = nxt[0]["seq"] if nxt else None
        why = f"Look-ahead found on re-test of #{c['seq']}: {detail[:1500]}"
        with _lock:
            db().execute("INSERT INTO lessons (objective_id, ts, model, candidate_id, text) VALUES (?,?,?,?,?)",
                         (obj["id"], now, "harness", cid,
                          f"AVOID the pattern in #{c['seq']}: its positions changed when future data was removed "
                          f"-- decisions used information from after the bar they are dated at."))
            db().commit()
        _auto_quarantine(obj, cid, c["seq"], c["code"], why)
        _board_post(obj["project_id"], "results", "harness",
                    f"LOOK-AHEAD on re-test: #{c['seq']} on \"{obj['title']}\" is disqualified.\n\n{detail[:3000]}\n\n"
                    "The look-ahead test now cuts the data just after each candidate's own trades. Do not build on "
                    f"#{c['seq']}; if you did, re-check that every value your position uses was known at its timestamp.",
                    {"objective_id": obj["id"], "candidate_id": cid, "seq": c["seq"], "lookahead": "fail"})
    st["done_at"] = time.time()
    st["current"] = st["current_rank"] = None


class Retest(BaseModel):
    top: int = Field(25, ge=1, le=500)


@router.post("/objectives/{oid}/lookahead/retest")
async def start_retest(oid: str, req: Retest) -> dict:
    """Re-run the look-ahead test on the current leaderboard's top candidates with the dense
    cuts. Leaks found are disqualified exactly as the harness would have done at submission."""
    obj = get_objective(oid)
    st = _retests.get(oid)
    if st and not st.get("done_at"):
        raise HTTPException(status_code=409, detail="a re-test is already running for this objective")
    ranked = _ranked(oid, _higher(obj), limit=req.top)
    ids = [c["id"] for c in ranked]
    _retests[oid] = {"started_at": time.time(), "total": len(ids), "results": [], "current": None,
                     "queue": [c["seq"] for c in ranked], "progress": {},
                     "current_rank": None, "current_started": None, "done_at": None, "recrowned": None,
                     "cancel": False, "cancelled": False}
    _retest_tasks[oid] = asyncio.create_task(_retest(obj, ids))
    return _retests[oid]


@router.post("/objectives/{oid}/lookahead/retest/cancel")
async def cancel_retest(oid: str) -> dict:
    """Stop now. The candidate in flight is abandoned (its sandbox runs are killed); verdicts
    already reached stand."""
    st = _retests.get(oid)
    task = _retest_tasks.get(oid)
    if not st or st.get("done_at") or task is None:
        raise HTTPException(status_code=409, detail="no re-test is running")
    st["cancel"] = True
    task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=15)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass
    st.update(cancelled=True, done_at=st.get("done_at") or time.time(), current=None, current_rank=None)
    return st


@router.get("/objectives/{oid}/lookahead/retest")
async def retest_status(oid: str) -> dict:
    get_objective(oid)
    return _retests.get(oid) or {"total": 0, "results": [], "done_at": None}


class Judged(BaseModel):
    score: float = Field(..., ge=0, le=10)
    notes: str = Field("", max_length=20_000)
    model: str = Field("", max_length=200)


@router.post("/objectives/{oid}/candidates/{cid}/judge")
async def judge(oid: str, cid: str, req: Judged) -> dict:
    obj = get_objective(oid)
    c = get_candidate(cid, light=True)
    if c["objective_id"] != oid or obj["metric"]["kind"] != "judge":
        raise HTTPException(status_code=400, detail="not a judged candidate of this objective")
    fields: dict[str, Any] = {"score": req.score, "is_score": req.score,
                              "score_note": f"[{req.model or 'judge'}] {req.notes}"[:4000]}
    best = _best(obj)
    if _better(req.score, best["score"] if best else None, True):
        if obj["require_audit"]:
            fields["audit"] = "pending"
        else:
            fields["champion_at"] = time.time()
    _update_candidate(cid, fields)
    if fields.get("champion_at"):
        _crown(oid, cid)
        _auto_review(oid, cid)
    return agent_view(obj, get_candidate(cid))


class Lesson(BaseModel):
    text: str = Field(..., min_length=3, max_length=2_000)
    model: str = Field("", max_length=200)
    candidate_id: str | None = None


@router.post("/objectives/{oid}/lessons")
async def add_lesson(oid: str, req: Lesson) -> dict:
    get_objective(oid)
    with _lock:
        db().execute("INSERT INTO lessons (objective_id, ts, model, candidate_id, text) VALUES (?,?,?,?,?)",
                     (oid, time.time(), req.model, req.candidate_id, req.text.strip()))
        db().commit()
    return {"ok": True}


class ReplaceLessons(BaseModel):
    lessons: list[str] = Field(..., min_length=1, max_length=40)
    model: str = Field("", max_length=200)


@router.post("/objectives/{oid}/lessons/replace")
async def replace_lessons(oid: str, req: ReplaceLessons) -> dict:
    """Consolidation: the agent's condensed list supersedes the active lessons."""
    get_objective(oid)
    now = time.time()
    with _lock:
        db().execute("UPDATE lessons SET active=0 WHERE objective_id=? AND active=1", (oid,))
        for text in req.lessons:
            if text.strip():
                db().execute("INSERT INTO lessons (objective_id, ts, model, text) VALUES (?,?,?,?)",
                             (oid, now, req.model or "consolidated", text.strip()[:2000]))
        db().execute("UPDATE objectives SET consolidating_until=0 WHERE id=?", (oid,))
        db().commit()
    return {"ok": True, "lessons": len(req.lessons)}


# =======================================================================================
# Operator clean-up: deleting polluted records outright
# =======================================================================================
# Demoting disqualifies a result but keeps it (struck through, reason attached) so the team
# learns from it. Some records teach nothing: twenty-seven clones of one weak strategy, a
# steering note that no longer applies, a lesson distilled from a leak. Those crowd what the
# agents are handed every iteration -- the top ranks become the parents they improve -- so
# the operator can remove them for good. Hard deletes: nothing here can be undone.

# The same filters `_ranked` and `_disqualified` use, as SQL, so "delete all ranked" removes
# exactly what the leaderboard shows. Keep them in step with those two functions.
_DELETE_SCOPES = {
    "ranked": ("status='ok' AND score IS NOT NULL AND lookahead NOT IN ('fail', 'error') "
               "AND audit NOT IN ('fail')"),
    "disqualified": "status='ok' AND score IS NOT NULL AND (audit='fail' OR lookahead='fail')",
    "all": "1=1",
}


class DeleteCandidates(BaseModel):
    """Either explicit ids (the operator's selection) or a whole scope ("delete all ...")."""
    ids: list[str] = Field(default_factory=list, max_length=10_000)
    scope: Literal["ranked", "disqualified", "all"] | None = None


class DeleteRows(BaseModel):
    ids: list[int] = Field(default_factory=list, max_length=10_000)
    all: bool = False


def _chunks(xs: list, n: int = 500):
    """SQLite caps the number of bound parameters per statement; a "delete all" can exceed it."""
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def _retest_held(oid: str) -> set[int]:
    """Seqs a running leaderboard re-test has yet to finish, the one in flight included.

    `_retest` works from a list of ids fixed when it started and looks each one up as it
    reaches it; a lookup of a deleted id raises outside its per-candidate guard, which ends
    the whole re-test without ever marking it done -- and a re-test that never finishes
    blocks every later one. Those candidates are skipped instead, with the reason.
    """
    st = _retests.get(oid)
    if not st or st.get("done_at"):
        return set()
    return set(st.get("queue") or []) - {r["seq"] for r in st.get("results") or []}


@router.post("/objectives/{oid}/candidates/delete")
async def delete_candidates(oid: str, req: DeleteCandidates) -> dict:
    """Delete candidates for good -- a selection, or every ranked / disqualified / any one.

    Everything that points at a deleted candidate is tidied in the same transaction: its
    library-usage rows (or modules keep counting it as evidence), and the parent link of any
    candidate built on it (the child stands on its own score). If the champion goes, the best
    remaining ranked candidate is crowned -- the same hand-over a demotion does, so it also
    counts as a new best and restarts the stuck clock (escalation.assess).

    Skipped, and reported: candidates still evaluating (the evaluation would write its result
    to a row that no longer exists and could crown it), and candidates a running re-test has
    not reached yet. A pending audit is NOT a reason to skip: the auditor's verdict then
    lands on a missing row and is refused, which is what deleting it should mean.

    The team is told on #results, because agents keep citing seqs they have read on the
    board and in earlier contexts; without the note they go looking for a parent that is gone.
    """
    obj = get_objective(oid)
    if not req.ids and not req.scope:
        raise HTTPException(status_code=400, detail="give the candidate ids to delete, or a scope")
    held = _retest_held(oid)
    skipped: list[dict] = []
    with _lock:
        if req.ids:
            want = list(dict.fromkeys(req.ids))
            rows = []
            for part in _chunks(want):
                rows += db().execute(
                    f"SELECT id, seq, status FROM candidates WHERE objective_id=? AND id IN "
                    f"({','.join('?' * len(part))})", (oid, *part)).fetchall()
            found = {r["id"] for r in rows}
            skipped += [{"id": i, "seq": None, "reason": "not a candidate of this objective"}
                        for i in want if i not in found]
        else:
            rows = db().execute(f"SELECT id, seq, status FROM candidates WHERE objective_id=? AND "
                                f"{_DELETE_SCOPES[req.scope]}", (oid,)).fetchall()
        doomed: list[sqlite3.Row] = []
        for r in rows:
            if r["status"] == "evaluating":
                skipped.append({"id": r["id"], "seq": r["seq"], "reason": "still evaluating"})
            elif r["seq"] in held:
                skipped.append({"id": r["id"], "seq": r["seq"],
                                "reason": "queued in the running look-ahead re-test -- cancel it or let it finish"})
            else:
                doomed.append(r)
        ids = [r["id"] for r in doomed]
        for part in _chunks(ids):
            marks = ",".join("?" * len(part))
            db().execute(f"DELETE FROM lib_usage WHERE candidate_id IN ({marks})", part)
            db().execute(f"UPDATE candidates SET parent_id=NULL WHERE objective_id=? AND parent_id IN ({marks})",
                         (oid, *part))
            db().execute(f"DELETE FROM candidates WHERE objective_id=? AND id IN ({marks})", (oid, *part))
        # Checked by existence, not by membership of this batch, so a best_id already left
        # dangling by some earlier path is repaired too rather than shown as "no best".
        best_id = db().execute("SELECT best_id FROM objectives WHERE id=?", (oid,)).fetchone()[0]
        lost_best = bool(best_id) and db().execute(
            "SELECT 1 FROM candidates WHERE id=?", (best_id,)).fetchone() is None
        if lost_best:
            db().execute("UPDATE objectives SET best_id=NULL, updated_at=? WHERE id=?", (time.time(), oid))
        db().commit()

    recrowned = None
    if lost_best:
        nxt = _ranked(oid, _higher(obj), limit=1)
        if nxt:
            _crown(oid, nxt[0]["id"])
            recrowned = nxt[0]["seq"]
    seqs = sorted(r["seq"] for r in doomed)
    if seqs:
        logger.info("operator deleted %d candidates of %s: %s", len(seqs), oid, seqs[:50])
        shown = ", #".join(str(s) for s in seqs[:300]) + (f" (+{len(seqs) - 300} more)" if len(seqs) > 300 else "")
        _board_post(
            obj["project_id"], "results", "operator",
            f"REMOVED by the operator from \"{obj['title']}\": #{shown}.\n\n"
            f"These candidates no longer exist. Do not build on them, cite them as a parent, or "
            f"re-submit their code -- they were deleted as duplicates or polluted results."
            + (f"\n\nThe best was among them; #{recrowned} now holds the title." if recrowned
               else "\n\nThe best was among them; nothing ranked is left to take the title." if lost_best
               else ""),
            {"objective_id": oid, "deleted": seqs, "recrowned": recrowned})
    return {"deleted": seqs, "skipped": skipped, "recrowned": recrowned, "lost_best": lost_best}


def _delete_rows(table: str, oid: str, req: DeleteRows, all_where: str = "") -> int:
    """Delete by id, or every row of the objective (narrowed by `all_where`). `table` is one
    of the fixed names below, never user input."""
    if not req.ids and not req.all:
        raise HTTPException(status_code=400, detail="give the ids to delete, or all=true")
    n = 0
    with _lock:
        if req.all:
            n = db().execute(f"DELETE FROM {table} WHERE objective_id=?{all_where}", (oid,)).rowcount
        else:
            for part in _chunks(list(dict.fromkeys(req.ids))):
                n += db().execute(f"DELETE FROM {table} WHERE objective_id=? AND id IN "
                                  f"({','.join('?' * len(part))})", (oid, *part)).rowcount
        db().commit()
    return n


@router.post("/objectives/{oid}/lessons/delete")
async def delete_lessons(oid: str, req: DeleteRows) -> dict:
    """Delete team lessons. "All" means the ACTIVE ones -- what agents read and the console
    shows; lessons already superseded by a consolidation are an inert archive and stay."""
    get_objective(oid)
    return {"deleted": _delete_rows("lessons", oid, req, " AND active=1")}


@router.post("/objectives/{oid}/notes/delete")
async def delete_notes(oid: str, req: DeleteRows) -> dict:
    """Delete steering notes. Agents read the newest ten each iteration, so removing a stale
    one also lets an older, still-valid note back into their view."""
    get_objective(oid)
    return {"deleted": _delete_rows("notes", oid, req)}


@router.post("/objectives/{oid}/ideas/delete")
async def delete_ideas(oid: str, req: DeleteRows) -> dict:
    """Delete escalation ideas. The table belongs to escalation.py and only exists once it
    has started, so a missing table means there is nothing to delete.

    The escalation ladder climbs one rung per idea since the last new best; deleting them
    steps it back down, and with none left an objective still stuck is asked again soon.
    """
    get_objective(oid)
    with _lock:
        exists = db().execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='ideas'").fetchone()
    return {"deleted": _delete_rows("ideas", oid, req) if exists else 0}


@router.get("/objectives/{oid}/context")
async def context(oid: str, model: str = "") -> dict:
    """Everything an agent needs for one iteration -- and what it should do in it.

    The search policy lives here, in one place: whether this iteration EXPLORES (a new idea)
    or IMPROVES (mutates a parent), and which parent -- a tournament over the top ranks, so
    good candidates are built on more often without the search collapsing onto one line.
    Pending audits and lesson consolidation are handed out as leases so two agents do not
    both take the same chore.
    """
    obj = get_objective(oid)
    higher = _higher(obj)
    ranked = _ranked(oid, higher, 50)
    now = time.time()

    # Chores first. An audit that has been pending 20 minutes was orphaned (runner restart).
    with _lock:
        pending = db().execute(
            f"SELECT {_LIGHT}, code, answer FROM candidates WHERE objective_id=? AND audit='pending' "
            "AND audit_started < ? ORDER BY score " + ("DESC" if higher else "ASC") + " LIMIT 1",
            (oid, now - 1200),
        ).fetchone()
        if pending is not None:
            db().execute("UPDATE candidates SET audit_started=? WHERE id=?", (now, pending["id"]))
            db().commit()
        n_lessons = db().execute("SELECT count(*) FROM lessons WHERE objective_id=? AND active=1",
                                 (oid,)).fetchone()[0]
        consolidate = n_lessons >= LESSONS_CONSOLIDATE_AT and obj["consolidating_until"] < now
        if consolidate:
            db().execute("UPDATE objectives SET consolidating_until=? WHERE id=?", (now + 900, oid))
            db().commit()
        lessons = [r[0] for r in db().execute(
            "SELECT text FROM lessons WHERE objective_id=? AND active=1 ORDER BY ts DESC LIMIT ?",
            (oid, 60 if consolidate else 25)).fetchall()]
        notes = [dict(r) for r in db().execute(
            "SELECT ts, author, text FROM notes WHERE objective_id=? ORDER BY ts DESC LIMIT 10", (oid,)).fetchall()]
        recent = [_cand_row(r) for r in db().execute(
            f"SELECT {_LIGHT} FROM candidates WHERE objective_id=? ORDER BY seq DESC LIMIT 8", (oid,)).fetchall()]

    parent = None
    mode = "explore"
    # BUILD: grow the shared library -- write or improve one reusable module, test it, then
    # prove it in a candidate. Handed out more while the library is small, and more to the
    # stronger coding models, which otherwise spent whole iterations in private experiments.
    from .library import list_modules

    n_modules = len(list_modules(obj["project_id"], include_retired=False))
    p_build = 0.5 if n_modules < 6 else 0.25
    if any(k in model.lower() for k in ("qwen", "coder", "deepseek", "120b")):
        p_build = min(0.65, p_build * 1.3)
    roll = random.random()
    if roll < p_build:
        mode = "build"
    elif ranked and roll > p_build + (1 - p_build) * EXPLORE_PROBABILITY:
        mode = "improve"
        pool = ranked[:8]
        # Tournament of 3: favours the top without always picking it.
        parent = min(random.sample(pool, min(3, len(pool))), key=lambda c: pool.index(c))
    if mode == "build" and ranked:
        # Build on what is winning: offer the leader's code as the thing to factor into modules.
        parent = ranked[0]
    parent_doc = None
    if parent:
        full = get_candidate(parent["id"])
        parent_doc = {"id": full["id"], "seq": full["seq"], "rationale": full["rationale"],
                      "code": full["code"], "answer": full["answer"],
                      "in_sample": (full["metrics"] or {}).get("in_sample"),
                      "in_sample_score": full["is_score"],
                      # In-sample: before costs, after costs, flipped -- what to change first.
                      "diagnosis": ((full["metrics"] or {}).get("costs") or {}).get("verdict"),
                      "rank": ranked.index(parent) + 1}

    def brief(c: dict) -> dict:
        return {"id": c["id"], "seq": c["seq"], "model": c["model"], "status": c["status"],
                "rationale": (c["rationale"] or "")[:400], "in_sample_score": c.get("is_score"),
                "lookahead": c.get("lookahead"),
                "problem": (c.get("score_note") or "")[:300] if c["status"] == "error" or c.get("score") is None else "",
                "rank": next((i + 1 for i, x in enumerate(ranked) if x["id"] == c["id"]), None)}

    project = projects.get(obj["project_id"]) or {}
    catalog = datasource.catalog(project.get("data_dir", "")) if project else []
    from .escalation import ideas_for_context  # escalation imports this module

    return {
        "objective": {k: obj[k] for k in ("id", "title", "description", "metric", "split_date", "dataset",
                                          "time_column", "lookahead_check", "require_audit", "status", "cooldown_s",
                                          "eval_timeout_s")},
        "metric_label": METRIC_LABEL.get(obj["metric"]["kind"], obj["metric"]["kind"]),
        "datasets": [c["view"] for c in catalog][:60],
        "mode": mode,
        "parent": parent_doc,
        "leaderboard": [brief(c) for c in ranked[:6]],
        "recent": [brief(c) for c in recent],
        "total_candidates": (recent[0]["seq"] if recent else 0),
        "lessons": lessons,
        "notes": notes,
        # New directions from a stronger model, asked for because the search stopped improving.
        "ideas": ideas_for_context(oid),
        "audit": (_cand_row(pending) | {"code": pending["code"], "answer": pending["answer"]}) if pending else None,
        "consolidate": consolidate,
        "features": [{k: f.get(k) for k in ("view", "params", "rows", "columns", "skill", "usage")} for f in list_features(oid)],
        "library": _library_brief(obj["project_id"]),
        "playbook": _playbook(obj["project_id"]),
        # Only lease the practices rewrite when this agent is not already handed another chore.
        # ... and never while a mentor is on duty: it rewrites them instead of a searcher.
        "refresh_practices": (not pending and not consolidate and not _mentor_active(obj["project_id"])
                              and _practices_due(obj["project_id"])),
        "fields": field_guide(obj, project.get("data_dir", "")) if project else {},
        "field_scan": _field_scan_brief(obj["project_id"]),
        "forecast_lab": _lab_brief(obj["project_id"], obj["id"]),
        "regime_maps": _regime_brief(obj["project_id"]),
        "forecasters": _forecasters_brief(obj),
        "forecast_board": _forecast_board(obj),
    }


# What each column family of the GEX dataset measures, keyed by name prefix. Unknown families
# still appear in the guide (grouped by prefix), just without a description.
FIELD_FAMILIES = {
    "(base)": "bar OHLCV; HistVol = historical volatility, IntrVol = intraday volatility; GEX = net dealer gamma exposure (sign = long/short gamma regime)",
    "Delta": "dealer delta exposure (raw, per contract, normalised, % and notional)",
    "Gamma": "dealer gamma exposure -- how much hedging flow each move forces (raw, per contract, normalised, %, notional)",
    "Charm": "delta decay with time: hedging flow that builds into the close, incl. daily/hourly charm",
    "Vanna": "delta sensitivity to implied vol: hedging flow when IV moves",
    "IV": "implied volatility: per-contract, normalised, ATM for today (D0) and next expiry (D1), term slope/gap",
    "Pinning": "pin strength toward high-gamma strikes: pin index, local/total |GEX|, pin band, nearest wall and its z-distance, composite",
    "SkewRR": "risk-reversal skew, its 1h and overnight changes",
    "Gex": "GEX in notional and share terms",
    "GexFlip": "gamma flip levels -- where net dealer gamma changes sign (above/below)",
    "CallWall": "call walls: largest-OI call strikes, top-5 strikes and their OI",
    "PutWall": "put walls: largest-OI put strikes, top-5 strikes and their OI",
    "Pressure": "hedging pressure total, below and above spot",
    "Meta": "total OI, strike count, spot GEX, overnight rate",
    "Front": "front expiry: put/call OI and GEX ratios, its share of total gamma, days to expiry",
    "Imb": "flow imbalances per expiry (D0 = today...): IV-weighted, notional, delta, gamma, charm, volga, theta, OI, liquidity",
    "PinTrend": "pin-vs-trend day score and probabilities",
    "Surf": "greek surface around spot: gamma/vanna/charm level, strike gradient, curvature, velocity, acceleration",
    "Doi": "day-over-day OI change: call/put OI change, build skew, above/below skew, wall growth, crossover drift",
    "Bsa": "session cumulative measure", "Bsc": "correlation measure",
}
_field_cache: dict[str, dict] = {}


def field_guide(obj: dict, data_dir: str) -> dict:
    """Every column of the objective's dataset, grouped by family, with what the family is."""
    if not obj.get("dataset") or not data_dir:
        return {}
    if obj["id"] in _field_cache:
        return _field_cache[obj["id"]]
    try:
        cols = [c["name"] for c in datasource.describe(data_dir, obj["dataset"])["columns"]]
    except Exception:  # noqa: BLE001
        return {}
    fams: dict[str, list[str]] = {}
    for c in cols:
        fam = c.split("_", 1)[0] if "_" in c else "(base)"
        fams.setdefault(fam, []).append(c)
    guide = {fam: {"about": FIELD_FAMILIES.get(fam, ""), "columns": cs} for fam, cs in fams.items()}
    _field_cache[obj["id"]] = guide
    return guide


def _lab_brief(project_id: str, oid: str) -> dict | None:
    """The latest finished Forecast Lab analysis for this project, compacted for the brief."""
    try:
        from .tslab import _db as _lab_db

        conn = _lab_db()
        with _lock:
            r = conn.execute("SELECT results, created_at FROM tslab_runs WHERE project_id=? AND status='done' "
                             "ORDER BY (objective_id = ?) DESC, created_at DESC LIMIT 1", (project_id, oid)).fetchone()
    except Exception:  # noqa: BLE001
        return None
    if r is None:
        return None
    res = json.loads(r["results"] or "{}")
    if "best" not in res:
        return None
    sig = [i for i in res.get("impact", []) if i.get("impact_significant")]
    return {"target": res.get("target"), "model": res.get("model"), "points": res.get("anchors"),
            "baseline_skill": (res.get("baseline") or {}).get("skill"),
            "best_inputs": res["best"].get("inputs"), "best_skill": res["best"].get("skill"),
            "best_gain": res["best"].get("vs_baseline_gain"), "best_significant": res["best"].get("vs_baseline_significant"),
            "significant_impacts": [(i["input"], i["impact_gain"]) for i in sig][:8],
            "top_solo": [(s["input"], s["lift_gain"], s.get("lift_significant")) for s in res.get("solo", [])[:5]]}


def _field_scan_brief(project_id: str) -> dict | None:
    from .library import compact_scan, latest_field_scan

    fs = latest_field_scan(project_id)
    return compact_scan(fs["result"], 15) if fs else None


def _playbook(project_id: str) -> dict:
    from .playbook import get

    pb = get(project_id)
    return {"charter": pb["charter"], "practices": pb["practices"], "practices_version": pb["practices_version"],
            "pitfalls": pb.get("pitfalls") or ""}


def _forecast_board(obj: dict) -> list[dict]:
    from .mentor import forecast_scoreboard

    try:
        return forecast_scoreboard(obj)[:15]
    except Exception:  # noqa: BLE001 -- a scoreboard must never cost an agent its brief
        logger.exception("forecast scoreboard failed for %s", obj["id"])
        return []


def _mentor_active(project_id: str) -> bool:
    from .mentor import mentor_active

    return mentor_active(project_id)


def _practices_due(project_id: str) -> bool:
    from .playbook import practices_due

    return practices_due(project_id)


def _library_brief(project_id: str) -> list[dict]:
    from .library import brief

    return brief(project_id)


def _regime_brief(project_id: str) -> list[dict]:
    from .library import regime_brief

    return regime_brief(project_id)


def _forecasters_brief(obj: dict) -> list[dict]:
    from .tsfm import ts_manager

    project = projects.get(obj["project_id"]) or {}
    allowed = project.get("models")
    out = []
    for st in ts_manager.statuses():
        if st.get("state") != "running" or (allowed is not None and st["model_id"] not in allowed):
            continue
        h = st.get("health") or {}
        out.append({"model": st["model_id"], "family": h.get("family"), "context_length": h.get("context_length"),
                    "native_horizon": h.get("native_horizon"),
                    # Chronos-2: reads other columns as inputs (ft.forecast(inputs=[...])).
                    "supports_covariates": bool(h.get("supports_covariates"))})
    return out


class Scratch(BaseModel):
    code: str = Field(..., max_length=400_000)
    timeout_s: int = Field(120, ge=5, le=600)


@router.post("/objectives/{oid}/python")
async def scratch_python(oid: str, req: Scratch) -> dict:
    """An agent's experiment: run code against the IN-SAMPLE data only (the holdout stays
    hidden). Nothing is scored or recorded."""
    obj = get_objective(oid)
    project = projects.get(obj["project_id"])
    if project is None:
        raise HTTPException(status_code=404, detail="no such project")
    catalog = await asyncio.to_thread(datasource.catalog, project["data_dir"])
    mirror = await asyncio.to_thread(build_mirror, obj, project["data_dir"]) if obj.get("split_date") else None
    async with _EVAL_SLOTS:
        rep = await _run_forecasting(req.code, project["data_dir"], catalog, mirror, req.timeout_s, obj,
                                     obj.get("split_date"), requested_by="agent experiment")
    return {"ok": rep["ok"], "stdout": rep["stdout"][-12_000:], "stderr": rep["stderr"][-6_000:],
            "artifacts": [a["name"] for a in rep["artifacts"]], "duration_s": rep["duration_s"],
            "data": "in-sample only (rows before " + obj["split_date"] + ")" if mirror else "full"}


@router.post("/objectives/{oid}/features")
async def create_feature(oid: str, req: FeatureReq) -> dict:
    """forecast_feature: a causal forecast of one column, as a dataset candidates can load."""
    obj = get_objective(oid)
    meta = await build_feature(obj, req)
    return {k: meta[k] for k in ("view", "adjusted", "params", "rows", "columns", "skill", "skill_note", "usage") if meta.get(k) is not None} | {
        "cached": meta.get("cached", False), "seconds": meta.get("seconds")}


@router.get("/objectives/{oid}/features")
async def get_features(oid: str) -> dict:
    get_objective(oid)
    return {"features": list_features(oid)}


class ForecastByName(BaseModel):
    column: str = Field(..., max_length=500)
    dataset: str | None = Field(None, max_length=300)
    horizon: int = Field(12, ge=1, le=256)
    context: int = Field(512, ge=16, le=8192)
    model: str | None = None


@router.post("/objectives/{oid}/forecast")
async def forecast_by_name(oid: str, req: ForecastByName) -> dict:
    """One forecast of a column's most recent IN-SAMPLE values, for an agent to look at. The
    series is read here, so the agent names it instead of pasting thousands of numbers."""
    obj = get_objective(oid)
    project = projects.get(obj["project_id"])
    if project is None:
        raise HTTPException(status_code=404, detail="no such project")
    mgr, info = _forecaster(req.model)
    context = min(req.context, int(info.get("context_length") or req.context))
    dataset = req.dataset or obj.get("dataset")
    times, values, _ = await asyncio.to_thread(_load_series, project["data_dir"], obj, dataset or "", req.column,
                                               obj.get("split_date"), context)
    if len(values) < 16:
        raise HTTPException(status_code=400, detail="not enough in-sample points")
    res = await mgr.forecast(info["model"], {"series": values, "horizon": req.horizon, "quantiles": FEATURE_QUANTILES})
    fc = (res.get("forecasts") or [{}])[0]
    return {"model": info["model"], "column": req.column, "history_from": str(times[0]), "history_to": str(times[-1]),
            "last_value": values[-1], "horizon": req.horizon, "median": fc.get("median"),
            "q10": (fc.get("quantiles") or {}).get("0.1"), "q90": (fc.get("quantiles") or {}).get("0.9"),
            "note": "To use forecasts in a strategy, create a feature with forecast_feature -- scripts cannot call the model."}


class ObjQuery(BaseModel):
    sql: str = Field(..., max_length=100_000)
    max_rows: int = Field(200, ge=1, le=500)


def _insample_query(obj: dict, data_dir: str, sql: str, max_rows: int) -> dict:
    """query_data for an objective: views over the in-sample copies, so an agent exploring
    the data cannot read the holdout period. File paths are refused (they would bypass the
    views); DuckDB confinement and the SELECT-only check are the same as datasource.query."""
    from sqlglot import exp

    stmt = datasource._check_select(sql)  # noqa: SLF001
    for t in stmt.find_all(exp.Table):
        if Path(t.name).suffix.lower() in datasource._READERS:  # noqa: SLF001
            raise datasource.DataError("in objective mode refer to data by its view name, not a file path")
    for lit in stmt.find_all(exp.Literal):
        if lit.is_string and Path(lit.this).suffix.lower() in datasource._READERS:  # noqa: SLF001
            raise datasource.DataError("in objective mode refer to data by its view name, not a file path")
    mirror = build_mirror(obj, data_dir) if obj.get("split_date") else {"items": [], "root": ""}
    swapped = {it["path"]: it for it in mirror["items"]}
    root = str(Path(data_dir).resolve())
    con = duckdb.connect(":memory:")
    try:
        con.execute("SET threads = 4")
        for item in datasource.catalog(root):
            reader = datasource._READERS[Path(item["path"]).suffix.lower()]  # noqa: SLF001
            if item["path"] in swapped:
                it = swapped[item["path"]]
                src = Path(mirror["root"]) / it["mount_rel"]
                full = (src / "*.parquet").as_posix() if it["kind"] == "dir" else src.as_posix()
            else:
                full = (Path(root) / item["path"]).as_posix()
            con.execute(f'CREATE OR REPLACE VIEW "{item["view"]}" AS SELECT * FROM {reader}(\'{full.replace(chr(39), chr(39) * 2)}\')')
        fdir = features_dir(obj, obj.get("split_date"))
        for f in _feature_catalog(obj["id"]) if fdir else []:
            con.execute(f'CREATE OR REPLACE VIEW "{f["view"]}" AS SELECT * FROM read_parquet(\'{(Path(fdir) / f["path"]).as_posix()}\')')
        dirs = [root] + ([str(Path(mirror["root"]).resolve())] if mirror["root"] else []) \
            + ([str(Path(fdir).resolve())] if fdir else [])
        con.execute("SET allowed_directories = ?", [dirs])
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")
        t0 = time.time()
        cur = con.execute(stmt.sql(dialect="duckdb"))
        cols = [d[0] for d in (cur.description or [])]
        rows = cur.fetchmany(max_rows + 1)
    except duckdb.Error as exc:
        raise datasource.DataError(str(exc).splitlines()[0]) from None
    finally:
        con.close()
    return {"columns": cols, "rows": [[datasource._cell(v) for v in r] for r in rows[:max_rows]],  # noqa: SLF001
            "row_count": min(len(rows), max_rows), "truncated": len(rows) > max_rows,
            "seconds": round(time.time() - t0, 3),
            "data": f"in-sample only (rows before {obj['split_date']})" if obj.get("split_date") else "full"}


@router.post("/objectives/{oid}/data/query")
async def objective_query(oid: str, req: ObjQuery) -> dict:
    obj = get_objective(oid)
    project = projects.get(obj["project_id"])
    if project is None:
        raise HTTPException(status_code=404, detail="no such project")
    try:
        return await asyncio.to_thread(_insample_query, obj, project["data_dir"], req.sql, req.max_rows)
    except datasource.DataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
