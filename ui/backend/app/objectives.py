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

* **Hidden holdout.** Data is split at a date. Ranking needs the returns AFTER the split to
  hold up (by default the score is the weaker of in-sample and holdout, times the equity
  curve's smoothness -- see ``_robust``); agents are shown only their in-sample numbers, and
  their exploratory data access (run_python, query_data in objective mode) sees only data
  BEFORE the split.
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
import shutil
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
from .sandbox import SANDBOX_DOWN, execute

from . import task_objectives as T
from . import trade_book

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
# When the operator has flagged runs they like, this share of IMPROVE iterations builds on one.
LIKED_PARENT_PROBABILITY = 0.4
# A single dataset larger than this is not truncated for the look-ahead test (reported).
MAX_TRUNCATE_BYTES = 20 << 30
TIME_NAMES = ("timestamp", "datetime", "date", "time", "ts", "trade_date", "bar_time", "dt")

MetricKind = Literal["sharpe", "sortino", "calmar", "total_return", "cagr", "max_drawdown",
                     "reported", "judge", "task"]
RETURN_METRICS = {"sharpe", "sortino", "calmar", "total_return", "cagr", "max_drawdown"}
METRIC_LABEL = {
    "sharpe": "Sharpe ratio", "sortino": "Sortino ratio", "calmar": "Calmar ratio",
    "total_return": "total return", "cagr": "CAGR", "max_drawdown": "max drawdown",
    "reported": "reported score", "judge": "judge score (0-10)", "task": "task server score",
}
RANK_NOTE = ("Ranking rewards a SMOOTH equity curve that holds up in BOTH periods: the score is the weaker "
             "of your in-sample and the hidden holdout metric, times the R^2 of the whole equity curve "
             "(in_sample.smoothness shows yours before the split). A strategy that loses for months and "
             "then makes it all back in one burst ranks low however good its final number looks.")
TASK_RANK_NOTE = ("Ranking rewards a strategy that holds up in BOTH periods: the score is the weaker of your "
                  "in-sample and the hidden holdout value, as the task server's value function measures them. "
                  "A result that only works in-sample ranks low however good its number looks.")

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
        cols = {r[1] for r in _conn.execute("PRAGMA table_info(candidates)").fetchall()}
        if "idea_id" not in cols:
            _conn.execute("ALTER TABLE candidates ADD COLUMN idea_id INTEGER")
        # The operator's own taste: runs flagged as "this is the shape I want", with a why.
        if "liked" not in cols:
            _conn.execute("ALTER TABLE candidates ADD COLUMN liked REAL")
            _conn.execute("ALTER TABLE candidates ADD COLUMN liked_note TEXT NOT NULL DEFAULT ''")
        # Anything still "evaluating" belongs to a previous control-plane process that died
        # mid-run; it will never finish, so say so instead of showing it as in progress.
        _conn.execute("UPDATE candidates SET status='error', score_note='evaluation interrupted "
                      "(control plane restarted)' WHERE status='evaluating'")
        _conn.execute("UPDATE candidates SET lookahead='error', lookahead_detail='the look-ahead test was "
                      "interrupted (control plane restarted) -- re-run the evaluation' WHERE lookahead='pending'")
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
          "eval_seconds, champion_at, idea_id, liked, liked_note")


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
    from . import ensembles  # ensembles built on a member that just went down go with it

    hit += ensembles.disqualify_dependents(obj, seqs=set(hit), why=reason)
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


def _liked(oid: str, limit: int = 20) -> list[dict]:
    """Candidates the operator flagged as the shape they want, newest flag first. Disqualified
    ones are left out: a liked shape built on a leak is not something to build on."""
    with _lock:
        rows = db().execute(
            f"SELECT {_LIGHT} FROM candidates WHERE objective_id=? AND liked IS NOT NULL AND status='ok' "
            "AND lookahead NOT IN ('fail', 'error', 'pending') AND audit NOT IN ('fail') ORDER BY liked DESC LIMIT ?",
            (oid, limit)).fetchall()
    return [_cand_row(r) for r in rows]


def _ranked(oid: str, higher: bool, limit: int = 1000) -> list[dict]:
    """Eligible candidates, best first: scored, not caught looking ahead, not failed audit."""
    order = "DESC" if higher else "ASC"
    with _lock:
        rows = db().execute(
            f"SELECT {_LIGHT} FROM candidates WHERE objective_id=? AND status='ok' AND score IS NOT NULL "
            f"AND lookahead NOT IN ('fail', 'error', 'pending') AND audit NOT IN ('fail') ORDER BY score {order}, seq ASC LIMIT ?",
            (oid, limit),
        ).fetchall()
    return [_cand_row(r) for r in rows]


def _clone_of(oid: str, cid: str, is_score: float | None, score: float | None) -> int | None:
    """The seq of an earlier candidate of this objective with exactly this in-sample score (and the
    same ranked score), else None. #149, #151 and #153 (10-02) all scored 4.616659 in-sample -- one
    sparse strategy resubmitted from three different parents, each told only its score, so each agent
    believed it had made progress."""
    if is_score is None:
        return None
    with _lock:
        rows = db().execute(
            "SELECT seq, score FROM candidates WHERE objective_id=? AND id != ? AND status='ok' "
            "AND ABS(is_score - ?) < 1e-9 ORDER BY seq ASC", (oid, cid, float(is_score))).fetchall()
    for seq, other in rows:
        if (other is None and score is None) or (other is not None and score is not None and abs(other - score) < 1e-9):
            return int(seq)
    return None


def holdout_check(c: dict, higher: bool = True) -> str | None:
    """How a RANKED candidate's in-sample result carried over to the hidden holdout, in words and
    without its number: 'holds up' (holdout at least 70% of in-sample), 'weakens' (30-70%) or
    'collapses'. Agents saw only in-sample scores and built on whatever scored best there; the
    top in-sample candidates of 19f971 (#157: 4.695 in-sample, 0.147 holdout) were the ones that
    generalised worst. None for an unranked candidate or one with nothing to carry over."""
    if not higher or c.get("score") is None:
        return None
    ho = ((c.get("metrics") or {}).get("rank") or {}).get("holdout")
    is_score = c.get("is_score")
    if ho is None or is_score is None or float(is_score) <= 0:
        return None
    ratio = float(ho) / float(is_score)
    return "holds up" if ratio >= 0.7 else "weakens" if ratio >= 0.3 else "collapses"


def _distinct(ranked: list[dict]) -> list[dict]:
    """The ranking without clones: a candidate scoring exactly what a better-ranked (or earlier)
    one scores, in-sample and on the holdout, is the same strategy resubmitted. Six copies of
    #59 (-2.657) filled the leaderboard agents saw and the pool their parents were drawn from."""
    seen, out = set(), []
    for c in ranked:
        key = (round(float(c["score"]), 9), None if c.get("is_score") is None else round(float(c["is_score"]), 9))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


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
        "smoothness": _smoothness(rets),
    })
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}


def _smoothness(rets: list[float]) -> float | None:
    """How straight the equity curve is: R^2 of log equity against time, signed by its slope.

    1.0 is a steady climb; near 0 is a random walk or a curve that went nowhere for months and
    then jumped; negative is a steady decline. Flat days count as time, so a strategy that
    earned everything in one burst scores low even if the burst was large."""
    n = len(rets)
    if n < 3:
        return None
    eq = np.cumsum(np.log1p(np.maximum(np.asarray(rets, dtype=float), -0.999999)))
    t = np.arange(n, dtype=float)
    tc, ec = t - t.mean(), eq - eq.mean()
    ss_e = float((ec * ec).sum())
    if ss_e < 1e-18:
        return None
    r = float((tc * ec).sum()) / math.sqrt(float((tc * tc).sum()) * ss_e)
    return math.copysign(r * r, r)


def _side_gap(m: dict, sides: dict | None) -> str | None:
    """Why a candidate is too one-sided to rank, or None. With min_side_share set (and both
    sides allowed), longs and shorts must each be at least that share of the in-sample trades:
    a strategy that only ever buys is riding the market's drift, not reading the signal.
    `sides` is None when not measured (an ensemble, a candidate scored before it was)."""
    share = float(m.get("min_side_share") or 0.0)
    if not share or (m.get("direction") or "both") != "both" or not sides:
        return None
    lo, sh = int(sides.get("long") or 0), int(sides.get("short") or 0)
    n = lo + sh
    if not n:
        return ("no trades: the strategy never opened a position in-sample -- its entry conditions never fire "
                "together. Loosen the strictest threshold or drop a gate, and check each condition's hit rate "
                "on its own before combining them")
    if min(lo, sh) >= share * n:
        return None
    weak = "short" if sh <= lo else "long"
    return (f"one-sided: {lo} long and {sh} short trades in-sample -- {weak} trades must be at least "
            f"{share:.0%} of them. Add the mirrored {weak} entry (the same conditions reversed) so it fires "
            f"when the signal points that way")


def _score_returns(obj: dict, returns: list[list], sides: dict | None = None) -> tuple[float | None, float | None, str, dict]:
    """(holdout score, in-sample score, note, metrics) for a trading-style objective. `sides`
    (in-sample long and short trade counts) lets a one-sided candidate go unranked when the
    objective asks for both sides -- see _side_gap."""
    score, is_score, note, metrics = _score_returns_unsided(obj, returns)
    gap = _side_gap(obj["metric"], sides)
    if gap:
        metrics["one_sided"] = gap
        return None, is_score, gap, metrics
    return score, is_score, note, metrics


def _score_returns_unsided(obj: dict, returns: list[list]) -> tuple[float | None, float | None, str, dict]:
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

    is_score, is_note = pick(metrics["in_sample"])
    if not split:
        score, note = pick(metrics["full"])
        return score, is_score, note, metrics
    ho_score, note = pick(metrics["holdout"])
    if ho_score is None and note:
        note = f"holdout: {note}" + T._sparse_hint(metrics["in_sample"], {"note": note, **metrics["holdout"]})
    if m.get("rank", "robust") != "robust":
        return ho_score, is_score, note, metrics
    score, metrics["rank"] = _robust(is_score, ho_score, metrics["full"].get("smoothness"),
                                     bool(m.get("higher_is_better", True)))
    if score is None and not note:
        note = f"in-sample: {is_note}"
    return score, is_score, note, metrics


def _robust(is_score: float | None, ho_score: float | None, smooth: float | None,
            higher: bool) -> tuple[float | None, dict]:
    """The leaderboard score: the WEAKER of the in-sample and holdout metric, times how smooth
    the equity curve is over the whole period.

    Ranking on the holdout alone crowned #1031 (holdout Sharpe 3.40): it lost money for the
    whole in-sample year and made everything in the last few months -- a lucky regime, not a
    strategy (full-period R^2 0.01). A strategy worth trading earns in both periods and keeps
    climbing through the split, so the worse period caps the score and a jagged curve discounts
    it. A negative base is left as it is: scaling a loss by a low R^2 would reward the noise."""
    if is_score is None or ho_score is None:
        return None, {}
    base = min(is_score, ho_score) if higher else max(is_score, ho_score)
    s = max(0.0, smooth or 0.0)
    score = base * s if higher and base > 0 else base
    return round(score, 6), {"method": "robust", "base": round(base, 6), "smoothness": round(s, 6),
                             "weaker": "in_sample" if (is_score <= ho_score) == higher else "holdout",
                             "holdout": ho_score}


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

    out: dict[str, Any] = {"metric": kind, "cost_bps": m.get("cost_bps"), "direction": m.get("direction") or "both",
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
            "bar (every size change is a trade that pays costs) -- set the size once at entry instead "
            "(ft.size, e.g. with ft.inverse_vol).")
    if abs(gross) <= noise:
        return (head + " No clear edge before costs in either direction: the idea itself does not work here, "
                "not just its costs -- change the signal, not its thresholds.")
    if gross > 0:
        return head + (" The signal has an edge before costs; trading it this often gives it away. " + less
                       if net < 0 or gross - net > gross / 3 else "")
    side = costs.get("direction") or "both"
    if side != "both":
        # The flipped strategy is on the side this objective forbids: there is nothing to flip to.
        return (head + f" It loses before costs, and this objective is {side.upper()} ONLY, so flipping it is not "
                "an option: change the entry signal, not its thresholds.")
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
    from .deci_core import plain_math

    names = {c.lower(): c for c in columns}
    if expr in columns:
        return f'"{expr}"'
    expr = plain_math(expr)
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
                import difflib

                close = difflib.get_close_matches(node.name.lower(), list(names), n=3, cutoff=0.6)
                raise HTTPException(status_code=400, detail=f"unknown column {node.sql()!r}" + (
                    f" -- did you mean {', '.join(names[c] for c in close)}?" if close else
                    " -- use a column of the dataset (describe_data lists them); a change or lag is not a "
                    "column, compute it in run_python"))
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


def _iso(x: Any) -> str | None:
    try:
        return str(np.datetime64(x, "s"))
    except (TypeError, ValueError):
        return None if x is None else str(x)


def input_streams(kind: str, model: str, dataset: str | None, times: Any, anchors: list[int], context: int,
                  horizon: int, streams: list[tuple[str, str]], *, every: int | None = None, bar: str | None = None,
                  requested_by: str | None = None, inclusive: bool = True, also_without_inputs: bool = False) -> dict:
    """What a forecast request sends, with dates: the as-of time(s) the forecasts are made
    from, and per input stream its role, source and the date range actually sent.

    `streams` is [(name, role)], role one of "target", "past covariate", "calendar (past)",
    "known ahead" (future covariates) or "candidate input" (the Forecast Lab). The context
    for anchor a is the `context` rows ending AT a (`inclusive`, the feature builders) or
    just before it (the lab). The forecaster never sees any of this -- only arrays -- so it is
    recorded here, where the timestamps are still known, for the agent inspector."""
    n = len(times)
    if not anchors or n == 0:
        return {"kind": kind, "model": model, "streams": []}
    ts = np.asarray(times, dtype="datetime64[s]")
    diffs = np.diff(ts[-200:]).astype("int64") if n > 2 else np.array([60])
    step = int(np.median(diffs[diffs > 0])) if (diffs > 0).any() else 60

    def window(a: int) -> tuple[int, int]:
        hi = a if inclusive else a - 1
        return max(0, hi - context + 1), hi

    lo0, _ = window(anchors[0])
    _, hi1 = window(anchors[-1])
    first_asof, last_asof = ts[window(anchors[0])[1]], ts[hi1]
    ahead = np.timedelta64(step * horizon, "s")
    out_streams = []
    for name, role in streams:
        if role == "known ahead":
            span = (first_asof + np.timedelta64(step, "s"), last_asof + ahead, horizon)
        else:
            span = (ts[lo0], ts[hi1], hi1 - lo0 + 1)
        out_streams.append({"name": name, "role": role, "dataset": dataset, "from": _iso(span[0]), "to": _iso(span[1]),
                            "points": int(span[2]), "bar": bar or f"{step}s"})
    picks = sorted({0, len(anchors) // 2, len(anchors) - 1})
    samples = []
    for k in picks:
        lo, hi = window(anchors[k])
        samples.append({"as_of": _iso(ts[hi]), "from": _iso(ts[lo]), "to": _iso(ts[hi]), "context": hi - lo + 1,
                        "horizon_end": _iso(ts[hi] + ahead)})
    return {"kind": kind, "model": model, "dataset": dataset, "requested_by": requested_by,
            "bar": bar or f"{step}s", "step_seconds": step,
            "as_of": {"first": _iso(first_asof), "last": _iso(last_asof), "anchors": len(anchors), "every": every},
            "horizon": {"bars": horizon, "end": _iso(last_asof + ahead), "approximate": True},
            "context_bars": context, "streams": out_streams, "samples": samples,
            "also_without_inputs": also_without_inputs}


def _note_inputs(model: str, detail: dict) -> None:
    """Hand a request's input description to the agent inspector (never fails the forecast)."""
    try:
        from .agent_activity import note_inputs

        note_inputs(model, detail)
    except Exception:  # noqa: BLE001
        logger.debug("note_inputs failed", exc_info=True)


def _capture(detail: dict, times: Any, anchors: list[int], context: int, horizon: int, split: str | None,
             inclusive: bool = True):
    """The values behind the request's sample anchors, for the agent inspector
    (forecast_values.ValueCapture). Inert, never raising, if it cannot be set up."""
    from . import forecast_values

    return forecast_values.start(detail, times, anchors, context=context, horizon=horizon, split=split,
                                 inclusive=inclusive)


class _Col:
    """Column `k` of a list of rows, sliced on demand (the candle closes, without a copy)."""

    def __init__(self, rows: list, k: int) -> None:
        self.rows, self.k = rows, k

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, s: slice) -> list:
        return [r[self.k] for r in self.rows[s]]


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
    detail = input_streams("feature", info["model"], dataset, times, anchors, context, horizon,
                           [("OHLCV candles", "target")], every=every, bar=bar)
    _note_inputs(info["model"], detail)
    cap = _capture(detail, times, anchors, context, horizon, obj.get("split_date"))
    cap.context_values("OHLCV candles", "target", {"close": _Col(candles, 3)})
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
        for j, fc in enumerate(res.get("forecasts") or []):
            cap.output(chunk[j] if j < len(chunk) else -1, "OHLCV candles", fc)
            qs = fc.get("quantiles") or {}
            med.append(fc["median"][-1])
            mean_path.append(sum(fc["median"]) / len(fc["median"]))
            q10.append((qs.get("0.1") or fc["median"])[-1])
            q90.append((qs.get("0.9") or fc["median"])[-1])
            hi.append(max(fc.get("high_q90") or fc["median"]))
            lo.append(min(fc.get("low_q10") or fc["median"]))
    closes = [c[3] for c in candles]
    cap.realized("OHLCV candles", closes)
    cap.publish(info["model"])
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
    detail = input_streams(
        "feature", info["model"], dataset, t, anchors, context, horizon,
        [(c, "target") for c in targets] + [(c, "past covariate") for c in covs]
        + ([("minute_of_day, weekday", "calendar (past)"), ("minute_of_day, weekday", "known ahead")] if req.calendar else []),
        every=every, bar=req.bar, also_without_inputs=bool(covs or req.calendar))
    _note_inputs(info["model"], detail)
    cap = _capture(detail, t, anchors, context, horizon, obj.get("split_date"))
    if cap.ok:
        try:
            for c in targets:
                cap.context_values(c, "target", {c: cols[c]})
            for c in covs:
                cap.context_values(c, "past covariate", {c: cols[c]})
            if req.calendar:
                cap.context_values("minute_of_day, weekday", "calendar (past)", _calendar(t, horizon)[0])
                cap.ahead_values("minute_of_day, weekday", "known ahead",
                                 lambda a: _calendar(t[a - context + 1:a + 1], horizon)[1])
        except Exception:  # noqa: BLE001 -- the inspector's view must never fail the build
            logger.warning("forecast values: input capture failed", exc_info=True)
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
            for j, fc in enumerate(res.get("forecasts") or []):
                for c, v in zip(targets, fc.get("variates") or [fc]):
                    cap.output(chunk[j] if j < len(chunk) else -1, c, v, with_inputs=with_inputs)
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
    for c in targets:
        cap.realized(c, cols[c])
    cap.publish(info["model"])
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
    detail = input_streams("feature", info["model"], dataset, times, anchors, context, req.horizon,
                           [(e, "target") for e, _ in aligned], every=every)
    _note_inputs(info["model"], detail)
    cap = _capture(detail, times, anchors, context, req.horizon, obj.get("split_date"))
    batch = max(1, min(256, 90_000 // context))
    cols: dict[str, Any] = {"t": pa.array([times[i] for i in anchors], type=pa.timestamp("us"))}
    described: dict[str, str] = {"t": "bar timestamp the forecasts were made AT (data up to and including t)"}
    skills: dict[str, dict] = {}
    single = len(aligned) == 1
    for expr, values in aligned:
        cap.context_values(expr, "target", {expr: values})
        cap.realized(expr, values)
        med, q10, q90, mean_path = [], [], [], []
        for b in range(0, len(anchors), batch):
            chunk = anchors[b:b + batch]
            try:
                res = await mgr.forecast(info["model"], {
                    "series": [values[i - context + 1:i + 1] for i in chunk],
                    "horizon": req.horizon, "quantiles": FEATURE_QUANTILES})
            except Exception as exc:  # noqa: BLE001 -- TsError and transport errors alike
                raise HTTPException(status_code=502, detail=f"forecaster failed on {expr}: {exc}") from None
            for j, fc in enumerate(res.get("forecasts") or []):
                cap.output(chunk[j] if j < len(chunk) else -1, expr, fc)
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
    cap.publish(info["model"])
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
# The brief says "use polars" and ft.load_pl() / ft.rows_pl() return polars, but ft.load(),
# ft.rows() and every ft helper that returns a frame (ft.resample, ft.inverse_vol, ft.size,
# ft.align, ...) return pandas -- so a script holds both kinds, and "'DataFrame' object has no
# attribute 'with_columns'" (the most repeated agent error in the logs, bug #52) does not say
# which kind it was holding. The traceback does: pandas raises that error from its own
# __getattr__ (pandas/core/generic.py), polars raises it bare.
_NO_ATTR = re.compile(r"AttributeError: '(DataFrame|Series|LazyFrame)' object has no attribute '(\w+)'")
_PANDAS_GETATTR = re.compile(r'File "[^"]*pandas[^"]*generic\.py", line \d+, in __getattr__')
# pandas methods models call on a polars frame (the control plane does not import pandas to ask).
_PANDAS_NAMES = frozenset((
    "sort_values", "sort_index", "reset_index", "set_index", "copy", "iloc", "loc", "index", "values", "assign",
    "groupby", "astype", "fillna", "ffill", "bfill", "dropna", "isna", "notna", "isnull", "notnull", "ewm",
    "expanding", "cumsum", "apply", "iterrows", "itertuples", "resample", "T", "tolist", "nunique", "merge"))
# The third kind a script holds: a numpy array, from its own .values / .to_numpy() / np.where(...)
# (or an ft helper that returns one, e.g. ft.admit), then a pandas/polars method called on it --
# "'numpy.ndarray' object has no attribute 'rolling'" (candidate 4ac34e667a: skew = df[c].values;
# skew.rolling(2000)). pandas Series methods an ndarray lacks, beyond _PANDAS_NAMES:
_NP_NO_ATTR = re.compile(r"AttributeError: 'numpy\.ndarray' object has no attribute '(\w+)'")
_PANDAS_SERIES_NAMES = _PANDAS_NAMES | frozenset((
    "rolling", "shift", "diff", "pct_change", "abs", "rank", "cummax", "cummin", "cumprod", "where", "mask",
    "median", "quantile", "between", "map", "str", "dt", "iat", "at", "head", "tail", "to_numpy", "unique",
    "value_counts", "isin", "replace", "interpolate", "corr", "cov", "skew", "kurt", "idxmax", "idxmin",
    "to_list", "to_frame", "reindex", "rename", "combine_first"))
# polars has two kinds of column: a Series is data (df["x"], df.get_column("x")); an expression
# (pl.col("x")...) is a recipe that only runs inside df.select / with_columns / filter. Models mix
# them -- "'Series' object has no attribute 'over'" (#130: s.cum_sum().over('session')),
# "'Expr' object has no attribute 'to_numpy'" -- and a reduction on a Series is a plain number:
# "'float' object has no attribute 'over'" (df["x"].mean().over(...)). None of these is the
# pandas/polars mix-up above, and each cost a submission on the night of 2026-09-30.
_EXPR_NO_ATTR = re.compile(r"AttributeError: 'Expr' object has no attribute '(\w+)'")
_SCALAR_NO_ATTR = re.compile(r"AttributeError: '(float|int|bool|NoneType)' object has no attribute '(\w+)'")
_ANY_NO_ATTR = re.compile(r"AttributeError: '[\w.]+' object has no attribute '(\w+)'")
# Methods models expect that neither library has; Python's "Did you mean: 'cum_max'?" points the wrong way.
_RUNNING_MEAN = ("there is no cum_mean in polars: a running mean is x.cum_sum() / x.cum_count() -- per session, "
                 "df.with_columns((pl.col('x').cum_sum() / pl.col('x').cum_count()).over('session').alias('x_mean')) "
                 "(pandas: x.expanding().mean(), per session x.groupby(session).transform(lambda s: s.expanding().mean()))")
_NTH = ("a polars column has no .nth: one value is .get(i) -- per session pl.col('x').get(0).over('session') for the "
        "session's first -- several are .gather([i, j]); pl.nth(i) picks the i-th COLUMN of a frame")
# 10-01 01:0x: pd.Series(p, index=bars_fc.to_datetime('SlotUtc')) -- pandas' to_datetime is a module function.
_TO_DATETIME = ("to_datetime is a pandas FUNCTION, not a method: pd.to_datetime(df['x']) -- and the rows' t (or a "
                "dataset's SlotUtc) is already a datetime; polars text to time is pl.col('x').str.to_datetime()")
_NO_SUCH_METHOD = {"nth": _NTH, "to_datetime": _TO_DATETIME,
                   "cum_mean": _RUNNING_MEAN, "cummean": _RUNNING_MEAN, "cumulative_mean": _RUNNING_MEAN,
                   "expanding_mean": _RUNNING_MEAN, "cum_std": _RUNNING_MEAN.replace("running mean", "running mean "
                   "(a running std needs the running mean of x and of x**2)")}


def _polars_kind_hint(kind: str, attr: str) -> str:
    """The line for a Series method called on an expression, an expression method called on a
    polars Series, or either one's method on the number a reduction returned; "" otherwise."""
    import polars as pl

    if kind == "Expr":
        if hasattr(pl.Series, attr) or attr in _PANDAS_SERIES_NAMES:
            return (f"pl.col(...) and anything built from it is a polars EXPRESSION -- a recipe, not data -- so it has "
                    f"no .{attr}. Run it on the frame first: df.select(expr).to_series().{attr}(...), or add it as a "
                    f"column with df.with_columns(expr.alias('x')) and use df['x'].{attr}(...)")
        return ""
    if kind == "DataFrame":
        if not hasattr(pl.DataFrame, attr) and (hasattr(pl.Expr, attr) or hasattr(pl.Series, attr)):
            return (f"that is a whole polars DataFrame and .{attr} belongs to one column: df['x'].{attr}(...), or "
                    f"pl.col('x').{attr}(...) inside df.select / df.with_columns")
        return ""
    if kind == "Series":
        if hasattr(pl.Expr, attr) and not hasattr(pl.Series, attr):
            return (f"that Series is polars DATA (df['x'] / get_column) and .{attr} exists only on EXPRESSIONS. Write "
                    f"the whole calculation as one expression inside the frame, e.g. "
                    f"df.with_columns(pl.col('x').cum_sum().{attr}('session').alias('y'))" if attr == "over" else
                    f"that Series is polars DATA (df['x'] / get_column) and .{attr} exists only on EXPRESSIONS: use "
                    f"pl.col('x').{attr}(...) inside df.select / df.with_columns")
        return ""
    if hasattr(pl.Expr, attr) or hasattr(pl.Series, attr) or attr in _PANDAS_SERIES_NAMES:
        return (f"that is a plain {kind}, not a column: a reduction (.mean(), .std(), .sum(), .max(), .item(), ...) on "
                f"a Series returns one number. For a per-session value keep it an expression: "
                f"df.with_columns(pl.col('x').mean().over('session').alias('x_mean'))")
    return ""


def _numpy_hint(attr: str) -> str:
    """The line for a pandas/polars method called on a numpy array; "" for any other name."""
    if attr in ("values", "to_numpy"):
        return (f"that is already a NUMPY array (.values / .to_numpy() was taken earlier) -- drop the .{attr}, "
                "or keep the pandas/polars column until the series maths is done")
    src = ("a NUMPY array (from .values / .to_numpy(), np.where / np.* maths, or an ft helper that returns an "
           "array)")
    if attr in _PANDAS_SERIES_NAMES:
        return (f"that is {src} and .{attr} is a pandas method. Wrap it -- pd.Series(x, index=df.index).{attr}(...)"
                f" -- or call .{attr} on the pandas column BEFORE .values / .to_numpy() (numpy: x[i], np.roll, "
                "np.diff, np.abs, ...)")
    if attr in dir(list) or attr == "len":
        return ""                                  # a list/numpy habit, not the other libraries' method
    import polars as pl

    if hasattr(pl.Series, attr) or hasattr(pl.Expr, attr):
        return (f"that is {src} and .{attr} is a polars method. Wrap it -- pl.Series(x).{attr}(...) -- or call "
                f".{attr} on the polars column before .to_numpy()")
    if hasattr(pl.DataFrame, attr):
        return (f"that is {src} and .{attr} is a polars DataFrame method. Keep the polars frame, or build one: "
                f"pl.DataFrame({{'x': x}}).{attr}(...)")
    return ""                                      # a typo or a cut-off name: no hint to mislead with


def _frame_hint(stderr: str) -> str:
    """"" unless the script called a polars method on a pandas frame, or the other way round, or
    either one's method on a numpy array: then one line saying which kind of object it holds and
    how to get the other. Also a polars Series/expression mix-up, and methods no library has."""
    stderr = stderr or ""
    last: dict[str, re.Match] = {}
    for name, rx in (("frame", _NO_ATTR), ("numpy", _NP_NO_ATTR), ("expr", _EXPR_NO_ATTR),
                     ("scalar", _SCALAR_NO_ATTR), ("any", _ANY_NO_ATTR)):
        for hit in rx.finditer(stderr):
            last[name] = hit                       # the last one is the error the run died of
    if "any" in last and last["any"].group(1) in _NO_SUCH_METHOD:
        died = last["any"]
        if all(died.start() >= h.start() for h in last.values()):
            return _NO_SUCH_METHOD[died.group(1)]
    m, n = last.get("frame"), last.get("numpy")
    other = max((last[k] for k in ("expr", "scalar") if k in last), key=lambda h: h.start(), default=None)
    if other is not None and all(other.start() > h.start() for h in (m, n) if h is not None):
        kind, attr = ("Expr", other.group(1)) if other.re is _EXPR_NO_ATTR else other.groups()
        return _polars_kind_hint(kind, attr)
    if n is not None and (m is None or n.start() > m.start()):
        return _numpy_hint(n.group(1))
    if m is None:
        return ""
    kind, attr = m.groups()
    frame = stderr.rfind('File "', 0, m.start())
    if frame < 0:
        return ""                                  # no traceback left to tell the two apart
    if kind != "LazyFrame" and _PANDAS_GETATTR.match(stderr, frame):
        import polars as pl

        if not any(hasattr(c, attr) for c in (pl.DataFrame, pl.Series, pl.Expr)):
            return ""                              # a typo or a cut-off name, not the other library's method
        return (f"that {kind} is PANDAS and .{attr} is a polars method. ft.load(), ft.rows() and every ft helper "
                "that returns a frame or series (ft.resample, ft.inverse_vol, ft.size, ft.align, ...) give pandas, "
                "even when you pass polars in; only ft.load_pl() / ft.rows_pl() give polars. Use the pandas call, "
                "or convert with pl.from_pandas(x)")
    if attr in _PANDAS_NAMES:
        return (f"that {kind} is POLARS (from ft.load_pl / ft.rows_pl) and .{attr} is a pandas method. Use the "
                "polars call, or convert with x.to_pandas() -- a library module written for pandas needs "
                "module.signal(df.to_pandas())")
    return _polars_kind_hint(kind, attr) if kind in ("Series", "DataFrame") else ""


def error_hint(stderr: str, code: str = "") -> str:
    """The one-line fix for a failed run's stderr, or "": run_python, submissions, the regime lab and
    library smoke tests all read it from here, so a new hint reaches every tool at once. `code`, the
    script, sharpens the hints that depend on what was written (a misplaced .alias())."""
    # .alias on a literal number reads like a reduction's scalar to _frame_hint -- answer it first (10-01 16:32).
    if _ALIAS_ON_NUMBER.search(stderr or ""):
        return _misc_hint(stderr, code)
    return _frame_hint(stderr) or _precedence_hint(stderr) or _dtype_hint(stderr) or _misc_hint(stderr, code)


def _failure_note(stderr: str, code: str = "") -> str:
    """The failure line agents read, with the fix for mistakes the team keeps repeating."""
    note = "the script failed -- see stderr"
    hint = error_hint(stderr, code)
    if hint:
        note += f". Hint: {hint}"
    return note


# `a > 0 & b < 1` is `a > (0 & b) < 1`: & and | bind tighter than comparisons. pandas says
# "Cannot perform 'rand_' with a dtyped [float64] array and scalar of type [bool]" (bug #168),
# plain Python "unsupported operand type(s) for &: 'float' and 'bool'" -- neither names the cause.
_PRECEDENCE = re.compile(r"[Cc]annot perform '(?:r?and_|r?or_|r?xor_)' with a dtyped|"
                         r"unsupported operand type\(s\) for [&|^]: '\w+' and 'bool'")


def _precedence_hint(stderr: str) -> str:
    if not _PRECEDENCE.search(stderr or ""):
        return ""
    return ("& and | bind tighter than > < ==, so `a > 0 & b < 1` runs as `a > (0 & b) < 1`. Put every comparison in "
            "its own parentheses: (a > 0) & (b < 1)")


# Bug #134 (16 times): the rows' `t` is already a Datetime, and models parse it as text --
# pl.col('t').str.strptime(...) / .str.to_datetime() -> "SchemaError: invalid series dtype:
# expected `String`, got `datetime[ns]` for series with name `t`".
_STR_ON_DATETIME = re.compile(r"expected `String`, got `(datetime[^`]*|date)` for series with name `(\w+)`")


def _dtype_hint(stderr: str) -> str:
    m = _STR_ON_DATETIME.search(stderr or "")
    if not m:
        return ""
    col = m.group(2)
    return (f"`{col}` is already a polars {m.group(1).split('[')[0].title()} -- drop the .str.strptime / "
            f".str.to_datetime and use pl.col('{col}').dt.date(), .dt.hour(), .dt.minute(); for New York session "
            f"and minute of day use ft.clock(rows['{col}'])")


# Seen on 2026-10-01 after the restart, each with no hint: pl.col('Close').list().over(...)
# ("'ExprListNameSpace' object is not callable"); a boolean column with nulls (rolling warm-up)
# taken .to_numpy().astype(int) ("int() argument must be ... not 'NoneType'"); a column the frame
# was not loaded with ("unable to find column \"Volume\"; valid columns: [...6 names]").
_NAMESPACE_CALL = re.compile(r"'Expr(\w+)NameSpace' object is not callable")
_DUPLICATE_NAME = re.compile(r"DuplicateError: \w+ contained duplicate output name '([^']+)'")
_NONE_TO_NUMBER = re.compile(r"(int|float)\(\) argument must be .* not 'NoneType'")
_MISSING_COLUMN = re.compile(r'unable to find column "(\w+)"; valid columns: \[([^\]]*)\]')
# 10-01 12:12/12:22: with_columns(pl.col('Close').shift(-1).alias('ret_1').drop_nulls()) -- a column
# shorter than its frame: "can't broadcast Series 'ret_1' of length 496481 to length 496482".
_LENGTH_CHANGE = re.compile(r"can't broadcast Series '([^']*)' of length (\d+) to length (\d+)")
# 10-01 12:20: rolling_quantile(0.9, 20) -- 20 landed in `interpolation`: "TypeError: 'int' object is
# not an instance of 'str'\nwhile processing 'interpolation'". The frame names the polars method.
_WRONG_SLOT = re.compile(r"TypeError: (?:argument '(?P<arg>\w+)': )?'(?P<got>\w+)' object (?:is not an instance of "
                         r"'\w+'|cannot be converted to '\w+')(?:\s*while processing '(?P<param>\w+)')?")
_POLARS_FRAME = re.compile(r'File "[^"]*polars[/\\][^"]*", line \d+, in (\w+)')
# 10-01 12:16: clock = ft.clock(rows['t']); clock['session'] -- a tuple read as a dict.
_TUPLE_BY_NAME = re.compile(r"tuple indices must be integers or slices, not str")
# 10-01 14:10: fwd = close[h:] - close[:-h]; np.corrcoef(z[:-h], fwd[:-h]) -- a shifted slice taken twice.
_ARRAY_LENGTHS = re.compile(r"array at index 0 has size (\d+) and the array at index 1 has size (\d+)|"
                            r"operands could not be broadcast together with shapes \((\d+),\) \((\d+),\)")
# 10-01 14:49: range_z[mask][valid] = r[valid] - ..., with `valid` built over ALL rows and r one session's.
_MASK_LENGTH = re.compile(r"boolean index did not match indexed array along (?:axis|dimension) 0; "
                          r"(?:size of axis|dimension) is (\d+) but (?:size of )?corresponding boolean (?:axis|dimension) is (\d+)")
# 10-01 15:28: with_columns([(Close.shift(-6) - Close) / Close, (Close.shift(-30) - Close) / Close]) -- two
# expressions without .alias both named Close.
_DUP_WITH_COLUMNS = re.compile(r"the name '([^']+)' passed to `(?:Lazy)?(?:Frame|DataFrame)\.with_columns` is duplicate")
# 10-01 15:32: f"{lo if lo else '-inf':.2f}" -- text into a number format.
_TEXT_NUMBER_FORMAT = re.compile(r"Unknown format code '[a-zA-Z%]' for object of type 'str'")
# 10-01 15:28: a Python for-loop over every row -- killed at the time limit.
_TIMED_OUT = re.compile(r"\[killed: exceeded the \d+s limit\]")
# 10-01 15:08: np.linalg.lstsq on windows holding NaN -- "SVD did not converge in Linear Least Squares".
_LSTSQ_NAN = re.compile(r"LinAlgError: SVD did not converge")
# 10-02 00:06: p (indexed per bar) * scale (reindexed by the day of each bar) -- pandas aligns two Series on the
# UNION of their labels, so the product had 2N rows: "Length of values (992964) does not match length of index (496482)".
_LENGTH_VALUES = re.compile(r"Length of values \((\d+)\) does not match length of index \((\d+)\)")
# 10-02 00:0x: ts_a / ts_b on Timestamps.
_TIMESTAMP_DIV = re.compile(r"unsupported operand type\(s\) for /: 'Timestamp' and 'Timestamp'")
# 10-01 22:19: pl.col('Close').head(30).over('session') -- an expression inside .over() changed the group's length.
_WINDOW_LENGTH = re.compile(r"the length of the window expression did not match that of the group")
# 10-01 15:03: f"{x:.4f if x else 'N/A'}" -- a conditional inside the format spec.
_SPEC_CONDITIONAL = re.compile(r"Invalid format specifier '[^']*\bif\b")
# 10-01 14:37: pl.col('Close').shift(-pl.col(...)) -- shift takes one number for the whole column.
_SHIFT_BY_EXPR = re.compile(r"ShapeError: 'n' must be a scalar value")
# 10-01 14:16: pl.col('t').dt.date().diff().fill_null(0) -- the diff of dates is a Duration, not a number.
_DURATION_FILL = re.compile(r"invalid or ambiguous dtypes: '\[duration\[\w+\], dyn int\]'")
# ft now words a missing polars column '"ret_60f" not found -- did you mean ...? The frame has: t, Close, ...'
# (10-01 15:06), and polars itself sometimes says only '"Doi_MultiSlope" not found' (10-01 00:59).
_MISSING_COLUMN_FT = re.compile(r'"(\w+)" not found(?: -- did you mean [^?\n]*\?)? The frame has: ([^\n]*)')
_NOT_FOUND = re.compile(r'ColumnNotFoundError: "(\w+)" not found')
# 10-01 00:42-03:59 (4 runs): ft.rows_pl(columns=[..., 'GexFlip_Pos_vs_price_bps', 'minutes_into_session'])
# -- names the TRADE REVIEW (trade_book.py) gives its features, read off the brief as if they were columns.
_MISSING_NAMES = re.compile(r'unable to find column "([^"]+)"|"([^"]+)" not found|KeyError: "?\'([^\']+)\'')
_ROWS_HAVE_NO = re.compile(r"the rows have no (.*?)(?: -- ft\.task|$)", re.M)
_REVIEW_FEATURE = re.compile(r"(\w+)_vs_price_bps|minutes_into_session|price_chg_(\d+)m(?:_bps)?|"
                             r"price_since_open(?:_bps)?|price_in_day_range")
# 10-01 06:17-14:20 (3 runs): group_by_dynamic('SlotUtc', every='15min') -- pandas' spelling of a duration.
_DURATION_UNIT = re.compile(r"unit: '(\w+)' not supported; available units are")
# 10-01 03:31 (and 00:2x with numpy): `a == b & c != 0` / `x and False` / `if col:` -- Python asks a whole
# column for ONE True/False.
_TRUTH_VALUE = re.compile(r"[Tt]he truth value of an? (?:Expr|Series|DataFrame|array with more than one element) "
                          r"is ambiguous")
# 10-01 00:4x (3 runs): pl.col('sd').clip(lower=1e-9), .rolling_std(20).max(1e-9), .rolling(20, min_periods=1)
# -- a polars method called with pandas' / numpy's arguments; the TypeError does not say what it does take.
_BAD_CALL = re.compile(r"TypeError: (?P<cls>Expr|Series|DataFrame|LazyFrame)\.(?P<fn>\w+)\(\) "
                       r"(?:got an unexpected keyword argument '(?P<kw>\w+)'|missing \d+ required "
                       r"(?:keyword-only |positional )?arguments?: [^\n]*|takes \d+ positional arguments? but \d+ "
                       r"(?:was|were) given)")
_KW_ALIASES = {"lower": "lower_bound", "upper": "upper_bound", "min": "lower_bound", "max": "upper_bound",
               "min_periods": "min_samples", "window": "window_size", "ascending": "descending",
               "periods": "n", "inplace": None}
# 10-01 00:5x / 01:2x: pl.col('SlotUtc').dt.cast(pl.Date), .dt.with_time_zone('UTC'), df['t'].diff().dt.seconds(),
# s.dt.strftime(...).str.alias(...) -- a namespace (.dt / .str) treated as a column or given a pandas name.
_NS_NO_ATTR = re.compile(r"AttributeError: '(?:Expr)?(\w+?)NameSpace' object has no attribute '(\w+)'")
_NS_ACCESSOR = {"DateTime": ("dt", "pl.col('t').dt.date()"), "String": ("str", "pl.col('s').str.slice(0, 4)"),
                "List": ("list", "pl.col('l').list.first()"), "Array": ("arr", "pl.col('a').arr.first()"),
                "Struct": ("struct", "pl.col('s').struct.field('x')"), "Name": ("name", "pl.col('x').name.suffix('_z')"),
                "Categorical": ("cat", "pl.col('c').cat.get_categories()")}
_DURATION_PARTS = {"seconds": "total_seconds", "minutes": "total_minutes", "hours": "total_hours",
                   "days": "total_days", "milliseconds": "total_milliseconds", "microseconds": "total_microseconds"}
# 10-01 00:3x (twice): df['SlotUtc'].dt.date.alias('day') -- a method named, never called.
_UNCALLED = re.compile(r"AttributeError: '(?:function|method|builtin_function_or_method)' object has no attribute "
                       r"'(\w+)'")
# 10-01 04:1x: rows.with_columns(pl.cut('ny_min', breaks=bins)) -- an expression method looked up on the module.
_POLARS_MODULE = re.compile(r"AttributeError: module 'polars' has no attribute '(\w+)'")
# 10-01 02:4x: pl.col('date') >= pl.Date(2024, 7, 19) -- pl.Date is a dtype, not a date.
_DTYPE_AS_VALUE = re.compile(r"TypeError: (Date|Time)\(\) takes no arguments")
# 10-01 02:4x: pl.col('t') >= pl.col('t').min() + 60 -- a number added to a timestamp.
_DATETIME_PLUS_NUMBER = re.compile(r"[-+] not allowed on (?:datetime|date)\S* and (?:dyn int|dyn float|[iuf]\d+)|"
                                   r"[-+] not allowed on (?:dyn int|dyn float|[iuf]\d+) and (?:datetime|date)")
# 10-01 05:3x: .rolling_quantile(...).over() -- an empty over().
_EMPTY_OVER = re.compile(r"At least one of `partition_by` and `order_by` must be specified in `over`")
# 10-01 04:3x / 15:xx: f"{score:.3f}" with score a dict (ft.quick_score) -- a container into a number format.
_FORMAT_CONTAINER = re.compile(r"unsupported format string passed to (dict|list|tuple|NoneType|Series|DataFrame|"
                               r"numpy\.ndarray|Expr)\.__format__")
# 10-01 05:2x: comp_z_prev = comp_z.to_numpy(); comp_z_prev[0] = ... -- a read-only view of a column.
_READ_ONLY = re.compile(r"ValueError: assignment destination is read-only")
# 10-01 00:1x: 'top10% n=%d' % (...) -- a bare % in a %-format string.
_PERCENT_FORMAT = re.compile(r"unsupported format character '.' \(0x[0-9a-f]+\) at index \d+")
# 10-01 03:4x: s.filter(pl.col(c).is_not_null()) on a Series; ft.trend_exits(pl.col(...) > 0, ...) -- an
# expression where data was needed.
_EXPR_AS_DATA = re.compile(r"Series constructor called with unsupported type 'Expr'|"
                           r"argument must be [^\n]*, not 'Expr'")
# 10-01 04:2x: (ts.dt.hour() * 3600 + ...) -- dt.hour() is Int8, 3600 does not fit.
_INT_OVERFLOW = re.compile(r"OverflowError: number too large to fit in target type")
# 10-01 04:2x: pl.col('Bsa_SessionCum').rank().cast(pl.UInt16) -- a strict cast to a type too small.
_NARROW_CAST = re.compile(r"conversion from `([iuf]\d+)` to `([iu]\d+)` failed in column '([^']+)'")
# 10-01 00:5x: bars['x'].rolling(60, min_periods=1) assigned without a statistic.
_BARE_WINDOW = re.compile(r"object of type '(Rolling|Expanding|ExponentialMovingWindow)' has no len\(\)|"
                          r"'(Rolling|Expanding|ExponentialMovingWindow)' object (?:is not|has no attribute "
                          r"'(?:astype|values|to_numpy|shift|diff)')")
# 10-01 01:4x: bars_fc.at[i, 'agree_long'] = False into a column created as 0.0 / NaN.
_BOOL_INTO_NUMBERS = re.compile(r"Invalid value '(True|False)' for dtype '(float|int)\w*'")
# 10-01 05:1x: active_bars.dt.date on a column that is not a datetime (object / text / numbers).
_DT_ON_NON_DATETIME = re.compile(r"Can only use \.dt accessor with datetimelike values")
# 10-01 05:1x: bars.index[i].date() after reset_index -- the index is row numbers.
_INT_AS_TIME = re.compile(r"AttributeError: '(?:int|numpy\.int64)' object has no attribute "
                          r"'(date|time|hour|minute|year|month|day|weekday|strftime|tz_localize|tz_convert|normalize)'")
# 10-01 05:16: bars = df.resample('15min', on='SlotUtc').agg({...}); bars['SlotUtc'] -- the time is now the INDEX.
_KEY_ERROR_NAME = re.compile(r"KeyError: \"?'(\w+)'")
# 10-01 (library smoke test): np.cumsum((df['t'].diff().dt.total_seconds() > 300).to_numpy()) -- the first diff
# is null, so the array holds None: "unsupported operand type(s) for +: 'NoneType' and 'bool'" from numpy.
_NONE_IN_NUMPY = re.compile(r"unsupported operand type\(s\) for [-+*/]: '(?:NoneType' and '(?:bool|int|float)|"
                            r"(?:bool|int|float)' and 'NoneType)'")
_NUMPY_FRAME = re.compile(r'File "[^"]*numpy[/\\]')


def _alias_of(code: str, col: str) -> bool:
    return bool(re.search(r"""\.alias\(\s*['"]""" + re.escape(col) + r"""['"]\s*\)""", code or ""))


def _wrong_slot_hint(stderr: str) -> str:
    m = _WRONG_SLOT.search(stderr)
    frames = _POLARS_FRAME.findall(stderr[:m.start()]) if m else []
    if not m or not frames:
        return ""
    import inspect

    import polars as pl

    fn = frames[-1]
    method = getattr(pl.Expr, fn, None) or getattr(pl.Series, fn, None)
    if method is None:
        return ""
    try:
        sig = str(inspect.signature(method)).replace("self, ", "").replace("'", "")
    except (TypeError, ValueError):
        return ""
    param = m.group("param") or m.group("arg")
    slot = f" its {param!r} parameter got a {m.group('got')}" if param else f" an argument got a {m.group('got')}"
    return (f"polars .{fn}:{slot} -- an argument passed by POSITION landed in the wrong slot (polars' order is "
            f"not pandas'). Pass everything after the first by keyword: .{fn}{sig}")


# Two forecast features merged keep the same names (t, fc_median, ...), so pandas makes fc_median_x /
# fc_median_y and 'fc_median' is gone -- in submissions for weeks, in run_python too (10-01 14:31).
_FORECAST_CLASH = re.compile(r"KeyError: \"?'(fc_\w+)'|unable to find column \"(fc_\w+)\"|\"(fc_\w+)\" not found")


# 10-01 16:32: (pl.col('Close').shift(-h) - pl.col('Close')) / pl.col('Close') * 10000 <newline> .alias(...) -- the
# .alias binds to the number 10000, not to the expression.
_ALIAS_ON_NUMBER = re.compile(r"AttributeError: '(int|float)' object has no attribute '(alias|over|cast|round|abs|fill_null)'")


def _misc_hint(stderr: str, code: str = "") -> str:
    stderr = stderr or ""
    m = _ALIAS_ON_NUMBER.search(stderr)
    if m:
        return (f".{m.group(2)}(...) attached to a plain NUMBER: a method binds tighter than * / + -, so "
                f"`a / b * 10000 .{m.group(2)}(...)` calls it on 10000. Put the whole expression in parentheses: "
                f"((pl.col('Close').shift(-h) - pl.col('Close')) / pl.col('Close') * 10000).{m.group(2)}(...)")
    m = _FORECAST_CLASH.search(stderr)
    if m:
        return (f"{m.group(1) or m.group(2) or m.group(3)!r} is missing: every forecast feature has the same column names (t, "
                "fc_median, fc_q10, ...), so after merging two of them pandas renames them fc_median_x / fc_median_y. "
                'Load each with a prefix -- ft.load("fc_x", prefix="x_") gives x_fc_median -- and merge those')
    hint = _review_feature_hint(stderr)
    if hint:
        return hint
    m = _LENGTH_CHANGE.search(stderr)
    if m:
        name, got, want = m.group(1), int(m.group(2)), int(m.group(3))
        return (f"{name!r} came out with {got} rows for a frame of {want}: every expression in with_columns must "
                f"keep the frame's length, so .drop_nulls() / .filter() / .unique() / .head() / .tail() / .slice() do "
                f"not belong inside one (shift(-n) already leaves nulls at the end -- keep them). Make the column, "
                f"then drop rows from the whole frame: df.with_columns(pl.col('Close').shift(-30).alias('fwd')"
                f").drop_nulls('fwd')")
    hint = _wrong_slot_hint(stderr)
    if hint:
        return hint
    if _TUPLE_BY_NAME.search(stderr):
        return ("that value is a TUPLE, not a dict or frame -- index it by position or unpack it: "
                "session, minute = ft.clock(rows['t'])")
    m = _ARRAY_LENGTHS.search(stderr)
    if m:
        a, b = (int(x) for x in (m.groups()[:2] if m.group(1) else m.groups()[2:]))
        return (f"two arrays of different lengths ({a} and {b}, {abs(a - b)} apart) were combined -- usually a "
                f"shifted slice taken twice: fwd = close[h:] - close[:-h] is ALREADY n-h long, so pair it with x[:-h], "
                f"not fwd[:-h]. Keeping full length avoids it: fwd = np.r_[close[h:] - close[:-h], [np.nan] * h], "
                f"then drop the NaNs from both together")
    m = _DUP_WITH_COLUMNS.search(stderr)
    if m:
        c = m.group(1)
        return (f"two expressions in one with_columns are both named {c!r}: without .alias an expression keeps the "
                f"name of the column it starts from, so (pl.col('{c}').shift(-6) - pl.col('{c}')) and the -30 one "
                f"collide -- and on its own each would REPLACE {c!r}. Name each: .alias('fwd6'), .alias('fwd30')")
    if _TEXT_NUMBER_FORMAT.search(stderr):
        return ("a TEXT value reached a number format (:.2f / :.4f) -- e.g. f\"{lo if lo else '-inf':.2f}\" formats the "
                "string '-inf'. Use numbers on both sides (float('-inf')), or format first: "
                "(f\"{x:.2f}\" if x is not None else 'n/a')")
    if _TIMED_OUT.search(stderr):
        return ("the script ran out of time -- almost always a Python for-loop over every row (~500k): replace it with "
                "column maths -- polars expressions (.shift / .rolling_* / .cum_sum, per session with .over('session')) "
                "or numpy on whole arrays; ft.trend_exits / ft.noise_area_breakout handle entries, stops and exits "
                "without a loop. Test on rows.head(20000) first")
    if _LSTSQ_NAN.search(stderr):
        return ("np.linalg.lstsq / np.polyfit got NaN or inf (rolling warm-up, shift, a missing value) -- the SVD "
                "cannot converge on them. Keep only finite rows of X and y together first: ok = np.isfinite(X).all(1) & "
                "np.isfinite(y); np.linalg.lstsq(X[ok], y[ok], rcond=None) -- and skip a window with fewer rows than "
                "columns")
    m = _LENGTH_VALUES.search(stderr)
    if m:
        got, want = int(m.group(1)), int(m.group(2))
        why = ("exactly twice the frame -- typically two Series with DIFFERENT indexes combined (a * b, a + b): pandas "
               "lines them up on the UNION of their labels, e.g. one indexed per bar and one by the day of each bar. "
               if got == 2 * want else "")
        return (f"{got} values for {want} rows: {why}Bring both to the same index first (b.reindex(a.index), or "
                "work in .to_numpy() arrays of equal length), and check len() of each piece before assigning")
    if _TIMESTAMP_DIV.search(stderr):
        return ("Timestamps cannot be divided. Subtract them to get a Timedelta, then divide by a unit: "
                "(t1 - t0) / pd.Timedelta('1min') -- or for minutes of the day use ft.clock(rows['t'])")
    if _WINDOW_LENGTH.search(stderr):
        return ("an expression inside .over('session') must give ONE value per row of the session (or a single "
                "value): .head(n) / .tail(n) / .filter / .drop_nulls / .unique change the length. For the close at "
                "bar 30 of each session use pl.col('Close').gather(29).over('session'); for the first n rows use "
                "pl.int_range(pl.len()).over('session') < n as a condition")
    if _SPEC_CONDITIONAL.search(stderr):
        return ("everything after ':' in an f-string field is the FORMAT, so f\"{x:.4f if x else 'N/A'}\" is not a "
                "conditional. Put the condition outside: (f\"{x:.4f}\" if x is not None else 'N/A'), or "
                "f\"{x if x is None else round(x, 4)}\"")
    m = _MASK_LENGTH.search(stderr)
    if m:
        return (f"a True/False mask of {m.group(2)} rows was used on an array of {m.group(1)} -- usually a mask built "
                "over ALL rows applied to one session's slice (r = x[mask]; r[valid]). Build both from the same rows: "
                "sel = mask & valid; out[sel] = x[sel] ... And never assign through two indexes -- out[mask][valid] = v "
                "writes into a COPY and changes nothing; use out[mask & valid] = v. Faster still: per-session work "
                "in polars with .over('session')")
    if _SHIFT_BY_EXPR.search(stderr):
        return ("shift(n) / head(n) / tail(n) take ONE number for the whole column, not a column of numbers. For a "
                "per-session value use an aggregate: the session's last close is pl.col('Close').last().over('session'); "
                "the close k bars ahead is pl.col('Close').shift(-k).over('session')")
    if _DURATION_FILL.search(stderr):
        return ("the difference of two dates/datetimes is a Duration, so fill_null(0) does not fit it. For a session "
                "counter compare instead: (pl.col('date') != pl.col('date').shift()).fill_null(True).cum_sum(), or use "
                ".over('session') / ft.clock(rows['t']) for per-session work; for a gap use .dt.total_days()")
    m = _NAMESPACE_CALL.search(stderr)
    if m:
        ns = m.group(1).lower()
        return (f"pl.col(...).{ns} is a namespace of methods, not a function -- write pl.col('x').{ns}.<method>(...). "
                "A per-session value is an aggregate with .over: pl.col('x').last().over('session') (or .first(), "
                ".max(), .mean())")
    m = _DUPLICATE_NAME.search(stderr)
    if m:
        # 10-01 11:40: select(pl.col(c).quantile(0.1), pl.col(c).quantile(0.9)) -- both keep the name c.
        c = m.group(1)
        return (f"two expressions in one select / with_columns both produce a column named {c!r} -- an expression "
                f"keeps its input's name, so pl.col('{c}').quantile(0.1) and pl.col('{c}').quantile(0.9) clash; give "
                f"each its own name: pl.col('{c}').quantile(0.1).alias('{c}_q10'). In ft.rows_pl(columns=[...]) / "
                "ft.load_pl, list each column once")
    if _NONE_TO_NUMBER.search(stderr) or (_NONE_IN_NUMPY.search(stderr) and _NUMPY_FRAME.search(stderr)):
        return ("a null became None in numpy -- polars comparisons are null where an input is null (rolling warm-up, "
                "shift, a missing value); fill first: pl.col('x').fill_null(False) / .fill_null(0), then .to_numpy()")
    m = _MISSING_COLUMN.search(stderr) or _MISSING_COLUMN_FT.search(stderr) or _NOT_FOUND.search(stderr)
    if m and _alias_of(code, m.group(1)):
        # 10-01 12:16: (pl.col('a') - pl.col('m')) / pl.col('s').alias('pb_z') -- the alias names only s, and
        # the result kept a's name, quietly overwriting column a.
        c = m.group(1)
        return (f"the script writes .alias('{c}') but no column {c!r} was made: .alias binds to the expression "
                f"right before it, so `(a - b) / c.alias('{c}')` renames only c, and the result keeps a's name -- "
                f"overwriting column a. Wrap the whole calculation: ((a - b) / c).alias('{c}')")
    listed = m.group(2) if m and m.re.groups >= 2 else None
    if m and m.group(1) == "t" and listed is not None and re.search(r'\bSlotUtc\b', listed):
        # 10-01 03:16: ft.load_pl('sql_exports_dbo_gexbar10s', columns=['t', ...]) -- 't' is the rows' name for it.
        return ("this frame is a raw dataset, whose time column is 'SlotUtc' -- 't' is the name ft.rows() / "
                "ft.rows_pl() give it. Use pl.col('SlotUtc') here, or load the task rows: ft.rows_pl(columns=[...])")
    if listed is not None and listed.count(",") < 40 and not listed.rstrip().endswith("..."):
        # the frame holds a chosen subset, not the whole dataset
        return (f"this frame holds only the columns listed -- if {m.group(1)!r} is a dataset column, add it to "
                f"ft.rows_pl(columns=[...]) / ft.load_pl(..., columns=[...]); if you made it, check the .alias() on "
                "the expression that should create it")
    return _more_hint(stderr, code)


def _review_feature_hint(stderr: str) -> str:
    """A trade-review feature name (trade_book.py's X_vs_price_bps, minutes_into_session, ...) asked
    of the rows as if it were a column: what it is made of, as a polars expression; "" otherwise."""
    names = [g for m in _MISSING_NAMES.finditer(stderr) for g in m.groups() if g]
    for m in _ROWS_HAVE_NO.finditer(stderr):        # ft.rows(columns=...): "the rows have no 'a' (did you mean ..); 'b'"
        names += re.findall(r"'([^']+)'", re.sub(r"\([^)]*\)", "", m.group(1)))
    for name in names:
        f = _REVIEW_FEATURE.fullmatch(name)
        if not f:
            continue
        if name.endswith("_vs_price_bps"):
            expr = f"(pl.col('{f.group(1)}') / pl.col('Close') - 1) * 1e4"
        elif name == "minutes_into_session":
            expr = "(pl.col('t') - pl.col('t').first().over('session')).dt.total_minutes()"
        elif name.startswith("price_chg_"):
            expr = (f"(pl.col('Close') / pl.col('Close').shift(n).over('session') - 1) * 1e4  # n = rows in "
                    f"{f.group(2)} minutes")
        elif name.startswith("price_since_open"):
            expr = "(pl.col('Close') / pl.col('Close').first().over('session') - 1) * 1e4"
        else:
            expr = ("(pl.col('Close') - pl.col('Close').cum_min().over('session')) / (pl.col('Close').cum_max()"
                    ".over('session') - pl.col('Close').cum_min().over('session'))")
        return (f"{name!r} is a feature the TRADE REVIEW computes, not a column of the rows -- load the columns it is "
                f"made of and compute it: rows.with_columns(({expr}).alias('{name}'))")
    return ""


def _plain_signature(fn) -> str:
    """`fn`'s parameters without annotations -- (window_size, weights=None, *, min_samples=None) --
    or "" when it has no inspectable signature."""
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return ""
    parts, star = [], False
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.kind is p.VAR_POSITIONAL:
            parts.append(f"*{p.name}")
            star = True
            continue
        if p.kind is p.VAR_KEYWORD:
            parts.append(f"**{p.name}")
            continue
        if p.kind is p.KEYWORD_ONLY and not star:
            parts.append("*")
            star = True
        parts.append(p.name if p.default is p.empty else f"{p.name}={p.default!r}")
    out = f"({', '.join(parts)})"
    return out if len(out) <= 260 else out[:257] + "...)"


def _call_hint(stderr: str) -> str:
    """A polars method (or pandas' merge) called with arguments it does not take: the specific fix
    for the habits seen, else its real signature; "" when the library cannot be told."""
    m = _BAD_CALL.search(stderr)
    if not m:
        return ""
    import polars as pl

    cls, fn, kw = m.group("cls"), m.group("fn"), m.group("kw")
    if cls == "DataFrame" and fn == "merge" and kw in ("tolerance", "direction", "allow_exact_matches"):
        return (f"{kw}= belongs to pd.merge_asof, not .merge: pd.merge_asof(left.sort_values('t'), right.sort_values("
                "'t'), on='t', direction='backward', tolerance=pd.Timedelta('20min')) -- both sides sorted by the key")
    if cls == "Expr" and fn == "rolling":
        return ("pl.col(...).rolling(...) is a TIME-window group (it needs index_column/period), not pandas' "
                ".rolling(n): a statistic over the last n rows is .rolling_mean(n) / .rolling_std(n) / .rolling_max(n) "
                "/ .rolling_quantile(q, window_size=n), with min_samples=k for a shorter warm-up -- per session add "
                ".over('session')")
    if cls in ("Expr", "Series") and fn in ("max", "min") and kw is None:
        bound, what = ("lower_bound", "floor") if fn == "max" else ("upper_bound", "cap")
        return (f".{fn}() takes no value: it is the column's own {fn}imum, one number. A {what} per row is "
                f".clip({bound}=1e-9); the row-wise {fn} of two columns is pl.{fn}_horizontal(a, b)")
    owners = {"Expr": [pl.Expr], "LazyFrame": [pl.LazyFrame], "Series": [pl.Series], "DataFrame": [pl.DataFrame]}[cls]
    if cls in ("Series", "DataFrame"):
        try:                                       # the control plane may run without pandas
            import pandas as pd

            owners.append(getattr(pd, cls))
        except ImportError:
            pass
    owners = [o for o in owners if callable(getattr(o, fn, None))]
    if kw is not None and len(owners) == 2:
        # both libraries have the method: the one whose signature takes the keyword is the one NOT in use
        takes = [o for o in owners if re.search(rf"\b{kw}\b", _plain_signature(getattr(o, fn)))]
        if len(takes) != 1:
            return ""
        owners = [o for o in owners if o is not takes[0]]
    if len(owners) != 1:
        return ""
    owner = owners[0]
    lib = "polars" if owner.__module__.startswith("polars") else "pandas"
    sig = _plain_signature(getattr(owner, fn))
    if not sig:
        return ""
    if kw is None:
        return f"{lib} .{fn}() was called with the wrong arguments -- it is .{fn}{sig}; pass options by keyword"
    if kw in _KW_ALIASES and _KW_ALIASES[kw] is None:
        fix = f"{lib} never changes a frame in place -- drop {kw}= and assign the result: df = df.{fn}(...)"
    elif _KW_ALIASES.get(kw) and re.search(rf"\b{_KW_ALIASES[kw]}\b", sig):
        fix = f"{kw}= is spelled {_KW_ALIASES[kw]}= here" + (
            " -- and means the opposite: descending=True for largest first" if kw == "ascending" else "")
    else:
        import difflib

        near = difflib.get_close_matches(kw, re.findall(r"(\w+)(?:=|,|\))", sig), n=2, cutoff=0.6)
        fix = f"it has no {kw}=" + (f" -- did you mean {' or '.join(n + '=' for n in near)}?" if near else "")
    return f"{lib} .{fn}(): {fix}. Its parameters: .{fn}{sig}"


def _namespace_hint(stderr: str) -> str:
    """A polars namespace (.dt / .str / .list ...) given a column method or a pandas name; "" otherwise."""
    m = _NS_NO_ATTR.search(stderr)
    if not m or m.group(1) not in _NS_ACCESSOR:
        return ""
    import polars as pl

    ns, attr = m.group(1), m.group(2)
    acc, example = _NS_ACCESSOR[ns]
    if ns == "DateTime" and attr in _DURATION_PARTS:
        field = attr[:-1]
        return (f"for a DURATION (a difference of two times) the length is .dt.{_DURATION_PARTS[attr]}(), e.g. "
                f"pl.col('t').diff().dt.{_DURATION_PARTS[attr]}() -- .dt.{field}() (Python's suggestion) is the "
                f"{field} FIELD of a timestamp, not a length")
    if ns == "DateTime" and attr in ("with_time_zone", "tz_localize", "tz_convert", "tz"):
        return ("polars time zones: .dt.replace_time_zone('UTC') labels naive times (the rows' t is naive UTC), "
                ".dt.convert_time_zone('America/New_York') then gives New York clock time -- or ft.clock(rows['t']) "
                "for (session, minute of day)")
    if attr == "cast":
        return (f".{acc} is a namespace, so .{acc}.cast does not exist -- cast the column itself: "
                f"pl.col('x').cast(pl.Date) (for a datetime's date: pl.col('t').dt.date())")
    if hasattr(pl.Expr, attr):
        return (f".{acc} on its own is a namespace of methods, not a column -- end it with one of them before "
                f".{attr}: {example}.{attr}(...)")
    return ""


def _more_hint(stderr: str, code: str = "") -> str:
    """The hints of the 10-01 archive sweep: failures that each had one recognisable cause and no hint."""
    hint = _call_hint(stderr) or _namespace_hint(stderr)
    if hint:
        return hint
    m = _DURATION_UNIT.search(stderr)
    if m:
        return (f"polars durations are not pandas': minutes are 'm' (and months 'mo') -- every='15m', not "
                f"'15{m.group(1)}'; hours '1h', days '1d', seconds '10s'")
    if _TRUTH_VALUE.search(stderr):
        return ("a whole column / expression / array was used where Python needs ONE True or False: `and`, `or`, "
                "`not`, `if col:`, or a chain like `a == b & c != 0` (& binds tighter, so that is a == (b & c) != 0). "
                "Use & | ~ with each comparison in parentheses -- (pl.col('a') == pl.col('b')) & (pl.col('c') != 0) "
                "-- and pl.when(cond).then(x).otherwise(y) / np.where(cond, x, y) instead of if")
    m = _UNCALLED.search(stderr)
    if m:
        return (f"a method was named without calling it -- pl.col('t').dt.date.{m.group(1)}(...) asks the METHOD "
                f"dt.date for .{m.group(1)}; add the parentheses: pl.col('t').dt.date().{m.group(1)}(...)")
    m = _POLARS_MODULE.search(stderr)
    if m:
        import polars as pl

        name = m.group(1)
        if hasattr(pl.Expr, name):
            return (f"pl.{name} is not a function -- .{name} is a method of a column expression: "
                    f"pl.col('x').{name}(...)")
        return ""
    if _DTYPE_AS_VALUE.search(stderr):
        return ("pl.Date / pl.Time are data TYPES, not values -- a date literal is datetime.date(2024, 7, 19) (or "
                "pl.date(2024, 7, 19)): pl.col('t').dt.date() >= datetime.date(2024, 7, 19)")
    if _DATETIME_PLUS_NUMBER.search(stderr):
        return ("a plain number was added to a timestamp -- polars does not know its unit. Add a duration: "
                "pl.col('t') + pl.duration(minutes=60) (datetime.timedelta works too); to skip the first n ROWS of "
                "each session use pl.int_range(pl.len()).over('session') >= n")
    if _EMPTY_OVER.search(stderr):
        return (".over() needs the column(s) to group by: .over('session') for per-session values -- or drop .over() "
                "to compute over the whole column")
    m = _FORMAT_CONTAINER.search(stderr)
    if m:
        kind = m.group(1)
        if kind == "dict":
            return ("a DICT went into a number format (:.3f) -- e.g. ft.quick_score(...) returns {'sharpe': ..., "
                    "'trades_per_day': ...}: format one of its values, f\"{score['sharpe']:.3f}\"")
        if kind == "NoneType":
            return ("None went into a number format (:.3f) -- the value came from a function that returned nothing "
                    "(no `return`) or a lookup that found nothing; check it, or (f\"{x:.3f}\" if x is not None else "
                    "'n/a')")
        return (f"a whole {kind} went into a number format (:.3f) -- reduce it to one number first: .mean(), .sum(), "
                ".item(), x[-1] / .iloc[-1]")
    if _READ_ONLY.search(stderr):
        return ("that numpy array is a READ-ONLY view of a column (.to_numpy() / .values) -- copy it before writing "
                "into it: a = s.to_numpy().copy()")
    if _PERCENT_FORMAT.search(stderr):
        return ("a '%' in a '...' % (...) string that is not a placeholder -- a literal percent sign is written %%: "
                "'top 10%% n=%d' % n; or use an f-string: f'top 10% n={n}'")
    if _EXPR_AS_DATA.search(stderr):
        return ("a polars EXPRESSION (pl.col(...)...) went where DATA is needed -- a Series method, an ft helper, "
                "numpy or float(). Evaluate it on the frame first: df.select(expr).to_series(), or add it with "
                "df.with_columns(expr.alias('x')) and pass df['x']; a Series filters with a Series: "
                "s.filter(s.is_not_null())")
    if _INT_OVERFLOW.search(stderr) and _POLARS_FRAME.search(stderr):
        return ("a polars integer column is too small for the result -- .dt.hour() / .dt.minute() / .dt.second() are "
                "Int8, so hour * 3600 does not fit. Cast first: pl.col('t').dt.hour().cast(pl.Int32) * 3600 (minute "
                "of day in New York: ft.clock(rows['t']))")
    m = _NARROW_CAST.search(stderr)
    if m:
        return (f"{m.group(3)!r} has values that do not fit {m.group(2)} -- .cast() is strict: cast to a wider type "
                f"(pl.Int64 / pl.Float64); ranks and counts over all rows need at least 32 bits")
    if _BARE_WINDOW.search(stderr):
        return ("s.rolling(n) is a WINDOW, not values -- finish it with a statistic: s.rolling(60, min_periods=1)"
                ".mean() / .std() / .max() / .quantile(0.9)")
    m = _BOOL_INTO_NUMBERS.search(stderr)
    if m:
        return (f"a {m.group(1)} was written into a column that holds numbers (created as 0.0 / NaN) -- create it as "
                f"booleans (df['x'] = False) or write 1.0 / 0.0; faster, build the whole column at once: "
                f"df['x'] = np.where(cond, 1.0, 0.0)")
    if _DT_ON_NON_DATETIME.search(stderr):
        return ("that pandas column is not a datetime (text, numbers or objects) -- convert it first: "
                "s = pd.to_datetime(s), then s.dt.date / s.dt.hour")
    m = _INT_AS_TIME.search(stderr)
    if m:
        return (f"that is a plain int where a timestamp was expected -- usually df.index[i] after reset_index() (the "
                f"index is row NUMBERS) or a time already turned into epoch ns. Take the time from its column: "
                f"df['t'].iloc[i].{m.group(1)}()")
    m = _KEY_ERROR_NAME.search(stderr)
    if m:
        c = re.escape(m.group(1))
        moved = re.search(rf"""(?:set_index\(\s*\[?|groupby\(\s*\[?|resample\([^)]*\bon\s*=\s*)['"]{c}['"]""",
                          code or "")
        if moved and not re.search(r"reset_index\(\s*\)|as_index\s*=\s*False", code[moved.end():]):
            return (f"{m.group(1)!r} is no longer a column: .set_index('{m.group(1)}') / .groupby('{m.group(1)}')... "
                    f"/ .resample(..., on='{m.group(1)}') make it the INDEX of the result. Add .reset_index() after "
                    f"that step (or read it as df.index)")
    return ""


def _uncalled_reporter(code: str) -> str | None:
    """The function holding the script's ft.report_* call when nothing at top level ever reaches
    it -- `def main(): ... ft.report_positions(pos)` with no `main()` call (#1770, #1745, #1744
    ended "no positions reported" so) -- else None."""
    import ast

    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return None
    defs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    def names(nodes) -> set[str]:
        return {x.id for n in nodes for x in ast.walk(n) if isinstance(x, ast.Name)}

    def reports(nodes) -> bool:
        return any(isinstance(x, ast.Attribute) and x.attr.startswith("report_") for n in nodes for x in ast.walk(n))

    top = [n for n in tree.body if n.__class__.__name__ not in ("FunctionDef", "AsyncFunctionDef")]
    if reports(top):
        return None
    seen, todo = set(), [d for d in names(top) if d in defs]
    while todo:
        d = todo.pop()
        if d not in seen:
            seen.add(d)
            todo += [x for x in names([defs[d]]) if x in defs and x not in seen]
    return next((d for d, n in defs.items() if d not in seen and reports([n])), None)


def _nothing_reported(what: str, call: str, code: str) -> str:
    fn = _uncalled_reporter(code)
    if fn:
        return (f"no {what} reported -- the script defines {fn}() with the ft.report_* call inside but never calls "
                f"it: add `{fn}()` at the end of the script")
    return f"no {what} reported -- call {call}"


HARNESS = """import sys
sys.path.insert(0, "/work/.ft")
try:
    import ft as _ft
    _ft._pandas_compat()
except Exception:
    pass
_src = open("/work/.ft/candidate.py", encoding="utf-8").read()
exec(compile(_src, "candidate.py", "exec"), {"__name__": "__main__", "__file__": "candidate.py"})
"""


_MEMBER_REF = re.compile(r"candidate_positions\(\s*['\"]?#?(\d+)")


def member_seqs(code: str) -> list[int]:
    """Candidate numbers a script runs through ft.candidate_positions(n), in order of appearance."""
    return list(dict.fromkeys(int(m) for m in _MEMBER_REF.findall(code or "")))


def member_files(oid: str, code: str) -> dict[str, str]:
    """{".ft/members/<seq>.py": source} for every VERIFIED candidate the script (or one of those
    candidates, in turn) runs through ft.candidate_positions. A number that is not verified --
    unscored, look-ahead failed or pending, disqualified, an ensemble -- is left out, so the
    script fails with ft's message naming it instead of building on it."""
    wanted, out = member_seqs(code), {}
    seen: set[int] = set()
    while wanted:
        seq = wanted.pop(0)
        if seq in seen or len(seen) >= 24:
            continue
        seen.add(seq)
        with _lock:
            row = db().execute(
                "SELECT code FROM candidates WHERE objective_id=? AND seq=? AND status='ok' AND mode != 'ensemble' "
                "AND lookahead = 'pass' AND audit != 'fail'", (oid, seq)).fetchone()
        if row and row["code"]:
            out[f".ft/members/{seq}.py"] = row["code"]
            wanted += [s for s in member_seqs(row["code"]) if s not in seen]
    return out


async def _run(code: str, data_dir: str, catalog: list[dict], mirror: dict | None, timeout_s: int,
               obj: dict | None = None, cut: str | None = None, extra_files: dict[str, str] | None = None,
               task_dir: Path | None = None, task_datasets: list[dict] | None = None) -> dict:
    """`cut` also truncates the objective's forecast features (None = full). The project's
    code library is copied in as /work/.ft/lib (``from lib import x``), and the research
    library's Python listings as /work/.ft/research (``from research.<doc> import <module>``).

    With `task_dir` (a task objective) the ONLY data mounted is that folder, at /task: the
    task server's rows, already cut for a look-ahead run. The project's data folder is not
    mounted, so the code cannot read around the cut. `task_datasets` (an agent's experiment
    only, never a scored run) adds single in-sample files to that: catalog entries with a
    `root` and the `host` file mounted read-only at <root>/<path>."""
    entries = [{k: c[k] for k in ("view", "path", "format")} for c in catalog]
    mounts = _mounts(data_dir, mirror)
    if task_dir is not None:
        entries, mounts = [], [(str(task_dir), "/task")]
        for d in task_datasets or []:
            entries.append({k: d[k] for k in ("view", "path", "format", "root")})
            mounts.append((d["host"], f"{d['root'].rstrip('/')}/{d['path']}"))
    elif obj is not None:
        fdir = await asyncio.to_thread(features_dir, obj, cut)
        if fdir:
            entries += _feature_catalog(obj["id"])
            mounts.append((fdir, "/features"))
    from .library import module_files
    from .research import code_files as research_files

    files = {
        ".ft/ft.py": FT_HELPER.read_text(encoding="utf-8"),
        ".ft/catalog.json": json.dumps(entries),
        ".ft/candidate.py": code,
        **(module_files(obj["project_id"]) if obj is not None else {}),
        **(research_files(obj["project_id"]) if obj is not None else {}),
        **(member_files(obj["id"], code) if obj is not None else {}),
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
                           obj: dict, cut: str | None = None, requested_by: str | None = None,
                           task_datasets: list[dict] | None = None) -> dict:
    """_run, building any forecast the script asked for with ft.forecast() and running it again.
    `task_datasets`: see _run (an agent's experiment on a task objective only).

    A script that calls ft.forecast(...) for a recipe not built yet stops with ForecastPending;
    the recipe is built here -- causally, by the loaded forecaster, named by the hash of the
    recipe so the same call finds it next time -- and the script is re-run. At most
    MAX_AUTO_FORECASTS new forecasts per call; what could not be built is said in stderr."""
    if T.is_task(obj):
        # A task objective: the task server's rows (up to `cut`) are the only data; no forecasts.
        folder = await T.export_dir(obj, cut)
        return await _run(code, data_dir, [], None, timeout_s, obj, None, task_dir=folder, task_datasets=task_datasets)
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


def _kept_positions(oid: str, cid: str) -> Path:
    """Where a scored candidate's positions are kept for the day chart. The sandbox prunes its
    run folders within a few evaluations, and the candidate row stores only daily returns."""
    return WORK_ROOT / oid / "positions" / f"{cid}.parquet"


def _keep_positions(oid: str, cid: str, src: Path, collapse: bool = True, dst: Path | None = None) -> Path:
    """Copy reported positions aside, keeping only the bars where the position changes: the
    as-of join that prices them reads the same position at every bar, and a strategy that holds
    for minutes shrinks from every bar of the dataset to a few thousand rows. `collapse=False`
    keeps every row -- task actions that are events (orders), where a repeat is a new order.
    `dst` keeps them elsewhere (a replay on another data source)."""
    dst = dst or _kept_positions(oid, cid)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    if not collapse:
        shutil.copyfile(src, tmp)
        tmp.replace(dst)
        return dst
    con = duckdb.connect(":memory:")
    try:
        con.execute("SET TimeZone = 'UTC'")
        con.execute(f"""
            COPY (
                SELECT t, pos FROM (
                    SELECT CAST(t AS TIMESTAMP) AS t, CAST(pos AS DOUBLE) AS pos,
                           lag(CAST(pos AS DOUBLE)) OVER (ORDER BY t) AS prev, row_number() OVER (ORDER BY t) AS n
                    FROM read_parquet('{src.as_posix()}')
                ) WHERE n = 1 OR pos IS DISTINCT FROM prev ORDER BY t
            ) TO '{tmp.as_posix()}' (FORMAT parquet)""")
    finally:
        con.close()
    tmp.replace(dst)
    return dst


def pos_bounds(m: dict) -> tuple[float, float]:
    """(lowest, highest) position the harness honours: |position| up to max_leverage, and only
    the sides the objective allows -- a long-only objective floors positions at 0 (flat), a
    short-only one caps them at 0."""
    lev = float(m.get("max_leverage") or 1.0)
    d = m.get("direction") or "both"
    return (0.0 if d == "long" else -lev), (0.0 if d == "short" else lev)


def pos_clip(m: dict, col: str = "pos") -> str:
    """SQL: the reported position clipped to pos_bounds, NULL as flat."""
    lo, hi = pos_bounds(m)
    return f"greatest({lo}, least({hi}, coalesce(CAST({col} AS DOUBLE), 0)))"


def analysis_price(obj: dict) -> str | None:
    """The price column the analysis tools (decile plots, field scan, regime maps) measure moves
    against: the objective's price column, or -- for a task objective -- the target its project
    chose. Scoring never uses this: a task objective is valued by its data/action MCP alone."""
    m = obj.get("metric") or {}
    if m.get("price_column"):
        return m["price_column"]
    t = m.get("target") if m.get("kind") == "task" else None
    return t if t and not any(ch in t for ch in "\"'\\;") and t.isprintable() else None


def held_sql(m: dict) -> str:
    """SQL over temp tables px (t, p) and pos (t, pos): every priced bar with the position in
    force over it -- the latest one reported at or before the bar (an as-of join). Under an
    intraday objective the position is forced flat at each day's last bar, so nothing is held
    from one day's close into the next day: every trade opens and closes on the same day, and
    one still wanted the next morning is entered again (and pays for it) at that day's bars."""
    q = "coalesce(pos.pos, 0)"
    if m.get("intraday"):
        q = f"CASE WHEN px.t = max(px.t) OVER (PARTITION BY CAST(px.t AS DATE)) THEN 0 ELSE {q} END"
    return f"SELECT px.t AS t, px.p AS p, {q} AS pos FROM px ASOF LEFT JOIN pos ON px.t >= pos.t"


def _trade_table(obj: dict, data_dir: str, positions: Path) -> dict[str, np.ndarray]:
    """Every trade the kept positions make, and how much of each day they spent in the market.

    A trade is a run of one direction: it opens when the position leaves flat (or flips side)
    and closes when it returns to flat (or flips); resizing along the way stays the same trade.
    Its gross return compounds the bars it held, priced exactly as _mark_to_market prices them;
    its net return also pays cost_bps on every unit it traded -- the entry, each resize, the exit
    (a flip's exit belongs to the closing trade, its entry to the opening one). A trade still
    open when the data ends has paid no exit.

    Cached beside the positions and rebuilt when they are newer: one full pass over the dataset."""
    cache = positions.with_name(positions.stem + (".intraday" if obj["metric"].get("intraday") else "") + ".trades.npz")
    if cache.is_file() and cache.stat().st_mtime >= positions.stat().st_mtime:
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    m = obj["metric"]
    item = next((i for i in datasource.catalog(data_dir) if obj["dataset"] in (i["view"], i["path"])), None)
    if item is None:
        raise HTTPException(status_code=400, detail=f"dataset {obj['dataset']!r} is gone from the data folder")
    tc, pc = obj["time_column"], m["price_column"]
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
            SELECT CAST(t AS TIMESTAMP) AS t, {pos_clip(m)} AS pos
            FROM read_parquet('{positions.as_posix()}')""")
        a = con.execute(f"SELECT epoch(t) AS t, p, pos AS q FROM ({held_sql(m)}) ORDER BY t").fetchnumpy()
    finally:
        con.close()
    t, p, q = (np.asarray(a[k], dtype=np.float64) for k in ("t", "p", "q"))
    n = len(t)
    sign = np.sign(q)
    qp = np.r_[0.0, q[:-1]]
    sp = np.sign(qp)
    opens = (sign != 0) & (sign != sp)
    tid = np.where(sign != 0, np.cumsum(opens) - 1, -1)
    held = np.r_[-1, tid[:-1]]  # the trade whose position is held over the bar ending at i
    ntr = int(opens.sum())
    g = np.zeros(n)
    g[1:] = q[:-1] * (p[1:] / p[:-1] - 1)
    h = held >= 0
    logs = np.zeros(ntr)
    np.add.at(logs, held[h], np.log1p(np.maximum(g[h], -0.999999)))
    turn = np.zeros(ntr)
    same = (sign == sp) & (sign != 0)
    resize = same & (q != qp)
    np.add.at(turn, tid[resize], np.abs(q[resize] - qp[resize]))
    exits = ~same & (qp != 0)
    np.add.at(turn, held[exits], np.abs(qp[exits]))
    entries = ~same & (q != 0)
    np.add.at(turn, tid[entries], np.abs(q[entries]))
    size = np.zeros(ntr)
    np.maximum.at(size, tid[sign != 0], np.abs(q[sign != 0]))
    first = np.flatnonzero(opens)
    last = np.flatnonzero((tid >= 0) & (np.r_[tid[1:], -1] != tid))
    out = {
        "entry_t": t[first],
        "exit_t": t[np.minimum(last + 1, n - 1)],
        "side": sign[first],
        "size": size,
        "gross": np.expm1(logs),
        "net": np.expm1(logs) - cost * turn,
        "open": last + 1 >= n,
        # Bars per UTC day, and how many of them held a position.
        "day": np.unique(t // 86400),
    }
    d = (t // 86400).astype(np.int64)
    base = int(d.min()) if n else 0
    out["day_bars"] = np.bincount(d - base)[(out["day"] - base).astype(np.int64)] if n else np.zeros(0)
    out["day_in"] = np.bincount(d - base, weights=(sign != 0).astype(np.float64))[(out["day"] - base).astype(np.int64)] if n else np.zeros(0)
    tmp = cache.with_name(cache.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **out)
    tmp.replace(cache)
    return out


def _utc_day(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


def _trade_calendar(obj: dict, data_dir: str, positions: Path) -> dict:
    """Per UTC day: the trades opened that day (a trade belongs to the day it opened, however
    long it runs), how many won net of costs, long and short (and how many of each won), the best and worst, the average
    hold, and the share of the day's bars spent in the market. Plus the same over every trade."""
    tr = _trade_table(obj, data_dir, positions)
    days: dict[str, dict] = {
        _utc_day(float(dn) * 86400): {"trades": 0, "wins": 0, "long": 0, "short": 0, "long_wins": 0,
                                      "short_wins": 0, "exposure": round(float(i) / float(b), 4) if b else 0.0}
        for dn, b, i in zip(tr["day"], tr["day_bars"], tr["day_in"])
    }
    net, side, hold = tr["net"], tr["side"], tr["exit_t"] - tr["entry_t"]
    for k in range(len(net)):
        dd = days.setdefault(_utc_day(float(tr["entry_t"][k])), {"trades": 0, "wins": 0, "long": 0, "short": 0,
                                                                  "long_wins": 0, "short_wins": 0, "exposure": 0.0})
        r = float(net[k])
        dd["trades"] += 1
        dd["wins"] += r > 0
        dd["long" if side[k] > 0 else "short"] += 1
        dd["long_wins" if side[k] > 0 else "short_wins"] += r > 0
        dd["best"] = max(dd.get("best", r), r)
        dd["worst"] = min(dd.get("worst", r), r)
        dd["hold_sum"] = dd.get("hold_sum", 0.0) + float(hold[k])
    for dd in days.values():
        if dd["trades"]:
            dd["hold_s"] = round(dd.pop("hold_sum") / dd["trades"], 1)
            dd["best"], dd["worst"] = round(dd["best"], 6), round(dd["worst"], 6)
    wins, losses = net[net > 0], net[net <= 0]
    longs = side > 0

    def rate(mask: np.ndarray) -> float | None:
        return round(float((net[mask] > 0).mean()), 4) if mask.any() else None

    summary = {
        "trades": int(len(net)),
        "win_rate": rate(np.ones(len(net), bool)),
        "long": int(longs.sum()),
        "short": int((~longs).sum()),
        "long_win_rate": rate(longs),
        "short_win_rate": rate(~longs),
        "avg_win": round(float(wins.mean()), 6) if len(wins) else None,
        "avg_loss": round(float(losses.mean()), 6) if len(losses) else None,
        "profit_factor": round(float(wins.sum() / -losses.sum()), 3) if len(losses) and losses.sum() < 0 else None,
        "expectancy": round(float(net.mean()), 6) if len(net) else None,
        "avg_hold_s": round(float(hold.mean()), 1) if len(net) else None,
    }
    return {"days": days, "summary": summary, "cost_bps": float(obj["metric"].get("cost_bps") or 0.0)}


def _day_trades(obj: dict, data_dir: str, positions: Path, day: str) -> list[dict]:
    """The trades that were open at any point in one UTC day, including one carried in from the
    day before (``carried``) or still open at the day's end."""
    tr = _trade_table(obj, data_dir, positions)
    lo = float(np.datetime64(day, "s").astype(np.int64))  # UTC midnight
    hi = lo + 86400
    k = np.flatnonzero((tr["entry_t"] < hi) & (tr["exit_t"] > lo))
    return [{"entry_t": float(tr["entry_t"][i]), "exit_t": float(tr["exit_t"][i]), "side": int(tr["side"][i]),
             "size": round(float(tr["size"][i]), 4), "gross": round(float(tr["gross"][i]), 6),
             "net": round(float(tr["net"][i]), 6), "carried": bool(tr["entry_t"][i] < lo),
             "open": bool(tr["open"][i])} for i in k]


# The OHLC columns a dataset may carry, matched by exact name: "Pinning_BandLow" is not a low.
_OHLC_NAMES = {"open": "o", "high": "h", "low": "l"}
# Candle widths for the day chart, finest first: the finest that keeps a day under _DAY_CANDLES.
_DAY_BUCKETS_S = (10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600)
_DAY_CANDLES = 400


def _day_bars(obj: dict, data_dir: str, positions: Path, day: str) -> dict:
    """One calendar day (UTC, as the daily returns are dated) of the dataset's prices with the
    candidate's positions over them.

    Candles are the dataset's bars merged into the finest width that keeps the day readable.
    Positions stay at bar resolution, as spans: the position decided at bar t is held from t to
    the next bar (exactly how _mark_to_market prices it), and a span's return is that position
    times the price move over it, before costs. Positions reported before the day still count --
    the as-of join carries the overnight position into the first bar."""
    from .deciplot import dataset_columns

    m = obj["metric"]
    item = next((i for i in datasource.catalog(data_dir) if obj["dataset"] in (i["view"], i["path"])), None)
    if item is None:
        raise HTTPException(status_code=400, detail=f"dataset {obj['dataset']!r} is gone from the data folder")
    tc, pc = obj["time_column"], m["price_column"]
    lev = float(m.get("max_leverage") or 1.0)
    names = {n.lower(): n for n, _ in dataset_columns(data_dir, obj["dataset"])}
    ohlc = {alias: names[k] for k, alias in _OHLC_NAMES.items() if k in names}
    has_ohlc = len(ohlc) == 3
    extra = "".join(f', TRY_CAST("{col}" AS DOUBLE) AS {alias}' for alias, col in ohlc.items()) if has_ohlc else ""
    agg = ", avg(o) AS o, max(h) AS h, min(l) AS l" if has_ohlc else ""
    con = duckdb.connect(":memory:")
    try:
        con.execute("SET TimeZone = 'UTC'")
        con.execute(f"""
            CREATE TEMP TABLE px AS
            SELECT t, avg(p) AS p{agg} FROM (
                SELECT TRY_CAST("{tc}" AS TIMESTAMP) AS t, TRY_CAST("{pc}" AS DOUBLE) AS p{extra}
                FROM {_reader(item)}('{_abs(data_dir, item)}')
            ) WHERE t >= DATE '{day}' AND t < DATE '{day}' + INTERVAL 1 DAY AND p IS NOT NULL AND p > 0
            GROUP BY t""")
        con.execute(f"""
            CREATE TEMP TABLE pos AS
            SELECT CAST(t AS TIMESTAMP) AS t, {pos_clip(m)} AS pos
            FROM read_parquet('{positions.as_posix()}') WHERE CAST(t AS TIMESTAMP) < DATE '{day}' + INTERVAL 1 DAY""")
        bars = con.execute(f"SELECT epoch(t), p, pos FROM ({held_sql(m)}) ORDER BY t").fetchall()
        if not bars:
            return {"day": day, "bars": 0}
        first, last = bars[0][0], bars[-1][0]
        bucket = next((b for b in _DAY_BUCKETS_S if (last - first) / b <= _DAY_CANDLES), _DAY_BUCKETS_S[-1])
        # Open is the first price in the bucket, close the last; high/low over the dataset's own
        # high/low when it has them, over the price when it does not.
        o, h, lo = ("arg_min(o, t)", "max(h)", "min(l)") if has_ohlc else ("arg_min(p, t)", "max(p)", "min(p)")
        candles = con.execute(f"""
            SELECT epoch(time_bucket(INTERVAL {bucket} SECOND, t)) AS b, {o}, {h}, {lo}, arg_max(p, t)
            FROM px GROUP BY b ORDER BY b""").fetchall()
    finally:
        con.close()

    step = min((b - a for (a, *_), (b, *_) in zip(bars, bars[1:]) if b > a), default=bucket)
    # Runs of one position; a run ends where the next begins. The last runs to the end of the
    # day's last bar, its return up to that bar's price (what it made overnight is tomorrow's).
    spans: list[dict] = []
    for t, p, q in bars:
        if spans and spans[-1]["pos"] == q:
            continue
        if spans:
            spans[-1].update(to=t, exit=p)
        spans.append({"from": t, "pos": q, "entry": p})
    spans[-1].update(to=last + step, exit=bars[-1][1])
    for s in spans:
        s["ret"] = s["pos"] * (s.pop("exit") / s.pop("entry") - 1)
    return {
        "day": day,
        "bars": len(bars),
        "bar_s": step,
        "bucket_s": bucket,
        "ohlc": has_ohlc,
        "price": pc,
        "max_leverage": lev,
        "candles": [[c[0], *(None if v is None else round(v, 6) for v in c[1:])] for c in candles],
        "spans": spans,
        "changes": len(spans) - 1,
    }


def _regime_segments(obj: dict, report: dict, meta: dict, returns: list[list]) -> dict | None:
    """Which regime (and so which routed signal) each day was in, for the equity chart.

    The script records bar-level labels (ft.route does it automatically, ft.report_regime
    explicitly); a day takes the label it spent the most bars in. Per-label stats split the
    daily returns by that label, in-sample and holdout apart, so it shows at a glance whether
    each regime's signal kept working after the split. Display only -- nothing here scores."""
    import polars as pl

    path = Path(report["run_dir"]) / ".ft" / "regime.parquet"
    if not path.is_file():
        return None
    df = pl.read_parquet(path)
    if df.is_empty() or "label" not in df.columns:
        return None
    df = df.with_columns(pl.col("t").cast(pl.Datetime("us")).dt.strftime("%Y-%m-%d").alias("d"),
                         pl.col("label").cast(pl.String))
    # The label the day spent the most bars in (ties: the first seen, which is stable).
    counts = df.group_by("d", "label", maintain_order=True).len()
    day_label = dict(counts.sort("len", descending=True, maintain_order=True).unique("d", keep="first")
                     .sort("d").select("d", "label").iter_rows())
    out: dict[str, Any] = {"name": meta.get("name") or "regime", "routes": meta.get("routes") or {},
                           "days": [[d, str(v)] for d, v in day_label.items()]}
    if "value" in df.columns:
        vals = (df.with_columns(pl.col("value").cast(pl.Float64, strict=False).fill_nan(None))
                .group_by("d").agg(pl.col("value").mean()).drop_nulls("value").sort("d"))
        if vals.height:
            out["signal"] = [[d, round(float(v), 6)] for d, v in vals.iter_rows()]
    ppy = float(obj["metric"].get("periods_per_year") or 252)
    split = obj.get("split_date")
    by: dict[str, dict] = {}
    for d, r in returns:
        lab = day_label.get(d)
        if lab is None:
            continue
        seg = "holdout" if split and d >= split else "in_sample"
        by.setdefault(str(lab), {}).setdefault(seg, []).append(r)
    out["by_label"] = {lab: {seg: {k: _stats(rs, ppy).get(k) for k in ("days", "sharpe", "total_return")}
                             for seg, rs in segs.items()} for lab, segs in by.items()}
    return out


# Swing capture for positions objectives. Task servers (mcp/) carry their own copy: the control
# plane never imports server code, it only talks to servers over MCP.
def _zigzag(p: np.ndarray, thr: float) -> list[tuple[int, int, int]]:
    """Swing legs (start, end, +1 up / -1 down) of one day's prices: a pivot is confirmed when the
    price reverses at least `thr` from the running extreme."""
    n = len(p)
    legs: list[tuple[int, int, int]] = []
    if n < 2:
        return legs
    d, piv, ext, lo, hi = 0, 0, 0, 0, 0
    for i in range(1, n):
        x = p[i]
        if d == 0:
            lo = i if x < p[lo] else lo
            hi = i if x > p[hi] else hi
            if x >= p[lo] * (1 + thr):
                d, piv, ext = 1, lo, i
            elif x <= p[hi] * (1 - thr):
                d, piv, ext = -1, hi, i
        elif d == 1:
            if x > p[ext]:
                ext = i
            elif x <= p[ext] * (1 - thr):
                legs.append((piv, ext, 1))
                d, piv, ext = -1, ext, i
        else:
            if x < p[ext]:
                ext = i
            elif x >= p[ext] * (1 + thr):
                legs.append((piv, ext, -1))
                d, piv, ext = 1, ext, i
    if d:
        legs.append((piv, ext, d))
    return legs


def _swings(t: np.ndarray, p: np.ndarray, pos: np.ndarray, thr: float,
           split_t: float | None) -> dict[str, dict[str, Any]]:
    """How well positions sat on the right side of the price's swings, per segment.

    `t` is epoch seconds (UTC), `p` the price, `pos` the position DECIDED at each row -- the one
    held over the following row, which is how it is priced. The prices are cut into swing legs
    day by day (`zigzag`). Over a leg's rows the alignment is the mean of sign(position) x the
    leg's direction, from -1 (always the wrong side) to +1 (always the right side), flat counting
    0. A leg is a HIT when alignment is at least 0.5 -- long through most of an up leg, short
    through most of a down leg -- a MISS at -0.5 or below, and neither when mostly flat or mixed.
    `net` is hits minus misses, `net_per_leg` that over all legs, and `capture` the alignment
    weighted by each leg's move. Riding the drift (always long) nets about zero here, so this
    rewards reading the turns, both ways. Segments: in_sample / holdout split at `split_t`
    (epoch seconds), or one `full` segment without a split."""
    out: dict[str, dict[str, Any]] = {}
    if len(t) < 2:
        return out
    held = np.sign(np.r_[0.0, pos[:-1]])            # the position earning row i was set at i-1
    cs = np.r_[0.0, np.cumsum(held)]
    day = (t // 86400).astype(np.int64)
    cuts = np.flatnonzero(np.diff(day)) + 1
    legs: list[tuple[int, int, int]] = []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(t)]):
        legs += [(a + i, a + j, d) for i, j, d in _zigzag(p[a:b], thr)]
    if not legs:
        return out
    L = np.array(legs)
    i, j, d = L[:, 0], L[:, 1], L[:, 2]
    align = (cs[j + 1] - cs[i + 1]) / np.maximum(1, j - i) * d   # rows i+1..j earn the leg
    move = np.abs(p[j] / p[i] - 1)
    segs = ({"in_sample": t[i] < split_t, "holdout": t[i] >= split_t} if split_t
            else {"full": np.ones(len(L), bool)})
    for name, k in segs.items():
        if not k.any():
            continue
        hit, miss, up = align[k] >= 0.5, align[k] <= -0.5, d[k] > 0
        n = int(k.sum())
        out[name] = {
            "legs": n, "up_legs": int(up.sum()), "down_legs": int((~up).sum()),
            "up_caught_long": int((hit & up).sum()), "down_caught_short": int((hit & ~up).sum()),
            "up_while_short": int((miss & up).sum()), "down_while_long": int((miss & ~up).sum()),
            "hits": int(hit.sum()), "misses": int(miss.sum()), "net": int(hit.sum() - miss.sum()),
            "net_per_leg": round(float((hit.sum() - miss.sum()) / n), 4),
            "capture": round(float((align[k] * move[k]).sum() / move[k].sum()), 4) if move[k].sum() else None,
            "legs_per_day": round(n / max(1, len(np.unique(day[i[k]]))), 2),
        }
    return out


def _mark_to_market(obj: dict, data_dir: str, positions: Path) -> tuple[list[list], dict]:
    """Daily returns of the reported positions, from the dataset's own prices.

    On the dataset's bar grid (every distinct timestamp with a price), the position in force
    at bar t is the latest one reported at or before t (an as-of join). The return realised at
    bar t is pos[t-1] * (price[t] / price[t-1] - 1), less cost_bps on |pos[t-1] - pos[t-2]| --
    the trade made at the previous bar. Returns are compounded per calendar day. Under an
    intraday objective the position is flat at each day's last bar (see held_sql).

    Two more daily series ride along in ``info`` (popped by the caller, never stored there):
    ``_gross`` -- the same positions without costs -- and ``_inverted`` -- every position's sign
    flipped, same costs -- and ``_swings`` (see _swings). From them evaluate() says whether a loser lacks an edge, pays too much
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
            SELECT CAST(t AS TIMESTAMP) AS t, {pos_clip(m)} AS pos
            FROM read_parquet('{positions.as_posix()}')""")
        n_pos, n_changes = con.execute(
            "SELECT count(*), count(*) FILTER (WHERE pos IS DISTINCT FROM prev) FROM "
            "(SELECT pos, lag(pos) OVER (ORDER BY t) AS prev FROM pos)").fetchone()
        rows = con.execute(f"""
            WITH g AS ({held_sql(m)}), l AS (
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
        # Trades opened on each side (a flip opens one), in-sample only: the holdout stays hidden.
        split = obj.get("split_date")
        sides = con.execute(f"""
            WITH g AS ({held_sql(m)}),
                 e AS (SELECT t, sign(pos) AS s, sign(lag(pos) OVER (ORDER BY t)) AS ps FROM g)
            SELECT count(*) FILTER (WHERE s > 0 AND s IS DISTINCT FROM ps),
                   count(*) FILTER (WHERE s < 0 AND s IS DISTINCT FROM ps)
            FROM e {f"WHERE t < DATE '{split}'" if split else ""}""").fetchone()
        # The swing legs the positions sat on the right or wrong side of (_swings).
        g = con.execute(f"SELECT epoch(t) AS t, p, pos FROM ({held_sql(m)}) ORDER BY t").fetchnumpy()
        split_t = float(np.datetime64(split, "s").astype(np.int64)) if split else None
        swings = _swings(np.asarray(g["t"], dtype=np.float64), np.asarray(g["p"], dtype=np.float64),
                         np.asarray(g["pos"], dtype=np.float64), float(m.get("swing_pct") or 0.25) / 100.0, split_t)
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
            "cost_bps": m.get("cost_bps"), "max_leverage": lev, "direction": m.get("direction") or "both",
            "intraday": bool(m.get("intraday")), "price_column": pc,
            "sides": {"long": int(sides[0] or 0), "short": int(sides[1] or 0)},
            "positions_in_data_range": span[0],
            "positions_from": str(span[1]) if span[1] is not None else None,
            "positions_to": str(span[2]) if span[2] is not None else None,
            "data_from": str(bars[1]) if bars[1] is not None else None,
            "data_to": str(bars[2]) if bars[2] is not None else None}
    ok = [x for x in rows if all(v is not None and math.isfinite(v) for v in x[1:])]
    info["_gross"] = [[d, float(g)] for d, _, g, _ in ok]
    info["_swings"] = swings
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


_LOAD_RX = re.compile(r"""ft\.(?:load|load_pl|path)\(\s*(?:name\s*=\s*)?(['"])([^'"]+)\1""")
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
        if len(literal) != src.count("ft.load(") + src.count("ft.load_pl(") + src.count("ft.path("):
            return None  # a load with a computed name
        names.update(n for _, n in literal)
    return names or None


async def _lookahead(obj: dict, code: str, data_dir: str, catalog: list[dict], positions_mode: bool,
                     full_pos: Path | None, returns: list[list], seed: Any = None,
                     concurrency: int = LOOKAHEAD_CONCURRENCY, progress: dict | None = None) -> tuple[str, str]:
    """Run the look-ahead test; (verdict, detail). Verdict: pass | fail | error.

    `progress`, if given, is kept current as {"cuts_done", "cuts_total"} for a progress bar."""
    if T.is_task(obj):
        if full_pos is None:
            return "error", "no actions to compare"

        async def run(c: str, folder: Path) -> dict:
            return await _run(c, data_dir, [], None, obj["eval_timeout_s"], obj, None, task_dir=folder)

        return await T.lookahead(obj, code, full_pos, run, seed=seed, concurrency=concurrency, progress=progress)
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


async def evaluate(obj: dict, req: Submit, rerun: dict | None = None) -> dict:
    """Score a submission. With `rerun` (a stored candidate), score that candidate again in
    place -- same number, same code -- instead of adding a new one."""
    project = projects.get(obj["project_id"])
    if project is None:
        raise HTTPException(status_code=404, detail="the objective's project no longer exists")
    kind = obj["metric"]["kind"]
    if kind != "judge" and not req.code.strip():
        raise HTTPException(status_code=400, detail="submit a complete Python script in `code`")

    cid = rerun["id"] if rerun else uuid.uuid4().hex[:10]
    now = time.time()
    if rerun:
        seq = rerun["seq"]
        # Everything the last attempt produced goes; the operator is asking for a clean run.
        _update_candidate(cid, {"status": "evaluating", "score": None, "is_score": None, "score_note": "",
                                "metrics": "{}", "returns": "[]", "lookahead": "skipped", "lookahead_detail": "",
                                "audit": "none", "audit_notes": "", "stdout": "", "stderr": "",
                                "eval_seconds": None, "champion_at": None})
    else:
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
    if not rerun:
        record_usage(obj["project_id"], cid, req.code)
    fields: dict[str, Any] = {}
    deferred: tuple | None = None
    t0 = time.time()
    try:
        async with _EVAL_SLOTS:
            if req.code.strip():
                full = await _run_forecasting(req.code, data_dir, catalog, None, obj["eval_timeout_s"], obj, None,
                                              requested_by=f"candidate #{seq}")
                fields.update(stdout=full["stdout"][-20_000:], stderr=full["stderr"][-20_000:], run_id=full["run_id"])
                res = full["result"]
                if not full["ok"]:
                    fields.update(status="error", score_note=_failure_note(full["stderr"], req.code))
                elif kind == "task":
                    acts = T.actions_file(full)
                    ev = await T.evaluate_actions(obj, acts) if acts else {}
                    if acts is None:
                        fields.update(status="error", score_note=_nothing_reported(
                            "actions", "ft.report_actions(series indexed by the rows' t column)", req.code))
                    elif ev.get("problem"):
                        fields.update(status="error", score_note=str(ev["problem"])[:2000])
                    else:
                        score, is_score, note, metrics, curve = T.score(obj, ev)
                        if res.get("extra"):
                            metrics["extra"] = res["extra"]
                        if req.parent_id:
                            try:
                                parent = get_candidate(req.parent_id)
                                kind_of_change = _change_kind(parent.get("code"), req.code)
                                if kind_of_change:
                                    metrics["change"] = {"kind": kind_of_change, "parent_seq": parent["seq"]}
                            except HTTPException:
                                pass
                        twin = await asyncio.to_thread(_clone_of, obj["id"], cid, is_score, score)
                        if twin:
                            note = (f"IDENTICAL result to #{twin} (same in-sample score to 9 decimals): this is the same "
                                    f"strategy resubmitted, so it adds nothing. Change the LOGIC (entry rule, filters, "
                                    f"exits), not the wording" + (f". {note}" if note else ""))
                        fields.update(status="ok", score=score, is_score=is_score, score_note=note,
                                      metrics=json.dumps(metrics), returns=json.dumps(curve))
                        try:
                            await asyncio.to_thread(_keep_positions, obj["id"], cid, acts, T.actions_hold(obj))
                        except Exception:  # noqa: BLE001 -- keeping them must never cost the score
                            logger.exception("keeping actions failed for %s", cid)
                        if obj["lookahead_check"] and cuts(obj):
                            kept = WORK_ROOT / obj["id"] / "pending" / f"{cid}.parquet"
                            kept.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(acts, kept)
                            deferred = (data_dir, catalog, True, kept, curve)
                            fields.update(lookahead="pending",
                                          lookahead_detail="the look-ahead test is running in the background")
                elif kind in RETURN_METRICS:
                    positions_mode = bool(obj["metric"].get("price_column") and obj.get("dataset")
                                          and obj.get("time_column"))
                    full_pos = _positions_file(full)
                    returns, problem, mtm = [], None, None
                    if positions_mode:
                        if full_pos is None:
                            problem = _nothing_reported("positions", "ft.report_positions(series)", req.code)
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
                        score, is_score, note, metrics = _score_returns(obj, returns, (mtm or {}).get("sides"))
                        if res.get("extra"):
                            metrics["extra"] = res["extra"]
                        if full.get("features_used"):
                            metrics["features_used"] = full["features_used"]
                        if positions_mode:
                            try:
                                regime = _regime_segments(obj, full, res.get("regime") or {}, returns)
                                if regime:
                                    metrics["regime"] = regime
                            except Exception:  # noqa: BLE001 -- a chart must never cost the score
                                logger.exception("regime segments failed for %s", cid)
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
                            swings = mtm.pop("_swings", None)
                            if swings:
                                metrics["swings"] = {**swings, "swing_pct": float(obj["metric"].get("swing_pct") or 0.25)}
                            metrics["execution"] = mtm
                            try:
                                metrics["costs"] = _costs(obj, returns, gross, inverted,
                                                          int(mtm.get("position_changes") or 0))
                            except Exception:  # noqa: BLE001 -- a diagnostic must never cost the score
                                logger.exception("cost breakdown failed for %s", cid)
                        metrics["source"] = "positions (marked to market by the harness)" if positions_mode \
                            else "self-reported returns"
                        if positions_mode and full_pos is not None:
                            try:
                                await asyncio.to_thread(_keep_positions, obj["id"], cid, full_pos)
                            except Exception:  # noqa: BLE001 -- a chart must never cost the score
                                logger.exception("keeping positions failed for %s", cid)
                        fields.update(status="ok", score=score, is_score=is_score, score_note=note,
                                      metrics=json.dumps(metrics), returns=json.dumps(returns))
                        if obj["lookahead_check"] and cuts(obj):
                            # The look-ahead test is ten more sandbox runs -- most of an evaluation --
                            # and the agent needs none of it to carry on: its in-sample score and
                            # diagnosis are known now. So it runs in the background (_settle_lookahead)
                            # and the model goes back to work instead of idling for a minute or two.
                            # The full run's positions are copied aside: the sandbox prunes old runs.
                            kept = None
                            if full_pos is not None:
                                kept = WORK_ROOT / obj["id"] / "pending" / f"{cid}.parquet"
                                kept.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copyfile(full_pos, kept)
                            deferred = (data_dir, catalog, positions_mode, kept, returns)
                            fields.update(lookahead="pending",
                                          lookahead_detail="the look-ahead test is running in the background")
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
        why = exc.detail if isinstance(exc, HTTPException) else f"evaluation failed ({type(exc).__name__}): {exc}"
        if not rerun and str(why).startswith(SANDBOX_DOWN):
            # Docker never started the script: there is no result, so no candidate -- kept, it read
            # as the model's failure in its history and in the monitor (bug #417).
            from .library import _db as library_db

            conn = library_db()
            with _lock:
                conn.execute("DELETE FROM lib_usage WHERE candidate_id=?", (cid,))
                conn.execute("DELETE FROM candidates WHERE id=?", (cid,))
                conn.commit()
            raise
        # Keep what the run already produced. submit() only rewrites score_note on the way
        # out, so without this the agent got "evaluation failed: <harness internals>" and none
        # of its own stdout/stderr -- nothing to learn from (and an HTTPException, e.g. prices
        # that cannot be marked, left the candidate stuck at 'evaluating').
        # The harness's own traceback goes under the script's stderr: "evaluation failed: 0"
        # (a KeyError) told nobody where to look.
        import traceback

        harness = "".join(traceback.format_exception(exc))[-6000:]
        try:
            _update_candidate(cid, {**{k: fields[k] for k in ("stdout", "run_id") if k in fields},
                                    "stderr": (fields.get("stderr") or "")
                                    + "\n\n--- harness error (control plane) ---\n" + harness,
                                    "status": "error", "score_note": str(why)[:2000],
                                    "eval_seconds": round(time.time() - t0, 1)})
        except Exception:  # noqa: BLE001 -- never mask the original failure
            logger.exception("could not record the failed evaluation of %s", cid)
        raise
    fields["eval_seconds"] = round(time.time() - t0, 1)
    # Which of its trades were big winners, big losers or scratch, and what the winners had in
    # common at entry (in-sample): the agent's next change should aim at exactly that.
    review = None
    if kind == "task" and fields.get("status") == "ok":
        _update_candidate(cid, fields)
        review = await trade_book.after_scoring(obj, cid, seq)
    if deferred is not None:
        _update_candidate(cid, fields)
        _spawn(_settle_lookahead(obj, cid, seq, req.code, *deferred), f"look-ahead test of #{seq}")
        return _with_review(agent_view(obj, get_candidate(cid)), review)
    _settle(obj, cid, seq, req.code, fields)
    return _with_review(agent_view(obj, get_candidate(cid)), review)


def _with_review(view: dict, review: str | None) -> dict:
    if review:
        view["trade_review"] = review
        view["trade_review_how"] = trade_book.GOAL
    return view


def _settle(obj: dict, cid: str, seq: int | None, code: str, fields: dict) -> bool:
    """Record a scored candidate's final verdict and act on it: a contender for the title gets
    an audit (or the title), a proven leak retires its modules. Returns whether it contends."""
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
        _auto_quarantine(obj, cid, seq, code,
                         f"Look-ahead detected by the harness on #{seq}: "
                         f"{(fields.get('lookahead_detail') or '')[:1500]}")
    if contender and not obj["require_audit"]:
        _crown(obj["id"], cid)
        _auto_review(obj["id"], cid)
    return contender


# Background look-ahead tests: at most this many at once (each runs its cuts in parallel too),
# apart from the slots live submissions score in.
_LOOKAHEAD_SLOTS = asyncio.Semaphore(2)


def _spawn(coro, what: str) -> None:
    task = asyncio.create_task(coro)
    _bg.add(task)

    def _finished(t: asyncio.Task) -> None:
        _bg.discard(t)
        if not t.cancelled() and t.exception() is not None:
            logger.error("%s failed: %r", what, t.exception())

    task.add_done_callback(_finished)


async def _settle_lookahead(obj: dict, cid: str, seq: int | None, code: str, data_dir: str, catalog: list[dict],
                            positions_mode: bool, full_pos: Path | None, returns: list[list]) -> None:
    """Run a candidate's look-ahead test after its agent has moved on, then settle it."""
    try:
        async with _LOOKAHEAD_SLOTS:
            # A crash inside the look-ahead test is the harness's problem, not proof of a leak:
            # record it as an `error` verdict (which keeps the candidate off the title) instead
            # of discarding the score and output already earned.
            try:
                worst, detail = await _lookahead(obj, code, data_dir, catalog, positions_mode, full_pos, returns)
            except HTTPException as exc:
                worst, detail = "error", f"the look-ahead test could not run: {exc.detail}"
            except Exception as exc:  # noqa: BLE001
                logger.exception("look-ahead test crashed for %s", cid)
                worst, detail = "error", f"the look-ahead test could not run: {type(exc).__name__}: {exc}"
    finally:
        if full_pos is not None:
            full_pos.unlink(missing_ok=True)
    try:
        c = get_candidate(cid, light=True)
    except HTTPException:
        return                                   # deleted while it was being tested
    if c["lookahead"] != "pending":
        return                                   # re-run or re-tested meanwhile: that verdict stands
    fresh = get_objective(obj["id"])             # the current champion, not the one at submission
    contender = _settle(fresh, cid, seq, code, {"status": c["status"], "score": c["score"], "lookahead": worst,
                                                "lookahead_detail": detail[:2000]})
    tag = {"objective_id": obj["id"], "candidate_id": cid, "seq": seq, "lookahead": worst}
    if worst == "error":
        _board_post(obj["project_id"], "errors", "harness",
                    f"The look-ahead test of #{seq} could not run, so it cannot take the title: {detail[:800]}", tag)
    elif contender:
        _board_post(obj["project_id"], "results", "harness",
                    f"#{seq} passed the look-ahead test and is a contender for best"
                    + (" -- audit pending." if fresh["require_audit"] else " -- NEW BEST."), tag)


def rescore_stored(oid: str) -> dict:
    """Recompute every scored candidate's score from its stored returns under the objective's
    current ranking, then hand the title to the best candidate that is eligible to hold it.

    Nothing is re-run: the returns were marked to market at submission and are kept. The
    diagnostics evaluation added to `metrics` (costs, execution, extra) are kept too."""
    obj = get_objective(oid)
    if T.is_task(obj):
        return _rescore_task_stored(obj)
    if obj["metric"]["kind"] not in RETURN_METRICS:
        return {"rescored": 0}
    with _lock:
        rows = db().execute("SELECT id, metrics, returns FROM candidates WHERE objective_id=? AND status='ok' "
                            "AND returns != '[]'", (oid,)).fetchall()
    updates = []
    for r in rows:
        returns = json.loads(r["returns"] or "[]")
        if not returns:
            continue
        metrics = json.loads(r["metrics"] or "{}")
        score, is_score, note, fresh = _score_returns(obj, returns, (metrics.get("execution") or {}).get("sides"))
        metrics.pop("rank", None)
        metrics.pop("one_sided", None)
        metrics.update(fresh)
        updates.append((score, is_score, note, json.dumps(metrics), r["id"]))
    with _lock:
        db().executemany("UPDATE candidates SET score=?, is_score=?, score_note=?, metrics=? WHERE id=?", updates)
        db().commit()
    return {"rescored": len(updates), **_recrown(get_objective(oid))}


def _rescore_task_stored(obj: dict) -> dict:
    """A task objective's scores again from the segments its server returned (a ranking switch)."""
    with _lock:
        rows = db().execute("SELECT id, metrics FROM candidates WHERE objective_id=? AND status='ok'",
                            (obj["id"],)).fetchall()
    updates = []
    for r in rows:
        metrics = json.loads(r["metrics"] or "{}")
        t = metrics.get("task") or {}
        if not t.get("segments"):
            continue
        score, is_score, note, fresh, _ = T.score(obj, t)
        metrics.pop("rank", None)
        metrics.pop("too_few_trades", None)
        metrics.update({k: v for k, v in fresh.items() if k != "task"})
        updates.append((score, is_score, note, json.dumps(metrics), r["id"]))
    with _lock:
        db().executemany("UPDATE candidates SET score=?, is_score=?, score_note=?, metrics=? WHERE id=?", updates)
        db().commit()
    return {"rescored": len(updates), **_recrown(get_objective(obj["id"]))}


def _recrown(obj: dict) -> dict:
    """After the scores changed: crown the best candidate that may hold the title (clean look-ahead
    pass, audit passed if required), and queue an audit for the new leader if it has none yet --
    the audit hands it the title when it passes, exactly as at submission."""
    ranked = _ranked(obj["id"], _higher(obj), 200)
    la_ok = (lambda c: c["lookahead"] == "pass") if obj["lookahead_check"] and cuts(obj) else (lambda c: True)
    champ = next((c for c in ranked if la_ok(c) and (c["audit"] == "pass" or not obj["require_audit"])), None)
    if champ and champ["id"] != obj.get("best_id"):
        _crown(obj["id"], champ["id"])
    top = next((c for c in ranked if la_ok(c)), None)
    queued = None
    if obj["require_audit"] and top and top is not champ and top["audit"] == "none":
        _update_candidate(top["id"], {"audit": "pending", "audit_started": 0})
        queued = top["seq"]
    return {"champion": champ["seq"] if champ else None, "audit_queued": queued}


def migrate_ranking() -> None:
    """Objectives created before robust ranking scored on the holdout alone. Switch each one
    over once (the `rank` key marks it done) and rescore its candidates."""
    with _lock:
        objs = [_obj_row(r) for r in db().execute("SELECT * FROM objectives").fetchall()]
    for obj in objs:
        if "rank" in obj["metric"] or not obj.get("split_date"):
            continue
        metric = {**obj["metric"], "rank": "robust"}
        with _lock:
            db().execute("UPDATE objectives SET metric=? WHERE id=?", (json.dumps(metric), obj["id"]))
            db().commit()
        try:
            res = rescore_stored(obj["id"])
            logger.info("objective %s switched to robust ranking: %s", obj["id"], res)
        except Exception:  # noqa: BLE001 -- a failed migration must not stop the control plane
            logger.exception("could not rescore objective %s", obj["id"])


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
    if T.is_task(obj):
        view.update(T.agent_notes(m))
    sw = m.get("swings") or {}
    if sw.get("in_sample") or sw.get("full"):
        view["swings_in_sample"] = {
            **(sw.get("in_sample") or sw["full"]),
            "how": (f"price cut into swing legs of at least {sw.get('swing_pct', 0.25)}% within each day; a leg is a "
                    "hit when you were on its side (long up / short down) for most of it, a miss when on the wrong "
                    "side; net = hits - misses; capture = alignment weighted by leg size (-1..+1). Always-long nets "
                    "about zero here: this rewards catching the turns, both ways.")}
    sides = (m.get("execution") or {}).get("sides")
    if sides:
        view["trades_in_sample"] = sides
        if (obj["metric"].get("direction") or "both") == "both" and not sides.get("short") and sides.get("long"):
            view["one_sided"] = (m.get("one_sided") or
                                 f"long only: {sides['long']} long and 0 short trades in-sample. Shorts are allowed "
                                 "(positions may be negative, down to -max_leverage) -- add the mirrored short entry "
                                 "so the strategy also trades when the signal points down.")
    if m.get("extra"):
        view["extra"] = m["extra"]
    change = m.get("change") or {}
    if change.get("kind") in ("parameters only", "identical"):
        view["change"] = (f"This is #{change.get('parent_seq')} with only its numbers changed (thresholds, windows, "
                          "multipliers). Tuning numbers overfits the in-sample period and teaches the team nothing; "
                          "next time change the LOGIC -- a new signal, filter, regime condition or exit rule -- and "
                          "say why it should work.")
    view["lookahead"] = c.get("lookahead")
    if c.get("lookahead") == "pending":
        view["lookahead"] = ("pending -- the look-ahead test runs in the background; its verdict shows in "
                             "RECENT ATTEMPTS next iteration (a leak disqualifies the candidate then)")
    if c.get("lookahead") in ("fail", "error"):
        view["lookahead_detail"] = c.get("lookahead_detail")
    if c.get("score") is None and c.get("score_note"):
        view["not_ranked"] = c["score_note"]
    view["rank"] = f"{rank} of {len(ranked)}" if rank else "unranked"
    view["contender_for_best"] = c.get("audit") == "pending" or bool(c.get("champion_at"))
    if obj.get("split_date"):
        view["note"] = (TASK_RANK_NOTE if T.is_task(obj) else RANK_NOTE) \
            if obj["metric"].get("rank", "robust") == "robust" else (
            "Ranking uses the hidden holdout period (after the split); in-sample "
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
    # Which sides a position may take: positions the other way are held as flat when priced.
    direction: Literal["both", "long", "short"] = "both"
    # Intraday only: the harness flattens every position at each day's last bar, so no trade is
    # held overnight -- every trade opens and closes on the same day.
    intraday: bool = False
    # Both sides: with both allowed, longs and shorts must each be at least this share of a
    # candidate's in-sample trades for it to be ranked (0 = no requirement). See _side_gap.
    min_side_share: float = Field(0.0, ge=0, le=0.5)
    # Swing legs: price must reverse this many PERCENT from an extreme to confirm a pivot (_swings).
    swing_pct: float = Field(0.25, gt=0, le=20)
    # kind == "task": a registered task server (an MCP implementing the task contract) scores
    # the candidates -- see app/task_objectives.py and docs/task-servers.md.
    task_server: str | None = Field(None, max_length=200)
    task: str | None = Field(None, max_length=200)
    # The column the task is valued on, when the project chose one of the server's target_options.
    target: str | None = Field(None, max_length=200)
    # ...and the value function that ranks it, one of the server's value_functions.
    value_function: str | None = Field(None, max_length=200)
    # ...and the rule its actions are managed under (e.g. intraday / max_3_days / open).
    action_rule: str | None = Field(None, max_length=200)
    # ...and the data source its rows come from, one of the server's sources (e.g. which producer's
    # bars of the table); None = the server's default.
    source: str | None = Field(None, max_length=200)
    # What the leaderboard ranks on when there is a holdout: "robust" (weaker of in-sample and
    # holdout, times equity-curve smoothness) or "holdout" (the holdout metric alone).
    rank: Literal["robust", "holdout"] = "robust"


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
        lookahead_pending = db().execute("SELECT count(*) FROM candidates WHERE objective_id=? AND lookahead='pending'",
                                         (obj["id"],)).fetchone()[0]
    best = _best(obj)
    kind = obj["metric"]["kind"]
    return {
        **obj,
        "metric_label": METRIC_LABEL.get(kind, kind),
        "candidates": sum(counts.values()),
        "candidates_ok": counts.get("ok", 0),
        "candidates_error": counts.get("error", 0),
        "evaluating": counts.get("evaluating", 0),
        "lookahead_pending": lookahead_pending,
        "improvements": champs[0] or 0,
        "last_improvement_at": champs[1],
        "last_candidate_at": last,
        "best": _slim(best),
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
    dataset = req.dataset
    if metric["kind"] == "task":
        # A project's objectives are scored by the project's own data/action MCP.
        own = project.get("task_server")
        if own and not metric.get("task_server"):
            metric["task_server"] = own
        elif own and metric.get("task_server") != own:
            raise HTTPException(status_code=400, detail=f"this project's data/action MCP is {own!r}, not "
                                                        f"{metric.get('task_server')!r} -- change it on the Projects page")
        # ...valued on the target the project chose for this task (Projects -> Data/action MCP).
        chosen = (project.get("task_options") or {}).get(metric.get("task") or "") or {}
        if chosen.get("target") and not metric.get("target"):
            metric["target"] = chosen["target"]
        if chosen.get("value_function") and not metric.get("value_function"):
            metric["value_function"] = chosen["value_function"]
        if chosen.get("action_rule") and not metric.get("action_rule"):
            metric["action_rule"] = chosen["action_rule"]
        if chosen.get("source") and not metric.get("source"):
            metric["source"] = chosen["source"]
        # The sides trades may take: the project's choice unless the request names one (the
        # metric's "both" default must not override it, nor reach a server that offers no choice).
        if "direction" not in req.metric.model_fields_set:
            metric["direction"] = chosen.get("direction") or None
        metric, split = await T.prepare_objective(metric)
        tc, dataset = None, None
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
    if metric["kind"] == "task":
        # The MCP's in-sample rows as the objective's dataset: every analysis tool sees its schema.
        try:
            view = await T.ensure_view({"id": oid, "metric": metric, "split_date": split}, project["data_dir"])
        except HTTPException as exc:          # built on the next iteration's context instead
            logger.warning("task view of new objective %s not built yet: %s", oid, exc.detail)
            view = None
        if view:
            dataset, tc = view, "t"
    with _lock:
        db().execute(
            "INSERT INTO objectives (id, project_id, title, description, metric, status, dataset, time_column, "
            "split_date, lookahead_check, require_audit, eval_timeout_s, cooldown_s, created_at, updated_at) "
            "VALUES (?,?,?,?,?, 'running', ?,?,?,?,?,?,?,?,?)",
            (oid, project_id, req.title, req.description, json.dumps(metric), dataset, tc, split,
             int(req.lookahead_check and bool(split)), int(req.require_audit), req.eval_timeout_s,
             req.cooldown_s, now, now),
        )
        db().commit()
    return _summary(get_objective(oid))


def task_servers_of_project(project_id: str) -> set[str]:
    """Task servers the project's task objectives are scored by."""
    with _lock:
        rows = db().execute("SELECT metric FROM objectives WHERE project_id=?", (project_id,)).fetchall()
    out = set()
    for (metric,) in rows:
        try:
            m = json.loads(metric or "{}")
        except ValueError:
            continue
        if m.get("kind") == "task" and m.get("task_server"):
            out.add(m["task_server"])
    return out


@router.get("/objectives/{oid}/candidates/{cid}/actions")
async def candidate_actions(oid: str, cid: str, start: str | None = None, end: str | None = None,
                            limit: int = 500, source: str | None = None) -> dict:
    """A task candidate's managed actions between start and end, as its task server reports them
    (harness_actions): the trades or schedule it produced and the state it drove. Operator-facing.
    `source`: its replay on that data source (POST .../source-runs) instead of its scored run."""
    obj = get_objective(oid)
    if not T.is_task(obj):
        raise HTTPException(status_code=409, detail="only task objectives have a server-managed action log")
    c = get_candidate(cid, light=True)
    if c["objective_id"] != oid:
        raise HTTPException(status_code=404, detail="candidate belongs to another objective")
    if source:
        saved = _source_run_saved(oid, cid, source)
        kept = _source_run_dir(oid, source) / f"{cid}.parquet"
        if saved is None or not kept.is_file():
            raise HTTPException(status_code=404, detail=f"no replay of this candidate on source {source!r}")
        return await T.action_log({**obj, "metric": saved["metric"]}, kept, start, end, max(1, min(limit, 5000)))
    kept = _kept_positions(oid, cid)
    if not kept.is_file():
        raise HTTPException(status_code=404, detail="no actions kept for this candidate")
    return await T.action_log(obj, kept, start, end, max(1, min(limit, 5000)))


@router.get("/task-servers")
async def task_servers() -> dict:
    """Registered MCP servers that implement the task contract, with their tasks -- what a new
    task objective can be scored by (docs/task-servers.md)."""
    return await T.list_servers()


@router.get("/task-servers/{server}/tasks/{task}")
async def task_detail(server: str, task: str) -> dict:
    """One task's full description (rows, target, action, score, columns, holdout, cuts)."""
    return await T.describe(server, task)


@router.get("/task-servers/{server}/tasks/{task}/leak-scan")
async def task_leak_scan(server: str, task: str) -> dict:
    """The operator's data-timing check of a task (the server's harness_leak_scan): columns
    whose change predicts the NEXT row's target move better than the current one -- probably
    filed before they were known. Not offered to agents: it lists exactly the columns that leak."""
    return await T.call(server, "harness_leak_scan", {"task": task, "top": 15}, timeout_s=900)


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


class Like(BaseModel):
    liked: bool
    note: str = Field("", max_length=2000)


@router.post("/objectives/{oid}/candidates/{cid}/like")
async def like(oid: str, cid: str, req: Like) -> dict:
    """The operator flags a run as the kind they want (or takes the flag back). Agents see the
    liked runs and the reason in their brief, and improve iterations build on them."""
    c = get_candidate(cid, light=True)
    if c["objective_id"] != oid:
        raise HTTPException(status_code=404, detail="candidate belongs to another objective")
    _update_candidate(cid, {"liked": time.time() if req.liked else None,
                            "liked_note": req.note.strip() if req.liked else ""})
    return {"liked": req.liked}


class Ranking(BaseModel):
    rank: Literal["robust", "holdout"]


@router.post("/objectives/{oid}/ranking")
async def set_ranking(oid: str, req: Ranking) -> dict:
    """Switch what the leaderboard ranks on and rescore every candidate under it."""
    obj = get_objective(oid)
    with _lock:
        db().execute("UPDATE objectives SET metric=?, updated_at=? WHERE id=?",
                     (json.dumps({**obj["metric"], "rank": req.rank}), time.time(), oid))
        db().commit()
    return await asyncio.to_thread(rescore_stored, oid)


class Direction(BaseModel):
    direction: Literal["both", "long", "short"]


@router.post("/objectives/{oid}/direction")
async def set_direction(oid: str, req: Direction) -> dict:
    """Allow long trades, short trades or both. Agents are told from their next brief and new
    candidates are priced under it; candidates already scored keep the returns they were
    scored on (their positions are not re-run)."""
    obj = get_objective(oid)
    if not obj["metric"].get("price_column"):
        raise HTTPException(status_code=400, detail="direction applies to objectives the harness prices from positions")
    with _lock:
        db().execute("UPDATE objectives SET metric=?, updated_at=? WHERE id=?",
                     (json.dumps({**obj["metric"], "direction": req.direction}), time.time(), oid))
        db().commit()
    return _summary(get_objective(oid))


class Intraday(BaseModel):
    intraday: bool


# Objectives whose candidates are being re-marked under a changed execution rule -> progress.
_REMARKS: dict[str, dict] = {}


@router.post("/objectives/{oid}/intraday")
async def set_intraday(oid: str, req: Intraday) -> dict:
    """Trade intraday only (flat at each day's last bar) or allow holding overnight. Agents are
    told from their next brief and new candidates are priced under it. Candidates already scored
    are marked to market again under the new rule in the background, best first -- from their
    kept positions, or by running their code once more when none were kept."""
    obj = get_objective(oid)
    if not obj["metric"].get("price_column"):
        raise HTTPException(status_code=400, detail="intraday applies to objectives the harness prices from positions")
    changed = bool(obj["metric"].get("intraday")) != req.intraday
    with _lock:
        db().execute("UPDATE objectives SET metric=?, updated_at=? WHERE id=?",
                     (json.dumps({**obj["metric"], "intraday": req.intraday}), time.time(), oid))
        db().commit()
    if changed or oid not in _REMARKS:
        _spawn(_remark(oid), f"re-marking the candidates of {oid}")
    return {**_summary(get_objective(oid)), "remark": _REMARKS.get(oid)}


class SideShare(BaseModel):
    min_side_share: float = Field(..., ge=0, le=0.5)


@router.post("/objectives/{oid}/sides")
async def set_side_share(oid: str, req: SideShare) -> dict:
    """Require longs and shorts to each be at least this share of a candidate's in-sample trades
    (0 = off). Candidates already scored are re-marked in the background so their trade counts
    are measured and the rule applied, best first."""
    obj = get_objective(oid)
    if not obj["metric"].get("price_column"):
        raise HTTPException(status_code=400, detail="this applies to objectives the harness prices from positions")
    with _lock:
        db().execute("UPDATE objectives SET metric=?, updated_at=? WHERE id=?",
                     (json.dumps({**obj["metric"], "min_side_share": req.min_side_share}), time.time(), oid))
        db().commit()
    _spawn(_remark(oid), f"re-marking the candidates of {oid}")
    return {**_summary(get_objective(oid)), "remark": _REMARKS.get(oid)}


class TradeLimit(BaseModel):
    max_trades_per_day: int = Field(..., ge=0, le=200, description="0 = no limit")


@router.post("/objectives/{oid}/trade-limit")
async def set_trade_limit(oid: str, req: TradeLimit) -> dict:
    """A task objective's daily trade limit -- the task server ignores entries past it (exits and
    stops always go through). Tighten it as the swarm gets better at choosing its entries: start
    loose so signals can be found at all, end at the goal. The server must accept the rule
    (it is refused otherwise); agents are told from their next brief, and every scored candidate
    is valued again from its kept actions under the new limit, best first -- no code re-runs."""
    obj = get_objective(oid)
    if not T.is_task(obj):
        raise HTTPException(status_code=400, detail="the trade limit applies to task objectives")
    m = obj["metric"]
    info = m.get("task_info") or {}
    rule = T.with_trade_limit(m.get("action_rule") or info.get("action_rule"), req.max_trades_per_day or None)
    d = await T.describe(m["task_server"], m["task"], m.get("target"), m.get("value_function"), rule,
                         T._rule(obj).get("direction"), **T._src(m.get("source")))
    info = {**info, **{k: d.get(k) for k in ("action", "action_rule", "action_rules", "guidance", "valuation")}}
    with _lock:
        db().execute("UPDATE objectives SET metric=?, updated_at=? WHERE id=?",
                     (json.dumps({**m, "action_rule": d.get("action_rule") or rule, "task_info": info}), time.time(), oid))
        db().commit()
    _spawn(_rescore_task(oid), f"re-scoring the candidates of {oid}")
    return {**_summary(get_objective(oid)), "remark": _REMARKS.get(oid)}


class MinTrades(BaseModel):
    min_trades: int | None = Field(None, ge=0, le=100_000, description="in-sample trades in all; 0 = no floor")
    min_trades_per_day: float | None = Field(None, ge=0, le=100, description="the older daily quota; 0 = off")


@router.post("/objectives/{oid}/min-trades")
async def set_min_trades(oid: str, req: MinTrades) -> dict:
    """A task objective's trade floor: a candidate with fewer in-sample trades IN ALL than
    min_trades is not ranked (0 = off). Without it, when every strategy loses after costs the
    ranking drifts to strategies that barely trade. min_trades_per_day, the older daily quota, can
    only be cleared or set here too: it ranks only strategies that trade every day, which rules
    out the selective ones. Fields left out keep their value. Agents are told from their next
    brief; scored candidates are ranked again at once from the trade counts their server already
    reported -- nothing re-runs."""
    obj = get_objective(oid)
    if not T.is_task(obj):
        raise HTTPException(status_code=400, detail="the trade floor applies to task objectives")
    m = dict(obj["metric"])
    if req.min_trades is not None:
        m["min_trades"] = req.min_trades
    if req.min_trades_per_day is not None:
        m["min_trades_per_day"] = req.min_trades_per_day
    with _lock:
        db().execute("UPDATE objectives SET metric=?, updated_at=? WHERE id=?", (json.dumps(m), time.time(), oid))
        db().commit()
    res = await asyncio.to_thread(rescore_stored, oid)
    return {**_summary(get_objective(oid)), "rescored": res}


@router.post("/objectives/{oid}/rescore")
async def rescore(oid: str) -> dict:
    """Value every scored task candidate again from its kept actions -- after the task server's
    valuation changed (costs, fills, a new diagnostic). Progress as for re-marking."""
    obj = get_objective(oid)
    if not T.is_task(obj):
        raise HTTPException(status_code=400, detail="re-scoring from kept actions applies to task objectives")
    _spawn(_rescore_task(oid), f"re-scoring the candidates of {oid}")
    return {**_summary(get_objective(oid)), "remark": _REMARKS.get(oid)}


_TASK_METRIC_KEYS = ("task", "in_sample", "holdout", "rank", "source", "too_few_trades")


async def _rescore_task(oid: str) -> None:
    """Every scored candidate of a task objective valued again by its server from the actions kept
    when it was scored, under the objective's CURRENT choices (action rule, target, value
    function), best first; the title moves to the best eligible one. A candidate without kept
    actions is left as it was and counted as failed. A newer call supersedes a running one."""
    run = _REMARKS[oid] = {"started": time.time(), "done": 0, "failed": 0, "total": 0, "running": True}
    obj = get_objective(oid)
    trade_book.forget(oid)                  # the new rule makes new trades: the book is rebuilt below
    higher = _higher(obj)
    with _lock:
        rows = db().execute(
            "SELECT id, seq FROM candidates WHERE objective_id=? AND status='ok' "
            f"ORDER BY score IS NULL, score {'DESC' if higher else 'ASC'}", (oid,)).fetchall()
    run["total"] = len(rows)
    for k, r in enumerate(rows):
        if _REMARKS.get(oid) is not run:
            return
        try:
            obj = get_objective(oid)
            kept = _kept_positions(oid, r["id"])
            if not kept.is_file():
                raise ValueError("no kept actions")
            ev = await T.evaluate_actions(obj, kept)
            if ev.get("problem"):
                raise ValueError(str(ev["problem"]))
            score, is_score, note, fresh, curve = T.score(obj, ev)
            old = get_candidate(r["id"], light=True)["metrics"] or {}
            metrics = {**{x: v for x, v in old.items() if x not in _TASK_METRIC_KEYS}, **fresh}
            _update_candidate(r["id"], {"score": score, "is_score": is_score, "score_note": note,
                                        "metrics": json.dumps(metrics), "returns": json.dumps(curve)})
            run["done"] += 1
        except Exception as exc:  # noqa: BLE001 -- one candidate must not stop the rest
            logger.warning("re-scoring #%s failed: %s", r["seq"], getattr(exc, "detail", exc))
            run["failed"] += 1
        if k % 10 == 9 or k == len(rows) - 1:
            await asyncio.to_thread(_recrown, get_objective(oid))
    run.update(running=False, finished=time.time())
    trade_book.spawn_build(oid)


@router.get("/objectives/{oid}/remark")
async def remark_progress(oid: str) -> dict:
    return _REMARKS.get(oid) or {}


async def _remark(oid: str) -> None:
    """Mark every scored candidate to market again under the objective's current execution rule,
    best first, then hand the title to the best eligible one. A candidate whose positions cannot
    be had (its code no longer runs) is left as it was and counted as failed. A second call
    while one runs restarts it under the newest rule."""
    run = _REMARKS[oid] = {"started": time.time(), "done": 0, "failed": 0, "total": 0, "running": True}
    obj = get_objective(oid)
    higher = _higher(obj)
    with _lock:
        rows = db().execute(
            "SELECT id, seq FROM candidates WHERE objective_id=? AND status='ok' AND returns != '[]' "
            f"ORDER BY score IS NULL, score {'DESC' if higher else 'ASC'}", (oid,)).fetchall()
    run["total"] = len(rows)
    for k, r in enumerate(rows):
        if _REMARKS.get(oid) is not run:
            return                                  # superseded by a newer rule change
        try:
            obj = get_objective(oid)
            _, data_dir, kept, _ = await _candidate_positions(oid, r["id"])
            returns, mtm = await asyncio.to_thread(_mark_to_market, obj, data_dir, kept)
            if len(returns) < 10:
                raise ValueError(f"only {len(returns)} days of returns")
            score, is_score, note, fresh = _score_returns(obj, returns, mtm.get("sides"))
            c = get_candidate(r["id"], light=True)
            metrics = c["metrics"]
            metrics.pop("rank", None)
            metrics.pop("one_sided", None)
            metrics.update(fresh)
            gross, inverted = mtm.pop("_gross", []), mtm.pop("_inverted", [])
            swings = mtm.pop("_swings", None)
            if swings:
                metrics["swings"] = {**swings, "swing_pct": float(obj["metric"].get("swing_pct") or 0.25)}
            metrics["execution"] = mtm
            try:
                metrics["costs"] = _costs(obj, returns, gross, inverted, int(mtm.get("position_changes") or 0))
            except Exception:  # noqa: BLE001 -- a diagnostic must never cost the score
                logger.exception("cost breakdown failed for %s", r["id"])
            _update_candidate(r["id"], {"score": score, "is_score": is_score, "score_note": note,
                                        "metrics": json.dumps(metrics), "returns": json.dumps(returns)})
            run["done"] += 1
        except Exception as exc:  # noqa: BLE001 -- one candidate must not stop the rest
            logger.warning("re-marking #%s failed: %s", r["seq"], getattr(exc, "detail", exc))
            run["failed"] += 1
        if k % 10 == 9 or k == len(rows) - 1:
            await asyncio.to_thread(_recrown, get_objective(oid))
    run.update(running=False, finished=time.time())


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


def _slim(c: dict | None) -> dict | None:
    """A candidate for a LIST: an ensemble's daily member returns and weights (up to ~150 KB)
    are only drawn in its own view, which fetches the full candidate."""
    ens = ((c or {}).get("metrics") or {}).get("ensemble")
    if isinstance(ens, dict) and ("member_returns" in ens or "weights" in ens):
        c = {**c, "metrics": {**c["metrics"], "ensemble": {k: v for k, v in ens.items()
                                                           if k not in ("member_returns", "weights")}}}
    return c


@router.get("/objectives/{oid}/candidates")
async def list_candidates(oid: str, order: Literal["rank", "recent"] = "rank", limit: int = 50) -> dict:
    obj = get_objective(oid)
    higher = _higher(obj)
    limit = max(1, min(limit, 5000))              # the console pages through a long history 50 at a time

    def row(c: dict) -> dict:
        return {**_slim(c), "holdout_check": holdout_check(c, higher)}

    if order == "rank":
        return {"candidates": [row(c) for c in _ranked(oid, higher, limit)],
                "disqualified": [_slim(c) for c in _disqualified(oid, higher)]}
    with _lock:
        rows = db().execute(f"SELECT {_LIGHT} FROM candidates WHERE objective_id=? ORDER BY seq DESC LIMIT ?",
                            (oid, limit)).fetchall()
    return {"candidates": [row(_cand_row(r)) for r in rows]}


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
    if c.get("mode") == "ensemble":
        # Its `code` is a spec, not a script: the members run, the ensemble is arithmetic.
        raise HTTPException(status_code=409, detail="an ensemble has no script to run -- open a member to run it")
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


# -- Replay a task candidate on another data source of its task server -----------------------
# The candidate's code as it was scored, run unchanged (no agent, no training) on the full rows
# of another source -- e.g. a strategy built on the original GEX bars run on Source 2's -- and
# valued by the task server under that source. Nothing about the candidate changes: the result,
# its kept actions (for the day drill-down) and a comparison with the scored curve are saved
# under WORK_ROOT/<oid>/sources/<source>/.
_SOURCE_RUNS: dict[tuple[str, str, str], dict] = {}


def _source_run_dir(oid: str, source: str) -> Path:
    return WORK_ROOT / oid / "sources" / (re.sub(r"[^A-Za-z0-9_-]", "_", source)[:64] or "_")


def _source_run_saved(oid: str, cid: str, source: str) -> dict | None:
    try:
        out = json.loads((_source_run_dir(oid, source) / f"{cid}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return out if isinstance(out, dict) else None


def _curve_stats(curve: list, lo: str | None, hi: str | None, additive: bool) -> dict | None:
    """Days, active days, total return (or summed value), annualised Sharpe, worst drawdown and
    win rate of the daily curve between lo (inclusive) and hi (exclusive)."""
    import numpy as np

    pts = [(str(d)[:10], float(v)) for d, v in curve or [] if isinstance(v, (int, float))
           and (lo is None or str(d)[:10] >= lo) and (hi is None or str(d)[:10] < hi)]
    if not pts:
        return None
    r = np.array([v for _, v in pts])
    if additive:
        eq = np.cumsum(r)
        total, dd = float(eq[-1]), float((eq - np.maximum.accumulate(np.r_[0.0, eq])[1:]).min())
    else:
        eq = np.cumprod(1.0 + r)
        total, dd = float(eq[-1] - 1.0), float((eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1.0).min())
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    active = r[r != 0]
    return {"first": pts[0][0], "last": pts[-1][0], "days": len(r), "active_days": int(len(active)),
            "total_return": round(total, 6), "sharpe": round(float(r.mean()) / sd * 252 ** 0.5, 4) if sd > 0 else None,
            "max_drawdown": round(dd, 6), "win_rate": round(float((active > 0).mean()), 4) if len(active) else None}


def _source_comparison(obj: dict, scored: list, replay: list, additive: bool) -> dict:
    """The replay against the scored run, period by period: the dates the strategy was built on
    (before the objective's holdout), the objective's holdout, and the dates its data never had --
    plus how alike the two runs' days are where both have them."""
    import datetime as dt

    split = obj.get("split_date")
    info = obj["metric"].get("task_info") or {}
    last = str((info.get("shape") or {}).get("last") or info.get("last") or (scored[-1][0] if scored else ""))[:10]
    after = (dt.date.fromisoformat(last) + dt.timedelta(days=1)).isoformat() if last else None
    periods = []
    if split:
        periods.append({"name": "built on", "about": f"before the objective's holdout ({split}): the dates the strategy "
                                                     "was developed on", "from": None, "to": split})
        periods.append({"name": "objective holdout", "about": f"{split} to {last}: held out from the agents",
                        "from": split, "to": after})
    elif after:
        periods.append({"name": "objective's dates", "about": f"up to {last}", "from": None, "to": after})
    if after:
        periods.append({"name": "never seen", "about": f"after {last}: dates the objective's data does not have",
                        "from": after, "to": None})
    periods.append({"name": "all", "about": "every day of each run", "from": None, "to": None})
    for p in periods:
        p["scored"] = _curve_stats(scored, p["from"], p["to"], additive)
        p["replay"] = _curve_stats(replay, p["from"], p["to"], additive)
    a = {str(d)[:10]: float(v) for d, v in scored or [] if isinstance(v, (int, float))}
    b = {str(d)[:10]: float(v) for d, v in replay or [] if isinstance(v, (int, float))}
    both = sorted(set(a) & set(b))
    corr = None
    if len(both) > 10:
        import numpy as np

        x, y = np.array([a[d] for d in both]), np.array([b[d] for d in both])
        if x.std() > 0 and y.std() > 0:
            corr = round(float(np.corrcoef(x, y)[0, 1]), 4)
    return {"periods": periods, "shared_days": len(both), "daily_correlation": corr, "split": split, "last": last}


async def _source_metric(obj: dict, source: str) -> dict:
    """The objective's metric under another data source: the same target, value function,
    action rule and direction, described by the task server for that source's rows."""
    m = obj["metric"]
    d = await T.describe(m["task_server"], m["task"], m.get("target"), m.get("value_function"), m.get("action_rule"),
                         T._rule(obj).get("direction"), **T._src(source))
    return {**m, "source": source, "task_info": T.snapshot(d)}


async def _task_sources(obj: dict) -> tuple[list[dict], str | None]:
    """(the task server's data sources, the objective's own). An objective made before sources
    existed has neither in its snapshot: its own is the server's default."""
    m = obj["metric"]
    info = m.get("task_info") or {}
    if info.get("sources") is not None:
        return info["sources"] or [], m.get("source") or info.get("source")
    d = await T.describe(m["task_server"], m["task"])
    return d.get("sources") or [], m.get("source") or d.get("source")


async def _do_source_run(oid: str, cid: str, source: str, job: dict) -> None:
    try:
        obj, c = get_objective(oid), get_candidate(cid)
        project = projects.get(obj["project_id"])
        if project is None:
            raise RuntimeError("the objective's project is gone")
        job["phase"] = "asking the task server for the source"
        metric = await _source_metric(obj, source)
        other = {**obj, "metric": metric}
        job["phase"] = "exporting the source's rows"
        folder = await T.export_dir(other, None)
        # The same code over more rows takes longer: scale the objective's timeout by the size.
        own_rows = ((obj["metric"].get("task_info") or {}).get("rows")) or 0
        rows = metric["task_info"].get("rows") or own_rows
        timeout = int(min(600, obj["eval_timeout_s"] * max(1.0, rows / own_rows if own_rows else 1.0) * 1.5))
        job["phase"] = "waiting for a run slot"
        async with _EVAL_SLOTS:
            job["phase"] = f"running the code on {rows:,} rows"
            rep = await _run(c["code"], project["data_dir"], [], None, timeout, other, None, task_dir=folder)
        if not rep["ok"]:
            why = f"timed out after {timeout}s" if rep.get("timed_out") else "failed"
            raise RuntimeError(f"the script {why} on source {source}: {(rep.get('stderr') or '')[-3000:]}")
        acts = T.actions_file(rep)
        if acts is None:
            raise RuntimeError("the script reported no actions (ft.report_actions) on this source")
        job["phase"] = "valuing the actions"
        ev = await T.evaluate_actions(other, acts)
        if ev.get("problem"):
            raise RuntimeError(str(ev["problem"])[:2000])
        score, is_score, note, metrics, curve = T.score(other, ev)
        out = _source_run_dir(oid, source)
        out.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(_keep_positions, oid, cid, acts, T.actions_hold(other), out / f"{cid}.parquet")
        additive = (metrics.get("task") or {}).get("curve_kind") == "additive"
        saved = {"source": source, "candidate_id": cid, "seq": c["seq"], "finished_at": time.time(),
                 "duration_s": rep["duration_s"], "metric": metric, "score": score, "is_score": is_score, "note": note,
                 "metrics": metrics, "curve": curve, "holdout_from": metric["task_info"].get("holdout_from"),
                 "comparison": _source_comparison(obj, c.get("returns") or [], curve, additive)}
        tmp = out / f"{cid}.json.tmp"
        tmp.write_text(json.dumps(saved), encoding="utf-8")
        tmp.replace(out / f"{cid}.json")
        job.update(state="done", phase=None)
    except HTTPException as exc:
        job.update(state="failed", error=str(exc.detail))
    except Exception as exc:  # noqa: BLE001 -- reported on the job
        logger.exception("source replay of %s on %s failed", cid, source)
        job.update(state="failed", error=f"{exc}" if isinstance(exc, RuntimeError) else f"{type(exc).__name__}: {exc}")
    finally:
        job["finished_at"] = time.time()


def _source_run_view(oid: str, cid: str, source: str) -> dict | None:
    job = _SOURCE_RUNS.get((oid, cid, source))
    if job and job["state"] in ("running", "failed"):
        return {k: v for k, v in job.items() if k != "task"}
    saved = _source_run_saved(oid, cid, source)
    if saved is None:
        return None
    return {"state": "done", **{k: v for k, v in saved.items() if k != "metric"}}


@router.get("/objectives/{oid}/candidates/{cid}/source-runs")
async def candidate_source_runs(oid: str, cid: str) -> dict:
    """The task server's data sources, the objective's own, and this candidate's replay on each other one."""
    obj = get_objective(oid)
    if not T.is_task(obj):
        raise HTTPException(status_code=409, detail="only task objectives have data sources")
    c = get_candidate(cid, light=True)
    if c["objective_id"] != oid:
        raise HTTPException(status_code=404, detail="candidate belongs to another objective")
    sources, own = await _task_sources(obj)
    return {"own": own, "sources": sources, "split_date": obj.get("split_date"),
            "runs": {s["name"]: _source_run_view(oid, cid, s["name"]) for s in sources if s["name"] != own}}


class SourceRun(BaseModel):
    source: str = Field(..., min_length=1, max_length=200)


@router.post("/objectives/{oid}/candidates/{cid}/source-runs")
async def start_source_run(oid: str, cid: str, req: SourceRun) -> dict:
    """Replay the candidate's code, unchanged, on another data source and value it there (a
    background job: the export and the run take minutes; poll GET .../source-runs)."""
    obj = get_objective(oid)
    if not T.is_task(obj):
        raise HTTPException(status_code=409, detail="only task objectives have data sources")
    c = get_candidate(cid, light=True)
    if c["objective_id"] != oid:
        raise HTTPException(status_code=404, detail="candidate belongs to another objective")
    if c.get("mode") == "ensemble":
        raise HTTPException(status_code=409, detail="an ensemble has no script to run -- replay its members")
    if c.get("status") != "ok":
        raise HTTPException(status_code=409, detail="only a scored candidate can be replayed on another source")
    sources, own = await _task_sources(obj)
    names = [s["name"] for s in sources]
    if req.source not in names:
        raise HTTPException(status_code=400, detail=f"source {req.source!r} is not one of {names}")
    if req.source == own:
        raise HTTPException(status_code=400, detail="that is the objective's own source: its scored run is that one")
    key = (oid, cid, req.source)
    job = _SOURCE_RUNS.get(key)
    if not (job and job["state"] == "running"):
        job = {"state": "running", "phase": "starting", "source": req.source, "started_at": time.time(),
               "finished_at": None, "error": None}
        _SOURCE_RUNS[key] = job
        _spawn(_do_source_run(oid, cid, req.source, job), f"replaying {cid} on source {req.source}")
    return _source_run_view(oid, cid, req.source) or job


# One recovery re-run per candidate at a time, however many times its day chart is asked for.
_POSITION_RERUNS: dict[str, asyncio.Lock] = {}


@router.get("/objectives/{oid}/candidates/{cid}/day")
async def candidate_day(oid: str, cid: str, day: str) -> dict:
    """One day of the dataset's bars with the candidate's positions over them, for the equity
    chart's click-through. Positions kept at scoring are used; a candidate scored before they
    were kept is re-run once to recover them (the same run POST .../run makes), then kept."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise HTTPException(status_code=400, detail="day must be YYYY-MM-DD")
    obj, data_dir, kept, recovered = await _candidate_positions(oid, cid)
    out = await asyncio.to_thread(_day_bars, obj, data_dir, kept, day)
    if out.get("bars"):
        out["trades"] = await asyncio.to_thread(_day_trades, obj, data_dir, kept, day)
        out["cost_bps"] = float(obj["metric"].get("cost_bps") or 0.0)
    out["recovered"] = recovered
    return out


@router.get("/objectives/{oid}/candidates/{cid}/calendar")
async def candidate_calendar(oid: str, cid: str) -> dict:
    """Trade statistics per day -- trades, wins, long and short, best and worst, time in the
    market -- for the candidate's P&L calendar, and the same over all its trades."""
    obj, data_dir, kept, recovered = await _candidate_positions(oid, cid)
    out = await asyncio.to_thread(_trade_calendar, obj, data_dir, kept)
    out["recovered"] = recovered
    return out


async def _candidate_positions(oid: str, cid: str) -> tuple[dict, str, Path, str | None]:
    """The objective, its data folder and the candidate's kept positions -- recovering them
    first (once, however many requests ask) for a candidate scored before they were kept."""
    c = get_candidate(cid)
    if c["objective_id"] != oid:
        raise HTTPException(status_code=404, detail="candidate belongs to another objective")
    obj = get_objective(oid)
    if not (obj["metric"].get("price_column") and obj.get("dataset") and obj.get("time_column")):
        raise HTTPException(status_code=409, detail="this objective scores self-reported returns, not positions")
    if c.get("mode") == "ensemble":
        raise HTTPException(status_code=409, detail="an ensemble has no positions of its own -- open a member")
    project = projects.get(obj["project_id"])
    if project is None:
        raise HTTPException(status_code=404, detail="no such project")
    data_dir = project["data_dir"]
    kept = _kept_positions(oid, cid)
    recovered = None
    if not kept.is_file():
        async with _POSITION_RERUNS.setdefault(cid, asyncio.Lock()):
            if not kept.is_file():
                recovered = await _recover_positions(obj, c, data_dir)
    return obj, data_dir, kept, recovered


async def _recover_positions(obj: dict, c: dict, data_dir: str) -> str:
    """Positions for a candidate scored before they were kept: from its scoring run's folder if
    the sandbox has not pruned it, else by running its code again. Says which it was."""
    from .sandbox import RUNS_ROOT

    run_pos = RUNS_ROOT / (c.get("run_id") or "-") / ".ft" / "positions.parquet"
    if c.get("run_id") and run_pos.is_file():
        await asyncio.to_thread(_keep_positions, obj["id"], c["id"], run_pos)
        return "scoring run"
    if not c.get("code"):
        raise HTTPException(status_code=409, detail="this candidate has no code to recover its positions from")
    catalog = await asyncio.to_thread(datasource.catalog, data_dir)
    async with _EVAL_SLOTS:
        rep = await _run_forecasting(c["code"], data_dir, catalog, None, obj["eval_timeout_s"], obj, None,
                                     requested_by=f"day chart of #{c['seq']}")
    pos = _positions_file(rep) if rep["ok"] else None
    if pos is None:
        why = _failure_note(rep["stderr"], c["code"]) if not rep["ok"] else "no ft.report_positions call"
        raise HTTPException(status_code=409, detail=f"re-running the candidate gave no positions: {why}")
    await asyncio.to_thread(_keep_positions, obj["id"], c["id"], pos)
    return f"re-run ({rep['duration_s']:.0f}s)"


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
                         (f"evaluation failed ({type(exc).__name__}): {exc}"[:2000], oid, req.model,
                          time.time() - 3600))
            db().commit()
        raise HTTPException(status_code=500, detail=f"evaluation failed ({type(exc).__name__}): {exc}") from None


@router.post("/objectives/{oid}/candidates/{cid}/rerun")
async def rerun(oid: str, cid: str) -> dict:
    """The operator re-scores a candidate whose evaluation failed (a harness crash, a timeout, a
    look-ahead test that could not run) -- same number, same code, a clean run. Works while the
    objective is paused too: the point is to fix a result, not to search."""
    obj = get_objective(oid)
    c = get_candidate(cid)
    if c["objective_id"] != oid:
        raise HTTPException(status_code=404, detail="candidate belongs to another objective")
    if c["status"] == "evaluating":
        raise HTTPException(status_code=409, detail=f"#{c['seq']} is being evaluated right now")
    if c.get("mode") == "ensemble":
        raise HTTPException(status_code=409, detail="ensembles are recomputed, not re-run")
    if not (c["status"] == "error" or c.get("lookahead") == "error" or c.get("score") is None):
        raise HTTPException(status_code=409, detail=f"#{c['seq']} scored fine; only failed evaluations are re-run")
    if obj.get("best_id") == cid:
        raise HTTPException(status_code=409, detail=f"#{c['seq']} is the current best")
    req = Submit(code=c.get("code") or "", answer=c.get("answer") or "", rationale=c.get("rationale") or "",
                 parent_id=c.get("parent_id"), model=c.get("model") or "", mode=c.get("mode") or "improve",
                 idea_id=c.get("idea_id"))
    try:
        return await evaluate(obj, req, rerun=c)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 -- evaluate() already recorded it on the candidate
        raise HTTPException(status_code=500, detail=f"re-run failed ({type(exc).__name__}): {exc}") from None


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
        # ... and so do the ensembles built on it (see ensembles.disqualify_dependents).
        from . import ensembles

        ensembles.disqualify_dependents(obj, ids={cid}, why=f"Audit failed on #{c.get('seq')}: {note[:1500]}")
    return {"audit": "pass" if req.passed else "fail", "champion": crowned, "quarantined": quarantined}


AUDIT_MAX_TOKENS = 6000


def _audit_prompt(obj: dict, c: dict) -> str:
    """The same checklist the swarm's auditor gets (swarm_runner Worker.audit)."""
    m = obj["metric"]
    ho = ((c.get("metrics") or {}).get("holdout") or {}).get(m["kind"], c.get("score"))
    body = c.get("code") or c.get("answer") or ""
    if c.get("mode") == "ensemble":
        from . import ensembles

        body = ensembles.audit_body(c)   # the spec runs nothing: the members' code is what to review
    fence = "`" * 3
    return (
        "You are auditing a candidate solution before it can become the champion of this objective.\n\n"
        f"OBJECTIVE: {obj['title']}\n{obj.get('description') or ''}\n\n"
        f"It scored {ho} ({m['kind']}) on a hidden holdout and {c.get('is_score')} in-sample (ranking score "
        f"{c.get('score')}). The harness already verified mechanically that its positions do not change when "
        "future rows are removed, and it computes returns itself from prices.\n\n"
        "Check the code for problems the mechanical tests cannot catch:\n"
        "1. Hard-coded dates, prices or thresholds that encode knowledge of the specific data "
        "(e.g. 'go long after 2023-03-01', a parameter that only makes sense in hindsight).\n"
        "2. Reading files or data other than through ft.load, or touching /work/.ft/ files (importing the "
        "project library -- `from lib import x` -- or research code -- `from research.<doc> import x` -- is fine).\n"
        "3. Degenerate or fragile logic (e.g. trades only a handful of days, extreme leverage "
        "flipping, relies on a single outlier).\n"
        "4. Anything else that makes the score untrustworthy -- shown in the code, not supposed.\n\n"
        "A FAIL disqualifies the result for good, so it needs a CONCRETE defect you can point to in the code "
        "(quote the line). These are NOT defects and must not fail a candidate: ordinary parameter choices "
        "(thresholds, windows, hold times) with no sign of being fitted to specific dates; constant position "
        "size; simple exit rules; generic worries that it 'may overfit' or 'might not generalise'; thresholds "
        "written to several digits (-0.06554) or comments calling a rule 'proven' -- choosing parameters on the "
        "in-sample data is how candidates are made; variables or code that are never used and do not change the "
        "positions. The author never saw the holdout, so a parameter cannot have been tuned to it. Mention such "
        "concerns in notes and PASS. Reply with ONLY a JSON object: "
        '{"passed": true|false, "issues": ["..."], "notes": "one or two sentences"}\n\n'
        f"CODE:\n{fence}python\n{body[:16000]}\n{fence}")


def _json_verdict(text: str) -> dict | None:
    """The first JSON object in a reply that carries a verdict (models wrap it in prose)."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text or ""):
        if ch != "{":
            continue
        try:
            v, _ = dec.raw_decode(text[i:])
        except ValueError:
            continue
        if isinstance(v, dict) and "passed" in v:
            return v
    return None


class RunAudit(BaseModel):
    model: str | None = Field(None, max_length=300)


@router.post("/objectives/{oid}/candidates/{cid}/audit/run")
async def run_audit(oid: str, cid: str, req: RunAudit) -> dict:
    """Audit a candidate NOW instead of waiting for an agent to take the chore -- the operator's
    way to clear a pending audit. Uses the project's best idea model (the first ladder rung
    that is not the model that wrote the code) unless one is named; the verdict goes through
    the same path as the swarm's (crowning, quarantine on failure)."""
    from . import escalation as E, swarm_policy

    obj = get_objective(oid)
    c = get_candidate(cid)
    if c["objective_id"] != oid:
        raise HTTPException(status_code=404, detail="candidate belongs to another objective")
    if c["audit"] in ("pass", "fail"):
        raise HTTPException(status_code=409, detail=f"#{c['seq']} was already audited ({c['audit']})")
    if c["status"] != "ok" or c.get("score") is None:
        raise HTTPException(status_code=409, detail=f"#{c['seq']} has no score to audit")
    if E._complete is None or E._loaded is None:
        raise HTTPException(status_code=503, detail="no model runner is available (the escalation loop is not running)")
    model = req.model
    if not model:
        project = projects.get(obj["project_id"]) or {}
        ladder = [r["model"] for r in swarm_policy.plan(project, E._loaded())["ladder"]]
        model = next((m for m in ladder if m != c.get("model")), None)
        if model is None:
            raise HTTPException(status_code=409, detail="no model other than the candidate's author to audit with; "
                                                        "name one in `model`")
    # Hold the lease so an agent does not take the same audit meanwhile.
    _update_candidate(cid, {"audit": "pending", "audit_started": time.time()})
    try:
        text = await E._complete(model, [{"role": "user", "content": _audit_prompt(obj, c)}], AUDIT_MAX_TOKENS,
                                 f"audit:{oid}")
    except HTTPException as exc:
        _update_candidate(cid, {"audit_started": 0})
        raise HTTPException(status_code=exc.status_code, detail=f"{model} could not audit: {exc.detail}") from None
    except Exception as exc:  # noqa: BLE001
        _update_candidate(cid, {"audit_started": 0})
        raise HTTPException(status_code=502, detail=f"{model} could not audit ({type(exc).__name__}): {exc}") from None
    verdict = _json_verdict(text or "")
    if verdict is None:
        _update_candidate(cid, {"audit_started": 0})
        raise HTTPException(status_code=502, detail=f"{model}'s reply had no JSON verdict: {(text or '')[:600]}")
    notes = (verdict.get("notes") or "") + ("" if not verdict.get("issues") else
                                            " Issues: " + "; ".join(map(str, verdict["issues"]))[:1500])
    res = await audit(oid, cid, Audit(passed=bool(verdict.get("passed")), notes=notes,
                                      model=f"{model} (run by operator)"))
    return {**res, "model": model, "notes": notes}


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
    # An ensemble holding this result stands on it: it goes too (and hands on the title).
    from . import ensembles

    cascaded = sorted(set(cascaded) | set(ensembles.disqualify_dependents(
        obj, ids={cid}, why=f"#{c['seq']} was demoted by {req.reviewer}: {lesson}")))

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
        if c.get("mode") == "ensemble":           # no script to re-run (start_retest filters these too)
            st.setdefault("skipped", []).append({"seq": c["seq"], "reason": "ensemble: its members are tested"})
            continue
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
                full_pos = T.actions_file(full) if T.is_task(obj) else _positions_file(full)
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
        from . import ensembles  # an ensemble built on the leak goes with it

        ensembles.disqualify_dependents(obj, ids={cid}, why=why)
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
    top = _ranked(oid, _higher(obj), limit=req.top)
    # Ensembles run no script -- their members are what the test covers -- so they are skipped
    # and reported as such rather than silently missing from the queue.
    ranked = [c for c in top if c.get("mode") != "ensemble"]
    skipped = [{"seq": c["seq"], "reason": "ensemble: its members are tested, not the ensemble"}
               for c in top if c.get("mode") == "ensemble"]
    ids = [c["id"] for c in ranked]
    _retests[oid] = {"started_at": time.time(), "total": len(ids), "results": [], "current": None,
                     "queue": [c["seq"] for c in ranked], "progress": {}, "skipped": skipped,
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
    "ranked": ("status='ok' AND score IS NOT NULL AND lookahead NOT IN ('fail', 'error', 'pending') "
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
        trade_book.forget(oid, ids)
        for i in ids:
            kept = _kept_positions(oid, i)
            kept.unlink(missing_ok=True)
            for cache in kept.parent.glob(kept.stem + ".*.npz"):      # .trades / .intraday.trades
                cache.unlink(missing_ok=True)
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


def _audit_code(row: sqlite3.Row) -> str:
    """The code an agent auditor reviews: the script, or for an ensemble its spec plus members."""
    if row["mode"] != "ensemble":
        return row["code"]
    from . import ensembles

    return ensembles.audit_body(_cand_row(row))


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
    if T.is_task(obj):
        try:
            proj = projects.get(obj["project_id"]) or {}
            view = await T.ensure_view(obj, proj.get("data_dir", ""))
            if view and (view != obj.get("dataset") or obj.get("time_column") != "t"):
                with _lock:
                    db().execute("UPDATE objectives SET dataset=?, time_column='t' WHERE id=?", (view, oid))
                    db().commit()
                obj = get_objective(oid)
            # The server's advice to agents evolves with its code: brief them with the current one.
            fresh = await T.refresh_guidance(obj)
            if fresh:
                with _lock:
                    db().execute("UPDATE objectives SET metric=? WHERE id=?", (json.dumps(fresh), oid))
                    db().commit()
                obj = get_objective(oid)
        except HTTPException as exc:          # the server is down or needs sign-in: carry on without analyses
            logger.warning("task view of %s not refreshed: %s", oid, exc.detail)
    higher = _higher(obj)
    ranked = _distinct(_ranked(oid, higher, 50))
    # Nothing ranked makes money: improving the leader only polishes a loser (every Gex2
    # "improve" of the -2.657 composite scored the same or worse), so explore new ideas and
    # build modules from scratch instead of refactoring the leader's code.
    losing = bool(ranked) and higher and all(float(c["score"]) <= 0 for c in ranked[:8])
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
        lesson_rows = db().execute(
            "SELECT l.text, c.id, c.seq, c.status, c.score, c.score_note, c.lookahead, c.audit, c.is_score, c.metrics "
            "FROM lessons l LEFT JOIN candidates c ON c.id = l.candidate_id WHERE l.objective_id=? AND l.active=1 "
            "ORDER BY l.ts DESC LIMIT ?", (oid, 60 if consolidate else 40)).fetchall()
    # Consolidation rewrites the lessons themselves: it gets them as written.
    lessons = ([r[0] for r in lesson_rows] if consolidate
               else _lesson_lines(lesson_rows, {c["id"]: i + 1 for i, c in enumerate(ranked)}, higher))
    with _lock:
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
    if any(k in model.lower() for k in ("qwen", "coder", "120b")):
        p_build = min(0.65, p_build * 1.3)
    if losing:
        p_build *= 0.5
    # While nothing ranked makes money, the candidates worth building on are the few with a
    # POSITIVE in-sample score that something fixable keeps off the leaderboard (one-sided,
    # too few active days) or that rank low. Without them every iteration since #107 (09-30)
    # was an explore with no parent: 0 of 27 candidates in a day built on another, though the
    # mentor and the practices kept saying "make #131 symmetric" and #114 scored +3.4 in-sample.
    # Parents that make money, best first. With only one or two of them (19f971 on 10-04: #157 and
    # #159 positive, the rest of its top 8 at -0.8..-1.3) the pool used to be the top 8 whatever
    # their sign, and agents spent iterations improving -1.2 candidates; the promising unranked
    # runs fill the pool instead.
    scripts = [c for c in ranked if c.get("mode") != "ensemble"]
    winners = [c for c in scripts if not higher or float(c["score"]) > 0][:8]
    promising = _promising(oid) if losing or len(winners) < 3 else []
    roll = random.random()
    if roll < p_build:
        mode = "build"
    elif (ranked and not losing or promising) and roll > p_build + (1 - p_build) * EXPLORE_PROBABILITY:
        mode = "improve"
        # An ensemble is arithmetic over other candidates, not a script: never a parent to mutate.
        have = {c["id"] for c in winners}
        pool = (winners + [c for c in promising if c["id"] not in have])[:8] or scripts[:8]
        liked = [c for c in _liked(oid, 8) if c.get("mode") != "ensemble"]
        if liked and random.random() < LIKED_PARENT_PROBABILITY:
            # The operator flagged these as the shape they want: build on them directly
            # some of the time, whatever their rank.
            pool = liked
        # Tournament of 3: favours the top without always picking it.
        if pool:
            parent = min(random.sample(pool, min(3, len(pool))), key=lambda c: pool.index(c))
        else:
            mode = "explore"
    if mode == "build" and ranked and not losing:
        # Build on what is winning: offer the leader's code as the thing to factor into modules.
        parent = next((c for c in ranked if c.get("mode") != "ensemble"), None)
    parent_doc = None
    if parent:
        full = get_candidate(parent["id"])
        parent_doc = {"id": full["id"], "seq": full["seq"], "rationale": full["rationale"],
                      "code": full["code"], "answer": full["answer"], "model": full.get("model"),
                      "in_sample": (full["metrics"] or {}).get("in_sample"),
                      "in_sample_score": full["is_score"],
                      # In-sample: before costs, after costs, flipped -- what to change first.
                      "diagnosis": ((full["metrics"] or {}).get("costs") or {}).get("verdict"),
                      # A promising parent may be unranked: None, and the reason it is not ranked.
                      "rank": next((i + 1 for i, c in enumerate(ranked) if c["id"] == parent["id"]), None),
                      "holdout_check": holdout_check(full, higher),
                      "problem": (full.get("score_note") or "")[:400] if full.get("score") is None else ""}

    def brief(c: dict) -> dict:
        return {"id": c["id"], "seq": c["seq"], "model": c["model"], "status": c["status"],
                "rationale": (c["rationale"] or "")[:400], "in_sample_score": c.get("is_score"),
                "lookahead": c.get("lookahead"),
                "problem": (c.get("score_note") or "")[:300] if c["status"] == "error" or c.get("score") is None else "",
                "rank": next((i + 1 for i, x in enumerate(ranked) if x["id"] == c["id"]), None),
                "holdout_check": holdout_check(c, higher),
                # Member numbers of an ensemble ("ensemble of #a+#b" in the brief); absent otherwise.
                **({"ensemble": [m.get("seq") for m in ((c.get("metrics") or {}).get("ensemble") or {}).get("members") or []]}
                   if c.get("mode") == "ensemble" else {})}

    project = projects.get(obj["project_id"]) or {}
    catalog = datasource.catalog(project.get("data_dir", "")) if project else []
    from .escalation import ideas_for_context  # escalation imports this module

    # The trade book: big winners vs the rest, of the whole swarm and of the parent (in-sample).
    trades = None
    if T.is_task(obj):
        try:
            trades = await asyncio.wait_for(trade_book.brief(obj, parent["id"] if parent else None), 120)
        except asyncio.TimeoutError:
            logger.warning("trade book brief of %s timed out", oid)
    if trades and parent_doc and trades.get("parent"):
        parent_doc["trade_review"] = trades.pop("parent")

    return {
        "objective": {k: obj[k] for k in ("id", "title", "description", "metric", "split_date", "dataset",
                                          "time_column", "lookahead_check", "require_audit", "status", "cooldown_s",
                                          "eval_timeout_s")},
        "metric_label": METRIC_LABEL.get(obj["metric"]["kind"], obj["metric"]["kind"]),
        "datasets": [c["view"] for c in catalog][:60],
        "mode": mode,
        "parent": parent_doc,
        "leaderboard": [brief(c) for c in ranked[:6]],
        # While fewer than 3 ranked candidates make money: the positive in-sample runs worth fixing or building on.
        "promising": [brief(c) | {"problem": (c.get("score_note") or "")[:300] if c.get("score") is None else ""}
                      for c in promising[:4]],
        # Runs the operator flagged as the shape they want, with their reason.
        "liked": [brief(c) | {"operator_note": c.get("liked_note") or ""} for c in _liked(oid, 6)],
        "recent": [brief(c) for c in recent],
        "total_candidates": (recent[0]["seq"] if recent else 0),
        "lessons": lessons,
        "notes": notes,
        # New directions from a stronger model, asked for because the search stopped improving,
        # and ideas from the research library's documents.
        "ideas": ideas_for_context(oid),
        # The research library: documents, their ideas and code (research_search / research_get).
        "research": _research_brief(obj["project_id"]),
        # An ensemble's own `code` is only its spec: the auditor gets the members' code with it.
        "audit": (_cand_row(pending) | {"code": _audit_code(pending), "answer": pending["answer"]}) if pending else None,
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
        # The Regime Lab: which verified candidate works in which regime, and the router script.
        "regime_lab": _regime_lab_brief(obj),
        "forecasters": _forecasters_brief(obj),
        "forecast_board": _forecast_board(obj),
        # What the team already knows about signals and forecast inputs, so it builds on it
        # instead of re-running the same study: decile studies and explored input combinations.
        "deci_studies": _deci_brief(obj),
        "forecast_inputs": _combo_brief(obj),
        "trade_book": trades,
        "activity": _activity_brief(obj),
    } | (_task_context(obj) if T.is_task(obj) else {})


def _activity_brief(obj: dict) -> dict | None:
    """The active-days floor in numbers the agent can act on, from the in-sample side only: how
    many sessions each period has, the share of sessions a strategy must trade on in-sample to
    expect enough active holdout days, and how many recent candidates missed it. A sparse
    strategy scores well in-sample and is then unrankable; agents only learned that after
    submitting, so whole iterations ended on unranked candidates."""
    if not obj.get("split_date"):
        return None
    with _lock:
        rows = db().execute(
            "SELECT status, score, score_note, metrics FROM candidates WHERE objective_id=? "
            "ORDER BY seq DESC LIMIT 40", (obj["id"],)).fetchall()
    is_days = ho_days = None
    need = int(obj["metric"].get("min_active_days") or 0)
    for r in rows:
        m = json.loads(r["metrics"] or "{}") if isinstance(r["metrics"], str) else (r["metrics"] or {})
        ins, ho = m.get("in_sample") or {}, m.get("holdout") or {}
        if ins.get("days") and ho.get("days"):
            is_days, ho_days = int(ins["days"]), int(ho["days"])
            if not need:
                found = re.search(r"need (\d+)", str(ho.get("note") or "") + str(ins.get("note") or ""))
                need = int(found.group(1)) if found else 0
            break
    need = need or 20
    if not (is_days and ho_days):
        return None
    share = min(1.0, need / ho_days * 1.5)
    sparse = sum(1 for r in rows if r["status"] == "ok" and r["score"] is None
                 and re.search(r"active days|SPARSE|too few trades", r["score_note"] or ""))
    return {"need": need, "in_sample_days": is_days, "holdout_days": ho_days, "share": round(share, 2),
            "in_sample_active": math.ceil(share * is_days), "recent": len(rows), "recent_sparse": sparse}


LESSONS_IN_BRIEF = 25
LESSON_CHARS = 400


def _lesson_lines(rows: list, ranks: dict[str, int], higher: bool = True) -> list[str]:
    """The team lessons for an iteration brief, newest first: near-repeats dropped, each cut to
    LESSON_CHARS, and each tagged with how the candidate it was written from fared. A lesson reads
    the same whether its candidate ranked or never reached the leaderboard; untagged, "KEEP:" lessons
    from unranked runs were followed as if they were results.
    `rows` are (text, candidate id, seq, status, score, score_note, lookahead, audit, is_score, metrics
    JSON); the candidate columns are None for a lesson without a candidate (a consolidated one)."""
    out: list[str] = []
    seen: set[str] = set()
    for text, cid, seq, status, score, note, lookahead, audit, is_score, metrics in rows:
        text = " ".join(str(text or "").split())
        key = re.sub(r"[^a-z]+", " ", re.sub(r"^(keep|avoid|try)\b", "", text.lower()))[:160].strip()
        if not key or key in seen:
            continue
        seen.add(key)
        if len(text) > LESSON_CHARS:
            text = text[:LESSON_CHARS - 3].rstrip() + "..."
        if seq is None:
            tag = ""
        elif status == "error":
            tag = f"[#{seq}: failed to run] "
        elif lookahead in ("fail", "error") or audit == "fail":
            tag = f"[#{seq}: DISQUALIFIED -- {'look-ahead' if lookahead in ('fail', 'error') else 'failed audit'}] "
        elif score is None:
            why = " ".join(str(note or "no score").split())
            tag = f"[#{seq}: NOT RANKED -- {why[:90]}{'...' if len(why) > 90 else ''}] "
        else:
            check = holdout_check({"score": score, "is_score": is_score,
                                   "metrics": json.loads(metrics) if isinstance(metrics, str) else metrics}, higher)
            tag = (f"[#{seq}: {f'rank {ranks[cid]}' if cid in ranks else 'ranked'}"
                   f"{f', {check} on unseen data' if check else ''}] ")
        out.append(tag + text)
        if len(out) >= LESSONS_IN_BRIEF:
            break
    return out


def _promising(oid: str, limit: int = 8) -> list[dict]:
    """Candidates with a POSITIVE in-sample score that ran cleanly, passed (or await) the
    look-ahead test and were not disqualified -- ranked or not, best in-sample first, one per
    distinct in-sample score. The parents to build on while no ranked candidate makes money:
    what keeps them off the leaderboard (one-sided, too few active days) is often one fix away."""
    with _lock:
        rows = db().execute(
            f"SELECT {_LIGHT} FROM candidates WHERE objective_id=? AND status='ok' AND is_score > 0 "
            "AND COALESCE(mode, '') != 'ensemble' AND lookahead NOT IN ('fail', 'error') "
            "AND COALESCE(audit, '') != 'fail' "
            # Unranked because the holdout could not be scored: nothing the agent can fix.
            "AND NOT (score IS NULL AND COALESCE(score_note, '') LIKE 'holdout%') "
            "ORDER BY is_score DESC, seq DESC LIMIT 40", (oid,)).fetchall()
    seen, out = set(), []
    for c in (_cand_row(r) for r in rows):
        key = round(float(c["is_score"]), 9)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out[:limit]


def _research_brief(project_id: str) -> dict | None:
    try:
        from . import research

        return research.brief(project_id)
    except Exception:  # noqa: BLE001 -- the brief goes out without the library
        logger.exception("research brief for %s failed", project_id)
        return None


def _task_context(obj: dict) -> dict:
    """A task objective's data is the task server's rows. Candidates read them with ft.rows(); the
    analysis tools read the same rows, in-sample, as the dataset view `obj["dataset"]` -- so the
    field guide and the studies are about the MCP's schema. Forecast features, the Regime Lab and
    ensembles do not apply (a task candidate cannot read features, has no positions to replay)."""
    return {"datasets": [obj["dataset"]] if obj.get("dataset") else [], "features": [],
            "fields": T.field_guide(obj), "forecast_lab": None, "regime_lab": None, "forecasters": None,
            "forecast_board": None, "forecast_inputs": None, "task": obj["metric"].get("task_info") or {}}


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
        board = forecast_scoreboard(obj)[:15]
    except Exception:  # noqa: BLE001 -- a scoreboard must never cost an agent its brief
        logger.exception("forecast scoreboard failed for %s", obj["id"])
        return []
    # The fairer "did it help": against the candidate's own parent where one exists without
    # the forecast (tslab.forecast_report), not only against the median of everyone else.
    try:
        from .tslab import forecast_report

        fair = {r["view"]: r for r in forecast_report(obj)}
        for row in board:
            r = fair.get(row["view"])
            if r:
                row["verdict"] = f"{r['verdict']} ({r['basis']}, n={r['n']}, effect {r['effect']})"
    except Exception:  # noqa: BLE001
        logger.exception("forecast report failed for %s", obj["id"])
    return board


def _deci_brief(obj: dict) -> dict | None:
    from .deciplot import brief

    return brief(obj)


def _combo_brief(obj: dict) -> list[dict]:
    from .tslab import combo_brief

    return combo_brief(obj["project_id"], obj)


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


def _regime_lab_brief(obj: dict) -> list[dict]:
    from .regimes import brief

    try:
        return brief(obj)
    except Exception:  # noqa: BLE001 -- a broken lab row must not cost an agent its brief
        logger.exception("regime lab brief failed")
        return []


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
    task = T.is_task(obj)
    # A task objective's experiment reads the task rows cut at the split (_run_forecasting); the
    # project's datasets are not mounted, so there is nothing to mirror.
    mirror = (await asyncio.to_thread(build_mirror, obj, project["data_dir"])
              if obj.get("split_date") and not task else None)
    # ... except the trade book the brief points agents at (in-sample trades only).
    book = trade_book.sandbox_dataset(obj) if task else None
    async with _EVAL_SLOTS:
        rep = await _run_forecasting(req.code, project["data_dir"], catalog, mirror, req.timeout_s, obj,
                                     obj.get("split_date"), requested_by="agent experiment",
                                     task_datasets=[book] if book else None)
    in_sample = bool(mirror) or (task and bool(obj.get("split_date")))
    hint = "" if rep["ok"] else error_hint(rep["stderr"], req.code)
    return {"ok": rep["ok"], "stdout": rep["stdout"][-12_000:], "stderr": rep["stderr"][-6_000:],
            **({"hint": hint} if hint else {}),
            "artifacts": [a["name"] for a in rep["artifacts"]], "duration_s": rep["duration_s"],
            "data": "in-sample only (rows before " + obj["split_date"] + ")" if in_sample else "full"}


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
    # No upper bound here: a longer horizon is read as the longest one (FORECAST_MAX_HORIZON), with
    # a note -- qwen asked for 360 steps (10-01) and got a raw pydantic 422 for an obvious intent.
    horizon: int = Field(12, ge=1)
    context: int = Field(512, ge=16, le=8192)
    model: str | None = None


FORECAST_MAX_HORIZON = 256


@router.post("/objectives/{oid}/forecast")
async def forecast_by_name(oid: str, req: ForecastByName) -> dict:
    """One forecast of a column's most recent IN-SAMPLE values, for an agent to look at. The
    series is read here, so the agent names it instead of pasting thousands of numbers."""
    obj = get_objective(oid)
    project = projects.get(obj["project_id"])
    if project is None:
        raise HTTPException(status_code=404, detail="no such project")
    mgr, info = _forecaster(req.model)
    horizon = min(req.horizon, FORECAST_MAX_HORIZON)
    context = min(req.context, int(info.get("context_length") or req.context))
    dataset = req.dataset or obj.get("dataset")
    times, values, _ = await asyncio.to_thread(_load_series, project["data_dir"], obj, dataset or "", req.column,
                                               obj.get("split_date"), context)
    if len(values) < 16:
        raise HTTPException(status_code=400, detail="not enough in-sample points")
    detail = input_streams("single forecast", info["model"], dataset, times, [len(values) - 1],
                           len(values), horizon, [(req.column, "target")])
    _note_inputs(info["model"], detail)
    cap = _capture(detail, times, [len(values) - 1], len(values), horizon, obj.get("split_date"))
    cap.context_values(req.column, "target", {req.column: values})
    res = await mgr.forecast(info["model"], {"series": values, "horizon": horizon, "quantiles": FEATURE_QUANTILES})
    fc = (res.get("forecasts") or [{}])[0]
    cap.output(len(values) - 1, req.column, fc)
    cap.publish(info["model"])
    return {"model": info["model"], "column": req.column, "history_from": str(times[0]), "history_to": str(times[-1]),
            "last_value": values[-1], "horizon": horizon, "median": fc.get("median"),
            "q10": (fc.get("quantiles") or {}).get("0.1"), "q90": (fc.get("quantiles") or {}).get("0.9"),
            "note": ((f"horizon {req.horizon} is past the longest forecast ({FORECAST_MAX_HORIZON} steps), so "
                      f"{FORECAST_MAX_HORIZON} steps were forecast. ") if req.horizon > horizon else "")
            + "To use forecasts in a strategy, create a feature with forecast_feature -- scripts cannot call the model."}


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
    views: list[str] = []                  # what this query may name: a mistyped table is told these
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
            views.append(item["view"])
        fdir = features_dir(obj, obj.get("split_date"))
        for f in _feature_catalog(obj["id"]) if fdir else []:
            con.execute(f'CREATE OR REPLACE VIEW "{f["view"]}" AS SELECT * FROM read_parquet(\'{(Path(fdir) / f["path"]).as_posix()}\')')
            views.append(f["view"])
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
        raise datasource.DataError(datasource.sql_error(exc, views)) from None
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
