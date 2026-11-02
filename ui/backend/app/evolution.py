"""The search's memory: which strategies are really different, what "good" means, and what works.

Picking parents from the top of one score collapses the search: on b27d52cb27 the top 303 candidates
were one strategy (the same 23 in-sample trades), so 27 "new bests" reshuffled it, the last eight for
+0.31 in 663 candidates. This module keeps the search spread and measures itself, after AlphaEvolve
(Novikov et al. 2025, arXiv 2506.13131) and its program database:

* **Islands.** Candidates whose in-sample daily returns correlate above ISLAND_RHO are one strategy,
  however different their code. Parents are drawn across islands, weighted away from the ones the
  team has just worked on, so progress is not one lineage and a strategy can come back later.
* **Niches.** Each strategy has a place on a map (how often it trades, how many days, which side), and
  the best of each place is kept, MAP-Elites style. An EXPLORE iteration is sent to an empty place.
* **Lenses.** Several definitions of "good" (smooth, shallow drawdown, many trades, unlike the leader):
  strategies strong on different ones have different structure and feed different proposals, which
  often improves the main score too. A lens is chosen per iteration.
* **Instruction experiments.** The words of an assignment are a treatment: variants are drawn by
  Thompson sampling, each candidate records the one it got, and the win rate decides which survive.
  The mentor proposes new ones. The first variant is the control and is never retired.

**In-sample only.** Islands, niches and lenses read in-sample numbers. Ranking already leans on the
holdout; every further choice made from it uses it up. A parent with a collapsing holdout is still
kept out -- one bit, not a score. An experiment's outcome does use the ranked score, because the
question it answers is what generalises; that is many candidates' worth of a few bits, and agents
never see those numbers either way.

The tournament over the leaderboard stays as a control arm (candidates record `parent_via`), so the
archive has to earn its share in numbers.
"""

from __future__ import annotations

import json
import logging
import math
import random
import statistics
import threading
import time
import uuid
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import objectives as O

logger = logging.getLogger("freetoken.evolution")

router = APIRouter(tags=["evolution"])

# Daily returns correlating above this are one strategy.
ISLAND_RHO = 0.9
ARCHIVE_MAX_MEMBERS = 800
# Share of IMPROVE iterations that take a parent from the archive; the rest are the control arm.
ARCHIVE_SHARE = 0.7
MIN_ISLANDS = 2
ISLANDS_IN_PLAY = 8
# One whole-strategy jump is worth a tenth of the leader's score, with this floor.
JUMP_SHARE = 0.1
JUMP_FLOOR = 0.05
INSPIRATIONS = 2
INSPIRATION_CODE_CHARS = 2500
# Experiments: a variant needs this many outcomes before it can be retired, and is retired when it
# has under this chance of being better than the best one.
RETIRE_AFTER = 15
RETIRE_BELOW = 0.05
MAX_ACTIVE_VARIANTS = 6
VARIANT_TEXT_MAX = 900

# Bucket edges of the niche map, on in-sample numbers.
FREQ_EDGES = (0.25, 1.0, 4.0)
FREQ_NAMES = ("rare (under 1 trade a week or so)", "selective (under 1 a day)", "daily (1-4 a day)", "active (4+ a day)")
DAYS_EDGES = (0.3, 0.6)
DAYS_NAMES = ("few days (under 30%)", "many days (30-60%)", "most days (60%+)")
SIDE_NAMES = {"long": "mostly long", "both": "both sides", "short": "mostly short"}
SIDE_SHARE = 0.7

LENSES: dict[str, str] = {
    "primary": "the objective's own in-sample score",
    "smooth": "the smoothest equity curve: steady gains over the whole period, no long flat or losing stretches",
    "drawdown": "the shallowest drawdown for what it earns",
    "breadth": "the most trades behind the result: a score from many trades is evidence, one from few is luck",
    "novel": "being unlike the leader: earning on different days, from a different source of return",
}

# The seed of each mode's experiment. The first of a mode is its control: what the assignment said
# before experiments existed (an empty text keeps the brief's own wording).
SEED_VARIANTS: dict[str, list[tuple[str, str]]] = {
    "improve": [
        ("focused", ""),
        ("structural",
         "Replace one whole component of it -- its entry signal, its regime gate or its exit system -- with a "
         "different mechanism, and keep the rest. Expect to rewrite 20-50% of the code. Do not touch its numbers: "
         "over the team's history changes of this size beat their parent 44% of the time, against 12-17% for "
         "smaller ones, which mostly reshuffled the same trades."),
        ("searcher",
         "Do not tune it by hand. Turn it into a FAMILY: replace its hand-set thresholds and windows with "
         "parameters, and submit the search itself -- `found = ft.search(family, {param: (low, high), ...}, frame)` "
         "then report found.positions. Each period's parameters are then chosen from earlier sessions only, and "
         "one iteration tests the whole family. Try it in run_python first and read found.table."),
    ],
    "explore": [
        ("open", ""),
        ("family-first",
         "Write the idea as a FAMILY from the start: a function of a few parameters that returns positions, tuned "
         "by `found = ft.search(family, {param: (low, high), ...}, frame)`, reporting found.positions. A family "
         "that comes out mostly flat has no edge beyond chance, and that is a real result: say so."),
    ],
}


_cache: dict[str, tuple[tuple, dict]] = {}
_cache_lock = threading.Lock()


def _ensure() -> None:
    with O._lock:
        O.db().executescript(
            """
            CREATE TABLE IF NOT EXISTS instruction_variants (
                id INTEGER PRIMARY KEY AUTOINCREMENT, objective_id TEXT NOT NULL, mode TEXT NOT NULL,
                name TEXT NOT NULL, text TEXT NOT NULL DEFAULT '', parent_id INTEGER, author TEXT NOT NULL DEFAULT 'seed',
                control INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'active', why TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL, retired_at REAL
            );
            CREATE INDEX IF NOT EXISTS variant_obj ON instruction_variants(objective_id, mode, status);
            """
        )
        O.db().commit()


# =======================================================================================
# What a candidate is: its place on the map, and how it reads through each lens
# =======================================================================================
def _trades(m: dict) -> tuple[float | None, float | None]:
    """(long, short) in-sample trades, by either harness's record of them."""
    sides = (m.get("execution") or {}).get("sides") or {}
    if sides:
        return float(sides.get("long") or 0), float(sides.get("short") or 0)
    d = ((m.get("task") or {}).get("diagnostics") or {}).get("in_sample") or {}
    n = d.get("trades")
    if isinstance(n, dict):
        return float(n.get("long") or 0), float(n.get("short") or 0)
    return None, None


def behaviour(c: dict) -> dict | None:
    """The in-sample facts a strategy's place on the map comes from, or None without them."""
    m = c.get("metrics") or {}
    ins = m.get("in_sample") or {}
    days = ins.get("days") or 0
    if not days:
        return None
    long_n, short_n = _trades(m)
    total = (long_n or 0.0) + (short_n or 0.0)
    d = ((m.get("task") or {}).get("diagnostics") or {}).get("in_sample") or {}
    per_day = total / days if long_n is not None else d.get("trades_per_day")
    return {"trades": total if long_n is not None else None, "trades_per_day": per_day,
            "active_share": (ins.get("active_days") or 0) / days,
            "long_share": long_n / total if total else None}


def niche(b: dict) -> tuple[int, int, str]:
    per_day = b.get("trades_per_day") or 0.0
    side = b.get("long_share")
    return (int(np.searchsorted(FREQ_EDGES, per_day, side="right")),
            int(np.searchsorted(DAYS_EDGES, b.get("active_share") or 0.0, side="right")),
            "both" if side is None or 1 - SIDE_SHARE <= side <= SIDE_SHARE else "long" if side > SIDE_SHARE else "short")


def niche_label(cell: tuple[int, int, str] | list) -> str:
    return f"{FREQ_NAMES[cell[0]]}, {DAYS_NAMES[cell[1]]}, {SIDE_NAMES[cell[2]]}"


def lens_value(c: dict, lens: str, rho_leader: float | None = None, higher: bool = True) -> float | None:
    """How good a candidate is through a lens, in-sample only and higher is better; None when it cannot say. Where a lower
    score is better (an error measure) the other lenses have no meaning: only `primary` speaks, and it reads as the score."""
    m = c.get("metrics") or {}
    ins = m.get("in_sample") or {}
    base = c.get("is_score")
    if base is None:
        return None
    if lens == "primary":
        return float(base)
    if not higher:
        return None
    if lens == "smooth":
        s = ins.get("smoothness")
        return float(base) * float(s) if s is not None and base > 0 else None
    if lens == "drawdown":
        v = ins.get("calmar")
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None
    if lens == "breadth":
        n = (behaviour(c) or {}).get("trades")
        return float(base) * math.log1p(n) if n and base > 0 else None
    if lens == "novel":
        return float(base) * (1.0 - abs(rho_leader)) if rho_leader is not None and base > 0 else None
    return None


# =======================================================================================
# The archive: every usable candidate, in islands and niches
# =======================================================================================
def _members(oid: str, higher: bool) -> list[dict]:
    """Candidates a parent can be: scored in-sample, money-making there, undisqualified, not an ensemble (an ensemble
    is arithmetic over other candidates). Unranked ones count -- what keeps them off the board may be one fix away."""
    sign = "DESC" if higher else "ASC"
    with O._lock:
        rows = O.db().execute(
            f"SELECT {O._LIGHT}, returns FROM candidates WHERE objective_id=? AND status='ok' AND is_score IS NOT NULL "
            "AND COALESCE(mode, '') != 'ensemble' AND lookahead NOT IN ('fail', 'error') AND COALESCE(audit, '') != 'fail' "
            f"ORDER BY is_score {sign}, seq DESC LIMIT ?", (oid, ARCHIVE_MAX_MEMBERS)).fetchall()
    out = []
    for r in rows:
        c = O._cand_row(r)
        if (float(c["is_score"]) > 0 or not higher) and O.holdout_check(c, higher) != "collapses":
            out.append(c)
    return out


def _in_sample_matrix(members: list[dict], split: str | None) -> tuple[np.ndarray, list[str]]:
    """members x days of in-sample daily returns, on the days any of them has (0 where one has none)."""
    days = sorted({str(d)[:10] for c in members for d, _ in c.get("returns") or [] if not split or str(d)[:10] < split})
    at = {d: i for i, d in enumerate(days)}
    x = np.zeros((len(members), len(days)))
    for i, c in enumerate(members):
        for d, v in c.get("returns") or []:
            j = at.get(str(d)[:10])
            if j is not None and isinstance(v, (int, float)) and math.isfinite(v):
                x[i, j] = v
    return x, days


def _standardise(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    z = x - x.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    return np.divide(z, norm, out=np.zeros_like(z), where=norm > 0)


def islands_of(x: np.ndarray, rho: float = ISLAND_RHO) -> list[int]:
    """Greedy clustering, members taken in order (best first): each joins the first island whose founder's daily
    returns it correlates with above `rho`, or founds a new one. A series with no variance is its own island."""
    z = _standardise(x)
    founders: list[int] = []
    out: list[int] = []
    for i in range(len(z)):
        if founders:
            c = z[founders] @ z[i]
            j = int(np.argmax(c))
            if c[j] >= rho:
                out.append(j)
                continue
        founders.append(i)
        out.append(len(founders) - 1)
    return out


def _roll_up(obj: dict) -> dict:
    """Build the archive from the candidates. Everything it needs from `metrics` and `returns` is here."""
    higher = O._higher(obj)
    members = _members(obj["id"], higher)
    if not members:
        return {"islands": [], "niches": {}, "members": 0, "by_id": {}}
    x, days = _in_sample_matrix(members, obj.get("split_date"))
    if not days:
        return {"islands": [], "niches": {}, "members": 0, "by_id": {}}
    island_of = islands_of(x)
    z = _standardise(x)
    rho_leader = z @ z[0]
    children: dict[str, int] = {}
    with O._lock:
        recent = O.db().execute(
            "SELECT parent_id FROM candidates WHERE objective_id=? AND parent_id IS NOT NULL ORDER BY seq DESC LIMIT 60",
            (obj["id"],)).fetchall()
    for r in recent:
        children[r[0]] = children.get(r[0], 0) + 1
    islands: dict[int, dict] = {}
    niches: dict[str, dict] = {}
    by_id: dict[str, dict] = {}
    for c, isl, rho in zip(members, island_of, rho_leader):
        b = behaviour(c) or {}
        cell = niche(b) if b else None
        entry = {"id": c["id"], "seq": c["seq"], "model": c.get("model"), "is_score": c["is_score"], "score": c.get("score"),
                 "island": isl, "niche": list(cell) if cell else None, "rho_leader": round(float(rho), 3),
                 "behaviour": b, "lens": {k: lens_value(c, k, float(rho), higher) for k in LENSES},
                 "rationale": (c.get("rationale") or "")[:300]}
        by_id[c["id"]] = entry
        home = islands.setdefault(isl, {"island": isl, "founder": c["seq"], "size": 0, "recent_children": 0, "members": []})
        home["size"] += 1
        home["recent_children"] += children.get(c["id"], 0)
        home["members"].append(entry)
        if cell:
            key = "|".join(map(str, cell))
            n = niches.setdefault(key, {"cell": list(cell), "label": niche_label(cell), "strategies": set(), "best": entry})
            n["strategies"].add(isl)
            if entry["is_score"] * (1 if higher else -1) > n["best"]["is_score"] * (1 if higher else -1):
                n["best"] = entry
    ordered = sorted(islands.values(), key=lambda i: i["island"])
    for n in niches.values():
        n["strategies"] = len(n["strategies"])
    return {"islands": ordered, "niches": niches, "members": len(members), "by_id": by_id, "at": time.time()}


def archive(obj: dict, fresh: bool = False) -> dict:
    """The objective's archive, rebuilt when its candidates change: a new one, a delete, a disqualification."""
    with O._lock:
        state = tuple(O.db().execute(
            "SELECT COALESCE(MAX(seq), 0), COUNT(*), COALESCE(SUM(lookahead IN ('fail', 'error')), 0), "
            "COALESCE(SUM(audit = 'fail'), 0), COALESCE(SUM(is_score IS NOT NULL), 0) FROM candidates WHERE objective_id=?",
            (obj["id"],)).fetchone())
    with _cache_lock:
        hit = _cache.get(obj["id"])
    if not fresh and hit and hit[0] == state:
        return hit[1]
    built = _roll_up(obj)
    with _cache_lock:
        _cache[obj["id"]] = (state, built)
    return built


def _applicable(obj: dict) -> bool:
    """Islands need a daily return stream: judged and self-scored objectives have none."""
    kind = obj["metric"]["kind"]
    return kind == "task" or kind in O.RETURN_METRICS


def _instructable(obj: dict) -> bool:
    """The instructions speak of positions and ft.search: an objective whose scripts report positions or actions."""
    m = obj["metric"]
    return m["kind"] == "task" or bool(m["kind"] in O.RETURN_METRICS and m.get("price_column") and obj.get("dataset"))


def _pick_lens(rng: random.Random) -> str:
    return rng.choice(list(LENSES))


def pick_parent(obj: dict, rng: random.Random | None = None) -> dict | None:
    """(parent candidate id, lens, island): an island drawn with weight falling with its recent children and its rank,
    then its best member through a lens chosen for this iteration. None while the archive has too few islands to
    be a choice at all."""
    if not _applicable(obj):
        return None
    rng = rng or random
    a = archive(obj)
    islands = a.get("islands") or []
    if len(islands) < MIN_ISLANDS:
        return None
    higher = O._higher(obj)
    sign = 1 if higher else -1
    best = {i["island"]: max(m["is_score"] * sign for m in i["members"]) for i in islands}
    in_play = sorted(islands, key=lambda i: -best[i["island"]])[:ISLANDS_IN_PLAY]
    weights = [(0.75 ** rank) / (1.0 + i["recent_children"]) for rank, i in enumerate(in_play)]
    island = rng.choices(in_play, weights=weights, k=1)[0]
    lens = _pick_lens(rng)
    scored = [(m["lens"].get(lens), m) for m in island["members"] if m["lens"].get(lens) is not None]
    if not scored:
        lens = "primary"
        scored = [(m["lens"]["primary"], m) for m in island["members"]]
    top = max(scored, key=lambda s: s[0] * (sign if lens == "primary" else 1))[1]
    return {"id": top["id"], "lens": lens, "island": island["island"]}


def inspirations(obj: dict, parent_id: str | None, rng: random.Random | None = None) -> list[dict]:
    """Strategies unlike the parent, with their code and scores, for the agent to borrow from: the paper's prompts carry
    several programs, because a proposal draws on what they do differently."""
    if not _applicable(obj):
        return []
    rng = rng or random
    a = archive(obj)
    own = (a.get("by_id") or {}).get(parent_id or "", {}).get("island")
    sign = 1 if O._higher(obj) else -1
    others = [i for i in a.get("islands") or [] if i["island"] != own]
    picks = []
    for i in sorted(others, key=lambda i: -max(m["is_score"] * sign for m in i["members"]))[:ISLANDS_IN_PLAY]:
        picks.append(max(i["members"], key=lambda m: m["is_score"] * sign))
    rng.shuffle(picks)
    out = []
    for m in picks[:INSPIRATIONS]:
        try:
            full = O.get_candidate(m["id"])
        except HTTPException:
            continue
        out.append({"seq": m["seq"], "model": m.get("model"), "in_sample_score": m["is_score"], "niche": niche_label(m["niche"])
                    if m.get("niche") else None, "rho_leader": m["rho_leader"], "rationale": m["rationale"],
                    "code": (full.get("code") or "")[:INSPIRATION_CODE_CHARS]})
    return out


def empty_niche(obj: dict, rng: random.Random | None = None) -> dict | None:
    """A place on the map worth sending an explorer to: one nobody has filled, in the band a strategy can rank from
    (never the rarest traders -- the leaderboard's activity floor is against them), both sides first."""
    if not _applicable(obj):
        return None
    rng = rng or random
    a = archive(obj)
    if not a.get("members"):
        return None
    taken = {tuple(n["cell"]) for n in (a.get("niches") or {}).values()}
    wanted = [(f, d, "both") for f in (1, 2, 3) for d in (1, 2)]
    free = [c for c in wanted if c not in taken]
    if not free:
        crowded = sorted((n for n in a["niches"].values() if tuple(n["cell"]) in wanted), key=lambda n: n["strategies"])
        free = [tuple(n["cell"]) for n in crowded[:2]]
    if not free:
        return None
    cell = rng.choice(free)
    return {"cell": list(cell), "label": niche_label(cell), "empty": cell not in taken}


def scorecard(obj: dict) -> dict | None:
    """What every agent reads: how concentrated the search is, and the best DISTINCT strategy by each definition of
    good. In-sample numbers only."""
    if not _applicable(obj):
        return None
    a = archive(obj)
    islands = a.get("islands") or []
    if not islands:
        return None
    sign = 1 if O._higher(obj) else -1
    top = sorted((m for i in islands for m in i["members"]), key=lambda m: -m["is_score"] * sign)[:20]
    same = sum(1 for m in top if m["island"] == top[0]["island"])
    leaders = {}
    for lens, about in LENSES.items():
        rows = [m for i in islands for m in i["members"] if m["lens"].get(lens) is not None]
        if rows:
            m = max(rows, key=lambda r: r["lens"][lens] * (sign if lens == "primary" else 1))
            leaders[lens] = {"about": about, "seq": m["seq"], "model": m.get("model"), "in_sample_score": m["is_score"],
                             "value": round(m["lens"][lens], 3), "island": m["island"],
                             "trades": (m.get("behaviour") or {}).get("trades")}
    return {"strategies": len(islands), "candidates": a["members"], "top": len(top), "top_same_strategy": same,
            "leaders": leaders}


# =======================================================================================
# Instruction experiments
# =======================================================================================
def variants(oid: str, mode: str | None = None, active_only: bool = True) -> list[dict]:
    _ensure()
    q = "SELECT * FROM instruction_variants WHERE objective_id=?"
    args: list[Any] = [oid]
    if mode:
        q += " AND mode=?"
        args.append(mode)
    if active_only:
        q += " AND status='active'"
    with O._lock:
        return [dict(r) for r in O.db().execute(q + " ORDER BY id", args).fetchall()]


def _seed(oid: str) -> None:
    _ensure()
    with O._lock:
        have = {r[0] for r in O.db().execute("SELECT DISTINCT mode FROM instruction_variants WHERE objective_id=?",
                                             (oid,)).fetchall()}
        for mode, seeds in SEED_VARIANTS.items():
            if mode in have:
                continue
            for i, (name, text) in enumerate(seeds):
                O.db().execute("INSERT INTO instruction_variants (objective_id, mode, name, text, control, created_at) "
                               "VALUES (?,?,?,?,?,?)", (oid, mode, name, text, 1 if i == 0 else 0, time.time()))
        O.db().commit()


def jump_size(leader_score: float | None) -> float:
    return max(JUMP_FLOOR, JUMP_SHARE * abs(leader_score or 0.0))


_ELIGIBLE = ("status='ok' AND score IS NOT NULL AND lookahead NOT IN ('fail', 'error', 'pending') "
             "AND COALESCE(audit, '') != 'fail'")


def _leader_score(oid: str, sign: int) -> float:
    """The best score that counts, in the direction of better: a disqualified candidate's sets no bar."""
    with O._lock:
        rows = O.db().execute(f"SELECT score FROM candidates WHERE objective_id=? AND {_ELIGIBLE}", (oid,)).fetchall()
    return max((r[0] * sign for r in rows), default=0.0)


def _merged(rows: list[dict]) -> dict:
    """One iteration's outcome from all it submitted: a crash, its repair and its retries are one trial of what the
    iteration was told, so it wins if any of them did, and failed only if all of them did."""
    best = max(rows, key=lambda r: (r["win"], r["beat_parent"], r["gain"] if r["gain"] is not None else -1e18))
    return {**best, "failed": all(r["failed"] for r in rows)}


def _outcomes(obj: dict) -> list[dict]:
    """Every iteration that carries an assignment and has a settled result: the experiments' raw data."""
    higher = O._higher(obj)
    sign = 1 if higher else -1
    with O._lock:
        rows = O.db().execute(
            "SELECT id, seq, mode, parent_id, status, score, is_score, lookahead, audit, metrics, assignment "
            "FROM candidates WHERE objective_id=? AND assignment != '{}' ORDER BY seq", (obj["id"],)).fetchall()
        known = {r[0]: r[1] if r[1] is not None else r[2] for r in O.db().execute(
            "SELECT id, score, is_score FROM candidates WHERE objective_id=?", (obj["id"],)).fetchall()}
    jump = jump_size(_leader_score(obj["id"], sign))
    iterations: dict[str, list[dict]] = {}
    for r in rows:
        a = json.loads(r["assignment"] or "{}")
        m = json.loads(r["metrics"] or "{}")
        valid = (r["status"] == "ok" and r["score"] is not None and r["lookahead"] not in ("fail", "error")
                 and r["audit"] != "fail")
        # A parent with no ranked score is measured by its in-sample one: the same bar for every route to it.
        parent = known.get(r["parent_id"]) if r["parent_id"] else None
        gain = (r["score"] - parent) * sign if valid and parent is not None else None
        rho = (m.get("novelty") or {}).get("rho_leader")
        if parent is not None:
            win = gain is not None and gain > jump
        else:
            # Nothing to beat: a new strategy that holds up, not one more copy of the leader.
            win = bool(valid and (not higher or r["score"] > 0) and (rho is None or abs(rho) < 0.7))
        iterations.setdefault(a.get("iteration") or r["id"], []).append({
            "seq": r["seq"], "mode": r["mode"], "assignment": a, "valid": valid, "win": win, "gain": gain,
            "failed": r["status"] != "ok", "beat_parent": bool(gain is not None and gain > 0),
            "change_size": (m.get("change") or {}).get("size"),
            "pending": r["lookahead"] == "pending" or r["status"] == "evaluating"})
    return [_merged(v) for v in iterations.values() if not any(x["pending"] for x in v)]


def _record(rows: list[dict]) -> dict:
    gains = [r["gain"] for r in rows if r["gain"] is not None]
    sizes = [r["change_size"] for r in rows if r.get("change_size") is not None]
    n = len(rows)
    return {"n": n, "wins": sum(1 for r in rows if r["win"]), "win_rate": round(sum(1 for r in rows if r["win"]) / n, 3) if n else None,
            "beat_parent": round(sum(1 for r in rows if r["beat_parent"]) / n, 3) if n else None,
            "failed": sum(1 for r in rows if r["failed"]),
            "median_gain": round(statistics.median(gains), 3) if gains else None,
            "median_change": round(statistics.median(sizes), 3) if sizes else None}


def _experiment_rows(obj: dict) -> list[dict]:
    """Each variant with its record so far, retired ones too."""
    outcomes = _outcomes(obj)
    rows = []
    for v in variants(obj["id"], active_only=False):
        mine = [o for o in outcomes if o["assignment"].get("variant") == v["id"]]
        rows.append({**{k: v[k] for k in ("id", "mode", "name", "text", "status", "author", "control", "why", "parent_id")},
                     **_record(mine)})
    return rows


def _beta_draws(wins: int, losses: int, rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.beta(1 + wins, 1 + losses, n)


def settle_experiments(obj: dict) -> list[str]:
    """Retire any non-control variant with enough outcomes and little chance of being the best of its mode."""
    rows = [r for r in _experiment_rows(obj) if r["status"] == "active"]
    retired = []
    rng = np.random.default_rng(0)
    for mode in {r["mode"] for r in rows}:
        # Only arms with a record: an untried one draws so widely it would take the chance of best from proven ones.
        arm = [r for r in rows if r["mode"] == mode and r["n"] >= RETIRE_AFTER]
        if len(arm) < 2:
            continue
        draws = np.vstack([_beta_draws(r["wins"], r["n"] - r["wins"], rng, 4000) for r in arm])
        chance = (draws.argmax(axis=0)[None, :] == np.arange(len(arm))[:, None]).mean(axis=1)
        for r, p in zip(arm, chance):
            if not r["control"] and p < RETIRE_BELOW:
                why = f"{r['wins']} wins in {r['n']}: {p:.1%} chance of being the best {mode} instruction"
                with O._lock:
                    O.db().execute("UPDATE instruction_variants SET status='retired', why=?, retired_at=? WHERE id=?",
                                   (why, time.time(), r["id"]))
                    O.db().commit()
                retired.append(r["name"])
    return retired


def pick_variant(obj: dict, mode: str, rng: random.Random | None = None) -> dict | None:
    """Thompson sampling: draw each variant's win rate from what it has shown, take the best draw. A variant with little
    data draws widely, so it gets tried; one that has lost keeps losing the draw."""
    if mode not in SEED_VARIANTS or not _instructable(obj):
        return None
    _seed(obj["id"])
    settle_experiments(obj)
    active = variants(obj["id"], mode)
    if not active:
        return None
    outcomes = _outcomes(obj)
    rng = rng or random
    draws = []
    for v in active:
        mine = [o for o in outcomes if o["assignment"].get("variant") == v["id"]]
        wins = sum(1 for o in mine if o["win"])
        draws.append((rng.betavariate(1 + wins, 1 + len(mine) - wins), v))
    return max(draws, key=lambda d: d[0])[1]


def add_variant(oid: str, mode: str, name: str, text: str, author: str, parent_id: int | None = None) -> dict:
    """A new instruction to test (from the mentor or the operator). Refused, not queued, while the mode is full: an
    experiment with too many arms learns nothing."""
    obj = O.get_objective(oid)
    if mode not in SEED_VARIANTS:
        raise HTTPException(status_code=400, detail=f"mode must be one of {', '.join(SEED_VARIANTS)}")
    _seed(oid)
    settle_experiments(obj)
    active = variants(oid, mode)
    if len(active) >= MAX_ACTIVE_VARIANTS:
        raise HTTPException(status_code=409, detail=f"{mode} already has {len(active)} instructions under test")
    if any(" ".join(v["text"].split()) == " ".join(text.split()) for v in active):
        raise HTTPException(status_code=409, detail="that instruction is already under test")
    with O._lock:
        cur = O.db().execute(
            "INSERT INTO instruction_variants (objective_id, mode, name, text, parent_id, author, created_at) "
            "VALUES (?,?,?,?,?,?,?)", (oid, mode, name[:60], text[:VARIANT_TEXT_MAX], parent_id, author[:200], time.time()))
        O.db().commit()
    return {"id": int(cur.lastrowid), "name": name[:60], "mode": mode}


# =======================================================================================
# An iteration's assignment
# =======================================================================================
def assign(obj: dict, mode: str, parent_via: str | None, parent_id: str | None, lens: str | None,
           practices_version: int | None, rng: random.Random | None = None) -> dict:
    """The treatments this iteration gets. The runner hands `assignment` back with the submission, so each candidate
    carries what it was told -- that is what the experiments measure."""
    out: dict[str, Any] = {"iteration": uuid.uuid4().hex[:12], "parent_via": parent_via, "lens": lens,
                           "practices_version": practices_version}
    try:
        v = pick_variant(obj, mode, rng)
    except Exception:  # noqa: BLE001 -- an experiment must never cost an agent its iteration
        logger.exception("no instruction variant for %s", obj["id"])
        v = None
    if v:
        out.update(variant=v["id"], variant_name=v["name"], instruction=v["text"])
    if mode == "explore":
        try:
            n = empty_niche(obj, rng)
        except Exception:  # noqa: BLE001
            logger.exception("no niche for %s", obj["id"])
            n = None
        if n:
            out["niche"] = n
    return out


def brief(obj: dict, parent_id: str | None) -> dict:
    """The evolution part of an iteration's context."""
    try:
        return {"scorecard": scorecard(obj), "inspirations": inspirations(obj, parent_id) if parent_id else []}
    except Exception:  # noqa: BLE001 -- a scoreboard must never cost an agent its brief
        logger.exception("evolution brief failed for %s", obj["id"])
        return {"scorecard": None, "inspirations": []}


def novelty(obj: dict, returns: list[list]) -> dict | None:
    """How a new candidate's in-sample daily returns correlate with the leader's: stored on the candidate, so an
    experiment can tell a new strategy from a copy."""
    try:
        best = O._best(obj)
        if not best:
            return None
        full = O.get_candidate(best["id"])
        x, days = _in_sample_matrix([{"returns": full.get("returns") or []}, {"returns": returns}], obj.get("split_date"))
        if not days:
            return None
        z = _standardise(x)
        return {"rho_leader": round(float(z[0] @ z[1]), 3), "leader": best["seq"]}
    except Exception:  # noqa: BLE001 -- a diagnostic must never cost the score
        logger.exception("novelty of a candidate of %s failed", obj["id"])
        return None


def mentor_section(obj: dict) -> dict:
    """What the mentor reads to evolve the instructions: each one's record, and the practices' versions."""
    if not _applicable(obj):
        return {}
    _seed(obj["id"])
    keep = ("id", "mode", "name", "text", "status", "control", "n", "wins", "win_rate", "failed", "median_change")
    outcomes = _outcomes(obj)
    by_version: dict[Any, list[dict]] = {}
    for o in outcomes:
        by_version.setdefault(o["assignment"].get("practices_version"), []).append(o)
    # Counts of wins only: the ranked gains behind them lean on the holdout, which the mentor does not see.
    return {"variants": [{k: r[k] for k in keep} for r in _experiment_rows(obj)],
            "practices": {str(k): {x: _record(v)[x] for x in ("n", "wins", "win_rate")}
                          for k, v in sorted(by_version.items(), key=lambda x: str(x[0])) if k is not None},
            "scorecard": scorecard(obj)}


# =======================================================================================
# The console
# =======================================================================================
def _yield_table(rows: list[dict], key) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        k = key(r)
        if k:
            groups.setdefault(str(k), []).append(r)
    return sorted(({"arm": k, **_record(v)} for k, v in groups.items()), key=lambda r: -(r["n"]))


def _size_band(size: float | None) -> str | None:
    if size is None:
        return None
    for top, name in ((0.05, "tiny (under 5%)"), (0.2, "small (5-20%)"), (0.5, "medium (20-50%)")):
        if size < top:
            return name
    return "rewrite (over 50%)"


_sizes: dict[str, tuple[int, list[dict]]] = {}


def _history_by_size(oid: str, higher: bool) -> list[dict]:
    """Every parent/child pair by how much of the code changed. Candidates from before sizes were recorded are measured
    here; comparing their code takes seconds, so the table waits for a new candidate."""
    sign = 1 if higher else -1
    with O._lock:
        rows = O.db().execute("SELECT id, parent_id, score, metrics, code FROM candidates WHERE objective_id=? AND status='ok'",
                              (oid,)).fetchall()
    if oid in _sizes and _sizes[oid][0] == len(rows):
        return _sizes[oid][1]
    by_id = {r["id"]: r for r in rows}
    out = []
    for r in rows:
        p = by_id.get(r["parent_id"])
        if p is None or p["score"] is None or r["score"] is None:
            continue
        size = (json.loads(r["metrics"] or "{}").get("change") or {}).get("size")
        if size is None:
            size = O._change_size(p["code"], r["code"])
        if size is None:
            continue
        gain = (r["score"] - p["score"]) * sign
        out.append({"change_size": size, "gain": gain, "beat_parent": gain > 0, "win": False, "failed": False})
    table = _yield_table(out, lambda r: _size_band(r["change_size"]))
    _sizes[oid] = (len(rows), table)
    return table


def stages(oid: str) -> dict:
    """Where submissions stopped, and how much look-ahead work the two-tier test saved."""
    with O._lock:
        rows = O.db().execute("SELECT status, metrics, lookahead FROM candidates WHERE objective_id=? "
                              "ORDER BY seq DESC LIMIT 500", (oid,)).fetchall()
    out = {"recent": len(rows), "static": 0, "smoke": 0, "full": 0, "ok": 0, "lookahead_short": 0, "lookahead_full": 0}
    for r in rows:
        stage = (json.loads(r["metrics"] or "{}").get("stage") or {}).get("stopped")
        if stage in ("static", "smoke"):
            out[stage] += 1
        elif r["status"] == "error":
            out["full"] += 1
        else:
            out["ok"] += 1
        if r["lookahead"] == "short":
            out["lookahead_short"] += 1
        elif r["lookahead"] == "pass":
            out["lookahead_full"] += 1
    out["cuts_saved"] = out["lookahead_short"] * max(0, O.LOOKAHEAD_ACTIVE_CUTS - O.LOOKAHEAD_FIRST_CUTS)
    return out


def view(obj: dict) -> dict:
    """The operator's whole picture. It may show what agents are not shown: the holdout-based ranked score."""
    higher = O._higher(obj)
    if not _applicable(obj):
        return {"applicable": False}
    _seed(obj["id"])
    a = archive(obj, fresh=True)
    outcomes = _outcomes(obj)
    sign = 1 if higher else -1

    def island_row(i: dict) -> dict:
        best = max(i["members"], key=lambda m: m["is_score"] * sign)
        ranked = [m for m in i["members"] if m["score"] is not None]
        b = best.get("behaviour") or {}
        return {"island": i["island"], "size": i["size"], "recent_children": i["recent_children"], "best_seq": best["seq"],
                "best_in_sample": best["is_score"],
                "best_ranked": max((m["score"] for m in ranked), key=lambda v: v * sign, default=None),
                "rho_leader": best["rho_leader"], "niche": niche_label(best["niche"]) if best.get("niche") else None,
                "trades": b.get("trades"), "trades_per_day": b.get("trades_per_day"), "active_share": b.get("active_share")}

    islands = sorted((island_row(i) for i in a.get("islands") or []), key=lambda r: -r["best_in_sample"] * sign)
    return {"applicable": True, "candidates": a.get("members", 0), "islands": islands[:40], "strategies": len(islands),
            "niches": [{"cell": n["cell"], "label": n["label"], "strategies": n["strategies"], "best_seq": n["best"]["seq"],
                        "best_in_sample": n["best"]["is_score"]} for n in (a.get("niches") or {}).values()],
            "scorecard": scorecard(obj), "experiments": _experiment_rows(obj),
            "parent_arms": _yield_table(outcomes, lambda o: str(o["assignment"].get("parent_via") or "").split(":")[0]),
            "lenses": _yield_table(outcomes, lambda o: o["assignment"].get("lens")),
            "practices": _yield_table(outcomes, lambda o: o["assignment"].get("practices_version")),
            "change_sizes": _history_by_size(obj["id"], higher), "stages": stages(obj["id"]),
            "jump": jump_size(_leader_score(obj["id"], sign)),
            "share": ARCHIVE_SHARE}


@router.get("/objectives/{oid}/evolution")
async def evolution_view(oid: str) -> dict:
    """Islands, niches, lenses, instruction experiments and evaluation stages, for the console."""
    import asyncio

    return await asyncio.to_thread(view, O.get_objective(oid))


class VariantIn(BaseModel):
    mode: str = Field(..., pattern=r"^(improve|explore)$")
    name: str = Field(..., min_length=2, max_length=60)
    text: str = Field(..., min_length=20, max_length=VARIANT_TEXT_MAX)
    author: str = Field("operator", max_length=200)
    based_on: int | None = None


@router.post("/objectives/{oid}/variants")
async def post_variant(oid: str, req: VariantIn) -> dict:
    """A new instruction to put under test."""
    return add_variant(oid, req.mode, req.name, req.text, req.author, req.based_on)


@router.post("/objectives/{oid}/variants/{vid}/retire")
async def retire_variant(oid: str, vid: int) -> dict:
    _ensure()
    with O._lock:
        r = O.db().execute("SELECT control FROM instruction_variants WHERE id=? AND objective_id=?", (vid, oid)).fetchone()
        if r is None:
            raise HTTPException(status_code=404, detail="no such instruction")
        if r["control"]:
            raise HTTPException(status_code=409, detail="the control stays: without it the others have nothing to beat")
        O.db().execute("UPDATE instruction_variants SET status='retired', why='retired by the operator', retired_at=? "
                       "WHERE id=?", (time.time(), vid))
        O.db().commit()
    return {"retired": vid}
