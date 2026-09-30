"""The trade book: every trade of every task candidate, sorted into three classes, and what the
BIG WINNERS have in common that the rest lack.

The leaderboard ranks strategies by their daily curve; that says a strategy is not working but not
which of its trades are the problem. Every trade is one of

    BIG WINNER   its result per unit of size, after costs, is at least +threshold
    BIG LOSER    at most -threshold
    SCRATCH      anything in between: flat, small wins, small losses -- costs and noise

and the search optimises for BIG WINNERS only: a strategy that takes only the setups big winners
come from, and skips the rest, wins. So for a candidate (and for the whole swarm's book) the
review contrasts the market at each trade's entry -- every signal column of the task's rows, the
price's own path, the time of day -- between the big winners and everything else, per side, and
keeps only conditions that hold in BOTH halves of the in-sample period. That, with the best and
worst trades themselves, is what agents are shown: in-sample only, like everything they see.

Trades come from the task server (harness_actions on the candidate's kept actions); the entry
conditions from the in-sample rows export the analysis view is built from, as of the last row
BEFORE the entry (the decision was made at or before it). Nothing here knows the task: a server
whose action log has no trades (entry/exit/side/size/net) simply has an empty book.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import projects
from . import task_objectives as T

logger = logging.getLogger("freetoken.trade_book")
router = APIRouter(tags=["trade-book"])

DAY_NS = 86_400_000_000_000
BIG_QUANTILE = 0.8          # auto threshold: this quantile of |result per unit| over the book's in-sample trades
BIG_COST_FLOOR = 8          # ... and never below this many one-way costs (a "big" win must beat the costs well)
MAX_EVENTS = 5000           # trades per harness_actions call (the server's cap); one call per segment
MIN_SIDE_TRADES = 24        # fewer trades on a side than this: no conditions for it
MIN_RANGE_TRADES = 12
# How clearly a range must concentrate winners (or losers) to be reported -- a z-score. Every field
# is tried in four ranges on each side (hundreds of tests), so chance alone clears z = 2 several
# times a review; the whole book has thousands of trades and can be held to more.
Z_MIN = 2.5
Z_MIN_BOOK = 3.0
HALF_Z = 1.0                # ... and in each half of the period on its own
MAX_OVERLAP = 0.7           # a condition selecting mostly the same trades as a stronger one is left out
TOP_CONDITIONS = 4
EXAMPLES = 4
DATASET_DIR = "trade_book"  # under the project's data folder: the in-sample book as a dataset for agents

Cls = Literal["win", "loss", "scratch", "all"]


# ---------------------------------------------------------------------------------------------
# Storage (in the objectives database, next to the candidates it describes)
# ---------------------------------------------------------------------------------------------
_schema_conn: Any = None


def _db():
    from . import objectives as O

    global _schema_conn
    con = O.db()
    if _schema_conn is not con:
        with O._lock:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    objective_id TEXT NOT NULL, candidate_id TEXT NOT NULL, entry TEXT NOT NULL,
                    exit TEXT NOT NULL, entry_ns INTEGER NOT NULL, side INTEGER NOT NULL, size REAL NOT NULL,
                    bars INTEGER, gross REAL, net REAL NOT NULL, unit REAL NOT NULL,
                    holdout INTEGER NOT NULL, open INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS trades_obj_unit ON trades(objective_id, holdout, unit);
                CREATE INDEX IF NOT EXISTS trades_cand ON trades(candidate_id);
                CREATE TABLE IF NOT EXISTS trade_index (
                    candidate_id TEXT PRIMARY KEY, objective_id TEXT NOT NULL, n INTEGER NOT NULL DEFAULT 0,
                    ts REAL NOT NULL, error TEXT NOT NULL DEFAULT ''
                );
                """
            )
            con.commit()
        _schema_conn = con
    return con, O._lock


# Candidates whose trades belong in the book: scored, and not caught leaking or failed on review
# (a look-ahead's "trades" are what cheating looks like -- the worst thing to learn from).
_ELIGIBLE = "c.status='ok' AND c.lookahead != 'fail' AND c.audit != 'fail'"


def _ns(iso: str) -> int:
    return int(np.datetime64(str(iso).replace("Z", "").replace(" ", "T")[:29], "ns").astype(np.int64))


def trades_of(events: list[dict], split: str | None) -> list[tuple]:
    """(entry, exit, entry_ns, side, size, bars, gross, net, unit, holdout, open) per TRADE among a
    server's action-log events; events that are not trades (a charge schedule, ...) are skipped."""
    out, seen = [], set()
    for e in events:
        if not all(k in e for k in ("entry", "exit", "side", "net")):
            continue
        key = (e["entry"], e["side"])
        if key in seen:                       # a trade across a window boundary comes back twice
            continue
        seen.add(key)
        size = abs(float(e.get("size") or 1.0)) or 1.0
        net = float(e["net"])
        entry = str(e["entry"])
        out.append((entry, str(e["exit"]), _ns(entry), 1 if str(e["side"]).lower().startswith("l") else -1, size,
                    int(e.get("bars") or 0), float(e.get("gross") or 0.0), net, net / size,
                    int(bool(split) and entry[:10] >= split), int(bool(e.get("open")))))
    return out


async def index_candidate(obj: dict, cid: str) -> int:
    """Put one candidate's trades in the book (replacing any it had), from its kept actions. A
    problem with the candidate itself (no kept actions, actions the server cannot read) is recorded
    so it is not retried; the server being unreachable raises -- the book tries again later."""
    from .objectives import _kept_positions

    kept = _kept_positions(obj["id"], cid)
    con, lock = _db()
    try:
        if not kept.is_file():
            raise ValueError("no actions kept for this candidate")
        split = obj.get("split_date") or None
        events: list[dict] = []
        for a, b in ([(None, split), (split, None)] if split else [(None, None)]):
            out = await T.action_log(obj, kept, a, b, MAX_EVENTS)
            if out.get("problem"):
                raise ValueError(str(out["problem"]))
            events += out.get("events") or []
        rows = trades_of(events, split)
        err = "" if rows or not events else "the task server's action log has no trades"
    except ValueError as exc:
        rows, err = [], str(exc)[:500]
    with lock:
        con.execute("DELETE FROM trades WHERE candidate_id=?", (cid,))
        con.executemany(
            "INSERT INTO trades (objective_id, candidate_id, entry, exit, entry_ns, side, size, bars, gross, net, unit, "
            "holdout, open) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [(obj["id"], cid, *r) for r in rows])
        con.execute("INSERT INTO trade_index (candidate_id, objective_id, n, ts, error) VALUES (?,?,?,?,?) "
                    "ON CONFLICT(candidate_id) DO UPDATE SET n=excluded.n, ts=excluded.ts, error=excluded.error",
                    (cid, obj["id"], len(rows), time.time(), err))
        con.commit()
    return len(rows)


def forget(oid: str, cids: list[str] | None = None) -> None:
    """Drop trades that no longer describe the candidates (re-scored under a new rule, deleted)."""
    con, lock = _db()
    with lock:
        if cids is None:
            con.execute("DELETE FROM trades WHERE objective_id=?", (oid,))
            con.execute("DELETE FROM trade_index WHERE objective_id=?", (oid,))
        else:
            for c in cids:
                con.execute("DELETE FROM trades WHERE candidate_id=?", (c,))
                con.execute("DELETE FROM trade_index WHERE candidate_id=?", (c,))
        con.commit()
    _POOL.pop(oid, None)


_BUILDS: dict[str, dict] = {}


async def ensure_book(oid: str) -> None:
    """Index every scored candidate not yet in the book (one at a time; one build per objective)."""
    b = _BUILDS.get(oid)
    if b and b.get("running"):
        return
    from .objectives import get_objective

    b = _BUILDS[oid] = {"running": True, "done": 0, "total": 0, "started": time.time()}
    try:
        con, lock = _db()
        with lock:
            missing = [r[0] for r in con.execute(
                "SELECT c.id FROM candidates c WHERE c.objective_id=? AND c.status='ok' AND NOT EXISTS "
                "(SELECT 1 FROM trade_index i WHERE i.candidate_id=c.id) ORDER BY c.seq DESC", (oid,)).fetchall()]
        b["total"] = len(missing)
        for cid in missing:
            obj = get_objective(oid)
            if not T.is_task(obj):
                break
            await index_candidate(obj, cid)
            b["done"] += 1
    except HTTPException as exc:          # the server is down or needs sign-in: the rest waits for next time
        b["error"] = str(exc.detail)[:300]
        logger.warning("trade book of %s paused: %s", oid, exc.detail)
    except Exception as exc:  # noqa: BLE001
        b["error"] = str(exc)[:300]
        logger.exception("building the trade book of %s failed", oid)
    finally:
        b.update(running=False, finished=time.time())


RETRY_S = 120


def spawn_build(oid: str) -> None:
    b = _BUILDS.get(oid)
    if b and (b.get("running") or (b.get("error") and time.time() - b.get("finished", 0) < RETRY_S)):
        return
    from .objectives import _spawn

    _spawn(ensure_book(oid), f"trade book of {oid}")


# ---------------------------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------------------------
def _info(obj: dict) -> dict:
    return (obj.get("metric") or {}).get("task_info") or {}


_ADDITIVE: dict[str, bool] = {}


def additive(obj: dict) -> bool:
    """Whether results are P&L in the target's own units (a signed target) rather than returns."""
    oid = obj["id"]
    if oid not in _ADDITIVE:
        import json

        con, lock = _db()
        with lock:
            r = con.execute("SELECT metrics FROM candidates WHERE objective_id=? AND status='ok' "
                            "AND metrics LIKE '%curve_kind%' LIMIT 1", (oid,)).fetchone()
        if r is None:
            return False
        kind = ((json.loads(r[0] or "{}").get("task") or {}).get("curve_kind"))
        _ADDITIVE[oid] = kind == "additive"
    return _ADDITIVE[oid]


def cost_floor(obj: dict) -> float:
    if additive(obj):
        return 0.0
    cost = float((_info(obj).get("valuation") or {}).get("cost_bps") or 1.0)
    return BIG_COST_FLOOR * cost / 1e4


def _book_units(oid: str) -> np.ndarray:
    """Per-unit results of the book's distinct in-sample trades (one per entry and side)."""
    con, lock = _db()
    with lock:
        rows = con.execute(
            "SELECT AVG(t.unit) FROM trades t JOIN candidates c ON c.id=t.candidate_id "
            f"WHERE t.objective_id=? AND t.holdout=0 AND {_ELIGIBLE} GROUP BY t.entry_ns, t.side", (oid,)).fetchall()
    return np.array([r[0] for r in rows], dtype=float)


def threshold(obj: dict, units: np.ndarray | None = None) -> tuple[float, str]:
    """(the per-unit result that makes a trade BIG, where it came from). Set by the operator
    (metric.big_trade), else the BIG_QUANTILE of the book's |results|, floored at BIG_COST_FLOOR costs."""
    m = obj.get("metric") or {}
    if m.get("big_trade"):
        return float(m["big_trade"]), "set"
    u = _book_units(obj["id"]) if units is None else units
    floor = cost_floor(obj)
    if not len(u):
        return (floor or 1.0), "floor"
    q = float(np.quantile(np.abs(u), BIG_QUANTILE))
    return (max(q, floor), "auto") if q > 0 else ((floor or 1.0), "floor")


def classes(unit: np.ndarray, thr: float) -> np.ndarray:
    """+1 big winner, -1 big loser, 0 scratch."""
    return np.where(unit >= thr, 1, np.where(unit <= -thr, -1, 0)).astype(np.int8)


def fmt_unit(u: float | None, add: bool) -> str:
    if u is None or not math.isfinite(u):
        return "n/a"
    return f"{u:+.4g}" if add else f"{u * 1e4:+.1f} bps"


# ---------------------------------------------------------------------------------------------
# Entry conditions: the market as of the last row before each entry
# ---------------------------------------------------------------------------------------------
_SNAP: dict[str, dict] = {}
_SNAP_LOCK = threading.Lock()
PRICE_NAMES = ("Open", "High", "Low", "Close")


def snapshots(path: Path, target: str | None, add: bool) -> dict:
    """Minute snapshots of the in-sample rows (each row: the minute's LAST row, stamped with that
    row's own time), as {t, day, X, names}. Signal columns as they are, except price levels (walls,
    strikes, bands -- a median near the price's) which become their distance from the price in bps;
    plus the price's own path: its change over 5 and 30 minutes and since the session's first bar,
    where it sits in the day's range so far, and the minutes since the session opened."""
    key = f"{path}|{path.stat().st_mtime}|{target}|{add}"
    with _SNAP_LOCK:
        if key in _SNAP:
            return _SNAP[key]
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    num = [c for c, d in schema.items() if c != "t" and d.is_numeric()]
    snap = (lf.select(["t", *num]).sort("t")
            .group_by_dynamic("t", every="1m", closed="left", label="left")
            .agg([pl.col(c).drop_nulls().last() for c in num] + [pl.col("t").last().alias("_t")])
            .collect())
    t = snap["_t"].cast(pl.Datetime("ns")).cast(pl.Int64).to_numpy()
    day = t // DAY_NS
    price = snap[target].cast(pl.Float64).to_numpy() if target and target in snap.columns else None
    pmed = float(np.nanmedian(price)) if price is not None and np.isfinite(price).any() else float("nan")
    cols: dict[str, np.ndarray] = {}
    for c in num:
        if c == target or c in PRICE_NAMES:
            continue
        x = snap[c].cast(pl.Float64).to_numpy()
        fin = x[np.isfinite(x)]
        if len(fin) < 100 or float(fin.std()) == 0.0:
            continue
        if price is not None and not add and pmed > 0 and 0.8 < float(np.median(fin)) / pmed < 1.25:
            with np.errstate(all="ignore"):
                cols[f"{c}_vs_price_bps"] = (x / price - 1.0) * 1e4
            continue
        cols[c] = x
    if len(t):
        starts = np.r_[0, np.flatnonzero(np.diff(day)) + 1]
        first = starts[np.searchsorted(starts, np.arange(len(t)), side="right") - 1]
        cols["minutes_into_session"] = (t - t[first]) / 60e9
        if price is not None:
            idx = np.arange(len(t))

            def back(k: int) -> np.ndarray:
                j = idx - k
                out = np.full(len(t), np.nan)
                ok = j >= first
                out[ok] = price[j[ok]]
                return out

            with np.errstate(all="ignore"):
                chg = (lambda a, b: a - b) if add else (lambda a, b: (a / b - 1.0) * 1e4)
                sfx = "" if add else "_bps"
                cols["price_chg_5m" + sfx] = chg(price, back(5))
                cols["price_chg_30m" + sfx] = chg(price, back(30))
                cols["price_since_open" + sfx] = chg(price, price[first])
                rng = (pl.DataFrame({"p": price, "d": day})
                       .with_columns(hi=pl.col("p").cum_max().over("d"), lo=pl.col("p").cum_min().over("d")))
                hi, lo = rng["hi"].to_numpy(), rng["lo"].to_numpy()
                cols["price_in_day_range"] = np.where(hi > lo, (price - lo) / (hi - lo), np.nan)
    names = list(cols)
    X = np.column_stack([cols[n].astype(np.float32) for n in names]) if names else np.zeros((len(t), 0), np.float32)
    out = {"t": t, "day": day, "X": X, "names": names}
    with _SNAP_LOCK:
        _SNAP.clear()                         # one task's snapshots at a time: they are ~100 MB
        _SNAP[key] = out
    return out


def entry_conditions(snap: dict, entry_ns: np.ndarray) -> np.ndarray:
    """The snapshot row as of the last row strictly BEFORE each entry, the same day; NaN otherwise."""
    t = snap["t"]
    i = np.searchsorted(t, entry_ns, side="left") - 1
    ok = (i >= 0) & (snap["day"][np.clip(i, 0, None)] == entry_ns // DAY_NS) if len(t) else np.zeros(len(entry_ns), bool)
    X = np.full((len(entry_ns), snap["X"].shape[1]), np.nan, np.float32)
    X[ok] = snap["X"][i[ok]]
    return X


# ---------------------------------------------------------------------------------------------
# The review
# ---------------------------------------------------------------------------------------------
def _share(m: np.ndarray) -> float | None:
    return round(float(m.mean()), 4) if len(m) else None


def _conditions(v: np.ndarray, u: np.ndarray, X: np.ndarray, names: list[str], when: np.ndarray,
                z_min: float = Z_MIN) -> tuple[list, list]:
    """Ranges of one field (bottom 20/40% or top 40/20% of this side's entries) where BIG WINNERS
    concentrate (good) or BIG LOSERS do (bad), each also holding in both halves of the period.

    Good ranks on how much MORE OFTEN a trade there is a big winner (a z-score of the winner share
    against this side's base), and also needs winners-minus-losers no worse than the base: a
    quiet market that merely has fewer losers is not where the big winners are, and a wild one
    with more of both is not a winner condition either. Bad ranks the same way on big losers."""
    n = len(v)
    if n < MIN_SIDE_TRADES:
        return [], []
    half = when <= np.median(when)
    vf = v.astype(float)
    cands: dict[str, list] = {"good": [], "bad": []}

    def holds(y: np.ndarray, m: np.ndarray, part: np.ndarray) -> bool:
        """Whether y (0/1) happens clearly more often inside m than on the whole of `part` (a z of
        HALF_Z): a real effect shows in each half on its own, chance rarely in both."""
        k, p = int((m & part).sum()), float(y[part].mean()) if part.any() else 0.0
        if k < 5 or p <= 0 or p >= 1:
            return False
        return (y[m & part].mean() - p) / math.sqrt(p * (1 - p) / k) >= HALF_Z

    for f in range(X.shape[1]):
        x = X[:, f]
        ok = np.isfinite(x)
        nv = int(ok.sum())
        if nv < MIN_SIDE_TRADES:
            continue
        vv, xs, h = vf[ok], x[ok], half[ok]
        win, loss = (vv == 1).astype(float), (vv == -1).astype(float)
        qs = np.quantile(xs, [0.2, 0.4, 0.6, 0.8])
        if qs[0] == qs[-1]:
            continue
        best: dict[str, tuple | None] = {"good": None, "bad": None}
        for op, q in (("<=", qs[0]), ("<=", qs[1]), (">=", qs[2]), (">=", qs[3])):
            m = xs <= q if op == "<=" else xs >= q
            k = int(m.sum())
            if k < max(MIN_RANGE_TRADES, 0.1 * nv) or k > 0.6 * nv:      # a range must actually filter
                continue
            net = vv[m].mean() - vv.mean()                               # winners minus losers vs the base
            for kind, y, sign in (("good", win, 1.0), ("bad", loss, -1.0)):
                p = y.mean()
                if p <= 0 or p >= 1 or sign * net < 0:
                    continue
                z = (y[m].mean() - p) / math.sqrt(p * (1 - p) / k)
                if z >= z_min and holds(y, m, h) and holds(y, m, ~h) and (best[kind] is None or z > best[kind][0]):
                    best[kind] = (z, op, q, m)
        for kind, rec in best.items():
            if rec is None:
                continue
            z, op, q, m = rec
            full = np.zeros(n, bool)
            full[np.flatnonzero(ok)[m]] = True
            uu = u[ok]
            cands[kind].append({
                "field": names[f], "op": op, "value": float(q), "trades": int(m.sum()),
                "kept": round(float(m.mean()), 3), "win": _share(vv[m] == 1), "loss": _share(vv[m] == -1),
                "win_all": _share(vv == 1), "loss_all": _share(vv == -1),
                "mean": float(uu[m].mean()), "mean_all": float(uu.mean()), "z": round(float(z), 2), "_mask": full})
    out = []
    for kind in ("good", "bad"):
        picked: list[dict] = []
        for c in sorted(cands[kind], key=lambda c: -c["z"]):
            if any((c["_mask"] & p["_mask"]).sum() / max(1, (c["_mask"] | p["_mask"]).sum()) > MAX_OVERLAP for p in picked):
                continue
            picked.append(c)
            if len(picked) >= TOP_CONDITIONS:
                break
        out.append([{k: v for k, v in c.items() if k != "_mask"} for c in picked])
    return out[0], out[1]


def _local_minutes(entry_ns: np.ndarray, tz: str) -> np.ndarray:
    s = pl.Series(entry_ns, dtype=pl.Int64).cast(pl.Datetime("ns")).dt.replace_time_zone("UTC")
    try:
        s = s.dt.convert_time_zone(tz)
    except Exception:  # noqa: BLE001 -- an unknown zone: UTC
        pass
    return (s.dt.hour().cast(pl.Int64) * 60 + s.dt.minute().cast(pl.Int64)).to_numpy()


def _local_text(entry_ns: np.ndarray, tz: str) -> list[str]:
    s = pl.Series(entry_ns, dtype=pl.Int64).cast(pl.Datetime("ns")).dt.replace_time_zone("UTC")
    try:
        s = s.dt.convert_time_zone(tz)
    except Exception:  # noqa: BLE001
        pass
    return s.dt.strftime("%Y-%m-%d %H:%M").to_list()


def review(unit: np.ndarray, side: np.ndarray, entry_ns: np.ndarray, bars: np.ndarray, X: np.ndarray,
           names: list[str], thr: float, tz: str = "UTC", step_s: float | None = None,
           z_min: float = Z_MIN) -> dict[str, Any]:
    """The three classes of a set of trades, and what separates the BIG WINNERS from the rest."""
    v = classes(unit, thr)
    n = len(v)
    out: dict[str, Any] = {"trades": n, "threshold": thr}
    if not n:
        return out
    for name, k in (("big_winners", 1), ("big_losers", -1), ("scratch", 0)):
        m = v == k
        held = bars[m] * (step_s or 0) / 60 if step_s else bars[m]
        out[name] = {"n": int(m.sum()), "share": _share(m), "total": float(unit[m].sum()),
                     "mean": float(unit[m].mean()) if m.any() else None,
                     "held": float(np.median(held)) if m.any() else None}
    out["held_unit"] = "minutes" if step_s else "bars"
    out["total"] = float(unit.sum())
    sides = {}
    for label, s in (("long", 1), ("short", -1)):
        m = side == s
        if not m.any():
            continue
        good, bad = _conditions(v[m], unit[m], X[m], names, entry_ns[m], z_min)
        sides[label] = {"trades": int(m.sum()), "win": _share(v[m] == 1), "loss": _share(v[m] == -1),
                        "mean": float(unit[m].mean()), "winner_conditions": good, "loser_conditions": bad}
    out["sides"] = sides
    mins = _local_minutes(entry_ns, tz)
    slots = []
    for s in np.unique(mins // 30 * 30):
        m = mins // 30 * 30 == s
        if m.sum() < 8:
            continue
        slots.append({"from": f"{s // 60:02d}:{s % 60:02d}", "to": f"{(s + 30) // 60:02d}:{(s + 30) % 60:02d}",
                      "trades": int(m.sum()), "win": _share(v[m] == 1), "loss": _share(v[m] == -1),
                      "mean": float(unit[m].mean())})
    out["time_of_day"] = {"tz": tz, "slots": slots}
    # Each example shows the fields its own side's big winners are told apart by.
    focus = {s: [names.index(c["field"]) for c in (sides.get(lab) or {}).get("winner_conditions", [])[:3]]
             for lab, s in (("long", 1), ("short", -1))}
    local = _local_text(entry_ns, tz)

    def ex(i: int) -> dict:
        return {"when": local[i], "side": "long" if side[i] > 0 else "short", "unit": float(unit[i]),
                "held": float(bars[i] * (step_s or 0) / 60) if step_s else int(bars[i]),
                "at_entry": {names[j]: (None if not np.isfinite(X[i, j]) else round(float(X[i, j]), 4))
                             for j in focus.get(int(side[i]), [])}}
    def pick(idx: np.ndarray, k: int) -> list[dict]:
        """The first examples of distinct days: one big day would otherwise fill the list."""
        out_, days = [], set()
        for i in idx:
            if v[i] != k or entry_ns[i] // DAY_NS in days:
                continue
            days.add(entry_ns[i] // DAY_NS)
            out_.append(ex(int(i)))
            if len(out_) >= EXAMPLES:
                break
        return out_
    order = np.argsort(unit)
    out["best"] = pick(order[::-1], 1)
    out["worst"] = pick(order, -1)
    return out


def _cond_text(c: dict, add: bool, what: str) -> str:
    val = f"{c['value']:.4g}"

    def x(a: float, b: float) -> str:
        return f"{a:.0%} ({a / b:.1f}x the {b:.0%} base)" if b else f"{a:.0%}"
    rate = (f"big winners {x(c['win'], c['win_all'])}, big losers {c['loss']:.0%} vs {c['loss_all']:.0%}"
            if what == "win" else
            f"big losers {x(c['loss'], c['loss_all'])}, big winners {c['win']:.0%} vs {c['win_all']:.0%}")
    return (f"{c['field']} {c['op']} {val} ({c['kept']:.0%} of these entries, {c['trades']} trades): {rate}; "
            f"avg {fmt_unit(c['mean'], add)} vs {fmt_unit(c['mean_all'], add)} (z {c['z']:.1f})")


def render(r: dict, add: bool, title: str) -> str:
    """A review in words, for an agent's prompt."""
    if not r.get("trades"):
        return f"{title}: no trades."
    thr = fmt_unit(r["threshold"], add).lstrip("+")
    w, l, s = r["big_winners"], r["big_losers"], r["scratch"]
    unit = r.get("held_unit", "bars")

    def cls(name: str, c: dict) -> str:
        held = f", held {c['held']:.0f} {unit} (median)" if c.get("held") is not None else ""
        return f"{name} {c['n']} ({c['share']:.0%}) = {fmt_unit(c['total'], add)} in all{held}"
    lines = [f"{title} -- {r['trades']} trades; BIG = at least {thr} per unit of size after costs:",
             "  " + "; ".join([cls("BIG WINNERS", w), cls("BIG LOSERS", l), cls("SCRATCH", s)])
             + f"; all together {fmt_unit(r['total'], add)}."]
    if s["n"] and s["total"] < 0:
        lines.append(f"  Scratch trades alone cost {fmt_unit(s['total'], add)}: every one you skip is a gain.")
    for label, sd in (r.get("sides") or {}).items():
        lines.append(f"  {label.upper()} ({sd['trades']} trades): big winners {sd['win']:.0%}, big losers {sd['loss']:.0%}, "
                     f"avg {fmt_unit(sd['mean'], add)}.")
        if sd["winner_conditions"]:
            lines.append(f"    Where {label} BIG WINNERS concentrate at entry (held in both halves of the period):")
            lines += [f"    + {_cond_text(c, add, 'win')}" for c in sd["winner_conditions"]]
        elif sd["trades"] >= MIN_SIDE_TRADES:
            lines.append(f"    No entry condition separates the {label} big winners reliably -- the entry signal itself "
                         "does not find them yet.")
        if sd["loser_conditions"]:
            lines.append(f"    Where {label} BIG LOSERS concentrate (avoid):")
            lines += [f"    - {_cond_text(c, add, 'loss')}" for c in sd["loser_conditions"]]
    slots = (r.get("time_of_day") or {}).get("slots") or []
    if len(slots) >= 2:
        lines.append(f"  By entry time ({r['time_of_day']['tz']}; half hour: trades, big winners / big losers, avg): "
                     + "; ".join(f"{x['from']} {x['trades']}, {x['win']:.0%}/{x['loss']:.0%}, {fmt_unit(x['mean'], add)}"
                                 for x in slots))

    def exs(xs: list[dict]) -> str:
        return "; ".join(f"{x['when']} {x['side']} {fmt_unit(x['unit'], add)}"
                         + (" [" + ", ".join(f"{k}={v}" for k, v in x["at_entry"].items()) + "]" if x["at_entry"] else "")
                         for x in xs)
    if r.get("best"):
        lines.append("  Best trades: " + exs(r["best"]))
    if r.get("worst"):
        lines.append("  Worst trades: " + exs(r["worst"]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Reviews of a candidate and of the whole book
# ---------------------------------------------------------------------------------------------
async def _snap(obj: dict) -> dict | None:
    """The in-sample rows' snapshots (None when the rows cannot be had)."""
    try:
        folder = await T.export_dir(obj, obj.get("split_date") or None)
    except HTTPException as exc:
        logger.warning("trade review of %s without entry conditions: %s", obj["id"], exc.detail)
        return None
    info = _info(obj)
    target = (obj.get("metric") or {}).get("target") or info.get("target")
    return await asyncio.to_thread(snapshots, folder / "rows.parquet", target, additive(obj))


def _rows(oid: str, where: str, args: tuple) -> dict[str, np.ndarray]:
    con, lock = _db()
    with lock:
        rows = con.execute(f"SELECT t.entry_ns, t.side, AVG(t.unit), AVG(t.bars), COUNT(DISTINCT t.candidate_id) "
                           f"FROM trades t JOIN candidates c ON c.id=t.candidate_id WHERE t.objective_id=? AND {where} "
                           "GROUP BY t.entry_ns, t.side ORDER BY t.entry_ns", (oid, *args)).fetchall()
    a = np.array(rows, dtype=float).reshape(-1, 5)
    return {"entry_ns": a[:, 0].astype(np.int64), "side": a[:, 1].astype(np.int64), "unit": a[:, 2],
            "bars": a[:, 3], "takers": a[:, 4].astype(np.int64)}


async def _review(obj: dict, tr: dict[str, np.ndarray], thr: float, z_min: float = Z_MIN) -> dict:
    snap = await _snap(obj)
    info = _info(obj)
    if snap is not None:
        X, names = entry_conditions(snap, tr["entry_ns"]), snap["names"]
    else:
        X, names = np.zeros((len(tr["unit"]), 0), np.float32), []
    step = (info.get("shape") or {}).get("step_s")
    return await asyncio.to_thread(review, tr["unit"], tr["side"], tr["entry_ns"], tr["bars"], X, names, thr,
                                   info.get("display_tz") or "UTC", float(step) if step else None, z_min)


async def candidate_review(obj: dict, cid: str, segment: str = "in_sample") -> dict:
    """One candidate's review: in-sample (what agents see) or the holdout (the operator's)."""
    hold = 1 if segment == "holdout" else 0
    tr = _rows(obj["id"], "t.candidate_id=? AND t.holdout=?", (cid, hold))
    thr, src = threshold(obj)
    r = await _review(obj, tr, thr)
    return {**r, "threshold_source": src, "segment": segment}


_POOL: dict[str, tuple[tuple, dict]] = {}


async def pool_review(obj: dict) -> dict:
    """The whole swarm's in-sample book: every distinct trade (entry and side) of every eligible
    candidate, reviewed together. Cached until the book or the threshold changes."""
    oid = obj["id"]
    tr = _rows(oid, f"t.holdout=0 AND {_ELIGIBLE}", ())
    thr, src = threshold(obj, tr["unit"])
    con, lock = _db()
    with lock:
        k = con.execute("SELECT COUNT(*), COALESCE(MAX(ts), 0) FROM trade_index WHERE objective_id=?", (oid,)).fetchone()
    key = (tuple(k), round(thr, 9), len(tr["unit"]))
    hit = _POOL.get(oid)
    if hit and hit[0] == key:
        return hit[1]
    r = await _review(obj, tr, thr, Z_MIN_BOOK)
    r.update(threshold_source=src, candidates=int(k[0]))
    _POOL[oid] = (key, r)
    try:
        await asyncio.to_thread(write_dataset, obj)
    except Exception:  # noqa: BLE001 -- a convenience for agents; never costs the review
        logger.exception("writing the trade book dataset of %s failed", oid)
    return r


def dataset_view(obj: dict) -> str:
    from . import datasource

    return datasource._view_name(Path(DATASET_DIR) / f"trades_{obj['id']}.parquet")


def write_dataset(obj: dict) -> str | None:
    """The in-sample book as a dataset of the project (<data_dir>/trade_book/trades_<objective>.parquet):
    one row per candidate's trade -- seq, entry/exit (UTC), side, size, bars, net, unit, cls -- so an
    agent can join it to the rows (as of the entry) and test a filter on real winners and losers.
    Only in-sample trades: the holdout never reaches an agent."""
    proj = projects.get(obj["project_id"]) or {}
    data_dir = proj.get("data_dir")
    if not data_dir or not Path(data_dir).is_dir():
        return None
    thr, _ = threshold(obj)
    con, lock = _db()
    with lock:
        rows = con.execute(
            "SELECT c.seq, t.entry, t.exit, t.side, t.size, t.bars, t.net, t.unit FROM trades t "
            f"JOIN candidates c ON c.id=t.candidate_id WHERE t.objective_id=? AND t.holdout=0 AND {_ELIGIBLE} "
            "ORDER BY c.seq, t.entry_ns", (obj["id"],)).fetchall()
    df = pl.DataFrame(rows, orient="row", schema={"seq": pl.Int64, "entry": pl.String, "exit": pl.String,
                                                  "side": pl.Int64, "size": pl.Float64, "bars": pl.Int64,
                                                  "net": pl.Float64, "unit": pl.Float64})
    df = df.with_columns(pl.col("entry").str.to_datetime(time_unit="ns"), pl.col("exit").str.to_datetime(time_unit="ns"),
                         pl.when(pl.col("unit") >= thr).then(pl.lit("big_winner"))
                         .when(pl.col("unit") <= -thr).then(pl.lit("big_loser")).otherwise(pl.lit("scratch")).alias("cls"))
    dst = Path(data_dir) / DATASET_DIR / f"trades_{obj['id']}.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(dst)
    return dataset_view(obj)


# ---------------------------------------------------------------------------------------------
# What agents get
# ---------------------------------------------------------------------------------------------
GOAL = ("Every trade is a BIG WINNER, a BIG LOSER or a SCRATCH trade (flat, small win or small loss). Optimise "
        "for BIG WINNERS ONLY: find what the market looks like at the entry of big winners and NOT at the others, "
        "and enter only then. Every scratch and every big loser you skip is a gain -- scratch trades pay costs for "
        "nothing and are most of what a weak strategy trades. More trades are not the goal; more of the RIGHT "
        "trades are.")


async def after_scoring(obj: dict, cid: str, seq: int | None = None) -> str | None:
    """Index a freshly scored candidate and review its in-sample trades, for the submitting agent.
    Never raises: the review is advice, the score stands without it."""
    try:
        if not await index_candidate(obj, cid):
            return None                       # a task whose actions are not trades, or none were made
        r = await candidate_review(obj, cid)
        if not r.get("trades"):
            return None
        return render(r, additive(obj), f"TRADE REVIEW of your candidate{f' #{seq}' if seq else ''} (in-sample)")
    except Exception:  # noqa: BLE001
        logger.exception("trade review of %s failed", cid)
        return None


async def brief(obj: dict, parent_id: str | None) -> dict | None:
    """The trade book's part of an agent's context: the goal, the swarm's book and the parent's review."""
    try:
        spawn_build(obj["id"])
        pool = await pool_review(obj)
        if not pool.get("trades"):
            return None                       # nothing to learn from yet (or the task's actions are not trades)
        add = additive(obj)
        out: dict[str, Any] = {"goal": GOAL, "threshold": fmt_unit(pool.get("threshold"), add).lstrip("+"),
                               "dataset": dataset_view(obj)}
        out["pool"] = render(pool, add, f"SWARM TRADE BOOK (distinct in-sample trades of {pool.get('candidates', 0)} "
                                        "candidates)")
        if parent_id:
            con, lock = _db()
            with lock:
                have = con.execute("SELECT 1 FROM trade_index WHERE candidate_id=?", (parent_id,)).fetchone()
            if not have:
                await index_candidate(obj, parent_id)
            r = await candidate_review(obj, parent_id)
            if r.get("trades"):
                out["parent"] = render(r, add, "TRADE REVIEW of the candidate you are improving (in-sample)")
        return out
    except Exception:  # noqa: BLE001 -- the iteration goes on without it
        logger.exception("trade book brief of %s failed", obj["id"])
        return None


# ---------------------------------------------------------------------------------------------
# Routes (the console's trade leaderboard)
# ---------------------------------------------------------------------------------------------
def _objective(oid: str) -> dict:
    from .objectives import get_objective

    obj = get_objective(oid)
    if not T.is_task(obj):
        raise HTTPException(status_code=409, detail="the trade book is kept for task objectives only")
    return obj


@router.get("/objectives/{oid}/trades")
async def list_trades(oid: str, cls: Cls = "win", side: Literal["all", "long", "short"] = "all",
                      segment: Literal["all", "in_sample", "holdout"] = "all", limit: int = 100,
                      offset: int = 0) -> dict:
    """The trade leaderboard: distinct trades (entry, exit, side) of eligible candidates, one class
    at a time, best first (big losers worst first), with every candidate that took each one."""
    obj = _objective(oid)
    spawn_build(oid)
    thr, src = threshold(obj)
    add = additive(obj)
    where = ["t.objective_id=?", _ELIGIBLE]
    args: list[Any] = [oid]
    if side != "all":
        where.append("t.side=?")
        args.append(1 if side == "long" else -1)
    if segment != "all":
        where.append("t.holdout=?")
        args.append(1 if segment == "holdout" else 0)
    counts_where, counts_args = " AND ".join(where), list(args)
    if cls == "win":
        where.append("t.unit >= ?")
        args.append(thr)
        order = "unit DESC"
    elif cls == "loss":
        where.append("t.unit <= ?")
        args.append(-thr)
        order = "unit ASC"
    elif cls == "scratch":
        where += ["t.unit > ?", "t.unit < ?"]
        args += [-thr, thr]
        order = "unit DESC"
    else:
        order = "unit DESC"
    con, lock = _db()
    with lock:
        rows = con.execute(
            "SELECT t.entry, t.exit, t.side, AVG(t.unit) AS unit, AVG(t.net) AS net, MAX(t.size) AS size, "
            "MAX(t.bars) AS bars, MAX(t.holdout) AS holdout, MAX(t.open) AS open, "
            "GROUP_CONCAT(c.seq || ':' || c.id) AS takers "
            f"FROM trades t JOIN candidates c ON c.id=t.candidate_id WHERE {' AND '.join(where)} "
            f"GROUP BY t.entry, t.exit, t.side ORDER BY {order} LIMIT ? OFFSET ?",
            (*args, max(1, min(limit, 500)), max(0, offset))).fetchall()
        cnt = con.execute(
            "SELECT holdout, SUM(u >= ?), SUM(u <= ?), SUM(u > ? AND u < ?), SUM(CASE WHEN u >= ? THEN u END), "
            "SUM(CASE WHEN u <= ? THEN u END), SUM(CASE WHEN u > ? AND u < ? THEN u END) FROM ("
            "SELECT t.holdout AS holdout, AVG(t.unit) AS u FROM trades t JOIN candidates c ON c.id=t.candidate_id "
            f"WHERE {counts_where} GROUP BY t.entry, t.exit, t.side) GROUP BY holdout",
            (thr, -thr, -thr, thr, thr, -thr, -thr, thr, *counts_args)).fetchall()
        idx = con.execute("SELECT COUNT(*), SUM(n), SUM(error != '') FROM trade_index WHERE objective_id=?",
                          (oid,)).fetchone()
        total = con.execute("SELECT COUNT(*) FROM candidates WHERE objective_id=? AND status='ok'", (oid,)).fetchone()[0]
    counts = {("holdout" if r[0] else "in_sample"): {"win": r[1] or 0, "loss": r[2] or 0, "scratch": r[3] or 0,
                                                      "win_total": r[4] or 0.0, "loss_total": r[5] or 0.0,
                                                      "scratch_total": r[6] or 0.0} for r in cnt}
    out = []
    for r in rows:
        takers = sorted(((int(s), c) for s, c in (x.split(":", 1) for x in (r["takers"] or "").split(",") if ":" in x)))
        u = float(r["unit"])
        out.append({"entry": r["entry"], "exit": r["exit"], "side": "long" if r["side"] > 0 else "short", "unit": u,
                    "net": float(r["net"]), "size": float(r["size"]), "bars": r["bars"], "holdout": bool(r["holdout"]),
                    "open": bool(r["open"]), "cls": "win" if u >= thr else "loss" if u <= -thr else "scratch",
                    "takers": [{"seq": s, "id": c} for s, c in takers]})
    b = _BUILDS.get(oid) or {}
    return {"threshold": thr, "threshold_source": src, "additive": add, "split_date": obj.get("split_date"),
            "display_tz": _info(obj).get("display_tz") or "UTC", "counts": counts, "rows": out,
            "indexed": idx[0] or 0, "indexed_trades": idx[1] or 0, "index_errors": idx[2] or 0, "candidates": total,
            "building": {k: b.get(k) for k in ("running", "done", "total")} if b else None}


@router.get("/objectives/{oid}/trades/review")
async def trades_review(oid: str, candidate: str | None = None,
                        segment: Literal["in_sample", "holdout"] = "in_sample") -> dict:
    """What the big winners have in common: of one candidate, or of the whole book (in-sample)."""
    obj = _objective(oid)
    add = additive(obj)
    if candidate:
        r = await candidate_review(obj, candidate, segment)
        title = f"Candidate review ({segment.replace('_', '-')})"
    else:
        r = await pool_review(obj)
        title = "Swarm trade book (distinct in-sample trades)"
    return {"review": r, "text": render(r, add, title), "additive": add}


class Threshold(BaseModel):
    #: The per-unit result that makes a trade BIG: bps for a returns target, the target's own units
    #: for a signed one. None or 0 = automatic.
    value: float | None = None


@router.post("/objectives/{oid}/trades/threshold")
async def set_threshold(oid: str, req: Threshold) -> dict:
    import json

    from . import objectives as O

    obj = _objective(oid)
    m = dict(obj["metric"])
    if req.value:
        m["big_trade"] = abs(float(req.value)) / (1.0 if additive(obj) else 1e4)
    else:
        m.pop("big_trade", None)
    with O._lock:
        O.db().execute("UPDATE objectives SET metric=?, updated_at=? WHERE id=?", (json.dumps(m), time.time(), oid))
        O.db().commit()
    _POOL.pop(oid, None)
    thr, src = threshold(O.get_objective(oid))
    return {"threshold": thr, "threshold_source": src}


@router.post("/objectives/{oid}/trades/reindex")
async def reindex(oid: str) -> dict:
    _objective(oid)
    forget(oid)
    spawn_build(oid)
    return {"ok": True}


@router.get("/objectives/{oid}/candidates/{cid}/trade-review")
async def agent_trade_review(oid: str, cid: str) -> dict:
    """A candidate's in-sample review in words -- the agents' tool (never the holdout)."""
    obj = _objective(oid)
    con, lock = _db()
    with lock:
        have = con.execute("SELECT 1 FROM trade_index WHERE candidate_id=?", (cid,)).fetchone()
    if not have:
        await index_candidate(obj, cid)
    r = await candidate_review(obj, cid, "in_sample")
    return {"text": render(r, additive(obj), "TRADE REVIEW (in-sample)")}
