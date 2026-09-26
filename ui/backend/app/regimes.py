"""Regime Lab: which verified strategy works in which market regime, and a router that trades
each regime with the one that works there.

A strategy's edge is rarely uniform. The same momentum rule that earns in short-gamma,
high-volatility stretches bleeds in long-gamma chop. The lab makes that visible and usable:

1. **Regimes.** Either a crossing of fields -- each smoothed by a trailing mean, ranked
   against its OWN trailing window and cut into buckets (``ft.regime_grid``; e.g. GEX tercile
   x IntrVol tercile = 9 regimes) -- or a library regime module's ``detect(df)``. Labels are
   computed in the sandbox by exactly the code a candidate script runs, so the routes the lab
   picks mean the same thing in the script.
2. **Members.** Verified candidates (scored, look-ahead pass, not disqualified, not
   ensembles), each replayed once on the full data to capture its bar-level positions (cached
   per data version -- a candidate's code never changes).
3. **Measurement.** On the dataset's bar grid, with the scorer's own rule (the position set at
   bar t-1 earns bar t's return, less cost_bps on the trade made at t-1), every bar's net
   return of every member is attributed to the regime in force when its position was set, and
   summed per day. Per (regime, member): daily Sharpe over the days the regime occurred, P&L,
   hit rate -- in-sample, in each HALF of the in-sample period, and in the holdout.
4. **Routing.** Start from the best single member everywhere; a regime switches to another
   member only when it beats that baseline by SWITCH_MARGIN Sharpe in BOTH in-sample halves,
   and goes flat only when the baseline loses in both (see suggest_routes for why the greedy
   best-per-regime rule was dropped). The routed position series is then marked to market
   EXACTLY -- switching costs included, and at ~2-3 regime changes a day they are large -- and
   its P&L is attributed to the member traded: the contribution chart.
5. **A candidate.** ``router_code`` writes the router as an ordinary script
   (``ft.regime_grid`` + ``ft.candidate_positions`` + ``ft.route``) that goes through the whole
   pipeline: look-ahead test (members re-run on the truncated data too), audit, the hidden
   holdout, the ranking.

**Agents see in-sample only.** ``compact`` (their tool result and brief) carries no holdout
number; routes are chosen from in-sample halves alone. The operator's console shows the
holdout next to everything so they can see whether regime edges persisted.
"""

from __future__ import annotations

import asyncio
import glob
import hashlib
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any, Literal

import duckdb
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import datasource, projects

logger = logging.getLogger("freetoken.regimes")

router = APIRouter(tags=["regime-lab"])

CACHE_ROOT = Path(__file__).resolve().parent.parent / "regime_cache"
MAX_MEMBERS = 8
DEFAULT_MEMBERS = 6
DEFAULT_FIELDS = ("GEX", "IntrVol")
MAX_REPLAY_SECONDS = 150          # a member slower than this makes the router script too slow to score
MIN_DAYS = 15                     # in-sample days a regime needs before it is routed
MIN_SHARE = 0.02                  # ... and its share of in-sample bars
MIN_HALF_DAYS = 5                 # days in each half for that half's Sharpe to count
SWITCH_MARGIN = 0.5               # Sharpe a member must beat the baseline by, in BOTH halves, to take a regime
FLAT = "flat"
SKIP_LABELS = {"warmup", "unknown", "nan", "None", ""}

_jobs: dict[str, dict] = {}
_events: dict[str, asyncio.Event] = {}
_ready = False


def _db():
    global _ready
    from .objectives import _lock, db

    conn = db()
    if _ready:
        return conn
    with _lock:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS regime_labs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL, objective_id TEXT NOT NULL,
                ts REAL NOT NULL, author TEXT, spec TEXT NOT NULL, members TEXT NOT NULL,
                routes TEXT NOT NULL, summary TEXT NOT NULL, result TEXT NOT NULL, candidate_seq INTEGER
            );
            CREATE INDEX IF NOT EXISTS regime_labs_obj ON regime_labs(objective_id, ts);
            """
        )
        conn.commit()
    _ready = True
    return conn


def _lock():
    from .objectives import _lock as lock

    return lock


# =======================================================================================
# Requests
# =======================================================================================
class FieldSplit(BaseModel):
    field: str = Field(..., max_length=120)
    n: int = Field(3, ge=2, le=5)


class Split(BaseModel):
    kind: Literal["fields", "module"] = "fields"
    fields: list[FieldSplit] = Field(default_factory=list, max_length=2)
    module: str | None = Field(None, max_length=48)
    smooth: int = Field(360, ge=0, le=50_000, description="trailing-mean bars before ranking (0 = raw)")
    window_days: float = Field(20, ge=2, le=250, description="sessions each field is ranked against")


class LabReq(BaseModel):
    split: Split | None = None
    members: list[int] | None = Field(None, max_length=MAX_MEMBERS)
    routes: dict[str, int | None] | None = None
    author: str = Field("operator", max_length=200)
    wait: bool = False
    compact: bool = False


class SubmitReq(BaseModel):
    routes: dict[str, int | None] | None = None
    model: str = Field("regime-lab (operator)", max_length=200)
    rationale: str = Field("", max_length=4000)


def describe(spec: dict) -> str:
    if spec.get("kind") == "module":
        return f"library regime module {spec.get('module')}"
    parts = " x ".join(f"{f['field']} ({f['n']} buckets)" for f in spec.get("fields") or [])
    smooth = int(spec.get("smooth") or 0)
    return (f"{parts}; each {'smoothed over ' + str(smooth) + ' bars, ' if smooth > 1 else ''}"
            f"ranked against its own trailing {spec.get('window_days')} sessions")


def _norm_spec(split: Split) -> dict:
    if split.kind == "module":
        if not split.module:
            raise HTTPException(status_code=400, detail="split.kind='module' needs split.module (a regime module name)")
        return {"kind": "module", "module": split.module}
    if not split.fields:
        raise HTTPException(status_code=400, detail="split.kind='fields' needs 1 or 2 fields, e.g. GEX and IntrVol")
    names = [f.field for f in split.fields]
    if len(set(names)) != len(names):
        raise HTTPException(status_code=400, detail="split the regime by two DIFFERENT fields")
    return {"kind": "fields", "fields": [{"field": f.field, "n": f.n} for f in split.fields],
            "smooth": split.smooth, "window_days": split.window_days}


# =======================================================================================
# Members: verified candidates and their positions
# =======================================================================================
def _eligible(obj: dict, limit: int = 60) -> list[dict]:
    """Verified candidates, best first, with what the picker shows."""
    from .objectives import _higher, _lock as lock, db

    order = "DESC" if _higher(obj) else "ASC"
    with lock:
        rows = db().execute(
            "SELECT id, seq, model, mode, rationale, score, is_score, eval_seconds, returns FROM candidates "
            "WHERE objective_id=? AND status='ok' AND score IS NOT NULL AND lookahead='pass' AND audit != 'fail' "
            f"AND mode != 'ensemble' ORDER BY score {order}, seq ASC LIMIT ?", (obj["id"], limit)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["returns"] = json.loads(d["returns"] or "[]")
        d["rationale"] = (d.get("rationale") or "")[:240]
        out.append(d)
    return out


def default_members(obj: dict, eligible: list[dict], limit: int = DEFAULT_MEMBERS) -> list[int]:
    """The best-ranked verified candidates that are not near-copies of one another.

    The top of a leaderboard is usually one idea in five variants; routing between clones
    teaches nothing. Greedy by rank, a candidate is skipped when its in-sample daily returns
    correlate above 0.9 with one already picked, or when the members together would run for more
    than half the eval timeout (a router runs every member it routes to inside its own script)."""
    split = obj.get("split_date")
    # A router runs its members inside its own script, under the objective's eval timeout.
    budget = 0.5 * float(obj.get("eval_timeout_s") or 300)
    spent = 0.0
    picked: list[dict] = []
    series: dict[int, dict[str, float]] = {}
    for c in eligible:
        secs = float(c.get("eval_seconds") or 0)
        if secs > MAX_REPLAY_SECONDS or spent + secs > budget:
            continue
        r = {d: v for d, v in c["returns"] if not split or d < split}
        if any((_corr(r, series[p["seq"]]) or 0) > 0.9 for p in picked):
            continue
        picked.append(c)
        spent += secs
        series[c["seq"]] = r
        if len(picked) >= limit:
            break
    return [c["seq"] for c in picked]


def _corr(a: dict[str, float], b: dict[str, float]) -> float | None:
    days = sorted(set(a) | set(b))
    if len(days) < 20:
        return None
    x = np.array([a.get(d, 0.0) for d in days])
    y = np.array([b.get(d, 0.0) for d in days])
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _member_rows(obj: dict, seqs: list[int]) -> list[dict]:
    from .objectives import _lock as lock, db

    out = []
    for s in dict.fromkeys(seqs):
        with lock:
            r = db().execute(
                "SELECT id, seq, model, rationale, score, is_score, eval_seconds, code FROM candidates "
                "WHERE objective_id=? AND seq=? AND status='ok' AND lookahead='pass' AND audit != 'fail' "
                "AND mode != 'ensemble'", (obj["id"], int(s))).fetchone()
        if r is None:
            raise HTTPException(status_code=400, detail=f"#{s} is not a verified candidate of this objective "
                                                        "(scored, look-ahead passed, not disqualified, not an ensemble)")
        out.append(dict(r))
    return out


def _dataset_item(project: dict, obj: dict) -> dict:
    item = next((i for i in datasource.catalog(project["data_dir"]) if obj["dataset"] in (i["view"], i["path"])), None)
    if item is None:
        raise HTTPException(status_code=400, detail=f"dataset {obj['dataset']!r} is gone from the data folder")
    return item


def _data_stamp(project: dict, obj: dict) -> str:
    """Changes when the dataset's files do, so cached positions and labels follow new data."""
    from .objectives import _abs

    item = _dataset_item(project, obj)
    files = glob.glob(_abs(project["data_dir"], item).replace("''", "'"))
    return str(int(max((Path(f).stat().st_mtime for f in files), default=0)))


def _cache_dir(obj: dict) -> Path:
    d = CACHE_ROOT / obj["id"]
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _replay(obj: dict, project: dict, member: dict, dest: Path) -> str | None:
    """Run a member's own script on the full data and keep the positions it reports."""
    from .objectives import _EVAL_SLOTS, _failure_note, _positions_file, _run_forecasting

    catalog = await asyncio.to_thread(datasource.catalog, project["data_dir"])
    async with _EVAL_SLOTS:
        rep = await _run_forecasting(member["code"], project["data_dir"], catalog, None, obj["eval_timeout_s"], obj,
                                     None, requested_by=f"regime lab replay of #{member['seq']}")
    if not rep["ok"]:
        return _failure_note(rep["stderr"])[:400]
    p = _positions_file(rep)
    if p is None:
        return "reports returns, not positions -- a router needs positions"
    shutil.copyfile(p, dest)
    return None


LABEL_SCRIPT = r"""
import importlib, json
import numpy as np, pandas as pd
import ft

CFG = json.loads(__CFG__)
tc = CFG["time_column"]
if CFG["kind"] == "fields":
    df = ft.load(CFG["dataset"], columns=[tc] + [f["field"] for f in CFG["fields"]])
else:
    df = ft.load(CFG["dataset"])
df = df.sort_values(tc).reset_index(drop=True)
if CFG["kind"] == "fields":
    labels = ft.regime_grid(df, {f["field"]: f["n"] for f in CFG["fields"]}, time=tc,
                            smooth=CFG["smooth"], window_days=CFG["window_days"]).to_numpy()
else:
    labels = np.asarray(importlib.import_module("lib." + CFG["module"]).detect(df))
    if len(labels) != len(df):
        raise ValueError(f"{CFG['module']}.detect returned {len(labels)} labels for {len(df)} rows")
out = pd.DataFrame({"t": ft._naive_times(df[tc]), "label": pd.Series(labels).astype(str).to_numpy()})
out.to_parquet("/work/.ft/regime_labels.parquet", index=False)
print("regimes:", out["label"].value_counts(normalize=True).round(3).head(16).to_dict())
"""


async def _labels(obj: dict, project: dict, spec: dict, stamp: str) -> Path:
    from .objectives import _run

    version = ""
    if spec["kind"] == "module":
        from .library import list_modules

        mod = next((m for m in list_modules(obj["project_id"], include_retired=False)
                    if m["name"] == spec["module"] and m["kind"] == "regime"), None)
        if mod is None:
            raise HTTPException(status_code=400, detail=f"{spec['module']!r} is not an active regime module")
        version = f"v{mod['version']}"
    key = hashlib.sha1(json.dumps([spec, version, stamp, obj["dataset"]], sort_keys=True).encode()).hexdigest()[:16]
    dest = _cache_dir(obj) / f"labels_{key}.parquet"
    if dest.is_file():
        return dest
    cfg = {"dataset": obj["dataset"], "time_column": obj["time_column"], **spec}
    code = LABEL_SCRIPT.replace("__CFG__", repr(json.dumps(cfg)), 1)
    catalog = await asyncio.to_thread(datasource.catalog, project["data_dir"])
    rep = await _run(code, project["data_dir"], catalog, None, 400, obj, None)
    src = Path(rep["run_dir"]) / ".ft" / "regime_labels.parquet"
    if not rep["ok"] or not src.is_file():
        raise RuntimeError("labelling regimes failed: " + (rep.get("stderr") or "")[-1200:])
    shutil.copyfile(src, dest)
    return dest


# =======================================================================================
# Measurement (pure: DuckDB over parquet files; unit-tested with synthetic data)
# =======================================================================================
def _r(v: float | None, nd: int = 3) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v), nd)


def _sharpe(x: np.ndarray) -> float | None:
    if len(x) < 5:
        return None
    sd = float(np.std(x, ddof=1))
    return float(np.mean(x) / sd * math.sqrt(252.0)) if sd > 0 else None


def _seg(x: np.ndarray, present: np.ndarray, mask: np.ndarray) -> dict:
    sel = present & mask
    v = x[sel]
    return {"days": int(sel.sum()), "sharpe": _r(_sharpe(v), 2), "pnl": _r(float(v.sum()), 6),
            "hit": _r(float((v > 0).mean()), 3) if len(v) else None}


def _segments(x: np.ndarray, present: np.ndarray, seg: dict[str, np.ndarray]) -> dict:
    return {k: _seg(x, present, m) for k, m in seg.items()}


def _label_order(labels: list[str], spec: dict) -> tuple[list[str], list[dict]]:
    """Regimes in grid order (each field's buckets low -> high) for field splits, by name
    otherwise; the warm-up label last. Also the axes of the grid."""
    def names(n: int) -> list[str]:
        return {2: ["low", "high"], 3: ["low", "mid", "high"]}.get(n) or [f"q{i + 1}" for i in range(n)]

    if spec.get("kind") != "fields":
        return sorted(labels, key=lambda s: (s in SKIP_LABELS, s)), []
    axes = [{"field": f["field"], "buckets": names(int(f["n"]))} for f in spec["fields"]]

    def key(label: str):
        if label in SKIP_LABELS:
            return (1, [])
        parts = dict(p.split(":", 1) for p in label.split("|") if ":" in p)
        return (0, [a["buckets"].index(parts.get(a["field"], "")) if parts.get(a["field"]) in a["buckets"] else 99
                    for a in axes])

    return sorted(labels, key=key), axes


def _pos_sql(path: Path, lev: float) -> str:
    return (f"SELECT CAST(t AS TIMESTAMP) AS t, arg_max(greatest(-{lev}, least({lev}, coalesce(CAST(pos AS DOUBLE), 0))), t) AS pos "
            f"FROM read_parquet('{path.as_posix()}') GROUP BY 1")


def analyze(price_sql: str, labels_path: Path, members: list[tuple[int, Path]], *, cost_bps: float,
            max_leverage: float, split: str | None, mid: str | None, spec: dict,
            routes: dict[str, int | None] | None = None) -> dict:
    """Every member inside every regime, then the router, from the bars up.

    `price_sql` selects (t TIMESTAMP, p DOUBLE) -- the dataset's time and price columns."""
    cost = float(cost_bps or 0) / 10_000.0
    lev = float(max_leverage or 1.0)
    seqs = [s for s, _ in members]
    con = duckdb.connect(":memory:")
    try:
        con.execute("SET TimeZone = 'UTC'")
        con.execute(f"CREATE TEMP TABLE px AS SELECT t, avg(p) AS p FROM ({price_sql}) "
                    "WHERE t IS NOT NULL AND p IS NOT NULL AND p > 0 GROUP BY t")
        con.execute(f"CREATE TEMP TABLE lab AS SELECT CAST(t AS TIMESTAMP) AS t, arg_max(CAST(label AS VARCHAR), t) AS label "
                    f"FROM read_parquet('{labels_path.as_posix()}') GROUP BY 1")
        joins, cols = [], []
        for i, (_, path) in enumerate(members):
            con.execute(f"CREATE TEMP TABLE p{i} AS {_pos_sql(path, lev)}")
            joins.append(f"ASOF LEFT JOIN p{i} ON px.t >= p{i}.t")
            cols.append(f"coalesce(p{i}.pos, 0) AS m{i}")
        con.execute(
            "CREATE TEMP TABLE g AS SELECT px.t, px.p, coalesce(lab.label, 'warmup') AS label"
            + "".join(f", {c}" for c in cols)
            + " FROM px ASOF LEFT JOIN lab ON px.t >= lab.t " + " ".join(joins))
        # Bar t's return belongs to the regime in force when the position earning it was set (t-1).
        lagged = "".join(f", lag(m{i}) OVER w AS a{i}, lag(m{i}, 2) OVER w AS b{i}" for i in range(len(members)))
        con.execute(
            "CREATE TEMP TABLE b AS SELECT t, strftime(CAST(t AS DATE), '%Y-%m-%d') AS d, lag(label) OVER w AS lab, "
            f"p / lag(p) OVER w - 1 AS ret{lagged} FROM g WINDOW w AS (ORDER BY t)")
        nets = "".join(f", sum(coalesce(a{i} * ret, 0) - {cost} * abs(coalesce(a{i}, 0) - coalesce(b{i}, 0)))"
                       for i in range(len(members)))
        rows = con.execute(f"SELECT d, lab, count(*){nets} FROM b WHERE lab IS NOT NULL AND ret IS NOT NULL "
                           "GROUP BY 1, 2 ORDER BY 1, 2").fetchall()
        switches = dict(con.execute(
            "SELECT d, count(*) FILTER (WHERE label IS DISTINCT FROM prev AND prev IS NOT NULL) FROM "
            "(SELECT strftime(CAST(t AS DATE), '%Y-%m-%d') AS d, label, lag(label) OVER (ORDER BY t) AS prev FROM g) "
            "GROUP BY 1").fetchall())
        entries = dict(con.execute(
            "SELECT label, count(*) FILTER (WHERE label IS DISTINCT FROM prev) FROM "
            "(SELECT label, lag(label) OVER (ORDER BY t) AS prev FROM g) GROUP BY 1").fetchall())
        bar_s = con.execute("SELECT median(epoch(t) - epoch(pt)) FROM (SELECT t, lag(t) OVER (ORDER BY t) AS pt FROM g)").fetchone()[0]

        dates = sorted({r[0] for r in rows})
        labels_seen = sorted({r[1] for r in rows})
        labels, axes = _label_order(labels_seen, spec)
        di = {d: i for i, d in enumerate(dates)}
        nd, nl = len(dates), len(labels)
        li = {lab: i for i, lab in enumerate(labels)}
        bars = np.zeros((nl, nd))
        cell = np.zeros((nl, len(members), nd))
        for r in rows:
            i, j = li[r[1]], di[r[0]]
            bars[i, j] = r[2]
            cell[i, :, j] = [v or 0.0 for v in r[3:]]
        present = bars > 0
        d_arr = np.array(dates)
        is_mask = d_arr < split if split else np.ones(nd, dtype=bool)
        is_days = d_arr[is_mask]
        if not mid or not (len(is_days) and is_days[0] < mid <= is_days[-1]):
            mid = str(is_days[len(is_days) // 2]) if len(is_days) else None
        seg = {"is": is_mask, "a": is_mask & (d_arr < mid) if mid else is_mask,
               "b": is_mask & (d_arr >= mid) if mid else np.zeros(nd, dtype=bool), "ho": ~is_mask}

        total_is = max(1.0, float(bars[:, is_mask].sum()))
        total_ho = max(1.0, float(bars[:, ~is_mask].sum()))
        regimes = []
        for lab in labels:
            i = li[lab]
            n_bars = float(bars[i].sum())
            regimes.append({
                "label": lab,
                "parts": [p.split(":", 1) for p in lab.split("|") if ":" in p] if spec.get("kind") == "fields" else [],
                "share": {"is": _r(float(bars[i, is_mask].sum()) / total_is, 4),
                          "ho": _r(float(bars[i, ~is_mask].sum()) / total_ho, 4) if (~is_mask).any() else None},
                "days": {"is": int(present[i, is_mask].sum()), "ho": int(present[i, ~is_mask].sum())},
                "dwell_min": _r(n_bars / max(1, entries.get(lab, 1)) * float(bar_s or 0) / 60.0, 1),
            })
        sw = np.array([switches.get(d, 0) for d in dates], dtype=float)
        switches_per_day = _r(float(sw[is_mask].mean()) if is_mask.any() else None, 1)

        cells: dict[str, dict[str, dict]] = {}
        for lab in labels:
            i = li[lab]
            cells[lab] = {str(s): _segments(cell[i, k], present[i], seg) for k, s in enumerate(seqs)}
        member_daily = cell.sum(axis=0)
        any_day = present.any(axis=0)
        member_stats = {str(s): _segments(member_daily[k], any_day, seg) for k, s in enumerate(seqs)}

        best = max(seqs, key=lambda s: member_stats[str(s)]["is"]["sharpe"] or -1e9) if seqs else None
        suggested = suggest_routes(labels, regimes, cells, seqs, best)
        use = {lab: suggested.get(lab) for lab in labels} if routes is None else {
            lab: (int(routes[lab]) if routes.get(lab) is not None and int(routes[lab]) in seqs else None) for lab in labels}

        # The router, exactly: its own position series and its own switching costs.
        case = " ".join(f"WHEN '{lab.replace(chr(39), chr(39) * 2)}' THEN m{seqs.index(s)}"
                        for lab, s in use.items() if s is not None)
        routed_sql = f"CASE label {case} ELSE 0 END" if case else "0"
        rrows = con.execute(
            f"WITH r AS (SELECT t, {routed_sql} AS pos FROM g), "
            "l AS (SELECT t, lag(pos) OVER w AS a, lag(pos, 2) OVER w AS b FROM r WINDOW w AS (ORDER BY t)) "
            f"SELECT b.d, b.lab, sum(coalesce(l.a * b.ret, 0) - {cost} * abs(coalesce(l.a, 0) - coalesce(l.b, 0))) "
            "FROM l JOIN b USING (t) WHERE b.lab IS NOT NULL AND b.ret IS NOT NULL GROUP BY 1, 2").fetchall()
    finally:
        con.close()

    router_by_label = np.zeros((nl, nd))
    for d, lab, v in rrows:
        if lab in li and d in di:
            router_by_label[li[lab], di[d]] = v or 0.0
    router_daily = router_by_label.sum(axis=0)
    contrib: dict[str, np.ndarray] = {str(s): np.zeros(nd) for s in seqs}
    contrib[FLAT] = np.zeros(nd)
    for lab in labels:
        s = use.get(lab)
        contrib[str(s) if s is not None else FLAT] += router_by_label[li[lab]]
    router = _segments(router_daily, any_day, seg)

    def arr(x: np.ndarray) -> list[float]:
        return [float(f"{v:.6g}") for v in x]

    dominant = [int(np.argmax(bars[:, j])) if bars[:, j].any() else -1 for j in range(nd)]
    return {
        "version": 1, "spec": spec, "describe": describe(spec), "split_date": split, "mid_date": mid,
        "cost_bps": cost_bps, "dates": dates, "regimes": regimes, "axes": axes,
        "switches_per_day": switches_per_day, "bar_seconds": _r(float(bar_s or 0), 1),
        "cells": cells, "member_stats": member_stats,
        "suggested": {k: v for k, v in suggested.items()}, "routes": use,
        "routes_source": "suggested" if routes is None else "custom",
        "router": router,
        "best_single": {"seq": best, **({k: member_stats[str(best)][k] for k in ("is", "a", "b", "ho")} if best else {})},
        "daily": {
            "regime": dominant,
            "router": arr(router_daily),
            "contrib": {k: arr(v) for k, v in contrib.items()},
            "members": {str(s): arr(member_daily[k]) for k, s in enumerate(seqs)},
            "cells": {lab: {str(s): arr(cell[li[lab], k]) for k, s in enumerate(seqs)} for lab in labels},
            "bars": {lab: [int(v) for v in bars[li[lab]]] for lab in labels},
        },
    }


def suggest_routes(labels: list[str], regimes: list[dict], cells: dict, seqs: list[int],
                   baseline: int | None) -> dict[str, int | None]:
    """Routes that start from the best single member everywhere and change only on evidence.

    Picking the best of k members in each of r regimes is k x r free choices -- in-sample it
    always looks better, and on the first real data (6 members, 9 regimes) the greedy router
    went from in-sample Sharpe 3.3 to a NEGATIVE holdout while the best single member held
    3.2. So the default shrinks to the baseline: a regime switches to another member only
    when that member beats the baseline by SWITCH_MARGIN in BOTH in-sample halves, and goes
    flat only when the baseline loses in both halves there. Rare regimes keep the baseline."""
    reg = {r["label"]: r for r in regimes}
    out: dict[str, int | None] = {}
    for lab in labels:
        r = reg[lab]
        if lab in SKIP_LABELS:
            out[lab] = None
            continue
        out[lab] = baseline
        if baseline is None or (r["share"]["is"] or 0) < MIN_SHARE or r["days"]["is"] < MIN_DAYS:
            continue
        base = cells[lab][str(baseline)]
        ba, bb = base["a"]["sharpe"], base["b"]["sharpe"]
        if ba is not None and bb is not None and ba < 0 and bb < 0:
            out[lab] = None
        bar_a, bar_b = max(ba or 0.0, 0.0) + SWITCH_MARGIN, max(bb or 0.0, 0.0) + SWITCH_MARGIN
        best, best_v = None, -math.inf
        for s in seqs:
            if s == baseline:
                continue
            a, b = cells[lab][str(s)]["a"], cells[lab][str(s)]["b"]
            if a["days"] < MIN_HALF_DAYS or b["days"] < MIN_HALF_DAYS or a["sharpe"] is None or b["sharpe"] is None:
                continue
            if a["sharpe"] > bar_a and b["sharpe"] > bar_b and min(a["sharpe"], b["sharpe"]) > best_v:
                best, best_v = s, min(a["sharpe"], b["sharpe"])
        if best is not None:
            out[lab] = best
    return out


# =======================================================================================
# The candidate a router becomes
# =======================================================================================
def router_code(obj: dict, spec: dict, routes: dict[str, int | None], run_id: int | None) -> str:
    tc, ds = obj["time_column"], obj["dataset"]
    used = sorted({int(s) for s in routes.values() if s is not None})
    if not used:
        raise HTTPException(status_code=400, detail="every regime is flat -- route at least one regime to a candidate")
    head = [f'"""Regime router{f" from Regime Lab run {run_id}" if run_id else ""}.', "",
            f"Regime: {describe(spec)}.",
            "Each routed regime trades the verified candidate that worked there in BOTH halves of the in-sample",
            "period; every other regime (and the warm-up) stays flat.", '"""', "import ft"]
    if spec["kind"] == "fields":
        cols = [tc] + [f["field"] for f in spec["fields"]]
        grid = "{" + ", ".join(f"{f['field']!r}: {int(f['n'])}" for f in spec["fields"]) + "}"
        body = ["", f"df = ft.load({ds!r}, columns={cols!r}).sort_values({tc!r}).reset_index(drop=True)",
                f"regime = ft.regime_grid(df, {grid}, time={tc!r}, smooth={int(spec['smooth'])}, "
                f"window_days={spec['window_days']!r})"]
    else:
        mod = spec["module"]
        head += ["import numpy as np", "import pandas as pd", f"from lib import {mod}"]
        body = ["", f"df = ft.load({ds!r}).sort_values({tc!r}).reset_index(drop=True)",
                f"regime = pd.Series(np.asarray({mod}.detect(df)).astype(str), index=pd.to_datetime(df[{tc!r}]), "
                f"name={mod!r})"]
    body += ["", "# Each member's own script runs here on the same data (a look-ahead test truncates it for them too)."]
    body += [f"m{s} = ft.candidate_positions({s})" for s in used]
    body += ["", "positions = ft.route(regime, {"]
    body += [f"    {lab!r}: m{int(s)}," for lab, s in routes.items() if s is not None]
    body += ["})", "ft.report_positions(positions)", ""]
    return "\n".join(head + body)


# =======================================================================================
# Runs: a background job per objective
# =======================================================================================
def _store(obj: dict, author: str, spec: dict, members: list[dict], result: dict) -> int:
    summary = _summary(result, members)
    conn = _db()
    with _lock():
        cur = conn.execute(
            "INSERT INTO regime_labs (project_id, objective_id, ts, author, spec, members, routes, summary, result) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (obj["project_id"], obj["id"], time.time(), author, json.dumps(spec),
             json.dumps([m["seq"] for m in members]), json.dumps(result["routes"]), json.dumps(summary),
             json.dumps(result, separators=(",", ":"))))
        conn.commit()
        return int(cur.lastrowid)


def _summary(result: dict, members: list[dict]) -> dict:
    return {"describe": result["describe"], "members": [m["seq"] for m in members],
            "regimes": sum(1 for r in result["regimes"] if r["label"] not in SKIP_LABELS),
            "routed": sum(1 for v in result["routes"].values() if v is not None),
            "router": {k: result["router"][k].get("sharpe") for k in ("is", "a", "b", "ho")},
            "best_single": {"seq": result["best_single"].get("seq"),
                            **{k: (result["best_single"].get(k) or {}).get("sharpe") for k in ("is", "ho")}},
            "routes_source": result["routes_source"]}


def _row(r, full: bool = False) -> dict:
    d = {k: r[k] for k in ("id", "objective_id", "ts", "author", "candidate_seq")}
    d["spec"] = json.loads(r["spec"])
    d["members"] = json.loads(r["members"])
    d["routes"] = json.loads(r["routes"])
    d["summary"] = json.loads(r["summary"])
    if full:
        d["result"] = json.loads(r["result"])
    return d


def get_run(rid: int) -> dict:
    conn = _db()
    with _lock():
        r = conn.execute("SELECT * FROM regime_labs WHERE id=?", (rid,)).fetchone()
    if r is None:
        raise HTTPException(status_code=404, detail=f"no regime lab run {rid}")
    return _row(r, full=True)


def list_runs(oid: str, limit: int = 30) -> list[dict]:
    conn = _db()
    with _lock():
        rows = conn.execute("SELECT id, objective_id, ts, author, candidate_seq, spec, members, routes, summary "
                            "FROM regime_labs WHERE objective_id=? ORDER BY ts DESC LIMIT ?", (oid, limit)).fetchall()
    return [_row(r) for r in rows]


async def _job(job: dict, obj: dict, project: dict, spec: dict, members: list[dict],
               routes: dict[str, int | None] | None, author: str) -> None:
    from .objectives import _abs, _reader

    try:
        stamp = await asyncio.to_thread(_data_stamp, project, obj)
        have: list[dict] = []
        paths: list[tuple[int, Path]] = []
        for i, m in enumerate(members):
            dest = _cache_dir(obj) / f"pos_{m['id']}_{stamp}.parquet"
            if not dest.is_file():
                job["step"] = f"replaying #{m['seq']} for its positions ({i + 1}/{len(members)})"
                err = await _replay(obj, project, m, dest)
                if err:
                    job["errors"][str(m["seq"])] = err
                    continue
            have.append(m)
            paths.append((m["seq"], dest))
            job["done"] += 1
        if not paths:
            raise RuntimeError("no member produced positions: " + "; ".join(f"#{k}: {v}" for k, v in job["errors"].items()))
        job["step"] = "labelling regimes"
        lp = await _labels(obj, project, spec, stamp)
        job["done"] += 1
        job["step"] = "measuring every member in every regime"
        item = await asyncio.to_thread(_dataset_item, project, obj)
        price_sql = (f'SELECT TRY_CAST("{obj["time_column"]}" AS TIMESTAMP) AS t, '
                     f'TRY_CAST("{obj["metric"]["price_column"]}" AS DOUBLE) AS p '
                     f"FROM {_reader(item)}('{_abs(project['data_dir'], item)}')")
        mid = str(obj["metric"].get("mid_cut") or "")[:10] or None
        result = await asyncio.to_thread(
            analyze, price_sql, lp, paths, cost_bps=obj["metric"].get("cost_bps") or 0,
            max_leverage=obj["metric"].get("max_leverage") or 1.0, split=obj.get("split_date"), mid=mid,
            spec=spec, routes=routes)
        result["members"] = [{k: m.get(k) for k in ("seq", "id", "model", "score", "is_score", "eval_seconds")}
                             | {"rationale": (m.get("rationale") or "")[:240]} for m in have]
        result["errors"] = dict(job["errors"])
        rid = await asyncio.to_thread(_store, obj, author, spec, have, result)
        result["code"] = router_code(obj, spec, result["routes"], rid) if any(result["routes"].values()) else None
        job.update(phase="done", run_id=rid)
    except HTTPException as exc:
        job.update(phase="error", error=str(exc.detail))
    except Exception as exc:  # noqa: BLE001 -- a job must end in a readable state
        logger.exception("regime lab run failed")
        job.update(phase="error", error=f"{type(exc).__name__}: {exc}"[:2000])
    finally:
        job["finished_at"] = time.time()
        job["step"] = None
        ev = _events.get(obj["id"])
        if ev:
            ev.set()


def _setup(oid: str) -> tuple[dict, dict]:
    from .objectives import get_objective

    obj = get_objective(oid)
    project = projects.get(obj["project_id"])
    if project is None:
        raise HTTPException(status_code=404, detail="the objective's project no longer exists")
    if not (obj.get("dataset") and obj.get("time_column") and obj["metric"].get("price_column")):
        raise HTTPException(status_code=400, detail="the Regime Lab needs an objective with a dataset, time column and price column")
    return obj, project


def compact(obj: dict, run: dict, with_code: bool = True) -> dict:
    """What an agent sees: IN-SAMPLE numbers only, the suggested routes and the script."""
    res = run["result"]

    def s(seg: dict | None) -> float | None:
        return (seg or {}).get("sharpe")

    regimes = []
    for r in res["regimes"]:
        lab = r["label"]
        ranked = sorted(((int(k), v) for k, v in res["cells"].get(lab, {}).items()),
                        key=lambda kv: kv[1]["is"]["sharpe"] if kv[1]["is"]["sharpe"] is not None else -1e9, reverse=True)
        regimes.append({"regime": lab, "share": r["share"]["is"], "days": r["days"]["is"],
                        "route": f"#{res['routes'][lab]}" if res["routes"].get(lab) else FLAT,
                        "best": [f"#{k}: IS Sharpe {s(v['is'])} (halves {s(v['a'])} / {s(v['b'])})" for k, v in ranked[:3]]})
    bs = res.get("best_single") or {}
    out = {
        "run": run["id"], "regime": res["describe"], "switches_per_day": res.get("switches_per_day"),
        "members": {f"#{k}": f"IS Sharpe {s(v['is'])} (halves {s(v['a'])} / {s(v['b'])})"
                    for k, v in res["member_stats"].items()},
        "regimes": regimes,
        "router_in_sample": {"sharpe": s(res["router"]["is"]), "halves": [s(res["router"]["a"]), s(res["router"]["b"])],
                             "vs_best_single": f"#{bs.get('seq')}: {s(bs.get('is'))}" if bs.get("seq") else None},
        "errors": res.get("errors") or {},
        "submitted_as": f"#{run['candidate_seq']}" if run.get("candidate_seq") else None,
        "note": ("In-sample only; daily Sharpe of the P&L earned while each regime was in force, net of costs. "
                 "Routes start from the best single member; a regime switches to another member only when it "
                 "beats that one by 0.5 Sharpe in BOTH in-sample halves (goes flat only when the baseline loses in "
                 "both). Switching members costs trades at every regime change, so a router must clearly beat the "
                 "best single member in-sample to be worth submitting; say why it should generalise."),
    }
    if with_code and any(res["routes"].values()):
        out["code"] = router_code(obj, res["spec"], res["routes"], run["id"])
    return out


def brief(obj: dict) -> list[dict]:
    """The latest lab runs for the iteration brief: in-sample, the newest with its script."""
    runs = list_runs(obj["id"], limit=2)
    return [compact(obj, get_run(r["id"]), with_code=(i == 0)) for i, r in enumerate(runs)]


# =======================================================================================
# Routes
# =======================================================================================
@router.get("/objectives/{oid}/regime-lab")
async def overview(oid: str) -> dict:
    from .deciplot import _numeric, dataset_columns
    from .library import list_modules

    obj, project = _setup(oid)
    eligible = await asyncio.to_thread(_eligible, obj)
    cols = await asyncio.to_thread(dataset_columns, project["data_dir"], obj["dataset"])
    fields = [c for c in _numeric(cols) if c not in (obj["time_column"], obj["metric"]["price_column"])]
    return {
        "runs": await asyncio.to_thread(list_runs, oid),
        "job": {k: v for k, v in (_jobs.get(oid) or {}).items()} or None,
        "fields": fields,
        "default_fields": [f for f in DEFAULT_FIELDS if f in fields],
        "modules": [m["name"] for m in list_modules(obj["project_id"], include_retired=False) if m["kind"] == "regime"],
        "eligible": [{k: c.get(k) for k in ("seq", "id", "model", "mode", "score", "is_score", "eval_seconds", "rationale")}
                     for c in eligible],
        "default_members": default_members(obj, eligible),
        "split_date": obj.get("split_date"),
        "eval_timeout_s": obj.get("eval_timeout_s"),
    }


@router.get("/regime-lab/{rid}")
async def run_detail(rid: int) -> dict:
    run = await asyncio.to_thread(get_run, rid)
    from .objectives import get_objective

    obj = get_objective(run["objective_id"])
    res = run["result"]
    if any(res["routes"].values()):
        res["code"] = router_code(obj, res["spec"], res["routes"], rid)
    return run


@router.post("/objectives/{oid}/regime-lab")
async def start(oid: str, req: LabReq) -> dict:
    """Measure verified candidates inside regimes and build the router. Runs in the background;
    `wait` blocks until it finishes (agents), `compact` returns the in-sample view."""
    obj, project = _setup(oid)
    running = _jobs.get(oid)
    if running and running["phase"] == "running":
        if not req.wait:
            return {"job": running, "note": "a Regime Lab run is already in progress for this objective"}
    else:
        split = req.split or Split(fields=[FieldSplit(field=f) for f in DEFAULT_FIELDS])
        spec = _norm_spec(split)
        if spec["kind"] == "fields":
            from .deciplot import _numeric, dataset_columns

            cols = set(_numeric(await asyncio.to_thread(dataset_columns, project["data_dir"], obj["dataset"])))
            bad = [f["field"] for f in spec["fields"] if f["field"] not in cols]
            if bad:
                raise HTTPException(status_code=400, detail=f"not numeric columns of {obj['dataset']}: {', '.join(bad)}")
        seqs = req.members or default_members(obj, await asyncio.to_thread(_eligible, obj))
        if not seqs:
            raise HTTPException(status_code=400, detail="no verified candidates to measure yet")
        members = await asyncio.to_thread(_member_rows, obj, seqs[:MAX_MEMBERS])
        running = {"objective_id": oid, "phase": "running", "step": "starting", "done": 0, "total": len(members) + 1,
                   "errors": {}, "started_at": time.time(), "finished_at": None, "run_id": None, "error": None,
                   "author": req.author, "describe": describe(spec)}
        _jobs[oid] = running
        _events[oid] = asyncio.Event()
        asyncio.create_task(_job(running, obj, project, spec, members, req.routes, req.author))
    if not req.wait:
        return {"job": running}
    await _events[oid].wait()
    job = _jobs[oid]
    if job["phase"] != "done":
        return {"ok": False, "error": job.get("error") or "the run did not finish", "member_errors": job.get("errors")}
    run = await asyncio.to_thread(get_run, job["run_id"])
    return {"ok": True, **compact(obj, run)} if req.compact else {"ok": True, "run": run}


@router.post("/regime-lab/{rid}/submit")
async def submit_router(rid: int, req: SubmitReq) -> dict:
    """The router (the run's routes, or edited ones) as a candidate, scored like any other."""
    from .objectives import Submit, _lock as lock, db, evaluate, get_objective

    run = await asyncio.to_thread(get_run, rid)
    obj = get_objective(run["objective_id"])
    if obj["status"] != "running":
        raise HTTPException(status_code=409, detail=f"objective is {obj['status']}")
    res = run["result"]
    member_seqs = {int(k) for k in res["member_stats"]}
    routes = res["routes"] if req.routes is None else {
        lab: (int(v) if v is not None and int(v) in member_seqs else None) for lab, v in req.routes.items()}
    code = router_code(obj, res["spec"], routes, rid)
    routed = [f"{lab} -> #{s}" for lab, s in routes.items() if s is not None]
    rationale = req.rationale or (
        f"Regime router (Regime Lab run {rid}): {res['describe']}. Routes: {'; '.join(routed)}; other regimes flat. "
        f"Each route is a candidate that worked in that regime in both in-sample halves.")
    best = res.get("best_single", {}).get("seq")
    with lock:
        parent = db().execute("SELECT id FROM candidates WHERE objective_id=? AND seq=?", (obj["id"], best)).fetchone()
    body = Submit(code=code, rationale=rationale[:8000], parent_id=parent["id"] if parent else None,
                  model=req.model, mode="regime")
    started = time.time()
    task = asyncio.create_task(evaluate(obj, body))
    task.add_done_callback(lambda t: t.exception() and logger.error("router candidate failed: %s", t.exception()))
    seq = None
    for _ in range(40):  # evaluate() numbers the candidate before its first await
        await asyncio.sleep(0.05)
        with lock:
            row = db().execute("SELECT seq FROM candidates WHERE objective_id=? AND code=? AND created_at >= ? "
                               "ORDER BY seq DESC LIMIT 1", (obj["id"], code, started - 1)).fetchone()
        if row:
            seq = row["seq"]
            break
    if seq is not None:
        conn = _db()
        with _lock():
            conn.execute("UPDATE regime_labs SET candidate_seq=? WHERE id=?", (seq, rid))
            conn.commit()
    return {"ok": True, "seq": seq, "code": code}
