"""Decile studies ("deci-plots"), run in the sandbox on in-sample data and kept for the team.

A field scan ranks every field by one number (its rank correlation with the forward return).
A decile study looks at ONE signal properly: the average forward return in each of its ten
deciles, on 10 s / 20 s / 30 s / 1 min / 5 min bars, at several horizons, and whether the
shape holds in each sub-period -- enough to tell a real, monotone, stable relationship from a
lucky extreme bucket. The statistics live in deci_core.py (pure pandas, unit-tested); this
module validates the request, runs that core in the sandbox, and stores the result.

Nothing here may see the holdout, and the study itself must not look ahead:

* The run mounts the objective's IN-SAMPLE mirror (every row at or after the split removed
  from the files themselves) and the forecast features truncated at the same cut; the core
  drops rows at/after the cut again before computing anything.
* Deciles are assigned against ROLLING edges from past sessions only (never a full-sample
  qcut), and forward returns never cross the session close or the cut. See deci_core.

**Every study is stored and cached** by what determines its numbers (dataset, signal, the
feature's build, timeframes, horizons, window, cut, core version): asking again returns the
stored study at once, so the team's understanding of each signal accumulates instead of
being recomputed -- and the brief each agent reads lists what is already known.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import deci_core, projects

logger = logging.getLogger("freetoken.deciplot")

router = APIRouter(tags=["deci-plots"])

CORE_SRC = Path(__file__).resolve().parent / "deci_core.py"
TIMEOUT_S = 500
BATCH_CHUNK = 8                     # signals per sandbox run in a batch (the data loads once)
# Columns a batch leaves out by default: the bar itself, not a signal about it.
BATCH_SKIP = {"open", "high", "low", "close", "schemaver", "symbol"}

# One study at a time, interactive and batch alike: each is a container with 2 CPUs.
_SLOT = asyncio.Semaphore(1)
_batches: dict[str, dict] = {}

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
            CREATE TABLE IF NOT EXISTS deci_plots (
                id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL, objective_id TEXT,
                key TEXT NOT NULL, ts REAL NOT NULL, author TEXT, signal TEXT NOT NULL,
                params TEXT NOT NULL, result TEXT NOT NULL, summary TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS deci_key ON deci_plots(project_id, key);
            """
        )
        conn.commit()
    _ready = True
    return conn


def _lock():
    from .objectives import _lock as lock

    return lock


# =======================================================================================
# The sandbox harness
# =======================================================================================
# The core and the config are INLINED into the script (see harness_code), not shipped as extra
# files: sandbox.execute copies at most 32 files per run, and a project library of ~30 modules
# already fills them -- extra files past the cap are dropped without a word.
DECI_HARNESS = r"""
import json, sys, types
sys.path.insert(0, "/work/.ft")
import numpy as np, polars as pl
import ft

D = types.ModuleType("deci_core")
exec(compile(__CORE__, "deci_core.py", "exec"), D.__dict__)
CFG = json.loads(__CFG__)
tc, pc, cut = CFG["time_column"], CFG["price_column"], CFG.get("cut")
need = {tc, pc}
for s in CFG["signals"]:
    if s["kind"] == "dataset":
        need |= set(s["columns"])


def naive_times(frame, col):
    # Parquet may carry a tz; compare in UTC wall time, as the harness does everywhere.
    dt = frame.schema[col]
    if dt == pl.String:
        frame = frame.with_columns(pl.col(col).str.to_datetime())
    elif not isinstance(dt, pl.Datetime):
        frame = frame.with_columns(pl.col(col).cast(pl.Datetime("us")))
    if getattr(frame.schema[col], "time_zone", None):
        frame = frame.with_columns(pl.col(col).dt.convert_time_zone("UTC").dt.replace_time_zone(None))
    return frame.with_columns(pl.col(col).cast(pl.Datetime("us")))


df = naive_times(ft.load_pl(CFG["dataset"], columns=sorted(need)), tc).sort(tc, maintain_order=True)
if cut:
    # The mounted data already ends before the cut; this is the second lock on the same door.
    from datetime import datetime
    df = df.filter(pl.col(tc) < datetime.fromisoformat(str(cut)))
price = D._floats(df[pc])
out, errors = {}, {}
for s in CFG["signals"]:
    try:
        if s["kind"] == "feature":
            f = naive_times(ft.load_pl(s["view"]), "t").sort("t", maintain_order=True)
            f = f.with_columns(pl.Series("_sig", D.eval_expression(f, s["expr"])))
            # A forecast stamped t was made from data up to t: attach it to bars at or after t.
            m = df.select(tc).join_asof(f.select("t", "_sig"), left_on=tc, right_on="t", strategy="backward")
            x = D._floats(m["_sig"])
        else:
            x = D.eval_expression(df, s["expr"])
        out[s["signal"]] = D.study(df[tc], x, price, cut=cut, timeframes=CFG["timeframes"],
                                   horizons=CFG["horizons"], window_days=CFG["window_days"],
                                   window_bars=CFG.get("window_bars"))
    except Exception as exc:
        errors[s["signal"]] = f"{type(exc).__name__}: {exc}"
ft._merge({"deci": out, "deci_errors": errors})
print("deci-plots:", {k: v["summary"]["verdict"] for k, v in out.items()}, "errors:", errors)
"""


def harness_code(cfg: dict) -> str:
    """The script a study runs: the harness with deci_core's source and the config inlined."""
    return (DECI_HARNESS.replace("__CORE__", repr(CORE_SRC.read_text(encoding="utf-8")), 1)
            .replace("__CFG__", repr(json.dumps(cfg, default=str)), 1))


# =======================================================================================
# Signals: validate on the host before a container is spent on a typo
# =======================================================================================
_columns_cache: dict[str, list[tuple[str, str]]] = {}


def dataset_columns(data_dir: str, dataset: str) -> list[tuple[str, str]]:
    """(name, type) of every column of a dataset (DESCRIBE only, no scan)."""
    import duckdb

    from . import datasource
    from .objectives import _abs, _reader

    key = f"{data_dir}|{dataset}"
    if key in _columns_cache:
        return _columns_cache[key]
    item = next((i for i in datasource.catalog(data_dir) if dataset in (i["view"], i["path"])), None)
    if item is None:
        raise HTTPException(status_code=400, detail=f"no dataset {dataset!r}")
    con = duckdb.connect(":memory:")
    try:
        cols = [(c[0], str(c[1])) for c in con.execute(
            f"DESCRIBE SELECT * FROM {_reader(item)}('{_abs(data_dir, item)}')").fetchall()]
    finally:
        con.close()
    _columns_cache[key] = cols
    return cols


def _numeric(cols: list[tuple[str, str]]) -> list[str]:
    return [n for n, t in cols if any(k in t.upper() for k in ("DOUBLE", "FLOAT", "DECIMAL", "REAL", "INT", "NUMERIC"))]


def feature_columns(oid: str, view: str) -> tuple[dict, list[str]]:
    import pyarrow.parquet as pq

    from .objectives import _features_root, list_features

    feat = next((f for f in list_features(oid) if f.get("view") == view), None)
    if feat is None:
        raise HTTPException(status_code=400, detail=f"no forecast feature {view!r} in this objective")
    try:
        names = pq.read_schema(_features_root(oid) / feat["file"]).names
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"{view}: {exc}") from None
    return feat, [c for c in names if c != "t"]


def parse_signal(signal: str) -> tuple[str, str | None, str]:
    """("dataset", None, expr) or ("feature", "fc_view", expr).

    A dataset signal is a column or an expression over columns (``GEX / Pinning_TotalAbsGex``);
    a forecast feature's column is ``fc_<name>:<column or expression over its columns>``
    (``fc_close_h30:fc_change / last``); a bare ``fc_<name>`` means its fc_change."""
    s = " ".join((signal or "").split())
    if not s:
        raise HTTPException(status_code=400, detail="give the signal: a column, an expression, or fc_<feature>:<column>")
    if s.startswith("fc_"):
        view, _, expr = s.partition(":")
        return "feature", view.strip(), (expr.strip() or "fc_change")
    return "dataset", None, s


def resolve_signal(obj: dict, data_dir: str, signal: str) -> dict:
    """The validated signal spec the harness runs, or a 400 that says what is wrong."""
    from .objectives import series_expression

    kind, view, expr = parse_signal(signal)
    if kind == "feature":
        feat, cols = feature_columns(obj["id"], view)
        try:
            used = deci_core.expression_columns(expr, cols)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{view}: {exc}; its columns: {', '.join(cols)}") from None
        return {"signal": f"{view}:{expr}", "kind": "feature", "view": view, "expr": expr, "columns": used,
                "feature_built": feat.get("created_at")}
    cols = [n for n, _ in dataset_columns(data_dir, obj["dataset"])]
    series_expression(expr, cols)   # the SQL vocabulary the rest of the app uses (raises a 400)
    try:
        used = deci_core.expression_columns(expr, cols)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return {"signal": expr, "kind": "dataset", "view": None, "expr": expr, "columns": used}


# =======================================================================================
# Cache and store
# =======================================================================================
def cache_key(obj: dict, spec: dict, timeframes: list[str], horizons: list[int], window_days: int,
              window_bars: int | None) -> str:
    """What determines a study's numbers -- and nothing else, so a repeat is a hit."""
    doc = {"dataset": obj.get("dataset"), "price": obj["metric"].get("price_column"), "signal": spec["signal"],
           "feature_built": spec.get("feature_built"), "timeframes": list(timeframes),
           "horizons": [int(h) for h in horizons],
           "window": {"bars": int(window_bars)} if window_bars else {"days": int(window_days)},
           "cut": obj.get("split_date"), "core": deci_core.CORE_VERSION}
    return hashlib.sha1(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()[:20]


def cached(project_id: str, key: str) -> dict | None:
    conn = _db()
    with _lock():
        r = conn.execute("SELECT * FROM deci_plots WHERE project_id=? AND key=? ORDER BY ts DESC LIMIT 1",
                         (project_id, key)).fetchone()
    return _row(r, full=True) if r is not None else None


def store(project_id: str, oid: str, key: str, author: str, spec: dict, params: dict, result: dict) -> int:
    conn = _db()
    with _lock():
        cur = conn.execute(
            "INSERT INTO deci_plots (project_id, objective_id, key, ts, author, signal, params, result, summary) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (project_id, oid, key, time.time(), author or "", spec["signal"], json.dumps(params),
             json.dumps(result), json.dumps(result.get("summary") or {})))
        conn.commit()
    return int(cur.lastrowid)


def _row(r, full: bool = False) -> dict:
    params = json.loads(r["params"])
    out = {"id": r["id"], "objective_id": r["objective_id"], "key": r["key"], "ts": r["ts"], "author": r["author"],
           "signal": r["signal"], "kind": params.get("kind"), "timeframes": params.get("timeframes"),
           "horizons": params.get("horizons"), "window": params.get("window"), "cut": params.get("cut"),
           "summary": json.loads(r["summary"])}
    if full:
        out["result"] = json.loads(r["result"])
    return out


def _visible(obj: dict, row: dict) -> bool:
    """A study is shown for an objective only if it saw no more than that objective's
    in-sample period: one run for an objective with a LATER split read this one's holdout."""
    split = obj.get("split_date")
    return not split or (row.get("cut") is not None and str(row["cut"]) <= str(split))


def list_studies(obj: dict, limit: int = 500) -> list[dict]:
    """Latest study per cache key (with how many times it was run), newest first."""
    conn = _db()
    with _lock():
        rows = conn.execute(
            "SELECT d.*, (SELECT count(*) FROM deci_plots x WHERE x.project_id=d.project_id AND x.key=d.key) AS runs "
            "FROM deci_plots d WHERE d.project_id=? AND d.id = (SELECT max(id) FROM deci_plots y "
            "WHERE y.project_id=d.project_id AND y.key=d.key) ORDER BY d.ts DESC LIMIT ?",
            (obj["project_id"], limit)).fetchall()
    out = []
    for r in rows:
        row = _row(r)
        row["runs"] = r["runs"]
        if _visible(obj, row):
            out.append(row)
    return out


def get_study(sid: int) -> dict:
    conn = _db()
    with _lock():
        r = conn.execute("SELECT * FROM deci_plots WHERE id=?", (sid,)).fetchone()
    if r is None:
        raise HTTPException(status_code=404, detail="no such study")
    return _row(r, full=True)


def compact(study: dict) -> dict:
    """A study trimmed for a model's context: the verdict, the best cell per timeframe and
    the decile means of the strongest cell."""
    res = study.get("result") or {}
    s = study.get("summary") or res.get("summary") or {}
    best = s.get("best") or {}
    shape = None
    if best:
        cell = (((res.get("timeframes") or {}).get(best["timeframe"]) or {}).get("horizons") or {}).get(str(best["horizon"]))
        if cell:
            shape = [d["mean_bps"] for d in cell["deciles"]]
    return {"id": study.get("id"), "signal": study.get("signal"), "verdict": s.get("verdict"),
            "direction": s.get("direction"), "best": best or None, "by_timeframe": s.get("by_timeframe"),
            "mean_bps_by_decile_at_best": shape, "cached": study.get("cached", False),
            "note": ("in-sample only, rolling (trailing) decile edges, forward returns within the session; "
                     "t-stats are overlap-adjusted (n/h). verdict: monotone = |t| >= 3, Spearman >= 0.7 and the "
                     "same sign in every sub-period; unstable = sign flips between sub-periods; extremes = only "
                     "the end deciles differ; flat = nothing there. Compare spreads with the cost per trade.")}


# =======================================================================================
# Running
# =======================================================================================
class DeciReq(BaseModel):
    signal: str = Field(..., min_length=1, max_length=500)
    timeframes: list[str] | None = Field(None, max_length=8)
    horizons: list[int] | None = Field(None, max_length=8)
    window_days: int = Field(deci_core.WINDOW_DAYS, ge=2, le=250, description="past sessions the decile edges come from")
    window_bars: int | None = Field(None, ge=100, le=200_000, description="trailing bars instead of sessions")
    author: str = Field("", max_length=200)
    force: bool = Field(False, description="run again even if the same study is stored (history is kept)")
    compact: bool = Field(False, description="return the trimmed form an agent reads")


def _norm(req) -> tuple[list[str], list[int]]:
    import re

    tfs = list(dict.fromkeys(req.timeframes or deci_core.TIMEFRAMES))
    for tf in tfs:
        if not re.fullmatch(r"\d{1,4}(s|min|h)", tf):
            raise HTTPException(status_code=400, detail=f"timeframe {tf!r}: use e.g. 10s, 20s, 30s, 1min, 5min")
    hs = sorted({int(h) for h in (req.horizons or deci_core.HORIZONS)})
    if not hs or hs[0] < 1 or hs[-1] > 500:
        raise HTTPException(status_code=400, detail="horizons are bars of the timeframe, 1..500")
    return tfs, hs


def _setup(oid: str) -> tuple[dict, dict]:
    from .objectives import get_objective

    obj = get_objective(oid)
    project = projects.get(obj["project_id"])
    if project is None:
        raise HTTPException(status_code=404, detail="no such project")
    if not (obj.get("dataset") and obj.get("time_column") and obj["metric"].get("price_column")):
        raise HTTPException(status_code=400, detail="a decile study needs an objective with a dataset, time column and price column")
    return obj, project


async def _run_specs(obj: dict, project: dict, specs: list[dict], tfs: list[str], hs: list[int],
                     window_days: int, window_bars: int | None) -> tuple[dict, dict, str]:
    """Run the harness for several signals at once (the data loads once). In-sample only."""
    from . import datasource
    from .objectives import _run, build_mirror

    split = obj.get("split_date")
    cfg = {"dataset": obj["dataset"], "time_column": obj["time_column"], "price_column": obj["metric"]["price_column"],
           "cut": split, "signals": specs, "timeframes": tfs, "horizons": hs, "window_days": window_days,
           "window_bars": window_bars}
    mirror = await asyncio.to_thread(build_mirror, obj, project["data_dir"]) if split else None
    catalog = await asyncio.to_thread(datasource.catalog, project["data_dir"])
    async with _SLOT:
        rep = await _run(harness_code(cfg), project["data_dir"], catalog, mirror, TIMEOUT_S, obj, split)
    res = rep.get("result") or {}
    return res.get("deci") or {}, res.get("deci_errors") or {}, (rep.get("stderr") or "")[-3000:]


def _params(obj: dict, spec: dict, tfs, hs, window_days, window_bars) -> dict:
    return {"kind": spec["kind"], "view": spec.get("view"), "expr": spec["expr"], "dataset": obj["dataset"],
            "price_column": obj["metric"]["price_column"], "timeframes": tfs, "horizons": hs,
            "window": {"bars": window_bars} if window_bars else {"days": window_days},
            "cut": obj.get("split_date"), "core_version": deci_core.CORE_VERSION,
            "feature_built": spec.get("feature_built")}


async def run_study(oid: str, req: DeciReq) -> dict:
    obj, project = _setup(oid)
    tfs, hs = _norm(req)
    spec = await asyncio.to_thread(resolve_signal, obj, project["data_dir"], req.signal)
    key = cache_key(obj, spec, tfs, hs, req.window_days, req.window_bars)
    if not req.force:
        hit = cached(obj["project_id"], key)
        if hit and _visible(obj, hit):
            return {**hit, "cached": True}
    out, errors, stderr = await _run_specs(obj, project, [spec], tfs, hs, req.window_days, req.window_bars)
    result = out.get(spec["signal"])
    if not result:
        raise HTTPException(status_code=400, detail=f"decile study failed: {errors.get(spec['signal']) or stderr or 'no result'}")
    sid = store(obj["project_id"], oid, key, req.author, spec,
                _params(obj, spec, tfs, hs, req.window_days, req.window_bars), result)
    return {**get_study(sid), "cached": False}


@router.post("/objectives/{oid}/deci-plots")
async def create_study(oid: str, req: DeciReq) -> dict:
    """Run (or fetch the stored) decile study of one signal on in-sample data."""
    study = await run_study(oid, req)
    return compact(study) if req.compact else study


@router.get("/objectives/{oid}/deci-plots")
async def studies(oid: str, compact: bool = False) -> dict:
    from .objectives import get_objective

    obj = get_objective(oid)
    rows = await asyncio.to_thread(list_studies, obj)
    if compact:
        # What an agent reads before deciding to run a study: one line per signal studied.
        return {"studied": len(rows), "studies": [
            {"signal": r["signal"], "verdict": (r["summary"] or {}).get("verdict"),
             "direction": (r["summary"] or {}).get("direction"), "best": (r["summary"] or {}).get("best")}
            for r in rows[:120]], "note": "in-sample, rolling deciles; call deci_plot(signal=...) for one in full"}
    return {"studies": rows, "batch": _batches.get(oid), "split_date": obj.get("split_date"),
            "defaults": {"timeframes": list(deci_core.TIMEFRAMES), "horizons": list(deci_core.HORIZONS),
                         "window_days": deci_core.WINDOW_DAYS}}


@router.get("/deci-plots/{sid}")
async def study_detail(sid: int) -> dict:
    return await asyncio.to_thread(get_study, sid)


@router.get("/objectives/{oid}/deci-plots/signals")
async def signals(oid: str) -> dict:
    """What can be studied: the dataset's numeric columns and each forecast feature's columns."""
    from .objectives import list_features

    obj, project = _setup(oid)
    cols = await asyncio.to_thread(dataset_columns, project["data_dir"], obj["dataset"])
    feats = []
    for f in list_features(oid):
        try:
            feats.append({"view": f["view"], "columns": feature_columns(oid, f["view"])[1]})
        except HTTPException:
            continue
    return {"columns": [c for c in _numeric(cols) if c != obj["time_column"]], "features": feats}


# =======================================================================================
# Batch: every column, one sandbox run per chunk, in the background
# =======================================================================================
class BatchReq(BaseModel):
    columns: list[str] | None = Field(None, max_length=400)
    timeframes: list[str] | None = Field(None, max_length=8)
    horizons: list[int] | None = Field(None, max_length=8)
    window_days: int = Field(deci_core.WINDOW_DAYS, ge=2, le=250)
    window_bars: int | None = Field(None, ge=100, le=200_000)
    author: str = Field("operator", max_length=200)


async def _batch(job: dict, obj: dict, project: dict, specs: list[dict], tfs, hs, req: BatchReq) -> None:
    try:
        for i in range(0, len(specs), BATCH_CHUNK):
            if job.get("cancel"):
                job["phase"] = "cancelled"
                return
            chunk = specs[i:i + BATCH_CHUNK]
            job["current"] = ", ".join(s["signal"] for s in chunk)
            out, errors, stderr = await _run_specs(obj, project, chunk, tfs, hs, req.window_days, req.window_bars)
            for s in chunk:
                if s["signal"] in out:
                    key = cache_key(obj, s, tfs, hs, req.window_days, req.window_bars)
                    await asyncio.to_thread(store, obj["project_id"], obj["id"], key, req.author, s,
                                            _params(obj, s, tfs, hs, req.window_days, req.window_bars), out[s["signal"]])
                    job["done"] += 1
                else:
                    job["failed"].append({"signal": s["signal"], "error": errors.get(s["signal"]) or stderr[-300:] or "no result"})
        job["phase"] = "done"
    except Exception as exc:  # noqa: BLE001 -- a batch must end with a readable state
        logger.exception("deci-plot batch failed")
        job.update(phase="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        job["finished_at"] = time.time()
        job["current"] = None


@router.post("/objectives/{oid}/deci-plots/batch")
async def start_batch(oid: str, req: BatchReq) -> dict:
    """Study every numeric column (or the given ones) that has no stored study yet."""
    obj, project = _setup(oid)
    running = _batches.get(oid)
    if running and running["phase"] == "running":
        return running
    tfs, hs = _norm(req)
    cols = await asyncio.to_thread(dataset_columns, project["data_dir"], obj["dataset"])
    wanted = req.columns or [c for c in _numeric(cols) if c.lower() not in BATCH_SKIP
                             and c not in (obj["time_column"], obj["metric"]["price_column"])]
    specs, skipped = [], 0
    for c in wanted:
        try:
            s = resolve_signal(obj, project["data_dir"], c)
        except HTTPException:
            continue
        hit = cached(obj["project_id"], cache_key(obj, s, tfs, hs, req.window_days, req.window_bars))
        if hit and _visible(obj, hit):
            skipped += 1
            continue
        specs.append(s)
    job = {"objective_id": oid, "phase": "running" if specs else "done", "total": len(specs), "done": 0,
           "already_studied": skipped, "failed": [], "started_at": time.time(), "current": None, "error": None}
    _batches[oid] = job
    if specs:
        asyncio.create_task(_batch(job, obj, project, specs, tfs, hs, req))
    return job


@router.get("/objectives/{oid}/deci-plots/batch")
async def batch_status(oid: str) -> dict:
    return {"batch": _batches.get(oid)}


@router.post("/objectives/{oid}/deci-plots/batch/cancel")
async def cancel_batch(oid: str) -> dict:
    job = _batches.get(oid)
    if job and job["phase"] == "running":
        job["cancel"] = True
    return {"batch": job}


# =======================================================================================
# What agents and the mentor read
# =======================================================================================
def brief(obj: dict, per_timeframe: int = 4) -> dict | None:
    """The team's accumulated decile knowledge, compact: the strongest signals per timeframe
    and the ones shown to be flat (so nobody studies them again)."""
    try:
        rows = list_studies(obj)
    except Exception:  # noqa: BLE001 -- the brief must never cost an agent its iteration
        logger.exception("deci brief failed")
        return None
    if not rows:
        return None
    by_tf: dict[str, list[dict]] = {}
    flat, unstable = [], []
    for r in rows:
        s = r["summary"] or {}
        if s.get("verdict") == "flat":
            flat.append(r["signal"])
        elif s.get("verdict") == "unstable":
            unstable.append(r["signal"])
        for tf, b in (s.get("by_timeframe") or {}).items():
            if b.get("verdict") in ("monotone", "weak", "extremes") and abs(b.get("t_spread") or 0) >= 2:
                by_tf.setdefault(tf, []).append({"signal": r["signal"], "h": b["horizon"], "spread_bps": b["spread_bps"],
                                                 "t": b["t_spread"], "rho": b["spearman"],
                                                 "consistency": b["consistency"], "verdict": b["verdict"]})
    order = {tf: i for i, tf in enumerate(deci_core.TIMEFRAMES)}
    best = {tf: sorted(v, key=lambda x: -abs(x["t"] or 0))[:per_timeframe]
            for tf, v in sorted(by_tf.items(), key=lambda kv: order.get(kv[0], 99))}
    return {"studied": len(rows), "best_by_timeframe": best, "flat": flat[:40], "unstable": unstable[:15],
            "shapes": _shapes(rows)}


SHAPES_IN_BRIEF = 8


def shape_of(means: list, rho: float | None) -> str:
    """One word for how forward return moves across the ten deciles: a straight line is
    tradeable in proportion to the signal; a U or an edge-only effect is only worth trading at
    the extremes -- which a top-minus-bottom spread alone does not reveal."""
    m = [x for x in means if x is not None]
    if len(m) < 6:
        return "too few buckets"
    mid = sum(m[3:7]) / len(m[3:7])
    lo, hi = m[0] - mid, m[-1] - mid
    span = (max(m) - min(m)) or 1e-9
    if rho is not None and abs(rho) >= 0.7:
        return "rising" if rho > 0 else "falling"
    if lo * hi > 0 and min(abs(lo), abs(hi)) > 0.35 * span:
        return "U-shaped (both extremes above the middle)" if lo > 0 else "inverted U (both extremes below the middle)"
    if abs(hi) > 2 * abs(lo) and abs(hi) > 0.35 * span:
        return f"top decile only ({'up' if hi > 0 else 'down'})"
    if abs(lo) > 2 * abs(hi) and abs(lo) > 0.35 * span:
        return f"bottom decile only ({'up' if lo > 0 else 'down'})"
    return "no clear shape"


def _shapes(rows: list[dict]) -> list[dict]:
    """The decile curve of the strongest studied signals (in-sample, at each one's best cell)."""
    ranked = sorted((r for r in rows if ((r["summary"] or {}).get("best") or {}).get("t_spread") is not None
                     and (r["summary"] or {}).get("verdict") not in ("flat", "unstable")),
                    key=lambda r: -abs(r["summary"]["best"]["t_spread"]))[:SHAPES_IN_BRIEF]
    out = []
    for r in ranked:
        try:
            res = get_study(r["id"]).get("result") or {}
        except Exception:  # noqa: BLE001 -- one unreadable study must not cost the brief
            continue
        # The cell worth trading is the one with the LARGEST spread that is still solid, not
        # the highest t: 10 s x 1 bar always wins on t (the most samples) with a spread far
        # below one round trip's cost.
        cells = [(tf, h, c) for tf, t in (res.get("timeframes") or {}).items()
                 for h, c in (t.get("horizons") or {}).items()
                 if c.get("spread_bps") is not None and abs(c.get("t_spread") or 0) >= 3
                 and c.get("verdict") not in ("unstable", "flat")]
        if not cells:
            continue
        tf, h, c = max(cells, key=lambda x: abs(x[2]["spread_bps"]))
        means = [d["mean_bps"] for d in c["deciles"]]
        out.append({"signal": r["signal"], "timeframe": tf, "h": int(h),
                    "means": [None if v is None else round(v, 2) for v in means],
                    "spread_bps": c["spread_bps"], "t": c["t_spread"], "rho": c.get("spearman"),
                    "shape": shape_of(means, c.get("spearman"))})
    return sorted(out, key=lambda s: -abs(s["spread_bps"]))
