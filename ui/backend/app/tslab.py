"""Forecast Lab: which inputs actually make a time-series forecast better?

A covariate model (Chronos-2) can forecast a target -- Close, IntrVol, a GEX field -- while
reading any other columns as inputs. Whether a given input HELPS is an empirical question, and
this module answers it the same way for the quick test and the full analysis: forecast at many
points in the IN-SAMPLE period (the objective's holdout is never touched), compare each forecast
with what actually happened, and score it.

Scores (all in-sample, per run):

* ``skill``      -- 1 - MAE(median forecast) / MAE(no-change forecast). > 0 beats "it stays put".
* ``direction``  -- share of points where the forecast called the direction of the move.
* ``coverage``   -- share of outcomes inside the 10-90% band (0.80 is calibrated).
* ``qloss``      -- mean pinball loss over the 10/50/90% quantiles, relative to no-change
                    (lower is better): rewards a well-placed band, not just a good median.

"Reversing the model" -- the full analysis:

1. **Baseline**: the target alone.
2. **All inputs** together.
3. **Leave-one-out**: all inputs minus one, for each input. How much worse the forecast gets
   without an input is its IMPACT (positive = it helped, negative = it hurt).
4. **Solo**: the target plus one input, for each. Its gain over the baseline is its SOLO LIFT.
5. **Best combination**: greedy forward selection from the baseline, adding at each step the
   input (among the strongest candidates) that improves skill most, until nothing does.

Everything runs as a background job with progress; results are stored (objectives.sqlite3) so
the page, the agents' brief and a forecast feature can all use the best combination found.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import datasource, projects

router = APIRouter(prefix="/tslab", tags=["forecast-lab"])

QS = [0.1, 0.5, 0.9]
BATCH_POINTS = 1_500_000        # floats per forecast request (targets + inputs x context x batch)
MAX_CANDIDATES = 40
GREEDY_POOL = 10
GREEDY_MAX = 8

_jobs: dict[str, dict] = {}


def _db():
    from .objectives import _lock, db

    conn = db()
    with _lock:
        conn.execute("""CREATE TABLE IF NOT EXISTS tslab_runs (
            id TEXT PRIMARY KEY, project_id TEXT NOT NULL, objective_id TEXT, created_at REAL NOT NULL,
            status TEXT NOT NULL, params TEXT NOT NULL, results TEXT NOT NULL DEFAULT '{}')""")
        conn.commit()
    return conn


def _lock():
    from .objectives import _lock as lock

    return lock


# =======================================================================================
# Data: target + inputs, aligned, in-sample only
# =======================================================================================
class Setup(BaseModel):
    project_id: str
    objective_id: str | None = None
    dataset: str | None = None
    target: str = Field(..., max_length=500)
    inputs: list[str] = Field(default_factory=list, max_length=MAX_CANDIDATES)
    horizon: int = Field(30, ge=1, le=512)
    context: int = Field(1024, ge=64, le=8192)
    bar: str | None = Field(None, max_length=10, pattern=r"^\d+(s|min|h)$")
    points: int = Field(300, ge=30, le=2000, description="forecast points to score")
    model: str | None = None
    end: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}", description="evaluate before this date")


def _context_for(req: Setup) -> tuple[dict, dict | None, str, str | None]:
    """(project, objective, dataset, end date) -- defaulting to the project's objective so the
    holdout stays unseen."""
    project = projects.get(req.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="no such project")
    obj = None
    if req.objective_id:
        from .objectives import get_objective

        obj = get_objective(req.objective_id)
    dataset = req.dataset or (obj or {}).get("dataset")
    if not dataset:
        raise HTTPException(status_code=400, detail="choose a dataset (or an objective that has one)")
    end = req.end or (obj or {}).get("split_date")
    return project, obj, dataset, end


def load_frame(project: dict, obj: dict | None, dataset: str, columns: list[str], bar: str | None,
               end: str | None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(timestamps, {expression: values}) on one time grid, rows before `end` only. With `bar`,
    rows are bucketed and each column takes its LAST value in the bar (stamped at the bar's last
    row, so it is causal)."""
    from .objectives import _abs, _reader, detect_time_column, series_expression

    data_dir = project["data_dir"]
    item = next((i for i in datasource.catalog(data_dir) if dataset in (i["view"], i["path"])), None)
    if item is None:
        raise HTTPException(status_code=400, detail=f"no dataset {dataset!r}")
    tc = detect_time_column(data_dir, item, (obj or {}).get("time_column"))
    if tc is None:
        raise HTTPException(status_code=400, detail=f"{dataset} has no time column")
    con = duckdb.connect(":memory:")
    try:
        cols = [c[0] for c in con.execute(f"DESCRIBE SELECT * FROM {_reader(item)}('{_abs(data_dir, item)}')").fetchall()]
    finally:
        con.close()
    exprs = {c: series_expression(c, cols) for c in dict.fromkeys(columns)}
    where = f"AND t < TIMESTAMP '{end}'" if end else ""
    names = list(exprs)
    if bar:
        n, unit = int(bar[:-3] if bar.endswith("min") else bar[:-1]), ("min" if bar.endswith("min") else bar[-1])
        secs = n * {"s": 1, "min": 60, "h": 3600}[unit]
        inner = ", ".join(f"TRY_CAST(({e}) AS DOUBLE) AS c{i}" for i, e in enumerate(exprs.values()))
        agg = ", ".join(f"arg_max(c{i}, t) AS c{i}" for i in range(len(names)))
        sql = (f"SELECT max(t) AS t, {agg} FROM (SELECT TRY_CAST(\"{tc}\" AS TIMESTAMP) AS t, {inner} "
               f"FROM {_reader(item)}('{_abs(data_dir, item)}')) WHERE t IS NOT NULL {where} "
               f"GROUP BY time_bucket(INTERVAL '{secs} seconds', t) ORDER BY t")
    else:
        agg = ", ".join(f"avg(TRY_CAST(({e}) AS DOUBLE)) AS c{i}" for i, e in enumerate(exprs.values()))
        sql = (f"SELECT t, * EXCLUDE (t) FROM (SELECT TRY_CAST(\"{tc}\" AS TIMESTAMP) AS t, {agg} "
               f"FROM {_reader(item)}('{_abs(data_dir, item)}') GROUP BY 1) WHERE t IS NOT NULL {where} ORDER BY t")
    con = duckdb.connect(":memory:")
    try:
        con.execute("SET TimeZone = 'UTC'")
        got = con.execute(sql).fetchnumpy()
    except duckdb.Error as exc:
        raise HTTPException(status_code=400, detail=f"could not load the columns: {str(exc).splitlines()[0]}") from None
    finally:
        con.close()
    t = np.ma.filled(np.ma.asarray(got["t"]), np.datetime64("NaT"))
    # fetchnumpy returns MASKED arrays where a column has NULLs; np.asarray would expose the
    # masked slots' arbitrary contents as real numbers. Fill them with NaN explicitly.
    vals = {name: np.ma.filled(np.ma.asarray(got[f"c{i}"]).astype(np.float64), np.nan) for i, name in enumerate(names)}
    # Keep rows where every column is present; forward-fill small gaps in inputs first.
    for name in names:
        v = vals[name]
        if np.isnan(v).any():
            idx = np.where(~np.isnan(v), np.arange(len(v)), 0)
            np.maximum.accumulate(idx, out=idx)
            vals[name] = v[idx]
    ok = np.ones(len(t), dtype=bool)
    for v in vals.values():
        ok &= np.isfinite(v)
    return t[ok], {k: v[ok] for k, v in vals.items()}


def _anchors(n: int, context: int, horizon: int, points: int) -> list[int]:
    lo, hi = context, n - horizon
    if hi - lo < 10:
        raise HTTPException(status_code=400, detail=f"not enough in-sample history: {n} rows for context {context} + horizon {horizon}")
    step = max(1, (hi - lo) // points)
    return list(range(lo, hi, step))[:points]


# =======================================================================================
# Scoring one combination
# =======================================================================================
async def run_combo(mgr, model: str, target: np.ndarray, inputs: dict[str, np.ndarray], anchors: list[int],
                    context: int, horizon: int) -> dict:
    per_item = context * (1 + len(inputs))
    batch = max(1, min(512, BATCH_POINTS // per_item))
    med, lo, hi = [], [], []
    for b in range(0, len(anchors), batch):
        chunk = anchors[b:b + batch]
        items = []
        for a in chunk:
            it: dict[str, Any] = {"target": target[a - context:a].tolist()}
            if inputs:
                it["past_covariates"] = {k: v[a - context:a].tolist() for k, v in inputs.items()}
            items.append(it)
        res = await mgr.forecast(model, {"inputs": items, "horizon": horizon, "quantiles": QS})
        for fc in res.get("forecasts") or []:
            q = fc.get("quantiles") or {}
            med.append(fc["median"][-1])
            lo.append((q.get("0.1") or fc["median"])[-1])
            hi.append((q.get("0.9") or fc["median"])[-1])
    return score(target, anchors, horizon, np.array(med), np.array(lo), np.array(hi))


def score(target: np.ndarray, anchors: list[int], horizon: int, med, lo, hi) -> dict:
    last = np.array([target[a - 1] for a in anchors])
    real = np.array([target[a + horizon - 1] for a in anchors])
    mae_fc = float(np.mean(np.abs(med - real)))
    mae_nv = float(np.mean(np.abs(last - real)))
    moved = (real != last) & (med != last)
    direction = float(np.mean((med[moved] > last[moved]) == (real[moved] > last[moved]))) if moved.any() else None

    def pinball(q, pred):
        d = real - pred
        return np.mean(np.maximum(q * d, (q - 1) * d))

    ql = float((pinball(0.1, lo) + pinball(0.5, med) + pinball(0.9, hi)) / 3)
    ql_nv = float((pinball(0.1, last) + pinball(0.5, last) + pinball(0.9, last)) / 3)
    return {
        # Per-point errors, scaled by the no-change MAE, for paired comparisons between runs at
        # the same points (dropped before anything is stored or sent).
        "_err": (np.abs(med - real) / mae_nv) if mae_nv > 0 else None,
        "skill": round(1 - mae_fc / mae_nv, 5) if mae_nv > 0 else None,
        "direction": round(direction, 4) if direction is not None else None,
        "coverage": round(float(np.mean((real >= lo) & (real <= hi))), 4),
        "qloss_rel": round(ql / ql_nv, 5) if ql_nv > 0 else None,
        "points": len(anchors),
    }


# =======================================================================================
# Model
# =======================================================================================
def _covariate_model(name: str | None):
    from .tsfm import ts_manager

    running = [i for i in ts_manager.running()]
    stats = {s["model_id"]: s for s in ts_manager.statuses()}
    cands = [i for i in running if (stats.get(i.model_id, {}).get("health") or {}).get("supports_covariates")]
    if name:
        inst = next((i for i in running if i.model_id == name), None)
        if inst is None:
            raise HTTPException(status_code=400, detail=f"{name} is not loaded")
        if inst not in cands:
            raise HTTPException(status_code=400, detail=f"{name} cannot take input series; load amazon/chronos-2")
        return ts_manager, inst.model_id
    if not cands:
        raise HTTPException(status_code=409, detail="load a model that takes input series (amazon/chronos-2) on the Models page")
    return ts_manager, cands[0].model_id


# =======================================================================================
# Quick test
# =======================================================================================
@router.post("/test")
async def test(req: Setup) -> dict:
    """One combination vs the target alone, plus example forecasts to plot."""
    project, obj, dataset, end = _context_for(req)
    mgr, model = _covariate_model(req.model)
    t, cols = await asyncio.to_thread(load_frame, project, obj, dataset, [req.target, *req.inputs], req.bar, end)
    target = cols[req.target]
    inputs = {k: cols[k] for k in req.inputs if k != req.target}
    anchors = _anchors(len(target), req.context, req.horizon, req.points)
    t0 = time.time()
    base = await run_combo(mgr, model, target, {}, anchors, req.context, req.horizon)
    combo = await run_combo(mgr, model, target, inputs, anchors, req.context, req.horizon) if inputs else base
    # Examples: 3 points spread through the period, history tail + forecast band + actual.
    examples = []
    for a in [anchors[len(anchors) // 6], anchors[len(anchors) // 2], anchors[(5 * len(anchors)) // 6]]:
        res = []
        for use in ({}, inputs):
            item: dict[str, Any] = {"target": target[a - req.context:a].tolist()}
            if use:
                item["past_covariates"] = {k: v[a - req.context:a].tolist() for k, v in use.items()}
            f = (await mgr.forecast(model, {"inputs": [item], "horizon": req.horizon, "quantiles": QS}))["forecasts"][0]
            res.append({"median": f["median"], "q10": f["quantiles"]["0.1"], "q90": f["quantiles"]["0.9"]})
        tail = min(req.horizon * 4, a)
        examples.append({"t": str(t[a - 1]), "history": target[a - tail:a].tolist(),
                         "actual": target[a:a + req.horizon].tolist(), "baseline": res[0], "with_inputs": res[1]})
    return {"model": model, "dataset": dataset, "end": end, "target": req.target, "inputs": list(inputs),
            "rows": len(target), "baseline": _public(base), "combo": _public(combo),
            "lift": paired(base, combo) if inputs else None, "examples": examples,
            "seconds": round(time.time() - t0, 1),
            "note": None if end else "no split date: scored on all data (choose an objective to keep its holdout unseen)"}


# =======================================================================================
# Full analysis (background job)
# =======================================================================================
def _key(inputs: list[str]) -> str:
    return "|".join(sorted(inputs))


async def _analyze(job: dict, req: Setup) -> None:
    try:
        project, obj, dataset, end = _context_for(req)
        mgr, model = _covariate_model(req.model)
        cands = [c for c in dict.fromkeys(req.inputs) if c != req.target][:MAX_CANDIDATES]
        job.update(phase="loading data", model=model, dataset=dataset, end=end)
        t, cols = await asyncio.to_thread(load_frame, project, obj, dataset, [req.target, *cands], req.bar, end)
        target = cols[req.target]
        anchors = _anchors(len(target), req.context, req.horizon, req.points)
        runs: dict[str, dict] = {}
        k = len(cands)
        job["total"] = 2 + 2 * k + GREEDY_MAX * min(GREEDY_POOL, k)
        job["done"] = 0

        async def run(inputs: list[str], kind: str) -> dict:
            key = _key(inputs)
            if key not in runs:
                job["current"] = f"{kind}: {', '.join(inputs) or '(target only)'}"
                sc = await run_combo(mgr, model, target, {c: cols[c] for c in inputs}, anchors, req.context, req.horizon)
                runs[key] = {"inputs": sorted(inputs), "kind": kind, **sc}
            job["done"] += 1
            job["runs"] = [_public(r) for r in runs.values()]
            return runs[key]

        job["phase"] = "baseline and all inputs"
        base = await run([], "baseline")
        full = await run(cands, "all") if cands else base
        job["phase"] = "leave-one-out"
        impact = []
        for c in cands:
            without = await run([x for x in cands if x != c], "leave-one-out")
            impact.append({"input": c, **{f"impact_{k2}": v for k2, v in paired(without, full).items()},
                           "impact_skill": _d(full["skill"], without["skill"]),
                           "impact_direction": _d(full["direction"], without["direction"]),
                           "impact_qloss": _d(without["qloss_rel"], full["qloss_rel"])})
        job["phase"] = "solo"
        solo = []
        for c in cands:
            one = await run([c], "solo")
            solo.append({"input": c, **{f"lift_{k2}": v for k2, v in paired(base, one).items()},
                         "lift_skill": _d(one["skill"], base["skill"]),
                         "lift_direction": _d(one["direction"], base["direction"]),
                         "lift_qloss": _d(base["qloss_rel"], one["qloss_rel"])})
        job["phase"] = "best combination"
        # The pool: the strongest candidates by solo lift and by impact.
        rank = {c: 0.0 for c in cands}
        for s in solo:
            rank[s["input"]] += (s["lift_skill"] or 0)
        for i in impact:
            rank[i["input"]] += (i["impact_skill"] or 0)
        pool = sorted(cands, key=lambda c: rank[c], reverse=True)[:GREEDY_POOL]
        chosen: list[str] = []
        best = base
        path = [{"inputs": [], "skill": base["skill"], "direction": base["direction"], "added": None}]
        while pool and len(chosen) < GREEDY_MAX:
            trials = []
            for c in pool:
                trials.append((await run(chosen + [c], "greedy"), c))
            top, c = max(trials, key=lambda x: (x[0]["skill"] if x[0]["skill"] is not None else -9))
            if (top["skill"] or -9) <= (best["skill"] or -9) + 0.0005:
                break
            chosen.append(c)
            pool.remove(c)
            best = top
            step = paired(best, top)
            path.append({"inputs": list(chosen), "skill": top["skill"], "direction": top["direction"], "added": c,
                         "gain": step["gain"], "se": step["se"], "significant": step["significant"]})
        results = {
            "model": model, "dataset": dataset, "end": end, "target": req.target, "anchors": len(anchors),
            "baseline": _public(base), "all": {**_public(full), **{f"vs_baseline_{k2}": v for k2, v in paired(base, full).items()}}, "impact": sorted(impact, key=lambda x: -(x["impact_skill"] or 0)),
            "solo": sorted(solo, key=lambda x: -(x["lift_skill"] or 0)), "greedy_path": path,
            "best": {"inputs": chosen, **{k2: best.get(k2) for k2 in ("skill", "direction", "coverage", "qloss_rel")},
                     **{f"vs_baseline_{k2}": v for k2, v in paired(base, best).items()}},
            "runs": [_public(r) for r in runs.values()],
        }
        job.update(phase="done", results=results, finished_at=time.time())
        conn = _db()
        with _lock():
            conn.execute("UPDATE tslab_runs SET status='done', results=? WHERE id=?", (json.dumps(results), job["id"]))
            conn.commit()
    except HTTPException as exc:
        job.update(phase="error", error=exc.detail)
    except Exception as exc:  # noqa: BLE001
        job.update(phase="error", error=f"{type(exc).__name__}: {exc}")
    if job.get("phase") == "error":
        conn = _db()
        with _lock():
            conn.execute("UPDATE tslab_runs SET status='error', results=? WHERE id=?",
                         (json.dumps({"error": job["error"]}), job["id"]))
            conn.commit()


def _d(a, b):
    return None if a is None or b is None else round(a - b, 5)


def paired(worse: dict, better: dict) -> dict:
    """How much `better` beats `worse`, point by point at the same forecast points: mean skill
    gain, its standard error, and whether the gain is outside +/- 2 SE (i.e. not just noise)."""
    a, b = worse.get("_err"), better.get("_err")
    if a is None or b is None or len(a) != len(b) or len(a) < 10:
        return {"gain": None, "se": None, "significant": False}
    d = a - b
    gain, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    return {"gain": round(gain, 5), "se": round(se, 5), "significant": abs(gain) > 2 * se}


def _public(run: dict) -> dict:
    return {k: v for k, v in run.items() if not k.startswith("_")}


@router.post("/analyze")
async def analyze(req: Setup) -> dict:
    if not req.inputs:
        raise HTTPException(status_code=400, detail="choose the candidate inputs to analyse")
    _context_for(req)
    _covariate_model(req.model)
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "phase": "queued", "done": 0, "total": 1, "started_at": time.time(), "error": None,
           "params": req.model_dump()}
    _jobs[jid] = job
    conn = _db()
    with _lock():
        conn.execute("INSERT INTO tslab_runs (id, project_id, objective_id, created_at, status, params) VALUES (?,?,?,?,?,?)",
                     (jid, req.project_id, req.objective_id, time.time(), "running", json.dumps(req.model_dump())))
        conn.commit()
    asyncio.create_task(_analyze(job, req))
    return job


@router.get("/jobs/{jid}")
async def job_status(jid: str) -> dict:
    job = _jobs.get(jid)
    if job is None:
        r = get_run(jid)
        return {"id": jid, "phase": r["status"], "results": r["results"], "params": r["params"]}
    return job


def get_run(rid: str) -> dict:
    conn = _db()
    with _lock():
        r = conn.execute("SELECT * FROM tslab_runs WHERE id=?", (rid,)).fetchone()
    if r is None:
        raise HTTPException(status_code=404, detail="no such analysis")
    return {**dict(r), "params": json.loads(r["params"]), "results": json.loads(r["results"] or "{}")}


@router.get("/analyses")
async def analyses(project_id: str) -> dict:
    conn = _db()
    with _lock():
        rows = conn.execute("SELECT id, objective_id, created_at, status, params, results FROM tslab_runs "
                            "WHERE project_id=? ORDER BY created_at DESC LIMIT 30", (project_id,)).fetchall()
    out = []
    for r in rows:
        res = json.loads(r["results"] or "{}")
        out.append({"id": r["id"], "objective_id": r["objective_id"], "created_at": r["created_at"], "status": r["status"],
                    "params": json.loads(r["params"]), "best": res.get("best"), "baseline": res.get("baseline"),
                    "error": res.get("error")})
    return {"analyses": out, "running": [j for j in _jobs.values() if j["phase"] not in ("done", "error")]}


@router.get("/options")
async def options(project_id: str, objective_id: str | None = None, dataset: str | None = None) -> dict:
    """What the lab can offer: datasets, numeric columns by family, covariate-capable models,
    the objective's split, and the field scan's top fields as suggested inputs."""
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="no such project")
    from .objectives import field_guide, get_objective
    from .tsfm import ts_manager

    obj = get_objective(objective_id) if objective_id else None
    ds = dataset or (obj or {}).get("dataset")
    cat = datasource.catalog(project["data_dir"])
    numeric: list[str] = []
    families: dict[str, list[str]] = {}
    if ds:
        desc = await asyncio.to_thread(datasource.describe, project["data_dir"], ds)
        numeric = [c["name"] for c in desc["columns"] if any(x in c["type"].upper() for x in
                   ("DOUBLE", "FLOAT", "DECIMAL", "REAL", "INT", "NUMERIC"))]
        guide = field_guide({"id": f"lab:{ds}", "dataset": ds, "time_column": (obj or {}).get("time_column")},
                            project["data_dir"])
        families = {fam: [c for c in g["columns"] if c in numeric] for fam, g in guide.items()}
        families = {k: v for k, v in families.items() if v}
    suggested: list[str] = []
    try:
        from .library import latest_field_scan

        fs = latest_field_scan(project_id)
        if fs:
            suggested = [f["field"] for f in fs["result"]["fields"][:15] if f["field"] in numeric]
    except Exception:  # noqa: BLE001
        pass
    models = []
    for s in ts_manager.statuses():
        h = s.get("health") or {}
        models.append({"model": s["model_id"], "state": s["state"], "family": h.get("family"),
                       "covariates": bool(h.get("supports_covariates"))})
    return {"datasets": [c["view"] for c in cat], "dataset": ds, "numeric": numeric, "families": families,
            "suggested": suggested, "models": models, "split_date": (obj or {}).get("split_date"),
            "objective": {"id": obj["id"], "title": obj["title"]} if obj else None}
