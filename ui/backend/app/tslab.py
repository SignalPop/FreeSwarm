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

**Exploration** (``/tslab/explore``, the agents' ``explore_forecast_inputs``) is the same idea as
a budgeted search an agent or the mentor can start: baseline, a solo screen of every candidate,
greedy forward selection that only takes a step significant at the same points, then a
leave-one-out prune. Every combination anyone scores -- lab, analysis or exploration -- goes into
``tslab_combos`` keyed by what determines its score, so no combination is ever forecast twice and
the brief can say, per target, which inputs help and which were tested and are useless.

The **forecast report** (``/tslab/forecast-report/{oid}``) is the other half: for every forecast
feature built, what was sent to the model, its skill, the lift from its inputs, and whether the
candidates that used it did better -- judged against their own parent where possible.
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
        # Every combination ever scored, keyed by what determines its score (see combo_key), so
        # no combination is forecast twice -- by the lab, an exploration or an agent. `err` keeps
        # the per-point errors so later runs can still be compared with it point by point.
        conn.execute("""CREATE TABLE IF NOT EXISTS tslab_combos (
            key TEXT PRIMARY KEY, project_id TEXT NOT NULL, objective_id TEXT, dataset TEXT NOT NULL,
            target TEXT NOT NULL, inputs TEXT NOT NULL, horizon INTEGER NOT NULL, bar TEXT, model TEXT NOT NULL,
            context INTEGER NOT NULL, points INTEGER NOT NULL, end_date TEXT, n_rows INTEGER NOT NULL,
            ts REAL NOT NULL, author TEXT, kind TEXT, score TEXT NOT NULL, err TEXT)""")
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


def _note_streams(model: str, kind: str, dataset: str, t, anchors: list[int], req: "Setup", inputs: list[str],
                  role: str, cols: dict | None = None, end: str | None = None):
    """Tell the agent inspector which streams (and dates) this lab run sends the forecaster.
    Lab anchors read the `context` rows BEFORE the anchor (not including it). Returns the
    value capture (forecast_values) with the sample anchors' context values already in it."""
    from .objectives import _capture, _note_inputs, input_streams

    detail = input_streams(kind, model, dataset, t, anchors, req.context, req.horizon,
                           [(req.target, "target")] + [(c, role) for c in inputs],
                           bar=req.bar, inclusive=False)
    _note_inputs(model, detail)
    cap = _capture(detail, t, anchors, req.context, req.horizon, end, inclusive=False)
    for name, r in [(req.target, "target")] + [(c, role) for c in inputs]:
        if cols is not None and name in cols:
            cap.context_values(name, r, {name: cols[name]})
    return cap


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
                    context: int, horizon: int, capture=None, target_name: str = "", with_inputs: bool = True) -> dict:
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
        for j, fc in enumerate(res.get("forecasts") or []):
            if capture is not None:
                capture.output(chunk[j] if j < len(chunk) else -1, target_name, fc, with_inputs=with_inputs)
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
    cap = _note_streams(model, "lab test", dataset, t, anchors, req, list(inputs), "past covariate", cols, end)
    t0 = time.time()
    base = await run_combo(mgr, model, target, {}, anchors, req.context, req.horizon,
                           capture=cap, target_name=req.target, with_inputs=not inputs)
    combo = await run_combo(mgr, model, target, inputs, anchors, req.context, req.horizon,
                            capture=cap, target_name=req.target) if inputs else base
    cap.realized(req.target, target)
    cap.publish(model)
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
        _note_streams(model, "lab analysis", dataset, t, anchors, req, cands, "candidate input", cols, end).publish(model)
        runs: dict[str, dict] = {}
        k = len(cands)
        job["total"] = 2 + 2 * k + GREEDY_MAX * min(GREEDY_POOL, k)
        job["done"] = 0
        where = {"project_id": req.project_id, "objective_id": req.objective_id, "dataset": dataset,
                 "target": req.target, "horizon": req.horizon, "bar": req.bar, "model": model,
                 "context": req.context, "points": req.points, "end": end, "n_rows": len(target)}

        async def run(inputs: list[str], kind: str) -> dict:
            key = _key(inputs)
            if key not in runs:
                job["current"] = f"{kind}: {', '.join(inputs) or '(target only)'}"
                sc = await asyncio.to_thread(load_combo, combo_key(where, inputs))
                if sc is None:
                    sc = await run_combo(mgr, model, target, {c: cols[c] for c in inputs}, anchors, req.context, req.horizon)
                    await asyncio.to_thread(save_combo, where, inputs, sc, "analysis", kind)
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


# =======================================================================================
# The combination cache: no combination is ever forecast twice
# =======================================================================================
def combo_key(where: dict, inputs: list[str]) -> str:
    """What determines a combination's score: the data (dataset, target, bar, the in-sample end
    and its row count -- which fix the anchors), the model and its settings, and the SET of
    inputs (order does not matter to the model, so it does not matter to the key)."""
    import hashlib

    doc = {k: where.get(k) for k in ("dataset", "target", "horizon", "bar", "model", "context", "points", "end", "n_rows")}
    doc["inputs"] = sorted(dict.fromkeys(inputs))
    return hashlib.sha1(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()[:24]


def load_combo(key: str) -> dict | None:
    """A stored score (with its per-point errors restored for paired comparisons), or None."""
    conn = _db()
    with _lock():
        r = conn.execute("SELECT score, err FROM tslab_combos WHERE key=?", (key,)).fetchone()
    if r is None:
        return None
    sc = json.loads(r["score"])
    err = json.loads(r["err"]) if r["err"] else None
    sc["_err"] = np.array(err, dtype=float) if err is not None else None
    sc["cached"] = True
    return sc


def save_combo(where: dict, inputs: list[str], sc: dict, author: str = "", kind: str = "") -> None:
    err = sc.get("_err")
    conn = _db()
    with _lock():
        conn.execute(
            "INSERT OR REPLACE INTO tslab_combos (key, project_id, objective_id, dataset, target, inputs, horizon, bar, "
            "model, context, points, end_date, n_rows, ts, author, kind, score, err) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (combo_key(where, inputs), where["project_id"], where.get("objective_id"), where["dataset"], where["target"],
             json.dumps(sorted(dict.fromkeys(inputs))), where["horizon"], where.get("bar"), where["model"], where["context"],
             where["points"], where.get("end"), where["n_rows"], time.time(), author, kind,
             json.dumps(_public({k: v for k, v in sc.items() if k != "cached"})),
             json.dumps([round(float(x), 6) for x in err]) if err is not None else None))
        conn.commit()


# =======================================================================================
# Exploration: a budgeted, systematic search over input combinations (background job)
# =======================================================================================
class ExploreReq(Setup):
    budget: int = Field(40, ge=3, le=300, description="NEW combinations to forecast; stored ones are free")
    author: str = Field("", max_length=200)


def suggest_inputs(project_id: str, obj: dict | None, target: str, limit: int = 20) -> list[str]:
    """Candidate inputs when none are given: the field scan's strongest fields, then signals
    the decile studies found monotone -- what the team already has evidence for."""
    out: list[str] = []
    try:
        from .library import latest_field_scan

        fs = latest_field_scan(project_id)
        if fs:
            out += [f["field"] for f in fs["result"].get("fields", [])[:15]]
    except Exception:  # noqa: BLE001
        pass
    if obj is not None:
        try:
            from .deciplot import list_studies

            out += [s["signal"] for s in list_studies(obj) if s.get("kind") == "dataset"
                    and (s.get("summary") or {}).get("verdict") == "monotone"]
        except Exception:  # noqa: BLE001
            pass
    return [c for c in dict.fromkeys(out) if c != target][:limit]


async def explore_search(run, cands: list[str], job: dict | None = None) -> dict:
    """The search itself, given `run(inputs, kind) -> score | None` (None = out of budget).

    baseline -> solo screen of every candidate -> greedy forward selection that only accepts
    a step beating the current best beyond +/- 2 SE at the same points -> leave-one-out prune
    of the chosen set. Separate from the job so it can be tested with a fake forecaster."""
    job = job if job is not None else {}
    job["phase"] = "baseline"
    base = await run([], "baseline")
    if base is None:
        raise HTTPException(status_code=400, detail="budget too small for even the baseline")
    job["phase"] = "solo screen"
    solo = []
    for c in cands:
        one = await run([c], "solo")
        if one is None:
            break
        solo.append({"input": c, **{f"lift_{k}": v for k, v in paired(base, one).items()},
                     "lift_skill": _d(one["skill"], base["skill"]),
                     "lift_direction": _d(one["direction"], base["direction"]),
                     "lift_qloss": _d(base["qloss_rel"], one["qloss_rel"])})

    def z(s):  # solo lift in standard errors: the pool is ordered by evidence, not by luck
        return (s["lift_gain"] or 0) / s["lift_se"] if s.get("lift_se") else 0.0

    pool = [s["input"] for s in sorted(solo, key=z, reverse=True) if (s["lift_gain"] or 0) > 0][:GREEDY_POOL]
    job["phase"] = "greedy selection"
    chosen: list[str] = []
    best = base
    path = [{"inputs": [], "skill": base["skill"], "direction": base["direction"], "added": None}]
    while pool and len(chosen) < GREEDY_MAX:
        trials = []
        for c in pool:
            r = await run(chosen + [c], "greedy")
            if r is not None:
                trials.append((r, c))
        if not trials:
            break
        top, c = max(trials, key=lambda x: (x[0]["skill"] if x[0]["skill"] is not None else -9))
        step = paired(best, top)
        # Only a step that beats the current best beyond noise is taken: a greedy search that
        # accepts every +0.0001 ends up with eight inputs and a lucky score.
        if not (step["significant"] and (step["gain"] or 0) > 0):
            break
        chosen.append(c)
        pool.remove(c)
        best = top
        path.append({"inputs": list(chosen), "skill": top["skill"], "direction": top["direction"], "added": c,
                     "gain": step["gain"], "se": step["se"], "significant": step["significant"]})
    job["phase"] = "leave-one-out prune"
    pruned = []
    if len(chosen) >= 2:
        for c in list(chosen):
            without = await run([x for x in chosen if x != c], "prune")
            if without is None:
                break
            keep = paired(without, best)
            if not (keep["significant"] and (keep["gain"] or 0) > 0):
                chosen.remove(c)      # the set does as well without it: it is not pulling weight
                best = without
                pruned.append(c)
    helpful = [s["input"] for s in solo if s.get("lift_significant") and (s["lift_gain"] or 0) > 0]
    return {"baseline": _public(base), "solo": sorted(solo, key=lambda x: -(x["lift_skill"] or 0)),
            "greedy_path": path, "pruned": pruned,
            "best": {"inputs": chosen, **{k: best.get(k) for k in ("skill", "direction", "coverage", "qloss_rel")},
                     **{f"vs_baseline_{k}": v for k, v in paired(base, best).items()}},
            "helpful": helpful, "useless": [s["input"] for s in solo if s["input"] not in helpful]}


async def _explore(job: dict, req: ExploreReq) -> None:
    """The exploration job: every combination goes through the cache; only new ones spend the budget."""
    try:
        project, obj, dataset, end = _context_for(req)
        mgr, model = _covariate_model(req.model)
        cands = [c for c in dict.fromkeys(req.inputs or suggest_inputs(req.project_id, obj, req.target))
                 if c != req.target][:MAX_CANDIDATES]
        if not cands:
            raise HTTPException(status_code=400, detail="no candidate inputs: give `inputs` (or run a field scan first)")
        job.update(phase="loading data", model=model, dataset=dataset, end=end, candidates=cands)
        t, cols = await asyncio.to_thread(load_frame, project, obj, dataset, [req.target, *cands], req.bar, end)
        target = cols[req.target]
        anchors = _anchors(len(target), req.context, req.horizon, req.points)
        _note_streams(model, "exploration", dataset, t, anchors, req, cands, "candidate input", cols, end).publish(model)
        where = {"project_id": req.project_id, "objective_id": req.objective_id, "dataset": dataset,
                 "target": req.target, "horizon": req.horizon, "bar": req.bar, "model": model,
                 "context": req.context, "points": req.points, "end": end, "n_rows": len(target)}
        job.update(total=req.budget, done=0, cache_hits=0, runs=[])
        seen: dict[str, dict] = {}

        async def run(inputs: list[str], kind: str) -> dict | None:
            key = _key(inputs)
            if key in seen:
                return seen[key]
            sc = await asyncio.to_thread(load_combo, combo_key(where, inputs))
            if sc is not None:
                job["cache_hits"] += 1
            else:
                if job["done"] >= req.budget:
                    job["budget_exhausted"] = True
                    return None
                job["current"] = f"{kind}: {', '.join(inputs) or '(target only)'}"
                sc = await run_combo(mgr, model, target, {c: cols[c] for c in inputs}, anchors, req.context, req.horizon)
                await asyncio.to_thread(save_combo, where, inputs, sc, req.author or "exploration", kind)
                job["done"] += 1
            seen[key] = {"inputs": sorted(inputs), "kind": kind, **sc}
            job["runs"] = [_public(r) for r in seen.values()]
            return seen[key]

        found = await explore_search(run, cands, job)
        results = {"mode": "explore", "model": model, "dataset": dataset, "end": end, "target": req.target,
                   "horizon": req.horizon, "bar": req.bar, "anchors": len(anchors), "candidates": cands, **found,
                   "new_runs": job["done"], "cache_hits": job["cache_hits"], "budget": req.budget,
                   "budget_exhausted": bool(job.get("budget_exhausted")),
                   "runs": [_public(r) for r in seen.values()]}
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


@router.post("/explore")
async def explore(req: ExploreReq) -> dict:
    """Start a budgeted exploration of input combinations for one target (in-sample only). A
    running exploration of the same target is returned instead of starting a second one."""
    _context_for(req)
    _covariate_model(req.model)
    for j in _jobs.values():
        p = j.get("params") or {}
        if (j["phase"] not in ("done", "error") and p.get("mode") == "explore" and p.get("project_id") == req.project_id
                and p.get("target") == req.target and p.get("horizon") == req.horizon and p.get("bar") == req.bar):
            return {**{k: v for k, v in j.items() if k not in ("results", "runs")}, "already_running": True}
    jid = uuid.uuid4().hex[:10]
    params = {**req.model_dump(), "mode": "explore"}
    job = {"id": jid, "phase": "queued", "done": 0, "total": req.budget, "started_at": time.time(), "error": None,
           "params": params}
    _jobs[jid] = job
    conn = _db()
    with _lock():
        conn.execute("INSERT INTO tslab_runs (id, project_id, objective_id, created_at, status, params) VALUES (?,?,?,?,?,?)",
                     (jid, req.project_id, req.objective_id, time.time(), "running", json.dumps(params)))
        conn.commit()
    asyncio.create_task(_explore(job, req))
    return job


def combo_groups(project_id: str, end_max: str | None = None, target: str | None = None) -> list[dict]:
    """Every explored combination, grouped by what makes them comparable (same target, data,
    horizon, bar, model and points), each compared with that group's target-only baseline.

    `end_max` hides groups evaluated past it: a combination scored on data beyond an
    objective's split has seen that objective's holdout."""
    conn = _db()
    with _lock():
        rows = conn.execute("SELECT * FROM tslab_combos WHERE project_id=? ORDER BY ts", (project_id,)).fetchall()
    groups: dict[tuple, dict] = {}
    for r in rows:
        if end_max and (r["end_date"] is None or str(r["end_date"]) > str(end_max)):
            continue
        if target and r["target"] != target:
            continue
        g = (r["dataset"], r["target"], r["horizon"], r["bar"], r["model"], r["context"], r["points"], r["end_date"], r["n_rows"])
        grp = groups.setdefault(g, {"dataset": r["dataset"], "target": r["target"], "horizon": r["horizon"],
                                    "bar": r["bar"], "model": r["model"], "context": r["context"],
                                    "points": r["points"], "end": r["end_date"], "rows": [], "last_ts": 0.0})
        sc = json.loads(r["score"])
        err = json.loads(r["err"]) if r["err"] else None
        grp["rows"].append({"inputs": json.loads(r["inputs"]), "ts": r["ts"], "author": r["author"], "kind": r["kind"],
                            **sc, "_err": np.array(err, dtype=float) if err is not None else None})
        grp["last_ts"] = max(grp["last_ts"], r["ts"])
    out = []
    for grp in groups.values():
        base = next((x for x in grp["rows"] if not x["inputs"]), None)
        combos = []
        for x in grp["rows"]:
            vs = paired(base, x) if base is not None and x["inputs"] else {"gain": None, "se": None, "significant": False}
            combos.append({**_public(x), "gain": vs["gain"], "se": vs["se"], "significant": vs["significant"]})
        combos.sort(key=lambda c: -(c.get("skill") if c.get("skill") is not None else -9))
        solo = [c for c in combos if len(c["inputs"]) == 1]
        helpful = [c["inputs"][0] for c in solo if c["significant"] and (c["gain"] or 0) > 0]
        hurts = [c["inputs"][0] for c in solo if c["significant"] and (c["gain"] or 0) < 0]
        useless = [c["inputs"][0] for c in solo if c["inputs"][0] not in helpful]
        winners = [c for c in combos if c["inputs"] and c["significant"] and (c["gain"] or 0) > 0]
        best = max(winners, key=lambda c: c["skill"] if c["skill"] is not None else -9, default=None)
        out.append({k: v for k, v in grp.items() if k != "rows"} | {
            "baseline": _public(base) if base else None, "combos": combos, "tested": len(combos),
            "helpful": helpful, "hurts": hurts, "useless": useless, "best": best})
    out.sort(key=lambda g: -g["last_ts"])
    return out


def combo_brief(project_id: str, obj: dict | None, limit: int = 6) -> list[dict]:
    """What the explored combinations say, per target -- for the agents' brief and the mentor."""
    try:
        groups = combo_groups(project_id, (obj or {}).get("split_date"))
    except Exception:  # noqa: BLE001 -- a brief must never cost an agent its iteration
        return []
    out = []
    for g in groups[:limit]:
        b = g.get("best")
        out.append({"target": g["target"], "horizon": g["horizon"], "bar": g["bar"], "model": g["model"],
                    "tested": g["tested"], "baseline_skill": (g.get("baseline") or {}).get("skill"),
                    "best_inputs": b["inputs"] if b else [], "best_skill": b["skill"] if b else None,
                    "best_gain": b["gain"] if b else None, "best_se": b["se"] if b else None,
                    "helpful": g["helpful"][:10], "hurts": g["hurts"][:10],
                    "useless": [c for c in g["useless"] if c not in g["hurts"]][:25]})
    return out


@router.get("/combos")
async def combos(project_id: str, objective_id: str | None = None, target: str | None = None) -> dict:
    end_max = None
    if objective_id:
        from .objectives import get_objective

        end_max = get_objective(objective_id).get("split_date")
    groups = await asyncio.to_thread(combo_groups, project_id, end_max, target)
    running = [{k: v for k, v in j.items() if k not in ("results", "runs")} for j in _jobs.values()
               if (j.get("params") or {}).get("mode") == "explore" and (j.get("params") or {}).get("project_id") == project_id
               and j["phase"] not in ("done", "error")]
    return {"groups": groups, "running": running}


# =======================================================================================
# Forecast report: every forecast feature, what it read, how good it is, and did it help?
# =======================================================================================
def _feature_skill(f: dict) -> dict:
    """Per series, one shape for both kinds of feature: skill / direction / coverage, and for a
    feature with inputs the same forecast without them and the lift (paired, +/- 2 SE)."""
    out = {}
    for series, s in (f.get("skill") or {}).items():
        if "with_inputs" in s:
            w, wo = s.get("with_inputs") or {}, s.get("without_inputs") or {}
            out[series] = {"skill": w.get("skill"), "direction": w.get("direction"), "coverage": w.get("coverage"),
                           "points": w.get("points"),
                           "without_inputs": {k: wo.get(k) for k in ("skill", "direction", "coverage")} if wo else None,
                           "lift": s.get("lift"), "lift_skill": s.get("lift_skill"), "lift_direction": s.get("lift_direction")}
        else:
            out[series] = {"skill": s.get("skill_vs_no_change"), "direction": s.get("direction_accuracy"),
                           "coverage": s.get("band_coverage_10_90"), "points": s.get("anchors_scored")}
    return out


def forecast_report(obj: dict, recent: int = 1000) -> list[dict]:
    """Every forecast feature: its recipe (what was sent to the model), its in-sample skill,
    the lift from its inputs, who used it -- and whether using it HELPED.

    "Helped" is judged as fairly as the record allows. The strongest evidence is a candidate
    that improved a parent which did NOT use the feature: the same idea with and without it,
    so the score difference is mostly the feature's (basis "vs parent"). Only when there are
    fewer than 3 such pairs does it fall back to comparing the median in-sample score of the
    candidates that used it with that of candidates using no forecast at all ("vs median") --
    weaker, because those candidates differ in everything else too. In-sample scores only."""
    from .objectives import db as _odb, list_features

    feats = list_features(obj["id"])
    with _lock():
        rows = [dict(r) for r in _odb().execute(
            # The code is only needed for candidates recorded before features_used existed.
            "SELECT id, seq, parent_id, is_score, metrics, CASE WHEN metrics LIKE '%\"features_used\"%' THEN '' "
            "ELSE code END AS code FROM candidates WHERE objective_id=? AND status='ok' "
            "AND is_score IS NOT NULL ORDER BY seq DESC LIMIT ?", (obj["id"], recent)).fetchall()]
    views = [f["view"] for f in feats]
    by_id = {}
    for c in rows:
        m = json.loads(c["metrics"] or "{}")
        # Recorded by the harness since features_used existed; older candidates: the views
        # their code names.
        c["used"] = set(m["features_used"]) if "features_used" in m else {v for v in views if v in (c["code"] or "")}
        by_id[c["id"]] = c
    no_fc = [c["is_score"] for c in rows if not c["used"]]
    med_none = float(np.median(no_fc)) if no_fc else None
    out = []
    for f in feats:
        v = f["view"]
        users = [c for c in rows if v in c["used"]]
        pairs = []
        for c in users:
            p = by_id.get(c["parent_id"] or "")
            if p is not None and v not in p["used"]:
                pairs.append({"seq": c["seq"], "parent_seq": p["seq"], "delta": round(c["is_score"] - p["is_score"], 4)})
        verdict, basis, n, effect = "unused", "none", len(users), None
        if len(pairs) >= 3:
            d = np.array([x["delta"] for x in pairs])
            effect, share = float(np.median(d)), float(np.mean(d > 0))
            basis, n = "vs parent", len(pairs)
            verdict = "helped" if effect > 0 and share >= 0.6 else "hurt" if effect < 0 and share <= 0.4 else "unclear"
        elif users:
            basis = "vs median"
            if med_none is not None and len(users) >= 5 and len(no_fc) >= 5:
                effect = float(np.median([c["is_score"] for c in users])) - med_none
                verdict = "helped" if effect > 0 else "hurt" if effect < 0 else "unclear"
            else:
                verdict = "unclear"
        p = f.get("params") or {}
        recipe = f.get("recipe") or {}
        out.append({
            "view": v, "created_at": f.get("created_at"), "rows": f.get("rows"), "seconds": f.get("seconds"),
            "auto": v.startswith("fc_auto_"), "requested_by": recipe.get("requested_by"),
            "inputs_sent": {"model": p.get("model"), "dataset": p.get("dataset"), "target": p.get("series"),
                            "covariates": p.get("covariates") or [], "calendar": bool(p.get("calendar")),
                            "horizon": p.get("horizon"), "every": p.get("every"), "context": p.get("context"),
                            "bar": p.get("bar"), "samples": p.get("samples")},
            "request": recipe.get("request"), "columns": f.get("columns"),
            "skill": _feature_skill(f),
            "used_by": len(users), "used_by_seqs": sorted(c["seq"] for c in users)[-30:],
            "verdict": verdict, "basis": basis, "n": n,
            "effect": round(effect, 4) if effect is not None else None,
            "pairs": sorted(pairs, key=lambda x: -x["seq"])[:30],
            "median_users": round(float(np.median([c["is_score"] for c in users])), 4) if users else None,
            "median_no_forecast": round(med_none, 4) if med_none is not None else None,
        })
    out.sort(key=lambda r: (-r["used_by"], -(r["created_at"] or 0)))
    return out


@router.get("/forecast-report/{oid}")
async def forecast_report_route(oid: str) -> dict:
    from .objectives import get_objective

    obj = get_objective(oid)
    feats = await asyncio.to_thread(forecast_report, obj)
    return {"features": feats, "split_date": obj.get("split_date"),
            "note": ("in-sample only. verdict 'vs parent' = median score change of candidates that added this "
                     "forecast to a parent without it; 'vs median' = median score of its users minus that of "
                     "candidates using no forecast (weaker evidence).")}


@router.get("/forecast-view/{oid}/{view}")
async def forecast_view_route(oid: str, view: str, anchors: int = 5, rerun: bool = True) -> dict:
    """One stored forecast feature drawn at a few in-sample anchors: its inputs over the context
    and the forecast cone against what followed (app/forecast_view.py). Nothing at or after the
    split date is read."""
    from .forecast_view import feature_view
    from .objectives import get_objective

    return await feature_view(get_objective(oid), view, anchors, rerun)
