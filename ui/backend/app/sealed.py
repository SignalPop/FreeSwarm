"""The sealed test: the search's best candidates, unchanged, on dates nothing in the search has touched.

The holdout stops being unseen the moment thousands of candidates are ranked on it. Agents never read
it, but they build on what it ranked, so over a long search it is fitted anyway: on b27d52cb27, after
2,600 candidates, the leader scores 2.26 in-sample and 2.27 on the holdout. Going faster only spends it
faster, so something has to stay outside the search:

* **The sealed run.** A cohort of the objective's candidates -- the top, a spread down the board as a
  control, and the best of each other island -- replayed on rows past the last date the objective ever
  had: another data source of its task server, or for a positions objective a folder of the same
  dataset's rows. Forecasts they read are rebuilt from those rows by the same recipe. What comes back
  is a verdict on the board, not a new way to pick: if rank does not predict the sealed result, the
  board's order means little.
* **The deflated score** (Bailey & Lopez de Prado): the probability that the leader's Sharpe beats what
  the best of that many trials reaches by luck, given how long and how lumpy its record is.

Nothing here reaches an agent, the mentor included: sealed results are not in any brief or rank. Every
run is logged, because each look the operator acts on spends the seal too.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import shutil
import statistics
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any

import duckdb
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import datasource, projects
from . import objectives as O
from . import task_objectives as T
from .config import settings

logger = logging.getLogger("freetoken.sealed")

router = APIRouter(tags=["sealed"])

COHORT_TOP = 20
COHORT_SPREAD = 10
COHORT_ISLANDS = 8
# A replay is only evidence about the strategy if it trades the same way where both runs have data.
COMPARABLE_RHO = 0.5
HOLDS_SHARE = 0.5
MAX_REPLAY_TIMEOUT_S = 600
SOURCES_WAIT_S = 15.0

_live: dict[str, asyncio.Task] = {}


def _ensure() -> None:
    with O._lock:
        O.db().executescript(
            """
            CREATE TABLE IF NOT EXISTS sealed_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, objective_id TEXT NOT NULL, started_at REAL NOT NULL,
                finished_at REAL, state TEXT NOT NULL, phase TEXT NOT NULL DEFAULT '', source TEXT NOT NULL DEFAULT '',
                rows_path TEXT NOT NULL DEFAULT '', total INTEGER NOT NULL DEFAULT 0, results TEXT NOT NULL DEFAULT '[]',
                summary TEXT NOT NULL DEFAULT '{}', error TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS sealed_obj ON sealed_runs(objective_id, started_at);
            """
        )
        O.db().commit()


def _update(run_id: int, **fields: Any) -> None:
    sets = ", ".join(f"{k}=?" for k in fields)
    with O._lock:
        O.db().execute(f"UPDATE sealed_runs SET {sets} WHERE id=?", (*fields.values(), run_id))
        O.db().commit()


# =======================================================================================
# The deflated score
# =======================================================================================
def expected_max_sharpe(days: int, trials: float) -> float:
    """What the best of `trials` strategies with no edge reaches on average over `days`, per day. Their spread is the
    sampling noise of a Sharpe, 1 / sqrt(days); the spread of the team's actual Sharpes would mostly measure how badly
    its early candidates lost to costs (on b27d52cb27 it put the bar at an annual Sharpe of 29)."""
    if trials <= 1 or days <= 1:
        return 0.0
    gamma, inv = 0.5772156649, NormalDist().inv_cdf
    return ((1.0 - gamma) * inv(1.0 - 1.0 / trials) + gamma * inv(1.0 - 1.0 / (trials * math.e))) / math.sqrt(days)


def deflated_sharpe(returns: list[float], trials: float) -> float | None:
    """The probability that this series' true Sharpe beats the best of `trials` by luck. A short or lumpy record (fat
    tails, few active days) counts for less."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3 or not r.std() > 0:
        return None
    sr = float(r.mean() / r.std(ddof=1))
    z = (r - r.mean()) / r.std()
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        return None
    return NormalDist().cdf((sr - expected_max_sharpe(len(r), trials)) * math.sqrt(len(r) - 1) / math.sqrt(denom))


def _all_trials(oid: str) -> tuple[int, int]:
    """(scored candidates, trials their ft.search calls ran inside them)."""
    with O._lock:
        rows = O.db().execute("SELECT metrics FROM candidates WHERE objective_id=? AND status='ok' AND is_score IS NOT NULL",
                              (oid,)).fetchall()
    return len(rows), sum(int((json.loads(r[0] or "{}").get("search") or {}).get("trials") or 0) for r in rows)


def integrity(obj: dict) -> dict | None:
    """How much the leader's score means after this many tries. Operator-only."""
    from . import evolution

    if not (T.is_task(obj) or obj["metric"]["kind"] in O.RETURN_METRICS):
        return None
    best = O._best(obj)
    if not best:
        return None
    full = O.get_candidate(best["id"])
    series = [v for _, v in full.get("returns") or [] if isinstance(v, (int, float))]
    n, inner = _all_trials(obj["id"])
    if not series:
        return None
    trials = n + inner
    distinct = len(evolution.archive(obj).get("islands") or [])
    m = full.get("metrics") or {}
    b = evolution.behaviour(full) or {}
    return {"leader": best["seq"], "score": best.get("score"), "trials": trials, "search_trials": inner,
            "distinct_strategies": distinct, "days": len(series), "trades": b.get("trades"),
            "active_days": {"in_sample": (m.get("in_sample") or {}).get("active_days"),
                            "holdout": (m.get("holdout") or {}).get("active_days")},
            "luck_bar_sharpe": round(expected_max_sharpe(len(series), trials) * math.sqrt(252.0), 3),
            "luck_bar_distinct": round(expected_max_sharpe(len(series), distinct) * math.sqrt(252.0), 3),
            "deflated_all_trials": _round(deflated_sharpe(series, trials)),
            "deflated_distinct": _round(deflated_sharpe(series, max(distinct, 1)))}


def _round(v: float | None, n: int = 4) -> float | None:
    return None if v is None else round(v, n)


# =======================================================================================
# Which candidates
# =======================================================================================
def cohort(obj: dict, top: int = COHORT_TOP, spread: int = COHORT_SPREAD, islands: int = COHORT_ISLANDS) -> list[dict]:
    """The board's top, an even spread below it, and the best of other islands: the spread is the control. If rank means
    something the top beats it on sealed dates; the clones at the top alone could not show that."""
    from . import evolution

    higher = O._higher(obj)
    ranked = [c for c in O._distinct(O._ranked(obj["id"], higher)) if c.get("mode") != "ensemble"]
    for i, c in enumerate(ranked):
        c["rank"] = i + 1
    chosen = ranked[:top]
    rest = ranked[top:]
    if rest and spread:
        step = max(1, len(rest) // spread)
        chosen += rest[step // 2::step][:spread]
    have = {c["id"] for c in chosen}
    by_id = {c["id"]: c for c in ranked}
    extra = []
    for island in (evolution.archive(obj).get("islands") or []) if evolution._applicable(obj) else []:
        best = next((by_id[m["id"]] for m in sorted(island["members"], key=lambda m: by_id[m["id"]]["rank"] if m["id"] in by_id else 1 << 30)
                     if m["id"] in by_id), None)
        if best and best["id"] not in have:
            extra.append(best)
    extra.sort(key=lambda c: c["rank"])
    return chosen + extra[:islands]


def _feature_views(obj: dict, cands: list[dict]) -> set[str]:
    """Every forecast the cohort reads, members of ft.candidate_positions chains included."""
    views: set[str] = set()
    seen: set[int] = set()

    def walk(c: dict) -> None:
        views.update((c.get("metrics") or {}).get("features_used") or [])
        for seq in O.member_seqs(c.get("code") or ""):
            if seq in seen:
                continue
            seen.add(seq)
            with O._lock:
                r = O.db().execute("SELECT code, metrics FROM candidates WHERE objective_id=? AND seq=?",
                                   (obj["id"], seq)).fetchone()
            if r is not None:
                walk({"code": r["code"], "metrics": json.loads(r["metrics"] or "{}")})

    for c in cands:
        walk(c if c.get("code") else O.get_candidate(c["id"]))
    return views


# =======================================================================================
# Replays
# =======================================================================================
def _resolve_rows(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path(settings.repo_root) / p
    p = p.resolve()
    root = Path(settings.repo_root).resolve()
    if root != p and root not in p.parents:
        raise HTTPException(status_code=400, detail="the rows folder must be inside the repository")
    if not p.is_dir() or not any(p.glob("*.parquet")):
        raise HTTPException(status_code=400, detail=f"no parquet files in {path}")
    return p


def _schema(glob: str) -> dict[str, str]:
    con = duckdb.connect(":memory:")
    try:
        return {r[0]: r[1] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{glob}')").fetchall()}
    finally:
        con.close()


def prepare_rows(obj: dict, rows_dir: Path) -> dict:
    """A data folder holding `rows_dir` under the objective's dataset's own path, so a script's ft.load finds its dataset
    unchanged. Returns the objective as the replay sees it. The files are hard links where the disk allows."""
    project = projects.get(obj["project_id"]) or {}
    data_dir = project.get("data_dir") or ""
    item = next((c for c in datasource.catalog(data_dir) if c["view"] == obj.get("dataset")), None)
    if item is None or not item["path"].endswith("/*.parquet"):
        raise HTTPException(status_code=409, detail=f"the objective's dataset {obj.get('dataset')!r} is not a parquet folder")
    own = _schema(O._abs(data_dir, item).replace("\\", "/"))
    theirs = _schema((rows_dir / "*.parquet").as_posix())
    missing = sorted(set(own) - set(theirs))
    if missing:
        raise HTTPException(status_code=409, detail=f"those rows lack {len(missing)} of the dataset's columns: "
                                                    f"{', '.join(missing[:12])}")
    tag = hashlib.sha1(str(rows_dir).lower().encode()).hexdigest()[:10]
    home = f"{obj['id']}/sealed/{tag}"
    root = O.WORK_ROOT / home
    rel = item["path"][: -len("/*.parquet")]
    folder = root / "data" / rel
    folder.mkdir(parents=True, exist_ok=True)
    for src in rows_dir.glob("*.parquet"):
        dst = folder / src.name
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            continue
        dst.unlink(missing_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
    (folder / ".export.json").write_text(json.dumps({"source": str(rows_dir)}), encoding="utf-8")
    return {**obj, "_replay": {"data_dir": str(root / "data"), "features_id": home}}


async def rebuild_forecasts(obj: dict, replay: dict, views: set[str], on_phase) -> list[str]:
    """Rebuild the forecasts the cohort reads from the sealed rows, by each one's own recipe, and say which could not be."""
    wanted = {f["view"]: f for f in O.list_features(obj["id"]) if f["view"] in views}
    failed = sorted(views - set(wanted))
    for i, (view, meta) in enumerate(sorted(wanted.items())):
        on_phase(f"rebuilding forecast {i + 1} of {len(wanted)}: {view}")
        p = meta.get("params") or {}
        stem = str(meta.get("file") or "").rsplit(".", 1)[0]
        # What actually ran, not what was asked for: a request may say every=0 or no model, and the
        # sealed rows would then choose another grid or whichever forecaster happens to be loaded.
        asked = {k: v for k, v in ((meta.get("recipe") or {}).get("request") or {}).items() if v is not None}
        ran = {k: p[k] for k in ("every", "context", "model", "horizon", "dataset") if p.get(k) is not None}
        request = {**({} if asked else {"columns": p.get("series")}), **asked, **ran}
        try:
            await O.build_feature(replay, O.FeatureReq(**{**request, "name": stem}), requested_by="sealed test")
        except HTTPException as exc:
            logger.warning("forecast %s not rebuilt: %s", view, exc.detail)
            failed.append(view)
        except Exception:  # noqa: BLE001 -- the candidates that read it are the ones to report
            logger.exception("forecast %s not rebuilt", view)
            failed.append(view)
    return failed


async def _last_date(obj: dict) -> str | None:
    """The last date the objective's own data has: what "never seen" starts after, for every candidate alike."""
    if T.is_task(obj):
        glob, col = (await T.export_dir(obj)).joinpath("rows.parquet").as_posix(), "t"
    else:
        project = projects.get(obj["project_id"]) or {}
        data_dir = project.get("data_dir") or ""
        item = next((c for c in datasource.catalog(data_dir) if c["view"] == obj.get("dataset")), None)
        if item is None or not obj.get("time_column"):
            return None
        glob, col = O._abs(data_dir, item).replace("\\", "/"), obj["time_column"]

    def read() -> str | None:
        con = duckdb.connect(":memory:")
        try:
            v = con.execute(f"SELECT max(TRY_CAST(\"{col}\" AS TIMESTAMP)) FROM read_parquet('{glob}')").fetchone()[0]
        finally:
            con.close()
        return str(v)[:10] if v else None

    return await asyncio.to_thread(read)


async def _replay_positions(obj: dict, replay: dict, c: dict, catalog: list[dict], timeout: int, last: str | None) -> dict:
    data_dir = replay["_replay"]["data_dir"]
    replay = {**replay, "_library_as_of": c.get("created_at")}
    async with O._EVAL_SLOTS:
        rep = await O._run_forecasting(c["code"], data_dir, catalog, None, timeout, replay, None,
                                       requested_by=f"sealed test of #{c['seq']}")
    if not rep["ok"]:
        why = f"timed out after {timeout}s" if rep.get("timed_out") else O._failure_note(rep["stderr"], c["code"])
        raise RuntimeError(why)
    pos = O._positions_file(rep)
    if pos is None:
        raise RuntimeError("no positions reported")
    returns, mtm = await asyncio.to_thread(O._mark_to_market, replay, data_dir, pos)
    problem = O._positions_off_data(mtm)
    if problem:
        raise RuntimeError(problem)
    return {"curve": returns, "comparison": O._source_comparison(obj, c.get("returns") or [], returns, False, last)}


async def _replay_task(obj: dict, c: dict, source: str, last: str | None) -> dict:
    job = {"state": "running", "phase": "starting"}
    await O._do_source_run(obj["id"], c["id"], source, job)
    if job.get("state") != "done":
        raise RuntimeError(job.get("error") or "the replay failed")
    saved = O._source_run_saved(obj["id"], c["id"], source) or {}
    curve = saved.get("curve") or []
    additive = ((saved.get("metrics") or {}).get("task") or {}).get("curve_kind") == "additive"
    return {"curve": curve, "comparison": O._source_comparison(obj, c.get("returns") or [], curve, additive, last)}


def _row(c: dict, out: dict) -> dict:
    """One candidate's sealed result, from a replay's comparison with its scored run."""
    periods = {p["name"]: p for p in (out.get("comparison") or {}).get("periods") or []}
    sealed = (periods.get("never seen") or {}).get("replay") or {}
    series = [v for d, v in out.get("curve") or [] if sealed.get("first") and str(d)[:10] >= sealed["first"]]
    m = c.get("metrics") or {}

    def sharpe(period: str, side: str) -> float | None:
        return ((periods.get(period) or {}).get(side) or {}).get("sharpe")

    return {"seq": c["seq"], "id": c["id"], "rank": c.get("rank"), "score": c.get("score"), "in_sample": c.get("is_score"),
            "holdout": (m.get("rank") or {}).get("holdout"), "days": sealed.get("days"), "active_days": sealed.get("active_days"),
            "sharpe": sealed.get("sharpe"), "total_return": sealed.get("total_return"), "max_drawdown": sealed.get("max_drawdown"),
            "win_rate": sealed.get("win_rate"), "smoothness": _round(O._smoothness(series), 3) if series else None,
            "overlap_rho": (out.get("comparison") or {}).get("daily_correlation"),
            # The same dates on both sets of rows: what it earned where it was scored, and on the other rows.
            "same_dates": {"built_on": {"scored": sharpe("built on", "scored"), "replay": sharpe("built on", "replay")},
                           "holdout": {"scored": sharpe("objective holdout", "scored"), "replay": sharpe("objective holdout", "replay")}},
            "first": sealed.get("first"), "last": sealed.get("last"), "status": "ok"}


def _ranks(xs: list[float]) -> np.ndarray:
    return np.argsort(np.argsort(np.asarray(xs))).astype(float)


def summarise(rows: list[dict], top: int = COHORT_TOP) -> dict:
    """What the sealed dates say about the board. `verdict` is one of holds, weakens, fails, not_comparable, no_dates."""
    ok = [r for r in rows if r.get("status") == "ok" and r.get("sharpe") is not None]
    out: dict[str, Any] = {"replayed": len(ok), "failed": sum(1 for r in rows if r.get("status") != "ok")}
    if not ok:
        out["verdict"] = "no_dates"
        out["reading"] = "No candidate has results past the objective's last date: these rows hold nothing it has not had."
        return out
    head = [r for r in ok if (r.get("rank") or 10 ** 9) <= top]
    tail = [r for r in ok if (r.get("rank") or 10 ** 9) > top]
    rhos = [r["overlap_rho"] for r in ok if r.get("overlap_rho") is not None]
    med = lambda xs: round(statistics.median(xs), 3) if xs else None  # noqa: E731
    out.update(first=ok[0].get("first"), last=ok[0].get("last"), days=ok[0].get("days"),
               median_sharpe_top=med([r["sharpe"] for r in head]), median_sharpe_rest=med([r["sharpe"] for r in tail]),
               median_score_top=med([r["score"] for r in head if r.get("score") is not None]),
               median_active_days_top=med([r["active_days"] for r in head if r.get("active_days") is not None]),
               positive_share=round(sum(1 for r in ok if r["sharpe"] > 0) / len(ok), 3),
               median_overlap_rho=med(rhos))
    for period in ("built_on", "holdout"):
        for side in ("scored", "replay"):
            vals = [x for x in (((r.get("same_dates") or {}).get(period) or {}).get(side) for r in head) if x is not None]
            out[f"same_dates_{period}_{side}"] = med(vals)
    scored = [r for r in ok if r.get("score") is not None]
    if len(scored) >= 5:
        a, b = _ranks([r["score"] for r in scored]), _ranks([r["sharpe"] for r in scored])
        if a.std() > 0 and b.std() > 0:
            out["rank_rho"] = round(float(np.corrcoef(a, b)[0, 1]), 3)
    top_sr, top_score = out["median_sharpe_top"], out["median_score_top"]
    if out["median_overlap_rho"] is not None and out["median_overlap_rho"] < COMPARABLE_RHO:
        out["verdict"] = "not_comparable"
        scored, replay = out.get("same_dates_built_on_scored"), out.get("same_dates_built_on_replay")
        out["reading"] = (f"Where both runs have data their days correlate only {out['median_overlap_rho']}: the strategies "
                          "trade these rows differently, so the sealed dates say little about them as scored."
                          + (f" On the very dates they were built on, the top's median Sharpe is {scored} on its own rows and "
                             f"{replay} on these." if scored is not None and replay is not None else ""))
    elif top_sr is None or top_sr <= 0:
        out["verdict"] = "fails"
        out["reading"] = (f"The top of the board made nothing on dates it never saw (median Sharpe {top_sr}): "
                          "its score was fitted to the data it was searched on.")
    elif top_score and top_sr >= HOLDS_SHARE * top_score:
        out["verdict"] = "holds"
        out["reading"] = f"The top of the board keeps most of its score on sealed dates (median Sharpe {top_sr} against {top_score})."
    else:
        out["verdict"] = "weakens"
        out["reading"] = f"The top of the board still earns on sealed dates, well under its score (median Sharpe {top_sr} against {top_score})."
    return out


async def _job(run_id: int, oid: str, cands: list[dict], source: str, rows_path: str, top: int = COHORT_TOP) -> None:
    results: list[dict] = []
    try:
        obj = O.get_objective(oid)
        catalog, replay, missing = [], None, []
        timeout = min(MAX_REPLAY_TIMEOUT_S, int(obj["eval_timeout_s"]) * 3)
        if not T.is_task(obj):
            _update(run_id, phase="linking the rows")
            replay = await asyncio.to_thread(prepare_rows, obj, _resolve_rows(rows_path))
            catalog = await asyncio.to_thread(datasource.catalog, replay["_replay"]["data_dir"])
            views = await asyncio.to_thread(_feature_views, obj, cands)
            missing = await rebuild_forecasts(obj, replay, views, lambda p: _update(run_id, phase=p))
        last = await _last_date(obj)
        for i, light in enumerate(cands):
            _update(run_id, phase=f"replaying #{light['seq']} ({i + 1} of {len(cands)})")
            c = {**O.get_candidate(light["id"]), "rank": light.get("rank")}
            try:
                out = await (_replay_task(obj, c, source, last) if T.is_task(obj)
                             else _replay_positions(obj, replay, c, catalog, timeout, last))
                results.append(_row(c, out))
            except HTTPException as exc:
                results.append({"seq": c["seq"], "id": c["id"], "rank": c.get("rank"), "status": "error", "error": str(exc.detail)[:600]})
            except Exception as exc:  # noqa: BLE001 -- one candidate's failure is its row, not the run's
                results.append({"seq": c["seq"], "id": c["id"], "rank": c.get("rank"), "status": "error",
                                "error": f"{exc}"[:600]})
            _update(run_id, results=json.dumps(results))
        _update(run_id, state="done", phase="", finished_at=time.time(),
                summary=json.dumps({**summarise(results, top), "missing_forecasts": missing}))
    except HTTPException as exc:
        _update(run_id, state="failed", phase="", finished_at=time.time(), error=str(exc.detail)[:2000])
    except Exception as exc:  # noqa: BLE001 -- reported on the run
        logger.exception("sealed run %s failed", run_id)
        _update(run_id, state="failed", phase="", finished_at=time.time(), error=f"{type(exc).__name__}: {exc}"[:2000])
    finally:
        _live.pop(oid, None)


def runs(oid: str) -> list[dict]:
    _ensure()
    with O._lock:
        rows = [dict(r) for r in O.db().execute("SELECT * FROM sealed_runs WHERE objective_id=? ORDER BY started_at DESC",
                                                (oid,)).fetchall()]
    for r in rows:
        r["results"], r["summary"] = json.loads(r["results"] or "[]"), json.loads(r["summary"] or "{}")
        if r["state"] == "running" and oid not in _live:
            r["state"], r["error"] = "interrupted", "the control plane restarted during this run"
    return rows


async def _rows_offered(obj: dict) -> list[str]:
    """Folders in the repository holding rows under the same folder name as the objective's dataset."""
    project = projects.get(obj["project_id"]) or {}
    item = next((c for c in await asyncio.to_thread(datasource.catalog, project.get("data_dir") or "")
                 if c["view"] == obj.get("dataset")), None)
    if item is None or "/" not in item["path"]:
        return []
    name = item["path"].split("/")[-2]
    root = Path(settings.repo_root)
    found = [p for p in (root / "mcp").glob(f"*/data/*/{name}") if any(p.glob("*.parquet"))]
    return sorted(p.relative_to(root).as_posix() for p in found if "backup" not in p.as_posix())


@router.get("/objectives/{oid}/sealed")
async def sealed_view(oid: str) -> dict:
    """The operator's view of the seal: past runs and how much the leader's score means. Agents have no route to this."""
    obj = O.get_objective(oid)
    sources: list[dict] = []
    own = None
    if T.is_task(obj):
        try:
            # An older objective's snapshot does not list them: asking the server can take minutes.
            sources, own = await asyncio.wait_for(O._task_sources(obj), SOURCES_WAIT_S)
        except (HTTPException, asyncio.TimeoutError):
            pass
    return {"runs": await asyncio.to_thread(runs, oid), "integrity": await asyncio.to_thread(integrity, obj),
            "task": T.is_task(obj), "own_source": own, "sources": [s["name"] for s in sources if s["name"] != own],
            "rows_offered": [] if T.is_task(obj) else await _rows_offered(obj)}


class SealedRun(BaseModel):
    source: str = Field("", max_length=200)
    rows_path: str = Field("", max_length=1000)
    top: int = Field(COHORT_TOP, ge=1, le=100)
    spread: int = Field(COHORT_SPREAD, ge=0, le=100)


@router.post("/objectives/{oid}/sealed")
async def start_sealed(oid: str, req: SealedRun) -> dict:
    """Open the seal: replay a cohort on the sealed rows (a background job; poll GET .../sealed). Every run is logged."""
    obj = O.get_objective(oid)
    _ensure()
    if not O._higher(obj):
        raise HTTPException(status_code=409, detail="the sealed test compares Sharpe ratios: this objective's score is an error measure")
    if oid in _live:
        raise HTTPException(status_code=409, detail="a sealed run of this objective is already going")
    if T.is_task(obj):
        sources, own = await O._task_sources(obj)
        names = [s["name"] for s in sources if s["name"] != own]
        if req.source not in names:
            raise HTTPException(status_code=400, detail=f"source must be another of the task's sources: {names}")
    else:
        if obj["metric"]["kind"] not in O.RETURN_METRICS or not obj["metric"].get("price_column"):
            raise HTTPException(status_code=409, detail="only an objective with positions or a task server can be replayed")
        _resolve_rows(req.rows_path)
    cands = await asyncio.to_thread(cohort, obj, req.top, req.spread)
    if not cands:
        raise HTTPException(status_code=409, detail="no ranked candidate to replay")
    with O._lock:
        cur = O.db().execute("INSERT INTO sealed_runs (objective_id, started_at, state, phase, source, rows_path, total) "
                             "VALUES (?,?,?,?,?,?,?)", (oid, time.time(), "running", "starting", req.source, req.rows_path,
                                                        len(cands)))
        O.db().commit()
    run_id = int(cur.lastrowid)
    _live[oid] = asyncio.create_task(_job(run_id, oid, cands, req.source, req.rows_path, req.top))
    return {"id": run_id, "state": "running", "total": len(cands)}
