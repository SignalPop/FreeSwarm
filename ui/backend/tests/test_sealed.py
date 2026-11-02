"""The sealed test: what the board's candidates do on dates the search never touched, and the deflated score."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pytest

from app import evolution as E
from app import external, library
from app import objectives as O
from app import sealed as S


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "DB_PATH", tmp_path / "objectives.sqlite3")
    monkeypatch.setattr(O, "_conn", None)
    monkeypatch.setattr(O, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(external, "CONFIG_PATH", tmp_path / "external.json")
    monkeypatch.setattr(library, "_ready", False)
    E._cache.clear()
    now = time.time()
    O.db().execute("INSERT INTO objectives (id, project_id, title, metric, split_date, created_at, updated_at, require_audit, best_id, dataset) "
                   "VALUES ('o1', 'p1', 't', ?, '2024-03-01', ?, ?, 1, 'c1', 'px')",
                   (json.dumps({"kind": "sharpe", "price_column": "Close"}), now, now))
    O.db().commit()
    yield O.db()
    O.db().close()


def _cand(seq, score, returns=None, metrics=None, code="x = 1", mode="improve"):
    rng = np.random.default_rng(seq)
    series = returns if returns is not None else [[f"2024-01-{d:02d}", float(v)] for d, v in zip(range(1, 29), rng.normal(0.001, 0.01, 28))]
    O.db().execute(
        "INSERT INTO candidates (id, objective_id, seq, created_at, model, mode, rationale, code, status, score, is_score, metrics, returns, "
        "lookahead, audit) VALUES (?, 'o1', ?, ?, 'M', ?, 'why', ?, 'ok', ?, ?, ?, ?, 'pass', 'none')",
        (f"c{seq}", seq, time.time(), mode, code, score, score,
         json.dumps(metrics or {"in_sample": {"days": 28, "active_days": 20}}), json.dumps(series)))
    O.db().commit()


# ---------------------------------------------------------------------------------------
# The deflated score
# ---------------------------------------------------------------------------------------
def test_the_luck_bar_rises_with_trials_and_falls_with_days():
    assert S.expected_max_sharpe(320, 1) == 0
    assert S.expected_max_sharpe(320, 1000) > S.expected_max_sharpe(320, 10) > 0
    assert S.expected_max_sharpe(1000, 100) < S.expected_max_sharpe(100, 100)
    assert S.expected_max_sharpe(320, 1000) * 252 ** 0.5 == pytest.approx(2.9, abs=0.3)


def test_more_trials_mean_less_belief():
    r = np.random.default_rng(1).normal(0.0015, 0.01, 320).tolist()
    assert S.deflated_sharpe(r, 1) > S.deflated_sharpe(r, 50) > S.deflated_sharpe(r, 5000)


def test_a_lumpy_record_counts_for_less_than_a_steady_one():
    steady = np.random.default_rng(2).normal(0.001, 0.005, 320)
    lumpy = np.zeros(320)
    lumpy[::16] = steady[::16] * 16
    lumpy *= steady.mean() / lumpy.mean()
    lumpy = lumpy / lumpy.std() * steady.std()
    assert S.deflated_sharpe(lumpy.tolist(), 100) < S.deflated_sharpe(steady.tolist(), 100)


def test_a_flat_or_short_record_has_no_deflated_score():
    assert S.deflated_sharpe([0.0] * 50, 10) is None and S.deflated_sharpe([0.1, 0.2], 10) is None


def test_integrity_counts_the_trials_inside_searches(db):
    _cand(1, 2.0, metrics={"in_sample": {"days": 28, "active_days": 20}, "search": {"trials": 24}})
    _cand(2, 1.0)
    got = S.integrity(O.get_objective("o1"))
    assert got["leader"] == 1 and got["trials"] == 26 and got["search_trials"] == 24
    assert got["luck_bar_sharpe"] > got["luck_bar_distinct"] > 0


# ---------------------------------------------------------------------------------------
# The cohort and the summary
# ---------------------------------------------------------------------------------------
def test_the_cohort_is_the_top_a_spread_as_control_and_no_ensembles(db):
    for i in range(1, 61):
        _cand(i, 5.0 - i / 100)
    _cand(100, 9.0, mode="ensemble")
    got = S.cohort(O.get_objective("o1"), top=5, spread=4, islands=0)
    ranks = [c["rank"] for c in got]
    assert ranks[:5] == [1, 2, 3, 4, 5] and len(got) == 9 and all(c["seq"] != 100 for c in got)
    assert ranks[5] > 5 and ranks[-1] > 40


def _row(rank, score, sharpe, rho=0.9):
    return {"rank": rank, "score": score, "sharpe": sharpe, "overlap_rho": rho, "status": "ok", "days": 300, "first": "2025-01-02",
            "last": "2026-06-10", "active_days": 60}


def test_a_board_that_holds_up():
    got = S.summarise([_row(i, 2.0 - i / 100, 1.6 - i / 100) for i in range(1, 21)] + [_row(30, 1.0, 0.4)], top=20)
    assert got["verdict"] == "holds" and got["rank_rho"] > 0.9 and got["positive_share"] == 1.0


def test_a_board_fitted_to_its_data_fails_the_seal():
    got = S.summarise([_row(i, 2.0, -0.5) for i in range(1, 21)], top=20)
    assert got["verdict"] == "fails" and "fitted" in got["reading"]


def test_a_weaker_edge_on_sealed_dates_says_so():
    assert S.summarise([_row(i, 2.0, 0.5) for i in range(1, 21)], top=20)["verdict"] == "weakens"


def test_rows_the_strategies_trade_differently_prove_nothing():
    assert S.summarise([_row(i, 2.0, 1.9, rho=0.1) for i in range(1, 21)], top=20)["verdict"] == "not_comparable"


def test_the_same_dates_on_other_rows_are_reported():
    """A strategy that loses its edge on the very dates it was built on, in another feed, is fitted to the feed."""
    rows = [{**_row(i, 2.0, -0.4, rho=-0.07), "same_dates": {"built_on": {"scored": 2.3, "replay": 0.1},
                                                             "holdout": {"scored": 2.2, "replay": -0.2}}} for i in range(1, 21)]
    got = S.summarise(rows, top=20)
    assert got["verdict"] == "not_comparable" and "2.3" in got["reading"] and "0.1" in got["reading"]
    assert got["same_dates_holdout_replay"] == -0.2


def test_rows_with_no_new_dates_are_said_to_be_that():
    got = S.summarise([{"rank": 1, "status": "ok", "sharpe": None}, {"rank": 2, "status": "error"}])
    assert got["verdict"] == "no_dates" and got["failed"] == 1


# ---------------------------------------------------------------------------------------
# Replays
# ---------------------------------------------------------------------------------------
def _rows_folder(path: Path, days: tuple[str, ...], cols: str = "t TIMESTAMP, Close DOUBLE") -> None:
    path.mkdir(parents=True, exist_ok=True)
    names = [c.split()[0] for c in cols.split(",")]
    con = duckdb.connect(":memory:")
    con.execute(f"CREATE TABLE r ({cols})")
    for i, d in enumerate(days):
        con.execute("INSERT INTO r VALUES (?, ?" + ", 1.0" * (len(names) - 2) + ")", (f"{d} 15:00:00", 100.0 + i))
    con.execute(f"COPY r TO '{(path / 'part-0.parquet').as_posix()}' (FORMAT parquet)")
    con.close()


def test_the_rows_land_where_the_dataset_is_and_columns_are_checked(db, tmp_path, monkeypatch):
    from app import projects

    data = tmp_path / "data"
    _rows_folder(data / "exports" / "px", ("2024-01-02",))
    (data / "exports" / "px" / ".export.json").write_text("{}")
    monkeypatch.setattr(projects, "get", lambda pid: {"id": pid, "data_dir": str(data)})
    obj = {**O.get_objective("o1"), "dataset": "exports_px"}
    rows = tmp_path / "sealed_rows"
    _rows_folder(rows, ("2024-01-02", "2025-06-02"))
    replay = S.prepare_rows(obj, rows)
    home = Path(replay["_replay"]["data_dir"])
    from app import datasource

    assert [c["view"] for c in datasource.catalog(str(home))] == ["exports_px"]
    assert (home / "exports" / "px" / "part-0.parquet").is_file()
    assert O._feature_home(replay).startswith("o1/sealed/") and O._feature_home(obj) == "o1"
    thin = tmp_path / "thin_rows"
    _rows_folder(thin, ("2025-06-02",), "t TIMESTAMP, Other DOUBLE")
    with pytest.raises(S.HTTPException, match="lack"):
        S.prepare_rows(obj, thin)


def test_forecasts_are_rebuilt_by_their_own_recipe_and_gaps_reported(db, monkeypatch):
    built = []

    async def fake_build(replay, req, requested_by=None):
        built.append((replay["_replay"]["features_id"], req.name, req.every, req.columns))
        return {}

    monkeypatch.setattr(O, "build_feature", fake_build)
    monkeypatch.setattr(O, "list_features", lambda oid: [
        {"view": "fc_a", "file": "a.parquet", "params": {"series": ["X"], "dataset": "px", "horizon": 3, "every": 12, "context": 64, "model": "m"}}])
    obj = O.get_objective("o1")
    phases = []
    missing = asyncio.run(S.rebuild_forecasts(obj, {**obj, "_replay": {"features_id": "h"}}, {"fc_a", "fc_gone"}, phases.append))
    assert built == [("h", "a", 12, ["X"])] and missing == ["fc_gone"] and "fc_a" in phases[0]


def test_a_forecast_is_rebuilt_as_it_ran_not_as_it_was_asked_for(db, monkeypatch):
    """every=0 or no model in the request would let the sealed rows pick another grid, or any loaded forecaster."""
    built = []

    async def fake_build(replay, req, requested_by=None):
        built.append((req.every, req.model, req.context, req.columns))
        return {}

    monkeypatch.setattr(O, "build_feature", fake_build)
    monkeypatch.setattr(O, "list_features", lambda oid: [
        {"view": "fc_a", "file": "a.parquet", "params": {"series": ["X"], "dataset": "px", "horizon": 3, "every": 35, "context": 512, "model": "chronos"},
         "recipe": {"request": {"columns": ["X"], "every": 0, "model": None, "context": 256, "horizon": 3}}}])
    obj = O.get_objective("o1")
    asyncio.run(S.rebuild_forecasts(obj, {**obj, "_replay": {"features_id": "h"}}, {"fc_a"}, lambda p: None))
    assert built == [(35, "chronos", 512, ["X"])]


def test_a_replay_runs_the_library_as_it_stood(db):
    now = time.time()
    conn = library._db()
    conn.execute("INSERT INTO lib_modules (project_id, name, description, version, kind, author, status, created_at, updated_at) "
                 "VALUES ('p1', 'sig', 'd', 2, 'signal', 'M', 'retired', ?, ?)", (now, now))
    for version, ts, code in ((1, now - 100, "OLD = 1"), (2, now - 10, "NEW = 2")):
        conn.execute("INSERT INTO lib_versions (project_id, name, version, code, author, ts, test_ok) VALUES ('p1', 'sig', ?, ?, 'M', ?, 1)",
                     (version, code, ts))
    conn.commit()
    assert library.files_as_of("p1", now - 50)[".ft/lib/sig.py"] == "OLD = 1"
    assert library.files_as_of("p1", now)[".ft/lib/sig.py"] == "NEW = 2"
    assert library.files_as_of("p1", now - 500) == {}
    assert ".ft/lib/sig.py" not in library.module_files("p1")        # retired: only a replay still sees it


def test_the_sealed_window_is_the_objectives_not_one_candidates(db):
    scored = [["2024-01-02", 0.01], ["2024-06-03", 0.01]]         # an older candidate, ending early
    replay = [["2024-06-03", 0.01], ["2024-10-01", 0.01], ["2025-01-06", 0.02], ["2025-01-07", 0.01]]
    got = O._source_comparison(O.get_objective("o1"), scored, replay, False, "2024-12-17")
    never = next(p for p in got["periods"] if p["name"] == "never seen")
    assert never["replay"]["first"] == "2025-01-06" and never["replay"]["days"] == 2


def test_the_summary_uses_the_cohort_size_asked_for():
    rows = [_row(i, 2.0, 1.5 if i <= 5 else -1.0) for i in range(1, 21)]
    assert S.summarise(rows, top=5)["median_sharpe_top"] == 1.5
    assert S.summarise(rows, top=20)["median_sharpe_top"] == -1.0


def test_the_seal_is_refused_for_an_error_measure(db):
    O.db().execute("UPDATE objectives SET metric=? WHERE id='o1'", (json.dumps({"kind": "task", "higher_is_better": False}),))
    O.db().commit()
    with pytest.raises(S.HTTPException, match="error measure"):
        asyncio.run(S.start_sealed("o1", S.SealedRun()))


def test_the_forecasts_a_chain_reads_are_all_found(db):
    _cand(1, 2.0, metrics={"features_used": ["fc_a"]})
    _cand(2, 2.0, code="ft.candidate_positions(1)", metrics={"features_used": ["fc_b"]})
    assert S._feature_views(O.get_objective("o1"), [O.get_candidate("c2")]) == {"fc_a", "fc_b"}


def test_a_candidate_row_reads_the_sealed_period_of_its_comparison():
    c = {"seq": 7, "id": "c7", "rank": 1, "score": 2.0, "is_score": 2.1, "metrics": {"rank": {"holdout": 2.2}}}
    never = {"first": "2025-01-02", "last": "2026-01-02", "days": 3, "active_days": 3, "sharpe": 1.1, "total_return": 0.1, "max_drawdown": -0.1, "win_rate": 0.6}
    out = {"curve": [["2024-12-01", 0.5], ["2025-01-02", 0.01], ["2025-01-03", 0.01], ["2026-01-02", 0.01]],
           "comparison": {"daily_correlation": 0.8, "periods": [{"name": "never seen", "replay": never}]}}
    out["comparison"]["periods"].append({"name": "built on", "scored": {"sharpe": 2.1}, "replay": {"sharpe": 0.3}})
    row = S._row(c, out)
    assert row["sharpe"] == 1.1 and row["overlap_rho"] == 0.8 and row["holdout"] == 2.2 and row["status"] == "ok"
    assert row["same_dates"]["built_on"] == {"scored": 2.1, "replay": 0.3}
    assert row["smoothness"] == pytest.approx(1.0, abs=0.01)


def test_the_rows_folder_must_be_inside_the_repository(tmp_path):
    with pytest.raises(S.HTTPException, match="inside the repository"):
        S._resolve_rows(str(tmp_path))


def test_a_run_is_logged_replays_every_candidate_and_keeps_going_past_a_failure(db, monkeypatch):
    for i in range(1, 4):
        _cand(i, 3.0 - i / 10)
    obj = O.get_objective("o1")

    async def fake_last(o):
        return "2024-12-17"

    async def fake_task(o, c, source, last):
        assert last == "2024-12-17"                          # one sealed window for the whole cohort
        if c["seq"] == 2:
            raise RuntimeError("no such column")
        never = {"first": "2025-01-02", "last": "2026-01-02", "days": 200, "active_days": 40, "sharpe": 1.5}
        return {"curve": [], "comparison": {"daily_correlation": 0.9, "periods": [{"name": "never seen", "replay": never}]}}

    monkeypatch.setattr(S, "_replay_task", fake_task)
    monkeypatch.setattr(S, "_last_date", fake_last)
    monkeypatch.setattr(S.T, "is_task", lambda o: True)
    S._ensure()
    cands = S.cohort(obj)
    cur = O.db().execute("INSERT INTO sealed_runs (objective_id, started_at, state, total) VALUES ('o1', ?, 'running', ?)", (time.time(), len(cands)))
    O.db().commit()
    asyncio.run(S._job(cur.lastrowid, "o1", cands, "2", ""))
    run = S.runs("o1")[0]
    assert run["state"] == "done" and len(run["results"]) == 3
    assert [r["status"] for r in run["results"]] == ["ok", "error", "ok"]
    assert run["summary"]["verdict"] == "holds" and run["summary"]["failed"] == 1


def test_a_run_the_control_plane_lost_says_so(db):
    S._ensure()
    O.db().execute("INSERT INTO sealed_runs (objective_id, started_at, state) VALUES ('o1', ?, 'running')", (time.time(),))
    O.db().commit()
    assert S.runs("o1")[0]["state"] == "interrupted"


def test_nothing_sealed_reaches_an_agent(db, monkeypatch):
    """The whole point: no context, brief or agent view may carry a sealed result."""
    from app import projects

    monkeypatch.setattr(projects, "get", lambda pid: {"id": pid, "data_dir": ""})
    _cand(1, 2.0)
    S._ensure()
    O.db().execute("INSERT INTO sealed_runs (objective_id, started_at, state, summary) VALUES ('o1', ?, 'done', ?)",
                   (time.time(), json.dumps({"verdict": "SEALED-MARKER"})))
    O.db().commit()
    ctx = asyncio.run(O.context("o1", "M"))
    assert "SEALED-MARKER" not in json.dumps(ctx, default=str)
    assert "sealed" not in {k.lower() for k in ctx}
