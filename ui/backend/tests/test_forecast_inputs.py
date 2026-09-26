"""Forecast transparency (app/tslab.py): the input-combination cache, the budgeted exploration,
and the forecast report's "did it help" -- judged against the candidate's own parent."""

from __future__ import annotations

import asyncio
import json
import time
import zlib

import numpy as np
import pytest

from app import objectives as O
from app import tslab


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "DB_PATH", tmp_path / "objectives.sqlite3")
    monkeypatch.setattr(O, "_conn", None)
    monkeypatch.setattr(O, "WORK_ROOT", tmp_path / "work")
    now = time.time()
    O.db().execute(
        "INSERT INTO objectives (id, project_id, title, metric, split_date, created_at, updated_at) "
        "VALUES ('o1', 'p1', 'Best strategy', ?, '2024-07-19', ?, ?)",
        (json.dumps({"kind": "sharpe", "higher_is_better": True, "price_column": "Close", "cost_bps": 2}), now, now))
    O.db().commit()
    yield O.db()
    O.db().close()


WHERE = {"project_id": "p1", "objective_id": "o1", "dataset": "bars", "target": "Close", "horizon": 30, "bar": None,
         "model": "amazon/chronos-2", "context": 1024, "points": 300, "end": "2024-07-19", "n_rows": 5000}


def fake_score(inputs: list[str], seed: int = 0) -> dict:
    """Per-point errors as a forecaster would produce them: `good` cuts the error by 20%,
    `good2` by a further 10% on top of it, `noise` does nothing, `bad` makes it worse."""
    rng = np.random.default_rng(seed)
    err = 1.0 + rng.normal(scale=0.3, size=300).clip(-0.9, None)
    jitter = np.random.default_rng(zlib.crc32("|".join(sorted(inputs)).encode())).normal(scale=0.01, size=300)
    f = 1.0
    if "good" in inputs:
        f *= 0.8
    if "good2" in inputs:
        f *= 0.9
    if "bad" in inputs:
        f *= 1.2
    e = err * f + jitter
    return {"_err": e, "skill": round(1 - float(e.mean()), 5), "direction": 0.5, "coverage": 0.8,
            "qloss_rel": 0.7, "points": 300}


def test_combo_key_ignores_order_but_not_the_data():
    assert tslab.combo_key(WHERE, ["a", "b"]) == tslab.combo_key(WHERE, ["b", "a", "a"])
    assert tslab.combo_key(WHERE, ["a"]) != tslab.combo_key({**WHERE, "horizon": 12}, ["a"])
    assert tslab.combo_key(WHERE, ["a"]) != tslab.combo_key({**WHERE, "n_rows": 5001}, ["a"])
    assert tslab.combo_key(WHERE, []) != tslab.combo_key(WHERE, ["a"])


def test_a_stored_combination_comes_back_with_its_point_errors(db):
    sc = fake_score(["good"])
    tslab.save_combo(WHERE, ["good"], sc, "agent", "solo")
    got = tslab.load_combo(tslab.combo_key(WHERE, ["good"]))
    assert got["cached"] and got["skill"] == sc["skill"]
    assert np.allclose(got["_err"], sc["_err"], atol=1e-6)
    # paired comparisons still work on the stored errors
    tslab.save_combo(WHERE, [], fake_score([]), "agent", "baseline")
    assert tslab.paired(tslab.load_combo(tslab.combo_key(WHERE, [])), got)["significant"]


def _runner(forecasts: list):
    """run(inputs, kind) through the persistent cache, counting real forecasts."""
    async def run(inputs, kind):
        key = tslab.combo_key(WHERE, inputs)
        sc = tslab.load_combo(key)
        if sc is None:
            forecasts.append(sorted(inputs))
            sc = fake_score(inputs)
            tslab.save_combo(WHERE, inputs, sc, "t", kind)
        return {"inputs": sorted(inputs), "kind": kind, **sc}
    return run


def test_exploration_finds_the_inputs_that_help_and_never_repeats_a_combination(db):
    cands = ["noise", "good", "bad", "good2", "noise2"]
    forecasts: list = []
    res = asyncio.run(tslab.explore_search(_runner(forecasts), cands))
    assert sorted(res["best"]["inputs"]) == ["good", "good2"]
    assert res["best"]["vs_baseline_significant"]
    assert "good" in res["helpful"] and "noise" in res["useless"] and "bad" in res["useless"]
    assert len({tuple(f) for f in forecasts}) == len(forecasts)      # no combination twice in one run
    n_first = len(forecasts)
    again: list = []
    res2 = asyncio.run(tslab.explore_search(_runner(again), cands))
    assert again == [] and res2["best"]["inputs"] == res["best"]["inputs"]   # all served from the store
    assert n_first > 0


def test_combination_summary_for_the_brief(db):
    asyncio.run(tslab.explore_search(_runner([]), ["noise", "good", "bad"]))
    groups = tslab.combo_groups("p1", "2024-07-19")
    assert len(groups) == 1
    g = groups[0]
    assert g["helpful"] == ["good"] and "bad" in g["hurts"] and "noise" in g["useless"]
    brief = tslab.combo_brief("p1", {"split_date": "2024-07-19"})
    assert brief[0]["target"] == "Close" and brief[0]["best_inputs"] == ["good"]
    # A group evaluated past an objective's split saw its holdout: not shown for it.
    assert tslab.combo_groups("p1", "2024-01-01") == []


# ---------------------------------------------------------------------------------------
# The forecast report: helped / hurt, judged against the parent
# ---------------------------------------------------------------------------------------
def _feature(tmp_root, name: str, covariates=None):
    root = tmp_root / "o1" / "features"
    root.mkdir(parents=True, exist_ok=True)
    meta = {"view": f"fc_{name}", "file": f"{name}.parquet", "created_at": time.time(), "rows": 100,
            "params": {"dataset": "bars", "series": ["Close"], "covariates": covariates or [], "calendar": True,
                       "horizon": 30, "every": 30, "context": 512, "model": "amazon/chronos-2", "bar": None},
            "skill": {"Close": {"with_inputs": {"skill": 0.02, "direction": 0.53, "coverage": 0.8, "points": 900},
                                "without_inputs": {"skill": 0.01, "direction": 0.51, "coverage": 0.8},
                                "lift": {"gain": 0.01, "se": 0.004, "significant": True}, "lift_skill": 0.01}},
            "recipe": {"requested_by": "agent-x", "request": {"column": "Close"}}}
    (root / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")


def _cand(seq, score, used=None, parent=None):
    O.db().execute(
        "INSERT INTO candidates (id, objective_id, seq, created_at, model, status, is_score, metrics, parent_id, code) "
        "VALUES (?, 'o1', ?, ?, 'm', 'ok', ?, ?, ?, 'x=1')",
        (f"c{seq}", seq, time.time(), score, json.dumps({"features_used": used or []}), parent))
    O.db().commit()


def test_forecast_report_judges_a_feature_against_the_parent(db, tmp_path):
    _feature(tmp_path / "work", "close_x", ["GEX"])
    _feature(tmp_path / "work", "useless")
    # Parents without the forecast; children that added it scored higher each time -- even
    # though the children's scores are BELOW the no-forecast median (the global comparison
    # would call it harmful).
    for i, (p, c) in enumerate([(0.1, 0.3), (0.2, 0.4), (0.0, 0.1), (0.5, 0.6)]):
        _cand(10 + i, p)
        _cand(20 + i, c, ["fc_close_x"], parent=f"c{10 + i}")
    for i in range(6):
        _cand(40 + i, 2.0)
    rows = {r["view"]: r for r in tslab.forecast_report(O.get_objective("o1"))}
    r = rows["fc_close_x"]
    assert r["verdict"] == "helped" and r["basis"] == "vs parent" and r["n"] == 4 and r["effect"] > 0
    assert r["inputs_sent"]["covariates"] == ["GEX"] and r["inputs_sent"]["calendar"] is True
    assert r["skill"]["Close"]["lift"]["significant"] and r["requested_by"] == "agent-x"
    assert r["median_users"] < r["median_no_forecast"]          # the naive comparison would say "hurt"
    assert rows["fc_useless"]["verdict"] == "unused" and rows["fc_useless"]["used_by"] == 0


# ---------------------------------------------------------------------------------------
# What each forecast request sends, with dates (for the agent inspector)
# ---------------------------------------------------------------------------------------
def _grid(n: int, start: str = "2024-01-02T14:00:00"):
    return list(np.datetime64(start, "s") + np.arange(n) * np.timedelta64(10, "s"))


def test_input_streams_describe_the_dates_actually_sent():
    times = _grid(1000)
    d = O.input_streams("feature", "amazon/chronos-2", "bars", times, [99, 199, 299], 100, 6,
                        [("Close", "target"), ("GEX", "past covariate"), ("minute_of_day, weekday", "known ahead")],
                        every=100)
    assert d["as_of"] == {"first": "2024-01-02T14:16:30", "last": "2024-01-02T14:49:50", "anchors": 3, "every": 100}
    close, gex, cal = d["streams"]
    # the context of the first anchor starts at row 0; the last anchor's ends AT it (inclusive)
    assert (close["from"], close["to"], close["points"]) == ("2024-01-02T14:00:00", "2024-01-02T14:49:50", 300)
    assert gex["role"] == "past covariate" and gex["from"] == close["from"] and gex["bar"] == "10s"
    # known-ahead inputs span the horizon after each as-of, never the past
    assert cal["from"] == "2024-01-02T14:16:40" and cal["to"] == "2024-01-02T14:50:50" and cal["points"] == 6
    assert d["horizon"]["end"] == "2024-01-02T14:50:50"
    assert [s["context"] for s in d["samples"]] == [100, 100, 100]
    assert d["samples"][0] == {"as_of": "2024-01-02T14:16:30", "from": "2024-01-02T14:00:00",
                               "to": "2024-01-02T14:16:30", "context": 100, "horizon_end": "2024-01-02T14:17:30"}
    # the Forecast Lab reads the rows BEFORE its anchor
    lab = O.input_streams("lab test", "m", "bars", times, [100], 100, 6, [("Close", "target")], inclusive=False)
    assert lab["as_of"]["first"] == "2024-01-02T14:16:30" and lab["streams"][0]["from"] == "2024-01-02T14:00:00"


def test_input_streams_reach_the_forecast_row():
    from app import agent_activity as A

    A.reset()
    token = A._caller.set({"rid": 9001, "agent": "qwen", "path": "/api/objectives/o1/features", "recipe": {"column": "Close"}})
    try:
        detail = O.input_streams("feature", "amazon/chronos-2", "bars", _grid(500), [99, 499], 100, 6, [("Close", "target")])
        A.note_inputs("amazon/chronos-2", detail)          # described before the first call goes out
        A.note_forecast("amazon/chronos-2", {"inputs": [{"target": [1.0] * 100}], "horizon": 6}, 0.2, response=object())
        A.note_forecast("amazon/chronos-2", {"inputs": [{"target": [1.0] * 100}], "horizon": 6}, 0.2, response=object())
        second = O.input_streams("feature", "amazon/chronos-2", "bars", _grid(500), [199], 100, 6, [("GEX", "target")])
        A.note_inputs("amazon/chronos-2", second)          # a second build in the same request
    finally:
        A._caller.reset(token)
    rows = A.forecasts("amazon/chronos-2")
    assert len(rows) == 1 and rows[0]["calls"] == 2 and rows[0]["agent"] == "qwen"
    assert [i["streams"][0]["name"] for i in rows[0]["inputs"]] == ["Close", "GEX"]
    assert rows[0]["inputs"][0]["as_of"]["anchors"] == 2
    A.reset()
