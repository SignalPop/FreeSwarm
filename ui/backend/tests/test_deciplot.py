"""Decile studies (app/deci_core.py, app/deciplot.py): no look-ahead, nothing past the split, caching.

The core runs in the sandbox on in-sample data; here it is exercised directly on synthetic
bars. The properties that matter most are the ones a decile plot usually gets wrong:
deciles from FUTURE data (full-sample qcut), returns measured into the holdout, and returns
measured across the overnight gap.
"""

from __future__ import annotations

import ast
import json
import time

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from app import deci_core as D  # noqa: E402


def bars(days: int = 30, per_day: int = 240, seed: int = 0, start: str = "2024-01-02 14:00:00",
         signal_edge: float = 0.0):
    """10 s bars, `per_day` per session, sessions on consecutive weekdays. With `signal_edge`,
    the signal predicts the NEXT bar's return (a planted, causal relationship)."""
    rng = np.random.default_rng(seed)
    t0 = np.datetime64(start.replace(" ", "T"), "us")
    first, offset = t0.astype("datetime64[D]"), t0 - t0.astype("datetime64[D]")
    sessions = np.busday_offset(first, np.arange(days), roll="forward")   # consecutive weekdays
    step = np.arange(per_day) * np.timedelta64(10, "s")
    times = np.concatenate([d.astype("datetime64[us]") + offset + step for d in sessions])
    n = len(times)
    x = rng.normal(size=n)
    r = rng.normal(scale=1e-4, size=n)
    r[1:] += signal_edge * 1e-4 * x[:-1]      # x at bar t moves the return from t to t+1
    price = 100 * np.exp(np.cumsum(r))
    return times, x, price


def day(t: np.ndarray) -> np.ndarray:
    return t.astype("datetime64[D]")


# ---------------------------------------------------------------------------------------
# Causality of the rolling edges
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("window_bars", [None, 500])
def test_appending_future_data_never_changes_earlier_deciles(window_bars):
    t, x, _ = bars(days=40)
    cut = 30 * 240
    short = D.assign_deciles(x[:cut], D.rolling_edges(t[:cut], x[:cut], window_days=10, window_bars=window_bars))
    full = D.assign_deciles(x, D.rolling_edges(t, x, window_days=10, window_bars=window_bars))
    assert (short >= 0).sum() > 1000            # the test is not vacuous: most rows are bucketed
    assert np.array_equal(short, full[:cut])


def test_a_bar_never_sets_its_own_edges():
    """Days mode: the edges for a session come from the sessions BEFORE it. Changing every
    value of today leaves today's edges alone."""
    t, x, _ = bars(days=25)
    e1 = D.rolling_edges(t, x, window_days=10)
    today = day(t) == day(t)[-1]
    x2 = x.copy()
    x2[today] = 1e6
    e2 = D.rolling_edges(t, x2, window_days=10)
    assert np.allclose(e1[today], e2[today])
    # bars mode: excluding the current bar -- changing bar i does not move bar i's edges
    e3 = D.rolling_edges(t, x, window_bars=300)
    x3 = x.copy()
    x3[-1] = 1e9
    e4 = D.rolling_edges(t, x3, window_bars=300)
    assert np.allclose(e3[-1], e4[-1])


def test_warm_up_is_skipped_not_estimated():
    t, x, _ = bars(days=15)
    dec = D.assign_deciles(x, D.rolling_edges(t, x, window_days=10))
    first_day = day(t) == day(t)[0]
    assert (dec[first_day] == -1).all()
    tenth = np.unique(day(t))[9]
    assert (dec[day(t) <= tenth] == -1).all()
    assert (dec[day(t) > tenth] >= 0).all()


def test_resampled_bars_are_stamped_at_their_last_underlying_bar():
    t, x, p = bars(days=1, per_day=12)
    b = D.resample_last(t, x, p, "30s")
    assert b["t"].to_numpy()[0] == t[2]          # 14:00:00, :10, :20 -> stamped :20
    assert b["x"][0] == x[2] and b["p"][0] == p[2]


# ---------------------------------------------------------------------------------------
# Nothing after the split; nothing across the session close
# ---------------------------------------------------------------------------------------
def test_nothing_after_the_cut_is_read():
    t, x, p = bars(days=40)
    cut = str(np.unique(day(t))[30])
    before = t < np.datetime64(cut, "us")
    clean = D.study(t[before], x[before], p[before], cut=cut, timeframes=("10s", "1min"), horizons=(1, 6), window_days=10)
    # Garbage after the cut -- a price that would dominate every statistic if it leaked in.
    x2, p2 = x.copy(), p.copy()
    x2[~before] = 1e9
    p2[~before] = 1e9
    dirty = D.study(t, x2, p2, cut=cut, timeframes=("10s", "1min"), horizons=(1, 6), window_days=10)
    assert json.dumps(clean, sort_keys=True) == json.dumps(dirty, sort_keys=True)
    assert dirty["timeframes"]["10s"]["to"] < cut


def test_forward_returns_stop_at_the_session_close():
    t, x, p = bars(days=2, per_day=10)
    b = D.resample_last(t, x, p, "10s")
    r = D.forward_returns_bps(b, 3)
    assert np.isnan(r[7:10]).all()               # the last 3 bars of day 1 have no same-day outcome
    assert np.isfinite(r[:7]).all()
    assert np.isnan(r[-3:]).all()


# ---------------------------------------------------------------------------------------
# It finds what is there, and not what is not
# ---------------------------------------------------------------------------------------
def test_planted_signal_is_monotone_and_noise_is_not():
    t, x, p = bars(days=40, signal_edge=0.5, seed=1)
    res = D.study(t, x, p, timeframes=("10s",), horizons=(1,), window_days=10)
    cell = res["timeframes"]["10s"]["horizons"]["1"]
    assert cell["verdict"] == "monotone" and cell["spread_bps"] > 0 and cell["spearman"] > 0.9
    assert cell["consistency"] == 1.0
    assert res["summary"]["direction"] == "higher -> up"
    t, x, p = bars(days=40, signal_edge=0.0, seed=2)
    res = D.study(t, x, p, timeframes=("10s",), horizons=(1,), window_days=10)
    assert res["timeframes"]["10s"]["horizons"]["1"]["verdict"] != "monotone"


# ---------------------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------------------
def test_expressions_evaluate_and_refuse_anything_else():
    df = pl.DataFrame({"GEX": [1.0, -2.0, 4.0], "Pinning_TotalAbsGex": [2.0, 0.0, 8.0], "Odd Name": [1.0, 2.0, 3.0]})
    got = D.eval_expression(df, "gex / Pinning_TotalAbsGex")
    assert got[0] == 0.5 and np.isnan(got[1]) and got[2] == 0.5     # division by zero -> NaN
    assert np.allclose(D.eval_expression(df, 'abs(GEX) + "Odd Name"'), [2.0, 4.0, 7.0])
    assert D.expression_columns("greatest(GEX, 0) * 2", list(df.columns)) == ["GEX"]
    for bad in ("__import__('os')", "GEX.real", "GEX[0]", "open('x')", "GEX ^ 2", "nope + 1", "lambda: 1"):
        with pytest.raises(ValueError):
            D.expression_columns(bad, list(df.columns))


def test_the_harness_is_valid_python():
    from app import deciplot

    ast.parse(deciplot.harness_code({"signals": []}))


# ---------------------------------------------------------------------------------------
# Store and cache
# ---------------------------------------------------------------------------------------
@pytest.fixture
def store(tmp_path, monkeypatch):
    from app import deciplot, projects
    from app import objectives as O

    monkeypatch.setattr(O, "DB_PATH", tmp_path / "objectives.sqlite3")
    monkeypatch.setattr(O, "_conn", None)
    monkeypatch.setattr(O, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(deciplot, "_ready", False)
    now = time.time()
    O.db().execute(
        "INSERT INTO objectives (id, project_id, title, metric, dataset, time_column, split_date, created_at, updated_at) "
        "VALUES ('o1', 'p1', 'Best strategy', ?, 'bars', 'SlotUtc', '2024-07-19', ?, ?)",
        (json.dumps({"kind": "sharpe", "higher_is_better": True, "price_column": "Close", "cost_bps": 2}), now, now))
    O.db().commit()
    monkeypatch.setattr(projects, "get", lambda pid: {"id": pid, "data_dir": str(tmp_path)})
    monkeypatch.setattr(deciplot, "dataset_columns",
                        lambda data_dir, ds: [("SlotUtc", "TIMESTAMP"), ("Close", "DOUBLE"), ("GEX", "DOUBLE"),
                                              ("IntrVol", "DOUBLE")])
    calls = []

    async def fake_run(obj, project, specs, tfs, hs, window_days, window_bars):
        calls.append([s["signal"] for s in specs])
        t, x, p = bars(days=20)
        return ({s["signal"]: D.study(t, x, p, cut=obj["split_date"], timeframes=tfs, horizons=hs,
                                      window_days=window_days) for s in specs}, {}, "")

    monkeypatch.setattr(deciplot, "_run_specs", fake_run)
    yield deciplot, calls
    O.db().close()


def test_a_repeated_study_is_served_from_the_store(store):
    import asyncio

    deciplot, calls = store
    req = deciplot.DeciReq(signal="GEX", timeframes=["10s", "1min"], horizons=[1, 3], window_days=5, author="a")
    first = asyncio.run(deciplot.run_study("o1", req))
    assert first["cached"] is False and len(calls) == 1
    again = asyncio.run(deciplot.run_study("o1", deciplot.DeciReq(signal=" GEX ", timeframes=["10s", "1min"],
                                                                  horizons=[3, 1], window_days=5)))
    assert again["cached"] is True and again["id"] == first["id"] and len(calls) == 1
    # A different window is a different study.
    asyncio.run(deciplot.run_study("o1", deciplot.DeciReq(signal="GEX", timeframes=["10s", "1min"], horizons=[1, 3],
                                                          window_days=10)))
    assert len(calls) == 2
    # force re-runs, and history is kept: the listing shows the latest with its run count.
    asyncio.run(deciplot.run_study("o1", req.model_copy(update={"force": True})))
    assert len(calls) == 3
    from app.objectives import get_objective

    rows = deciplot.list_studies(get_objective("o1"))
    assert len(rows) == 2 and max(r["runs"] for r in rows) == 2
    assert deciplot.brief(get_objective("o1"))["studied"] == 2


def test_a_study_that_saw_this_objectives_holdout_is_hidden(store):
    deciplot, _ = store
    from app.objectives import get_objective

    obj = get_objective("o1")
    spec = {"signal": "GEX", "kind": "dataset", "expr": "GEX", "columns": ["GEX"]}
    later = {**obj, "split_date": "2025-01-01"}
    deciplot.store("p1", "o2", "k-late", "x", spec, {"cut": "2025-01-01", "kind": "dataset"}, {"summary": {}})
    deciplot.store("p1", "o1", "k-ok", "x", spec, {"cut": "2024-07-19", "kind": "dataset"}, {"summary": {}})
    assert [r["key"] for r in deciplot.list_studies(obj)] == ["k-ok"]
    assert len(deciplot.list_studies(later)) == 2


def test_bad_signals_are_refused_before_a_container_is_spent(store):
    import asyncio

    from fastapi import HTTPException

    deciplot, calls = store
    with pytest.raises(HTTPException):
        asyncio.run(deciplot.run_study("o1", deciplot.DeciReq(signal="NoSuchColumn / GEX")))
    with pytest.raises(HTTPException):
        asyncio.run(deciplot.run_study("o1", deciplot.DeciReq(signal="GEX", timeframes=["1 week"])))
    assert calls == []


def test_runs_on_read_only_arrays():
    """Polars' to_numpy() hands back read-only views (so did pandas 3's): the core must never
    write into one (it did once -- `keep &= ...` -- and every study failed)."""
    t, x, p = bars(days=15)
    frame = pl.DataFrame({"t": t, "x": x, "p": p})
    xs, ps = frame["x"].to_numpy(), frame["p"].to_numpy()
    xs.flags.writeable = False
    ps.flags.writeable = False
    res = D.study(frame["t"], xs, ps, cut="2024-01-15", timeframes=("10s",), horizons=(1,), window_days=5)
    assert res["rows"] > 0


def test_shape_labels_describe_the_curve_not_just_its_ends():
    from app.deciplot import shape_of

    assert shape_of([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5], 0.99) == "rising"
    assert shape_of([5, 4, 3, 2, 1, 0, -1, -2, -3, -4], -0.99) == "falling"
    assert shape_of([3, 1, 0, 0, 0, 0, 0, 0, 1, 3], 0.0).startswith("U-shaped")
    assert shape_of([0, 0, 0, 0, 0, 0, 0, 0, 0, 3], 0.4).startswith("top decile only")
    assert shape_of([None] * 7 + [1, 2, 3], None) == "too few buckets"
