"""The optional taskkit helpers: as-of alignment of sparse actions, the polars Task base (exports,
in-sample reads, the holdout boundary, leak scan), JSON TableTasks (as-of joins, delayed columns)
and the ready valuations. And the battery sample's simulator and oracle."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from taskkit import evaluators
from taskkit.table import TableTask
from taskkit.task import Task, align_actions, read_actions, ts_ns

MCP = Path(__file__).resolve().parent.parent


def test_align_actions_holds_forward_and_starts_at_initial():
    keys = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    out = align_actions(keys, np.array([20, 40], dtype=np.int64), np.array([1.0, -1.0]), initial=0.5)
    assert out.tolist() == [0.5, 1.0, 1.0, -1.0, -1.0]
    out = align_actions(keys, np.array([25], dtype=np.int64), np.array([1.0]))
    assert out.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]                   # never applied to an earlier row


def test_read_actions_keeps_the_last_duplicate_and_drops_nan(tmp_path):
    t = np.array(["2024-01-01T10:00", "2024-01-01T09:00", "2024-01-01T10:00", "2024-01-01T11:00"], dtype="datetime64[ns]")
    pl.DataFrame({"t": t, "pos": [1.0, 0.5, -1.0, float("nan")]}).write_parquet(tmp_path / "a.parquet")
    tt, a = read_actions(tmp_path / "a.parquet")
    assert a.tolist() == [0.5, -1.0] and (np.diff(tt) > 0).all()


def test_ts_ns_converts_zones_and_dates():
    assert ts_ns("2024-01-02T07:30:00+02:00") == ts_ns("2024-01-02 05:30:00")
    assert ts_ns("2024-01-02") == ts_ns("2024-01-02T00:00:00")


class Toy(Task):
    name = "toy"
    target = "y"
    holdout_from = "2024-01-03T15:00:00"                                # a time of day: snapped to midnight
    action = {"kind": "value", "min": -1, "max": 1, "initial": 0.0}

    def load_rows(self):
        t = np.arange(np.datetime64("2024-01-01T00", "h"), np.datetime64("2024-01-04T00", "h"), np.timedelta64(1, "h"))
        return pl.DataFrame({"t": t.astype("datetime64[ns]"), "y": np.arange(72, dtype=float) + 100})

    def evaluate(self, rows, actions):
        return {"segments": {"in_sample": {"score": float(actions.sum())}}, "curve": [], "curve_kind": "additive"}


def test_task_boundary_exports_and_in_sample_reads(tmp_path):
    task = Toy()
    assert task.describe()["holdout_from"] == "2024-01-03T00:00:00" and len(task.in_sample()) == 48
    info = task.export(tmp_path / "rows.parquet", until="2024-01-02 05:30")
    assert info["rows"] == 30 and pl.read_parquet(tmp_path / "rows.parquet")["t"].max() == np.datetime64("2024-01-02T05:00")
    assert len(task.sample(limit=500)["rows"]) == 48
    assert task.query("SELECT count(*) AS n FROM rows")["rows"] == [{"n": 48}]
    pl.DataFrame({"t": task.rows()["t"], "pos": [5.0] * 72}).write_parquet(tmp_path / "a.parquet")
    assert task.evaluate_file(tmp_path / "a.parquet")["segments"]["in_sample"]["score"] == 72.0   # clipped to 1


def test_table_task_asof_join_and_delayed_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("TASKKIT_CACHE", str(tmp_path / "cache"))
    t = np.arange(np.datetime64("2024-01-01T09:00"), np.datetime64("2024-01-01T10:00"), np.timedelta64(10, "m"))
    pl.DataFrame({"ts": t.astype("datetime64[ns]"), "price": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]}).write_parquet(tmp_path / "fast.parquet")
    (tmp_path / "slow.csv").write_text("when,vix\n2024-01-01 09:05:00,15\n2024-01-01 09:35:00,20\n")
    cfg = {"name": "joined", "target": "price", "holdout_from": "2024-01-02",
           "sources": [{"path": "fast.parquet", "time_column": "ts"},
                       {"path": "slow.csv", "time_column": "when", "prefix": "s_"}],
           "evaluator": {"kind": "forecast", "horizon": 1, "min_rows": 1}}
    rows = TableTask(cfg, tmp_path).rows()
    assert rows.columns == ["t", "price", "s_vix"]
    assert rows["s_vix"].to_list() == [None, 15, 15, 15, 20, 20]         # never a later slow row
    cfg2 = dict(cfg, name="delayed", shift_rows={"rows": 1, "columns": ["s_*"]})
    assert TableTask(cfg2, tmp_path).rows()["s_vix"].to_list() == [None, None, 15, 15, 15, 20]


def test_leak_scan_flags_a_late_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("TASKKIT_CACHE", str(tmp_path / "cache"))
    rng = np.random.default_rng(7)
    n = 3000
    t = np.datetime64("2024-01-01T14:00:00") + np.arange(n) * np.timedelta64(10, "s")
    price = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    late = np.r_[price[1:], price[-1]] * 3 + rng.normal(0, 1e-3, n)       # computed from the NEXT row
    pl.DataFrame({"ts": t.astype("datetime64[ns]"), "Close": price, "late": late}).write_parquet(tmp_path / "bars.parquet")
    cfg = {"name": "late", "target": "Close", "holdout_from": "2024-01-02",
           "sources": [{"path": "bars.parquet", "time_column": "ts"}], "evaluator": {"kind": "trading"}}
    assert TableTask(cfg, tmp_path).leak_scan()["suspects"] == ["late"]
    fixed = TableTask(dict(cfg, name="late2", shift_rows={"rows": 1, "except": ["Close"]}), tmp_path)
    assert fixed.leak_scan()["suspects"] == []


def test_ready_valuations():
    t = np.arange(np.datetime64("2024-01-01T00", "h"), np.datetime64("2024-01-13T12", "h"), np.timedelta64(1, "h"))
    y = np.sin(np.arange(len(t)) / 7) * 10 + 50
    rows = pl.DataFrame({"t": t.astype("datetime64[ns]"), "y": y})
    hold = ts_ns("2024-01-10")
    perfect = evaluators.forecast(rows, np.r_[y[1:], np.nan], target="y", holdout_ns=hold)
    naive = evaluators.forecast(rows, y, target="y", holdout_ns=hold)
    assert perfect["segments"]["in_sample"]["score"] == 1.0 and abs(naive["segments"]["in_sample"]["score"]) < 1e-9
    longonly = evaluators.trading(rows, np.where(np.arange(len(t)) % 10 < 5, 1.0, 0.0), price="y", holdout_ns=hold,
                                  min_active_days=1, min_side_share=0.2)
    assert longonly["unranked"].startswith("one-sided")
    never = evaluators.trading(rows, np.zeros(len(t)), price="y", holdout_ns=hold, min_active_days=1,
                               min_side_share=0.2)
    assert never["unranked"].startswith("no trades")


# ---------------------------------------------------------------------------------------------
# The battery sample
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def battery():
    sys.path.insert(0, str(MCP / "test" / "battery"))
    import server as B
    return B


def test_battery_simulator_respects_capacity_and_losses(battery):
    B = battery
    price = np.full(40, 50.0)
    _, soc, moved = B.simulate(price, np.ones(40))
    assert soc.max() <= B.CAPACITY_KWH + 1e-9 and moved[moved > 0].max() <= B.POWER_KW + 1e-9
    assert (moved[-5:] == 0).all()                                        # full: nothing more is bought
    _, soc, moved = B.simulate(price, -np.ones(40))
    assert soc.min() >= -1e-9 and -moved.sum() <= B.START_SOC * B.CAPACITY_KWH * B.EFF_OUT + 1e-9


def test_battery_oracle_beats_every_strategy(battery):
    B = battery
    rng = np.random.default_rng(1)
    h = np.arange(24 * 20)
    price = 50 + 30 * np.sin(2 * np.pi * h / 24) + rng.normal(0, 5, len(h))
    best = B.oracle(price, levels=28).sum()
    for a in (np.zeros(len(h)), np.sign(np.sin(2 * np.pi * (h + 7) / 24)), rng.uniform(-1, 1, len(h))):
        assert B.simulate(price, a)[0].sum() <= best + 1e-6
    assert best > 0
