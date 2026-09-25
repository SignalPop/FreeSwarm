"""Candidate scoring: costs vs gross vs flipped, failure hints, and ft.forecast auto-building."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import duckdb
import pytest

from app import objectives as O


# ---------------------------------------------------------------------------------------
# Mark to market: gross and flipped series ride along with the net one
# ---------------------------------------------------------------------------------------
def _write_prices(data_dir: Path) -> None:
    # Two days of 4 bars, price rising 1% a bar.
    con = duckdb.connect(":memory:")
    con.execute(f"""COPY (SELECT * FROM (VALUES
        (TIMESTAMP '2024-01-02 10:00', 100.0), (TIMESTAMP '2024-01-02 10:01', 101.0),
        (TIMESTAMP '2024-01-02 10:02', 102.01), (TIMESTAMP '2024-01-02 10:03', 103.0301),
        (TIMESTAMP '2024-01-03 10:00', 104.060401), (TIMESTAMP '2024-01-03 10:01', 105.10100501),
        (TIMESTAMP '2024-01-03 10:02', 106.1520150601), (TIMESTAMP '2024-01-03 10:03', 107.213535210701)
    ) v(ts, close)) TO '{(data_dir / "px.parquet").as_posix()}' (FORMAT parquet)""")
    con.close()


def _write_positions(path: Path, rows: list[tuple[str, float]]) -> None:
    con = duckdb.connect(":memory:")
    values = ", ".join(f"(TIMESTAMP '{t}', {p})" for t, p in rows)
    con.execute(f"COPY (SELECT * FROM (VALUES {values}) v(t, pos)) TO '{path.as_posix()}' (FORMAT parquet)")
    con.close()


def _obj(cost_bps: float) -> dict:
    return {"id": "o1", "dataset": "px.parquet", "time_column": "ts", "split_date": "2024-01-03",
            "metric": {"kind": "sharpe", "price_column": "close", "cost_bps": cost_bps, "max_leverage": 1.0,
                       "periods_per_year": 252}}


def test_mark_to_market_gross_and_inverted(tmp_path):
    _write_prices(tmp_path)
    pos = tmp_path / "pos.parquet"
    # Long from the first bar, flat at 10:02 on day 2: two trades.
    _write_positions(pos, [("2024-01-02 10:00", 1.0), ("2024-01-03 10:02", 0.0)])
    net, info = O._mark_to_market(_obj(cost_bps=10.0), str(tmp_path), pos)
    gross, inverted = info.pop("_gross"), info.pop("_inverted")
    assert [d for d, _ in net] == [d for d, _ in gross] == [d for d, _ in inverted]
    for (_, n), (_, g), (_, i) in zip(net, gross, inverted):
        assert g >= n                    # costs only ever take away
        assert i < 0 < g                 # rising prices: long wins before costs, the flip loses
    # Day 1: three bars held long at +1% each, no costs on a return that has no trade before it.
    assert math.isclose(gross[0][1], 1.01 ** 3 - 1, rel_tol=1e-9)
    assert not any(k.startswith("_") for k in info)


def test_mark_to_market_without_costs_net_equals_gross(tmp_path):
    _write_prices(tmp_path)
    pos = tmp_path / "pos.parquet"
    _write_positions(pos, [("2024-01-02 10:00", 1.0)])
    net, info = O._mark_to_market(_obj(cost_bps=0.0), str(tmp_path), pos)
    assert net == info["_gross"]


# ---------------------------------------------------------------------------------------
# The cost breakdown and what the agent is told
# ---------------------------------------------------------------------------------------
def _costs(net, gross, inv, kind="sharpe"):
    return {"metric": kind, "changes_per_day": 104.9,
            "in_sample": {"net": net, "gross": gross, "inverted": inv,
                          "return_net": -0.0139, "return_gross": -0.0020, "return_inverted": -0.0099}}


def test_verdict_edge_given_away_by_costs():
    v = O._cost_verdict(_costs(net=-3.1, gross=0.9, inv=-5.0))
    assert "edge before costs" in v and "Trade LESS" in v


def test_verdict_flip_when_flipped_survives_costs():
    v = O._cost_verdict(_costs(net=-2.5, gross=-2.0, inv=1.6))
    assert "FLIPPED" in v and "1.60" in v


def test_verdict_flip_and_trade_less():
    # Candidate #532: wrong way before costs, and the flipped version still loses to costs.
    v = O._cost_verdict(_costs(net=-3.97, gross=-0.88, inv=-3.95))
    assert "points the wrong way" in v and "Flip it AND trade LESS" in v


def test_verdict_no_edge():
    v = O._cost_verdict(_costs(net=-3.0, gross=0.2, inv=-3.2))
    assert "No clear edge" in v


def test_verdict_healthy_strategy_gets_no_advice():
    v = O._cost_verdict(_costs(net=1.1, gross=1.2, inv=-1.5))
    assert v.endswith("position changes/day).")


def test_verdict_missing_numbers():
    assert O._cost_verdict(_costs(net=None, gross=None, inv=None)) is None


def test_costs_splits_segments_and_counts_changes():
    days = [f"2024-01-{d:02d}" for d in range(1, 11)]
    net = [[d, 0.001 * ((-1) ** k)] for k, d in enumerate(days)]
    gross = [[d, r + 0.0005] for d, r in net]
    inv = [[d, -r - 0.0005] for d, r in gross]
    obj = {"split_date": "2024-01-06", "metric": {"kind": "sharpe", "periods_per_year": 252, "cost_bps": 2}}
    c = O._costs(obj, net, gross, inv, changes=50)
    assert c["changes_per_day"] == 5.0
    assert set(c) >= {"in_sample", "holdout", "verdict"}
    assert c["in_sample"]["gross"] > c["in_sample"]["net"]


def test_agent_view_shows_in_sample_costs_only(monkeypatch):
    c = {"id": "c1", "seq": 1, "status": "ok", "is_score": -1.0, "lookahead": "pass", "stdout": "",
         "metrics": {"in_sample": {"sharpe": -1.0},
                     "costs": {"changes_per_day": 3, "verdict": "told",
                               "in_sample": {"net": -1.0, "gross": 0.5, "inverted": -2.0},
                               "holdout": {"net": 9.0, "gross": 9.0, "inverted": 9.0}}}}
    monkeypatch.setattr(O, "_ranked", lambda *a, **k: [])  # no database needed for this view
    view = O.agent_view({"id": "o1", "metric": {"kind": "sharpe", "higher_is_better": True}}, c)
    assert view["costs_in_sample"] == {"after_costs": -1.0, "before_costs": 0.5, "flipped": -2.0,
                                       "position_changes_per_day": 3}
    assert view["diagnosis"] == "told"
    assert "9.0" not in str(view)  # the holdout never reaches the agent


# ---------------------------------------------------------------------------------------
# Failure hints
# ---------------------------------------------------------------------------------------
def test_failure_note_explains_clashing_forecast_columns():
    note = O._failure_note("Traceback ...\nKeyError: 'fc_median'")
    assert "fc_median_x" in note and 'prefix="x_"' in note


def test_failure_note_plain_otherwise():
    assert O._failure_note("ZeroDivisionError: division by zero") == "the script failed -- see stderr"


# ---------------------------------------------------------------------------------------
# ft.forecast: build what the script asked for, then run it again
# ---------------------------------------------------------------------------------------
@pytest.fixture
def fake_harness(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "WORK_ROOT", tmp_path)
    state = {"runs": 0, "built": []}

    def make_run(requests_per_run):
        async def fake_run(code, data_dir, catalog, mirror, timeout_s, obj, cut=None, extra_files=None):
            state["runs"] += 1
            asked = requests_per_run(state["runs"])
            return {"ok": not asked, "stderr": "ForecastPending" if asked else "", "stdout": "",
                    "forecast_requests": asked, "run_dir": str(tmp_path)}
        monkeypatch.setattr(O, "_run", fake_run)

    async def fake_build(obj, req, requested_by=None):
        root = O._features_root(obj["id"])
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{req.name}.parquet").write_bytes(b"x")
        state["built"].append((req.name, req.column, req.covariates, requested_by))
        return {"file": f"{req.name}.parquet"}

    monkeypatch.setattr(O, "build_feature", fake_build)
    state["make_run"] = make_run
    return state


def _req(name, col="GEX"):
    return {"name": name, "recipe": {"column": col, "covariates": ["A"], "horizon": 6, "model": None}}


def test_run_forecasting_builds_then_reruns(fake_harness):
    fake_harness["make_run"](lambda n: [_req("auto_aaa")] if n == 1 else [])
    rep = asyncio.run(O._run_forecasting("code", "d", [], None, 30, {"id": "o1"}, requested_by="candidate #7"))
    assert rep["ok"] and fake_harness["runs"] == 2
    assert fake_harness["built"] == [("auto_aaa", "GEX", ["A"], "candidate #7")]


def test_run_forecasting_does_not_rebuild_existing(fake_harness):
    fake_harness["make_run"](lambda n: [_req("auto_aaa")] if n == 1 else [])
    asyncio.run(O._run_forecasting("code", "d", [], None, 30, {"id": "o1"}))
    # A run that fails for another reason while listing an already-built forecast is not retried.
    fake_harness["make_run"](lambda n: [_req("auto_aaa")])
    rep = asyncio.run(O._run_forecasting("code", "d", [], None, 30, {"id": "o1"}))
    assert not rep["ok"] and len(fake_harness["built"]) == 1


def test_run_forecasting_caps_new_forecasts(fake_harness):
    many = [_req(f"auto_{i}", col=f"c{i}") for i in range(5)]
    fake_harness["make_run"](lambda n: many)
    rep = asyncio.run(O._run_forecasting("code", "d", [], None, 30, {"id": "o1"}))
    assert len(fake_harness["built"]) == O.MAX_AUTO_FORECASTS
    assert "could not be built" in rep["stderr"] and "at most" in rep["stderr"]


def test_run_forecasting_reports_build_errors(fake_harness, monkeypatch):
    fake_harness["make_run"](lambda n: [_req("auto_bad")])

    async def failing_build(obj, req, requested_by=None):
        raise O.HTTPException(status_code=409, detail="no time-series model is loaded")

    monkeypatch.setattr(O, "build_feature", failing_build)
    rep = asyncio.run(O._run_forecasting("code", "d", [], None, 30, {"id": "o1"}))
    assert not rep["ok"] and "no time-series model is loaded" in rep["stderr"]
