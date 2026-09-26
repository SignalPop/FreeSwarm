"""Regime Lab: attribution of member P&L to regimes, route selection and the exact router."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import regimes as R

FT_PATH = Path(__file__).resolve().parents[2] / "sandbox" / "ft.py"


def _ft():
    spec = importlib.util.spec_from_file_location("ft_under_test", FT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _market(tmp: Path, days: int = 60, per_day: int = 40, seed: int = 0):
    """Bars of a random walk, two regimes alternating by day, and two members:
    `trend` holds +1 (earns on the up days of regime A), `fade` holds -1."""
    rng = np.random.default_rng(seed)
    t = []
    for d in pd.bdate_range("2024-01-02", periods=days):
        t += list(pd.date_range(d + pd.Timedelta(hours=14, minutes=30), periods=per_day, freq="1min"))
    t = pd.DatetimeIndex(t)
    day = np.repeat(np.arange(days), per_day)
    regime = np.where(day % 2 == 0, "A", "B")
    drift = np.where(regime == "A", 4e-4, -4e-4)
    p = 100 * np.exp(np.cumsum(drift + rng.normal(0, 1e-3, len(t))))
    pd.DataFrame({"t": t, "p": p}).to_parquet(tmp / "px.parquet", index=False)
    pd.DataFrame({"t": t, "label": regime}).to_parquet(tmp / "labels.parquet", index=False)
    pd.DataFrame({"t": t, "pos": 1.0}).to_parquet(tmp / "trend.parquet", index=False)
    pd.DataFrame({"t": t, "pos": -1.0}).to_parquet(tmp / "fade.parquet", index=False)
    return f"SELECT t, p FROM read_parquet('{(tmp / 'px.parquet').as_posix()}')"


def _run(tmp: Path, routes=None, cost_bps=0.0):
    sql = _market(tmp)
    return R.analyze(sql, tmp / "labels.parquet", [(1, tmp / "trend.parquet"), (2, tmp / "fade.parquet")],
                     cost_bps=cost_bps, max_leverage=1.0, split="2024-03-01", mid=None,
                     spec={"kind": "module", "module": "x"}, routes=routes)


def test_cells_add_up_to_each_member(tmp_path):
    res = _run(tmp_path)
    for s in ("1", "2"):
        total = np.array(res["daily"]["members"][s])
        by_regime = sum(np.array(res["daily"]["cells"][lab][s]) for lab in res["daily"]["cells"])
        assert np.allclose(total, by_regime, atol=1e-9)


def test_routes_pick_the_member_that_works_in_both_halves(tmp_path):
    res = _run(tmp_path)
    assert res["suggested"] == {"A": 1, "B": 2}
    assert res["routes_source"] == "suggested"
    # Routing each regime to the member that earns there beats either member alone.
    assert res["router"]["is"]["sharpe"] > max(res["member_stats"][s]["is"]["sharpe"] or -9 for s in ("1", "2"))
    # Contributions are attributed to the member traded and sum to the router.
    contrib = sum(np.array(v) for v in res["daily"]["contrib"].values())
    assert np.allclose(contrib, res["daily"]["router"], atol=1e-9)


def test_router_routed_everywhere_to_one_member_equals_that_member(tmp_path):
    res = _run(tmp_path, routes={"A": 1, "B": 1}, cost_bps=2.0)
    assert res["routes_source"] == "custom"
    assert np.allclose(res["daily"]["router"], res["daily"]["members"]["1"], atol=1e-9)


def test_router_pays_switching_costs(tmp_path):
    free = _run(tmp_path, routes={"A": 1, "B": 2}, cost_bps=0.0)
    paid = _run(tmp_path, routes={"A": 1, "B": 2}, cost_bps=5.0)
    # +1 -> -1 at every regime change is a trade of 2; the members alone never trade.
    switches = sum(1 for a, b in zip(free["daily"]["regime"], free["daily"]["regime"][1:]) if a != b)
    gap = free["router"]["is"]["pnl"] + free["router"]["ho"]["pnl"] - paid["router"]["is"]["pnl"] - paid["router"]["ho"]["pnl"]
    assert gap == pytest.approx(switches * 2 * 5e-4, rel=0.05)


def test_holdout_is_kept_apart(tmp_path):
    res = _run(tmp_path)
    split = "2024-03-01"
    is_days = sum(1 for d in res["dates"] if d < split)
    assert res["router"]["is"]["days"] == is_days
    assert res["router"]["ho"]["days"] == len(res["dates"]) - is_days
    assert res["mid_date"] < split


def test_router_code_is_a_script(tmp_path):
    obj = {"time_column": "SlotUtc", "dataset": "bars"}
    spec = {"kind": "fields", "fields": [{"field": "GEX", "n": 3}, {"field": "IntrVol", "n": 3}],
            "smooth": 360, "window_days": 20}
    code = R.router_code(obj, spec, {"GEX:high|IntrVol:low": 12, "GEX:low|IntrVol:high": None,
                                     "GEX:mid|IntrVol:mid": 7}, 3)
    compile(code, "router.py", "exec")
    assert "ft.candidate_positions(12)" in code and "ft.candidate_positions(7)" in code
    assert "GEX:low|IntrVol:high" not in code
    from app.objectives import member_seqs

    assert member_seqs(code) == [7, 12]
    with pytest.raises(Exception):
        R.router_code(obj, spec, {"GEX:high|IntrVol:low": None}, 3)


def test_candidate_positions_captures_a_member(tmp_path, monkeypatch):
    ft = _ft()
    monkeypatch.setattr(ft, "_FT", str(tmp_path))
    monkeypatch.setattr(ft, "_RESULT", str(tmp_path / "result.json"))
    (tmp_path / "members").mkdir()
    (tmp_path / "members" / "5.py").write_text(
        "import pandas as pd, ft_under_test as ft\n"
        "t = pd.date_range('2024-01-02 14:30', periods=4, freq='15min')\n"
        "ft.report(trades=3)\n"
        "ft.report_positions(pd.Series([0, 1, 1, -1], index=t))\n", encoding="utf-8")
    import sys

    monkeypatch.setitem(sys.modules, "ft_under_test", ft)
    got = ft.candidate_positions(5)
    assert list(got.values) == [0, 1, 1, -1] and got.name == "#5"
    # The member's report() and positions did not become this run's result.
    assert not (tmp_path / "result.json").exists() and not (tmp_path / "positions.parquet").exists()
    with pytest.raises(FileNotFoundError):
        ft.candidate_positions(6)


def test_regime_grid_is_causal_and_route_aligns_timeframes():
    ft = _ft()
    n = 3000
    t = pd.date_range("2024-01-02 14:30", periods=n, freq="10s")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"T": t, "GEX": rng.normal(size=n).cumsum(), "IntrVol": rng.normal(size=n).cumsum()})
    full = ft.regime_grid(df, {"GEX": 3, "IntrVol": 2}, time="T", smooth=10, window=200)
    cut = ft.regime_grid(df.iloc[:1700], {"GEX": 3, "IntrVol": 2}, time="T", smooth=10, window=200)
    assert (full.iloc[:1700].to_numpy() == cut.to_numpy()).all()
    assert set(full.unique()) <= {"warmup"} | {f"GEX:{g}|IntrVol:{v}" for g in ("low", "mid", "high") for v in ("low", "high")}
    slow = pd.Series(np.sign(rng.normal(size=n // 90)), index=t[89::90])
    pos = ft.route(full, {lab: slow for lab in full.unique() if lab != "warmup"})
    assert len(pos) == n and pos.index.equals(full.index)
    # Carried as-of backward: bar i holds the latest slow position at or before it.
    i = 500
    expect = slow[slow.index <= t[i]].iloc[-1] if full.iloc[i] != "warmup" else 0.0
    assert pos.iloc[i] == expect
    with pytest.raises(TypeError, match="function or module"):
        ft.route(full, {"GEX:low|IntrVol:low": ft.route})
