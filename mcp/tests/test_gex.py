"""The GEX server's internals: position management (target and order modes, bounds, direction,
flat at each close), trades, valuation (curve, ratios, segment matching, side balance), and the
data layer's delay. Skips when mcp/gex is absent (it is git-ignored)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

GEX = Path(__file__).resolve().parent.parent / "gex"
if not (GEX / "positions.py").is_file():
    pytest.skip("mcp/gex is not present (it is git-ignored)", allow_module_level=True)
sys.path.insert(0, str(GEX))

import positions as P  # noqa: E402
import valuation as V  # noqa: E402
from data import GexData  # noqa: E402

NS = 1_000_000_000


def _keys(days=2, per_day=6):
    base = np.datetime64("2024-01-02T14:00:00", "ns").astype(np.int64)
    return np.concatenate([base + d * 86_400 * NS + np.arange(per_day) * 60 * NS for d in range(days)])


def test_target_mode_holds_bounds_and_flattens_each_close():
    k = _keys()
    t = k[[0, 3, 7]]
    pos = P.manage(k, t, np.array([5.0, -1.0, 2.0]), P.Rules(max_position=3, flat_each_day=True))
    # 5 is clipped to 3; the last bar of each day is flat; day 2 starts with the carried -1 (a new trade).
    assert pos.tolist() == [3, 3, 3, -1, -1, 0, -1, 2, 2, 2, 2, 0]
    longonly = P.manage(k, t, np.array([5.0, -1.0, 2.0]), P.Rules(max_position=3, direction="long"))
    assert longonly.min() == 0


def test_order_mode_fills_buys_and_sells_within_bounds():
    k = _keys()
    # buy 2, buy 2 (only 1 fills at max 3), sell 5 (to -2), next day: buy 1 from flat.
    t, a = k[[0, 1, 3, 8]], np.array([2.0, 2.0, -5.0, 1.0])
    pos = P.manage(k, t, a, P.Rules(mode="order", max_position=3, flat_each_day=True))
    assert pos.tolist() == [2, 3, 3, -2, -2, 0, 0, 0, 1, 1, 1, 0]
    # without the close rule the position carries over night
    pos = P.manage(k, t, a, P.Rules(mode="order", max_position=3))
    assert pos.tolist() == [2, 3, 3, -2, -2, -2, -2, -2, -1, -1, -1, -1]


def test_trades_and_valuation_of_a_known_path():
    k = _keys(days=30, per_day=40)
    rng = np.random.default_rng(5)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(k))))
    acts = np.sign(np.sin(np.arange(len(k)) / 7))
    pos = P.manage(k, k, acts, P.Rules(max_position=1, flat_each_day=True))
    tr = P.trades(k, price, pos, 1.0)
    assert (tr["side"] != 0).all() and (tr["exit_t"] >= tr["entry_t"]).all()
    assert ((tr["exit_t"] // (86_400 * NS)) == (tr["entry_t"] // (86_400 * NS))).all()   # all intraday
    hold = int(np.datetime64("2024-01-22", "ns").astype(np.int64))
    ev = V.value(k, price, pos, hold, {"score": "calmar", "cost_bps": 1, "min_active_days": 1, "min_side_share": 0.2})
    ins = ev["segments"]["in_sample"]
    assert ins["score"] == ins["calmar"] and ev["curve_kind"] == "returns" and len(ev["curve"]) == 30
    d = ev["diagnostics"]["in_sample"]
    assert d["trades"]["long"] > 0 and d["trades"]["short"] > 0 and ev["unranked"] is None
    assert d["segment_matching"] is None or "net" in d["segment_matching"]
    onesided = V.value(k, price, np.clip(pos, 0, None), hold, {"min_active_days": 1, "min_side_share": 0.2})
    assert onesided["unranked"].startswith("one-sided")


def test_segment_matching_rewards_the_right_side():
    k = _keys(days=1, per_day=6)
    p = np.array([100, 101, 102, 101, 100, 100.0])
    hold = 2**62
    right = V.segment_matching(k, p, np.array([1, 1, -1, -1, 0, 0.0]), 0.01, hold)["in_sample"]
    wrong = V.segment_matching(k, p, -np.array([1, 1, -1, -1, 0, 0.0]), 0.01, hold)["in_sample"]
    drift = V.segment_matching(k, p, np.ones(6), 0.01, hold)["in_sample"]
    assert (right["net"], wrong["net"], drift["net"]) == (2, -2, 0)


def test_data_layer_delays_options_columns_and_exports_before_the_cut(tmp_path):
    t = (np.datetime64("2024-01-02T14:00:00") + np.arange(10) * np.timedelta64(10, "s")).astype("datetime64[ns]")
    pl.DataFrame({"SlotUtc": t, "Close": np.arange(10) + 100.0, "GEX": np.arange(10) * 1.0}).write_parquet(tmp_path / "b.parquet")
    d = GexData({"path": str(tmp_path / "b.parquet"), "time_column": "SlotUtc", "price": "Close",
                 "delay": {"rows": 2, "except": ["Close"]}}, tmp_path)
    rows = d.rows()
    assert rows["Close"].to_list()[:3] == [100, 101, 102] and rows["GEX"].to_list()[:3] == [None, None, 0.0]
    out = d.export(str(tmp_path / "x.parquet"), str(t[4]).replace("T", " ")[:19])
    assert out["rows"] == 4
    assert pl.read_parquet(tmp_path / "x.parquet")["t"].cast(pl.Int64).max() < t[4].astype(np.int64)
    assert (tmp_path / ".cache").is_dir()                                 # cached beside its config


def test_a_signed_target_is_valued_in_its_own_units():
    """A target that crosses zero (e.g. GEX) has no returns: P&L = position x change, summed."""
    k = _keys(days=3, per_day=5)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0] * 3)                     # rises through zero every day
    pos = np.ones(len(k))
    gross, net = P.bar_returns(x, pos, 0.0, additive=True)
    assert gross.tolist()[:5] == [0.0, 1.0, 1.0, 1.0, 1.0]            # held long into each +1 step
    tr = P.trades(k, x, P.manage(k, k, pos, P.Rules(flat_each_day=True)), 0.0, additive=True)
    assert np.allclose(tr["gross"], 4.0)          # held into bars 1-4 (the bar-3 decision earns bar 4), flat after
    ev = V.value(k, x, P.manage(k, k, pos, P.Rules(flat_each_day=True)), 2**62,
                 {"additive": True, "cost_bps": 0, "min_active_days": 1})
    assert ev["curve_kind"] == "additive" and [v for _, v in ev["curve"]] == [4.0, 4.0, 4.0]
    assert ev["diagnostics"]["full" if "full" in ev["diagnostics"] else "in_sample"]["segment_matching"] is None
