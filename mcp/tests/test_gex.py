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
    pos = P.manage(k, t, np.array([5.0, -1.0, 2.0]), P.Rules(max_position=3, holding="intraday"))
    # 5 is clipped to 3; the last bar of each day is flat; day 2 starts with the carried -1 (a new trade).
    assert pos.tolist() == [3, 3, 3, -1, -1, 0, -1, 2, 2, 2, 2, 0]
    longonly = P.manage(k, t, np.array([5.0, -1.0, 2.0]), P.Rules(max_position=3, direction="long"))
    assert longonly.min() == 0


def test_order_mode_fills_buys_and_sells_within_bounds():
    k = _keys()
    # buy 2, buy 2 (only 1 fills at max 3), sell 5 (to -2), next day: buy 1 from flat.
    t, a = k[[0, 1, 3, 8]], np.array([2.0, 2.0, -5.0, 1.0])
    pos = P.manage(k, t, a, P.Rules(mode="order", max_position=3, holding="intraday"))
    assert pos.tolist() == [2, 3, 3, -2, -2, 0, 0, 0, 1, 1, 1, 0]
    # without the close rule the position carries over night
    pos = P.manage(k, t, a, P.Rules(mode="order", max_position=3))
    assert pos.tolist() == [2, 3, 3, -2, -2, -2, -2, -2, -1, -1, -1, -1]


def test_trades_and_valuation_of_a_known_path():
    k = _keys(days=30, per_day=40)
    rng = np.random.default_rng(5)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(k))))
    acts = np.sign(np.sin(np.arange(len(k)) / 7))
    pos = P.manage(k, k, acts, P.Rules(max_position=1, holding="intraday"))
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
    tr = P.trades(k, x, P.manage(k, k, pos, P.Rules(holding="intraday")), 0.0, additive=True)
    assert np.allclose(tr["gross"], 4.0)          # held into bars 1-4 (the bar-3 decision earns bar 4), flat after
    ev = V.value(k, x, P.manage(k, k, pos, P.Rules(holding="intraday")), 2**62,
                 {"additive": True, "cost_bps": 0, "min_active_days": 1})
    assert ev["curve_kind"] == "additive" and [v for _, v in ev["curve"]] == [4.0, 4.0, 4.0]
    assert ev["diagnostics"]["full" if "full" in ev["diagnostics"] else "in_sample"]["segment_matching"] is None


@pytest.mark.parametrize("rule,trades,longest_days", [("open", 1, 6), ("intraday", 6, 1), ("max_2_days", 3, 2),
                                                      ("max_3_days", 2, 3), ("max_10_days", 1, 6)])
def test_holding_rules_cap_how_many_days_a_trade_spans(rule, trades, longest_days):
    k = _keys(days=6, per_day=3)
    pos = P.manage(k, k, np.ones(len(k)), P.Rules(holding=rule))          # wants to be long the whole time
    tr = P.trades(k, np.linspace(100, 101, len(k)), pos, 0.0)
    spans = [len(np.unique(k[a:b + 1] // (86_400 * NS))) for a, b in zip(tr["entry_i"], tr["exit_i"] - 1)]
    assert len(tr["side"]) == trades and max(spans) == longest_days
    assert P.Rules(holding=rule).describe()["action_rule"] == rule


def test_holding_rules_for_orders_and_bad_rules():
    k = _keys(days=6, per_day=3)
    pos = P.manage(k, k[[0]], np.array([2.0]), P.Rules(mode="order", max_position=3, holding="max_2_days"))
    assert pos.tolist() == [2, 2, 2, 2, 2, 0] + [0] * 12                    # an order is not repeated: flat after
    for bad in ("forever", "max_0_days", "max_x_days"):
        with pytest.raises(ValueError):
            P.parse_holding(bad)


def test_an_open_trade_counts_every_bar_it_was_held():
    keys = np.arange(10, dtype=np.int64) * 10_000_000_000
    price = np.linspace(100, 101, 10)
    pos = np.r_[np.zeros(2), np.ones(3), -np.ones(5)]                  # a closed long, then a short still open
    tr = P.trades(keys, price, pos, 0.0)
    assert list(tr["bars"]) == [3, 5] and list(tr["open"]) == [False, True]


@pytest.mark.parametrize("direction, want", [("both", [-2.0, 0.0, 1.5]), ("long", [0.0, 0.0, 1.5]),
                                             ("short", [-2.0, 0.0, 0.0])])
def test_direction_limits_the_sides_trades_take(direction, want):
    r = P.Rules.from_cfg({"mode": "target", "max_position": 3, "holding": "open"}, None, direction)
    keys = np.arange(3, dtype=np.int64) * 10_000_000_000
    assert list(P.manage(keys, keys, np.array([-2.0, 0.0, 1.5]), r)) == want
    allowed = {a["name"] for a in r.describe()["allowed"]}
    assert ("open_short" in allowed) == (direction != "long") and ("open_long" in allowed) == (direction != "short")
    assert {"close", "hold"} <= allowed and r.describe()["direction"] == direction
    with pytest.raises(ValueError):
        P.Rules.from_cfg({}, None, "sideways")


def _trend_days(days=12, per_day=60, move=0.01, seed=3):
    """Days that trend steadily from the open, alternating up and down, with a little noise."""
    k = _keys(days=days, per_day=per_day)
    rng = np.random.default_rng(seed)
    price, p = [], 100.0
    for d in range(days):
        step = (1 if d % 2 == 0 else -1) * move / per_day
        for _ in range(per_day):
            p *= 1 + step + rng.normal(0, 2e-4)
            price.append(p)
    return k, np.array(price)


def test_trend_diagnostics_reward_holding_the_days_move_and_flag_early_exits():
    k, price = _trend_days()
    day = k // (86_400 * NS)
    first = np.r_[True, day[1:] != day[:-1]]
    up = np.array([1.0 if d % 2 == 0 else -1.0 for d in range(12)])[np.cumsum(first) - 1]
    rules = P.Rules(max_position=1, holding="intraday")
    cfg = {"cost_bps": 0.5, "min_active_days": 1}
    hold = P.manage(k, k, up, rules)                                       # the right side, all day
    ev = V.value(k, price, hold, 2**62, cfg)["diagnostics"]["in_sample"]
    assert ev["trend_capture"] > 0.8 and ev["day_bps"] > 0.8 * ev["oracle_day_bps"] - 20
    assert ev["random_entry_pctile"] >= 0.95                              # far better than random trades
    # The same side, but out after 5 bars each day: the move keeps going after the exit.
    early = up * (np.arange(len(k)) % 60 < 5)
    ev2 = V.value(k, price, P.manage(k, k, early, rules), 2**62, cfg)
    d2 = ev2["diagnostics"]["in_sample"]
    assert d2["trend_capture"] < 0.2 and d2["after_exit_bps"] > 50 and d2["exit_continued_share"] > 0.9
    assert "cut winners short" in ev2["notes"] and "Trend capture" in ev2["notes"]


def test_trend_notes_use_only_the_in_sample_segment():
    k, price = _trend_days(days=12)
    day = k // (86_400 * NS)
    hold_ns = int(np.unique(day)[8] * 86_400 * NS)                      # the last 4 days are the holdout
    wrong = -np.array([1.0 if d % 2 == 0 else -1.0 for d in range(12)])[np.searchsorted(np.unique(day), day)]
    pos = P.manage(k, k, np.where(day >= hold_ns // (86_400 * NS), -wrong, wrong), P.Rules(max_position=1, holding="intraday"))
    ev = V.value(k, price, pos, hold_ns, {"cost_bps": 0.5, "min_active_days": 1})
    ins, ho = ev["diagnostics"]["in_sample"], ev["diagnostics"]["holdout"]
    assert ins["trend_capture"] < 0 < ho["trend_capture"]
    assert f"{ins['trend_capture']:+.1%}" in ev["notes"] and f"{ho['trend_capture']:+.1%}" not in ev["notes"]


def test_trade_limit_ignores_entries_after_the_days_allowance_but_never_exits():
    k = _keys(days=2, per_day=12)
    #          long    flat  short  (flip) long   flat  short  flat | day 2: long again
    acts = np.array([1, 1, 0, -1, -1, 1, 1, 0, -1, -1, 0, 0,  1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], float)
    r = P.Rules.from_cfg({"max_position": 1, "holding": "intraday"}, "intraday+max_3_trades")
    assert (r.holding, r.max_trades_per_day, r.rule) == ("intraday", 3, "intraday+max_3_trades")
    pos = P.manage(k, k, acts, r)
    # trades 1-3 (long, short, long via the flip) go through; the 4th (short at bar 8) is held flat;
    # the next day has its own allowance.
    assert pos[:12].tolist() == [1, 1, 0, -1, -1, 1, 1, 0, 0, 0, 0, 0]
    assert pos[12:14].tolist() == [1, 1]
    assert len(P.trades(k, np.linspace(100, 101, len(k)), pos, 0.0)["side"]) == 4
    with pytest.raises(ValueError):
        P.split_rule("intraday+max_0_trades")
    assert P.split_rule("max_2_days+max_6_trades") == ("max_2_days", 6)


def test_fills_come_a_bar_after_the_decision_and_never_leak_overnight():
    k = _keys(days=2, per_day=6)
    acts = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], float)            # always wants to be long
    r = P.Rules.from_cfg({"max_position": 1, "holding": "intraday", "fill_delay_bars": 1})
    pos = P.manage(k, k, acts, r)
    assert pos.tolist() == [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0]             # filled next bar, flat at each close
    price = np.array([100, 101, 102, 103, 104, 105, 200, 201, 202, 203, 204, 205], float)
    gross, _ = P.bar_returns(price, pos, 0.0)
    assert gross[6] == 0.0                                                   # the overnight jump earns nothing
    undelayed = P.manage(k, k, acts, P.Rules.from_cfg({"max_position": 1, "holding": "intraday"}))
    assert P.bar_returns(price, undelayed, 0.0)[0].sum() > gross.sum()     # a bar later costs the first move
