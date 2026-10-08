"""The sandbox helper `ft` (ui/sandbox/ft.py): load(prefix=...) and forecast().

ft runs inside the candidate sandbox, where pandas is installed; these tests skip without it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

FT_PATH = Path(__file__).resolve().parents[2] / "sandbox" / "ft.py"


@pytest.fixture
def ft(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("ft_under_test", FT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    work = tmp_path / ".ft"
    work.mkdir()
    monkeypatch.setattr(mod, "_FT", str(work))
    monkeypatch.setattr(mod, "_REQUESTS", str(work / "forecast_requests.json"))
    monkeypatch.setattr(mod, "_CATALOG", [])
    mod._root = tmp_path
    return mod


def _add_feature(ft, name: str, frame) -> None:
    frame.to_parquet(ft._root / f"{name}.parquet")
    ft._CATALOG.append({"view": f"fc_{name}", "path": f"{name}.parquet", "format": "parquet", "root": str(ft._root)})


def _fc_frame(times, medians):
    return pd.DataFrame({"t": pd.to_datetime(times), "last": [1.0] * len(times), "fc_median": medians,
                         "fc_q10": medians, "fc_q90": medians})


def test_load_prefix_renames_all_but_t(ft):
    _add_feature(ft, "a", _fc_frame(["2024-01-01 10:00"], [2.0]))
    cols = list(ft.load("fc_a", prefix="a_").columns)
    assert cols == ["t", "a_last", "a_fc_median", "a_fc_q10", "a_fc_q90"]
    assert list(ft.load("fc_a").columns)[1] == "last"  # no prefix: unchanged


def test_recipe_name_is_stable_and_order_independent(ft):
    r1 = {"column": "GEX", "horizon": 6, "covariates": ["A", "B"], "model": "m"}
    r2 = {"model": "m", "covariates": ["A", "B"], "horizon": 6, "column": "GEX"}
    assert ft.recipe_name(r1) == ft.recipe_name(r2)
    assert ft.recipe_name(r1).startswith("auto_")
    assert ft.recipe_name(r1) != ft.recipe_name({**r1, "horizon": 7})
    # Keys outside the recipe (e.g. join options) never change the name.
    assert ft.recipe_name(r1) == ft.recipe_name({**r1, "prefix": "x_"})


def test_forecast_not_built_records_request_once_and_raises(ft):
    for _ in range(2):
        with pytest.raises(ft.ForecastPending):
            ft.forecast("GEX", inputs=["Pressure_Total"], horizon=6, model="amazon/chronos-2")
    asked = json.loads(Path(ft._REQUESTS).read_text(encoding="utf-8"))
    assert len(asked) == 1
    assert asked[0]["recipe"]["covariates"] == ["Pressure_Total"]
    assert asked[0]["name"] == ft.recipe_name(asked[0]["recipe"])


def test_forecast_needs_a_series(ft):
    with pytest.raises(ValueError):
        ft.forecast()


def test_forecast_join_is_backward_and_prefixed(ft):
    recipe_kwargs = {"horizon": 6, "model": "amazon/chronos-2"}
    with pytest.raises(ft.ForecastPending):
        ft.forecast("GEX", **recipe_kwargs)
    name = json.loads(Path(ft._REQUESTS).read_text(encoding="utf-8"))[0]["name"]
    _add_feature(ft, name, _fc_frame(["2024-01-01 10:00:00", "2024-01-01 10:00:20"], [1.0, 2.0]))

    df = pd.DataFrame({"SlotUtc": pd.to_datetime(["2024-01-01 10:00:10", "2024-01-01 09:59:50",
                                                  "2024-01-01 10:00:30"]), "x": [1, 2, 3]})
    out = ft.forecast("GEX", join=df, time_col="SlotUtc", **recipe_kwargs)
    # Original row order kept; each row sees only the forecast made at or before it.
    assert list(out["x"]) == [1, 2, 3]
    med = list(out["GEX_fc_median"])
    assert med[0] == 1.0          # 10:00:10 -> forecast made at 10:00:00, never the 10:00:20 one
    assert pd.isna(med[1])        # 09:59:50 -> no forecast existed yet
    assert med[2] == 2.0
    assert "GEX_t" in out.columns and "t" not in out.columns


def test_two_joined_forecasts_do_not_clash(ft):
    for col, val in (("GEX", 1.0), ("Imb", 5.0)):
        with pytest.raises(ft.ForecastPending):
            ft.forecast(col, horizon=3)
    for entry in json.loads(Path(ft._REQUESTS).read_text(encoding="utf-8")):
        v = 1.0 if entry["recipe"]["column"] == "GEX" else 5.0
        _add_feature(ft, entry["name"], _fc_frame(["2024-01-01 10:00"], [v]))
    df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01 10:01"])})
    out = ft.forecast("Imb", horizon=3, join=ft.forecast("GEX", horizon=3, join=df, time_col="ts"), time_col="ts")
    assert out.loc[0, "GEX_fc_median"] == 1.0
    assert out.loc[0, "Imb_fc_median"] == 5.0


def test_loads_are_recorded_for_the_scoreboard(ft):
    _add_feature(ft, "a", _fc_frame(["2024-01-01 10:00"], [2.0]))
    ft.load("fc_a")
    ft.load("fc_a", prefix="x_")
    assert json.loads((Path(ft._FT) / "used.json").read_text(encoding="utf-8")) == ["fc_a"]


# ---------------------------------------------------------------------------------------
# Intraday trend tools
# ---------------------------------------------------------------------------------------
def _session_rows(days=30, seed=0, step_s=60, drift=None):
    """1-minute rows 9:30-16:00 New York on consecutive weekdays (timestamps naive UTC, as ft.rows
    gives them), a random walk with an optional per-day drift."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out_t, out_p, p = [], [], 100.0
    for i, d in enumerate(pd.bdate_range("2024-02-26", periods=days)):          # spans the March DST change
        local = pd.date_range(d + pd.Timedelta("9h30min"), d + pd.Timedelta("15h59min"), freq=f"{step_s}s",
                              tz="America/New_York")
        mu = 0.0 if drift is None else drift[i % len(drift)]
        for ts in local.tz_convert("UTC").tz_localize(None):
            p *= 1 + mu + rng.normal(0, 4e-4)
            out_t.append(ts)
            out_p.append(p)
    return pd.DataFrame({"t": pd.to_datetime(out_t), "Close": out_p, "Volume": 100.0,
                         "GEX": np.sin(np.arange(len(out_p)) / 500.0)})


def test_clock_is_new_york_time_across_daylight_saving(ft):
    import numpy as np

    rows = _session_rows(days=15)
    session, minute = ft.clock(rows["t"])
    firsts = minute[np.r_[True, session[1:] != session[:-1]]]
    assert set(firsts) == {570.0}                          # 9:30 every day, before and after the change
    assert len(set(session)) == 15


def test_a_helper_called_with_a_wrong_keyword_names_the_right_one(ft):
    """The night of 2026-09-30 lost seven submissions to trend_exits(atr_lookback= / start_time= /
    size_func= / base= / close=): the error has to say what the helper does take."""
    import numpy as np

    rows = pd.DataFrame({"t": pd.date_range("2024-06-03 13:30", periods=50, freq="1min"), "Close": 100.0})
    for bad, good in (("lookback", "vol_window"), ("start_time", "no_entry_before"), ("size_func", "size"),
                      ("close", "price"), ("stop_multt", "stop_mult")):
        with pytest.raises(TypeError) as err:
            ft.trend_exits(np.ones(50), rows, **{bad: 1})
        msg = str(err.value)
        assert f"unexpected keyword argument '{bad}'" in msg and f"did you mean '{good}'" in msg, msg
        assert "ft.trend_exits(entries, rows, price" in msg and "vol_window" in msg
    with pytest.raises(TypeError, match="missing a required argument: 'rows'"):
        ft.trend_exits(np.ones(50))
    # a call that binds runs as before, polars in or not
    assert len(ft.trend_exits(np.ones(50), rows, vol_window=10)) == 50


def test_trend_exits_trails_the_best_price_and_limits_trades(ft):
    import numpy as np

    t = pd.date_range("2024-06-03 13:30", periods=200, freq="1min")          # 9:30 New York (summer)
    p = np.r_[np.linspace(100, 102, 100), np.linspace(102, 100.5, 100)]      # up, then a pullback
    rows = pd.DataFrame({"t": t, "Close": p})
    e = np.zeros(200)
    e[20] = 1                                                                # one long signal at 9:50
    pos = ft.trend_exits(e, rows, stop_pct=0.5, breakeven_at=None).to_numpy()
    assert pos[19] == 0 and pos[20] == 1                                     # opens on the signal
    out = int(np.flatnonzero((pos[:-1] == 1) & (pos[1:] == 0))[0]) + 1
    assert 100 < out < 200 and p[out] <= 102 * (1 - 0.005) + 1e-9           # exit 0.5% under the high
    # A persistent signal does not re-enter after the stop; a new one does, up to the daily limit.
    e2 = np.zeros(200)
    e2[20:] = 1
    assert ft.trend_exits(e2, rows, stop_pct=0.5, breakeven_at=None).to_numpy()[out:].max() == 0
    flips = np.tile([1, -1], 100)
    many = ft.trend_exits(flips, rows, stop_pct=5, max_trades_per_day=3).to_numpy()
    assert (np.diff(np.sign(many)) != 0).sum() <= 4                          # 3 entries (and at most one exit)


def test_trend_tools_are_causal_under_truncation(ft):
    import numpy as np

    rows = _session_rows(days=30, drift=[3e-5, -3e-5, 0.0])
    fns = {
        "noise_area": lambda r: ft.noise_area_breakout(r, lookback_days=5),
        "gated": lambda r: ft.noise_area_breakout(r, lookback_days=5, gate=ft.gamma_regime(r, smooth=60) < 0),
        "trend_exits": lambda r: ft.trend_exits(np.sign(r["Close"].diff(30).fillna(0)).to_numpy(), r, vol_window=60),
        "vwap": lambda r: ft.session_vwap(r),
    }
    for name, fn in fns.items():
        full = fn(rows).to_numpy()
        assert np.isfinite(full).all(), name
        for frac in (0.35, 0.62, 0.9):
            n = int(len(rows) * frac) + 7
            part = fn(rows.iloc[:n].reset_index(drop=True)).to_numpy()
            assert np.array_equal(part, full[:n]), f"{name} changed an earlier value when cut at row {n}"
    na = fns["noise_area"](rows).to_numpy()
    assert (na != 0).any() and (na > 0).any() and (na < 0).any()


def test_decision_points_and_outcomes(ft):
    rows = _session_rows(days=3)
    pts = ft.decision_points(rows, times=["10:00", "15:00"])
    _, minute = ft.clock(rows["t"].iloc[pts])
    assert len(pts) == 6 and set(minute) == {600.0, 900.0}
    lab = ft.label_outcomes(rows, pts, horizons=(30,), barrier=(0.1, 0.1))
    last = rows.groupby(rows["t"].dt.normalize())["Close"].transform("last").to_numpy()[pts]
    assert abs(lab["ret_close_bps"].to_numpy() - (last / rows["Close"].to_numpy()[pts] - 1) * 1e4).max() < 1e-9
    assert (lab["mfe_bps"] >= lab["mae_bps"]).all() and set(lab["barrier"]) <= {-1.0, 0.0, 1.0}


def test_meta_filter_learns_only_from_closed_sessions(ft):
    import numpy as np

    rng = np.random.default_rng(1)
    sessions = np.repeat(np.arange(60), 5)
    X = rng.normal(size=(300, 2))
    y = (X[:, 0] > 0).astype(float)
    prob = ft.meta_filter(X, y, sessions, min_train_sessions=20, refit_every=1)
    assert np.isnan(prob[sessions < 20]).all() and np.isfinite(prob[sessions >= 20]).all()
    assert ((prob[sessions >= 20] > 0.5) == (y[sessions >= 20] == 1)).mean() > 0.9     # it learns
    # Scrambling the labels of session 40 onwards cannot change any prediction for session <= 40.
    y2 = y.copy()
    y2[sessions >= 40] = 1 - y2[sessions >= 40]
    prob2 = ft.meta_filter(X, y2, sessions, min_train_sessions=20, refit_every=1)
    keep = sessions <= 40
    assert np.allclose(np.nan_to_num(prob[keep], nan=-1), np.nan_to_num(prob2[keep], nan=-1))


def test_admit_takes_at_most_per_day_in_time_order_and_only_learns_its_bar_from_the_past(ft):
    import numpy as np

    rng = np.random.default_rng(2)
    sessions = np.repeat(np.arange(30), 10)
    scores = rng.random(300)
    take = ft.admit(scores, sessions, per_day=3, window_sessions=10, min_sessions=5)
    per = np.bincount(sessions[take], minlength=30)
    assert per[:5].sum() == 0 and per.max() <= 3 and 1.5 < per[5:].mean() <= 3
    # A later, better candidate never displaces an earlier one that already cleared the bar.
    s2 = scores.copy()
    s2[sessions == 20] = np.r_[np.full(3, 0.99), np.full(7, 1.0)]
    t2 = ft.admit(s2, sessions, per_day=3, window_sessions=10, min_sessions=5)
    assert np.flatnonzero(t2 & (sessions == 20)).tolist() == list(np.flatnonzero(sessions == 20)[:3])
    # Changing today's and later scores never changes an earlier decision.
    s3 = scores.copy()
    s3[sessions >= 20] = 0.0
    assert np.array_equal(ft.admit(s3, sessions, per_day=3, window_sessions=10, min_sessions=5)[sessions < 20],
                          take[sessions < 20])
    assert not ft.admit(np.full(300, np.nan), sessions).any()


def _task_rows(ft, tmp_path, monkeypatch, n=5):
    task = tmp_path / "task"
    task.mkdir()
    t = pd.date_range("2024-01-02 14:30", periods=n, freq="10s")
    pd.DataFrame({"t": t[::-1], "Close": range(n)}).to_parquet(task / "rows.parquet")   # stored unsorted
    monkeypatch.setattr(ft, "_TASK", str(task))
    return t


def test_report_actions_takes_values_keyword_and_row_numbered_actions(ft, tmp_path, monkeypatch):
    import numpy as np

    t = _task_rows(ft, tmp_path, monkeypatch)
    out = Path(ft._FT) / "actions.parquet"
    ft.report_actions(values=[0, 1, 1, -1, 0], t=t)                 # the keyword the docstrings suggest
    assert pd.read_parquet(out)["pos"].tolist() == [0, 1, 1, -1, 0]
    ft.report_actions(np.array([1, 0, 0, 0, -1.0]))                 # no times: one per row, in rows() order
    got = pd.read_parquet(out)
    assert got["t"].tolist() == list(t) and got["pos"].tolist() == [1, 0, 0, 0, -1]
    ft.report_actions(pd.Series([0, 0, 1, 1, 1]))                    # a RangeIndex series, same
    assert pd.read_parquet(out)["pos"].tolist() == [0, 0, 1, 1, 1]
    with pytest.raises(ValueError, match="RangeIndex"):
        ft.report_actions(pd.Series([1, 0, 1]))                      # not one per row: still refused
    with pytest.raises(TypeError, match="once"):
        ft.report_actions([1] * 5, values=[1] * 5)


def test_pandas_compat_restores_fillna_method(ft, monkeypatch):
    import inspect

    for cls in (pd.Series, pd.DataFrame):                            # pandas 3: fillna without `method`
        orig = cls.fillna
        if "method" in inspect.signature(orig).parameters:
            def no_method(self, value=None, *, axis=None, inplace=False, limit=None, _o=orig):
                return _o(self, value, axis=axis, inplace=inplace, limit=limit)
            monkeypatch.setattr(cls, "fillna", no_method)
        else:
            monkeypatch.setattr(cls, "fillna", orig)                 # restored after the test
    ft._pandas_compat()
    s = pd.Series([1.0, None, None, 4.0])
    assert s.fillna(method="ffill").tolist() == [1, 1, 1, 4]
    assert s.fillna(method="bfill", limit=1).fillna(0).tolist() == [1, 0, 4, 4]
    assert s.fillna(0).tolist() == [1, 0, 0, 4]
    df = pd.DataFrame({"a": [1.0, None]})
    df.fillna(method="pad", inplace=True)
    assert df["a"].tolist() == [1, 1]


def test_a_column_listed_twice_is_loaded_once(ft, tmp_path, monkeypatch):
    """Models repeat a column in long lists (['IV_AtmD0', ..., 'IV_AtmD0']); polars refused that
    with a DuplicateError deep in its planner and the experiment was lost (bug #63)."""
    pl = pytest.importorskip("polars")
    task = tmp_path / "task"
    task.mkdir()
    pd.DataFrame({"t": pd.date_range("2024-01-02 14:30", periods=3, freq="10s"), "A": [1.0, 2.0, 3.0],
                  "B": [4.0, 5.0, 6.0]}).to_parquet(task / "rows.parquet")
    monkeypatch.setattr(ft, "_TASK", str(task))
    assert ft.rows_pl(columns=["t", "A", "B", "A"]).columns == ["t", "A", "B"]
    assert list(ft.rows(columns=["A", "t", "A"]).columns) == ["t", "A"]
    (ft._root / "d.parquet").write_bytes((task / "rows.parquet").read_bytes())
    ft._CATALOG.append({"view": "d", "path": "d.parquet", "format": "parquet", "root": str(ft._root)})
    assert ft.load_pl("d", columns=["A", "A", "t"]).columns == ["A", "t"]
    assert list(ft.load("d", columns=["B", "B"]).columns) == ["B"]


def test_size_takes_one_direction_for_every_bar_and_names_a_length_mismatch(ft):
    """Bugs #88/#95: ft.size(1, scale) made the direction a one-row series and died in numpy's
    broadcast ("remapped shapes (3,) and requested shape (1,)") -- an error naming neither argument."""
    inv = pd.Series([0.5, 1.0, 2.0], index=pd.date_range("2024-01-02 14:30", periods=3, freq="10s"))
    pos = ft.size(1, inv, base=2.0, max_leverage=3.0)
    # One trade, opened on the first bar at base * scale and held (rebalance="entry").
    assert pos.tolist() == [1.0, 1.0, 1.0] and pos.index.equals(inv.index)
    assert ft.size(-1, [1.0, 1.0], base=2.0).tolist() == [-2.0, -2.0]
    assert ft.size(pd.Series([1, 0, -1]), inv, base=2.0).tolist() == [1.0, 0.0, -4.0]     # per bar, as before
    assert ft.size(pd.Series([1, -1]), 1.5).tolist() == [1.5, -1.5]                        # one number
    with pytest.raises(ValueError, match="direction has 2 values but scale has 3"):
        ft.size(pd.Series([1, -1]), inv, base=2.0, max_leverage=3.0)


def test_resample_takes_a_frame_indexed_by_its_time_column_or_holding_nothing_else(ft):
    """Two frames agents really passed, each of which died inside pandas: bar.index = bar[tc]
    ("'SlotUtc' is both an index level and a column label") and a frame with the time column
    alone ("No objects to concatenate")."""
    t = pd.date_range("2024-01-02 14:30", periods=6, freq="10s")
    bar = pd.DataFrame({"SlotUtc": t, "Close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    want = ft.resample(bar, "30s")
    assert want["Close"].tolist() == [3.0, 6.0] and want["SlotUtc"].tolist() == [t[2], t[5]]
    indexed = bar.copy()
    indexed.index = pd.to_datetime(indexed["SlotUtc"])
    pd.testing.assert_frame_equal(ft.resample(indexed, "30s"), want)
    only_t = ft.resample(bar[["SlotUtc"]], "30s")
    assert list(only_t.columns) == ["SlotUtc", "bar_rows"] and only_t["bar_rows"].tolist() == [3, 3]


def test_row_numbered_positions_are_refused_with_the_way_out(ft):
    """Bug #104: a library signal(df) returns positions "indexed like df" -- row numbers -- and
    report_positions refused them without naming `t=`, the one-argument fix."""
    t = pd.date_range("2024-01-02 14:30", periods=3, freq="10s")
    pos = pd.Series([0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match=r"ft\.report_positions\(pos, t=df\[time_col\]\)"):
        ft.report_positions(pos)
    ft.report_positions(pos, t=t)
    got = pd.read_parquet(Path(ft._FT) / "positions.parquet")
    assert got["t"].tolist() == list(t) and got["pos"].tolist() == [0.0, 1.0, 1.0]


def test_report_with_a_positions_series_reports_the_positions(ft, monkeypatch):
    """Bug #6: ft.report(positions=pos) kept str(pos)[:200] as an "extra number" and reported no
    positions, so the submission failed with "no positions reported". A series under
    positions= / returns= now goes to report_positions / report_returns; numbers stay extras."""
    monkeypatch.setattr(ft, "_RESULT", str(Path(ft._FT) / "result.json"))
    t = pd.date_range("2024-01-02 14:30", periods=3, freq="10s")
    ft.report(positions=pd.Series([0.0, 1.0, -1.0], index=t), trades=2, positions_note="flip")
    got = pd.read_parquet(Path(ft._FT) / "positions.parquet")
    assert got["pos"].tolist() == [0.0, 1.0, -1.0]
    extra = json.loads((Path(ft._FT) / "result.json").read_text(encoding="utf-8"))["extra"]
    assert extra == {"trades": 2.0, "positions_note": "flip"}
    ft.report(positions=12)                                  # a count, not a series: an extra number
    assert json.loads((Path(ft._FT) / "result.json").read_text(encoding="utf-8"))["extra"] == {"positions": 12.0}


# ---------------------------------------------------------------------------------------
# Fast research: quick_score, sweep, direction_scan
# ---------------------------------------------------------------------------------------
def test_quick_score_charges_costs_fills_a_bar_late_and_closes_each_day(ft):
    import numpy as np

    rows = _session_rows(days=20, drift=[3e-5, -3e-5])       # up days and down days in turn
    session, minute = ft.clock(rows["t"])
    day_sign = np.where(np.unique(session, return_inverse=True)[1] % 2 == 0, 1.0, -1.0)
    oracle = np.where(minute >= 600, day_sign, 0.0)          # the day's direction from 10:00
    q = ft.quick_score(oracle, rows, cost_bps=2.0)
    assert q["sharpe"] > 5 and q["sharpe_flipped"] < 0 and q["sharpe_gross"] > q["sharpe"]
    assert q["trades_per_day"] == 1.0 and q["long_share"] == q["short_share"] == 0.5
    assert not q["floors_ok"]                                # one trade a day is under the floor
    assert np.isfinite(q["worst_half"]) and q["days"] == 20 and q["active_share"] == 1.0
    flat = ft.quick_score(np.zeros(len(rows)), rows)
    assert flat["trades_per_day"] == 0 and not flat["floors_ok"] and flat["active_share"] == 0.0
    first_days = np.where(np.unique(session, return_inverse=True)[1] < 3, oracle, 0.0)
    sparse = ft.quick_score(first_days, rows)
    assert sparse["active_days"] == 3 and sparse["active_share"] == 0.15 and not sparse["floors_ok"]
    with pytest.raises(ValueError, match="one per row"):
        ft.quick_score(oracle[:-1], rows)


def test_sweep_ranks_variants_by_their_weaker_half(ft):
    import numpy as np

    rows = _session_rows(days=20, drift=[3e-5, -3e-5])
    session, minute = ft.clock(rows["t"])
    day_sign = np.where(np.unique(session, return_inverse=True)[1] % 2 == 0, 1.0, -1.0)

    def make(start, flip):
        return np.where(minute >= start, day_sign * flip, 0.0)

    table = ft.sweep(make, {"start": [600, 840], "flip": [1, -1]}, rows)
    top = table.row(0, named=True) if hasattr(table, "row") else table.iloc[0].to_dict()
    assert (top["start"], top["flip"]) == (600, 1)
    with pytest.raises(ValueError, match="at most"):
        ft.sweep(make, {"start": list(range(10)), "flip": list(range(10))}, rows)


def test_direction_scan_finds_a_field_that_tells_the_days_direction(ft):
    import numpy as np

    rows = _session_rows(days=80, drift=[4e-5, -4e-5, 4e-5, 4e-5, -4e-5])
    session, _ = ft.clock(rows["t"])
    idx = np.unique(session, return_inverse=True)[1]
    rows["Tell"] = np.where(np.array([4e-5, -4e-5, 4e-5, 4e-5, -4e-5])[idx % 5] > 0, 1.0, -1.0)
    rows["Noise"] = np.random.default_rng(1).normal(size=len(rows))
    tab = ft.direction_scan(rows, fields=["Tell", "Noise"], times=["10:00", "11:00"], cost_bps=0.5)
    tab = tab.to_pandas() if hasattr(tab, "to_pandas") else tab
    best = tab.iloc[0]
    assert best["field"] == "Tell" and best["best"] == "follow" and best["worst_half_bps"] > 0
    assert set(tab["field"]) >= {"Tell", "Noise", "ret_since_open", "gap"}


def test_a_polars_expression_is_evaluated_on_the_rows_beside_it(ft):
    """trend_exits(pl.when(...).then(1).otherwise(-1), rows) died in float() -- 'not Expr' (bug #182,
    Muse-Glimmer 10-01 11:20); it now means what it says. start_bar/start_minute name first_check."""
    import numpy as np
    pl = pytest.importorskip("polars")

    t = pd.date_range("2024-06-03 13:30", periods=200, freq="1min")
    rows = pl.DataFrame({"t": t, "Close": np.r_[np.linspace(100, 102, 100), np.linspace(102, 100.5, 100)]})
    e = pl.when(pl.col("Close") > pl.col("Close").shift(1)).then(1).otherwise(-1)
    want = ft.trend_exits(rows.select(e).to_series().to_numpy(), rows, stop_pct=0.5).to_numpy()
    assert (ft.trend_exits(e, rows, stop_pct=0.5).to_numpy() == want).all()
    assert (ft.trend_exits(e, rows.to_pandas(), stop_pct=0.5).to_numpy() == want).all()
    with pytest.raises(TypeError, match="no frame to evaluate it on"):
        ft.inverse_vol(pl.col("Close"))
    for bad in ("start_bar", "start_minute"):
        with pytest.raises(TypeError, match="did you mean 'first_check'"):
            ft.noise_area_breakout(rows, **{bad: 3})
    with pytest.raises(TypeError, match="did you mean 'no_entry_before'"):
        ft.trend_exits(np.ones(200), rows, start_time="10:00")


def test_expr_where_keeps_the_column_length_as_pandas_where_does(ft):
    """Qwen3.6 10-04: rows.with_columns(pl.col('minute').where(pl.col('sig') != 0).forward_fill())
    died -- polars' where is filter. Now: the value where the condition holds, null (or other) elsewhere."""
    pl = pytest.importorskip("polars")

    df = pl.DataFrame({"minute": [1, 2, 3, 4], "sig": [0, 1, 0, 1]})
    out = df.with_columns(pl.col("minute").where(pl.col("sig") != 0).forward_fill().alias("m"))
    assert out["m"].to_list() == [None, 2, 2, 4]
    assert df.select(pl.col("minute").where(pl.col("sig") != 0, 0))["minute"].to_list() == [0, 2, 0, 4]
    assert df.select(pl.col("minute").where(pl.col("sig") != 0).sum()).item() == 6


def test_boolean_series_with_nulls_converts_to_a_bool_array(ft):
    """Muse 10-07 (candidate c3c03b51c8): (rows['Doi_MultiSlope'] >= 0.14).to_numpy() over a column
    with nulls was an object array holding None, and `(sig == 1) & long1` died. A null condition is
    False, as a NaN comparison is in numpy; other dtypes keep their own conversion."""
    import numpy as np
    pl = pytest.importorskip("polars")

    s = pl.Series([0.2, None, 0.1])
    for arr in ((s >= 0.14).to_numpy(), np.asarray(s >= 0.14)):
        assert arr.dtype == bool and arr.tolist() == [True, False, False]
    assert ((np.array([1, 1, 0]) == 1) & (s >= 0.14).to_numpy()).tolist() == [True, False, False]
    assert np.isnan(s.to_numpy()[1])


def test_lookback_means_lookback_days_on_the_noise_area_and_is_explained_elsewhere(ft, capsys):
    """2026-10-04: ft.noise_area_breakout(rows, lookback=14) was refused though its window is
    lookback_days; trend_exits has no such window, so there lookback= is still refused with a hint."""
    import numpy as np

    rows = _session_rows(days=12, drift=[3e-5, -3e-5, 0.0])
    want = ft.noise_area_breakout(rows, lookback_days=5)
    got = ft.noise_area_breakout(rows, lookback=5)
    assert np.allclose(np.asarray(got, float), np.asarray(want, float), equal_nan=True)
    assert "lookback= is called lookback_days=" in capsys.readouterr().err
    with pytest.raises(TypeError, match="did you mean 'vol_window'"):
        ft.trend_exits(np.ones(len(rows)), rows, lookback=30)


def test_a_guessed_dataset_name_is_answered_with_the_nearest_real_ones(ft):
    """Bugs #402/#406: fc_imb_oinet_d0_forecast_1 (and fc_fc_...) do not exist; the error now leads
    with the close names instead of a list the reader loses the tail of. A doubled fc_ is forgiven."""
    import numpy as np

    _add_feature(ft, "imb_oinet_d0_forecast_3", pd.DataFrame({"t": [1, 2], "f": np.r_[0.1, 0.2]}))
    with pytest.raises(KeyError, match="did you mean 'fc_imb_oinet_d0_forecast_3'"):
        ft.load("fc_imb_oinet_d0_forecast_1")
    assert len(ft.load("fc_fc_imb_oinet_d0_forecast_3")) == 2
    with pytest.raises(KeyError) as far:
        ft.load("zzz")
    assert "did you mean" not in str(far.value)


def test_the_atr_names_mean_what_trend_exits_calls_them(ft, capsys):
    """The stop IS stop_mult x the typical move over vol_window rows -- an ATR-style stop -- and agents
    kept writing atr_lookback= / atr_mult= after being told the names. Those two are taken as said."""
    import numpy as np

    rows = pd.DataFrame({"t": pd.date_range("2024-06-03 13:30", periods=200, freq="1min"),
                         "Close": np.r_[np.linspace(100, 102, 100), np.linspace(102, 100.5, 100)]})
    e = np.zeros(200)
    e[20] = 1
    want = ft.trend_exits(e, rows, vol_window=30, stop_mult=2.0).to_numpy()
    got = ft.trend_exits(e, rows, atr_lookback=30, atr_mult=2.0).to_numpy()
    assert (got == want).all()
    err = capsys.readouterr().err
    assert "atr_lookback= is called vol_window=" in err and "atr_mult= is called stop_mult=" in err
    # the real name given too, or a name that means something else: still refused
    with pytest.raises(TypeError, match="unexpected keyword argument 'atr_lookback'"):
        ft.trend_exits(e, rows, atr_lookback=30, vol_window=60)
    with pytest.raises(TypeError, match="did you mean 'size'"):
        ft.trend_exits(e, rows, base=2.0)


def test_clock_also_reads_by_name(ft):
    """10-01 12:16: clock = ft.clock(rows['t']); clock['session'] -- still a pair to unpack."""
    rows = _session_rows(days=2)
    session, minute = ft.clock(rows["t"])
    clock = ft.clock(rows["t"])
    assert (clock["session"] == session).all() and (clock.minute == minute).all() and (clock[1] == minute).all()
    with pytest.raises(KeyError, match="session, minute"):
        clock["date"]


def test_polars_compat_takes_a_positional_window_as_meant(ft):
    """rolling_quantile(0.9, 20) put 20 in `interpolation`; rolling_mean(3, 2) put 2 in `weights`."""
    import polars as pl

    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    got = df.select(pl.col("x").rolling_quantile(0.5, 3).alias("q"), pl.col("x").rolling_mean(3, 2).alias("m"))
    want = df.select(pl.col("x").rolling_quantile(0.5, window_size=3).alias("q"),
                     pl.col("x").rolling_mean(3, min_samples=2).alias("m"))
    assert got.equals(want)
    assert df["x"].rolling_quantile(0.5, 3).equals(df["x"].rolling_quantile(0.5, window_size=3))
    assert df["x"].rolling_mean(3, [1.0, 1.0, 1.0]).to_list()[-1] == pytest.approx(4.0)   # real weights untouched


def test_an_expression_describe_in_a_select_gives_the_stats_table(ft):
    """rows.select(pl.col('GEX').describe()) -- 7 runs on 10-01 died of "'Expr' object has no
    attribute 'describe'"; it now means the frame's describe() over those columns."""
    import polars as pl

    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0], "c": [0.0] * 4})
    assert df.select(pl.col("a").describe()).equals(df.select("a").describe())
    assert df.select([pl.col("a").describe(), pl.col("b").describe()]).equals(df.select("a", "b").describe())
    assert df.select(pl.col("a"), pl.col("b") * 2).columns == ["a", "b"]           # ordinary selects untouched
    with pytest.raises(TypeError, match="on its own"):
        df.select(pl.col("a").describe(), pl.col("b"))


def _minute_rows():
    import numpy as np
    import polars as pl

    t = pl.datetime_range(pl.datetime(2024, 1, 2, 14, 30), pl.datetime(2024, 1, 2, 15, 29), "1m", eager=True)
    return pl.DataFrame({"t": t, "Close": np.linspace(100, 101, 60), "GEX": np.sin(np.arange(60))})


def test_parsing_a_datetime_column_as_text_uses_it_as_it_is(ft):
    """pl.col('t').str.strptime / .str.to_datetime / df['SlotUtc'].str.to_date on a column that is already
    a datetime -- "SchemaError: expected `String`, got `datetime[ns]`", 11 runs (bug #134)."""
    import polars as pl

    rows = _minute_rows()
    got = rows.select(pl.col("t").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f").dt.date().alias("d"),
                      pl.col("t").str.to_datetime("%Y-%m-%dT%H:%M:%S").dt.hour().alias("h"),
                      pl.col("t").str.slice(0, 10).alias("day"))
    assert got.row(0) == (rows["t"][0].date(), 14, "2024-01-02")
    raw = rows.rename({"t": "SlotUtc"})
    assert raw["SlotUtc"].str.to_date().n_unique() == 1
    assert raw.select(pl.col("SlotUtc").str.to_date()).dtypes == [pl.Date]
    assert pl.Series(["2024-01-02"]).str.to_date().dtype == pl.Date          # real text is still parsed
    assert pl.DataFrame({"s": ["a1"]}).select(pl.col("s").str.slice(0, 1)).item() == "a"


def test_pandas_habits_on_polars_frames_do_what_they_mean(ft):
    """df.sort_values / s.values / s.corr(other) on polars, df['x'] = values, .to_pandas() on pandas,
    clip(lower=), pl.col(x).rolling(n).mean(), '15min' durations -- each a repeated crash on 09-30/10-01."""
    import numpy as np
    import polars as pl

    rows = _minute_rows()
    assert type(rows.sort_values("t")).__module__.startswith("pandas")
    assert isinstance(rows["Close"].values, np.ndarray)
    assert rows["Close"].corr(rows["GEX"]) == pytest.approx(rows.to_pandas()["Close"].corr(rows.to_pandas()["GEX"]))
    assert isinstance(rows.copy(), pl.DataFrame)
    with pytest.raises(AttributeError, match="no attribute 'not_a_method'"):
        rows.not_a_method
    r = rows.clone()
    r["z"] = np.arange(60)
    r["k"] = 1.5
    r["e"] = pl.col("Close") * 2
    assert r.columns[-3:] == ["z", "k", "e"] and r["k"][0] == 1.5 and r["z"][59] == 59
    assert rows.to_pandas().to_pandas().shape == (60, 3)
    got = rows.select(pl.col("GEX").clip(lower=-0.5, upper=0.5).max().alias("c"),
                      pl.col("Close").rolling(5, min_periods=2).mean().alias("m"))
    want = rows.select(pl.col("GEX").clip(-0.5, 0.5).max().alias("c"),
                       pl.col("Close").rolling_mean(5, min_samples=2).alias("m"))
    assert got.equals(want)
    assert rows.select(pl.col("Close").rolling(5).quantile(0.5)).equals(
        rows.select(pl.col("Close").rolling_quantile(0.5, window_size=5)))
    assert rows.group_by_dynamic("t", every="15min").agg(pl.len()).height == 4
    assert rows.select(pl.col("t").dt.truncate("15min")).n_unique() == 4


def test_ft_helpers_find_the_time_column_under_its_dataset_name(ft):
    """quick_score / trend_exits on a frame loaded from the dataset (SlotUtc, no t): KeyError 't', 6 runs."""
    import numpy as np

    raw = _minute_rows().rename({"t": "SlotUtc"}).to_pandas()
    assert ft.quick_score(np.ones(60), raw) is not None
    with pytest.raises(KeyError, match="no time column 'when'"):
        ft._times(raw, "when")


def test_a_dataset_or_tool_imported_as_a_module_says_what_it_is(ft, monkeypatch):
    monkeypatch.setattr(ft, "datasets", lambda: ["trade_book_trades_19f971fff6"])
    finder = ft._ImportHelp()
    with pytest.raises(ModuleNotFoundError, match=r"DATASET.*ft.load\('trade_book_trades_19f971fff6'\)"):
        finder.find_spec("trade_book_trades_19f971fff6")
    for name in ("chronos__chronos_forecast", "deci_plot"):
        with pytest.raises(ModuleNotFoundError, match="is a TOOL you call"):
            finder.find_spec(name)
    assert finder.find_spec("some_real_package") is None


def test_rows_columns_derive_vwap_and_name_the_unknown(ft, tmp_path, monkeypatch):
    """ft.rows_pl(columns=[..., 'VWAP']) -- not a column, but ft can make it (Qwen, 10-01 14:08, twice)."""
    import polars as pl

    rows = _minute_rows().with_columns(pl.lit(1.0).alias("Volume"))
    rows.write_parquet(tmp_path / "rows.parquet")
    monkeypatch.setattr(ft, "_TASK", str(tmp_path))
    got = ft.rows_pl(columns=["Close", "VWAP"])
    assert got.columns == ["t", "Close", "VWAP"]
    assert got["VWAP"].to_list() == pytest.approx(ft.session_vwap(rows.to_pandas()).to_list())
    assert list(ft.rows(columns=["VWAP"]).columns) == ["t", "VWAP"]
    with pytest.raises(KeyError, match=r"no 'Clse' \(did you mean 'Close'\?\)"):
        ft.rows_pl(columns=["Clse"])


def test_alias_after_with_columns_and_a_missing_session_column(ft):
    """rows.with_columns(expr).alias('fwd30') (10-01 14:07: the unnamed expression overwrote Close) and
    pl.col(x).over('session') on rows without a session column."""
    import polars as pl

    rows = _minute_rows()
    got = rows.with_columns((pl.col("Close").shift(-30) / pl.col("Close") - 1) * 10000).alias("fwd30")
    assert got.columns == ["t", "Close", "GEX", "fwd30"] and got["Close"].equals(rows["Close"])
    got = rows.with_columns(pl.col("GEX").mean().over("session").alias("m"))
    assert "session" in got.columns and got["session"][0] == rows["t"][0].date()
    assert rows.group_by("day").agg(pl.len()).height == 1
    assert rows.with_columns(pl.col("GEX").alias("x")).columns == ["t", "Close", "GEX", "x"]   # untouched


def test_quick_score_reads_as_its_sharpe_where_a_number_is_written(ft):
    """f"{score:.4f}" on quick_score's dict (Qwen, 10-01 14:16)."""
    s = ft._Score({"sharpe": 1.23456, "trades_per_day": 3})
    assert f"{s:.2f}" == "1.23" and float(s) == 1.23456 and s > 1 and s < 2 and max([s, ft._Score({"sharpe": 2.0})])["sharpe"] == 2.0
    assert f"{s}" == str(dict(s)) and s["trades_per_day"] == 3
    assert s["in_sample"]["sharpe"] == 1.23456                    # the scored-candidate shape (10-01 14:19)
    with pytest.raises(KeyError, match="in-sample stats only: sharpe, trades_per_day"):
        s["holdout"]


def test_quick_score_takes_in_sample_names_and_refuses_holdout_ones(ft, capsys):
    """qs['in_sample_sharpe'] (Qwen3.6, 10-06 06:53): every quick-score stat is in-sample already."""
    s = ft._Score({"sharpe": 1.5, "trades": 40, "sharpe_h1": 0.9})
    assert s["in_sample_sharpe"] == s["is_sharpe"] == s["sharpe_is"] == s["IS_Sharpe"] == 1.5
    assert s["in_sample_trades"] == 40 and s["is_half1"] == 0.9 and s.get("insample_sharpe") == 1.5
    assert "[ft] quick_score['in_sample_sharpe']" in capsys.readouterr().err
    for key in ("holdout_sharpe", "oos_sharpe", "sharpe_test"):
        with pytest.raises(KeyError, match="never sees the holdout.*in-sample stats only: sharpe"):
            s[key]
    assert s.get("holdout_sharpe") is None and s.get("nope", 7) == 7
    with pytest.raises(KeyError, match="in-sample stats only"):
        s["is_bogus"]


def test_pandas_arithmetic_methods_on_polars(ft, capsys):
    """pl.col('t').dt.second().div(60) -- "'Expr' object has no attribute 'div'" (Qwen3.6, 10-06 06:59)."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1.0, 2.0, None], "b": [4.0, 8.0, 2.0]})
    got = df.select(pl.col("b").div(2).alias("d"), pl.col("b").divide(pl.col("a")).alias("q"),
                    pl.col("b").multiply(3).alias("m"), pl.col("b").subtract(1).alias("s"),
                    pl.col("a").rsub(10).alias("r"), pl.col("b").rdiv(16).alias("rd"),
                    pl.col("a").div(pl.col("b"), fill_value=0).alias("f"))
    assert got["d"].to_list() == [2.0, 4.0, 1.0] and got["q"].to_list()[:2] == [4.0, 4.0]
    assert got["m"].to_list() == [12.0, 24.0, 6.0] and got["s"].to_list() == [3.0, 7.0, 1.0]
    assert got["r"].to_list()[:2] == [9.0, 8.0] and got["rd"].to_list() == [4.0, 2.0, 8.0]
    assert got["f"].to_list() == [0.25, 0.25, 0.0]
    assert df["b"].div(2).to_list() == [2.0, 4.0, 1.0] and df["b"].mul(2).to_list() == [8.0, 16.0, 4.0]
    assert df["b"].add(df["b"]).to_list() == [8.0, 16.0, 4.0]
    assert "[ft] .div(x) is pandas" in capsys.readouterr().err
    assert df.select(pl.col("b").mul(2))["b"].to_list() == [8.0, 16.0, 4.0]   # polars' own: untouched


def test_a_number_or_time_column_used_as_a_filter_condition(ft, capsys):
    """agg.filter(pl.col('has_long') | pl.col('has_short')) on 0/1 Int8 flags (10-06 05:38) and
    rows.filter(pl.col('t')) as "every row" (10-06 06:53): "filter predicate must be of type `Boolean`"."""
    from datetime import datetime

    pl = pytest.importorskip("polars")
    agg = pl.DataFrame({"has_long": [1, 0, 0, None], "has_short": [0, 0, 1, 1]},
                       schema={"has_long": pl.Int8, "has_short": pl.Int8})
    assert agg.filter(pl.col("has_long") | pl.col("has_short")).height == 2       # the null row drops
    assert agg.filter(pl.col("has_short")).height == 2
    assert "read Int8 as non-zero" in capsys.readouterr().err
    rows = pl.DataFrame({"t": [datetime(2024, 1, 2, 15), None, datetime(2024, 1, 2, 16)], "x": [1, 2, 3]})
    assert rows.filter(pl.col("t"))["x"].to_list() == [1, 3]
    assert rows.filter(pl.col("x") > 1, pl.col("t")).height == 1                   # mixed with a real condition
    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.DataFrame({"s": ["a", ""]}).filter(pl.col("s"))                        # a string stays polars' error


def test_library_methods_across_pandas_and_polars(ft):
    """bars = ft.resample(...) (pandas); bars.sort('SlotUtc') -- and df.columns.tolist() on polars (10-01 14:16-17)."""
    import polars as pl

    rows = _minute_rows()
    assert rows.columns.tolist() == ["t", "Close", "GEX"] and rows.columns == ["t", "Close", "GEX"]
    pdf = rows.to_pandas()
    got = pdf.sort("t", descending=True)
    assert isinstance(got, pl.DataFrame) and got["t"][0] == rows["t"][-1]
    assert pdf.Close.iloc[0] == rows["Close"][0]                         # column attributes untouched
    with pytest.raises(AttributeError):
        pdf.not_a_method


def test_one_with_columns_may_use_a_column_it_creates(ft):
    """agg.with_columns((pl.col('pin') > 0).alias('pin_pos'), (pl.col('pin_pos') != ...).alias('flip')) -- 14:31, twice."""
    import polars as pl

    df = pl.DataFrame({"pin": [1.0, -1.0, -2.0, 3.0]})
    got = df.with_columns((pl.col("pin") > 0).alias("pin_pos"),
                          (pl.col("pin_pos") != pl.col("pin_pos").shift(1)).alias("flip"))
    assert got["flip"].to_list() == [None, True, False, True]
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        df.with_columns(pl.col("nope") * 2)                          # a column nobody makes: still an error


def test_minute_of_day_arithmetic_does_not_wrap(ft):
    """dt.hour() is Int8 in polars: dt.hour() * 60 + dt.minute() gave 102 for 14:30, and a minute filter then
    dropped every row (10-01 14:38) -- every minute-of-day gate written so had been silently wrong."""
    import numpy as np
    import polars as pl

    rows = _minute_rows()
    m = rows.select((pl.col("t").dt.hour() * 60 + pl.col("t").dt.minute()).alias("m"))["m"]
    assert m[0] == 14 * 60 + 30 and (rows["t"].dt.hour() * 60)[0] == 840
    with pytest.raises(ValueError, match="rows are EMPTY"):
        ft.quick_score(np.zeros(0), rows.head(0))


def test_a_pandas_method_on_a_series_numpy_array_runs_as_meant(ft):
    """valid[f].to_numpy().corr(other.to_numpy()) (10-01 14:44); .abs / .nunique / .values before."""
    import numpy as np
    import polars as pl

    s, o = pl.Series([1.0, 2.0, 4.0, 3.0]), pl.Series([2.0, 4.0, 8.0, 5.0])
    x = s.to_numpy()
    assert x.corr(o.to_numpy()) == pytest.approx(np.corrcoef(x, o.to_numpy())[0, 1])
    assert (-x).abs().tolist() == [1.0, 2.0, 4.0, 3.0] and x.nunique() == 4
    assert isinstance(x + 1, np.ndarray) and float(np.mean(x)) == 2.5
    with pytest.raises(AttributeError, match="no attribute 'not_a_thing'"):
        x.not_a_thing


def test_a_time_column_asked_for_under_the_other_datasets_name(ft, tmp_path, monkeypatch):
    """ft.load_pl('fc_...', columns=['SlotUtc', 'fc_median']) on a forecast whose time is t (10-01 14:44)."""
    import numpy as np
    import polars as pl

    t = _minute_rows()["t"]
    pl.DataFrame({"t": t, "fc_median": np.arange(len(t), dtype=float)}).write_parquet(tmp_path / "fc_x.parquet")
    monkeypatch.setattr(ft, "_CATALOG", [{"view": "fc_x", "path": "fc_x.parquet", "format": "parquet",
                                          "root": str(tmp_path)}])
    monkeypatch.setattr(ft, "_note_used", lambda v: None)
    assert ft.load_pl("fc_x", columns=["SlotUtc", "fc_median"]).columns == ["SlotUtc", "fc_median"]
    assert list(ft.load("fc_x", columns=["SlotUtc", "fc_median"]).columns) == ["SlotUtc", "fc_median"]
    assert ft.load_pl("fc_x", columns=["t"]).columns == ["t"]
    _, minute = ft.clock(t)
    assert minute[:2].values.tolist() == [570.0, 571.0]                # clock arrays take pandas calls too


def test_sort_by_on_a_frame_and_a_missing_column_names_the_nearest(ft):
    """df.sort_by('SlotUtc') (10-01 15:01); drop_nulls(subset=[..., 'ret_60f']) when the script made ret_60bps (15:02)."""
    import polars as pl

    df = pl.DataFrame({"SlotUtc": [3, 1, 2], "ret_10bps": [1.0, None, 2.0], "ret_60bps": [1.0, 2.0, None]})
    assert df.sort_by("SlotUtc")["SlotUtc"].to_list() == [1, 2, 3]
    with pytest.raises(pl.exceptions.ColumnNotFoundError, match="did you mean 'ret_60bps'"):
        df.drop_nulls(subset=["ret_10bps", "ret_60f"])
    with pytest.raises(pl.exceptions.ColumnNotFoundError, match="did you mean 'ret_60bps'"):
        df.with_columns(pl.col("ret_60f") * 2)                       # not mistaken for a column the call makes


def test_a_pandas_column_the_frame_lacks_names_the_nearest(ft):
    """bars['Imb_OINet_D0'] after an .agg({...}) that did not keep it (10-01 15:05)."""
    df = _minute_rows().to_pandas()
    with pytest.raises(KeyError, match=r"'GXE' -- the frame has no such column; did you mean 'GEX'\? It has: t, Close, GEX"):
        df["GXE"]
    assert df["GEX"].iloc[0] == df.GEX.iloc[0]
    with pytest.raises(KeyError):
        df[["GEX", "nope"]]                                           # list keys keep pandas' own message


def test_rolling_apply_returning_the_window_takes_its_last_value(ft, capsys):
    """s.rolling(60).apply(lambda x: (x - x.mean()) / x.std()) -- 'must be real number, not Series' (10-01 15:27);
    and an unaliased with_columns that replaces Close is said on stderr (15:25)."""
    import numpy as np
    import polars as pl

    s = pd.Series(np.arange(10, dtype=float) ** 1.5)
    z = s.rolling(5, min_periods=2).apply(lambda x: (x - x.mean()) / x.std())
    w = s.iloc[5:10]
    assert z.iloc[9] == pytest.approx((w.iloc[-1] - w.mean()) / w.std())
    assert s.rolling(3).apply(np.max, raw=True).iloc[2] == s.iloc[2]
    pl.DataFrame({"High": [1.0, 2.0]}).with_columns(pl.col("High").shift(-1) - pl.col("High"))
    assert "REPLACED the 'High' column" in capsys.readouterr().err   # once per column; no other test replaces High


def test_expression_corr_is_pl_corr(ft):
    """pl.col(c).corr(pl.col('fret30')) -- 'Expr' object has no attribute 'corr' (10-01 15:31)."""
    import polars as pl

    df = pl.DataFrame({"a": [1.0, 2.0, 4.0, 3.0], "b": [2.0, 4.0, 8.0, 5.0]})
    assert df.select(pl.col("a").corr(pl.col("b"))).item() == pytest.approx(df.select(pl.corr("a", "b")).item())
    assert df.select(pl.col("a").corr(pl.col("b"), method="spearman")).item() == pytest.approx(1.0)   # same ranks


def test_replayed_polars_habits_do_what_they_mean(ft):
    """Failures of the whole work archive replayed against ft (10-01): pl.col(x).nth(n) in an agg, two
    stats of one column in one select ("DuplicateError: ... duplicate output name"), rows[<bool expr>],
    .dt.date / .dt.hour written as pandas' properties, .dt.floor / .dt.cast, '5 hours', pl.cut(...)."""
    import polars as pl

    rows = _minute_rows().with_columns(pl.Series("d", [1] * 30 + [2] * 30))
    got = rows.group_by("d", maintain_order=True).agg(pl.col("Close").first(), pl.col("Close").nth(2),
                                                      pl.col("Close").nth(99))
    assert got.columns == ["d", "Close_first", "Close_first_2", "Close_first_3"]
    assert got["Close_first_2"].to_list() == [rows["Close"][2], rows["Close"][32]] and got["Close_first_3"].null_count() == 2
    q = rows.select(pl.col("GEX").quantile(0.1), pl.col("GEX").quantile(0.9))
    assert q.columns[0].startswith("GEX_quantile") and len(set(q.columns)) == 2 and q.item(0, 0) < q.item(0, 1)
    assert rows.group_by("d").len().select(pl.col("len").mean(), pl.col("len").max()).columns == ["len_mean", "len_max"]
    assert rows.select(pl.col("GEX"), pl.col("GEX").abs().alias("a")).columns == ["GEX", "a"]       # untouched
    with pytest.raises(pl.exceptions.DuplicateError):
        rows.select(pl.col("GEX"), pl.col("Close").alias("GEX"))         # nothing to tell apart: still an error

    assert rows[pl.col("GEX") > 0].equals(rows.filter(pl.col("GEX") > 0))
    assert rows[rows["GEX"] > 0].height == rows.filter(pl.col("GEX") > 0).height
    assert rows["GEX"].len() == 60 and rows[["t", "GEX"]].columns == ["t", "GEX"]                    # untouched

    got = rows.with_columns(pl.col("t").dt.date.alias("day"), (pl.col("t").dt.hour * 60 + pl.col("t").dt.minute).alias("m"))
    assert got["day"][0] == rows["t"][0].date() and got["m"][0] == 14 * 60 + 30
    assert rows.select(pl.col("t").dt.hour().alias("h"))["h"][0] == 14                           # the call: as before
    assert rows["t"].dt.date.alias("x").to_list()[0] == rows["t"][0].date() and rows["t"].dt.hour()[0] == 14
    assert rows.filter(pl.col("t").dt.minute == 31).height == 1

    got = rows.select(pl.col("t").dt.floor("15min").alias("b"), pl.col("t").dt.cast(pl.Date).alias("d"),
                      pl.col("t").dt.offset_by("5 hours").alias("o"))
    assert got["b"].n_unique() == 4 and got["d"][0] == rows["t"][0].date()
    assert (got["o"][0] - rows["t"][0]).total_seconds() == 5 * 3600
    assert rows["t"].dt.floor("1h").n_unique() == 2 and rows["t"].dt.floor("1h")[0].minute == 0
    assert rows.select(pl.cut("GEX", breaks=[0.0]).alias("b"))["b"].n_unique() == 2
    with pytest.raises(TypeError, match="break POINTS"):
        pl.cut("GEX", 3)


def test_values_of_a_pandas_series_takes_pandas_calls(ft):
    """df['SkewRR_Value'].values.rolling(2000) / rows['t'].values.astype('int64').values (10-01 replay, 3 submissions)."""
    import numpy as np

    s = pd.Series([1.0, 2.0, 4.0])
    v = s.values
    assert isinstance(v, np.ndarray) and v.rolling(2).mean().tolist()[1:] == [1.5, 3.0]
    assert v.astype("int64").values.tolist() == [1, 2, 4] and float(np.sum(v)) == 7.0
    sub = pd.DataFrame({"a": [1.0, -2.0, 3.0]}).loc[lambda d: d["a"] > 0].copy()     # index 0, 2
    sub["b"] = pd.Series([10.0, 30.0]).values                    # an array is placed by position, not aligned
    sub["c"] = pd.Series([1.0, 3.0]).to_numpy()
    assert sub["b"].tolist() == [10.0, 30.0] and sub["c"].tolist() == [1.0, 3.0]
    pos = pd.Series([1.0, 0.0, 1.0], index=pd.date_range("2024-01-02 15:00", periods=3, freq="12h"))
    assert pos[pos != 0].index.date.nunique() == 2                       # 10-01 replay, 2 runs
    import polars as pl

    try:
        qs = pl.Series([1.0, 2.0, 3.0, 4.0]).quantile([0.25, 0.75])      # a list of quantiles (polars >= 1.3x)
    except TypeError:
        qs = None
    if qs is not None:
        q1, q2 = qs.to_list()
        assert q1 < q2
    assert isinstance(pl.Series([1.0, 3.0]).quantile(0.5), float)        # one quantile: a number, as before
    assert pd.DataFrame({"a": [1, 2]}).values.shape == (2, 1)                # a frame's .values: untouched


def test_load_without_a_name_and_ft_imported_from_itself(ft, tmp_path, monkeypatch):
    """rows = ft.load() in a task objective's library test; `from ft import rows_pl, ft` (10-01 replay)."""
    import sys

    _minute_rows().write_parquet(tmp_path / "rows.parquet")
    with pytest.raises(TypeError, match="needs the dataset's name"):
        ft.load()
    monkeypatch.setattr(ft, "_TASK", str(tmp_path))
    assert list(ft.load().columns) == ["t", "Close", "GEX"] and ft.load_pl(columns=["GEX"]).columns == ["t", "GEX"]
    monkeypatch.setattr(ft, "_CATALOG", [{"view": "book", "path": "rows.parquet", "format": "parquet",
                                          "root": str(tmp_path)}])
    monkeypatch.setattr(ft, "_note_used", lambda v: None)
    assert ft.rows_pl("book").columns == ["t", "Close", "GEX"]          # a dataset's name where columns go
    assert ft.rows_pl("GEX").columns == ["t", "GEX"]                     # one column's name: that column
    monkeypatch.setitem(sys.modules, "ft_under_test", ft)
    assert ft.ft is ft
    with pytest.raises(AttributeError):
        ft.not_a_thing


def test_direction_scan_leaves_text_columns_out(ft):
    """A date-as-text column made direction_scan die on float('2023-08-01'): pandas 3 names its text dtype
    "str", which the check for "object"/"string" let through (10-01 replay)."""
    import numpy as np

    rows = _minute_rows().to_pandas()
    rows["day"] = rows["t"].dt.strftime("%Y-%m-%d")
    rows["flag"] = rows["GEX"] > 0
    got = ft.direction_scan(rows, times=("09:31",), min_sessions=1)
    assert "day" not in set(got["field"] if len(got) else []) and np.isfinite(len(got))


def test_rows_columns_make_the_trade_review_features(ft, tmp_path, monkeypatch):
    """ft.rows_pl(columns=[..., 'GexFlip_Pos_vs_price_bps', 'minutes_into_session']) -- names the trade review
    (app/trade_book.py) reports entries by, asked of the rows as columns (10-01 00:42-03:59, 6 runs)."""
    import numpy as np
    import polars as pl

    one = _minute_rows().with_columns(pl.Series("Wall", np.linspace(101, 102, 60)))
    two = one.with_columns(pl.col("t") + pl.duration(days=1), pl.col("Close") * 2)
    pl.concat([one, two]).write_parquet(tmp_path / "rows.parquet")
    monkeypatch.setattr(ft, "_TASK", str(tmp_path))
    got = ft.rows_pl(columns=["Close", "Wall_vs_price_bps", "minutes_into_session", "price_chg_5m_bps",
                              "price_since_open_bps", "price_in_day_range"])
    assert got.columns == ["t", "Close", "Wall_vs_price_bps", "minutes_into_session", "price_chg_5m_bps",
                           "price_since_open_bps", "price_in_day_range"]
    c, w = got["Close"].to_numpy(), np.r_[one["Wall"].to_numpy(), one["Wall"].to_numpy()]
    assert got["Wall_vs_price_bps"].to_numpy() == pytest.approx((w / c - 1) * 1e4)
    assert got["minutes_into_session"].to_list()[:2] == [0.0, 1.0] and got["minutes_into_session"][60] == 0.0
    chg = got["price_chg_5m_bps"].to_numpy()
    assert np.isnan(chg[:5]).all() and np.isnan(chg[60:65]).all()             # not across the session start
    assert chg[5] == pytest.approx((c[5] / c[0] - 1) * 1e4)
    assert got["price_since_open_bps"][61] == pytest.approx((c[61] / c[60] - 1) * 1e4)
    assert got["price_in_day_range"][59] == pytest.approx(1.0)                 # a rising Close: at its high
    assert list(ft.rows(columns=["minutes_into_session"]).columns) == ["t", "minutes_into_session"]
    with pytest.raises(KeyError, match="no 'Nope_vs_price_bps'"):
        ft.rows_pl(columns=["Nope_vs_price_bps"])


def test_sort_takes_reverse_and_ascending(ft):
    """trades.sort('unit', reverse=True) -- 'unexpected keyword argument reverse' (10-01 16:29)."""
    import polars as pl

    t = pl.DataFrame({"unit": [1.0, 3.0, 2.0]})
    assert t.sort("unit", reverse=True)["unit"].to_list() == [3.0, 2.0, 1.0]
    assert t.sort("unit", ascending=False)["unit"].to_list() == [3.0, 2.0, 1.0]
    assert t["unit"].sort(reverse=True).to_list() == [3.0, 2.0, 1.0] and t.sort("unit")["unit"].to_list() == [1.0, 2.0, 3.0]


def test_col_of_an_expression_pandas_columns_in_with_columns_and_datetime_elements(ft):
    """pl.col(<expression>) (16:36); with_columns(vwap=ft.session_vwap(rows)) -- a pandas Series (16:37);
    [ts.hour for ts in rows['t'].to_numpy()] -- numpy datetime64 has no .hour (16:39)."""
    import numpy as np
    import polars as pl

    rows = _minute_rows().with_columns(pl.lit(1.0).alias("Volume"))
    bw = pl.col("GEX") > 0
    assert rows.select(pl.col(bw).mean()).item() == rows.select(bw.mean()).item()
    assert rows.select(pl.col("GEX").sum()).item() == rows["GEX"].sum()
    got = rows.with_columns(vwap=ft.session_vwap(rows))
    assert got["vwap"].to_list() == pytest.approx(ft.session_vwap(rows).to_list())
    t = rows["t"].to_numpy()
    assert [ts.hour for ts in t][:2] == [14, 14] and t[0] == np.datetime64(rows["t"][0])
    assert len(np.unique(t)) == len(t)                               # numpy's own indexing untouched


def test_pl_min_max_of_values_is_the_row_wise_min_max(ft):
    """pl.min(pl.col('inv_vol') * 2.0, 3.0) -- "invalid input for `col`", Python's min(a, b) (10-05 21:30)."""
    import polars as pl

    df = pl.DataFrame({"inv_vol": [0.5, 1.0, 2.0], "b": [9.0, 0.0, 1.0]})
    assert df.select(pl.min(pl.col("inv_vol") * 2.0, 3.0).alias("s"))["s"].to_list() == [1.0, 2.0, 3.0]
    assert df.select(pl.max(pl.col("inv_vol"), 1.0).alias("s"))["s"].to_list() == [1.0, 1.0, 2.0]
    assert df.select(pl.min("inv_vol", "b"))["inv_vol"].to_list() == [0.5]           # names: polars' own
    assert df.select(pl.max(pl.col("b") * 2)).item() == 18.0                        # one expression: its max
    assert df.select(pl.min("inv_vol", pl.col("b")).alias("s"))["s"].to_list() == [0.5, 0.0, 1.0]


def test_the_task_rows_view_loaded_by_name_in_a_task_run_is_the_rows(ft, tmp_path, monkeypatch):
    """ft.load('mcp_tasks_gex_gex_intraday_src_2') in a library smoke test -- "no dataset ...; available:
    (none)" (bug #428): the analysis tools' copy of the task rows, which a scored run does not mount."""
    import polars as pl

    _minute_rows().write_parquet(tmp_path / "rows.parquet")
    monkeypatch.setattr(ft, "_TASK", str(tmp_path))
    monkeypatch.setattr(ft, "_CATALOG", [])
    got = ft.load_pl("mcp_tasks_gex_gex_intraday_src_2", columns=["Close"])
    assert got.columns == ["t", "Close"] and got.height == _minute_rows().height
    assert list(ft.load("mcp_tasks_gex_gex_intraday", columns=["GEX"], prefix="a_").columns) == ["t", "a_GEX"]
    with pytest.raises(KeyError, match="no dataset 'other_view'"):
        ft.load("other_view")
    monkeypatch.setattr(ft, "_TASK", str(tmp_path / "none"))
    with pytest.raises(KeyError, match="no dataset 'mcp_tasks_x'"):
        ft.load_pl("mcp_tasks_x")                                    # not a task run: still the plain refusal


def test_an_expressions_values_are_read_on_the_newest_frame_with_its_columns(ft):
    """size = (1.0 / pl.col('intraday_vol')).clip(...); size.to_numpy() -- "'Expr' object has no attribute
    'to_numpy'" (10-05 21:3x). No frame with its columns: still polars' AttributeError (the hint's cue)."""
    import polars as pl

    rows = pl.DataFrame({"IntrVol": [0.5, 1.0, 4.0]}).with_columns(pl.col("IntrVol").alias("intraday_vol"))
    pl.DataFrame({"other": [1.0]}).with_columns(pl.col("other").alias("o"))      # newer, but lacks the column
    size = (1.0 / pl.col("intraday_vol") * 2.0).clip(upper_bound=3.0)
    assert size.to_numpy().tolist() == [3.0, 2.0, 0.5]
    assert size.to_list() == [3.0, 2.0, 0.5]
    with pytest.raises(AttributeError, match="'Expr' object has no attribute 'to_numpy'"):
        pl.col("never_made_xyz").to_numpy()


def test_a_name_read_off_an_unaliased_with_columns_is_the_alias_it_lacked(ft):
    """df.with_columns(pl.when(...).then(1).otherwise(0))['entry'] -- '"entry" not found', the column was
    'literal' (10-05 21:3x). A name that is in the frame, or one after an aliased expression, is untouched."""
    import polars as pl

    df = pl.DataFrame({"z": [2.0, -2.0, 0.0]})
    sig = df.with_columns(pl.when(pl.col("z") > 1).then(1).when(pl.col("z") < -1).then(-1).otherwise(0))
    assert sig["entry"].to_list() == [1, -1, 0] and sig["entry"].name == "entry"
    assert sig["z"].to_list() == [2.0, -2.0, 0.0]
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        df.with_columns((pl.col("z") * 2).alias("z2"))["entry"]


def test_rows_columns_ignore_letter_case_when_one_column_matches(ft, tmp_path, monkeypatch):
    """ft.rows_pl(columns=['Gexflip_Neg']) for GexFlip_Neg (10-01 16:46)."""
    import polars as pl

    _minute_rows().with_columns(pl.col("GEX").alias("GexFlip_Neg")).write_parquet(tmp_path / "rows.parquet")
    monkeypatch.setattr(ft, "_TASK", str(tmp_path))
    got = ft.rows_pl(columns=["Close", "Gexflip_Neg"])
    assert got.columns == ["t", "Close", "Gexflip_Neg"]
    assert got["Gexflip_Neg"].to_list() == _minute_rows()["GEX"].to_list()
    assert list(ft.rows(columns=["gexflip_neg"]).columns) == ["t", "gexflip_neg"]


def test_groupby_apply_keeps_the_grouping_columns(ft):
    """df.groupby('date', group_keys=False).apply(compute_vwap) lost 'date' in pandas 3 -- KeyError "['date'] not in
    index" on the next line (10-01 17:21). The groups carry their key columns again, as in pandas 2."""
    df = pd.DataFrame({"date": ["a", "a", "b"], "Close": [1.0, 2.0, 3.0]})
    out = df.groupby("date", group_keys=False).apply(lambda g: g.assign(c2=g["Close"] * 2))
    assert out["date"].tolist() == ["a", "a", "b"] and out["c2"].tolist() == [2.0, 4.0, 6.0]
    assert df.groupby("date").apply(lambda g: g["Close"].sum()).tolist() == [3.0, 3.0]


def test_clip_takes_min_max_and_numpy_names(ft):
    """pl.col(x).clip(min=..., max=...) -- 'unexpected keyword argument min' (10-01 19:25)."""
    import polars as pl

    df = pl.DataFrame({"x": [-3.0, 0.5, 4.0]})
    assert df.select(pl.col("x").clip(min=-1, max=1))["x"].to_list() == [-1.0, 0.5, 1.0]
    assert df["x"].clip(a_min=0).to_list() == [0.0, 0.5, 4.0]
    assert df.select(pl.col("x").clip(-2, 2))["x"].to_list() == [-2.0, 0.5, 2.0]


def test_a_column_asked_for_without_its_group_prefix(ft, tmp_path, monkeypatch):
    """ft.load_pl(..., columns=['TotalAbsGex']) for Pinning_TotalAbsGex (10-01 19:36); two matches -> still an error."""
    import polars as pl

    t = _minute_rows()["t"]
    pl.DataFrame({"SlotUtc": t, "Pinning_TotalAbsGex": [1.0] * len(t), "A_x": 1.0, "B_x": 2.0}).write_parquet(tmp_path / "bars.parquet")
    monkeypatch.setattr(ft, "_CATALOG", [{"view": "bars", "path": "bars.parquet", "format": "parquet", "root": str(tmp_path)}])
    monkeypatch.setattr(ft, "_note_used", lambda v: None)
    assert ft.load_pl("bars", columns=["SlotUtc", "TotalAbsGex"]).columns == ["SlotUtc", "TotalAbsGex"]
    assert list(ft.load("bars", columns=["totalabsgex"]).columns) == ["totalabsgex"]
    assert ft._same_column("x", {"A_x", "B_x"}) is None


def test_polars_to_numpy_takes_dtype(ft):
    """s.to_numpy(dtype=float) on a polars Series -- 'unexpected keyword argument dtype' (10-01 20:27)."""
    import polars as pl

    a = pl.Series([1, 2, 3]).to_numpy(dtype=float)
    assert a.dtype.kind == "f" and a.tolist() == [1.0, 2.0, 3.0]


def test_loc_with_a_mask_over_fewer_rows(ft):
    """bars.loc[mask, 'r'] with mask built on .dropna() -- pandas 3: bare AssertionError (10-01 21:50)."""
    import numpy as np

    bars = pd.DataFrame({"z": [np.nan, 1.0, 2.0, 3.0], "r": [10.0, 20.0, 30.0, 40.0]})
    mask = bars["z"].dropna() > 1.5
    assert bars.loc[mask, "r"].tolist() == [30.0, 40.0] and bars.loc[mask].shape == (2, 2)
    assert bars.loc[bars["r"] > 25, "r"].tolist() == [30.0, 40.0] and bars.loc[1:2, "r"].tolist() == [20.0, 30.0]


def test_filter_takes_a_numpy_mask_with_missing_values_and_quick_score_counts_trades(ft):
    """rows['r'].filter(cond.to_numpy()) where cond had nulls -- 'Expected a boolean mask' (10-01 22:08);
    score['long_trades'] (22:01)."""
    import numpy as np
    import polars as pl

    rows = pl.DataFrame({"r": [1.0, 2.0, 3.0, 4.0], "g": [None, 1.0, -1.0, 2.0]})
    m = (rows["g"] > 0).to_numpy()
    assert rows["r"].filter(m).to_list() == [2.0, 4.0] and rows.filter(m).height == 2
    t = _minute_rows()
    s = ft.quick_score(np.sign(np.sin(np.arange(t.height) / 7)), t)
    assert s["trades"] == s["long_trades"] + s["short_trades"] and s["long_trades"] > 0


def test_a_single_value_frame_formats_as_its_number(ft):
    """f"{rows.select(pl.col('r').mean()):.3f}" -- 'unsupported format string passed to DataFrame.__format__' (22:18)."""
    import polars as pl

    df = pl.DataFrame({"r": [1.0, 2.5]})
    assert f"{df.select(pl.col('r').mean()):.3f}" == "1.750" and f"{df['r'].tail(1):.1f}" == "2.5"
    assert f"{df}".startswith("shape")                                # no spec: the table, as before


def test_a_single_value_frame_is_its_number_in_if_int_float(ft):
    """`ratio = n_short / n_long if n_long else 0` with 1x1 frames -- 'truth value of a DataFrame is ambiguous' (23:20)."""
    import polars as pl

    rows = pl.DataFrame({"l": [1, 0, 1], "s": [0, 0, 1]})
    n_long, n_short = rows.select(pl.col("l").sum()), rows.select(pl.col("s").sum())
    assert f"{n_short / n_long if n_long else 0:.3f}" == "0.500" and int(n_long) == 2 and not bool(rows.select(pl.lit(0)))
    with pytest.raises((TypeError, ValueError)):
        bool(rows)


def test_quick_score_indexes_like_a_tuple(ft):
    """score[0] (10-02 00:54): the values in order, sharpe first."""
    s = ft._Score({"sharpe": 1.5, "sharpe_gross": 2.0})
    assert s[0] == 1.5 and s[1] == 2.0 and s[-1] == 2.0 and s["sharpe"] == 1.5
    with pytest.raises(KeyError):
        s[5]


def test_a_library_module_called_like_its_function(ft, tmp_path, monkeypatch):
    """symmetric_dabs_iv_signal(df) after `from lib import symmetric_dabs_iv_signal` (10-02 01:49)."""
    import importlib
    import sys

    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "__init__.py").write_text("")
    (tmp_path / "lib" / "my_sig.py").write_text("def signal(df):\n    return df * 2\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m == "lib" or m.startswith("lib.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [ft._LibCallable()] + sys.meta_path)
    mod = importlib.import_module("lib.my_sig")
    assert mod(3) == 6 and mod.signal(3) == 6


def test_statistics_of_an_empty_numeric_series_are_nan(ft):
    """subset['ret30'].mean() * 10000 on an empty selection -- polars None crashed the next line (10-02 04:03)."""
    import math

    import polars as pl

    s = pl.Series("r", [1.0, 2.0, 3.0])
    empty = s.filter(s > 10)
    assert math.isnan(empty.mean() * 10000) and math.isnan(empty.max()) and s.mean() == 2.0
    assert pl.Series(["a"]).filter(pl.Series([False])).max() is None     # non-numeric keeps None


def test_quick_score_answers_the_names_models_guess(ft):
    """score['half1'] (10-02 05:06) -> sharpe_h1, and friends."""
    s = ft._Score({"sharpe": 1.0, "sharpe_h1": 0.5, "sharpe_h2": 1.5, "trades": 9})
    assert s["half1"] == 0.5 and s["h2"] == 1.5 and s["sharpe_ratio"] == 1.0 and s["n_trades"] == 9
    with pytest.raises(KeyError):
        s["nonsense"]


def test_row_numbered_positions_take_the_times_of_the_one_frame_of_that_length(ft, tmp_path, monkeypatch):
    """Bug #331 (10-06 06:3x): bar = ft.load(...).sort_values('SlotUtc').reset_index(drop=True); pos = lib.signal(bar)
    -- positions indexed by row number, three submissions refused. One loaded frame has that many bars: its times."""
    t = pd.date_range("2024-01-02 14:30", periods=4, freq="10s")
    pd.DataFrame({"SlotUtc": t[::-1], "Close": [4.0, 3.0, 2.0, 1.0]}).to_parquet(tmp_path / "bars.parquet")
    monkeypatch.setattr(ft, "_CATALOG", [{"view": "bars", "path": "bars.parquet", "format": "parquet", "root": str(tmp_path)}])
    monkeypatch.setattr(ft, "_note_used", lambda v: None)
    bar = ft.load("bars").sort_values("SlotUtc").reset_index(drop=True)
    ft.report_positions(pd.Series([0.0, 1.0, 1.0, -1.0], index=bar.index))
    got = pd.read_parquet(Path(ft._FT) / "positions.parquet")
    assert got["t"].tolist() == list(t) and got["pos"].tolist() == [0.0, 1.0, 1.0, -1.0]
    # 15-min bars from ft.resample count too; a length no frame has is still refused.
    bars = ft.resample(bar, "20s")
    ft.report_positions(pd.Series([1.0, -1.0], index=bars.index))
    got = pd.read_parquet(Path(ft._FT) / "positions.parquet")
    assert got["t"].tolist() == bars["SlotUtc"].tolist() and got["pos"].tolist() == [1.0, -1.0]
    with pytest.raises(ValueError, match="RangeIndex"):
        ft.report_positions(pd.Series([1.0, 0.0, 1.0]))


def test_a_frame_indexed_by_its_time_column_still_selects_it(ft):
    """Bug #331 (10-06 06:39): bar.set_index('SlotUtc') then a library signal's df[['SlotUtc', 'Close']] --
    KeyError "['SlotUtc'] not in index"."""
    t = pd.date_range("2024-01-02 14:30", periods=3, freq="10s")
    bar = pd.DataFrame({"SlotUtc": t, "Close": [1.0, 2.0, 3.0]}).set_index("SlotUtc")
    got = bar[["SlotUtc", "Close"]]
    assert list(got.columns) == ["SlotUtc", "Close"] and got["SlotUtc"].tolist() == list(t)
    assert bar["SlotUtc"].tolist() == list(t)
    with pytest.raises(KeyError, match="no such column"):
        bar["Nope"]


def test_rows_columns_session_initials_and_a_field_under_another_group(ft, tmp_path, monkeypatch):
    """Bug #362: ft.rows_pl(columns=[...]) refused 'session' (10-06 10:26), 'DABS' for Doi_AboveBelowSkew
    (10-05), 'Vwap' (10-05) and 'Pinning_ProbTrend' for PinTrend_ProbTrend (10-01, 10-06)."""
    import polars as pl

    rows = _minute_rows().with_columns(pl.lit(1.0).alias("Volume"), pl.col("GEX").alias("Doi_AboveBelowSkew"),
                                       pl.col("GEX").alias("PinTrend_ProbTrend"), pl.col("GEX").alias("IV_NotionalScaled"),
                                       pl.col("GEX").alias("Pinning_Index"))
    rows.write_parquet(tmp_path / "rows.parquet")
    monkeypatch.setattr(ft, "_TASK", str(tmp_path))
    got = ft.rows_pl(columns=["t", "Close", "DABS", "IV_NS", "Pinning_ProbTrend", "Vwap", "session"])
    assert got.columns == ["t", "Close", "DABS", "IV_NS", "Pinning_ProbTrend", "Vwap", "session"]
    assert got["DABS"].to_list() == rows["GEX"].to_list() == got["Pinning_ProbTrend"].to_list()
    assert got.schema["session"] == pl.Date and got["session"][0] == rows["t"][0].date()
    # usable as the session it is
    assert got.group_by("session").agg(pl.len()).height == 1
    p = ft.rows(columns=["session", "DABS"])
    assert list(p.columns) == ["t", "session", "DABS"] and p["session"].nunique() == 1
    # Unclear names stay errors: 'Pinning_Foo' matches nothing, 'XYZ' no initials, a field shared by two groups.
    with pytest.raises(KeyError, match="'Pinning_Foo'"):
        ft.rows_pl(columns=["Pinning_Foo"])
    assert ft._same_column("XYZ", {"Doi_AboveBelowSkew"}) is None
    assert ft._same_column("C_Index", {"A_Index", "B_Index"}) is None


def test_quick_score_carries_resampled_positions_onto_the_rows(ft):
    """Bug #331 (10-06 06:30): quick_score(pos_on_15min_bars, ft.load(10s bars)) -- '8178 positions for 713531 rows'."""
    import numpy as np

    t = pd.date_range("2024-01-02 14:30", periods=120, freq="10s")
    bar = pd.DataFrame({"SlotUtc": t, "Close": np.linspace(100, 101, 120)})
    bars = ft.resample(bar, "5min")
    pos = pd.Series(np.ones(len(bars)))                         # row-numbered, one per 5-min bar
    want = ft.quick_score(ft.align(pos, bars["SlotUtc"], t).to_numpy(), bar)
    got = ft.quick_score(pos, bar)
    assert got["bps_per_day"] == want["bps_per_day"] != 0 and got["long_share"] == want["long_share"] > 0
    timed = pd.Series(np.ones(len(bars)), index=pd.DatetimeIndex(bars["SlotUtc"]))
    assert ft.quick_score(timed, bar)["bps_per_day"] == want["bps_per_day"]
    with pytest.raises(ValueError, match="one per row"):
        ft.quick_score(np.ones(7), bar)


def _signal_module_rows():
    import numpy as np
    import polars as pl

    t = pl.datetime_range(pl.datetime(2024, 1, 2, 14, 30), pl.datetime(2024, 1, 2, 15, 30), "10s", eager=True)
    rng = np.random.default_rng(3)
    return pl.DataFrame({"t": t, "Close": 100 + rng.standard_normal(len(t)).cumsum() * 0.05,
                         "Volume": rng.integers(1, 100, len(t)).astype(float), "GEX": rng.standard_normal(len(t))})


def _local_frame_signal(df):
    """Bug #174's module shape: columns built on signal()'s OWN frame, an expression over them returned."""
    import polars as pl

    df = df.sort("t").with_columns(pl.col("t").dt.date().alias("session"))
    vwap = ((pl.col("Close") * pl.col("Volume")).cum_sum().over("session")
            / pl.col("Volume").cum_sum().over("session")).alias("vwap")
    df = df.with_columns(vwap)
    df = df.with_columns(pl.col("GEX").rolling_mean(5).over("session").alias("gex_smooth"))
    return (pl.when((pl.col("Close") > pl.col("vwap")) & (pl.col("gex_smooth") < 0)).then(1)
            .when(pl.col("Close") < pl.col("vwap")).then(-1).otherwise(0).alias("signal"))


def _local_frame_want(rows):
    import polars as pl

    df = rows.sort("t").with_columns(pl.col("t").dt.date().alias("session"))
    df = df.with_columns(((pl.col("Close") * pl.col("Volume")).cum_sum().over("session")
                          / pl.col("Volume").cum_sum().over("session")).alias("vwap"))
    df = df.with_columns(pl.col("GEX").rolling_mean(5).over("session").alias("gex_smooth"))
    return df.select(_local_frame_signal(df)).to_series()


def test_an_expression_over_columns_a_function_built_runs_with_them_on_the_same_rows(ft, capsys):
    """Bug #174 (10-05/06, the most common blocker): sig = lib.signal(rows) builds vwap / gex_smooth on a frame of
    its own and returns pl.when(...).alias('signal'); then sig.to_numpy(), rows.with_columns(sig), rows.select(sig),
    rows.filter(sig == 1) all died on 'unable to find column "vwap"' -- the frame was gone once signal() returned.
    They run on the caller's rows with the columns from that frame (same height, same t)."""
    import polars as pl

    rows = _signal_module_rows()
    want = _local_frame_want(rows)
    assert want.n_unique() == 3
    assert _local_frame_signal(rows).to_numpy().tolist() == want.to_list()                    # 1. expr.to_numpy()
    out = rows.with_columns(_local_frame_signal(rows))                                         # 2. with_columns
    assert out.columns == rows.columns + ["signal"] and out["signal"].to_list() == want.to_list()
    assert out["signal"].is_not_null().all()
    assert rows.select(_local_frame_signal(rows)).to_series().to_list() == want.to_list()     # 3. select (smoke test)
    assert rows.filter(_local_frame_signal(rows) == 1).height == (want == 1).sum()            # 4. filter
    assert rows.filter(_local_frame_signal(rows) == 1).columns == rows.columns
    assert "[ft] .with_columns(expr): the expression reads 'vwap'" in capsys.readouterr().err
    # ft.load() gives pandas: signal(df) and df.select(sig) both run on pl.from_pandas(df)
    pdf = rows.to_pandas()
    assert pdf.select(_local_frame_signal(pdf)).to_series().to_list() == want.to_list()
    # the caller's rows in another order: lined up by t, not by position
    shuffled = rows.sample(fraction=1.0, shuffle=True, seed=1)
    got = shuffled.with_columns(_local_frame_signal(shuffled)).sort("t")["signal"]
    assert got.to_list() == want.to_list()


def test_columns_are_never_taken_from_a_frame_of_other_rows(ft):
    import polars as pl

    rows = _signal_module_rows()
    sig = _local_frame_signal(rows)
    with pytest.raises(pl.exceptions.ColumnNotFoundError, match="vwap"):
        rows.head(100).select(sig)                                         # fewer rows
    later = rows.with_columns(pl.col("t") + pl.duration(days=1))
    with pytest.raises(pl.exceptions.ColumnNotFoundError, match="vwap"):
        later.with_columns(sig)                                            # same height, other times
