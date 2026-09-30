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
