"""ft.search: a strategy family and its walk-forward tuner, submitted in place of tuned numbers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("pyarrow")

FT_PATH = Path(__file__).resolve().parents[2] / "sandbox" / "ft.py"


@pytest.fixture
def ft(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("ft_search_under_test", FT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    work = tmp_path / ".ft"
    work.mkdir()
    monkeypatch.setattr(mod, "_FT", str(work))
    monkeypatch.setattr(mod, "_CATALOG", [])
    mod._root = tmp_path
    return mod


def _rows(days=260, seed=0, drift=None):
    """Hourly-ish rows on consecutive weekdays: a random walk with a per-day drift."""
    rng = np.random.default_rng(seed)
    t, p, price = [], [], 100.0
    for i, d in enumerate(pd.bdate_range("2023-01-02", periods=days)):
        mu = 0.0 if drift is None else drift[i]
        for ts in pd.date_range(d + pd.Timedelta("14h30min"), periods=26, freq="15min"):
            price *= 1 + mu + rng.normal(0, 8e-4)
            t.append(ts)
            p.append(price)
    return pd.DataFrame({"t": pd.to_datetime(t), "Close": p})


def _day_sign(ft, rows, flip_after=None):
    session, _ = ft.clock(rows["t"])
    index = np.unique(np.asarray(session), return_inverse=True)[1]
    return index, np.where(index % 2 == 0, 1.0, -1.0)


def test_session_pnl_matches_quick_score(ft):
    rows = _rows(days=40)
    pos = np.sign(np.sin(np.arange(len(rows)) / 9.0))
    session, _ = ft.clock(rows["t"])
    starts = ft._session_starts(np.asarray(session))
    ends = np.r_[starts[1:], len(rows)]
    daily = ft._session_pnl(pos, rows["Close"].to_numpy(), starts, ends, 2e-4, 1)
    assert np.mean(daily) * 1e4 == pytest.approx(ft.quick_score(pos, rows, cost_bps=2.0)["bps_per_day"], rel=1e-9)


def test_search_points_are_fixed_by_the_space_and_seed(ft):
    space = {"z": (0.5, 3.0), "window": (20, 240), "stop": [1.5, 2.0, 3.0]}
    a, b = ft._search_points(space, 24, 0), ft._search_points(space, 24, 0)
    assert a == b and len(a) == 24
    assert all(0.5 <= p["z"] <= 3.0 and isinstance(p["window"], int) and p["stop"] in (1.5, 2.0, 3.0) for p in a)
    assert ft._search_points(space, 24, 1) != a
    small = ft._search_points({"a": [1, 2], "b": [3, 4]}, 24, 0)
    assert small == [{"a": 1, "b": 3}, {"a": 1, "b": 4}, {"a": 2, "b": 3}, {"a": 2, "b": 4}]


def test_search_finds_the_real_edge_and_reports_trials(ft, tmp_path):
    days = 260
    drift = np.where(np.arange(days) % 2 == 0, 4e-4, -4e-4)
    rows = _rows(days=days, drift=drift)
    index, sign = _day_sign(ft, rows)

    def family(flip, scale):
        return sign * flip * scale

    found = ft.search(family, {"flip": [1.0, -1.0], "scale": [0.5, 1.0]}, rows)
    assert found.trials == 4 and found.best is not None and found.best["flip"] == 1.0
    assert found.stood_aside == 0.0
    assert (np.abs(found.positions[index < 60]) == 0).all()          # the warm-up is flat
    assert ft.quick_score(found, rows)["sharpe"] > 5                 # the result itself is a position series
    assert len(found) == len(rows)
    record = json.loads((tmp_path / ".ft" / "search.json").read_text())
    assert record[0]["trials"] == 4 and record[0]["family"] == "family" and record[0]["params"] == ["flip", "scale"]


def test_search_stands_aside_when_only_chance_is_on_offer(ft):
    rows = _rows(days=260, seed=3)
    n = len(rows)

    def family(seed):
        return np.random.default_rng(int(seed)).choice([-1.0, 0.0, 1.0], size=n)

    found = ft.search(family, {"seed": list(range(32))}, rows, trials=32)
    assert found.stood_aside > 0.5
    forced = ft.search(family, {"seed": list(range(32))}, rows, trials=32, confidence=0)
    assert forced.stood_aside == 0.0


def test_search_is_causal_on_truncated_rows(ft):
    """The look-ahead test: the same script on fewer rows must decide the same positions before the cut."""
    rows = _rows(days=260, seed=5, drift=1e-4 * np.sin(np.arange(260) / 7.0))

    def run(frame):
        close = frame["Close"]

        def family(window, sign):
            momentum = close / close.shift(int(window)) - 1.0
            return (np.sign(momentum.fillna(0.0)) * sign).to_numpy()

        return ft.search(family, {"window": (2, 60), "sign": [1.0, -1.0]}, frame, trials=12, refit="month",
                         confidence=0).positions

    full = run(rows)
    for cut in (len(rows) // 2, len(rows) * 3 // 4 + 7):       # mid-session on purpose
        trunc = run(rows.iloc[:cut].reset_index(drop=True))
        assert np.array_equal(full[:cut], trunc)


def test_a_family_on_resampled_bars_is_not_a_false_leak(ft):
    """Data cut short ends in a partial bar stamped at the last row. Carried at-or-before, its decision lands on a row
    that holds the previous bar's in the full run, and the look-ahead test calls that a leak (3 of 60 cuts)."""
    rows = _rows(days=200, seed=9, drift=1e-4 * np.sin(np.arange(200) / 5.0))

    def run(frame):
        bars = ft.resample(frame, "1h")

        def family(window, sign):
            close = bars["Close"]
            pos = np.sign((close / close.shift(int(window)) - 1.0).fillna(0.0)) * sign
            return pd.Series(pos.to_numpy(), index=pd.DatetimeIndex(bars["t"]))

        return ft.search(family, {"window": (2, 12), "sign": [1.0, -1.0]}, frame, trials=8, refit="month", confidence=0).positions

    full = run(rows)
    assert np.abs(full).sum() > 0
    for cut in range(len(rows) // 2, len(rows), len(rows) // 40):
        trunc = run(rows.iloc[:cut + 3].reset_index(drop=True))
        assert np.array_equal(full[:cut + 3], trunc), cut


def test_a_session_counts_as_seen_only_if_all_of_it_is_before_the_split(ft, tmp_path):
    """The harness splits on the UTC date; a New York evening session runs past midnight UTC."""
    t = pd.date_range("2023-01-02 14:00", "2023-09-01 23:00", freq="1h")
    t = t[t.dayofweek < 5]
    rows = pd.DataFrame({"t": t, "Close": 100.0 + np.arange(len(t)) * 0.01})
    (tmp_path / ".ft" / "run.json").write_text(json.dumps({"split": "2023-08-02", "timeout_s": 300}))
    found = ft.search(lambda a: np.ones(len(rows)) * a, {"a": [1.0, 0.5]}, rows, refit="month", confidence=0)
    assert all(p["from"] < "2023-08-01" for p in found.picks)


def test_a_forecast_being_built_is_not_a_failed_trial(ft):
    rows = _rows(days=30)

    def family(a):
        raise ft.ForecastPending("fc_x")

    with pytest.raises(ft.ForecastPending):
        ft.search(family, {"a": [1.0, 2.0]}, rows)


def test_search_stops_listing_at_the_split(ft, tmp_path):
    (tmp_path / ".ft" / "run.json").write_text(json.dumps({"split": "2023-07-03", "timeout_s": 300}))
    drift = np.where(np.arange(260) % 2 == 0, 4e-4, -4e-4)
    rows = _rows(days=260, drift=drift)
    _, sign = _day_sign(ft, rows)
    found = ft.search(lambda flip: sign * flip, {"flip": [1.0, -1.0]}, rows, refit="month")
    assert found.picks and all(p["from"] < "2023-07-03" for p in found.picks)
    assert found.positions[-1] != 0                            # the periods past the split still trade
    assert len(found.table) == 2


def test_search_says_when_the_family_is_too_slow_for_the_run(ft, tmp_path):
    import time

    (tmp_path / ".ft" / "run.json").write_text(json.dumps({"timeout_s": 1}))
    rows = _rows(days=30)

    def family(a):
        time.sleep(0.2)
        return np.zeros(len(rows))

    with pytest.raises(ValueError, match="fewer trials"):
        ft.search(family, {"a": (0.0, 1.0)}, rows, trials=24)


def test_search_survives_one_bad_trial_and_refuses_bad_arguments(ft):
    rows = _rows(days=120)
    _, sign = _day_sign(ft, rows)

    def family(a):
        if a == 2:
            raise ZeroDivisionError("boom")
        return sign * a

    assert ft.search(family, {"a": [1.0, 2, -1.0]}, rows, confidence=0).trials == 3

    def broken(a):
        raise ZeroDivisionError("boom")

    with pytest.raises(ZeroDivisionError):
        ft.search(broken, {"a": [1.0, 2.0]}, rows)
    with pytest.raises(ValueError, match="trials"):
        ft.search(family, {"a": [1.0]}, rows, trials=500)
    with pytest.raises(ValueError, match="refit"):
        ft.search(family, {"a": [1.0]}, rows, refit="weekly")
    with pytest.raises(ValueError, match="empty"):
        ft.search(family, {}, rows)


def test_search_takes_polars_rows(ft):
    pl = pytest.importorskip("polars")
    frame = _rows(days=120)
    _, sign = _day_sign(ft, frame)
    found = ft.search(lambda a: pl.Series(sign * a), {"a": [1.0, -1.0]}, pl.from_pandas(frame), confidence=0)
    assert len(found.positions) == len(frame)
