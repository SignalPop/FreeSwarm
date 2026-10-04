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
    info.pop("_swings")
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



def test_intraday_flattens_at_each_days_last_bar(tmp_path):
    _write_prices(tmp_path)
    pos = tmp_path / "pos.parquet"
    # Long once, never closed: held over both days and the night between them.
    _write_positions(pos, [("2024-01-02 10:00", 1.0)])
    obj = _obj(cost_bps=0.0)
    overnight, _ = O._mark_to_market(obj, str(tmp_path), pos)
    assert math.isclose(overnight[1][1], 1.01 ** 4 - 1, rel_tol=1e-9)    # the overnight bar counts
    obj["metric"]["intraday"] = True
    net, info = O._mark_to_market(obj, str(tmp_path), pos)
    assert info["intraday"] is True
    # Flat at 10:03 each day: day 2's first bar (the overnight move) earns nothing, and the
    # still-reported long is entered again for the rest of day 2.
    assert math.isclose(net[0][1], 1.01 ** 3 - 1, rel_tol=1e-9)
    assert math.isclose(net[1][1], 1.01 ** 3 - 1, rel_tol=1e-9)
    tr = O._trade_table(obj, str(tmp_path), pos)
    assert len(tr["entry_t"]) == 2
    for a, b in zip(tr["entry_t"], tr["exit_t"]):
        assert a // 86400 == b // 86400                                   # opened and closed the same day
    assert not tr["open"].any()
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


_HEAD = ('Traceback (most recent call last):\n  File "script.py", line 9, in <module>\n    exec(compile(_src, '
         '"candidate.py", "exec"))\n')
# As the sandbox prints them: pandas raises the AttributeError from its own __getattr__, polars bare.
_ON_PANDAS = (_HEAD + '  File "candidate.py", line 10, in <module>\n    bars = bars.{attr}([\n           ^^^^^^^^^^^^^^^^^\n'
              '  File "/usr/local/lib/python3.12/site-packages/pandas/core/generic.py", line 6194, in __getattr__\n'
              '    return object.__getattribute__(self, name)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
              "AttributeError: '{kind}' object has no attribute '{attr}'\n")
_ON_POLARS = (_HEAD + '  File "candidate.py", line 4, in <module>\n    fc = fc.sort(\'t\').{attr}()\n         ^^^^^^^^^^^^^^^^^^^^^^^^\n'
              "AttributeError: '{kind}' object has no attribute '{attr}'\n")


def test_the_wrong_kind_of_frame_is_named_in_the_hint():
    """Bug #52 and its family: a script holds pandas frames (ft.load, ft.resample, ...) and polars
    frames (ft.load_pl) side by side, and "'DataFrame' object has no attribute 'with_columns'"
    does not say which one it was holding."""
    for attr, kind in (("with_columns", "DataFrame"), ("to_pandas", "DataFrame"), ("alias", "Series")):
        hint = O._frame_hint(_ON_PANDAS.format(attr=attr, kind=kind))
        assert f"that {kind} is PANDAS and .{attr} is a polars method" in hint and "pl.from_pandas" in hint
    for attr in ("reset_index", "sort_values", "copy"):
        hint = O._frame_hint(_ON_POLARS.format(attr=attr, kind="DataFrame"))
        assert f"that DataFrame is POLARS" in hint and f".{attr} is a pandas method" in hint and "to_pandas()" in hint
    # A name cut off mid-word or mistyped is not the other library's method: no hint to mislead with.
    assert O._frame_hint(_ON_PANDAS.format(attr="with_", kind="DataFrame")) == ""
    assert O._frame_hint(_ON_POLARS.format(attr="sort_valu", kind="DataFrame")) == ""
    assert O._frame_hint("AttributeError: 'numpy.ndarray' object has no attribute 'rolling_avg'") == ""
    assert O._frame_hint("") == ""
    # The same line reaches a failed submission's error ...
    note = O._failure_note(_ON_PANDAS.format(attr="with_columns", kind="DataFrame"))
    assert note.startswith("the script failed -- see stderr. Hint: that DataFrame is PANDAS")


# As candidate 4ac34e667a died of it: the script took .values itself, then called a pandas method.
_ON_NUMPY = (_HEAD + '  File "candidate.py", line 52, in <module>\n    main()\n'
             '  File "candidate.py", line 14, in main\n    skew_mean = skew.{attr}(window=2000, min_periods=1000).mean()\n'
             '                ^^^^^^^^^^^^\n'
             "AttributeError: 'numpy.ndarray' object has no attribute '{attr}'\n")


def test_a_pandas_or_polars_method_on_a_numpy_array_is_named_in_the_hint():
    """skew = df[c].values; skew.rolling(2000) -- "'numpy.ndarray' object has no attribute
    'rolling'" says neither where the array came from nor how to get the series back."""
    for attr in ("rolling", "shift", "ewm", "diff", "fillna", "ffill", "pct_change", "iloc", "abs"):
        hint = O._frame_hint(_ON_NUMPY.format(attr=attr))
        assert hint.startswith("that is a NUMPY array (from .values / .to_numpy()"), attr
        assert f".{attr} is a pandas method" in hint and f"pd.Series(x, index=df.index).{attr}(" in hint, attr
    for attr in ("with_columns", "fill_null", "rolling_mean", "is_null"):
        hint = O._frame_hint(_ON_NUMPY.format(attr=attr))
        assert hint.startswith("that is a NUMPY array") and f".{attr} is a polars" in hint, attr
    assert "pl.DataFrame({'x': x}).with_columns(" in O._frame_hint(_ON_NUMPY.format(attr="with_columns"))
    assert "pl.Series(x).fill_null(" in O._frame_hint(_ON_NUMPY.format(attr="fill_null"))
    # .values on what already is an array (comp_vals = comp.values, a few lines further down)
    assert "already a NUMPY array" in O._frame_hint(_ON_NUMPY.format(attr="values"))
    # a list habit, a typo: nothing to say
    for attr in ("append", "len", "rolingg"):
        assert O._frame_hint(_ON_NUMPY.format(attr=attr)) == "", attr
    # the error the run died of is the LAST one: a pandas/polars mix-up earlier in stderr does not win
    both = _ON_PANDAS.format(attr="with_columns", kind="DataFrame") + _ON_NUMPY.format(attr="rolling")
    assert O._frame_hint(both).startswith("that is a NUMPY array")
    both = _ON_NUMPY.format(attr="rolling") + _ON_PANDAS.format(attr="with_columns", kind="DataFrame")
    assert O._frame_hint(both).startswith("that DataFrame is PANDAS")
    # it reaches a failed submission's note, the regime lab's note and run_python (all _failure_note /
    # _frame_hint), and library_save's smoke test (library._usage_hint falls through to _frame_hint)
    note = O._failure_note(_ON_NUMPY.format(attr="rolling"))
    assert note.startswith("the script failed -- see stderr. Hint: that is a NUMPY array")


def test_a_polars_series_expression_mixup_is_named_in_the_hint():
    """The night of 2026-09-30: #130 s.cum_sum().over('session') on a Series, #109/#121
    pl.col(...).to_numpy(), #129 .cum_mean() (Python suggested 'cum_max'), df['x'].mean().over()."""
    hint = O._frame_hint(_ON_POLARS.format(attr="over", kind="Series"))
    assert "polars DATA" in hint and ".over exists only on EXPRESSIONS" in hint and ".over('session')" in hint
    hint = O._frame_hint(_ON_POLARS.format(attr="to_numpy", kind="Expr"))
    assert "polars EXPRESSION" in hint and "df.select(expr).to_series().to_numpy(" in hint
    hint = O._frame_hint(_ON_POLARS.format(attr="over", kind="float"))
    assert hint.startswith("that is a plain float, not a column") and ".mean().over('session')" in hint
    for kind in ("Series", "Expr", "DataFrame"):
        hint = O._frame_hint(_ON_POLARS.format(attr="cum_mean", kind=kind) + "Did you mean: 'cum_max'?")
        assert hint.startswith("there is no cum_mean in polars") and "cum_sum() / pl.col('x').cum_count()" in hint
    # a polars Series method it does have, a typo, a scalar's own habit: nothing to say
    for attr, kind in (("rolling_mean", "Series"), ("cum_smu", "Expr"), ("is_integer", "float"), ("ovr", "float")):
        assert O._frame_hint(_ON_POLARS.format(attr=attr, kind=kind)) == "", (attr, kind)
    # the pandas/polars mix-up still wins for a pandas Series, and the LAST error is the one explained
    assert O._frame_hint(_ON_PANDAS.format(attr="alias", kind="Series")).startswith("that Series is PANDAS")
    both = _ON_POLARS.format(attr="to_numpy", kind="Expr") + _ON_NUMPY.format(attr="rolling")
    assert O._frame_hint(both).startswith("that is a NUMPY array")
    note = O._failure_note(_ON_POLARS.format(attr="over", kind="Series"))
    assert note.startswith("the script failed -- see stderr. Hint: that Series is polars DATA")


def test_a_library_smoke_test_on_a_numpy_array_carries_the_hint():
    from app import library as L

    req = L.SaveModule(name="skewz", code="def signal(df):\n    return df\n", test_code="skewz.signal(df)")
    out = L._usage_hint({"id": "p1"}, req, _ON_NUMPY.format(attr="rolling"))
    assert out.startswith("that is a NUMPY array") and "pd.Series(x, index=df.index).rolling(" in out


def test_a_failed_experiment_carries_the_hint(tmp_path, monkeypatch):
    """... and a failed run_python's result, as `hint` (absent when there is nothing to say)."""
    obj = {"id": "o1", "project_id": "p1", "split_date": None, "metric": {"kind": "sharpe"}}
    monkeypatch.setattr(O, "get_objective", lambda oid: obj)
    monkeypatch.setattr(O.projects, "get", lambda pid: {"id": "p1", "data_dir": str(tmp_path)})
    stderr = [_ON_PANDAS.format(attr="with_columns", kind="DataFrame")]

    async def run(*a, **k):
        return {"ok": False, "stdout": "", "stderr": stderr[0], "artifacts": [], "duration_s": 1.0}

    monkeypatch.setattr(O, "_run_forecasting", run)
    out = asyncio.run(O.scratch_python("o1", O.Scratch(code="x")))
    assert out["ok"] is False and out["hint"].startswith("that DataFrame is PANDAS")
    stderr[0] = "ZeroDivisionError: division by zero"
    assert "hint" not in asyncio.run(O.scratch_python("o1", O.Scratch(code="x")))


def test_a_mistyped_table_is_answered_with_the_tables_there_are():
    """Bug #103: only the first line of DuckDB's error was kept -- "Table with name
    sql_exports_dbo_gex_bar10s does not exist!" -- so the agent was not told the name it meant."""
    from app import datasource as D

    con = duckdb.connect(":memory:")
    con.execute('CREATE VIEW "sql_exports_dbo_gexbar10s" AS SELECT 1 AS "SlotUtc", 2 AS "Close"')
    views = ["sql_exports_dbo_gexbar10s", "fc_imb_oinet_d0_forecast_3"]

    def err(sql: str) -> str:
        with pytest.raises(duckdb.Error) as exc:
            con.execute(sql)
        return D.sql_error(exc.value, views)

    msg = err('SELECT * FROM "sql_exports_dbo_gex_bar10s"')
    assert msg.startswith("Catalog Error: Table with name sql_exports_dbo_gex_bar10s does not exist!")
    assert 'Did you mean "sql_exports_dbo_gexbar10s"?' in msg and "fc_imb_oinet_d0_forecast_3" in msg
    far = err("SELECT * FROM fc_trades")                  # nothing close: the list, not DuckDB's "pg_tables"
    assert "Did you mean" not in far and "pg_tables" not in far and "Tables you can query: sql_exports" in far
    col = err('SELECT "Slot" FROM sql_exports_dbo_gexbar10s')
    assert col.startswith("Binder Error") and 'Candidate bindings: "SlotUtc"' in col
    assert "\n" not in msg + far + col
    con.close()


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


# ---------------------------------------------------------------------------------------
# Robust ranking: the weaker period, times how smooth the whole equity curve is
# ---------------------------------------------------------------------------------------
def _days(n: int, start: str = "2023-01-01") -> list[str]:
    import datetime as dt

    d0 = dt.date.fromisoformat(start)
    return [(d0 + dt.timedelta(days=i)).isoformat() for i in range(n)]


def _rank_obj(rank: str = "robust") -> dict:
    return {"split_date": "2023-09-01", "metric": {"kind": "sharpe", "periods_per_year": 252,
                                                    "min_active_days": 5, "rank": rank}}


def test_smoothness_steady_climb_vs_late_burst():
    steady = [0.001 + 0.0005 * ((i % 3) - 1) for i in range(300)]
    late = [0.0] * 250 + [0.01] * 50
    assert O._smoothness(steady) > 0.95
    assert O._smoothness(late) < O._smoothness(steady)
    assert O._smoothness([-x for x in steady]) < -0.95


def test_robust_score_prefers_steady_over_lucky_holdout():
    import random as rnd

    rnd.seed(1)
    days = _days(360)
    # Loses in-sample, surges after the split (the #1031 shape).
    lucky = [[d, (-0.001 if d < "2023-09-01" else 0.004) + rnd.gauss(0, 0.006)] for d in days]
    steady = [[d, 0.0012 + rnd.gauss(0, 0.006)] for d in days]
    s_lucky, _, _, m_lucky = O._score_returns(_rank_obj(), lucky)
    s_steady, _, _, m_steady = O._score_returns(_rank_obj(), steady)
    assert m_lucky["holdout"]["sharpe"] > m_steady["holdout"]["sharpe"]   # holdout alone picks the wrong one
    assert s_steady > s_lucky                                           # robust picks the steady one
    assert s_lucky < 0 and m_lucky["rank"]["weaker"] == "in_sample"
    assert math.isclose(s_steady, m_steady["rank"]["base"] * m_steady["rank"]["smoothness"], rel_tol=1e-4)


def test_holdout_ranking_still_available():
    rets = [[d, 0.001 * ((i % 5) - 1)] for i, d in enumerate(_days(360))]
    score, _, _, m = O._score_returns(_rank_obj("holdout"), rets)
    assert score == m["holdout"]["sharpe"] and "rank" not in m


# ---------------------------------------------------------------------------------------
# Regime segments for the equity chart
# ---------------------------------------------------------------------------------------
def test_regime_segments_label_days_and_split_stats(tmp_path):
    import datetime as dt

    import polars as pl

    ft_dir = tmp_path / ".ft"
    ft_dir.mkdir()
    t0 = dt.datetime(2023, 8, 28, 9, 30)
    t = [t0 + dt.timedelta(hours=8 * i) for i in range(6 * 3)]          # 3 bars a day, 6 days
    labels = ["low", "low", "high"] * 3 + ["high", "high", "low"] * 3     # days 1-3 low, 4-6 high
    pl.DataFrame({"t": t, "label": labels, "value": list(range(len(t)))}).write_parquet(ft_dir / "regime.parquet")
    days = sorted({d.strftime("%Y-%m-%d") for d in t})
    returns = [[d, 0.01 * (i + 1)] for i, d in enumerate(days)]
    obj = {"split_date": days[4], "metric": {"periods_per_year": 252}}
    out = O._regime_segments(obj, {"run_dir": str(tmp_path)}, {"name": "vol", "routes": {"low": "mom"}}, returns)
    labels_by_day = dict(out["days"])
    assert out["name"] == "vol" and out["routes"] == {"low": "mom"}
    assert len(out["signal"]) == len(labels_by_day)
    assert {labels_by_day[d] for d in days[:3]} == {"low"} or labels_by_day[days[0]] == "low"
    assert out["by_label"]["high"]["holdout"]["days"] >= 1          # the split separates the stats
    assert O._regime_segments(obj, {"run_dir": str(tmp_path / "none")}, {}, returns) is None


# ---------------------------------------------------------------------------------------
# Both sides: long and short trades are counted, and a one-sided candidate can go unranked
# ---------------------------------------------------------------------------------------
def test_mark_to_market_counts_in_sample_trades_per_side(tmp_path):
    _write_prices(tmp_path)
    pos = tmp_path / "pos.parquet"
    # Day 1 (in-sample): long, flip short, flat. Day 2 is the holdout and is not counted.
    _write_positions(pos, [("2024-01-02 10:00", 1.0), ("2024-01-02 10:01", -1.0), ("2024-01-02 10:02", 0.0),
                           ("2024-01-03 10:00", -2.0)])
    _, info = O._mark_to_market(_obj(cost_bps=0.0), str(tmp_path), pos)
    assert info["sides"] == {"long": 1, "short": 1}


def test_side_gap_requires_each_side_share():
    m = {"min_side_share": 0.2}
    assert O._side_gap(m, {"long": 48, "short": 0}).startswith("one-sided: 48 long and 0 short")
    assert O._side_gap(m, {"long": 40, "short": 10}) is None          # 20% short
    assert O._side_gap(m, {"long": 5, "short": 45}) is not None       # too few longs
    assert O._side_gap({"min_side_share": 0.0}, {"long": 48, "short": 0}) is None
    assert O._side_gap({**m, "direction": "long"}, {"long": 48, "short": 0}) is None
    assert O._side_gap(m, None) is None                                # not measured


def test_one_sided_candidate_is_not_ranked():
    obj = {**_rank_obj(), "metric": {**_rank_obj()["metric"], "min_side_share": 0.2}}
    days = _days(300)
    returns = [[d, 0.001 * (1 if i % 3 else -0.5)] for i, d in enumerate(days)]
    score, is_score, note, metrics = O._score_returns(obj, returns, {"long": 48, "short": 0})
    assert score is None and note.startswith("one-sided") and metrics["one_sided"] == note
    assert is_score is not None
    ranked, *_ = O._score_returns(obj, returns, {"long": 30, "short": 20})
    assert ranked is not None


# ---------------------------------------------------------------------------------------
# Swing legs: hits for the right side of each move, misses for the wrong side
# ---------------------------------------------------------------------------------------
def test_zigzag_finds_legs_that_moved_the_threshold():
    import numpy as np

    # Up 2%, down 2%, then a 0.1% wiggle that is too small to be a leg.
    p = np.array([100, 101, 102, 101, 100.5, 99.96, 100.06])
    assert O._zigzag(p, 0.01) == [(0, 2, 1), (2, 5, -1)]


def test_swings_score_long_up_short_down_and_always_long_nets_zero():
    import numpy as np

    t = np.arange(6, dtype=float) * 60 + 86400 * 19000        # one day
    p = np.array([100, 101, 102, 101, 100, 100.0])
    # Position set at bar k is held over bar k+1: long for the up leg, short for the down leg.
    right = O._swings(t, p, np.array([1, 1, -1, -1, 0, 0.0]), 0.01, None)["full"]
    assert (right["hits"], right["misses"], right["net"], right["capture"]) == (2, 0, 2, 1.0)
    assert right["up_caught_long"] == right["down_caught_short"] == 1
    drift = O._swings(t, p, np.ones(6), 0.01, None)["full"]      # always long
    assert (drift["hits"], drift["misses"], drift["net"]) == (1, 1, 0)
    assert drift["down_while_long"] == 1 and abs(drift["capture"]) < 0.02   # +2% vs -1.96%
    wrong = O._swings(t, p, -np.array([1, 1, -1, -1, 0, 0.0]), 0.01, None)["full"]
    assert wrong["net"] == -2


def test_mark_to_market_reports_swings(tmp_path):
    _write_prices(tmp_path)
    pos = tmp_path / "pos.parquet"
    _write_positions(pos, [("2024-01-02 10:00", 1.0)])
    _, info = O._mark_to_market(_obj(cost_bps=0.0), str(tmp_path), pos)
    sw = info["_swings"]
    # Each day climbs 3% in a straight line: one up leg a day, held long -> a hit, split in two.
    assert sw["in_sample"]["up_caught_long"] == 1 and sw["holdout"]["up_caught_long"] == 1


def test_a_report_call_inside_a_function_never_called_is_named():
    """#1770, #1745, #1744: `def main(): ... ft.report_positions(pos)` and no main() call."""
    never = "import ft\ndef helper(x):\n    return x\ndef main():\n    ft.report_positions(helper(1))\n"
    assert O._uncalled_reporter(never) == "main"
    assert "defines main() with the ft.report_* call inside but never calls it" in \
        O._nothing_reported("positions", "ft.report_positions(series)", never)
    for ok in (never + "main()\n", never + 'if __name__ == "__main__":\n    main()\n',
               never + "def run():\n    main()\nrun()\n", "import ft\nft.report_positions(1)\n", "def (:\n", ""):
        assert O._uncalled_reporter(ok) is None, ok
    assert O._nothing_reported("positions", "ft.report_positions(series)", "x = 1\n") == \
        "no positions reported -- call ft.report_positions(series)"


def test_operator_precedence_mistakes_get_the_parentheses_hint():
    """Bug #168: `a > 0 & b < 1` -- pandas' 'rand_' message names neither & nor the cause."""
    for err in ("TypeError: Cannot perform 'rand_' with a dtyped [float64] array and scalar of type [bool]",
                "TypeError: cannot perform 'ror_' with a dtyped [float64] array and scalar of type [bool]",
                "TypeError: unsupported operand type(s) for &: 'float' and 'bool'"):
        assert "(a > 0) & (b < 1)" in O._failure_note(_HEAD + err), err
    assert "Hint" not in O._failure_note(_HEAD + "TypeError: unsupported operand type(s) for +: 'int' and 'str'")


def test_a_datetime_parsed_as_text_and_a_missing_nth_get_hints_in_every_tool():
    """Bug #134: pl.col('t').str.strptime(...) on the already-datetime `t` (16 times); Expr.nth
    (10-01 09:27). run_python used only _frame_hint, so the precedence hint never reached it."""
    schema = (_HEAD + '  File "candidate.py", line 9, in <module>\n    rows = rows.with_columns(\n'
              "polars.exceptions.SchemaError: invalid series dtype: expected `String`, got `datetime[ns]` for series "
              "with name `t`\n")
    hint = O.error_hint(schema)
    assert "`t` is already a polars Datetime" in hint and "pl.col('t').dt.date()" in hint and "ft.clock(rows['t'])" in hint
    assert O._failure_note(schema).endswith(hint)
    assert "pl.col('x').get(0).over('session')" in O.error_hint(_ON_POLARS.format(attr="nth", kind="Expr"))
    assert "(a > 0) & (b < 1)" in O.error_hint(_HEAD + "TypeError: unsupported operand type(s) for &: 'float' and 'bool'")
    assert O.error_hint(_HEAD + "SchemaError: invalid series dtype: expected `String`, got `i64` for series with name `x`") == ""


def test_the_new_crash_kinds_of_10_01_get_hints():
    """After the 09:12 restart: .list() called, nulls into int(), a column not loaded, .alias on a frame."""
    h = O.error_hint(_HEAD + "TypeError: 'ExprListNameSpace' object is not callable\n")
    assert "pl.col(...).list is a namespace" in h and ".last().over('session')" in h
    h = O.error_hint(_HEAD + "TypeError: int() argument must be a string, a bytes-like object or a real number, "
                             "not 'NoneType'\n")
    assert "fill_null(False)" in h
    h = O.error_hint(_HEAD + 'polars.exceptions.ColumnNotFoundError: unable to find column "Volume"; valid columns: '
                             '["t", "Close", "GEX"]\n')
    assert "'Volume'" in h and "columns=[...]" in h
    whole = ", ".join(f'"c{i}"' for i in range(140))   # the whole dataset listed: the name is just wrong
    assert O.error_hint(_HEAD + f'ColumnNotFoundError: unable to find column "VWAP"; valid columns: [{whole}]\n') == ""
    h = O.error_hint(_ON_POLARS.format(attr="alias", kind="DataFrame"))
    assert "whole polars DataFrame and .alias belongs to one column" in h
    assert O.error_hint(_ON_POLARS.format(attr="sort_values", kind="DataFrame")).startswith("that DataFrame is POLARS")


def test_two_expressions_with_one_name_get_the_alias_hint():
    """10-01 11:40: df.select(pl.col(c).quantile(0.1), pl.col(c).quantile(0.9)) -- polars says only
    'projections contained duplicate output name'."""
    err = (_HEAD + "polars.exceptions.DuplicateError: projections contained duplicate output name "
           "'Pinning_NearestWallDistZ'. It's possible that multiple expressions are returning the same default "
           "column name.\n")
    h = O.error_hint(err)
    assert "both produce a column named 'Pinning_NearestWallDistZ'" in h
    assert ".alias('Pinning_NearestWallDistZ_q10')" in h
    assert O._failure_note(err).endswith(h)


def test_the_crash_kinds_of_10_01_noon_get_hints():
    """12:08-12:22: a shortened column inside with_columns, rolling_quantile(0.9, 20), ft.clock read
    as a dict, and an .alias on the divisor -- each died with polars' words only."""
    h = O.error_hint(_HEAD + "polars.exceptions.ShapeError: can't broadcast Series 'ret_1' of length 496481 to "
                             "length 496482\n")
    assert "'ret_1' came out with 496481 rows for a frame of 496482" in h and ".drop_nulls('fwd')" in h
    err = (_HEAD + '  File "candidate.py", line 13, in <module>\n'
           '  File "/usr/local/lib/python3.12/site-packages/polars/_utils/deprecation.py", line 132, in wrapper\n'
           '  File "/usr/local/lib/python3.12/site-packages/polars/expr/expr.py", line 9530, in rolling_quantile\n'
           "TypeError: 'int' object is not an instance of 'str'\nwhile processing 'interpolation'\n")
    h = O.error_hint(err)
    assert "polars .rolling_quantile: its 'interpolation' parameter got a int" in h and "window_size" in h
    err = (_HEAD + '  File "/usr/local/lib/python3.12/site-packages/polars/expr/expr.py", line 1, in rolling_mean\n'
           "TypeError: argument 'weights': 'int' object cannot be converted to 'Sequence'\n")
    assert "its 'weights' parameter got a int" in O.error_hint(err)
    h = O.error_hint(_HEAD + "TypeError: tuple indices must be integers or slices, not str\n")
    assert "session, minute = ft.clock(rows['t'])" in h
    missing = (_HEAD + 'polars.exceptions.ColumnNotFoundError: unable to find column "pb_z"; valid columns: '
               '["t", "Pressure_Below", "pb_mean", "pb_std"]\n')
    code = "rows.with_columns((pl.col('Pressure_Below') - pl.col('pb_mean')) / pl.col('pb_std').alias('pb_z'))"
    h = O.error_hint(missing, code)
    assert "writes .alias('pb_z')" in h and "((a - b) / c).alias('pb_z')" in h
    assert O._failure_note(missing, code).endswith(h)
    assert "columns=[...]" in O.error_hint(missing)                 # without the script: the general hint


def test_shifted_slices_and_date_diffs_get_hints():
    """10-01 14:10 np.corrcoef(z[:-h], fwd[:-h]) with fwd already n-h long; 14:16 date diff .fill_null(0)."""
    h = O.error_hint(_HEAD + "ValueError: all the input array dimensions except for the concatenation axis must match "
                             "exactly, but along dimension 1, the array at index 0 has size 496452 and the array at "
                             "index 1 has size 496422\n")
    assert "(496452 and 496422, 30 apart)" in h and "ALREADY n-h long" in h
    h = O.error_hint(_HEAD + "polars.exceptions.InvalidOperationError: got invalid or ambiguous dtypes: "
                             "'[duration[μs], dyn int]' in expression 'fill_null'\n")
    assert "is a Duration" in h and ".cum_sum()" in h


def test_clashing_forecast_columns_get_the_hint_in_run_python_too():
    """10-01 14:31: two forecasts merged in run_python -- the hint used to reach only submissions."""
    h = O.error_hint(_HEAD + "KeyError: 'fc_median'\n")
    assert "fc_median_x / fc_median_y" in h and 'prefix="x_"' in h


def test_shift_by_a_column_gets_the_hint():
    """10-01 14:37: pl.col('Close').shift(-pl.col(...)) -- 'n' must be a scalar value."""
    h = O.error_hint(_HEAD + "polars.exceptions.ShapeError: 'n' must be a scalar value\n")
    assert "ONE number" in h and ".last().over('session')" in h


def test_a_mask_of_the_wrong_length_gets_the_hint():
    """10-01 14:49: range_z[mask][valid] with `valid` over all 496482 rows and the slice one session's 2371."""
    h = O.error_hint(_HEAD + "IndexError: boolean index did not match indexed array along axis 0; size of axis is "
                             "2371 but size of corresponding boolean axis is 496482\n")
    assert "mask of 496482 rows was used on an array of 2371" in h and "writes into a COPY" in h


def test_a_conditional_in_a_format_spec_gets_the_hint():
    """10-01 15:03: f"{corr30:.4f if corr30 else 'N/A'}"."""
    h = O.error_hint(_HEAD + "ValueError: Invalid format specifier '.4f if corr30 else 'N/A'' for object of type 'float'\n")
    assert "is the FORMAT" in h


def test_lstsq_on_nan_and_the_new_pandas_keyerror_get_hints():
    """10-01 15:08: np.linalg.lstsq over windows with NaN; the forecast-merge hint still matches ft's
    reworded pandas KeyError."""
    assert "only finite rows" in O.error_hint(_HEAD + "numpy.linalg.LinAlgError: SVD did not converge in Linear Least Squares\n")
    h = O.error_hint(_HEAD + "KeyError: \"'fc_median' -- the frame has no such column; did you mean 'fc_median_x'?\"\n")
    assert "fc_median_x / fc_median_y" in h


def test_the_15_28_batch_gets_hints():
    """Two unaliased expressions named Close, text into :.2f, and a per-row loop killed at the time limit."""
    h = O.error_hint(_HEAD + "polars.exceptions.ComputeError: the name 'Close' passed to `LazyFrame.with_columns` is duplicate\n")
    assert "both named 'Close'" in h and ".alias('fwd6')" in h
    assert "TEXT value" in O.error_hint(_HEAD + "ValueError: Unknown format code 'f' for object of type 'str'\n")
    assert "for-loop over every row" in O.error_hint("some output\n\n[killed: exceeded the 180s limit]")


def test_existing_hints_match_fts_reworded_missing_column():
    """ft words a missing polars column '"x" not found -- did you mean ...? The frame has: ...' (10-01 15:06):
    the forecast-clash, misplaced-.alias and subset hints still fire on it."""
    h = O.error_hint(_HEAD + 'polars.exceptions.ColumnNotFoundError: "fc_median" not found -- did you mean '
                             "'fc_median_x'? The frame has: t, fc_median_x, fc_median_y\n")
    assert "fc_median_x / fc_median_y" in h
    err = _HEAD + 'polars.exceptions.ColumnNotFoundError: "pb_z" not found The frame has: t, Pressure_Below, pb_std\n'
    code = "rows.with_columns((pl.col('Pressure_Below') - pl.col('pb_mean')) / pl.col('pb_std').alias('pb_z'))"
    assert "writes .alias('pb_z')" in O.error_hint(err, code)
    assert "columns=[...]" in O.error_hint(err)
    assert "writes .alias('pb_z')" in O.error_hint(_HEAD + 'ColumnNotFoundError: "pb_z" not found\n', code)
    many = ", ".join(f"c{i}" for i in range(40)) + ", ..."            # ft cuts the list at 40: not a chosen subset
    assert O.error_hint(_HEAD + f'ColumnNotFoundError: "zz" not found The frame has: {many}\n') == ""


def test_trade_review_features_asked_of_the_rows_get_their_formula():
    """10-01 00:42-03:59, 4 runs: ft.rows_pl(columns=[..., 'GexFlip_Pos_vs_price_bps', 'minutes_into_session'])."""
    whole = ", ".join(f'"c{i}"' for i in range(140))
    h = O.error_hint(_HEAD + f'polars.exceptions.ColumnNotFoundError: unable to find column "GexFlip_Pos_vs_price_bps"; '
                             f'valid columns: [{whole}]\n')
    assert "TRADE REVIEW" in h and "(pl.col('GexFlip_Pos') / pl.col('Close') - 1) * 1e4" in h
    h = O.error_hint(_HEAD + "KeyError: \"ft.rows(columns=...): the rows have no 'Pinning_ProbTrend' (did you mean "
                             "'PinTrend_ProbTrend'?); 'minutes_into_session' -- ft.task()['columns'] lists every column\"\n")
    assert "'minutes_into_session' is a feature" in h and ".dt.total_minutes()" in h
    assert "TRADE REVIEW" not in O.error_hint(_HEAD + "KeyError: \"ft.rows(columns=...): the rows have no 'Foo'\"\n")


def test_polars_calls_with_pandas_arguments_get_the_real_signature():
    """10-01 00:4x: .clip(lower=1e-9), .rolling_std(20).max(1e-9), .rolling(20, min_periods=1); merge(tolerance=)."""
    h = O.error_hint(_HEAD + "TypeError: Expr.clip() got an unexpected keyword argument 'lower'\n")
    assert "lower= is spelled lower_bound=" in h and ".clip(lower_bound=None, upper_bound=None)" in h
    h = O.error_hint(_HEAD + "TypeError: Expr.max() takes 1 positional argument but 2 were given\n")
    assert ".clip(lower_bound=1e-9)" in h and "pl.max_horizontal" in h
    for err in ("TypeError: Expr.rolling() got an unexpected keyword argument 'min_periods'",
                "TypeError: Expr.rolling() missing 1 required keyword-only argument: 'period'"):
        assert ".rolling_quantile(q, window_size=n)" in O.error_hint(_HEAD + err + "\n")
    h = O.error_hint(_HEAD + "TypeError: DataFrame.merge() got an unexpected keyword argument 'tolerance'\n")
    assert "pd.merge_asof" in h
    # pandas' sort_values takes ascending=, polars' sort does not: the frame was polars
    h = O.error_hint(_HEAD + "TypeError: DataFrame.sort() got an unexpected keyword argument 'ascending'\n")
    assert h.startswith("polars .sort(): ascending= is spelled descending=") and "opposite" in h
    assert O.error_hint(_HEAD + "TypeError: DataFrame.fillna() got an unexpected keyword argument 'zzz'\n").startswith(
        "pandas .fillna(): it has no zzz=")


def test_namespace_and_uncalled_method_mistakes_get_hints():
    """10-01: .dt.cast(pl.Date), .dt.with_time_zone, diff().dt.seconds(), a stray .str, and .dt.date.alias."""
    h = O.error_hint(_HEAD + "AttributeError: 'ExprDateTimeNameSpace' object has no attribute 'cast'\n")
    assert "pl.col('x').cast(pl.Date)" in h
    h = O.error_hint(_HEAD + "AttributeError: 'DateTimeNameSpace' object has no attribute 'seconds'. Did you mean: "
                             "'second'?\n")
    assert ".dt.total_seconds()" in h and "FIELD" in h
    h = O.error_hint(_HEAD + "AttributeError: 'ExprDateTimeNameSpace' object has no attribute 'with_time_zone'\n")
    assert "convert_time_zone('America/New_York')" in h
    h = O.error_hint(_HEAD + "AttributeError: 'StringNameSpace' object has no attribute 'alias'\n")
    assert ".str on its own is a namespace" in h
    h = O.error_hint(_HEAD + "AttributeError: 'function' object has no attribute 'alias'\n")
    assert "pl.col('t').dt.date().alias(...)" in h
    h = O.error_hint(_HEAD + "AttributeError: module 'polars' has no attribute 'cut'\n")
    assert "pl.col('x').cut(...)" in h
    assert O.error_hint(_HEAD + "AttributeError: module 'polars' has no attribute 'zzzz'\n") == ""


def test_the_archive_sweep_one_cause_failures_get_hints():
    """One-cause failures of 09-30..10-01 that went without a hint."""
    def hint(line, code=""):
        return O.error_hint(_HEAD + line + "\n", code)

    assert "every='15m'" in hint("polars.exceptions.InvalidOperationError: unit: 'min' not supported; available "
                                 "units are: 'y', 'mo', 'q', 'w', 'd', 'h', 'm', 's', 'ms', 'us', 'ns'")
    for line in ("TypeError: the truth value of an Expr is ambiguous",
                 "ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()"):
        assert "(pl.col('a') == pl.col('b')) & (pl.col('c') != 0)" in hint(line)
    assert "datetime.date(2024, 7, 19)" in hint("TypeError: Date() takes no arguments")
    assert "pl.duration(minutes=60)" in hint("polars.exceptions.InvalidOperationError: + not allowed on datetime[ns] "
                                             "and dyn int")
    assert ".over('session')" in hint("polars.exceptions.InvalidOperationError: At least one of `partition_by` and "
                                      "`order_by` must be specified in `over`")
    assert "score['sharpe']" in hint("TypeError: unsupported format string passed to dict.__format__")
    assert "reduce it to one number" in hint("TypeError: unsupported format string passed to Series.__format__")
    assert ".copy()" in hint("ValueError: assignment destination is read-only")
    assert "%%" in hint("ValueError: unsupported format character 'n' (0x6e) at index 16")
    assert "df.select(expr).to_series()" in hint("TypeError: Series constructor called with unsupported type 'Expr' "
                                                 "for the `values` parameter")
    assert "df.select(expr).to_series()" in hint("TypeError: float() argument must be a string or a real number, "
                                                 "not 'Expr'")
    polars = '  File "/usr/local/lib/python3.12/site-packages/polars/series/series.py", line 1200, in _arithmetic\n'
    assert ".cast(pl.Int32) * 3600" in hint(polars + "OverflowError: number too large to fit in target type")
    assert hint("OverflowError: number too large to fit in target type") == ""      # not polars: no claim
    assert "do not fit u16" in hint("polars.exceptions.InvalidOperationError: conversion from `u32` to `u16` failed "
                                    "in column 'Bsa_SessionCum' for 427627 out of 496482 values")
    assert ".mean() / .std()" in hint("TypeError: object of type 'Rolling' has no len()")
    assert "np.where(cond, 1.0, 0.0)" in hint("TypeError: Invalid value 'False' for dtype 'float64'")
    assert "pd.to_datetime(s)" in hint("AttributeError: Can only use .dt accessor with datetimelike values")
    assert "row NUMBERS" in hint("AttributeError: 'int' object has no attribute 'date'")
    assert "pd.to_datetime(df['x'])" in O.error_hint(_ON_PANDAS.format(attr="to_datetime", kind="DataFrame"))
    raw = ('polars.exceptions.ColumnNotFoundError: unable to find column "t"; valid columns: ["Symbol", "SlotUtc", '
           '"Close"]')
    assert "time column is 'SlotUtc'" in hint(raw)


def test_a_column_moved_into_the_index_and_nulls_in_numpy_get_hints():
    """10-01 05:16: df.resample('15min', on='SlotUtc').agg(...) then bars['SlotUtc']; a library module's
    np.cumsum over a polars comparison whose first row is null."""
    code = "bars = df.resample('15min', on='SlotUtc', label='right').agg({'Close': 'last'})\nx = bars['SlotUtc']"
    err = _HEAD + "KeyError: \"'SlotUtc' -- the frame has no such column; did you mean 'Close'?\"\n"
    assert "make it the INDEX" in O.error_hint(err, code)
    assert O.error_hint(err, code.replace("})\n", "}).reset_index()\n")) == ""
    assert O.error_hint(err) == ""
    numpy = '  File "/usr/local/lib/python3.12/site-packages/numpy/_core/fromnumeric.py", line 43, in _wrapit\n'
    assert "fill_null(False)" in O.error_hint(_HEAD + numpy + "TypeError: unsupported operand type(s) for +: "
                                                              "'NoneType' and 'bool'\n")
    assert O.error_hint(_HEAD + "TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'\n") == ""


def test_alias_on_a_number_gets_the_parentheses_hint():
    """10-01 16:32: (a - b) / b * 10000 <newline> .alias('ret') -- the alias went to 10000."""
    h = O.error_hint(_HEAD + "AttributeError: 'int' object has no attribute 'alias'\n")
    assert "plain NUMBER" in h and "* 10000).alias" in h


def test_a_window_expression_that_changes_length_gets_the_hint():
    """10-01 22:19: pl.col('Close').head(30).over('session')."""
    h = O.error_hint(_HEAD + "polars.exceptions.ShapeError: the length of the window expression did not match that of the group\n")
    assert "ONE value per row" in h and "gather(29).over('session')" in h



def test_union_aligned_lengths_and_timestamp_division_get_hints():
    """10-02 00:06: p (per bar) * scale (by day of bar) -> 2N values; Timestamp / Timestamp."""
    h = O.error_hint(_HEAD + "ValueError: Length of values (992964) does not match length of index (496482)\n")
    assert "exactly twice the frame" in h and "UNION of their labels" in h
    assert "different" not in O.error_hint(_HEAD + "ValueError: Length of values (10) does not match length of index (12)\n").lower()
    assert "pd.Timedelta('1min')" in O.error_hint(_HEAD + "TypeError: unsupported operand type(s) for /: 'Timestamp' and 'Timestamp'\n")
