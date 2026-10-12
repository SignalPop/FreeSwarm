"""The data audit (taskkit.audit, behind every server's harness_data_audit): a feed that dies,
freezes or skips a day inside the data is found from the rows alone, and the verified date range
stops where the data stops being trustworthy."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from taskkit.audit import audit_rows
from taskkit.task import ts_ns
from test_interface import provider  # noqa: F401 -- the fixture: every server (gex, battery, tables)

ROWS = 390                                   # one row a minute, 13:30-20:00 UTC


def _days(n: int) -> list[dt.date]:
    out, d = [], dt.date(2024, 1, 2)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def _bars(n_days: int = 60, seed: int = 1) -> tuple[pl.DataFrame, list[dt.date]]:
    """Clean minute bars: a price, a signal that moves all day, a level that changes once a day,
    and a patchy column (a third wall that exists on most days only)."""
    rng = np.random.default_rng(seed)
    days = _days(n_days)
    t = np.concatenate([np.datetime64(d) + np.timedelta64(13 * 60 + 30, "m") + np.arange(ROWS) * np.timedelta64(1, "m")
                        for d in days]).astype("datetime64[ns]")
    day_i = np.repeat(np.arange(n_days), ROWS)
    patchy = rng.normal(size=len(t))
    patchy[np.isin(day_i, rng.choice(n_days, n_days // 4, replace=False))] = np.nan
    return pl.DataFrame({
        "t": t, "Close": 400 * np.exp(np.cumsum(rng.normal(0, 1e-4, len(t)))),
        "GEX": rng.normal(size=len(t)), "Level": np.repeat(rng.normal(size=n_days), ROWS), "Wall3": patchy,
    }), days


def _break(df: pl.DataFrame, days: list[dt.date], col: str, frm: int, value=None) -> pl.DataFrame:
    """`col` empty (value None) or stuck at `value` from day index `frm` on."""
    after = pl.col("t") >= pl.lit(dt.datetime.combine(days[frm], dt.time()))
    return df.with_columns(pl.when(after).then(pl.lit(value, dtype=pl.Float64)).otherwise(pl.col(col)).alias(col))


def test_clean_data_is_verified_end_to_end():
    df, days = _bars()
    r = audit_rows(df, "Close", ts_ns(days[40].isoformat()))
    assert r["status"] == "ok", r["summary"]
    assert r["verified"]["from"] == days[0].isoformat() and r["verified"]["to"] == days[-1].isoformat()
    assert r["holdout"]["unverified_days"] == 0 and not r["exceptions"]
    assert "Level" not in str(r["exceptions"])                       # a daily level is not "frozen"
    assert [c["column"] for c in r["intermittent_columns"]] == ["Wall3"]   # patchy by nature: reported only


def test_a_feed_that_dies_in_the_holdout_ends_the_verified_range():
    df, days = _bars()
    df = _break(df, days, "GEX", 45)                                  # empty from day 45
    r = audit_rows(df, "Close", ts_ns(days[40].isoformat()))
    assert r["verified"]["to"] == days[44].isoformat()
    assert r["holdout"]["unverified_days"] == 15 and r["holdout"]["unverified_after"] == days[44].isoformat()
    assert r["status"] == "bad" and "holdout" in r["summary"]
    assert r["columns_end"] == [{"day": days[44].isoformat(), "count": 1, "columns": ["GEX"]}]
    assert r["excluded"][-1]["columns"][0]["column"] == "GEX"


def test_a_column_frozen_at_one_value_counts_as_dead():
    df, days = _bars()
    df = _break(df, days, "GEX", 50, 1.25)                            # stuck, not empty
    r = audit_rows(df, "Close")
    assert r["verified"]["to"] == days[49].isoformat()
    assert r["excluded"][-1]["columns"][0]["frozen_days"] == 10


def test_short_and_broken_single_days_are_exceptions_inside_the_range():
    df, days = _bars()
    d10, d20 = (dt.datetime.combine(days[i], dt.time()) for i in (10, 20))
    one = pl.col("t").is_between(d10, d10 + dt.timedelta(days=1), closed="left")
    df = df.with_columns(pl.when(one).then(None).otherwise(pl.col("GEX")).alias("GEX"))   # one empty day
    df = df.filter(~pl.col("t").is_between(d20 + dt.timedelta(hours=13, minutes=37), d20 + dt.timedelta(days=1)))
    r = audit_rows(df, "Close")
    assert r["verified"]["from"] == days[0].isoformat() and r["verified"]["to"] == days[-1].isoformat()
    why = {e["day"]: e["why"] for e in r["exceptions"]}
    assert set(why) == {days[10].isoformat(), days[20].isoformat()}
    assert "GEX" in why[days[10].isoformat()] and "rows of a typical" in why[days[20].isoformat()]
    assert r["status"] == "warn"


def test_a_late_starting_column_moves_the_start_of_the_range():
    df, days = _bars()
    before = pl.col("t") < pl.lit(dt.datetime.combine(days[30], dt.time()))
    df = df.with_columns(pl.when(before).then(None).otherwise(pl.col("GEX")).alias("GEX"))
    r = audit_rows(df, "Close")
    assert r["verified"]["from"] == days[30].isoformat()
    assert r["columns_start"][0]["columns"] == ["GEX"]


def test_ignored_columns_are_left_out():
    df, days = _bars()
    df = _break(df, days, "GEX", 45)
    r = audit_rows(df, "Close", settings={"ignore_columns": ["GE*"]})
    assert r["verified"]["to"] == days[-1].isoformat()


def test_every_server_audits_its_rows(provider, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setenv("TASKKIT_CACHE", str(tmp_path / "cache"))
    p, task = provider
    r = p.harness_data_audit(task)
    assert r["status"] in ("ok", "warn", "bad")
    assert r["verified"]["from"] <= r["verified"]["to"]
    d = p.task_describe(task, None)
    if d.get("holdout_from"):
        assert r["holdout"]["from"] == d["holdout_from"][:10]
    assert p.harness_data_audit(task) == r                           # the second call is the cached report
