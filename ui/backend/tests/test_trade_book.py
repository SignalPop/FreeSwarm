"""The trade book: trades classed as big winners / big losers / scratch, the review of what the big
winners have in common at entry, and the fence that keeps holdout trades away from agents."""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
import polars as pl
import pytest
from fastapi import HTTPException

from app import objectives as O
from app import task_objectives as T
from app import trade_book as B

DAY = 86_400_000_000_000


def test_trades_are_parsed_deduplicated_and_split_at_the_holdout():
    ev = [{"entry": "2024-07-18T14:00:00", "exit": "2024-07-18T15:00:00", "side": "long", "size": 2.0, "bars": 360,
           "gross": 0.004, "net": 0.003, "open": False},
          {"entry": "2024-07-18T14:00:00", "exit": "2024-07-18T15:00:00", "side": "long", "size": 2.0, "net": 0.003},
          {"entry": "2024-07-19T14:00:00", "exit": "2024-07-19T14:30:00", "side": "short", "size": 1.0, "net": -0.002},
          {"t": "2024-07-19T14:00:00", "action": 1.0}]                    # not a trade
    rows = B.trades_of(ev, "2024-07-19")
    assert len(rows) == 2
    long_, short = rows
    assert long_[3] == 1 and long_[8] == pytest.approx(0.0015) and long_[9] == 0          # per unit, in-sample
    assert short[3] == -1 and short[8] == pytest.approx(-0.002) and short[9] == 1         # holdout


def test_classes_and_the_threshold():
    u = np.array([0.003, 0.0016, 0.0005, -0.0005, -0.002])
    assert B.classes(u, 0.0016).tolist() == [1, 1, 0, 0, -1]
    obj = {"id": "x", "metric": {"big_trade": 0.002}}
    assert B.threshold(obj, u) == (0.002, "set")
    B._ADDITIVE["x"] = False
    obj = {"id": "x", "metric": {"task_info": {"valuation": {"cost_bps": 1.0}}}}
    thr, src = B.threshold(obj, np.full(100, 0.0001))
    assert src == "auto" and thr == pytest.approx(0.0008)            # floored at 8 one-way costs
    thr, src = B.threshold(obj, np.linspace(-0.01, 0.01, 101))
    assert thr == pytest.approx(0.008) and src == "auto"             # the 80th percentile of |result|


def _synthetic(n=1200, seed=1):
    """Trades where big winners come from high `edge`, and `calm` only makes losers rarer."""
    rng = np.random.default_rng(seed)
    edge, calm, noise = rng.random(n), rng.random(n), rng.random(n)
    unit = rng.normal(0, 0.0008, n)
    win = edge > 0.8
    unit[win & (rng.random(n) < 0.7)] += 0.004                        # most high-edge trades are big winners
    loss = (calm < 0.6) & (rng.random(n) < 0.3)
    unit[loss] -= 0.004                                               # losers only happen when it is not calm
    side = np.where(rng.random(n) < 0.5, 1, -1)
    entry = np.sort(rng.integers(0, 300, n)) * DAY + 14 * 3_600_000_000_000
    X = np.column_stack([edge, calm, noise]).astype(np.float32)
    return unit, side, entry, np.full(n, 100.0), X, ["edge", "calm", "noise"]


def test_review_finds_where_big_winners_concentrate_not_merely_fewer_losers():
    unit, side, entry, bars, X, names = _synthetic()
    r = B.review(unit, side, entry, bars, X, names, 0.0025, "America/New_York", 10.0)
    assert r["big_winners"]["n"] > 0 and r["big_losers"]["n"] > 0 and r["scratch"]["n"] > 0
    assert r["big_winners"]["n"] + r["big_losers"]["n"] + r["scratch"]["n"] == len(unit)
    for label in ("long", "short"):
        good = r["sides"][label]["winner_conditions"]
        assert good and good[0]["field"] == "edge" and good[0]["op"] == ">="
        assert good[0]["win"] > 2 * good[0]["win_all"]
        # "calm" only thins out the losers: it is not where the big winners are.
        assert all(c["field"] != "calm" for c in good)
        assert all(c["field"] != "noise" for c in good)
        bad = r["sides"][label]["loser_conditions"]
        assert any(c["field"] == "calm" and c["op"] == "<=" for c in bad)
    text = B.render(r, False, "TRADE REVIEW")
    assert "BIG WINNERS" in text and "edge >=" in text and "Best trades" in text
    assert r["held_unit"] == "minutes" and r["big_winners"]["held"] == pytest.approx(100 * 10 / 60)


def test_a_condition_must_hold_in_both_halves_of_the_period():
    unit, side, entry, bars, X, names = _synthetic(seed=2)
    late = entry > np.median(entry)
    X[late, 0] = np.random.default_rng(3).random(int(late.sum()))   # "edge" means nothing in the second half
    r = B.review(unit, side, entry, bars, X, names, 0.0025)
    for sd in r["sides"].values():
        assert all(c["field"] != "edge" for c in sd["winner_conditions"])


def test_entry_conditions_are_as_of_the_last_row_before_the_entry_the_same_day():
    t = np.array([10, 20, 30, DAY + 5], dtype=np.int64)
    snap = {"t": t, "day": t // DAY, "X": np.array([[1.0], [2.0], [3.0], [4.0]], np.float32), "names": ["f"]}
    X = B.entry_conditions(snap, np.array([20, 31, DAY + 1, DAY + 6], dtype=np.int64))
    assert X[0, 0] == 1.0          # the row AT the entry is not yet known: the one before it
    assert X[1, 0] == 3.0
    assert np.isnan(X[2, 0])       # the last row before is yesterday's: unknown
    assert X[3, 0] == 4.0


def test_snapshots_turn_price_levels_into_distances_and_add_the_price_path(tmp_path):
    t0 = np.datetime64("2024-01-02T14:30:00", "ns")
    n = 400
    t = t0 + np.arange(n) * np.timedelta64(30, "s")
    price = 100 + np.arange(n) * 0.01
    pl.DataFrame({"t": t, "Close": price, "Wall": 101.0 + np.arange(n) % 7 * 0.1,
                  "Sig": np.sin(np.arange(n))}).write_parquet(tmp_path / "r.parquet")
    s = B.snapshots(tmp_path / "r.parquet", "Close", False)
    assert "Wall_vs_price_bps" in s["names"] and "Wall" not in s["names"] and "Close" not in s["names"]
    assert {"Sig", "price_chg_5m_bps", "price_since_open_bps", "minutes_into_session", "price_in_day_range"} <= set(s["names"])
    assert len(s["t"]) == 200                                         # one snapshot per minute
    k = s["names"].index("Wall_vs_price_bps")
    assert s["X"][0, k] == pytest.approx((101.1 / 100.01 - 1) * 1e4, rel=1e-4)   # the minute's last row


def _objective(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "DB_PATH", tmp_path / "o.sqlite3")
    monkeypatch.setattr(O, "_conn", None)
    monkeypatch.setattr(O, "WORK_ROOT", tmp_path / "work")
    B._POOL.clear()
    B._ADDITIVE.clear()
    B._BUILDS.clear()
    now = time.time()
    metric = {"kind": "task", "higher_is_better": True, "task_server": "gex", "task": "t",
              "task_info": {"valuation": {"cost_bps": 1.0}, "display_tz": "UTC"}}
    O.db().execute("INSERT INTO objectives (id, project_id, title, metric, split_date, created_at, updated_at) "
                   "VALUES ('o1', 'p1', 't', ?, '2024-07-19', ?, ?)", (json.dumps(metric), now, now))
    for seq, look in ((1, "pass"), (2, "pass"), (3, "fail")):
        O.db().execute("INSERT INTO candidates (id, objective_id, seq, created_at, model, status, score, metrics, "
                       "lookahead) VALUES (?, 'o1', ?, ?, 'm', 'ok', 1.0, '{}', ?)", (f"c{seq}", seq, now, look))
        kept = O._kept_positions("o1", f"c{seq}")
        kept.parent.mkdir(parents=True, exist_ok=True)
        kept.write_bytes(b"x")
    O.db().commit()
    monkeypatch.setattr(B.projects, "get", lambda pid: {"data_dir": str(tmp_path / "data")})
    (tmp_path / "data").mkdir()

    async def no_rows(obj):
        return None
    monkeypatch.setattr(B, "_snap", no_rows)

    trades = {  # candidate -> its trades; c1 and c2 share the first one, c3 was caught leaking
        "c1": [("2024-07-18T14:00:00", "long", 0.004), ("2024-07-18T16:00:00", "short", -0.003),
               ("2024-07-22T14:00:00", "long", 0.009)],
        "c2": [("2024-07-18T14:00:00", "long", 0.004), ("2024-07-17T14:00:00", "short", 0.0001)],
        "c3": [("2024-07-16T14:00:00", "long", 0.05)],
    }

    async def fake_call(server, tool, args, timeout_s=600.0):
        assert tool == "harness_actions"
        cid = args["actions_path"].rsplit("\\", 1)[-1].rsplit("/", 1)[-1].split(".")[0]
        lo, hi = args["start"] or "0", args["end"] or "9"
        return {"events": [{"entry": e, "exit": e[:11] + "19:00:00", "side": s, "size": 1.0, "bars": 10, "net": u}
                           for e, s, u in trades[cid] if lo <= e[:10] < hi]}
    monkeypatch.setattr(T, "call", fake_call)
    return O.get_objective("o1")


def test_the_book_keeps_holdout_and_leaks_away_from_agents(tmp_path, monkeypatch):
    obj = _objective(tmp_path, monkeypatch)
    asyncio.run(B.ensure_book("o1"))
    con, _ = B._db()
    assert con.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 6
    pool = asyncio.run(B.pool_review(obj))
    # Distinct in-sample trades of eligible candidates: not c3's (look-ahead failed), not the holdout one.
    assert pool["trades"] == 3
    df = pl.read_parquet(tmp_path / "data" / B.DATASET_DIR / "trades_o1.parquet")
    from datetime import datetime

    assert df["entry"].max() < datetime(2024, 7, 19) and set(df["seq"].to_list()) == {1, 2}
    brief = asyncio.run(B.brief(obj, "c1"))
    assert brief["pool"].startswith("SWARM TRADE BOOK (") and "-- 3 trades" in brief["pool"] and "+90.0 bps" not in brief["pool"] + brief["parent"]
    assert brief["dataset"] == "trade_book_trades_o1"


def test_an_experiment_can_load_the_book_the_brief_points_at_but_a_scored_run_cannot(tmp_path, monkeypatch):
    """The brief says `ft.load_pl('trade_book_trades_<id>')` in run_python; a task run mounts only
    /task, so the book must be mounted for experiments -- and for nothing else."""
    import importlib.util
    from pathlib import Path

    obj = _objective(tmp_path, monkeypatch)
    asyncio.run(B.ensure_book("o1"))
    asyncio.run(B.pool_review(obj))                                   # writes the book's dataset
    (tmp_path / "task").mkdir()

    async def export_dir(o, cut):
        return tmp_path / "task"
    monkeypatch.setattr(T, "export_dir", export_dir)
    runs: list[dict] = []

    async def execute(code, *, timeout_s, files, mounts):
        runs.append({"files": files, "mounts": mounts})
        return {"ok": True, "stdout": "", "stderr": "", "artifacts": [], "duration_s": 0.1, "run_dir": str(tmp_path)}
    monkeypatch.setattr(O, "execute", execute)
    import app.library as L
    import app.research as RS

    monkeypatch.setattr(L, "module_files", lambda pid: {})
    monkeypatch.setattr(RS, "code_files", lambda pid: {})

    code = "import ft\ntrades = ft.load_pl('trade_book_trades_o1')\n"
    asyncio.run(O.scratch_python("o1", O.Scratch(code=code, timeout_s=60)))
    catalog = json.loads(runs[0]["files"][".ft/catalog.json"])
    book = tmp_path / "data" / "trade_book" / "trades_o1.parquet"
    assert [c["view"] for c in catalog] == ["trade_book_trades_o1"]
    # Only this objective's file, read-only, where ft.path resolves it -- never the data folder.
    assert runs[0]["mounts"] == [(str(tmp_path / "task"), "/task"), (str(book), "/trade_book/trades_o1.parquet")]

    spec = importlib.util.spec_from_file_location("ft_under_test", Path(O.FT_HELPER))
    ft = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ft)
    ft._CATALOG = catalog
    # The view name the brief gives and the path list_data shows both resolve.
    assert ft.path("trade_book_trades_o1").replace("\\", "/") == "/trade_book/trades_o1.parquet"
    assert ft.path("trade_book/trades_o1.parquet").replace("\\", "/") == "/trade_book/trades_o1.parquet"

    # A scored or look-ahead run of the same code sees the task rows only.
    asyncio.run(O._run_forecasting(code, str(tmp_path / "data"), [], None, 60, obj, None))
    assert json.loads(runs[1]["files"][".ft/catalog.json"]) == [] and runs[1]["mounts"] == [(str(tmp_path / "task"), "/task")]


def test_load_pl_counts_as_a_dataset_read_for_the_look_ahead_copies(monkeypatch):
    import app.library as L

    monkeypatch.setattr(L, "reachable_modules", lambda pid, code: {})
    obj = {"project_id": "p1"}
    assert O._datasets_used(obj, "import ft\na = ft.load('x')\nb = ft.load_pl('y')\n") == {"x", "y"}
    assert O._datasets_used(obj, "import ft\nb = ft.load_pl(name)\n") is None       # computed: copy all


def test_the_trade_leaderboard_groups_takers_and_ranks_each_class(tmp_path, monkeypatch):
    _objective(tmp_path, monkeypatch)
    monkeypatch.setattr(B, "spawn_build", lambda oid: None)
    asyncio.run(B.ensure_book("o1"))
    asyncio.run(B.set_threshold("o1", B.Threshold(value=20)))        # 20 bps
    win = asyncio.run(B.list_trades("o1", "win"))
    assert [r["entry"] for r in win["rows"]] == ["2024-07-22T14:00:00", "2024-07-18T14:00:00"]
    assert [t["seq"] for t in win["rows"][1]["takers"]] == [1, 2] and win["rows"][0]["holdout"]
    assert win["counts"]["in_sample"]["win"] == 1 and win["counts"]["holdout"]["win"] == 1
    assert win["threshold"] == pytest.approx(0.002) and win["threshold_source"] == "set"
    loss = asyncio.run(B.list_trades("o1", "loss"))
    assert [r["unit"] for r in loss["rows"]] == [-0.003]
    scratch = asyncio.run(B.list_trades("o1", "scratch", segment="in_sample"))
    assert [r["side"] for r in scratch["rows"]] == ["short"]
    assert all(r["unit"] != 0.05 for r in asyncio.run(B.list_trades("o1", "all"))["rows"])   # c3 is out


def test_an_unreachable_server_is_retried_but_a_bad_candidate_is_not(tmp_path, monkeypatch):
    obj = _objective(tmp_path, monkeypatch)

    async def down(server, tool, args, timeout_s=600.0):
        raise HTTPException(status_code=502, detail="task server 'gex': harness_actions failed: ConnectError")
    monkeypatch.setattr(T, "call", down)
    asyncio.run(B.ensure_book("o1"))
    con, _ = B._db()
    assert con.execute("SELECT COUNT(*) FROM trade_index").fetchone()[0] == 0     # nothing recorded: retried later
    assert "ConnectError" in B._BUILDS["o1"]["error"]
    O._kept_positions("o1", "c1").unlink()
    assert asyncio.run(B.index_candidate(obj, "c1")) == 0
    assert "no actions kept" in con.execute("SELECT error FROM trade_index WHERE candidate_id='c1'").fetchone()[0]


def test_agents_are_told_the_goal_and_the_reviews():
    import swarm_runner as R

    ctx = {"trade_book": {"goal": B.GOAL, "threshold": "16.0 bps", "pool": "SWARM TRADE BOOK: ...",
                          "dataset": "trade_book_trades_o1"}}
    text = "\n".join(R._trade_book_lines(ctx))
    assert "BIG WINNERS ONLY" in text and "SWARM TRADE BOOK" in text and "ft.load_pl('trade_book_trades_o1')" in text
    assert ".group_by('cls').len()" in text and "never df[mask]" in text      # the polars way to test a filter
    assert R._trade_book_lines({}) == []
