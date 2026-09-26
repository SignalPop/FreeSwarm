"""evaluate() end to end through the deferred look-ahead branch.

The background look-ahead shipped with `shutil` used but not imported: every evaluation that
reached the look-ahead step failed with a NameError, and no test ran that branch. This one
drives evaluate() with the sandbox and pricing stubbed, against a throwaway database."""

from __future__ import annotations

import asyncio

import pytest

from app import objectives as O


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "DB_PATH", tmp_path / "objectives.sqlite3")
    monkeypatch.setattr(O, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(O, "_conn", None)
    yield
    if O._conn is not None:
        O._conn.close()
    O._conn = None


def test_evaluate_defers_the_lookahead_and_returns_the_score(temp_db, tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    (run_dir / ".ft").mkdir(parents=True)
    (run_dir / ".ft" / "positions.parquet").write_bytes(b"PAR1")
    days = [f"2024-0{m}-{d:02d}" for m in (5, 6, 7, 8) for d in range(1, 29)]
    returns = [[d, 0.001 * ((i % 5) - 1)] for i, d in enumerate(days)]

    async def fake_run(*a, **k):
        return {"ok": True, "stdout": "", "stderr": "", "run_id": "r1", "result": {}, "run_dir": str(run_dir)}

    spawned = []
    monkeypatch.setattr(O.projects, "get", lambda pid: {"data_dir": str(tmp_path)})
    monkeypatch.setattr(O.datasource, "catalog", lambda d: [])
    monkeypatch.setattr(O, "_run_forecasting", fake_run)
    monkeypatch.setattr(O, "_mark_to_market", lambda obj, d, p: (returns, {"position_changes": 10}))
    monkeypatch.setattr(O, "_positions_off_data", lambda mtm: None)
    monkeypatch.setattr(O, "_spawn", lambda coro, what: (spawned.append(what), coro.close()))
    import app.library as L
    monkeypatch.setattr(L, "record_usage", lambda *a, **k: None)

    obj = {"id": "o1", "project_id": "p1", "title": "t", "status": "running", "dataset": "px", "time_column": "t",
           "split_date": "2024-07-19", "lookahead_check": True, "require_audit": True, "eval_timeout_s": 60,
           "best_id": None, "metric": {"kind": "sharpe", "price_column": "Close", "cost_bps": 2,
                                       "periods_per_year": 252, "min_active_days": 5, "mid_cut": "2024-06-01"}}
    view = asyncio.run(O.evaluate(obj, O.Submit(code="print(1)", model="m", mode="explore")))
    assert view["status"] == "ok", view
    assert str(view["lookahead"]).startswith("pending")
    assert spawned and "look-ahead" in spawned[0]
    c = O.get_candidate(view["candidate_id"])
    assert c["lookahead"] == "pending" and c["score"] is not None
    # the full run's positions were set aside for the background test
    assert (tmp_path / "work" / "o1" / "pending" / f"{c['id']}.parquet").exists()
