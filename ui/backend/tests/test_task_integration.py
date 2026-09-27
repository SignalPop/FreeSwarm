"""A task objective end to end: the real sandbox, the real battery-demo task server over MCP.

Opt-in (needs Docker, the freeswarm-sandbox image and the `battery-demo` entry in
mcp_servers.json; takes a few minutes):

    set FREESWARM_INTEGRATION=1
    python -m pytest tests/test_task_integration.py -q -s

It proves the whole contract on a throwaway database:
  * a task objective is created from the server's description (split, cuts, task snapshot);
  * an honest strategy is run in the sandbox on the exported rows, scored by the server, and
    passes the look-ahead test;
  * a strategy that peeks one row ahead (price.shift(-1)) scores brilliantly -- and fails it;
  * a strategy normalising with full-sample statistics fails it too;
  * agents see in-sample numbers only, the brief speaks about the task (ft.rows /
    ft.report_actions), and an agent experiment sees in-sample rows only.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from app import objectives as O

pytestmark = pytest.mark.skipif(os.environ.get("FREESWARM_INTEGRATION") != "1",
                                reason="integration test: set FREESWARM_INTEGRATION=1 (needs Docker)")

HONEST = '''
import ft, numpy as np, pandas as pd
rows = ft.rows()
fut = rows[[f"da_price_{h}" for h in range(1, 13)]]
nxt = rows["da_price_1"]
lo, hi = fut.quantile(0.25, axis=1), fut.quantile(0.75, axis=1)
act = np.where(nxt <= lo, 1.0, np.where(nxt >= hi, -1.0, 0.0))
ft.report_actions(pd.Series(act, index=rows["t"]))
print("rows", len(rows), "last", rows["t"].iloc[-1])
'''

# Peeks at the next hour's real price: a clairvoyant, and a leak.
PEEK = '''
import ft, numpy as np, pandas as pd
rows = ft.rows()
nxt = rows["price"].shift(-1)
day_mean = rows["price"].rolling(24, min_periods=1).mean()
act = np.where(nxt < day_mean * 0.8, 1.0, np.where(nxt > day_mean * 1.2, -1.0, 0.0))
ft.report_actions(pd.Series(act, index=rows["t"]))
'''

# Normalises with statistics of ALL rows (including the future): subtle, and a leak.
FULL_SAMPLE = '''
import ft, numpy as np, pandas as pd
rows = ft.rows()
z = (rows["da_price_1"] - rows["da_price_1"].mean()) / rows["da_price_1"].std()
act = np.where(z < -0.5, 1.0, np.where(z > 0.5, -1.0, 0.0))
ft.report_actions(pd.Series(act, index=rows["t"]))
'''

PROBE = '''
import ft
rows = ft.rows()
spec = ft.task()
print("LAST", rows["t"].max(), "TARGET", spec["target"], "N", len(rows))
'''


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "DB_PATH", tmp_path / "objectives.sqlite3")
    monkeypatch.setattr(O, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(O, "_conn", None)
    monkeypatch.setattr(O.projects, "get", lambda pid: {"id": pid, "data_dir": str(tmp_path / "data"),
                                                       "connectors": []})
    (tmp_path / "data").mkdir()
    import app.library as L
    monkeypatch.setattr(L, "record_usage", lambda *a, **k: None)
    spawned: list = []
    monkeypatch.setattr(O, "_spawn", lambda coro, what: spawned.append(coro))
    monkeypatch.setattr(O, "_auto_review", lambda *a, **k: None)      # no network review in a test
    yield spawned
    if O._conn is not None:
        O._conn.close()
    O._conn = None


async def _submit(obj: dict, code: str, spawned: list) -> dict:
    view = await O.evaluate(obj, O.Submit(code=code, model="test", mode="explore", rationale="integration test"))
    while spawned:                       # the deferred look-ahead test, awaited here
        await spawned.pop(0)
    return O.get_candidate(view["candidate_id"]) | {"_view": view}


def test_battery_task_end_to_end(world):
    spawned = world

    async def main():
        created = await O.create_objective("p1", O.CreateObjective(
            title="battery", metric=O.MetricSpec(kind="task", task_server="battery-demo", task="home_battery"),
            eval_timeout_s=180, require_audit=False))
        obj = O.get_objective(created["id"])
        m = obj["metric"]
        assert obj["split_date"] == "2024-09-01" and m["mid_cut"].startswith("2023-")
        assert m["task_info"]["target"] == "price" and obj["lookahead_check"]
        assert obj["dataset"] is None and m.get("price_column") is None

        honest = await _submit(obj, HONEST, spawned)
        print("\nHONEST", honest["status"], honest["score"], honest["is_score"], honest["lookahead"],
              honest["lookahead_detail"][:300], honest["score_note"])
        assert honest["status"] == "ok", honest["score_note"] or honest["stderr"][-2000:]
        assert honest["is_score"] is not None and honest["is_score"] > 0.3
        assert honest["lookahead"] == "pass", honest["lookahead_detail"]
        assert honest["returns"] and len(honest["returns"]) > 600          # one entry per day
        view = honest["_view"]
        flat = repr(view)
        assert "holdout" not in {k.lower() for k in view} and "oracle_profit': 8" not in flat
        assert "task_notes" in view and "In-sample" in view["task_notes"]

        peek = await _submit(obj, PEEK, spawned)
        print("PEEK", peek["status"], peek["is_score"], peek["lookahead"], peek["lookahead_detail"][:300])
        assert peek["status"] == "ok"                                  # it runs and scores...
        assert peek["lookahead"] == "fail", peek["lookahead_detail"]   # ...and the leak is caught

        full = await _submit(obj, FULL_SAMPLE, spawned)
        print("FULL_SAMPLE", full["status"], full["is_score"], full["lookahead"], full["lookahead_detail"][:300])
        assert full["lookahead"] == "fail", full["lookahead_detail"]

        # The leaderboard: only the honest candidate may hold the title.
        best = O.get_objective(obj["id"]).get("best_id")
        assert best in (None, honest["id"])

        # What an agent is given: a task brief, no project datasets, in-sample experiments.
        ctx = await O.context(obj["id"])
        assert ctx["datasets"] == [] and ctx["task"]["target"] == "price"
        import swarm_runner as S

        prompt = S.iteration_prompt(ctx)
        assert "ft.rows()" in prompt and "ft.report_actions" in prompt and "ft.load(" not in prompt
        assert "TASK" in prompt and "10-second" not in prompt
        exp = await O.scratch_python(obj["id"], O.Scratch(code=PROBE, timeout_s=120))
        print("EXPERIMENT", exp["stdout"].strip()[-200:], exp["data"])
        assert exp["ok"] and "LAST 2024-08-31" in exp["stdout"] and "TARGET price" in exp["stdout"]
        assert exp["data"].startswith("in-sample only")

    asyncio.run(main())
