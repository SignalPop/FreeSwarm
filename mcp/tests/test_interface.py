"""Every task server speaks the same interface: the same tools, the same result shapes, the same
guarantees (in-sample only for agents, exports strictly before a cut) -- whatever it uses inside.

Runs the interface against each provider: the battery sample (taskkit Task helper), the tables
sample (taskkit TableTask) and the GEX server (its own polars/numpy internals) on synthetic bars.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from taskkit.server import AGENT_TOOLS, HARNESS_TOOLS, TasksProvider

MCP = Path(__file__).resolve().parent.parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


def _gex_provider(tmp_path: Path):
    """The GEX server over synthetic 10-second bars (two sessions a day for 60 days)."""
    if not (MCP / "gex" / "server.py").is_file():
        pytest.skip("mcp/gex is not present (it is git-ignored)")
    rng = np.random.default_rng(3)
    days = np.arange(np.datetime64("2024-01-02"), np.datetime64("2024-03-02"))
    t = np.concatenate([d.astype("datetime64[s]") + np.timedelta64(13 * 3600 + 30 * 60, "s")
                        + np.arange(0, 2340) * np.timedelta64(10, "s") for d in days])
    close = 400 * np.exp(np.cumsum(rng.normal(0, 2e-4, len(t))))
    pl.DataFrame({"SlotUtc": t.astype("datetime64[ns]"), "Close": close, "GEX": rng.normal(size=len(t))}) \
        .write_parquet(tmp_path / "bars.parquet")
    cfg = {"data": {"path": str(tmp_path / "bars.parquet"), "time_column": "SlotUtc", "price": "Close",
                    "delay": {"rows": 2, "except": ["Close"]}},
           "tasks": [{"name": "synthetic", "title": "synthetic bars", "holdout_from": "2024-02-15",
                      "positions": {"mode": "target", "max_position": 3, "direction": "both", "flat_each_day": True},
                      "valuation": {"score": "sharpe", "cost_bps": 1, "min_active_days": 1}}]}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    server = _load(MCP / "gex" / "server.py", "gex_server_under_test")
    return server.GexProvider(tmp_path / "config.json"), "synthetic"


def _battery_provider(tmp_path: Path):
    b = _load(MCP / "test" / "battery" / "server.py", "battery_server_under_test")
    return TasksProvider(lambda: b.TASKS), "home_battery"


def _tables_provider(tmp_path: Path):
    s = _load(MCP / "test" / "tables" / "server.py", "tables_server_under_test")
    return TasksProvider(s.tasks), "bike_rentals"


@pytest.fixture(params=["gex", "battery", "tables"])
def provider(request, tmp_path):
    return {"gex": _gex_provider, "battery": _battery_provider, "tables": _tables_provider}[request.param](tmp_path)


def test_every_tool_of_the_interface_is_implemented(provider):
    p, _ = provider
    for tool in AGENT_TOOLS + HARNESS_TOOLS:
        assert callable(getattr(p, tool, None)), tool


def test_the_interface_end_to_end(provider, tmp_path):
    p, task = provider
    listing = p.task_list()
    assert any(t["name"] == task for t in listing["tasks"])
    d = p.task_describe(task, None)
    for k in ("name", "target", "target_options", "shape", "valuation", "action", "score", "rows", "in_sample_rows",
              "holdout_from", "cuts", "columns", "version"):
        assert k in d, k
    assert d["target"] in d["target_options"] and d["shape"]["rows"] == d["rows"] and d["valuation"].get("summary")
    with pytest.raises(Exception):
        p.task_describe(task, "no_such_column")                          # only the server's target options
    hold = d["holdout_from"]
    assert hold.endswith("T00:00:00")                                   # a midnight boundary

    # Agent-facing reads never reach the holdout.
    s = p.task_sample_rows(task, 500, max(0, d["in_sample_rows"] - 5), None)
    assert s["rows"] and all(r["t"] < hold for r in s["rows"])
    q = p.task_query(task, "SELECT max(t) AS last, count(*) AS n FROM rows", 5)
    assert q["rows"][0]["n"] == d["in_sample_rows"] and str(q["rows"][0]["last"]) < hold
    with pytest.raises(Exception):
        p.task_query(task, "DROP TABLE rows", 5)
    assert p.task_column_stats(task)["in_sample_rows"] == d["in_sample_rows"]

    # Exports stop strictly before the cut, and task.json hides the cuts.
    cut = d["cuts"][-1]
    out = p.harness_export_rows(task, str(tmp_path / "x" / "rows.parquet"), cut)
    rows = pl.read_parquet(tmp_path / "x" / "rows.parquet")
    assert out["rows"] == len(rows) and rows["t"].max() < np.datetime64(cut.replace("T", " "))
    assert "cuts" not in json.loads((tmp_path / "x" / "task.json").read_text())

    # Actions in -> managed -> valued: a curve and a valuation per segment.
    p.harness_export_rows(task, str(tmp_path / "all" / "rows.parquet"), None)
    allrows = pl.read_parquet(tmp_path / "all" / "rows.parquet")
    rng = np.random.default_rng(0)
    acts = np.sign(rng.normal(size=len(allrows))) * np.repeat(rng.random(len(allrows) // 50 + 1) > 0.5, 50)[: len(allrows)]
    pl.DataFrame({"t": allrows["t"], "pos": acts.astype(float)}).write_parquet(tmp_path / "a.parquet")
    ev = p.harness_evaluate(task, str(tmp_path / "a.parquet"))
    assert "problem" not in ev, ev.get("problem")
    assert {"in_sample", "holdout"} <= set(ev["segments"]) and "score" in ev["segments"]["in_sample"]
    assert ev["curve"] and ev["curve_kind"] in ("returns", "additive") and isinstance(ev["notes"], str)
    assert "holdout" not in ev["notes"].lower()                          # notes go to agents
    log = p.harness_actions(task, str(tmp_path / "a.parquet"), None, None, 20)
    assert "events" in log and "state" in log
    day = str(allrows["t"][len(allrows) // 2])[:10]
    nxt = str(np.datetime64(day) + np.timedelta64(1, "D"))
    drill = p.harness_actions(task, str(tmp_path / "a.parquet"), day, nxt, 50)
    bars = drill["bars"]
    assert bars["kind"] in ("ohlc", "line") and bars["rows"] and all(day <= r[0] < nxt for r in bars["rows"])
    assert isinstance(p.harness_leak_scan(task, 5)["suspects"], list)

    # Misdated actions are refused with a reason, not scored.
    pl.DataFrame({"t": np.arange(5).astype("datetime64[ns]"), "pos": [1.0] * 5}).write_parquet(tmp_path / "bad.parquet")
    assert "problem" in p.harness_evaluate(task, str(tmp_path / "bad.parquet"))
