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

    # Value functions: each one offered can rank, and becomes the score.
    fns = d["value_functions"]
    assert fns and d["value_function"] == fns[0]["name"] and all(f.get("description") is not None for f in fns)
    for f in fns:
        ev2 = p.harness_evaluate(task, str(tmp_path / "a.parquet"), None, f["name"])
        assert ev2["score_name"] == f["name"] and ev2["higher_is_better"] == f.get("higher_is_better", True)
        assert p.task_describe(task, None, f["name"])["value_function"] == f["name"]
    with pytest.raises(Exception):
        p.task_describe(task, None, "no_such_value_function")

    # Action rules: a server that offers them values under each; one that does not refuses them.
    rules = d.get("action_rules") or []
    for r in rules:
        ev3 = p.harness_evaluate(task, str(tmp_path / "a.parquet"), None, None, r["name"])
        assert "segments" in ev3 and p.task_describe(task, None, None, r["name"])["action_rule"] == r["name"]
    if not rules:
        with pytest.raises(Exception):
            p.task_describe(task, None, None, "intraday")

    # Misdated actions are refused with a reason, not scored.
    pl.DataFrame({"t": np.arange(5).astype("datetime64[ns]"), "pos": [1.0] * 5}).write_parquet(tmp_path / "bad.parquet")
    assert "problem" in p.harness_evaluate(task, str(tmp_path / "bad.parquet"))


def test_task_query_cannot_read_files(provider, tmp_path):
    """polars SQL can read files; an agent must not reach the holdout (or anything else) that way."""
    p, task = provider
    secret = tmp_path / "holdout.parquet"
    pl.DataFrame({"t": [1], "x": [42.0]}).write_parquet(secret)
    pl.DataFrame({"t": [1], "x": [42.0]}).write_csv(tmp_path / "holdout.csv")
    f = str(secret).replace("\\", "/")
    c = f[:-len("parquet")] + "csv"
    for sql in (f"SELECT * FROM read_parquet('{f}')",
                f"SELECT * FROM READ_PARQUET('{tmp_path.as_posix()}/*.parquet')",
                f'SELECT * FROM "read_parquet"(\'{f}\')',
                f"SELECT * FROM read_csv('{c}')",
                f"WITH x AS (SELECT * FROM read_csv('{c}')) SELECT * FROM x",
                f"SELECT count(*) AS n FROM rows WHERE t IN (SELECT t FROM read_parquet('{f}'))"):
        with pytest.raises(Exception, match="not allowed"):
            p.task_query(task, sql, 5)
    assert p.task_query(task, "SELECT count(*) AS n FROM rows a JOIN rows b ON a.t = b.t", 5)["rows"][0]["n"] > 0


def test_battery_reports_only_requests_it_could_not_fill(tmp_path):
    p, task = _battery_provider(tmp_path)
    p.harness_export_rows(task, str(tmp_path / "all" / "rows.parquet"), None)
    t = pl.read_parquet(tmp_path / "all" / "rows.parquet")["t"]
    i = np.arange(len(t))
    small = np.where(i < 20, np.where(i % 2 == 0, 0.2, -0.2), 0.0)     # 10 x (+1 kW, -1 kW), then idle
    pl.DataFrame({"t": t, "pos": small}).write_parquet(tmp_path / "small.parquet")
    ev = p.harness_evaluate(task, str(tmp_path / "small.parquet"))
    assert all(d["requests_clipped"] == 0 for d in ev["diagnostics"].values())
    pl.DataFrame({"t": t, "pos": -np.ones(len(t))}).write_parquet(tmp_path / "sell.parquet")
    ev = p.harness_evaluate(task, str(tmp_path / "sell.parquet"))     # always selling: empty after hour 2
    assert 0 < len(t) - 3 <= sum(d["requests_clipped"] for d in ev["diagnostics"].values()) < len(t) - 1


def test_directions_are_offered_or_refused(provider, tmp_path):
    """A server that offers directions (long only / short only / both) manages the actions under
    the chosen one; a server that offers none refuses one."""
    p, task = provider
    d = p.task_describe(task, None)
    if not d.get("directions"):
        with pytest.raises(Exception):
            p.task_describe(task, None, None, None, "long")
        return
    for name in [x["name"] for x in d["directions"]]:
        assert p.task_describe(task, None, None, None, name)["direction"] == name
    p.harness_export_rows(task, str(tmp_path / "all" / "rows.parquet"), None)
    t = pl.read_parquet(tmp_path / "all" / "rows.parquet")["t"]
    acts = np.where((np.arange(len(t)) // 500) % 2 == 0, -1.0, 1.0)            # alternating shorts and longs
    pl.DataFrame({"t": t, "pos": acts}).write_parquet(tmp_path / "a.parquet")
    both = p.harness_actions(task, str(tmp_path / "a.parquet"), None, None, 5000, None, None, "both")
    long = p.harness_actions(task, str(tmp_path / "a.parquet"), None, None, 5000, None, None, "long")
    assert {e["side"] for e in both["events"]} == {"long", "short"}
    assert {e["side"] for e in long["events"]} == {"long"}


def test_guidance_is_served_as_titled_sections(provider):
    """Advice only the server can give reaches the agents' brief as {title, text} sections; a server
    that offers directions words it for the chosen one."""
    p, task = provider
    g = p.task_describe(task, None)["guidance"]
    assert isinstance(g, list) and all(s["title"] and s["text"] for s in g)
    if p.task_describe(task, None).get("directions"):
        text = lambda d: " ".join(s["text"] for s in p.task_describe(task, None, None, None, d)["guidance"])  # noqa: E731
        assert "MIRROR" in text("both") and "LONG ONLY" in text("long") and "SHORT ONLY" in text("short")


def test_the_battery_keeps_its_backup_reserve(tmp_path):
    """The battery's action rule: a reserve it never sells below, and a benchmark that keeps it too."""
    b = _load(MCP / "test" / "battery" / "server.py", "battery_reserve_test")
    price = np.r_[np.full(10, 20.0), np.full(10, 200.0)]
    _, soc, _ = b.simulate(price, -np.ones(20), reserve=0.5)
    assert soc.min() >= 0.5 * b.CAPACITY_KWH - 1e-9                   # selling stops at the reserve
    free, keep = b.oracle(price, levels=28).sum(), b.oracle(price, levels=28, reserve=0.5).sum()
    assert keep < free                                                 # the reserve costs the oracle too
    p, task = TasksProvider(lambda: b.TASKS), "home_battery"
    d = p.task_describe(task, None, None, "reserve_20")
    assert d["action_rule"] == "reserve_20" and any(g["title"] == "Backup reserve" for g in d["guidance"])
    with pytest.raises(Exception):
        p.task_describe(task, None, None, "max_3_days")                 # only the battery's own rules


def test_a_table_offers_every_numeric_column_as_a_target(tmp_path):
    s = _load(MCP / "test" / "tables" / "server.py", "tables_numeric_test")
    p = TasksProvider(s.tasks)
    d = p.task_describe("bike_rentals", None)
    assert d["target_options"][0] == "rentals" and {"temperature_c", "humidity"} <= set(d["target_options"])
    assert p.task_describe("bike_rentals", "temperature_c")["target"] == "temperature_c"


def test_a_table_reads_an_excel_workbook(tmp_path):
    pytest.importorskip("fastexcel")
    pytest.importorskip("xlsxwriter")
    from taskkit.table import TableTask

    t = pl.datetime_range(pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 3), "1h", eager=True)
    pl.DataFrame({"time": t, "y": np.arange(len(t), dtype=float)}).write_excel(tmp_path / "data.xlsx")
    task = TableTask({"name": "xl", "target": "y", "sources": [{"path": "data.xlsx", "time_column": "time"}],
                      "evaluator": {"kind": "forecast", "horizon": 1}}, tmp_path)
    assert len(task.rows()) == len(t) and task.rows()["y"].to_list()[:3] == [0.0, 1.0, 2.0]
