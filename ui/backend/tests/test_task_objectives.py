"""The control plane's side of task objectives: ranking a server's valuation, the fence that keeps
harness_* tools away from agents, and -- across the process boundary -- that the GEX server's
valuation prices positions exactly as the positions harness does.

The server code under mcp/ is tested in mcp/tests. The control plane never imports it; only the
cross-check at the bottom loads mcp/gex, and skips when that folder is absent (it is git-ignored).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import mcp_registry
from app import objectives as O
from app import task_objectives as T

REPO = Path(__file__).resolve().parents[3]


def test_task_score_is_the_weaker_segment_and_holdout_mode():
    ev = {"segments": {"in_sample": {"score": 0.6}, "holdout": {"score": 0.4}}, "curve": [["2024-01-01", 1.0]]}
    s, is_s, note, m, curve = T.score({"metric": {"higher_is_better": True}}, ev)
    assert (s, is_s, note) == (0.4, 0.6, "") and m["rank"]["weaker"] == "holdout" and curve == [["2024-01-01", 1.0]]
    s, *_ = T.score({"metric": {"higher_is_better": False}}, ev)
    assert s == 0.6                                                    # lower is better: the worse is higher
    s, *_ = T.score({"metric": {"rank": "holdout"}}, ev)
    assert s == 0.4
    s, _, note, *_ = T.score({"metric": {}}, {"segments": {"in_sample": {"score": None, "note": "too few"},
                                                          "holdout": {"score": 1.0}}})
    assert s is None and "too few" in note
    notes = T.agent_notes({"task": {"notes": "in-sample", "diagnostics": {"in_sample": {"a": 1}, "holdout": {"b": 2}}}})
    assert notes == {"task_notes": "in-sample", "diagnostics_in_sample": {"a": 1}}


def test_unranked_and_holdout_note_reach_the_agent_without_numbers():
    ev = {"segments": {"in_sample": {"score": 0.5}, "holdout": {"score": 0.7}}, "unranked": "one-sided: ..."}
    s, is_s, note, *_ = T.score({"metric": {}}, ev)
    assert s is None and is_s == 0.5 and note.startswith("one-sided")
    ev = {"segments": {"in_sample": {"score": 0.5}, "holdout": {"score": None, "note": "only 3 scorable rows"}}}
    s, is_s, note, m, _ = T.score({"metric": {}}, ev)
    assert s is None and "3" not in note and note.startswith("holdout: no score")
    assert m["holdout"]["note"] == "only 3 scorable rows"              # kept for the operator


def test_the_total_trade_floor_ranks_selective_strategies_that_skip_most_days():
    ev = {"segments": {"in_sample": {"score": 0.5, "days": 223}, "holdout": {"score": 0.7}},
          "diagnostics": {"in_sample": {"trades": {"long": 29, "short": 10}, "trades_per_day": 0.175}}}
    s, *_ = T.score({"metric": {"min_trades": 30}}, ev)
    assert s == 0.5                                                    # 39 trades in all: ranked, at 0.175 a day
    s, is_s, note, m, _ = T.score({"metric": {"min_trades": 40}}, ev)
    assert s is None and is_s == 0.5 and note.startswith("too few trades: 39 in-sample")
    assert T.agent_notes(m)["not_ranked_because"] == note
    del ev["diagnostics"]["in_sample"]["trades"]                       # no per-side counts: per day x days
    assert T.in_sample_trades(ev) == pytest.approx(0.175 * 223)
    s, *_ = T.score({"metric": {"min_trades": 30, "min_trades_per_day": 1}}, ev)
    assert s is None                                                   # the older daily quota still holds where set


def test_trade_floor_leaves_rare_traders_unranked_and_tells_the_agent():
    ev = {"segments": {"in_sample": {"score": 0.5}, "holdout": {"score": 0.7}},
          "diagnostics": {"in_sample": {"trades_per_day": 0.4}}}
    s, is_s, note, m, _ = T.score({"metric": {"min_trades_per_day": 2}}, ev)
    assert s is None and is_s == 0.5 and note.startswith("too few trades: 0.4")
    assert T.agent_notes(m)["not_ranked_because"] == note
    s, *_ = T.score({"metric": {"min_trades_per_day": 0.3}}, ev)
    assert s == 0.5                                                    # above the floor: ranked as before
    s, _, _, m, _ = T.score({"metric": {}}, ev)
    assert s == 0.5 and "too_few_trades" not in m                      # no floor set


def test_harness_tools_are_fenced_off_from_agents():
    assert mcp_registry.harness_only("gex__harness_evaluate")
    assert mcp_registry.harness_only("battery-demo__harness_export_rows")
    assert mcp_registry.harness_only("custom__Harness_Dump") and mcp_registry.harness_only("x__HARNESS_rows")
    assert not mcp_registry.harness_only("gex__task_query")


def test_http_routes_hide_and_refuse_harness_tools(monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    monkeypatch.setattr(main.auth, "auth_enabled", lambda: False)
    monkeypatch.setattr(main, "_project_connectors", lambda pid: None)
    spec = mcp_registry.ServerSpec(name="gex", transport="http", url="http://127.0.0.1:1/mcp")
    monkeypatch.setattr(mcp_registry, "load_config", lambda: [spec])

    def tool(name):
        return {"type": "function", "function": {"name": f"gex__{name}", "description": "", "parameters": {}}}

    async def fake_probe(s, *a, **k):
        return {"name": s.name, "ok": True, "error": None,
                "tools": [tool("task_query"), tool("harness_evaluate"), tool("harness_actions")]}

    called = []

    async def fake_call(*a, **k):
        called.append(a)
        return {"is_error": False, "content": "{}", "structured": {}}

    monkeypatch.setattr(mcp_registry, "probe", fake_probe)
    monkeypatch.setattr(mcp_registry, "call_tool", fake_call)
    client = TestClient(main.app)
    names = [t["function"]["name"] for t in client.get("/api/mcp/tools").json()["tools"]]
    assert names == ["gex__task_query"]
    r = client.post("/api/mcp/call", json={"tool": "gex__harness_actions", "arguments": {}})
    assert r.status_code == 403 and "harness" in r.json()["detail"] and not called
    assert client.post("/api/mcp/call", json={"tool": "gex__task_query", "arguments": {}}).status_code == 200


# ---------------------------------------------------------------------------------------------
# Across the boundary: mcp/gex values positions exactly as the positions harness prices them
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gex():
    d = REPO / "mcp" / "gex"
    if not (d / "valuation.py").is_file():
        pytest.skip("mcp/gex is not present (it is git-ignored)")
    sys.path.insert(0, str(d))
    import positions
    import valuation
    return positions, valuation


@pytest.mark.parametrize("intraday", [False, True])
def test_gex_valuation_matches_the_positions_harness(tmp_path, gex, intraday):
    P, V = gex
    t = pd.to_datetime(["2024-01-02 10:00", "2024-01-02 10:01", "2024-01-02 10:02", "2024-01-02 10:03",
                        "2024-01-03 10:00", "2024-01-03 10:01", "2024-01-03 10:02", "2024-01-03 10:03"])
    p = np.array([100.0, 101.0, 100.5, 102.0, 101.0, 103.0, 102.5, 104.0])
    pd.DataFrame({"ts": t, "close": p}).to_parquet(tmp_path / "px.parquet")
    acts = np.array([1.0, 1.0, -2.0, -2.0, 0.0, 1.0, 1.0, 1.0])
    pd.DataFrame({"t": t, "pos": acts}).to_parquet(tmp_path / "pos.parquet")
    obj = {"id": "o", "dataset": "px.parquet", "time_column": "ts", "split_date": "2024-01-03",
           "metric": {"kind": "sharpe", "price_column": "close", "cost_bps": 10.0, "max_leverage": 3.0,
                      "intraday": intraday}}
    ref, _ = O._mark_to_market(obj, str(tmp_path), tmp_path / "pos.parquet")
    keys = t.values.astype("datetime64[ns]").astype(np.int64)
    pos = P.manage(keys, keys, acts, P.Rules(max_position=3.0, holding="intraday" if intraday else "open"))
    ev = V.value(keys, p, pos, int(pd.Timestamp("2024-01-03").value), {"cost_bps": 10.0, "min_active_days": 1})
    assert [d for d, _ in ev["curve"]] == [d for d, _ in ref]
    for (_, a), (_, b) in zip(ev["curve"], ref):
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    assert ev["diagnostics"]["in_sample"]["trades"] == {"long": 1, "short": 1}


def test_oauth_callback_matches_the_state_the_sdk_put_in_the_url():
    """The SDK mints its own `state` for the authorization URL; the callback carries that one.
    The flow must be found by it, or Connect can never complete."""
    from app import mcp_oauth

    flow = mcp_oauth.register_flow("srv")
    mcp_oauth.alias_flow(flow, "sdk-state-123")
    assert mcp_oauth.finish_flow("sdk-state-123", "the-code")
    assert flow.code == "the-code" and flow.returned_state == "sdk-state-123" and flow.code_ready.is_set()
    assert not mcp_oauth.finish_flow("forged-state", "x")


# ---------------------------------------------------------------------------------------------
# A project's data/action MCP
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def temp_projects(tmp_path, monkeypatch):
    from app import projects

    monkeypatch.setattr(projects, "PROJECTS_PATH", tmp_path / "projects.json")
    monkeypatch.setattr(projects, "DEFAULT_ROOT", tmp_path / "projects")
    return projects


def test_project_task_server_is_set_cleared_and_reported(temp_projects, monkeypatch):
    from app import main

    p = temp_projects.create("Gex", task_server="gex")
    assert temp_projects.get(p["id"])["task_server"] == "gex"
    spec = mcp_registry.ServerSpec(name="gex", kind="task", transport="http", url="http://127.0.0.1:1/mcp", oauth=True)
    monkeypatch.setattr(mcp_registry, "load_config", lambda: [spec])
    monkeypatch.setattr(main.mcp_oauth, "authorised_servers", lambda: set())
    assert main._project_view(temp_projects.get(p["id"]))["task_server_status"] == "needs sign-in"
    monkeypatch.setattr(main.mcp_oauth, "authorised_servers", lambda: {"gex"})
    assert main._project_view(temp_projects.get(p["id"]))["task_server_status"] == "ready"
    # its agents always reach it, whatever connectors are ticked
    assert "gex" in main._project_connectors(p["id"])
    temp_projects.update(p["id"], task_server="")
    assert "task_server" not in temp_projects.get(p["id"])
    assert main._project_view(temp_projects.get(p["id"]))["task_server_status"] is None


def test_task_objectives_use_the_projects_server(temp_projects, tmp_path, monkeypatch):
    import asyncio

    p = temp_projects.create("Gex", task_server="gex")
    monkeypatch.setattr(O, "DB_PATH", tmp_path / "o.sqlite3")
    monkeypatch.setattr(O, "_conn", None)
    seen = {}

    async def fake_prepare(metric):
        seen.update(metric)
        return {**metric, "task_info": {}}, "2024-07-19"

    monkeypatch.setattr(T, "prepare_objective", fake_prepare)
    try:
        created = asyncio.run(O.create_objective(p["id"], O.CreateObjective(title="t", metric=O.MetricSpec(kind="task", task="x"))))
        assert seen["task_server"] == "gex" and created["metric"]["task_server"] == "gex"
        with pytest.raises(Exception) as e:
            asyncio.run(O.create_objective(p["id"], O.CreateObjective(
                title="t", metric=O.MetricSpec(kind="task", task_server="battery-demo", task="home_battery"))))
        assert "data/action MCP is 'gex'" in str(getattr(e.value, "detail", e.value))
    finally:
        if O._conn is not None:
            O._conn.close()
        O._conn = None


def test_only_data_action_mcps_can_be_a_projects_server(temp_projects, monkeypatch):
    from fastapi import HTTPException

    from app import main

    tool = mcp_registry.ServerSpec(name="timeseries", command="python")          # kind defaults to "tool"
    task = mcp_registry.ServerSpec(name="gex", kind="task", transport="http", url="http://127.0.0.1:1/mcp")
    monkeypatch.setattr(mcp_registry, "load_config", lambda: [tool, task])
    main._check_task_server("gex")
    main._check_task_server("")                                                   # clearing is fine
    for bad, why in (("timeseries", "tool connector"), ("nope", "no MCP server")):
        with pytest.raises(HTTPException) as e:
            main._check_task_server(bad)
        assert why in e.value.detail
    assert mcp_registry.task_server_names() == {"gex"}
    p = temp_projects.create("P", task_server="timeseries")                       # e.g. hand-edited projects.json
    assert main._project_view(temp_projects.get(p["id"]))["task_server_status"] == "not a data/action MCP"


def test_task_objectives_refuse_a_tool_connector(monkeypatch):
    import asyncio

    from fastapi import HTTPException

    monkeypatch.setattr(mcp_registry, "load_config", lambda: [mcp_registry.ServerSpec(name="timeseries", command="py")])
    with pytest.raises(HTTPException) as e:
        asyncio.run(T.call("timeseries", "task_list", {}))
    assert "not a data/action MCP" in e.value.detail


def test_project_target_reaches_the_objective_and_every_harness_call(temp_projects, tmp_path, monkeypatch):
    import asyncio

    p = temp_projects.create("Gex", task_server="gex")
    temp_projects.update(p["id"], task_options={"gex_intraday": {"target": "Open", "value_function": "calmar", "action_rule": "max_3_days", "junk": "x"},
                                                "other": {}})
    assert temp_projects.get(p["id"])["task_options"] == {"gex_intraday": {"target": "Open", "value_function": "calmar",
                                                                          "action_rule": "max_3_days"}}
    monkeypatch.setattr(O, "DB_PATH", tmp_path / "o.sqlite3")
    monkeypatch.setattr(O, "_conn", None)
    calls = []

    async def fake_call(server, tool, args, timeout_s=0):
        calls.append((tool, dict(args)))
        return {"target": args.get("target", "Close"), "value_function": args.get("value_function", "sharpe"),
                "action_rule": args.get("action_rule", "intraday"),
                "holdout_from": "2024-07-19T00:00:00", "cuts": [], "score": {"higher_is_better": True}}

    monkeypatch.setattr(T, "call", fake_call)
    try:
        o = asyncio.run(O.create_objective(p["id"], O.CreateObjective(title="t", metric=O.MetricSpec(kind="task", task="gex_intraday"))))
        assert o["metric"]["target"] == "Open" and o["metric"]["value_function"] == "calmar"
        assert o["metric"]["action_rule"] == "max_3_days"
        assert ("task_describe", {"task": "gex_intraday", "target": "Open", "value_function": "calmar",
                                  "action_rule": "max_3_days"}) in calls
        # the in-sample view for the analysis tools is exported at the split, on the same target
        exports = [a for t_, a in calls if t_ == "harness_export_rows"]
        assert exports and exports[0]["until"] == "2024-07-19" and exports[0]["target"] == "Open"
        obj = O.get_objective(o["id"])
        asyncio.run(T.evaluate_actions(obj, tmp_path / "a.parquet"))
        asyncio.run(T.action_log(obj, tmp_path / "a.parquet", "2024-01-02", "2024-01-03"))
        assert all(a.get("target") == "Open" for t_, a in calls if t_.startswith("harness_"))
        assert all(a.get("value_function") == "calmar" for t_, a in calls if t_ in ("harness_evaluate", "harness_export_rows"))
        assert all(a.get("action_rule") == "max_3_days" for t_, a in calls if t_.startswith("harness_"))
    finally:
        if O._conn is not None:
            O._conn.close()
        O._conn = None


def test_stored_oauth_tokens_know_their_remaining_life(tmp_path, monkeypatch):
    """A stored token reports its remaining lifetime, so an expired one is refreshed silently
    instead of being sent stale (401) and starting a new browser sign-in."""
    import asyncio
    import time as _t

    from app import mcp_oauth

    monkeypatch.setattr(mcp_oauth, "TOKENS_PATH", tmp_path / "tokens.json")
    st = mcp_oauth.FileTokenStorage("srv")
    from mcp.shared.auth import OAuthToken

    asyncio.run(st.set_tokens(OAuthToken(access_token="a", expires_in=3600, refresh_token="r")))
    assert 3500 < asyncio.run(st.get_tokens()).expires_in <= 3570
    st._update(obtained_at=_t.time() - 7200)                          # two hours later
    assert asyncio.run(st.get_tokens()).expires_in == 0


# ---------------------------------------------------------------------------------------------
# Registering an MCP server from the console (a URL or a local path)
# ---------------------------------------------------------------------------------------------
def test_register_by_local_path_needs_trust_and_detects_a_task_server(tmp_path, monkeypatch):
    import asyncio
    import json as _json

    monkeypatch.setattr(mcp_registry, "CONFIG_PATH", tmp_path / "mcp_servers.json")
    (tmp_path / "mcp_servers.json").write_text(_json.dumps({"_comment": "keep me", "servers": []}))
    battery = REPO / "mcp" / "test" / "battery"
    with pytest.raises(ValueError, match="trust"):
        asyncio.run(mcp_registry.register("bat", str(battery)))
    (tmp_path / "evil.cmd").write_text("echo hi")
    with pytest.raises(ValueError, match="Python script"):
        asyncio.run(mcp_registry.register("evil", str(tmp_path / "evil.cmd"), trust_code=True))
    with pytest.raises(ValueError, match="name"):
        asyncio.run(mcp_registry.register("Bad Name!", str(battery), trust_code=True))
    with pytest.raises(ValueError, match="could not reach"):
        asyncio.run(mcp_registry.register("gone", "http://127.0.0.1:9/mcp"))

    import importlib.util

    if importlib.util.find_spec("mcp.server.mcpserver"):        # this interpreter can run the server: auto-detect
        out = asyncio.run(mcp_registry.register("bat", str(battery), trust_code=True))   # a folder holding server.py
        assert out["probe"]["task_server"]
    else:                                                        # (an older MCP SDK here): the kind is chosen
        with pytest.raises(ValueError, match="could not probe"):
            asyncio.run(mcp_registry.register("bat", str(battery), trust_code=True))
        out = asyncio.run(mcp_registry.register("bat", str(battery), kind="task", trust_code=True))
    assert out["server"]["kind"] == "task" and out["server"]["transport"] == "stdio"
    assert out["server"]["args"][0].endswith("server.py")
    saved = _json.loads((tmp_path / "mcp_servers.json").read_text())
    assert saved["_comment"] == "keep me" and [s["name"] for s in saved["servers"]] == ["bat"]
    assert mcp_registry.task_server_names() == {"bat"}
    with pytest.raises(ValueError, match="already registered"):
        asyncio.run(mcp_registry.register("bat", str(battery), trust_code=True))
    assert mcp_registry.unregister("bat") and not mcp_registry.unregister("bat")


def test_unregister_is_refused_while_a_project_uses_the_server(temp_projects, monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    monkeypatch.setattr(main.auth, "auth_enabled", lambda: False)
    temp_projects.create("Gex", task_server="gex")
    r = TestClient(main.app).delete("/api/mcp/servers/gex")
    assert r.status_code == 409 and "data/action MCP of Gex" in r.json()["detail"]


def test_task_options_merge_per_setting_so_quick_changes_never_overwrite(temp_projects):
    p = temp_projects.create("Gex", task_server="gex")
    temp_projects.update(p["id"], task_options={"gex_intraday": {"target": "Open"}})
    temp_projects.update(p["id"], task_options={"gex_intraday": {"value_function": "calmar"}})   # a second, separate change
    temp_projects.update(p["id"], task_options={"other": {"action_rule": "open"}})
    assert temp_projects.get(p["id"])["task_options"] == {"gex_intraday": {"target": "Open", "value_function": "calmar"},
                                                          "other": {"action_rule": "open"}}
    temp_projects.update(p["id"], task_options={"gex_intraday": {"target": ""}, "other": {"action_rule": ""}})
    assert temp_projects.get(p["id"])["task_options"] == {"gex_intraday": {"value_function": "calmar"}}


def test_a_shared_task_view_is_rebuilt_for_the_objective_reading_it(tmp_path, monkeypatch):
    """Every objective on a task shares one view file; each must read rows exported for itself."""
    import asyncio
    import json

    monkeypatch.setattr(O, "WORK_ROOT", tmp_path / "work")
    data = tmp_path / "data"
    data.mkdir()

    async def fake_call(server, tool, args, timeout_s=0):
        out = Path(args["path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        version = out.parents[3].name                               # <work>/<version>/<target>/<rule>/<tag>/rows.parquet
        pd.DataFrame({"t": pd.to_datetime(["2024-01-02"]), "made_for": [version]}).to_parquet(out)
        (out.parent / "task.json").write_text(json.dumps({"task": args["task"]}))
        return {"rows": 1}

    monkeypatch.setattr(T, "call", fake_call)

    def obj(oid, version):
        return {"id": oid, "split_date": "2024-07-19",
                "metric": {"kind": "task", "task_server": "gex", "task": "gex_intraday", "target": "Close",
                           "task_info": {"version": version}}}

    def made_for():
        return pd.read_parquet(data / T.VIEW_DIR / "gex_gex_intraday.parquet")["made_for"][0]

    a, b = obj("a", "v1"), obj("b", "v2")
    asyncio.run(T.ensure_view(a, str(data)))
    assert made_for() == "v1"
    asyncio.run(T.ensure_view(b, str(data)))
    assert made_for() == "v2"
    asyncio.run(T.ensure_view(a, str(data)))                        # A must not read B's rows
    assert made_for() == "v1"
    evil = obj("c", "../../../outside")                              # a server's version never shapes a path
    asyncio.run(T.ensure_view(evil, str(data)))
    assert not (tmp_path / "outside").exists() and (tmp_path / "work" / "c" / "task").is_dir()


def test_order_actions_keep_every_row_but_held_levels_collapse(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "WORK_ROOT", tmp_path / "work")
    src = tmp_path / "acts.parquet"
    pd.DataFrame({"t": pd.date_range("2024-01-02 14:30", periods=4, freq="10s"), "pos": [1.0, 1.0, 1.0, -1.0]}).to_parquet(src)
    kinds = {k: T.actions_hold({"metric": {"task_info": {"action": {"kind": k}}}})
             for k in ("position", "setpoint", "value", "order", "mystery")}
    assert kinds == {"position": True, "setpoint": True, "value": True, "order": False, "mystery": False}
    assert len(pd.read_parquet(O._keep_positions("o", "c1", src, collapse=True))) == 2
    assert len(pd.read_parquet(O._keep_positions("o", "c2", src, collapse=False))) == 4    # every order


def test_a_disabled_task_server_is_reported_as_disabled(temp_projects, monkeypatch):
    from app import main

    p = temp_projects.create("Gex", task_server="gex")
    spec = mcp_registry.ServerSpec(name="gex", kind="task", transport="http", url="http://127.0.0.1:1/mcp", enabled=False)
    monkeypatch.setattr(mcp_registry, "load_config", lambda: [spec])                   # the only server, disabled
    assert main._project_view(temp_projects.get(p["id"]))["task_server_status"] == "disabled"


def test_analysis_price_takes_only_a_plain_target_name():
    m = {"kind": "task", "task_server": "gex", "task": "x"}
    assert O.analysis_price({"metric": {**m, "target": "Pressure_Total"}}) == "Pressure_Total"
    assert O.analysis_price({"metric": {**m, "target": 'Close" AS p FROM x; --'}}) is None
    assert O.analysis_price({"metric": {"price_column": "Close"}}) == "Close"


def test_task_agents_are_told_how_a_task_objective_ranks():
    assert "R^2" not in O.TASK_RANK_NOTE and "weaker" in O.TASK_RANK_NOTE


def test_project_direction_reaches_the_objective_and_the_harness(temp_projects, tmp_path, monkeypatch):
    import asyncio

    p = temp_projects.create("Gex", task_server="gex")
    temp_projects.update(p["id"], task_options={"gex_intraday": {"direction": "long"}})
    assert temp_projects.get(p["id"])["task_options"] == {"gex_intraday": {"direction": "long"}}
    monkeypatch.setattr(O, "DB_PATH", tmp_path / "o.sqlite3")
    monkeypatch.setattr(O, "_conn", None)
    calls = []
    sides = [{"name": "both"}, {"name": "long"}, {"name": "short"}]

    async def fake_call(server, tool, args, timeout_s=0):
        calls.append((tool, dict(args)))
        offered = server == "gex"
        return {"target": "Close", "holdout_from": "2024-07-19T00:00:00", "cuts": [], "score": {"higher_is_better": True},
                "direction": args.get("direction", "both") if offered else None, "directions": sides if offered else []}

    monkeypatch.setattr(T, "call", fake_call)
    try:
        o = asyncio.run(O.create_objective(p["id"], O.CreateObjective(title="t", metric=O.MetricSpec(kind="task", task="gex_intraday"))))
        assert o["metric"]["direction"] == "long"
        obj = O.get_objective(o["id"])
        asyncio.run(T.evaluate_actions(obj, tmp_path / "a.parquet"))
        asyncio.run(T.action_log(obj, tmp_path / "a.parquet", "2024-01-02", "2024-01-03"))
        assert all(a.get("direction") == "long" for t_, a in calls if t_.startswith("harness_") or t_ == "task_describe")
        # A server that offers no choice of sides never gets one -- not even the metric's "both" default.
        q = temp_projects.create("Battery", task_server="battery-demo")
        calls.clear()
        b = asyncio.run(O.create_objective(q["id"], O.CreateObjective(title="b", metric=O.MetricSpec(kind="task", task="home_battery"))))
        asyncio.run(T.evaluate_actions(O.get_objective(b["id"]), tmp_path / "a.parquet"))
        assert b["metric"]["direction"] is None and not any("direction" in a for _, a in calls)
    finally:
        if O._conn is not None:
            O._conn.close()
        O._conn = None

def test_a_project_must_have_a_data_action_mcp(temp_projects, monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    monkeypatch.setattr(main.auth, "auth_enabled", lambda: False)
    spec = mcp_registry.ServerSpec(name="gex", kind="task", transport="http", url="http://127.0.0.1:1/mcp")
    monkeypatch.setattr(mcp_registry, "load_config", lambda: [spec])
    client = TestClient(main.app)
    r = client.post("/api/projects", json={"name": "NoMcp"})
    assert r.status_code == 400 and "data/action MCP" in r.json()["detail"]
    r = client.post("/api/projects", json={"name": "WithMcp", "task_server": "gex"})
    assert r.status_code == 200 and r.json()["task_server"] == "gex"
    pid = r.json()["id"]
    r = client.post(f"/api/projects/{pid}", json={"task_server": ""})
    assert r.status_code == 400 and "must keep" in r.json()["detail"]
    assert temp_projects.get(pid)["task_server"] == "gex"



def test_task_guidance_is_refreshed_from_the_server_but_not_every_iteration(monkeypatch):
    import asyncio

    calls = []

    async def describe(server, task, target=None, value_function=None, action_rule=None, direction=None):
        calls.append((server, task, action_rule, direction))
        return {"guidance": [{"title": "Trend days and exits", "text": "new advice"}], "version": "v2"}

    monkeypatch.setattr(T, "describe", describe)
    monkeypatch.setattr(T, "_guidance_checked", {})
    obj = {"id": "o-guid", "metric": {"task_server": "gex", "task": "gex_intraday", "action_rule": "intraday",
                                      "direction": "both",
                                      "task_info": {"version": "v1", "directions": [{"name": "both"}],
                                                    "guidance": [{"title": "Costs", "text": "old"}]}}}
    fresh = asyncio.run(T.refresh_guidance(obj))
    assert fresh["task_info"]["guidance"][0]["text"] == "new advice"
    assert fresh["task_info"]["version"] == "v1"                 # only the guidance changes
    assert calls == [("gex", "gex_intraday", "intraday", "both")]
    assert asyncio.run(T.refresh_guidance(obj)) is None and len(calls) == 1      # not again within the TTL
    monkeypatch.setattr(T, "_guidance_checked", {})
    assert asyncio.run(T.refresh_guidance({**obj, "metric": fresh})) is None     # unchanged: nothing to write


def test_trade_limit_rewrites_the_rule_and_rescores_from_kept_actions(tmp_path, monkeypatch):
    import asyncio
    import json
    import time

    monkeypatch.setattr(O, "DB_PATH", tmp_path / "objectives.sqlite3")
    monkeypatch.setattr(O, "_conn", None)
    monkeypatch.setattr(O, "WORK_ROOT", tmp_path / "work")
    now = time.time()
    metric = {"kind": "task", "higher_is_better": True, "task_server": "gex", "task": "gex_intraday",
              "action_rule": "intraday", "task_info": {"action_rule": "intraday", "version": "v1"}}
    O.db().execute("INSERT INTO objectives (id, project_id, title, metric, split_date, created_at, updated_at) "
                   "VALUES ('o1', 'p1', 't', ?, '2024-07-19', ?, ?)", (json.dumps(metric), now, now))
    for seq in (1, 2):
        O.db().execute("INSERT INTO candidates (id, objective_id, seq, created_at, model, status, score, is_score, "
                       "metrics, returns) VALUES (?, 'o1', ?, ?, 'm', 'ok', ?, ?, ?, '[]')",
                       (f"c{seq}", seq, now, 1.0 * seq, 1.0 * seq, json.dumps({"extra": {"n": seq}, "rank": {}})))
    O.db().commit()
    kept = O._kept_positions("o1", "c1")
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.write_bytes(b"x")                                          # c2 has none: counted as failed
    seen = {}

    async def describe(server, task, target=None, value_function=None, action_rule=None, direction=None):
        seen["describe"] = action_rule
        return {"action_rule": action_rule, "action": {"kind": "position"}, "guidance": [{"title": "cap"}]}

    async def evaluate(obj, actions):
        seen["rule"] = obj["metric"]["action_rule"]
        return {"segments": {"in_sample": {"score": -0.5}, "holdout": {"score": -0.7}},
                "curve": [["2024-01-02", 0.001]], "notes": "capped"}

    monkeypatch.setattr(T, "describe", describe)
    monkeypatch.setattr(T, "evaluate_actions", evaluate)
    monkeypatch.setattr(O, "_spawn", lambda coro, what: seen.setdefault("job", coro))
    asyncio.run(O.set_trade_limit("o1", O.TradeLimit(max_trades_per_day=3)))
    obj = O.get_objective("o1")
    assert obj["metric"]["action_rule"] == "intraday+max_3_trades" == seen["describe"]
    assert obj["metric"]["task_info"]["guidance"] == [{"title": "cap"}] and obj["metric"]["task_info"]["version"] == "v1"
    asyncio.run(seen["job"])
    assert seen["rule"] == "intraday+max_3_trades"
    c1 = O.get_candidate("c1")
    assert c1["score"] == -0.7 and c1["metrics"]["extra"] == {"n": 1} and c1["metrics"]["task"]["notes"] == "capped"
    assert O.get_candidate("c2")["score"] == 2.0                    # no kept actions: left as it was
    assert O._REMARKS["o1"]["done"] == 1 and O._REMARKS["o1"]["failed"] == 1
    O.db().close()


def test_an_unscorable_holdout_explains_sparseness_from_in_sample_only():
    """#139 (10-01): 29 active days of 223 in-sample, 7 in the 97-day holdout (20 needed) -- the agent got only
    'could not score that period'. Now it learns the strategy is too sparse, from in-sample numbers alone."""
    from app import task_objectives as T

    ins = {"days": 223, "active_days": 29, "score": 2.4}
    hold = {"days": 97, "active_days": 7, "score": None, "note": "only 7 active days (need 20)"}
    hint = T._sparse_hint(ins, hold)
    assert "SPARSE" in hint and "29 of 223" in hint and "~13" in hint and " 7 " not in hint
    assert T._sparse_hint({"days": 223, "active_days": 200}, hold) == ""        # dense enough: no lesson
