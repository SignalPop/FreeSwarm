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
    pos = P.manage(keys, keys, acts, P.Rules(max_position=3.0, flat_each_day=intraday))
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
    temp_projects.update(p["id"], task_options={"gex_intraday": {"target": "Open", "value_function": "calmar", "junk": "x"},
                                                "other": {}})
    assert temp_projects.get(p["id"])["task_options"] == {"gex_intraday": {"target": "Open", "value_function": "calmar"}}
    monkeypatch.setattr(O, "DB_PATH", tmp_path / "o.sqlite3")
    monkeypatch.setattr(O, "_conn", None)
    calls = []

    async def fake_call(server, tool, args, timeout_s=0):
        calls.append((tool, dict(args)))
        return {"target": args.get("target", "Close"), "value_function": args.get("value_function", "sharpe"),
                "holdout_from": "2024-07-19T00:00:00", "cuts": [], "score": {"higher_is_better": True}}

    monkeypatch.setattr(T, "call", fake_call)
    try:
        o = asyncio.run(O.create_objective(p["id"], O.CreateObjective(title="t", metric=O.MetricSpec(kind="task", task="gex_intraday"))))
        assert o["metric"]["target"] == "Open" and o["metric"]["value_function"] == "calmar"
        assert ("task_describe", {"task": "gex_intraday", "target": "Open", "value_function": "calmar"}) in calls
        # the in-sample view for the analysis tools is exported at the split, on the same target
        exports = [a for t_, a in calls if t_ == "harness_export_rows"]
        assert exports and exports[0]["until"] == "2024-07-19" and exports[0]["target"] == "Open"
        obj = O.get_objective(o["id"])
        asyncio.run(T.evaluate_actions(obj, tmp_path / "a.parquet"))
        asyncio.run(T.action_log(obj, tmp_path / "a.parquet", "2024-01-02", "2024-01-03"))
        assert all(a.get("target") == "Open" for t_, a in calls if t_.startswith("harness_"))
        assert all(a.get("value_function") == "calmar" for t_, a in calls if t_ in ("harness_evaluate", "harness_export_rows"))
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
