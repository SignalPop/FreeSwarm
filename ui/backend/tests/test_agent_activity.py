"""The agent inspector: the activity store and routes, forecast attribution, and the runner's recorder."""

from __future__ import annotations

import asyncio
import importlib.util
import types
from pathlib import Path

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from app import agent_activity as A

RUNNER = Path(__file__).resolve().parents[1] / "swarm_runner.py"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "DB_PATH", tmp_path / "agent_activity.sqlite3")
    monkeypatch.setattr(A, "_conn", None)
    A.reset()
    yield A
    if A._conn is not None:
        A._conn.close()
    A.reset()


@pytest.fixture
def client(store):
    app = FastAPI()
    api = APIRouter(prefix="/api")
    api.include_router(A.router)

    # Stand-in for objectives' feature build: several batched forecast calls in one request.
    @api.post("/objectives/{oid}/features")
    async def features(oid: str, body: dict) -> dict:
        for _ in range(3):
            A.note_forecast("amazon/chronos-2", {
                "inputs": [{"target": [1.0] * 64, "past_covariates": {"GEX": [0.0] * 64, "IntrVol": [0.0] * 64}}] * 10,
                "horizon": body.get("horizon", 12), "quantiles": [0.1, 0.5, 0.9]}, 0.2, types.SimpleNamespace(status_code=200))
        return {"echo": body}

    app.include_router(api)
    app.add_middleware(A.CallerMiddleware)
    return TestClient(app)


def rec(rid: str, started: float, **extra) -> dict:
    return {"id": rid, "mode": "explore", "status": "running", "started_at": started,
            "objective": {"id": "o1", "title": "Best strategy"}, "asked": [], "timeline": [], "chats": [],
            "submissions": [], **extra}


def post(client, agent="Qwen3.6-35B-A3B", model="Qwen3.6-35B-A3B", role="search", record=None, project="p1"):
    return client.post("/api/agents/activity", json={"agent": agent, "model": model, "role": role, "slot": 0,
                                                      "project_id": project, "record": record})


# ---------------------------------------------------------------------------------------
# Store and routes
# ---------------------------------------------------------------------------------------
def test_upsert_by_record_id_and_keep_newest(client, store, monkeypatch):
    monkeypatch.setattr(A, "KEEP_RECORDS", 3)
    for i in range(5):
        assert post(client, record=rec(f"r{i}", 100.0 + i)).status_code == 200
    # The same record again (a later snapshot of the running iteration) replaces, not appends.
    post(client, record=rec("r4", 104.0, status="submitted", submissions=[{"seq": 7, "status": "ok"}]))
    doc = client.get("/api/agents/activity/Qwen3.6-35B-A3B").json()
    (agent,) = doc["agents"]
    assert [r["id"] for r in agent["records"]] == ["r2", "r3", "r4"]
    assert agent["records"][-1]["status"] == "submitted"


def test_list_is_scoped_by_project_and_leaves_out_prompts(client, store):
    post(client, record=rec("a", 1.0, asked=[{"model": "Qwen3.6-35B-A3B", "system": "x" * 1000, "prompt": "y"}],
                            timeline=[{"kind": "tool", "name": "query_data"}], pending={"kind": "chat"}))
    post(client, agent="other", model="other", record=rec("b", 2.0), project="p2")
    doc = client.get("/api/agents/activity", params={"project_id": "p1"}).json()
    (a,) = doc["agents"]
    assert a["agent"] == "Qwen3.6-35B-A3B"
    assert a["current"]["tool_calls"] == 1 and a["current"]["last_tool"] == "query_data"
    assert "asked" not in a["current"]
    assert len(client.get("/api/agents/activity").json()["agents"]) == 2


def test_model_view_gathers_slots_mentor_and_peer_use(client, store):
    m = "gpt-oss-120b@groq"
    post(client, agent=m, model=m, record=rec("s0", 1.0))
    post(client, agent=f"{m} #2", model=m, record=rec("s1", 2.0))
    post(client, agent=f"{m} (mentor)", model=m, role="mentor", record=rec("mt", 3.0, mode="mentor"))
    # Another agent's audit went to this model.
    post(client, agent="Qwen", model="Qwen", record=rec("au", 4.0, mode="audit", chats=[
        {"model": m, "prompt_tokens": 900, "completion_tokens": 50}],
        asked=[{"model": m, "prompt": "You are auditing..."}]))
    doc = client.get(f"/api/agents/activity/{m}").json()
    assert [a["agent"] for a in doc["agents"]] == [m, f"{m} #2", f"{m} (mentor)"]
    (peer,) = doc["peer_calls"]
    assert peer["agent"] == "Qwen" and peer["mode"] == "audit" and peer["prompt_tokens"] == 900
    assert peer["asked"][0]["prompt"].startswith("You are auditing")


def test_model_path_with_slash(client, store):
    post(client, agent="org/model-7b", model="org/model-7b", record=rec("x", 1.0))
    assert client.get("/api/agents/activity/org/model-7b").json()["agents"][0]["model"] == "org/model-7b"


def test_rejects_record_without_id(client, store):
    assert post(client, record={"mode": "explore"}).status_code == 400


def test_persists_across_restart(client, store):
    post(client, record=rec("r1", 1.0))
    A.flush()
    A._agents.clear()
    A._loaded = False
    doc = client.get("/api/agents/activity/Qwen3.6-35B-A3B").json()
    assert doc["agents"][0]["records"][0]["id"] == "r1"


# ---------------------------------------------------------------------------------------
# Forecast attribution
# ---------------------------------------------------------------------------------------
def test_forecast_calls_grouped_by_request_with_caller_and_recipe(client, store):
    body = {"columns": ["GEX", "IntrVol"], "covariates": ["Pressure_Below"], "calendar": True,
            "horizon": 12, "every": 0, "model": "amazon/chronos-2"}
    r = client.post("/api/objectives/o%201/features", json=body,
                    headers={"X-FreeSwarm-Agent": "Qwen3.6-35B-A3B%20%232"})
    assert r.json() == {"echo": body}  # the middleware read the body and replayed it intact
    client.post("/api/objectives/o1/features", json={"column": "GEX"})
    rows = client.get("/api/agents/forecasts", params={"model": "amazon/chronos-2"}).json()["forecasts"]
    assert len(rows) == 2
    first = rows[1]  # newest first
    assert first["agent"] == "Qwen3.6-35B-A3B #2"
    assert first["via"] == "forecast_feature" and first["objective_id"] == "o 1"
    assert first["request"]["columns"] == ["GEX", "IntrVol"] and first["request"]["calendar"] is True
    assert first["calls"] == 3 and first["anchors"] == 30
    assert first["shape"]["past_covariates"] == ["GEX", "IntrVol"] and first["shape"]["context"] == 64
    assert rows[0]["agent"] is None and rows[0]["request"] == {"column": "GEX"}
    # ... and the forecaster's inspector view carries them.
    assert len(client.get("/api/agents/activity/amazon/chronos-2").json()["forecasts"]) == 2


def test_forecast_shapes():
    assert A._shape({"series": [1.0] * 100, "horizon": 6}) == {"horizon": 6, "kind": "series", "anchors": 1, "context": 100}
    s = A._shape({"series": [[1.0] * 50] * 4, "horizon": 6})
    assert s["anchors"] == 4 and s["context"] == 50
    k = A._shape({"candles": [[[1, 2, 3, 4]] * 20] * 3, "samples": 4, "freq_seconds": 60, "horizon": 12})
    assert k["kind"] == "candles" and k["anchors"] == 3 and k["context"] == 20


def test_note_forecast_never_raises(store):
    A.note_forecast("m", None, None, None)  # type: ignore[arg-type]
    A.note_forecast("m", {"inputs": "garbage"}, 0.1, object())
    assert all(e["model"] == "m" for e in A.forecasts())


def test_tsfm_forecast_is_noted(store, monkeypatch):
    from app import tsfm

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json):
            return types.SimpleNamespace(status_code=200, json=lambda: {"forecasts": []}, text="")

    monkeypatch.setattr(tsfm.httpx, "AsyncClient", FakeClient)
    mgr = tsfm.TsManager()
    inst = tsfm.TsInstance(id="ts1960", model_id="amazon/chronos-2", path=".", gpu="1", port=1960, state="running")
    mgr._instances[inst.id] = inst
    asyncio.run(mgr.forecast("amazon/chronos-2", {"series": [1.0] * 32, "horizon": 4}))
    (row,) = A.forecasts("amazon/chronos-2")
    assert row["calls"] == 1 and row["shape"]["context"] == 32 and inst.calls == 1


# ---------------------------------------------------------------------------------------
# The runner's recorder
# ---------------------------------------------------------------------------------------
@pytest.fixture
def runner(monkeypatch):
    spec = importlib.util.spec_from_file_location("swarm_runner_activity_test", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    posted: list[dict] = []
    monkeypatch.setattr(mod, "_poster", lambda: types.SimpleNamespace(put=lambda key, doc: posted.append(doc)))
    mod.posted = posted
    return mod


def worker(**kw):
    return types.SimpleNamespace(agent_name="Qwen #2", model="Qwen", role="search", slot=1, pid="p1", **kw)


def test_recorder_captures_an_iteration(runner):
    w = worker()
    act = runner._act(w)
    assert runner._agent_header() == {"X-FreeSwarm-Agent": "Qwen%20%232"}
    ctx = {"mode": "improve", "parent": {"id": "c9", "seq": 9, "rank": 2, "code": "x" * 50},
           "ideas": [{"id": 4, "model": "big", "tried": 1, "text": "try GEX"}], "lessons": ["a", "b"]}
    act.begin_iteration({"id": "o1", "title": "Best strategy"}, ctx)
    system, prompt = "S" * 300_000, "Improve candidate #9"
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    act.chat_start({"model": "Qwen", "messages": msgs, "tools": [{"function": {"name": "forecast_feature"}}]})
    call = {"function": {"name": "forecast_feature", "arguments": "{}"}}
    act.chat_done({"model": "Qwen", "messages": msgs}, {
        "usage": {"prompt_tokens": 1200, "completion_tokens": 80},
        "choices": [{"finish_reason": "tool_calls", "message": {"content": "", "tool_calls": [call]}}]})
    args = {"columns": ["GEX"], "covariates": ["IntrVol"], "calendar": True, "horizon": 12, "every": 0}
    act.tool_start("forecast_feature", args)
    act.tool_done("forecast_feature", args, {"view": "fc_gex", "skill": {"GEX": 0.1}}, True)
    msgs += [{"role": "assistant", "content": ""}, {"role": "user", "content": "Call submit_candidate NOW"}]
    act.chat_start({"model": "Qwen", "messages": msgs})
    act.chat_done({"model": "Qwen", "messages": msgs}, None, "cannot reach engine")
    sub = {"code": "import ft", "rationale": "gex trend", "idea": 4}
    act.tool_start("submit_candidate", sub)
    act.tool_done("submit_candidate", sub, {"candidate_id": "c10", "seq": 10, "status": "ok", "rank": 3,
                                            "in_sample_score": 1.2, "stdout_tail": "noise"}, True)
    act.end()

    r = runner.posted[-1]["record"]
    assert runner.posted[-1]["agent"] == "Qwen #2" and runner.posted[-1]["slot"] == 1
    assert r["mode"] == "improve" and r["parent"]["seq"] == 9 and "code" not in r["parent"]
    assert r["ideas_offered"][0]["id"] == 4 and r["context"]["lessons"] == 2
    (asked,) = r["asked"]
    assert asked["system_chars"] == 300_000 and len(asked["system"]) < 251_000 and "omitted" in asked["system"]
    assert asked["prompt"] == prompt and asked["tools"] == ["forecast_feature"]
    kinds = [e["kind"] for e in r["timeline"]]
    assert kinds == ["asked", "chat", "tool", "message", "chat", "tool"]
    tool = r["timeline"][2]
    assert tool["args"]["covariates"] == ["IntrVol"] and tool["ok"] and "fc_gex" in tool["result"]
    assert r["timeline"][4]["error"] == "cannot reach engine"
    assert r["timeline"][5]["result"] == {"candidate_id": "c10", "seq": 10, "status": "ok", "in_sample_score": 1.2,
                                          "rank": 3}
    assert r["submissions"][0]["seq"] == 10 and r["idea"] == 4
    assert r["tokens"] == {"prompt": 1200, "completion": 80, "chats": 2}
    assert r["status"] == "submitted" and r["outcome"] == "#10 ok" and r["pending"] is None


def test_recorder_marks_an_abandoned_job_and_chores(runner):
    act = runner._act(worker())
    act.begin_iteration({"id": "o1", "title": "t"}, {"mode": "explore"})
    first = runner.posted[-1]["record"]["id"]
    act.begin_iteration({"id": "o1", "title": "t"}, {"mode": "explore", "audit": {"id": "c3", "seq": 3}})
    abandoned = next(d["record"] for d in reversed(runner.posted) if d["record"]["id"] == first)
    assert abandoned["status"] == "interrupted"
    act.end()
    r = runner.posted[-1]["record"]
    assert r["mode"] == "audit" and r["audit_of"]["seq"] == 3 and r["status"] == "done"


def test_recorder_never_raises(runner):
    act = runner._act(worker())
    act.chat_start({"messages": "not a list"})
    act.tool_done("x", None, object(), True)
    act.chat_done({}, {"choices": "bad"})
    act.end()
