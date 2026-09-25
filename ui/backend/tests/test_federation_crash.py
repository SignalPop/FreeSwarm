"""A shared model that crashes is reported to connected computers, not silently dropped."""

from __future__ import annotations

import pytest

from app import federation, main
from app.engine import EngineSupervisor


class _Proc:
    def __init__(self, alive: bool):
        self.alive, self.pid = alive, 1

    def poll(self):
        return None if self.alive else 1


def _engine(key, model, state, alive, error=None, hint=None):
    e = EngineSupervisor(key, 30000)
    e.model_id, e.served_name, e.state, e._proc = model, model.lower(), state, _Proc(alive)
    e.error = error
    e.diagnosis = {"hint": hint} if hint else None
    return e


@pytest.fixture
def engines(monkeypatch):
    monkeypatch.setattr(main.manager, "_instances", {
        "a": _engine("a", "Qwen", "running", True),
        "b": _engine("b", "DeepSeek", "error", False, "torch.OutOfMemoryError: CUDA out of memory",
                     "The GPU ran out of VRAM. Lower KV pages"),
        "c": _engine("c", "Unloaded", "stopped", False),
    })


def test_local_models_include_crashed_not_unloaded(engines):
    by_name = {m["name"]: m for m in main._federation_models()}
    assert set(by_name) == {"Qwen", "DeepSeek"}
    assert by_name["Qwen"]["ready"] and "error" not in by_name["Qwen"]
    crashed = by_name["DeepSeek"]
    assert not crashed["ready"] and crashed["port"] is None
    assert "out of memory" in crashed["error"] and crashed["hint"] == "The GPU ran out of VRAM"


def test_live_instance_wins_over_crashed_one(monkeypatch):
    monkeypatch.setattr(main.manager, "_instances", {
        "old": _engine("old", "DeepSeek", "error", False, "oom"),
        "new": _engine("new", "DeepSeek", "running", True),
    })
    [m] = main._federation_models()
    assert m["ready"] and "error" not in m


def test_gateway_reports_crash_and_refuses_inference(engines, monkeypatch):
    from fastapi.testclient import TestClient

    federation.configure(main._federation_models)
    real_load = federation._load
    monkeypatch.setattr(federation, "_load", lambda: {**real_load(), "sharing": True, "shared_models": ["DeepSeek"]})
    monkeypatch.setattr(federation, "_client_for_token", lambda tok: "client-1")
    client = TestClient(federation.build_gateway(), client=("10.0.0.3", 5000))
    auth = {"Authorization": "Bearer t"}

    [m] = client.get("/fed/models", headers=auth).json()["models"]
    assert m["name"] == "DeepSeek" and m["ready"] is False and "out of memory" in m["error"]

    r = client.post("/fed/v1/chat/completions", headers=auth, json={"model": "DeepSeek", "messages": []})
    assert r.status_code == 503 and "stopped on this computer" in r.json()["detail"]


def test_using_side_never_routes_to_a_crashed_model(monkeypatch):
    monkeypatch.setitem(federation._remote_models, "n1", {"status": "ok", "models": [
        {"name": "X", "ready": True}, {"name": "D", "ready": False, "error": "oom"}]})
    real_load = federation._load
    monkeypatch.setattr(federation, "_load", lambda: {
        **real_load(), "peers": {"n1": {"name": "LAMBDA999", "address": "10.0.0.25", "port": 8443}}})
    assert [m["model"] for m in federation.remote_loaded()] == ["X@lambda999"]
    assert federation.route("D@lambda999") is None
