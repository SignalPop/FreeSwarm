"""Errors say what failed: the global 500 handler, and the operator's audit/re-run doors."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import main
from app import objectives as O


def test_unhandled_error_names_type_message_and_line():
    @main.app.get("/__boom_for_test")
    async def boom() -> dict:
        {}["missing"]  # noqa: B018 -- a KeyError, the classic bare "500"

    client = TestClient(main.app, raise_server_exceptions=False)
    r = client.get("/__boom_for_test")
    assert r.status_code == 500
    body = r.json()
    assert body["error_type"] == "KeyError"
    assert "KeyError" in body["detail"] and "'missing'" in body["detail"]
    assert "test_error_detail.py" in body["detail"]          # where it happened
    assert "Traceback" in body["traceback"]


def test_json_verdict_finds_the_object_inside_prose():
    assert O._json_verdict('Looks fine. {"passed": false, "issues": ["hard-coded date"]} thanks') == {
        "passed": False, "issues": ["hard-coded date"]}
    assert O._json_verdict("no json here") is None
    assert O._json_verdict('{"other": 1} then {"passed": true}') == {"passed": True}


def test_rerun_refuses_a_healthy_candidate(monkeypatch):
    obj = {"id": "o1", "best_id": None}
    cand = {"id": "c1", "objective_id": "o1", "seq": 7, "status": "ok", "lookahead": "pass", "score": 1.2}
    monkeypatch.setattr(O, "get_objective", lambda oid: obj)
    monkeypatch.setattr(O, "get_candidate", lambda cid, light=False: cand)
    with pytest.raises(HTTPException) as e:
        asyncio.run(O.rerun("o1", "c1"))
    assert e.value.status_code == 409 and "only failed evaluations" in e.value.detail


def test_rerun_scores_the_same_candidate_in_place(monkeypatch):
    obj = {"id": "o1", "best_id": None}
    cand = {"id": "c1", "objective_id": "o1", "seq": 7, "status": "error", "lookahead": "skipped", "score": None,
            "code": "print(1)", "rationale": "r", "model": "m", "mode": "improve", "parent_id": None, "idea_id": None}
    seen = {}

    async def fake_evaluate(o, req, rerun=None):
        seen.update(req=req, rerun=rerun)
        return {"seq": rerun["seq"]}

    monkeypatch.setattr(O, "get_objective", lambda oid: obj)
    monkeypatch.setattr(O, "get_candidate", lambda cid, light=False: cand)
    monkeypatch.setattr(O, "evaluate", fake_evaluate)
    assert asyncio.run(O.rerun("o1", "c1")) == {"seq": 7}
    assert seen["rerun"] is cand and seen["req"].code == "print(1)"


def test_no_get_route_requires_a_body():
    """A GET that demands a JSON body always fails with 422. That happened when a helper was
    inserted between @router.get(...) and its function: the decorator took the helper, whose
    dict parameter FastAPI read as the body, and the leaderboard went blank. Checked through
    the OpenAPI schema, which sees every mounted router (app.routes does not, here)."""
    spec = main.app.openapi()
    bad = [path for path, ops in spec["paths"].items() if "requestBody" in (ops.get("get") or {})]
    assert bad == []


def test_candidate_list_route_takes_the_list_parameters():
    spec = main.app.openapi()
    get = spec["paths"]["/api/objectives/{oid}/candidates"]["get"]
    names = {p["name"] for p in get.get("parameters", [])}
    assert {"oid", "order", "limit"} <= names and "requestBody" not in get
