"""The look-ahead test runs after the agent has moved on, then settles the candidate."""

from __future__ import annotations

import asyncio

from app import objectives as O


def _wire(monkeypatch, cand: dict, verdict):
    obj = {"id": "o1", "project_id": "p1", "require_audit": True, "lookahead_check": True,
           "split_date": "2024-07-19", "metric": {"kind": "sharpe"}, "best_id": None}
    settled, posts = [], []

    async def fake_lookahead(*a, **k):
        if isinstance(verdict, Exception):
            raise verdict
        return verdict

    monkeypatch.setattr(O, "_lookahead", fake_lookahead)
    monkeypatch.setattr(O, "get_candidate", lambda cid, light=False: cand)
    monkeypatch.setattr(O, "get_objective", lambda oid: obj)
    monkeypatch.setattr(O, "_settle", lambda o, cid, seq, code, fields: settled.append(fields) or True)
    monkeypatch.setattr(O, "_board_post", lambda *a: posts.append(a))
    return obj, settled, posts


def test_a_pass_settles_the_candidate_and_announces_a_contender(monkeypatch, tmp_path):
    cand = {"id": "c1", "status": "ok", "score": 2.3, "lookahead": "pending"}
    obj, settled, posts = _wire(monkeypatch, cand, ("pass", "10 cuts, identical"))
    kept = tmp_path / "c1.parquet"
    kept.write_bytes(b"x")
    asyncio.run(O._settle_lookahead(obj, "c1", 7, "code", "d", [], True, kept, []))
    assert settled == [{"status": "ok", "score": 2.3, "lookahead": "pass", "lookahead_detail": "10 cuts, identical"}]
    assert posts and "contender for best" in posts[0][3]
    assert not kept.exists()                                   # the copied positions are cleaned up


def test_a_verdict_already_replaced_is_left_alone(monkeypatch):
    cand = {"id": "c1", "status": "ok", "score": 2.3, "lookahead": "pass"}   # re-tested meanwhile
    obj, settled, _ = _wire(monkeypatch, cand, ("fail", "leak"))
    asyncio.run(O._settle_lookahead(obj, "c1", 7, "code", "d", [], True, None, []))
    assert settled == []


def test_a_crash_is_an_error_verdict_not_a_leak(monkeypatch):
    cand = {"id": "c1", "status": "ok", "score": 2.3, "lookahead": "pending"}
    obj, settled, posts = _wire(monkeypatch, cand, RuntimeError("docker went away"))
    asyncio.run(O._settle_lookahead(obj, "c1", 7, "code", "d", [], True, None, []))
    assert settled[0]["lookahead"] == "error" and "docker went away" in settled[0]["lookahead_detail"]
    assert posts[0][1] == "errors"
