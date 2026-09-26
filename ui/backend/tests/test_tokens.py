"""Tokens processed per model: engine-counter deltas, relayed usage blocks, the ledger merge,
persistence across restarts, and the /api/usage/tokens route."""

from __future__ import annotations

import asyncio
import time

import pytest

from app import external, main, tokens
from app.engine import EngineSupervisor


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    tokens._pending.clear()             # never flush a stray count into the real database
    tokens._marks.clear()
    tokens.close()
    monkeypatch.setattr(tokens, "DB_PATH", tmp_path / "token_usage.sqlite3")
    yield
    tokens.close()
    tokens._pending.clear()
    tokens._marks.clear()


def _by_model(doc: dict) -> dict:
    return {r["model"]: r for r in doc["models"]}


def _stats(pt: int, ct: int, done: int) -> dict:
    return {"requests": {"active": 0, "completed": done, "prompt_tokens_total": pt, "completion_tokens_total": ct}}


def test_totals_survive_a_restart():
    tokens.record("network", "qwen@box", 100, 20)
    tokens.record("network", "qwen@box", 50, 5)
    tokens.close()                      # flushes, then drops the connection like a shutdown
    row = _by_model(tokens.summary())["qwen@box"]
    assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"], row["requests"]) == (150, 25, 175, 2)
    assert row["source"] == "network" and row["first_seen"] <= row["last_seen"]


def test_engine_counters_are_diffed_per_process():
    tokens.observe_engine("a:1", "Qwen", _stats(1000, 200, 3))   # first reading: all of it is new
    tokens.observe_engine("a:1", "Qwen", _stats(1500, 260, 4))
    tokens.observe_engine("a:1", "Qwen", _stats(1400, 250, 4))   # an older poll answered late
    tokens.observe_engine("a:1", "Qwen", _stats(1500, 260, 4))   # nothing new
    tokens.observe_engine("a:2", "Qwen", _stats(10, 5, 1))       # restarted: a new baseline
    tokens.observe_engine("b:1", "Other", None)                  # stopped / unreachable
    tokens.observe_engine("c:1", "Old", {"requests": {"active": 0}})  # no lifetime counters
    doc = tokens.summary()
    row = _by_model(doc)["Qwen"]
    assert (row["prompt_tokens"], row["completion_tokens"], row["requests"]) == (1510, 265, 5)
    assert row["source"] == "local" and set(_by_model(doc)) == {"Qwen"}


def test_forgetting_keeps_a_grace_period(monkeypatch):
    tokens.observe_engine("a:1", "Qwen", _stats(100, 10, 1))
    tokens.forget_engines(set())
    tokens.observe_engine("a:1", "Qwen", _stats(100, 10, 1))     # a late reading after the stop
    assert _by_model(tokens.summary())["Qwen"]["total_tokens"] == 110
    monkeypatch.setattr(tokens, "FORGET_AFTER_S", -1.0)
    tokens.forget_engines(set())
    assert tokens._marks == {}


def test_usage_blocks_openai_anthropic_and_missing():
    tokens.record_usage("external", "claude@anthropic", {"input_tokens": 7, "output_tokens": 3})
    tokens.record_usage("network", "m@box", {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6})
    tokens.record_usage("network", "m@box", None)
    tokens.record_usage("network", "m@box", {"prompt_tokens": "junk"})
    tokens.record("bogus", "x", 1, 1)                             # unknown source: ignored
    rows = _by_model(tokens.summary())
    assert rows["claude@anthropic"]["total_tokens"] == 10
    m = rows["m@box"]
    assert (m["total_tokens"], m["requests"], m["unmetered"]) == (6, 3, 2)
    assert "x" not in rows


def test_metered_stream_finds_usage_split_across_chunks():
    body = (b'data: {"choices":[{"delta":{"content":"hi"},"index":0}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":34}}\n\n'
            b'data: [DONE]\n\n')
    pieces = [body[i:i + 7] for i in range(0, len(body), 7)]   # arbitrary byte boundaries
    seen: list = []

    async def source():
        for p in pieces:
            yield p

    async def drain():
        return b"".join([c async for c in tokens.metered(source(), seen.append)])

    assert asyncio.run(drain()) == body                          # passed through untouched
    assert seen == [{"prompt_tokens": 12, "completion_tokens": 34}]


def test_metered_stream_without_usage_still_reports_done():
    seen: list = []

    async def source():
        yield b'data: {"choices":[{"delta":{"content":"x"}}]}\n\ndata: [DONE]\n\n'

    async def drain():
        return [c async for c in tokens.metered(source(), seen.append)]

    asyncio.run(drain())
    assert seen == [None]


def test_stream_usage_requested_only_when_unset():
    assert tokens.with_stream_usage({"stream": True})["stream_options"] == {"include_usage": True}
    assert tokens.with_stream_usage({"stream": True, "stream_options": {"include_usage": False}})[
        "stream_options"] == {"include_usage": False}
    assert "stream_options" not in tokens.with_stream_usage({"stream": False})


def test_since_window_and_ledger_merge():
    now = time.time()
    tokens.record("local", "Qwen", 1000, 100, ts=now - 3 * 86_400)
    tokens.record("local", "Qwen", 10, 1, ts=now)
    tokens.record("external", "claude@anthropic", 5, 5, ts=now)
    ledger = [{"source": "external", "model": "gpt@groq", "prompt_tokens": 400, "completion_tokens": 100,
               "requests": 2, "unmetered": 0, "first_seen": now - 60, "last_seen": now},
              {"source": "external", "model": "claude@anthropic", "prompt_tokens": 1, "completion_tokens": 1,
               "requests": 1, "unmetered": 1, "first_seen": now - 5, "last_seen": now - 5}]
    doc = tokens.summary(since=now - 86_400, extra=ledger)
    assert [r["model"] for r in doc["models"]] == ["gpt@groq", "claude@anthropic", "Qwen"]   # largest first
    rows = _by_model(doc)
    assert rows["Qwen"]["total_tokens"] == 11
    assert (rows["claude@anthropic"]["total_tokens"], rows["claude@anthropic"]["requests"]) == (12, 2)
    assert doc["by_source"]["external"]["total_tokens"] == 512
    assert doc["totals"]["total_tokens"] == 523
    assert _by_model(tokens.summary())["Qwen"]["total_tokens"] == 1111


def test_ledger_rows_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(external, "LEDGER_PATH", tmp_path / "ledger.sqlite3")
    monkeypatch.setattr(external, "_ledger", None)
    try:
        external._record("openai/gpt-oss-120b@groq", {"prompt_tokens": 10, "completion_tokens": 4}, "chat")
        external._record("openai/gpt-oss-120b@groq", None, "chat")   # a stream cut off before usage
        [row] = external.token_rows()
        assert row["model"] == "openai/gpt-oss-120b@groq" and row["source"] == "external"
        assert (row["prompt_tokens"], row["completion_tokens"], row["requests"], row["unmetered"]) == (10, 4, 2, 1)
        assert external.token_rows(time.time() + 60) == []
    finally:
        if external._ledger is not None:
            external._ledger.close()


class _Proc:
    pid = 1

    def poll(self):
        return None


def test_sampler_reads_every_running_engine(monkeypatch):
    engines = {}
    for key, model in (("a", "Qwen"), ("b", "DeepSeek")):
        e = EngineSupervisor(key, 30000)
        e.model_id, e.served_name, e.state, e._proc, e.started_at = model, model.lower(), "running", _Proc(), 123.0
        engines[key] = e
    monkeypatch.setattr(main.manager, "_instances", engines)
    readings = {"a": _stats(300, 30, 2), "b": _stats(50, 5, 1)}

    async def fake_get(path, inst=None):
        assert path == "/v1/stats"
        return readings[inst.instance_id]

    monkeypatch.setattr(main, "_engine_get", fake_get)
    asyncio.run(main._sample_engines())
    readings["a"] = _stats(400, 40, 3)
    asyncio.run(main._sample_engines())
    rows = _by_model(tokens.summary())
    assert (rows["Qwen"]["total_tokens"], rows["Qwen"]["requests"]) == (440, 3)
    assert rows["DeepSeek"]["total_tokens"] == 55


def test_route(monkeypatch):
    from fastapi.testclient import TestClient

    from app import auth

    tokens.record("local", "Qwen", 900, 100)
    monkeypatch.setattr(external, "token_rows", lambda since=None: [
        {"source": "external", "model": "gpt@groq", "prompt_tokens": 5, "completion_tokens": 5,
         "requests": 1, "unmetered": 0, "first_seen": 1.0, "last_seen": 2.0}])
    main.app.dependency_overrides[auth.require_user] = lambda: "tester"
    try:
        c = TestClient(main.app)                 # no `with`: the lifespan (engines, federation) stays off
        doc = c.get("/api/usage/tokens").json()
        assert doc["window"] == "all" and doc["since"] is None
        assert [(r["model"], r["source"], r["total_tokens"]) for r in doc["models"]] == [
            ("Qwen", "local", 1000), ("gpt@groq", "external", 10)]
        assert doc["totals"]["total_tokens"] == 1010 and doc["by_source"]["network"]["requests"] == 0
        day = c.get("/api/usage/tokens?since=24h").json()
        assert day["window"] == "24h" and day["since"] > time.time() - 86_500
        assert c.get("/api/usage/tokens?since=1y").status_code == 422
    finally:
        main.app.dependency_overrides.pop(auth.require_user, None)
