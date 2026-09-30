"""Groq's HTTP 400 tool_use_failed: recover the model's rejected output instead of throwing
away the whole iteration (swarm_runner.converse + _bad_tool_call).

Groq refuses the reply when its parser takes the model's output for a tool call whose
arguments do not match any schema. `failed_generation` in the body is what the model actually
produced -- prose it wanted to say, or a tool call in a syntax Groq could not parse. Before
the fix, the retry silently dropped that text and the model repeated itself for two more
paid rounds before the iteration died.
"""

from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "swarm_runner.py"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("swarm_runner_under_test_bad_tool_call", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- _bad_tool_call: parse the Groq 400 envelope ---------------------------------------------

def _err_400(failed_gen: str) -> str:
    body = json.dumps({"error": {
        "message": "Failed to call a function. Please adjust your prompt.",
        "type": "invalid_request_error",
        "code": "tool_use_failed",
        "failed_generation": failed_gen}})
    return "/v1/chat/completions -> 400: " + body


def test_bad_tool_call_extracts_failed_generation(runner):
    assert runner._bad_tool_call(_err_400("I think the answer is 42")) == "I think the answer is 42"


def test_bad_tool_call_extracts_multiline_prose(runner):
    prose = 'Experiment budget is exhausted.\nBased on the leaderboard, the strongest insight is `x`.'
    assert runner._bad_tool_call(_err_400(prose)) == prose


def test_bad_tool_call_returns_none_for_other_errors(runner):
    assert runner._bad_tool_call("/v1/chat/completions -> 429: Rate limit reached") is None
    assert runner._bad_tool_call("connection refused") is None
    assert runner._bad_tool_call("prompt is too long: 15605 tokens > 8192 maximum") is None


def test_bad_tool_call_returns_empty_when_no_failed_generation(runner):
    body = '{"error":{"code":"tool_use_failed","message":"nope"}}'
    assert runner._bad_tool_call("/v1/chat/completions -> 400: " + body) == ""


def test_bad_tool_call_returns_empty_on_unparseable_body(runner):
    # Provider sometimes gives a bare text body around the phrase.
    assert runner._bad_tool_call('400: tool_use_failed but no JSON here') == ""


# ---- converse: what the retry does with the rejected text ------------------------------------

TOOLS = [{"type": "function", "function": {
    "name": "submit_candidate",
    "description": "submit",
    "parameters": {"type": "object", "properties": {
        "code": {"type": "string"}, "rationale": {"type": "string"}}}}}]


def _worker(runner, model="qwen/qwen3.8-27b@groq"):
    """Build a Worker without invoking __init__ (which starts threads and calls the board)."""
    w = runner.Worker.__new__(runner.Worker)
    w.model = model
    w.agent_name = model
    w.role = "search"
    w.slot = 0
    w.project = {"id": "p1", "slug": "test"}
    w._stop = threading.Event()
    w.retired = threading.Event()
    w._budget_streak = 0
    w.say = MagicMock()
    return w


def _sequential(responses):
    """Return a fake `_generate` that yields the queued responses (Exceptions are raised)."""
    def gen(payload):
        r = responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r
    return gen


@pytest.fixture(autouse=True)
def _quiet_activity(monkeypatch, runner):
    """Silence the activity poster and inspector during converse()."""
    monkeypatch.setattr(runner, "_act", lambda w: MagicMock())


def test_converse_preserves_prose_and_retries_on_tool_use_failed(runner):
    prose = "Experiment budget exhausted. The strongest insight is composite_skew_oinet_vwap."
    responses = [
        RuntimeError(_err_400(prose)),
        {"choices": [{"finish_reason": "stop",
                      "message": {"content": "Final answer: use vwap.", "tool_calls": []}}],
         "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
    ]
    w = _worker(runner)
    w._generate = _sequential(responses)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    ok, text, usage = w.converse(messages, TOOLS, MagicMock(), tag={"task_id": "t1"}, max_rounds=3)
    assert ok is True
    assert text == "Final answer: use vwap."
    # The rejected prose is kept as an assistant turn (never silently dropped) and appears
    # BEFORE the corrective user message that asks for a proper tool call.
    assistant_idx = next(i for i, m in enumerate(messages)
                         if m["role"] == "assistant" and prose in (m.get("content") or ""))
    corrective_idx = next(i for i, m in enumerate(messages)
                          if m["role"] == "user" and "tool_use_failed" in (m.get("content") or ""))
    assert assistant_idx < corrective_idx


def test_converse_salvages_xml_tool_call_from_failed_generation(runner):
    xml = ('<invoke name="submit_candidate">'
           '<parameter name="code">print("hi")</parameter>'
           '<parameter name="rationale">test</parameter>'
           '</invoke>')
    call_mock = MagicMock(return_value={"seq": 1, "ok": True})
    responses = [
        RuntimeError(_err_400(xml)),
        {"choices": [{"finish_reason": "stop",
                      "message": {"content": "done", "tool_calls": []}}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 1}},
    ]
    w = _worker(runner)
    w._generate = _sequential(responses)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    ok, text, usage = w.converse(messages, TOOLS, call_mock, tag={"task_id": "t1"}, max_rounds=3)
    assert ok is True
    call_mock.assert_called_once_with("submit_candidate",
                                     {"code": 'print("hi")', "rationale": "test"})
    # The salvage does NOT append a corrective user message (the round's work counted).
    assert not any("tool_use_failed" in (m.get("content") or "") for m in messages if m["role"] == "user")


def test_converse_treats_prose_as_answer_when_retries_exhaust(runner):
    prose = "I have hit the experiment budget; my saved module is enough."
    # Every generation fails the same way. After BAD_TOOL_CALL_RETRIES retries, the runner
    # keeps the prose rather than letting the iteration die on `generation failed:`.
    responses = [RuntimeError(_err_400(prose))
                 for _ in range(runner.BAD_TOOL_CALL_RETRIES + 1)]
    w = _worker(runner)
    w._generate = _sequential(responses)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    ok, text, usage = w.converse(messages, TOOLS, MagicMock(), tag={"task_id": "t1"}, max_rounds=3)
    assert ok is True
    assert prose in text


def test_converse_still_fails_when_400_carries_no_recoverable_text(runner):
    # Empty failed_generation and retries used up: don't invent an answer, still fail loudly.
    body = '{"error":{"code":"tool_use_failed","failed_generation":""}}'
    err = RuntimeError("/v1/chat/completions -> 400: " + body)
    responses = [err for _ in range(runner.BAD_TOOL_CALL_RETRIES + 1)]
    w = _worker(runner)
    w._generate = _sequential(responses)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    ok, text, usage = w.converse(messages, TOOLS, MagicMock(), tag={"task_id": "t1"}, max_rounds=3)
    assert ok is False
    assert "generation failed" in text
