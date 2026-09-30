"""Same-args, same-error tool re-issues (bug #12 -- Qwen/Qwen3-0.6B) and mid-turn retire
close of the activity record.

The console's bug list showed one agent re-sending an identical failing library_save 7 times
in a single iteration; another iteration by the same model repeated 3 times. Every other
tool had zero same-args repeats across 104 iterations (measured in agent_activity.sqlite3):
so on the very first duplicate we intercept without executing, and after REPEAT_TOOL_LIMIT
intercepts we end the iteration cleanly rather than burn the rest of the tool budget.

The other bug covered here: a Worker retired or stopped mid-turn used to leave its
agent-activity record with status "running" and a pending chat/tool forever, because
_act(self).end() only ran after iterate() returned normally.
"""

from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "swarm_runner.py"

TOOLS = [{"type": "function", "function": {
    "name": "library_save",
    "description": "save",
    "parameters": {"type": "object", "properties": {
        "name": {"type": "string"}, "kind": {"type": "string"}, "code": {"type": "string"}}}}}]


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("swarm_runner_under_test_repeats", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _worker(runner, model="Qwen/Qwen3-0.6B"):
    """Worker without __init__ (which starts threads and calls the board). `pid` is a property
    over `project["id"]`, so it is not settable directly."""
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
    def gen(payload):
        r = responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r
    return gen


def _mk_call(tool_name: str, args: dict) -> dict:
    return {"choices": [{"finish_reason": "tool_calls", "message": {
        "content": "", "tool_calls": [{"id": "c1", "type": "function",
                                        "function": {"name": tool_name,
                                                     "arguments": json.dumps(args)}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def _mk_stop(text: str = "done") -> dict:
    return {"choices": [{"finish_reason": "stop",
                         "message": {"content": text, "tool_calls": []}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}}


@pytest.fixture(autouse=True)
def _quiet_activity(monkeypatch, runner):
    """Silence the activity poster during converse(), but let _Activity itself run so end()
    reflects reality (some tests read the record state)."""
    class _NoopPoster:
        def put(self, key, doc):
            pass

    monkeypatch.setattr(runner, "_poster", lambda: _NoopPoster())


# ---- _tool_call_hash: what counts as "the same call" ---------------------------------------

def test_tool_call_hash_identical_across_whitespace_and_key_order(runner):
    a = {"name": "x", "kind": "signal", "code": "def main():\n    return 1\n"}
    b = {"code": "def main():\n\treturn 1", "kind": "signal", "name": "x"}
    assert runner._tool_call_hash("library_save", a) == runner._tool_call_hash("library_save", b)


def test_tool_call_hash_different_when_args_differ(runner):
    a = {"name": "x", "kind": "signal", "code": "def main(): pass"}
    b = {"name": "x", "kind": "signal", "code": "def signal(df): pass"}
    assert runner._tool_call_hash("library_save", a) != runner._tool_call_hash("library_save", b)


def test_tool_call_hash_different_when_tool_differs(runner):
    args = {"code": "print(1)"}
    assert runner._tool_call_hash("run_python", args) != runner._tool_call_hash("library_save", args)


# ---- converse: first failure runs, duplicate is intercepted --------------------------------

def test_repeat_failing_call_is_intercepted_without_executing(runner):
    """The same (name, args) is executed once (failure), then blocked on the second sighting.
    The blocked call must not reach `call(...)`, and the tool result must mention that the
    same call already failed."""
    args = {"name": "qwen_gex", "kind": "signal", "code": "def main(): pass"}
    responses = [
        _mk_call("library_save", args),   # round 1: model calls library_save (fails at API)
        _mk_call("library_save", args),   # round 2: model calls it again with identical args
        _mk_stop("giving up"),            # round 3: model finally answers
    ]
    calls_made: list[tuple] = []

    def tool_call(name, a):
        calls_made.append((name, a))
        return {"error": "a signal module must define signal(df, ...)"}

    w = _worker(runner)
    w._generate = _sequential(responses)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    ok, text, usage = w.converse(messages, TOOLS, tool_call, tag={"objective_id": "o"}, max_rounds=5)
    # Exactly one real invocation of the tool: the first. The duplicate never reached call().
    assert len(calls_made) == 1
    # The tool result on the intercepted round names the previous call and error.
    intercept = next(m for m in messages if m["role"] == "tool" and m.get("tool_call_id") == "c1"
                     and "already called" in json.dumps(m.get("content"), default=str).lower())
    assert "library_save" in json.dumps(intercept["content"], default=str)


def test_iteration_ends_after_REPEAT_TOOL_LIMIT_intercepts(runner):
    """More than REPEAT_TOOL_LIMIT duplicates of the same failing call end the iteration."""
    args = {"name": "qwen_gex", "kind": "signal", "code": "def main(): pass"}
    # 1 real failure + many duplicates: converse should exit before consuming them all.
    responses = [_mk_call("library_save", args) for _ in range(runner.REPEAT_TOOL_LIMIT + 4)]
    calls_made: list[tuple] = []

    def tool_call(name, a):
        calls_made.append((name, a))
        return {"error": "a signal module must define signal(df, ...)"}

    w = _worker(runner)
    # Prime a rec as iterate() would via begin_iteration() -- the direct _generate mock skips
    # the chat_start path that otherwise creates one.
    runner._act(w).begin("build", {"id": "o", "title": "T"})
    w._generate = _sequential(responses)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    ok, text, usage = w.converse(messages, TOOLS, tool_call, tag={"objective_id": "o"},
                                 max_rounds=runner.REPEAT_TOOL_LIMIT + 8)
    assert ok is False
    assert "identical repeat" in text
    # Real executions: just the first. Every re-issue was intercepted.
    assert len(calls_made) == 1
    # And the activity record was closed with a clean status/reason (not "running").
    rec = w.__dict__["_activity"].rec
    assert rec is not None
    assert rec["status"] == "stopped"
    assert rec.get("pending") is None
    assert rec.get("ended_at") is not None
    assert "identical repeat" in rec.get("reason", "")


def test_different_args_are_not_intercepted(runner):
    """Two failing calls that DIFFER in args must both be executed -- interception is only
    for exact re-issues."""
    a1 = {"name": "x", "kind": "signal", "code": "def main(): pass"}
    a2 = {"name": "x", "kind": "signal", "code": "def signal(df): pass\n# v2"}
    responses = [
        _mk_call("library_save", a1),
        _mk_call("library_save", a2),
        _mk_stop("ok"),
    ]
    calls_made: list[tuple] = []

    def tool_call(name, a):
        calls_made.append((name, a))
        return {"error": "smoke test failed"}

    w = _worker(runner)
    w._generate = _sequential(responses)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    ok, text, usage = w.converse(messages, TOOLS, tool_call, tag={"objective_id": "o"}, max_rounds=5)
    assert len(calls_made) == 2   # both attempts ran; neither was intercepted


def test_successful_call_is_not_tracked(runner):
    """A call that SUCCEEDED shouldn't cause an identical re-issue to be intercepted -- the
    tracker is for FAILING calls only (a legitimate re-check with the same args is fine)."""
    args = {"name": "x", "kind": "signal", "code": "def signal(df): pass"}
    responses = [
        _mk_call("library_save", args),
        _mk_call("library_save", args),
        _mk_stop("done"),
    ]
    calls_made: list[tuple] = []

    def tool_call(name, a):
        calls_made.append((name, a))
        return {"saved": True, "version": len(calls_made)}   # success

    w = _worker(runner)
    w._generate = _sequential(responses)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    ok, text, usage = w.converse(messages, TOOLS, tool_call, tag={"objective_id": "o"}, max_rounds=5)
    assert len(calls_made) == 2   # both ran (both succeeded); no intercept


# ---- retire mid-turn: the record is closed with a clear status -----------------------------

def test_stop_mid_turn_closes_activity_record(runner):
    """When _stop is set mid-turn, converse returns AND the activity record moves to
    status=interrupted with the reason, so the console's monitor stops reading it as stuck."""
    w = _worker(runner)
    # Prime an activity record as though the iteration had started.
    a = runner._Activity(w)
    a.rec = {"id": "rec-1", "mode": "build", "status": "running", "objective": None,
             "started_at": 1.0, "ended_at": None, "pending": {"kind": "chat", "model": w.model,
                                                              "since": 2.0},
             "asked": [], "timeline": [], "chats": [], "submissions": [],
             "tokens": {"prompt": 0, "completion": 0, "chats": 0}}
    w.__dict__["_activity"] = a
    w._stop.set()

    def gen(payload):
        raise AssertionError("_generate should not be called after stop")

    w._generate = gen
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    ok, text, usage = w.converse(messages, TOOLS, MagicMock(), tag={"objective_id": "o"}, max_rounds=3)
    assert ok is False
    assert a.rec["status"] == "interrupted"
    assert a.rec.get("pending") is None
    assert a.rec.get("ended_at") is not None
    assert a.rec.get("reason", "").startswith("stopped")


def test_retire_mid_turn_closes_activity_record_with_reason(runner):
    """Same as above, but for retirement (supervisor's `workers.pop(key).retired.set()`).
    Reason names retirement so the monitor page shows why."""
    w = _worker(runner)
    a = runner._Activity(w)
    a.rec = {"id": "rec-2", "mode": "explore", "status": "running", "objective": None,
             "started_at": 1.0, "ended_at": None, "pending": {"kind": "tool", "name": "library_save",
                                                              "since": 2.0, "args": {}},
             "asked": [], "timeline": [], "chats": [], "submissions": [],
             "tokens": {"prompt": 0, "completion": 0, "chats": 0}}
    w.__dict__["_activity"] = a
    w.retired.set()

    def gen(payload):
        raise AssertionError("_generate should not be called after retire")

    w._generate = gen
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    ok, _, _ = w.converse(messages, TOOLS, MagicMock(), tag={"objective_id": "o"}, max_rounds=3)
    assert ok is False
    assert a.rec["status"] == "interrupted"
    assert a.rec.get("pending") is None
    assert "retired" in a.rec.get("reason", "")


def test_end_reason_is_stored_on_the_record(runner):
    """_Activity.end(status, reason) stores the reason so the console can show it."""
    w = _worker(runner)
    a = runner._Activity(w)
    a.rec = {"id": "rec-3", "mode": "build", "status": "running", "objective": None,
             "started_at": 1.0, "ended_at": None, "pending": None, "asked": [], "timeline": [],
             "chats": [], "submissions": [], "tokens": {"prompt": 0, "completion": 0, "chats": 0}}
    a.end("stopped", "just because")
    assert a.rec["status"] == "stopped"
    assert a.rec["reason"] == "just because"
    assert a.rec["ended_at"] is not None
