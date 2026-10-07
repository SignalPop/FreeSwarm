"""Auto-repair budgets: a repair reply gets the runner's normal room to think, a reply cut off
before its code ends the repairs, and one tool call spends at most AUTO_REPAIR_MAX_S on them.

10-01 17:13, Muse-Glimmer: repair replies sized to the script (len(code)//3 + 512, + thinking
room) ran out at ~5.4k tokens mid-thought, were asked again, and were cut off again -- about 13
minutes of a ~45-minute iteration, for nothing.
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "swarm_runner.py"
MODEL = "Muse-Glimmer-30B-NVFP4"
BROKEN = "import ft\nrows = ft.rows_pl()\nx = rows['Close'].to_numpy()\nprint('mean', x.mean(), zz)\n"
FIXED = BROKEN.replace(", zz)", ")")
CRASHED = {"ok": False, "stderr": "Traceback (most recent call last):\n  File \"x.py\", line 4\n"
                                  "NameError: name 'zz' is not defined"}


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("swarm_runner_repair_budget_test", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reply(finish, content="", reasoning=""):
    return {"choices": [{"finish_reason": finish, "message": {"content": content, "reasoning_content": reasoning}}]}


def _worker(runner, replies):
    w = runner.Worker.__new__(runner.Worker)
    w.model = w.agent_name = MODEL
    w._stop, w.retired = threading.Event(), threading.Event()
    sent = []

    def generate(payload):
        sent.append({**payload, "_timeout": getattr(w, "_gen_timeout", None)})
        return replies.pop(0)

    w._generate = generate
    return w, sent


@pytest.fixture
def big_window(runner):
    runner._context[MODEL] = 131_072
    yield
    runner._context.pop(MODEL, None)


def test_a_repair_reply_gets_room_for_edits_and_thinking(runner, big_window):
    # 10-06: the reply is the changed lines (SEARCH/REPLACE), not the whole script again, so it
    # is no longer given the runner's full MAX_TOKENS -- the answer's room plus thinking room.
    w, sent = _worker(runner, [_reply("stop", f"```python\n{FIXED}\n```", reasoning="thinking " * 500)])
    crash = runner._script_crash(CRASHED)
    assert w._repair_code("run_python", BROKEN, crash) == FIXED.strip("\n")
    assert [p["max_tokens"] for p in sent] == [runner.AUTO_REPAIR_REPLY_TOKENS + runner.REASONING_ROOM]
    assert sent[0]["max_tokens"] < runner.MAX_TOKENS


def test_the_budget_still_fits_a_small_window(runner):
    runner._context[MODEL] = 8192
    try:
        w, sent = _worker(runner, [_reply("stop", f"```python\n{FIXED}\n```")])
        w._repair_code("run_python", BROKEN, runner._script_crash(CRASHED))
        assert runner.MIN_OUTPUT <= sent[0]["max_tokens"] < 8192
    finally:
        runner._context.pop(MODEL, None)


def test_a_cut_off_repair_reply_ends_the_repairs(runner, big_window):
    # Two attempts are allowed; the first reply runs out while thinking -- no second request.
    w, sent = _worker(runner, [_reply("length", "", reasoning="let me think " * 3000),
                               _reply("stop", f"```python\n{FIXED}\n```")])
    reruns = []
    out = runner._auto_repair("run_python", BROKEN, dict(CRASHED), w._repair_code,
                              lambda code: reruns.append(code) or {"ok": True, "stdout": "fine"}, who="t")
    assert len(sent) == 1 and reruns == []
    assert out["ok"] is False and "cut off" in out["auto_repair_failed"]


def test_a_cut_off_reply_that_still_holds_a_complete_block_is_used(runner, big_window):
    w, sent = _worker(runner, [_reply("length", f"```python\n{FIXED}\n```\nAnd one more thing ab")])
    out = runner._auto_repair("run_python", BROKEN, dict(CRASHED), w._repair_code,
                              lambda code: {"ok": True, "stdout": "fine"}, who="t")
    assert out["auto_repaired"]["attempts"] == 1 and out["code_ran"] == FIXED.strip("\n")


def test_repairs_stop_at_the_wall_time_limit(runner, monkeypatch):
    monkeypatch.setattr(runner, "AUTO_REPAIR_MAX_S", 0.0)
    asked = []
    out = runner._auto_repair("run_python", BROKEN, dict(CRASHED), lambda *a: asked.append(a) or FIXED,
                              lambda code: {"ok": True}, who="t")
    assert asked == [] and "time limit" in out["auto_repair_failed"]


def test_the_repair_request_waits_no_longer_than_the_time_left(runner, monkeypatch, big_window):
    monkeypatch.setattr(runner, "AUTO_REPAIR_MAX_S", 120.0)
    w, sent = _worker(runner, [_reply("stop", f"```python\n{FIXED}\n```")])
    runner._auto_repair("run_python", BROKEN, dict(CRASHED), w._repair_code, lambda code: {"ok": True}, who="t")
    assert 60 <= sent[0]["_timeout"] <= 120
    assert w._gen_timeout is None and runner._TL.repair_deadline is None    # nothing leaks to later calls
    # Outside a repair the normal generation timeout applies.
    w2, sent2 = _worker(runner, [_reply("stop", "ok")])
    w2._chat("hello")
    assert sent2[0]["_timeout"] is None


def test_no_repair_starts_with_under_30_seconds_left(runner, monkeypatch, big_window):
    monkeypatch.setattr(runner, "AUTO_REPAIR_MAX_S", 10.0)
    w, sent = _worker(runner, [_reply("stop", f"```python\n{FIXED}\n```")])
    out = runner._auto_repair("run_python", BROKEN, dict(CRASHED), w._repair_code, lambda code: {"ok": True}, who="t")
    assert sent == [] and "no repair time left" in out["auto_repair_failed"]
