"""Auto-repair routing (10-06): who writes a fix, and how much they write.

10-01..10-06, work.sqlite3: 256 auto-repairs failed, 233 (91%) without a reply at all -- the
crashing agent's OWN local model was asked to write the whole fixed script back: 156 refused up
front ("a fix from Qwen3.6-35B-A3B takes ~539s (4 tok/s measured), 360s left") and 77 timed out
after the full 360 s. Qwen3.6 repaired 12 of 190 crashes. A repair is now a few SEARCH/REPLACE
edits, asked of the fastest loaded model the project may use, falling back across models.
"""

from __future__ import annotations

import importlib.util
import threading
import time as _time
import traceback
import types
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "swarm_runner.py"
QWEN, MUSE, DEEPSEEK = "Qwen3.6-35B-A3B", "Muse-Glimmer-30B-NVFP4", "DeepSeek-V4-Flash-0731@lambda999"
GROQ, TINY = "qwen/qwen3.8-27b@groq", "Qwen/Qwen3-0.6B"
SPEEDS = {QWEN: 4.0, MUSE: 6.0, DEEPSEEK: 11.0, GROQ: 400.0, TINY: 200.0}

# The 10-06 case (project 6afb6a32, Qwen3.6-35B-A3B #2, seq 13), cut down: a hand-written trailing
# stop whose `price` is only set inside `if current_pos != 0:` -- NameError on the first entry.
EXAMPLE = """import numpy as np
close = np.array([10.0, 10.5, 11.0, 10.2, 9.8, 10.9, 11.5])
signal = np.array([0, 1, 1, 0, -1, -1, 0])
pos = np.zeros(len(close))
current_pos = 0
current_entry_price = None
trail = 0.05
for i in range(len(close)):
    if current_pos != 0:
        price = close[i]
        if current_pos > 0 and price < current_entry_price * (1 - trail):
            current_pos = 0
        elif current_pos < 0 and price > current_entry_price * (1 + trail):
            current_pos = 0
    if current_pos == 0 and signal[i] != 0:
        current_pos = int(signal[i])
        current_entry_price = price
    pos[i] = current_pos
print('positions', pos.tolist())
"""
EDIT = """The entry reads `price` before it is set. Fix:
<<<<<<< SEARCH
for i in range(len(close)):
    if current_pos != 0:
        price = close[i]
=======
for i in range(len(close)):
    price = close[i]
    if current_pos != 0:
>>>>>>> REPLACE
"""


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("swarm_runner_repair_routing_test", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._speed.update(SPEEDS)
    return mod


def _run(code: str) -> dict:
    """The sandbox, in-process: run_python's result shape."""
    import contextlib
    import io
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            exec(compile(code, "main.py", "exec"), {"__name__": "__main__"})
    except Exception:  # noqa: BLE001 -- the script's own error is the result
        return {"ok": False, "stdout": out.getvalue(), "stderr": traceback.format_exc()}
    return {"ok": True, "stdout": out.getvalue(), "stderr": ""}


def _worker(runner, replies, *, own=QWEN, loaded=(QWEN, MUSE, DEEPSEEK, TINY, GROQ), models=None):
    """A Worker of `own` with `loaded` models; `replies[model]` answers (a str, an exception, or a
    callable(the recorded payload) returning either) and every request is recorded."""
    w = runner.Worker.__new__(runner.Worker)
    w.model = w.agent_name = own
    w.project = {"id": "p1", "models": models}
    w._stop, w.retired = threading.Event(), threading.Event()
    w._sync = lambda: (list(loaded), [])
    sent = []

    def generate(payload):
        sent.append({**payload, "_timeout": w._gen_timeout})
        r = replies[payload["model"]]
        r = r(sent[-1]) if callable(r) else r
        if isinstance(r, Exception):
            raise r
        finish = "length" if r.startswith("[cut]") else "stop"
        return {"choices": [{"finish_reason": finish, "message": {"content": r.removeprefix("[cut]")}}]}

    w._generate = generate
    return w, sent


# ------------------------------------------------------------------------------------------
# Edits
# ------------------------------------------------------------------------------------------
def test_an_edit_reply_is_applied(runner):
    fixed, why = runner._reply_fix(EDIT, EXAMPLE)
    assert why == "" and "    price = close[i]\n    if current_pos != 0:\n        if current_pos > 0" in fixed
    assert _run(fixed)["ok"] is True and _run(EXAMPLE)["ok"] is False


def test_edits_match_despite_trailing_spaces_and_shifted_indentation(runner):
    code = "def f(x):\n    if x:   \n        y = 1\n    return y\n"
    # trailing spaces in the code, none in the edit
    assert runner._apply_edit(code, "    if x:\n        y = 1", "    y = 0\n    if x:\n        y = 1") == \
        "def f(x):\n    y = 0\n    if x:\n        y = 1\n    return y\n"
    # the model dropped the indentation: the replacement is shifted back in
    assert runner._apply_edit(code, "return y", "return y + 1") == code.replace("return y", "return y + 1")
    assert runner._apply_edit(code, "if x:\n    y = 1", "if x:\n    y = 2") == \
        "def f(x):\n    if x:\n        y = 2\n    return y\n"


def test_an_edit_that_is_missing_or_ambiguous_is_not_applied(runner):
    code = "a = 1\nb = 1\nprint(a + b)\n"
    assert runner._apply_edit(code, "c = 1", "c = 2") is None
    assert runner._apply_edit(code, " = 1", " = 2") is None                  # twice
    assert runner._apply_edit(code, "   \n", "x") is None                    # empty
    fixed, why = runner._reply_fix("<<<<<<< SEARCH\nc = 1\n=======\nc = 2\n>>>>>>> REPLACE", code)
    assert fixed is None and "not in the code exactly once" in why


def test_several_edits_fenced_or_not_and_a_full_script_still_work(runner):
    code = "x = 1\ny = 2\nprint(x, y, z)\n"
    reply = ("```\n<<<<<<< SEARCH\nx = 1\n=======\nx = 10\n>>>>>>> REPLACE\n```\n"
             "<<<<<<< SEARCH\nprint(x, y, z)\n=======\nprint(x, y)\n>>>>>>> REPLACE\n")
    assert runner._reply_fix(reply, code) == ("x = 10\ny = 2\nprint(x, y)\n", "")
    full = "```python\nx = 1\ny = 2\nprint(x, y)\n```"
    assert runner._reply_fix(full, code) == ("x = 1\ny = 2\nprint(x, y)", "")
    assert runner._reply_fix("I think z is undefined.", code)[0] is None


# ------------------------------------------------------------------------------------------
# Who is asked
# ------------------------------------------------------------------------------------------
def test_the_fastest_free_model_is_asked_first(runner):
    order, need = runner._repair_order(QWEN, [MUSE, DEEPSEEK, TINY], 1024, 360)
    assert order == [DEEPSEEK, MUSE, QWEN]                 # never the 0.6B (a smaller tier)
    assert need[QWEN] == pytest.approx(256)


def test_a_paid_peer_only_after_the_free_ones_or_when_none_fits(runner):
    assert runner._repair_order(QWEN, [MUSE, DEEPSEEK, GROQ], 1024, 360)[0] == [DEEPSEEK, MUSE, QWEN, GROQ]
    assert runner._repair_order(QWEN, [MUSE, DEEPSEEK, GROQ], 1024, 60)[0] == [GROQ]
    # the agent's own paid model is no extra spend: it competes on speed
    assert runner._repair_order(GROQ, [MUSE, DEEPSEEK], 1024, 360)[0] == [GROQ, DEEPSEEK, MUSE]


def test_a_paid_model_the_project_did_not_tick_is_never_asked(runner):
    w, sent = _worker(runner, {DEEPSEEK: EDIT}, models=None)
    assert GROQ not in w._repair_peers() and TINY in w._repair_peers()
    w, _ = _worker(runner, {}, models=[QWEN, DEEPSEEK, GROQ])
    assert w._repair_peers() == [DEEPSEEK, GROQ]


def test_a_model_that_timed_out_on_a_repair_is_not_asked_again_while_it_would_not_finish(runner):
    runner._note_reply_failed("repair", DEEPSEEK, 300.0)    # -> 450 s expected
    assert runner._repair_order(QWEN, [MUSE, DEEPSEEK], 1024, 360)[0] == [MUSE, QWEN]


# ------------------------------------------------------------------------------------------
# The worker's repair: routing, fallback, the example end to end
# ------------------------------------------------------------------------------------------
def test_the_example_is_repaired_by_the_fastest_model_end_to_end(runner):
    w, sent = _worker(runner, {DEEPSEEK: EDIT})
    crashed = _run(EXAMPLE)
    assert "NameError: name 'price' is not defined" in crashed["stderr"]
    out = runner._auto_repair("submit_candidate", EXAMPLE, crashed, w._repair_code, _run, who="t")
    assert out["ok"] is True and out["stdout"].count("positions") == 1
    assert out["auto_repaired"]["attempts"] == 1 and out["auto_repaired"]["models"] == [DEEPSEEK]
    assert [p["model"] for p in sent] == [DEEPSEEK]
    # a small reply budget, a wait that leaves time for the next model
    assert sent[0]["max_tokens"] <= runner.AUTO_REPAIR_REPLY_TOKENS + runner.REASONING_ROOM
    assert sent[0]["_timeout"] < runner.AUTO_REPAIR_MAX_S
    assert w._gen_timeout is None and runner._TL.repair_deadline is None and not runner._TL.repair_models


def test_a_timed_out_model_hands_the_repair_to_the_next_one(runner, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(runner, "time", types.SimpleNamespace(time=lambda: clock[0], strftime=_time.strftime))

    def times_out(payload):
        clock[0] += payload["_timeout"]                     # waited the whole cap
        return RuntimeError("/v1/chat/completions failed: TimeoutError: timed out")

    w, sent = _worker(runner, {DEEPSEEK: times_out, MUSE: EDIT})
    tokens = min(len(EXAMPLE) // 3 + 512, runner.AUTO_REPAIR_EXPECT_TOKENS)
    first = int(2 * tokens / SPEEDS[DEEPSEEK])                   # 2x its expected time (~65 s)
    then = int(min(runner.AUTO_REPAIR_MAX_S - first, max(runner.AUTO_REPAIR_CALL_MIN_S, 2 * tokens / SPEEDS[MUSE])))
    out = runner._auto_repair("submit_candidate", EXAMPLE, _run(EXAMPLE), w._repair_code, _run, who="t")
    assert out["ok"] is True and out["auto_repaired"]["models"] == [MUSE]
    assert [p["model"] for p in sent] == [DEEPSEEK, MUSE]
    # DeepSeek waited 2x its expected time, not the whole 360 s; Muse the time left (at most 2x
    # its own, as Qwen is still to come after it)
    assert sent[0]["_timeout"] == first < runner.AUTO_REPAIR_MAX_S / 2 and sent[1]["_timeout"] == then
    assert runner._reply_s[f"repair|{DEEPSEEK}"] == pytest.approx(1.5 * first, abs=2)


def test_a_refusing_model_sits_out_and_the_next_one_repairs(runner):
    w, sent = _worker(runner, {DEEPSEEK: RuntimeError("/v1/chat/completions -> 503: engine unavailable"),
                               MUSE: EDIT})
    assert _run(w._repair_code("run_python", EXAMPLE, runner._script_crash(_run(EXAMPLE))))["ok"] is True
    assert [p["model"] for p in sent] == [DEEPSEEK, MUSE]
    assert runner._repair_order(QWEN, [MUSE, DEEPSEEK], 1024, 360)[0] == [MUSE, QWEN]


def test_a_cut_off_reply_moves_on_and_only_failures_raise(runner):
    w, sent = _worker(runner, {DEEPSEEK: "[cut]<<<<<<< SEARCH\nfor i in", MUSE: RuntimeError("boom"),
                               QWEN: RuntimeError("boom too")})
    with pytest.raises(RuntimeError, match="cut off.*boom.*boom too"):
        w._repair_code("run_python", EXAMPLE, runner._script_crash(_run(EXAMPLE)))
    assert [p["model"] for p in sent] == [DEEPSEEK, MUSE, QWEN]


def test_replies_with_no_usable_fix_count_as_an_attempt_and_say_why(runner):
    w, sent = _worker(runner, {m: "The problem is that price is undefined." for m in (DEEPSEEK, MUSE, QWEN)})
    out = runner._auto_repair("run_python", EXAMPLE, _run(EXAMPLE), w._repair_code, _run, who="t")
    assert out["ok"] is False and "no corrected code came back (" in out["auto_repair_failed"]
    assert "no edit and no complete code block" in out["auto_repair_failed"]
    assert len(sent) == 3 * runner.AUTO_REPAIR_ATTEMPTS


def test_when_no_model_can_answer_in_time_none_is_asked(runner, monkeypatch):
    monkeypatch.setattr(runner, "AUTO_REPAIR_MAX_S", 50.0)        # DeepSeek needs ~65 s
    w, sent = _worker(runner, {}, loaded=(QWEN, MUSE, DEEPSEEK))
    out = runner._auto_repair("run_python", EXAMPLE, _run(EXAMPLE), w._repair_code, _run, who="t")
    assert sent == [] and "no loaded model can write a fix" in out["auto_repair_failed"]
    assert f"{DEEPSEEK} ~6" in out["auto_repair_failed"] and f"{QWEN} ~" in out["auto_repair_failed"]
