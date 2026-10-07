"""Auto-repair: a script that crashed goes back to the model that wrote it, inside the same
tool call, and a fix that runs is what the agent (and the Work/Bugs pages) see.

The recorded shape the pages rely on: the successful result's FIRST key is
``"auto_repaired": {"attempts": n, "errors": [...], "original_code_lines": k}`` and the tool
call is recorded as ok.
"""

from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "swarm_runner.py"

BROKEN = "\n".join([
    "import ft",
    "rows = ft.rows_pl()",
    "x = rows['Close'].to_numpy()",
    "m = x.mean()",
    "s = x.std()",
    "z = (x - m) / s",
    "sig = z > 1",
    "print('n', len(x))",
    "print('hits', sig.sum())",
    "print('mean', m, 'std', s, 'zz', zz)",
])
FIXED = BROKEN.replace(", 'zz', zz)", ")")


def _load_runner():
    spec = importlib.util.spec_from_file_location("swarm_runner_auto_repair_test", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def runner():
    return _load_runner()


def _crash(msg: str = "NameError: name 'zz' is not defined", hint: str | None = None) -> dict:
    tb = ("Traceback (most recent call last):\n  File \"/work/main.py\", line 10, in <module>\n"
          "    print('mean', m, 'std', s, 'zz', zz)\n" + msg + "\n")
    return {"ok": False, "stdout": "n 100\n", "stderr": tb, "artifacts": [], "duration_s": 0.2,
            **({"hint": hint} if hint else {})}


def _ok(text: str = "n 100\nhits 7\n") -> dict:
    return {"ok": True, "stdout": text, "stderr": "", "artifacts": [], "duration_s": 0.2, "data": "full"}


def _world(runner, monkeypatch, sandbox, library=None):
    """An ObjectiveWorld whose sandbox answers with `sandbox(code)` and whose library POST
    answers with `library(body)`."""
    calls = []

    def request(base, path, payload=None, **kw):
        calls.append((path, payload))
        if "/data/catalog" in path:
            return {"files": []}
        if path.startswith("/api/research/docs?"):
            return {"docs": []}
        if path.endswith("/python"):
            return sandbox((payload or {}).get("code", ""))
        if path.endswith("/library") and payload is not None and library is not None:
            return library(payload)
        return {}

    monkeypatch.setattr(runner, "request", request)
    monkeypatch.setattr(runner, "_mcp_tools", lambda pid: [])
    world = runner.ObjectiveWorld({"id": "p1", "name": "P"}, "me", [], [],
                                  {"id": "o1", "metric": {"kind": "sharpe"}}, on_submit=lambda a: None)
    return world, calls


class FakeModel:
    """The worker's repair call: hands back scripted replies and remembers what it was asked."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.asked = []

    def __call__(self, tool, code, crash, extra=""):
        self.asked.append({"tool": tool, "code": code, "crash": crash, "extra": extra})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _python_calls(calls):
    return [p for path, p in calls if path.endswith("/python")]


# ------------------------------------------------------------------------------------------
# run_python
# ------------------------------------------------------------------------------------------
def test_a_crashed_experiment_is_repaired_and_recorded(runner, monkeypatch):
    world, calls = _world(runner, monkeypatch, lambda code: _ok() if code == FIXED else _crash(hint="zz is never set"))
    world.repairer = model = FakeModel(FIXED)
    out = world.call("run_python", {"code": BROKEN})

    assert list(out)[0] == "auto_repaired"          # the head of the recorded JSON says so
    assert out["auto_repaired"] == {"attempts": 1, "errors": ["NameError: name 'zz' is not defined"],
                                    "original_code_lines": 10}
    assert out["ok"] is True and "error" not in out
    assert out["code_ran"] == FIXED
    assert out["stdout"].startswith("[Your script failed with NameError: name 'zz' is not defined; it was "
                                    "repaired automatically")
    assert "n 100\nhits 7" in out["stdout"]
    # One experiment for the call and its repair.
    assert out["experiments_left"] == runner.MAX_EXPERIMENTS - 1
    assert world.experiments == 1
    assert world.best_code == FIXED
    assert [p["code"] for p in _python_calls(calls)] == [BROKEN, FIXED]
    # The model got the code, the error line, the traceback and the hint.
    asked = model.asked[0]
    assert asked["tool"] == "run_python" and asked["code"] == BROKEN
    assert asked["crash"]["error"] == "NameError: name 'zz' is not defined"
    assert "Traceback" in asked["crash"]["traceback"] and asked["crash"]["hint"] == "zz is never set"


def test_the_second_attempt_starts_from_the_first_fix(runner, monkeypatch):
    half = BROKEN.replace("x.std()", "x.stdd()")   # still wrong in a new way
    half = half.replace(", 'zz', zz)", ")")

    def sandbox(code):
        if code == FIXED:
            return _ok()
        if code == half:
            return _crash("AttributeError: 'numpy.ndarray' object has no attribute 'stdd'")
        return _crash()

    world, calls = _world(runner, monkeypatch, sandbox)
    world.repairer = model = FakeModel(half, FIXED)
    out = world.call("run_python", {"code": BROKEN})
    assert out["auto_repaired"]["attempts"] == 2
    assert out["auto_repaired"]["errors"] == ["NameError: name 'zz' is not defined",
                                              "AttributeError: 'numpy.ndarray' object has no attribute 'stdd'"]
    assert model.asked[1]["code"] == half
    assert world.experiments == 1 and out["experiments_left"] == runner.MAX_EXPERIMENTS - 1


def test_two_failed_repairs_return_the_original_failure(runner, monkeypatch):
    other = BROKEN.replace("zz", "yy")
    third = BROKEN.replace("zz", "ww")
    world, calls = _world(runner, monkeypatch, lambda code: _crash())
    world.repairer = FakeModel(other, third)
    out = world.call("run_python", {"code": BROKEN})
    assert out["ok"] is False and list(out)[0] == "ok"
    assert "auto_repaired" not in out
    assert "NameError" in out["auto_repair_failed"]
    assert out["stderr"] == _crash()["stderr"]          # the original failure, unchanged
    assert len(_python_calls(calls)) == 3               # never more than 2 repairs
    assert world.experiments == 1
    assert world.best_code is None


def test_a_fix_that_deletes_most_of_the_script_is_not_run(runner, monkeypatch):
    world, calls = _world(runner, monkeypatch, lambda code: _ok() if code != BROKEN else _crash())
    world.repairer = FakeModel("import ft\nprint('ok')", "import ft\nprint(1)\nprint(2)")
    out = world.call("run_python", {"code": BROKEN})
    assert out["ok"] is False and "dropped most of the code" in out["auto_repair_failed"]
    assert len(_python_calls(calls)) == 1


def test_no_code_or_the_same_code_or_a_syntax_error_is_not_run(runner, monkeypatch):
    world, calls = _world(runner, monkeypatch, lambda code: _crash())
    world.repairer = FakeModel(None, BROKEN + "\n")
    out = world.call("run_python", {"code": BROKEN})
    assert out["ok"] is False and "same code" in out["auto_repair_failed"]
    world.repairer = FakeModel(BROKEN.replace("print('n', len(x))", "print('n', len(x)"), None)
    out = world.call("run_python", {"code": BROKEN})
    assert out["ok"] is False and "auto_repair_failed" in out
    assert len(_python_calls(calls)) == 2               # only the two originals ran


def test_a_failed_repair_request_returns_the_original(runner, monkeypatch):
    world, calls = _world(runner, monkeypatch, lambda code: _crash())
    world.repairer = FakeModel(RuntimeError("/v1/chat/completions -> 429: spending limit"))
    out = world.call("run_python", {"code": BROKEN})
    assert out["ok"] is False and "repair request failed" in out["auto_repair_failed"]
    assert len(_python_calls(calls)) == 1


@pytest.mark.parametrize("result", [
    {"ok": False, "stdout": "", "stderr": "\n\n[killed: exceeded the 180s limit]", "timed_out": True},
    {"ok": False, "stdout": "", "stderr": "Traceback (most recent call last):\n  ...\nMemoryError\n\n"
                                          "[killed: exceeded the 8g memory limit]"},
    {"ok": False, "stdout": "", "stderr": "exit code 2"},          # no traceback
    {"ok": True, "stdout": "fine", "stderr": ""},
])
def test_only_a_python_traceback_is_repaired(runner, monkeypatch, result):
    world, calls = _world(runner, monkeypatch, lambda code: result)
    world.repairer = model = FakeModel(FIXED)
    out = world.call("run_python", {"code": BROKEN})
    assert "auto_repaired" not in out and "auto_repair_failed" not in out
    assert model.asked == []


def test_budget_refusal_is_never_repaired(runner, monkeypatch):
    world, calls = _world(runner, monkeypatch, lambda code: _ok())
    world.repairer = model = FakeModel()
    world.experiments = runner.MAX_EXPERIMENTS
    out = world.call("run_python", {"code": BROKEN})
    assert "experiment budget used" in out["error"] and model.asked == []


def test_the_switch_turns_it_off(runner, monkeypatch):
    monkeypatch.setattr(runner, "AUTO_REPAIR", False)
    world, calls = _world(runner, monkeypatch, lambda code: _crash())
    world.repairer = model = FakeModel(FIXED)
    out = world.call("run_python", {"code": BROKEN})
    assert out["ok"] is False and model.asked == [] and "auto_repair_failed" not in out


def test_the_switch_reads_the_environment(monkeypatch):
    monkeypatch.setenv("FREESWARM_AUTO_REPAIR", "0")
    assert _load_runner().AUTO_REPAIR is False
    monkeypatch.delenv("FREESWARM_AUTO_REPAIR")
    assert _load_runner().AUTO_REPAIR is True


def test_a_world_without_a_repairer_leaves_failures_alone(runner, monkeypatch):
    world, calls = _world(runner, monkeypatch, lambda code: _crash())
    out = world.call("run_python", {"code": BROKEN})
    assert out["ok"] is False and "auto_repair_failed" not in out


def test_a_script_too_long_to_send_is_not_repaired(runner, monkeypatch):
    big = BROKEN + "\n" + "\n".join(f"# {'x' * 70}" for _ in range(400))
    world, calls = _world(runner, monkeypatch, lambda code: _crash())
    world.repairer = model = FakeModel(FIXED)
    world.call("run_python", {"code": big})
    assert model.asked == []


# ------------------------------------------------------------------------------------------
# library_save and submit_candidate
# ------------------------------------------------------------------------------------------
MODULE = "\n".join(["import polars as pl", "", "def signal(df):", "    x = df['Close']",
                    "    return (x - x.mean()) / x.stdd()"])
MODULE_FIXED = MODULE.replace("stdd", "std")
TEST = "from lib import zmod\nimport ft\nprint(zmod.signal(ft.load_pl('bars')).tail())"


def test_a_library_module_whose_smoke_test_crashed_is_repaired(runner, monkeypatch):
    def library(body):
        if body["code"] == MODULE_FIXED:
            return {"saved": True, "name": body["name"], "version": 1, "test_ok": True, "test_output": "ok"}
        return {"saved": False, "test_ok": False,
                "test_output": "Traceback (most recent call last):\n  File \"lib/zmod.py\", line 5, in signal\n"
                               "AttributeError: 'Series' object has no attribute 'stdd'",
                "error": "the smoke test failed -- fix the module and save again"}

    world, calls = _world(runner, monkeypatch, lambda code: _ok(), library)
    world.repairer = model = FakeModel(MODULE_FIXED)
    out = world.call("library_save", {"name": "zmod", "kind": "signal", "code": MODULE, "test_code": TEST,
                                      "description": "z-score"})
    assert list(out)[0] == "auto_repaired" and out["saved"] is True
    assert out["auto_repaired"]["errors"] == ["AttributeError: 'Series' object has no attribute 'stdd'"]
    assert out["code_ran"] == MODULE_FIXED and "repaired automatically" in out["note"]
    assert world.saved == ["zmod"]
    posts = [p for path, p in calls if path.endswith("/library") and p]
    assert [p["code"] for p in posts] == [MODULE, MODULE_FIXED]
    assert all(p["test_code"] == TEST for p in posts)          # the test stays as it is
    assert TEST in model.asked[0]["extra"]                      # and the model is shown it


def test_a_look_ahead_refusal_from_library_save_is_not_repaired(runner, monkeypatch):
    refusal = {"saved": False, "test_ok": False, "test_output": "Traceback (most recent call last):\nX\nValueError: y",
               "error": "signal changes when future rows are removed", "causality": {"verdict": "fail"}}
    world, calls = _world(runner, monkeypatch, lambda code: _ok(), lambda body: refusal)
    world.repairer = model = FakeModel(MODULE_FIXED)
    out = world.call("library_save", {"name": "zmod", "kind": "signal", "code": MODULE, "test_code": TEST})
    assert out["saved"] is False and model.asked == []


def test_a_candidate_that_crashed_is_repaired(runner):
    failed = {"candidate_id": "c1", "seq": 7, "status": "error", "eval_seconds": 2.0,
              "error": "KeyError: 'Clse' -- the column is 'Close'",
              "stderr_tail": "Traceback (most recent call last):\n  File \"/work/main.py\", line 3\nKeyError: 'Clse'",
              "stdout_tail": ""}
    scored = {"candidate_id": "c2", "seq": 8, "status": "ok", "in_sample_score": 1.2, "lookahead": "pass",
              "rank": 3}
    seen = []

    def rerun(code):
        seen.append(code)
        return scored

    model = FakeModel(FIXED)
    out = runner._auto_repair("submit_candidate", BROKEN, failed, model, rerun, who="me")
    assert list(out)[0] == "auto_repaired" and out["status"] == "ok" and out["seq"] == 8
    assert out["auto_repaired"]["errors"] == ["KeyError: 'Clse'"]
    assert "repaired automatically" in out["note"] and out["code_ran"] == FIXED
    assert "the column is 'Close'" in model.asked[0]["crash"]["hint"]
    assert seen == [FIXED]
    # What the activity record keeps of a submission keeps the repair too.
    assert runner._result_brief("submit_candidate", out)["auto_repaired"]["attempts"] == 1


def test_a_candidate_whose_fix_fails_another_way_keeps_the_original_error(runner):
    failed = {"seq": 7, "status": "error", "error": "boom",
              "stderr_tail": "Traceback (most recent call last):\nZeroDivisionError: division by zero"}
    other = {"seq": 8, "status": "error", "error": "no score reported -- call ft.report_score(x)", "stderr_tail": ""}
    out = runner._auto_repair("submit_candidate", BROKEN, failed, FakeModel(FIXED), lambda c: other)
    assert out["seq"] == 7 and out["error"] == "boom" and "no score reported" in out["auto_repair_failed"]


# ------------------------------------------------------------------------------------------
# The worker's side: the model call and the activity record
# ------------------------------------------------------------------------------------------
def _repair_worker(runner, model, reply):
    """A Worker with no peers loaded whose chat request answers with `reply(payload)`."""
    import threading
    w = runner.Worker.__new__(runner.Worker)
    w.model = w.agent_name = model
    w._stop, w.retired = threading.Event(), threading.Event()
    asked = []
    w._generate = lambda payload: asked.append(payload) or {
        "choices": [{"finish_reason": "stop", "message": {"content": reply(payload)}}]}
    return w, asked


def test_the_worker_asks_its_own_model_and_reads_the_code_block(runner):
    reply = "Here you go:\n```python\nprint(1)\n```\nand the full one:\n```python\n" + FIXED + "\n```"
    worker, asked = _repair_worker(runner, "me", lambda p: reply)
    crash = runner._script_crash(_crash(hint="define zz"))
    got = runner.Worker._repair_code(worker, "run_python", BROKEN, crash)
    assert got == FIXED                                  # the longest block
    assert asked[0]["model"] == "me"                     # nothing else loaded
    p = asked[0]["messages"][0]["content"]
    assert BROKEN in p and "NameError: name 'zz' is not defined" in p and "define zz" in p
    assert "<<<<<<< SEARCH" in p and "do NOT send the whole script back" in p
    assert runner.MIN_OUTPUT <= asked[0]["max_tokens"] <= runner.MAX_TOKENS


def test_the_worker_applies_an_edit_reply(runner):
    edit = "<<<<<<< SEARCH\nprint('mean', m, 'std', s, 'zz', zz)\n=======\nprint('mean', m, 'std', s)\n>>>>>>> REPLACE"
    worker, asked = _repair_worker(runner, "me", lambda p: edit)
    assert runner.Worker._repair_code(worker, "run_python", BROKEN, runner._script_crash(_crash())) == FIXED


def test_a_cut_off_reply_gives_no_code(runner):
    assert runner._fenced_code("```python\nimport ft\nrows = ft.rows_pl(") is None
    assert runner._fenced_code("I think the problem is the name zz.") is None
    assert runner._fenced_code("import ft\nprint(ft)") == "import ft\nprint(ft)"


def test_the_tool_event_carries_the_repair(runner, monkeypatch):
    act = runner._Activity(types.SimpleNamespace(agent_name="me"))
    act.rec = {"timeline": [], "submissions": [], "pending": None}
    monkeypatch.setattr(act, "_push", lambda: None)
    out = {"auto_repaired": {"attempts": 1, "errors": ["NameError: x"], "original_code_lines": 10},
           "ok": True, "stdout": "x" * 20_000, "code_ran": FIXED}
    act.tool_done("run_python", {"code": BROKEN}, out, True)
    e = act.rec["timeline"][-1]
    assert e["ok"] is True and e["auto_repaired"]["attempts"] == 1
    # The head of the (cut) result is the repair marker, never '{"ok": false'.
    assert e["result"].startswith('{"auto_repaired": {"attempts": 1')
    assert json.loads(json.dumps(e))                     # plain JSON for the poster


# ------------------------------------------------------------------------------------------
# Fixes that need no model (10-01 19:43: Qwen's "NameError: name 'pl_col' is not defined",
# whose repair request to the slow model timed out)
# ------------------------------------------------------------------------------------------
PL_BROKEN = "\n".join([
    "import polars as pl",
    "import ft",
    "rows = ft.rows_pl()",
    "close_prices = rows['Close']",
    "out = rows.select(pl_col('Close').mean())",
    "print(out, len(close_prices))",
])
PL_FIXED = PL_BROKEN.replace("pl_col(", "pl.col(")


def test_a_module_attribute_written_with_an_underscore_is_fixed_without_the_model(runner, monkeypatch):
    world, calls = _world(runner, monkeypatch,
                          lambda code: _ok() if code == PL_FIXED else _crash("NameError: name 'pl_col' is not defined"))
    world.repairer = model = FakeModel()                 # asking it would fail the test (no replies)
    out = world.call("run_python", {"code": PL_BROKEN})
    assert out["ok"] is True and out["code_ran"] == PL_FIXED and model.asked == []
    assert out["auto_repaired"] == {"attempts": 1, "errors": ["NameError: name 'pl_col' is not defined"],
                                    "original_code_lines": 6, "mechanical": True,
                                    "mechanical_fixes": ["pl_col -> pl.col"], "model_attempts": 0}
    assert [p["code"] for p in _python_calls(calls)] == [PL_BROKEN, PL_FIXED]


@pytest.mark.parametrize("code,error,fixed,how", [
    ("import numpy as np\nclose_prices = [1, 2]\nprint(np.mean(close_price))",
     "close_price", "import numpy as np\nclose_prices = [1, 2]\nprint(np.mean(close_prices))",
     "close_price -> close_prices"),
    ("import numpy as np\nx = [1.0]\nprint(np_nan_to_num(x))", "np_nan_to_num",
     "import numpy as np\nx = [1.0]\nprint(np.nan_to_num(x))", "np_nan_to_num -> np.nan_to_num"),
    ("import ft\nimport numpy as np\nrows = ft_load_pl('bars')", "ft_load_pl",
     "import ft\nimport numpy as np\nrows = ft.load_pl('bars')", "ft_load_pl -> ft.load_pl"),
    ("def signal(df):\n    return df\nprint(signall(1))", "signall", "def signal(df):\n    return df\nprint(signal(1))",
     "signall -> signal"),
])
def test_mechanical_fixes(runner, code, error, fixed, how):
    assert runner._mechanical_fix(code, {"error": f"NameError: name '{error}' is not defined"}) == (fixed, how)


@pytest.mark.parametrize("code,error", [
    ("val1 = 1\nval2 = 2\nprint(val)", "val"),                    # two close names: ambiguous
    ("x = 1\nprint(zz)", "zz"),                                   # nothing close
    ("print(pl_col('a'))", "pl_col"),                             # pl is not imported
    ("import polars as pl\nprint(pl_nosuchthing('a'))", "pl_nosuchthing"),
    ("import polars as pl\nprint('pl_col')", "pl_col"),           # only in a string
])
def test_no_mechanical_fix_without_one_obvious_answer(runner, code, error):
    assert runner._mechanical_fix(code, {"error": f"NameError: name '{error}' is not defined"}) is None
    assert runner._mechanical_fix(code, {"error": "AttributeError: 'list' has no attribute 'x'"}) is None


def test_a_mechanical_fix_that_uncovers_another_error_goes_on_to_the_model(runner, monkeypatch):
    model_fixed = PL_FIXED.replace("len(close_prices)", "close_prices.len()")

    def sandbox(code):
        if code == PL_FIXED:
            return _crash("TypeError: object of type 'Series' has no len()")
        return _ok() if code == model_fixed else _crash("NameError: name 'pl_col' is not defined")
    world, _ = _world(runner, monkeypatch, sandbox)
    world.repairer = model = FakeModel(model_fixed)
    out = world.call("run_python", {"code": PL_BROKEN})
    assert out["code_ran"] == model_fixed and model.asked[0]["code"] == PL_FIXED
    info = out["auto_repaired"]
    assert info["attempts"] == 2 and info["mechanical"] is False and info["model_attempts"] == 1
    assert info["mechanical_fixes"] == ["pl_col -> pl.col"]


# ------------------------------------------------------------------------------------------
# Waits sized to the model's measured speed
# ------------------------------------------------------------------------------------------
def test_the_speed_of_each_model_is_measured_from_its_replies(runner, monkeypatch):
    w = runner.Worker.__new__(runner.Worker)
    w.model, w.agent_name, w.role, w.slot, w.project = "slow", "slow", "search", 0, {"id": "p1", "slug": "t"}
    w._budget_streak = 0
    monkeypatch.setattr(runner, "_poster", lambda: types.SimpleNamespace(put=lambda k, d: None))
    clock = iter([100.0, 110.0])                         # the request took 10 s
    monkeypatch.setattr(runner, "time", types.SimpleNamespace(time=lambda: next(clock, 300.0)))
    monkeypatch.setattr(runner, "request", lambda *a, **k: {"choices": [{"message": {"content": "x"}}],
                                                            "usage": {"completion_tokens": 330}})
    monkeypatch.setattr(runner._Activity, "chat_start", lambda self, p: None)
    monkeypatch.setattr(runner._Activity, "chat_done", lambda self, p, r, e=None: None)
    w._generate({"model": "slow", "messages": []})
    assert runner._speed["slow"] == pytest.approx(33.0)
    runner._note_speed("slow", 30, 1.0)                  # too short a reply to say anything
    assert runner._speed["slow"] == pytest.approx(33.0)
    runner._note_speed("slow", 1000, 10.0)
    assert runner._speed["slow"] == pytest.approx(0.7 * 33 + 0.3 * 100)


def test_a_slow_model_is_not_asked_for_a_repair_it_cannot_finish_in_time(runner):
    import time
    worker, asked = _repair_worker(runner, "slow", lambda p: f"```python\n{FIXED}\n```")
    crash = runner._script_crash(_crash())
    code = BROKEN + "\n# " + "x" * 3000
    # An edit reply is expected to cost AUTO_REPAIR_EXPECT_TOKENS (1024), not the whole script.
    runner._speed["slow"] = 5.0                          # -> ~205 s for the fix
    runner._TL.repair_deadline = time.time() + 100
    try:
        with pytest.raises(RuntimeError, match="no repair time left: a fix from slow takes ~2"):
            runner.Worker._repair_code(worker, "run_python", code, crash)
        assert asked == []
        runner._speed["slow"] = 100.0                    # ~10 s: asked
        assert runner.Worker._repair_code(worker, "run_python", code, crash) == FIXED and len(asked) == 1
        # What repairs actually took counts too.
        runner._reply_s["repair|slow"] = 150.0
        with pytest.raises(RuntimeError, match="no repair time left"):
            runner.Worker._repair_code(worker, "run_python", code, crash)
    finally:
        runner._TL.repair_deadline = None


def test_the_feedback_time_limit_follows_the_models_speed(runner):
    assert runner._feedback_budget("unmeasured", 4) == (4, runner.FEEDBACK_MAX_S)
    runner._speed["fast"] = 200.0
    assert runner._feedback_budget("fast", 4) == (4, runner.FEEDBACK_MAX_S)
    runner._speed["slowish"] = 10.0                      # (4*250 + 4096) / 10 * 1.5 = 764 s
    n, cap = runner._feedback_budget("slowish", 4)
    assert n == 4 and cap == pytest.approx(764.4)
    runner._speed["crawl"] = 5.0                         # even one message needs > the hard cap
    assert runner._feedback_budget("crawl", 4) == (1, runner.FEEDBACK_HARD_MAX_S)
    assert runner.FEEDBACK_HARD_MAX_S == 900


def test_mechanical_fix_uses_pythons_own_suggestion(runner):
    """10-02 00:39: roll_70 with roll_q30 AND roll_q70 defined -- difflib sees two, Python said 'roll_q70'."""
    code = "roll_q30 = 1\nroll_q70 = 2\nx = roll_70 + 1\nprint(x)\n"
    crash = {"error": "NameError: name 'roll_70' is not defined. Did you mean: 'roll_q70'?", "traceback": ""}
    fixed, what = runner._mechanical_fix(code, crash)
    assert "roll_q70 + 1" in fixed and what == "roll_70 -> roll_q70"
