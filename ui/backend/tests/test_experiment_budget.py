"""Bug #11 -- "Agents run out of run_python experiments".

Evidence from ui/backend/agent_activity.sqlite3 for the two recent (non-Groq) sightings:

  * Qwen3.6-35B-A3B (2026-09-29 19:49): 6 run_python calls before refusal, 4 succeeded
    and 2 were the agent's own Python bugs. After the refusal the agent DID call
    submit_candidate, so the budget's purpose held -- but it burned every experiment
    with no wrap-up warning first, and the refusal did not tell it what to submit.

  * Muse-Glimmer-30B-NVFP4 (2026-09-29 19:36): 5 successful runs and 1 agent bug, then
    the wall. Same story.

  * Older qwen/qwen3.8-27b@groq cases were mostly SyntaxErrors from a truncated-code
    bug (fixed elsewhere) -- not relevant to the fix here.

Across all 93 failed run_python results in the DB, zero looked like platform failures
(harness / timeout); every single one carried an agent-side traceback. Refunding the
budget on a transport error is still a principled guard against future ones.

The fix:

  1. Bump the per-iteration budget from 6 to 8 (configurable via env, as it already was).
  2. Warn the agent with a ``note`` on the last two runs so it saves + submits before
     the wall.
  3. Track the last successful (ok=True) code as ``best_code`` and hand it back in the
     refusal, so submit_candidate has something concrete to send.
  4. If ``request`` to the sandbox raises (transport / harness error), refund the run --
     the agent never got to prove or disprove anything.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "swarm_runner.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("swarm_runner_budget_test", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def runner():
    return _load_runner()


def _world(runner, monkeypatch, sandbox_results):
    """An ObjectiveWorld wired to a scripted sandbox.

    ``sandbox_results`` is either a list of dicts consumed one per run_python call, or a
    callable that receives the code and returns a dict (or raises RuntimeError).
    """
    calls = []
    results = list(sandbox_results) if isinstance(sandbox_results, list) else None

    def request(base, path, payload=None, **kw):
        calls.append((path, payload))
        if path.startswith("/api/projects/") and "/data/catalog" in path:
            return {"files": []}
        if path.startswith("/api/research/docs?"):
            return {"docs": []}
        if path.endswith("/python"):
            if callable(sandbox_results):
                return sandbox_results((payload or {}).get("code", ""))
            return results.pop(0)
        return {}

    monkeypatch.setattr(runner, "request", request)
    monkeypatch.setattr(runner, "_mcp_tools", lambda pid: [])
    world = runner.ObjectiveWorld(
        {"id": "p1", "name": "P"}, "me", [], [],
        {"id": "o1", "metric": {"kind": "sharpe"}}, on_submit=lambda *a, **kw: None)
    return world, calls


# ------------------------------------------------------------------------------------------
# What the runner asks for
# ------------------------------------------------------------------------------------------
def test_default_budget_is_eight(runner):
    """The evidence in bug #11 showed 6 was too tight for the productive models.

    Both recent refusals came after 5-6 mostly-successful runs. Eight gives one or two
    experiments of head-room without going back to the "Qwen burns 10+ and never
    submits" world that led to a cap in the first place.
    """
    assert runner.MAX_EXPERIMENTS == 8


def test_budget_is_env_configurable(monkeypatch):
    """FREESWARM_SWARM_EXPERIMENTS still overrides the default."""
    monkeypatch.setenv("FREESWARM_SWARM_EXPERIMENTS", "3")
    mod = _load_runner()
    assert mod.MAX_EXPERIMENTS == 3


# ------------------------------------------------------------------------------------------
# The budget itself
# ------------------------------------------------------------------------------------------
def _ok(code_marker: str = "ran") -> dict:
    return {"ok": True, "stdout": code_marker, "stderr": "", "duration_s": 0.1, "artifacts": [], "data": {}}


def test_experiments_left_counts_down(runner, monkeypatch):
    """The tool reply carries experiments_left so the agent can pace itself."""
    world, _ = _world(runner, monkeypatch, [_ok() for _ in range(runner.MAX_EXPERIMENTS)])
    left = [world.call("run_python", {"code": f"print({i})"})["experiments_left"]
            for i in range(runner.MAX_EXPERIMENTS)]
    assert left == list(range(runner.MAX_EXPERIMENTS - 1, -1, -1))


def test_penultimate_and_last_runs_carry_a_wrap_up_note(runner, monkeypatch):
    """Agents burned through the whole budget in silence -- now they get a heads-up.

    The second-to-last successful run says "wrap up", the last one says "submit now".
    """
    world, _ = _world(runner, monkeypatch, [_ok() for _ in range(runner.MAX_EXPERIMENTS)])
    replies = [world.call("run_python", {"code": f"print({i})"})
               for i in range(runner.MAX_EXPERIMENTS)]
    # No note in the middle of the budget.
    for r in replies[:-2]:
        assert "note" not in r, r
    # Two-left warning
    penult = replies[-2]
    assert "note" in penult and "one run_python left" in penult["note"].lower()
    # Last-run warning
    last = replies[-1]
    assert "note" in last and "last run_python" in last["note"].lower()
    assert "submit_candidate" in last["note"]


def test_refusal_after_budget(runner, monkeypatch):
    """One over is refused with instructions to save + submit."""
    world, _ = _world(runner, monkeypatch, [_ok() for _ in range(runner.MAX_EXPERIMENTS)])
    for i in range(runner.MAX_EXPERIMENTS):
        world.call("run_python", {"code": f"print({i})"})
    refusal = world.call("run_python", {"code": "print('one more')"})
    assert "error" in refusal
    assert "experiment budget used" in refusal["error"]
    assert "submit_candidate" in refusal["error"]


def test_refusal_includes_best_working_code(runner, monkeypatch):
    """Bug #11 fix: the refusal hands the agent the last code that actually ran, so
    submit_candidate has something concrete to send instead of the agent trying to
    reconstruct it from memory (which is when they stopped submitting)."""
    fills = []
    # The last successful run's code is what should come back.
    for i in range(runner.MAX_EXPERIMENTS - 1):
        fills.append({"ok": False, "stderr": "NameError: x", "stdout": "", "duration_s": 0.1,
                      "artifacts": [], "data": {}})
    good_code = "import ft\nrows = ft.rows_pl()\nprint(rows.shape)"
    fills.append(_ok())  # only the FINAL one succeeded

    world, _ = _world(runner, monkeypatch, fills)
    for i in range(runner.MAX_EXPERIMENTS - 1):
        world.call("run_python", {"code": f"broken {i}"})
    world.call("run_python", {"code": good_code})

    refusal = world.call("run_python", {"code": "print('one more')"})
    assert refusal.get("best_working_code") == good_code
    assert "hint" in refusal and "submit_candidate" in refusal["hint"]


def test_refusal_without_any_ok_run_has_no_best_code(runner, monkeypatch):
    """If nothing ever ran cleanly there's no code to hand back -- the refusal still
    tells the agent to save + submit, but does not fabricate best_working_code."""
    fills = [{"ok": False, "stderr": "SyntaxError", "stdout": "", "duration_s": 0.1,
              "artifacts": [], "data": {}} for _ in range(runner.MAX_EXPERIMENTS)]
    world, _ = _world(runner, monkeypatch, fills)
    for _ in range(runner.MAX_EXPERIMENTS):
        world.call("run_python", {"code": "def foo(:"})
    refusal = world.call("run_python", {"code": "x"})
    assert "error" in refusal
    assert "best_working_code" not in refusal
    assert "hint" not in refusal


def test_platform_failure_does_not_burn_budget(runner, monkeypatch):
    """When the control-plane call itself fails (network, harness) the agent never got
    a chance to prove anything -- refunding the run keeps the budget honest.

    Evidence-driven: no platform failures actually showed up in agent_activity, but the
    93 agent-side failures in that DB were all clean Python tracebacks with ok=False,
    so this guard fires only on the transport case."""
    call_num = [0]

    def sandbox(code):
        call_num[0] += 1
        if call_num[0] == 3:
            raise RuntimeError("cannot reach control plane")
        return _ok()

    world, _ = _world(runner, monkeypatch, sandbox)
    for i in range(2):
        world.call("run_python", {"code": f"print({i})"})
    assert world.experiments == 2

    with pytest.raises(RuntimeError):
        world.call("run_python", {"code": "print('boom')"})
    # Refunded -- still at 2, not 3.
    assert world.experiments == 2

    # And we can still run the full remaining budget.
    remaining = runner.MAX_EXPERIMENTS - 2
    for i in range(remaining):
        r = world.call("run_python", {"code": f"print({i})"})
        assert "error" not in r
    # The one over is now refused.
    refused = world.call("run_python", {"code": "one more"})
    assert "experiment budget used" in refused.get("error", "")


def test_only_ok_runs_seed_best_code(runner, monkeypatch):
    """A failed run must not overwrite the last known-good code."""
    good = "import ft; ft.rows_pl()"
    seq = [
        _ok(),  # good
        {"ok": False, "stderr": "AttributeError", "stdout": "", "duration_s": 0.1,
         "artifacts": [], "data": {}},  # bad -- must NOT overwrite good
    ]
    world, _ = _world(runner, monkeypatch, seq)
    world.call("run_python", {"code": good})
    assert world.best_code == good
    world.call("run_python", {"code": "broken"})
    assert world.best_code == good  # unchanged
