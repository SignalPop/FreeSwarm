"""An iteration cut off by today's external spending limit is "interrupted", not "no submission".

Evidence (bug #3, nosubmit:qwen/qwen3.8-27b@groq, 28 sightings): both recent records --
1791270817354-6319 (10-06 00:20:30) and 1791237983622-2697 (10-05 15:08:53) -- end on the
console's own 429 "today's external-model spending limit for search is reached ($4.01 of
$4.00)". external_usage.sqlite3 puts the $4 search budget's exhaustion at 00:21 (09-28), 00:20
(09-29), 06:57 (09-30), 00:21 (10-01), 00:23 (10-02), 15:08 (10-05) and 00:20 (10-06): the
daily cluster of sightings just after midnight is the budget resetting at midnight, three Groq
agents spending it in ~20 minutes, and every iteration in flight being cut off. The runner
already backs off (hold_for_budget) and keeps the cut quiet on #errors, but closed the record
with no status, so it read "no submission" and the monitor counted it against the model.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app import monitor as M

RUNNER = Path(__file__).resolve().parents[1] / "swarm_runner.py"
MODEL = "qwen/qwen3.8-27b@groq"
LIMIT = ("/v1/chat/completions -> 429: today's external-model spending limit for search is reached "
         "($4.01 of $4.00); the remaining $1.00 of the $5.00 daily limit is held for ideas when stuck. "
         "Search resumes at midnight, or raise the limit / lower the reserve in Settings -> External models.")
SCRIPT = (
    "import ft, polars as pl\n\n"
    "def main():\n"
    "    rows = ft.rows_pl(columns=['t', 'Close'])\n"
    "    pos = (rows['Close'].diff().fill_null(0) > 0).cast(int)\n"
    "    ft.report_actions(pos, t=rows['t'])\n\n"
    "if __name__ == '__main__':\n"
    "    main()\n")
PROBE = "import ft\nrows = ft.rows_pl(columns=['t', 'Close'])\nprint(len(rows))\n"


@pytest.fixture
def runner(monkeypatch):
    spec = importlib.util.spec_from_file_location("swarm_runner_budget_cut_test", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_poster", lambda: type("P", (), {"put": lambda self, k, d: None})())
    monkeypatch.setattr(mod, "_mcp_tools", lambda pid: [])
    monkeypatch.setattr(mod, "AUTO_REPAIR", False)
    return mod


def _call(name: str, args: dict, i: int = 0) -> dict:
    return {"choices": [{"finish_reason": "tool_calls", "message": {"content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def _text(text: str) -> dict:
    return {"choices": [{"finish_reason": "stop", "message": {"content": text, "tool_calls": []}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class Harness:
    """Worker._turn on one explore iteration, replies scripted; the string LIMIT in the list
    is the console refusing the request for today's spending limit (as Worker._generate sees it)."""

    def __init__(self, runner, monkeypatch, replies):
        self.runner, self.replies, self.candidates = runner, list(replies), []
        monkeypatch.setattr(runner, "OBJECTIVE_TOOL_ROUNDS", 4)
        monkeypatch.setattr(runner, "MAX_EXPERIMENTS", 8)
        self.ctx = {"mode": "explore", "parent": None, "objective": {"title": "T", "metric": {"kind": "sharpe"}},
                    "metric_label": "Sharpe ratio", "library": [], "datasets": ["bars"], "playbook": {}}
        monkeypatch.setattr(runner, "request", self.request)
        w = runner.Worker.__new__(runner.Worker)
        w.model, w.agent_name, w.agent_id, w.role, w.slot = MODEL, f"{MODEL} #3", "a3", "search", 2
        w.project = {"id": "p1", "slug": "t", "name": "P"}
        w._stop, w.retired = threading.Event(), threading.Event()
        w._budget_block, w._budget_streak = None, 0
        w.say = MagicMock()
        w._sync = lambda: ([], [])
        w._inbox_since = 0.0
        w.beat = lambda *a, **k: None
        w.teammates = lambda: []
        w.mentor_coaching = lambda: None
        w.inbox = lambda: []
        w.record_collaboration = lambda *a, **k: None
        w.claim = lambda: None
        w.next_objective = lambda: {"id": "o1", "title": "T", "metric": {"kind": "sharpe"}}
        w.busy = None

        def generate(payload):
            act = runner._act(w)
            act.chat_start(payload)
            r = self.replies.pop(0) if self.replies else _text("(no more to say)")
            if r == LIMIT:
                act.chat_done(payload, None, LIMIT)
                w._budget_block = LIMIT          # as Worker._generate does for its own model
                raise RuntimeError(LIMIT)
            act.chat_done(payload, r)
            return r

        w._generate = generate
        self.w = w

    def request(self, base, path, payload=None, **kw):
        if "/context?" in path:
            return self.ctx
        if path.endswith("/python"):
            return {"ok": True, "stdout": "496482", "stderr": ""}
        if path.endswith("/candidates") and payload is not None:
            self.candidates.append(payload)
            return {"seq": 300 + len(self.candidates), "candidate_id": f"c{len(self.candidates)}",
                    "status": "ok", "score": 1.0}
        if path == "/mb/messages" and payload is not None:
            return {"seq": 1}
        return {"files": [], "docs": [], "modules": [], "candidates": [], "entries": []}

    def run(self):
        self.w._turn()
        return self

    @property
    def rec(self):
        return self.runner._act(self.w).rec


def test_iteration_cut_by_spending_limit_is_interrupted(runner, monkeypatch):
    # Record 6319 in shape: research calls, then the console refuses the next request.
    h = Harness(runner, monkeypatch, [_call("run_python", {"code": PROBE}, 0),
                                      _call("run_python", {"code": PROBE + "print(2)\n"}, 1),
                                      LIMIT]).run()
    assert h.rec["status"] == "interrupted"
    assert h.rec["reason"].startswith("spending limit: today's external-model spending limit")
    assert not h.candidates
    # No "Iteration ended without a submission" post on #errors either (unchanged behaviour).
    assert not [c for c in h.w.say.call_args_list if c.args[:2] == ("errors", "error")]
    # The monitor does not count it.
    a = {"agent": h.w.agent_name, "model": MODEL, "project_id": "p1", "updated_at": time.time(),
         "records": [h.rec]}
    assert not [f for f in M.record_findings(a, h.rec, time.time(), {}) if f["fingerprint"].startswith("nosubmit:")]


def test_iteration_that_submitted_before_the_limit_stays_submitted(runner, monkeypatch):
    # The reflection after a clean submission is refused: the iteration still delivered.
    h = Harness(runner, monkeypatch, [_call("submit_candidate", {"code": SCRIPT, "rationale": "r"}, 0),
                                      LIMIT]).run()
    assert len(h.candidates) == 1
    assert h.rec["status"] == "submitted"


def test_iteration_that_just_stops_is_still_no_submission(runner, monkeypatch):
    # No budget involved: the model talks instead of submitting, even in the forced turn.
    h = Harness(runner, monkeypatch, [_text("I think the signal is weak."), _text("Nothing to submit."),
                                      _text("Still nothing.")]).run()
    assert h.rec["status"] == "no submission"
    a = {"agent": h.w.agent_name, "model": MODEL, "project_id": "p1", "updated_at": time.time(),
         "records": [h.rec]}
    assert [f for f in M.record_findings(a, h.rec, time.time(), {}) if f["fingerprint"].startswith("nosubmit:")]


def _record(status: str, last_error: str | None) -> tuple[dict, dict]:
    now = time.time()
    tl = [{"kind": "chat", "at": now - 50, "model": MODEL, "error": None, "tool_calls": ["run_python"]},
          {"kind": "tool", "at": now - 40, "name": "run_python", "args": {"code": PROBE}, "ok": True,
           "result": "{\"ok\": true}"},
          {"kind": "chat", "at": now - 2, "model": MODEL, "error": last_error, "tool_calls": []}]
    r = {"id": "r1", "mode": "build", "status": status, "started_at": now - 100, "ended_at": now - 1,
         "pending": None, "timeline": tl, "submissions": []}
    return {"agent": f"{MODEL} #3", "model": MODEL, "project_id": "p1", "updated_at": now, "records": [r]}, r


def test_monitor_skips_no_submission_records_cut_by_the_limit():
    # Records written by a runner from before the fix still say "no submission".
    a, r = _record("no submission", LIMIT)
    assert not [f for f in M.record_findings(a, r, time.time(), {}) if f["fingerprint"].startswith("nosubmit:")]
    # Any other failed last request still counts.
    a, r = _record("no submission", "/v1/chat/completions -> 500: engine crashed")
    assert [f for f in M.record_findings(a, r, time.time(), {}) if f["fingerprint"] == f"nosubmit:{MODEL}"]
    a, r = _record("no submission", None)
    assert [f for f in M.record_findings(a, r, time.time(), {}) if f["fingerprint"] == f"nosubmit:{MODEL}"]
