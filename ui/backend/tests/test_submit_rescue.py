"""An explore/build iteration must not end "no submission" when the agent has a script.

Evidence (work.sqlite3, 2026-09-30..10-01): 22 explore/build iterations ended no_submission.
Every one with a reply had used all OBJECTIVE_TOOL_ROUNDS and was then asked, in a final round
that offered NO tools, to "call submit_candidate NOW":
  A. Muse-Glimmer #2 (Gex2, 10-01 21:36): answered with its script in a ```python block -- no
     tool call -- and the iteration ended without it being submitted.
  B. Muse-Glimmer (build, same minute): wrote a run_python call as harmony/XML text; the salvage
     RAN it (the last experiment) and the iteration ended without a submission.
Now the final round offers submit_candidate (named tool_choice), a script written as text is
submitted as what it is, a call to any other tool in that round is not run, the experiment
budget running out makes the next turn a submit turn, and a turn that still ends without a
submission gets one forced submit turn. Each salvage / forced turn is on the activity timeline.
"""

from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "swarm_runner.py"
MUSE = "Muse-Glimmer-30B-NVFP4"

SCRIPT = (
    "import ft, polars as pl\n\n"
    "def main():\n"
    "    rows = ft.rows_pl(columns=['t', 'Close'])\n"
    "    pos = (rows['Close'].diff().fill_null(0) > 0).cast(int)\n"
    "    ft.report_actions(pos, t=rows['t'])\n\n"
    "if __name__ == '__main__':\n"
    "    main()\n")
PROBE = "import ft\nrows = ft.rows_pl(columns=['t', 'Close'])\nprint(ft.quick_score(rows['Close'] * 0, rows))\n"

# B's reply, verbatim in shape (10-01 21:36, Muse-Glimmer build iteration).
HARMONY_RUN = (
    '<|start|>assistant to=run_python<|message|><atem:function_calls>\n'
    '<atem:invoke name="run_python">\n'
    '<atem:parameter name="code">\n'
    "import ft\nfrom lib import symmetric_contraction_breakout_v2\n"
    "rows = ft.rows_pl(columns=['t','Close','High','Low','Volume'])\n"
    "entries = symmetric_contraction_breakout_v2.signal(rows)\n"
    "import numpy as np\nprint(np.sum(entries==1), np.sum(entries==-1))\n"
    '</atem:parameter>\n</atem:invoke>\n</atem:function_calls>')


@pytest.fixture
def runner(monkeypatch):
    spec = importlib.util.spec_from_file_location("swarm_runner_submit_rescue_test", RUNNER)
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
    """Worker.iterate with the model's replies scripted and the control plane stubbed."""

    def __init__(self, runner, monkeypatch, replies, *, mode="explore", rounds=2, experiments=8,
                 refuse_tool_choice=False):
        self.runner, self.replies, self.payloads = runner, list(replies), []
        self.python_runs, self.candidates, self.saves = [], [], []
        monkeypatch.setattr(runner, "OBJECTIVE_TOOL_ROUNDS", rounds)
        monkeypatch.setattr(runner, "MAX_EXPERIMENTS", experiments)
        self.ctx = {"mode": mode, "parent": None, "objective": {"title": "T", "metric": {"kind": "sharpe"}},
                    "metric_label": "Sharpe ratio", "library": [], "datasets": ["bars"], "playbook": {}}
        monkeypatch.setattr(runner, "request", self.request)
        w = runner.Worker.__new__(runner.Worker)
        w.model, w.agent_name, w.agent_id, w.role, w.slot = MUSE, f"{MUSE} #2", "a2", "search", 1
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
        self.refuse_tool_choice = refuse_tool_choice

        def generate(payload):
            # As Worker._generate: the activity record sees each request and reply.
            self.payloads.append(json.loads(json.dumps(payload)))
            act = runner._act(w)
            act.chat_start(payload)
            if self.refuse_tool_choice and payload.get("tool_choice"):
                act.chat_done(payload, None, "400")
                raise RuntimeError("/v1/chat/completions -> 400: {\"detail\": \"tool_choice not supported\"}")
            if not payload.get("tools") and "Write ONE lesson" in str(payload["messages"][-1].get("content")):
                r = _text("KEEP: test lesson.")
            else:
                r = self.replies.pop(0)
            act.chat_done(payload, r)
            return r

        w._generate = generate
        self.w = w

    def request(self, base, path, payload=None, **kw):
        if "/context?" in path:
            return self.ctx
        if path.endswith("/python"):
            self.python_runs.append(payload["code"])
            return {"ok": True, "stdout": "quick_score {'sharpe': 1.0}", "stderr": ""}
        if path.endswith("/candidates") and payload is not None:
            self.candidates.append(payload)
            n = len(self.candidates)
            return {"seq": 200 + n, "candidate_id": f"c{n}", "status": "ok", "score": 1.0}
        if path == "/api/projects/p1/library" and payload is not None:
            self.saves.append(payload)
            return {"saved": True, "name": payload.get("name"), "version": 1, "test_ok": True}
        if path == "/mb/messages" and payload is not None:
            return {"seq": 1}
        return {"files": [], "docs": [], "modules": [], "candidates": [], "entries": []}

    def run(self):
        self.w.iterate({"id": "o1", "title": "T", "metric": {"kind": "sharpe"}})
        return self

    @property
    def timeline(self):
        return self.runner._act(self.w).rec["timeline"]

    def tool_names(self, payload):
        return [t["function"]["name"] for t in payload.get("tools") or []]


# ---------------------------------------------------------------------------------------
# The helpers
# ---------------------------------------------------------------------------------------
def test_text_script_takes_only_a_complete_reporting_script(runner):
    reply = f"I need to submit now. Plan: momentum.\n```python\n{SCRIPT}```\nDone."
    assert runner._text_script(reply) == SCRIPT.strip("\n")
    assert runner._text_rationale(reply) == "I need to submit now. Plan: momentum. Done."
    assert runner._text_script(f"```python\n{PROBE}```") is None                  # a probe: prints only
    assert runner._text_script(f"```python\n{SCRIPT[:-30]}") is None              # block never closed
    assert runner._text_script("```python\ndef main(:\n    ft.report_actions(x)\n```") is None  # no compile
    assert runner._text_rationale(f"```python\n{SCRIPT}```").startswith("script written as text")


def test_harmony_xml_run_python_is_recovered_with_its_code(runner):
    got = runner._text_tool_calls(HARMONY_RUN, {"run_python", "submit_candidate"})
    assert [n for n, _ in got] == ["run_python"]
    code = got[0][1]["code"]
    assert code.startswith("import ft\nfrom lib import symmetric_contraction_breakout_v2")
    assert code.endswith("print(np.sum(entries==1), np.sum(entries==-1))")
    compile(code, "<b>", "exec")


# ---------------------------------------------------------------------------------------
# A: the script written as text in the final round is submitted
# ---------------------------------------------------------------------------------------
def test_case_a_script_written_as_text_in_the_final_round_is_submitted(runner, monkeypatch):
    h = Harness(runner, monkeypatch, [
        _call("run_python", {"code": PROBE}, 0),
        _call("run_python", {"code": PROBE + "print(2)\n"}, 1),
        _text(f"```python\n{SCRIPT}```"),                       # the final round: text, no tool call
    ]).run()
    assert [c["code"] for c in h.candidates] == [SCRIPT.strip("\n")]
    assert h.candidates[0]["rationale"].startswith("script written as text")
    # The final round offered submit_candidate itself, by name, with a default script.
    final = h.payloads[2]
    assert h.tool_names(final) == ["submit_candidate"]
    assert final["tool_choice"] == {"type": "function", "function": {"name": "submit_candidate"}}
    assert "your last run_python script that ran" in final["messages"][-1]["content"]
    sub = [e for e in h.timeline if e.get("kind") == "tool" and e["name"] == "submit_candidate"]
    assert len(sub) == 1 and sub[0]["salvaged"] == "script written as text" and sub[0]["ok"]
    assert len(h.python_runs) == 2


# ---------------------------------------------------------------------------------------
# B: a run_python written as text in the final round is not run; a forced turn submits
# ---------------------------------------------------------------------------------------
def test_case_b_harmony_run_python_in_final_round_is_not_run_then_forced_submit(runner, monkeypatch):
    h = Harness(runner, monkeypatch, [
        _call("library_save", {"name": "symmetric_contraction_breakout_v2", "kind": "signal",
                               "description": "d", "code": "def signal(df):\n    return df\n",
                               "test": "print(1)"}, 0),
        _call("run_python", {"code": PROBE}, 1),
        _text(HARMONY_RUN),                                     # final round: harmony run_python
        _call("submit_candidate", {"code": SCRIPT, "rationale": "breakout"}, 3),   # forced turn
    ], mode="build").run()
    assert len(h.python_runs) == 1                             # the text-written check was NOT run
    assert len(h.saves) == 1 and [c["code"] for c in h.candidates] == [SCRIPT]
    forced = h.payloads[3]
    assert h.tool_names(forced) == ["submit_candidate"]
    assert forced["tool_choice"]["function"]["name"] == "submit_candidate"
    prompt = forced["messages"][-1]["content"]
    assert "without calling submit_candidate" in prompt and "symmetric_contraction_breakout_v2" in prompt
    assert PROBE.strip() in prompt                              # the last script that ran, as default
    steps = [e for e in h.timeline if e.get("kind") == "tool" and e["name"] == "forced_submit"]
    assert len(steps) == 1 and steps[0]["ok"]
    assert any(e.get("kind") == "message" and "without calling submit_candidate" in e["text"]
               for e in h.timeline)


def test_forced_turn_salvages_a_script_written_as_text(runner, monkeypatch):
    h = Harness(runner, monkeypatch, [
        _call("run_python", {"code": PROBE}, 0),
        _text("I think the idea is weak; stopping here."),       # final round: prose
        _text(f"Here it is:\n```python\n{SCRIPT}```"),          # forced turn: still text
    ], rounds=1).run()
    assert [c["code"] for c in h.candidates] == [SCRIPT.strip("\n")]
    assert any(e.get("name") == "forced_submit" for e in h.timeline)
    assert any(e.get("name") == "submit_candidate" and e.get("salvaged") == "script written as text"
               for e in h.timeline)


def test_no_endless_forcing_when_the_model_will_not_submit(runner, monkeypatch):
    h = Harness(runner, monkeypatch, [
        _call("run_python", {"code": PROBE}, 0),
        _text("nothing worth submitting"),
        _text("still nothing"),
        _text("no"),
    ], rounds=1).run()
    assert h.candidates == [] and h.replies == []               # final + forced turn (2 replies), then stop
    h.w.say.assert_any_call("errors", "error", "Iteration ended without a submission: no", {"objective_id": "o1"})


# ---------------------------------------------------------------------------------------
# The experiment budget spent: the next turn is a submit turn
# ---------------------------------------------------------------------------------------
def test_experiment_budget_spent_forces_the_next_turn_to_submit(runner, monkeypatch):
    h = Harness(runner, monkeypatch, [
        _call("run_python", {"code": PROBE}, 0),
        _call("submit_candidate", {"code": SCRIPT, "rationale": "r"}, 1),
    ], rounds=6, experiments=1).run()
    nxt = h.payloads[1]
    assert h.tool_names(nxt) == ["submit_candidate"]
    assert nxt["tool_choice"]["function"]["name"] == "submit_candidate"
    assert "All 1 run_python experiments" in nxt["messages"][-1]["content"]
    assert PROBE.strip() in nxt["messages"][-1]["content"]
    assert [c["code"] for c in h.candidates] == [SCRIPT]
    assert any(e.get("kind") == "message" and "All 1 run_python experiments" in e["text"] for e in h.timeline)


def test_budget_spent_call_to_another_tool_is_refused_without_running(runner, monkeypatch):
    h = Harness(runner, monkeypatch, [
        _call("run_python", {"code": PROBE}, 0),
        _call("team_board", {}, 1),                               # ignores the submit turn
        _call("submit_candidate", {"code": SCRIPT, "rationale": "r"}, 2),
    ], rounds=6, experiments=1).run()
    assert "team_board is not available now" in h.payloads[2]["messages"][-1]["content"]
    assert not any(e.get("name") == "team_board" for e in h.timeline)
    assert [c["code"] for c in h.candidates] == [SCRIPT]


def test_build_without_a_saved_module_may_still_save_in_the_submit_turn(runner, monkeypatch):
    h = Harness(runner, monkeypatch, [
        _call("run_python", {"code": PROBE}, 0),
        _call("submit_candidate", {"code": SCRIPT, "rationale": "r"}, 1),
    ], mode="build", rounds=6, experiments=1)
    h.replies.append(_call("library_save", {"name": "m", "kind": "signal", "description": "d",
                                            "code": "def signal(df):\n    return df\n", "test": "print(1)"}, 2))
    h.run()
    assert h.tool_names(h.payloads[1]) == ["library_save", "submit_candidate"]
    assert "tool_choice" not in h.payloads[1]
    assert [c["code"] for c in h.candidates] == [SCRIPT]


def test_server_refusing_tool_choice_is_resent_without_it(runner, monkeypatch):
    h = Harness(runner, monkeypatch, [
        _call("run_python", {"code": PROBE}, 0),
        _call("submit_candidate", {"code": SCRIPT, "rationale": "r"}, 1),
    ], rounds=6, experiments=1, refuse_tool_choice=True).run()
    assert h.payloads[1].get("tool_choice") and "tool_choice" not in h.payloads[2]
    assert h.tool_names(h.payloads[2]) == ["submit_candidate"]
    assert [c["code"] for c in h.candidates] == [SCRIPT]
    assert MUSE in runner._NO_TOOL_CHOICE


# ---------------------------------------------------------------------------------------
# Other callers of converse keep the old final round
# ---------------------------------------------------------------------------------------
def test_plain_converse_final_round_unchanged(runner, monkeypatch):
    h = Harness(runner, monkeypatch, [])
    w, sent = h.w, []
    replies = [_call("q", {"x": 1}), _text("the answer")]

    def gen(payload):
        sent.append(payload)
        return replies.pop(0)

    w._generate = gen
    tools = [{"type": "function", "function": {"name": "q", "description": "", "parameters": {
        "type": "object", "properties": {"x": {"type": "integer"}}}}}]
    ok, text, _ = w.converse([{"role": "user", "content": "hi"}], tools, lambda n, a: {"r": 1},
                             tag={"task_id": "t"}, max_rounds=1)
    assert ok and text == "the answer"
    assert "tools" not in sent[1] and "tool_choice" not in sent[1]
