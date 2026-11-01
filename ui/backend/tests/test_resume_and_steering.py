"""Iterations that survive a restart, messages that reach an agent mid-iteration, and the live
event stream (10-19).

Runner restarts were 257 of the work log's 361 interrupted iterations (~100 agent-hours): each
one threw away the research behind it. Now a search iteration is checkpointed at the top of
every tool round and a new runner carries it on. Everything the team learned also reached an
agent only through its next iteration's prompt; steering messages arrive between two rounds.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app import agent_activity as A
from app import events
from app import objectives as O
from app import steering as S
from app import work as W

RUNNER = Path(__file__).resolve().parents[1] / "swarm_runner.py"

TOOLS = [{"type": "function", "function": {
    "name": "run_python", "description": "run",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "submit_candidate", "description": "submit",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}}]


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("swarm_runner_under_test_resume", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _quiet_activity(monkeypatch, runner):
    class _NoopPoster:
        def put(self, key, doc):
            pass

    monkeypatch.setattr(runner, "_poster", lambda: _NoopPoster())


def _worker(runner, model="Muse", name=None):
    w = runner.Worker.__new__(runner.Worker)
    w.model = model
    w.agent_name = name or model
    w.agent_id = "a1"
    w.role = "search"
    w.slot = 0
    w.project = {"id": "p1", "slug": "test"}
    w._stop = threading.Event()
    w.retired = threading.Event()
    w._budget_streak = 0
    w._paused = False
    w.say = MagicMock()
    return w


def _scripted(responses, seen=None):
    def gen(payload):
        if seen is not None:
            seen.append(json.loads(json.dumps(payload)))
        return responses.pop(0)
    return gen


def _call(name: str, args: dict) -> dict:
    return {"choices": [{"finish_reason": "tool_calls", "message": {"content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def _stop(text: str = "done") -> dict:
    return {"choices": [{"finish_reason": "stop", "message": {"content": text, "tool_calls": []}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}}


def _msgs() -> list[dict]:
    return [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}]


# ---------------------------------------------------------------------------------------
# converse(): checkpoint, resume, pause on drain
# ---------------------------------------------------------------------------------------
def test_each_round_is_checkpointed_and_a_resume_carries_on_from_it(runner):
    w = _worker(runner)
    w._generate = _scripted([_call("run_python", {"code": "print(1)"}), _stop("all done")])
    saved, ran = [], []
    msgs = _msgs()
    ok, text, _ = w.converse(msgs, TOOLS, lambda n, a: ran.append(n) or {"ok": True}, tag={},
                             checkpoint=lambda st: saved.append((json.loads(json.dumps(st)), len(msgs))))
    assert ok and text == "all done"
    assert [s["round"] for s, _ in saved] == [0, 1]
    assert saved[1][0]["usage"]["tool_calls"] == 1
    # A new runner: the conversation as of round 2's top, carried on from there.
    state, n = saved[1]
    w2 = _worker(runner)
    w2._generate = _scripted([_stop("again")])
    ok, text, usage = w2.converse(msgs[:n], TOOLS, lambda n, a: {"ok": True}, tag={}, resume=state)
    assert ok and text == "again"
    assert usage["tool_calls"] == 1 and usage["rounds"] == 2      # the first round's work still counts


def test_a_drain_pauses_a_checkpointed_turn_at_the_next_round(runner, monkeypatch):
    w = _worker(runner)
    w._generate = MagicMock(side_effect=AssertionError("no model call after a drain"))
    monkeypatch.setattr(runner, "drain_requested", lambda: True)
    runner._act(w).begin("explore", {"id": "o1", "title": "T"})
    saved = []
    ok, text, _ = w.converse(_msgs(), TOOLS, lambda n, a: {}, tag={}, checkpoint=saved.append)
    assert (ok, text) == (False, "") and w._paused
    assert len(saved) == 1                                        # saved first, then paused
    rec = runner._act(w).rec
    assert rec["status"] == "interrupted" and "resumes from its checkpoint" in rec["reason"]


def test_without_a_checkpoint_a_drain_lets_the_turn_finish(runner, monkeypatch):
    w = _worker(runner)
    w._generate = _scripted([_stop("finished")])
    monkeypatch.setattr(runner, "drain_requested", lambda: True)
    assert w.converse(_msgs(), TOOLS, lambda n, a: {}, tag={})[:2] == (True, "finished")


def test_the_call_in_flight_is_marked_before_it_runs(runner):
    w = _worker(runner)
    w._generate = _scripted([_call("submit_candidate", {"code": "x = 1"}), _stop()])
    order = []
    w.converse(_msgs(), TOOLS, lambda n, a: order.append(("ran", n)) or {"ok": True}, tag={},
               pending=lambda n, a: order.append(("marked", n, a["code"])))
    assert order == [("marked", "submit_candidate", "x = 1"), ("ran", "submit_candidate")]


# ---------------------------------------------------------------------------------------
# converse(): steering
# ---------------------------------------------------------------------------------------
def test_advice_is_the_next_thing_the_model_reads(runner):
    w = _worker(runner)
    seen = []
    w._generate = _scripted([_stop()], seen)
    steer = lambda rnd: [{"kind": "steer", "text": "build on #12", "source": "operator"}] if rnd == 0 else []
    w.converse(_msgs(), TOOLS, lambda n, a: {}, tag={}, steer=steer)
    last = seen[0]["messages"][-1]
    assert last["role"] == "user" and "the operator" in last["content"] and "build on #12" in last["content"]


def test_wrap_up_makes_this_round_the_last(runner):
    w = _worker(runner)
    seen = []
    w._generate = _scripted([_stop("summary")], seen)
    steer = lambda rnd: [{"kind": "wrap_up", "text": "", "source": "operator"}]
    ok, text, _ = w.converse(_msgs(), TOOLS, lambda n, a: {}, tag={}, max_rounds=8, steer=steer,
                             final_prompt="LAST ROUND")
    assert seen[0]["messages"][-1]["content"] == "LAST ROUND"
    assert "tools" not in seen[0]                                  # a final round offers no tools
    assert (ok, text) == (True, "summary")


def test_abort_ends_the_turn_before_the_model_is_asked(runner):
    w = _worker(runner)
    w._generate = MagicMock(side_effect=AssertionError("aborted turns ask nothing"))
    steer = lambda rnd: [{"kind": "abort", "text": "wrong objective", "source": "operator"}]
    ok, text, _ = w.converse(_msgs(), TOOLS, lambda n, a: {}, tag={}, steer=steer)
    assert not ok and text.startswith("aborted by the operator") and "wrong objective" in text


def test_a_steering_failure_never_stops_the_turn(runner):
    w = _worker(runner)
    w._generate = _scripted([_stop("fine")])

    def broken(rnd):
        raise RuntimeError("console down")

    assert w.converse(_msgs(), TOOLS, lambda n, a: {}, tag={}, steer=broken)[:2] == (True, "fine")


# ---------------------------------------------------------------------------------------
# The checkpoint store and picking a checkpoint up
# ---------------------------------------------------------------------------------------
def test_checkpoints_round_trip(runner):
    ck = runner._ckpt()
    ck.save("p1|Muse", {"saved_at": 1.0, "messages": [{"role": "user", "content": "go"}], "pending": None})
    ck.mark("p1|Muse", {"name": "submit_candidate", "at": 2.0})
    doc = ck.load("p1|Muse")
    assert doc["messages"][0]["content"] == "go" and doc["pending"]["name"] == "submit_candidate"
    ck.drop("p1|Muse")
    assert ck.load("p1|Muse") is None


def _ck(runner, **over) -> dict:
    return {"version": runner.CHECKPOINT_VERSION, "saved_at": time.time(), "started_at": time.time() - 600,
            "record_id": "r-old", "resumes": 0, "objective_id": "o1", "ctx": {"mode": "explore"},
            "messages": _msgs(), "converse": {"round": 3}, "submits": [], "parents": [], **over}


def test_a_fresh_checkpoint_on_a_running_objective_is_resumed(runner, monkeypatch):
    w = _worker(runner)
    runner._ckpt().save("p1|Muse", _ck(runner))
    monkeypatch.setattr(runner, "request", lambda base, path, *a, **k: {"objectives": [{"id": "o1", "title": "T"}]})
    ck = w._resumable()
    assert ck["objective"]["title"] == "T" and ck["converse"]["round"] == 3
    assert runner._ckpt().load("p1|Muse")["resumes"] == 1             # counted before the attempt


@pytest.mark.parametrize("over, objectives", [
    ({"saved_at": time.time() - 99_999}, [{"id": "o1"}]),           # too old
    ({"resumes": 3}, [{"id": "o1"}]),                                # resumed too often
    ({}, [{"id": "other"}]),                                         # objective no longer running
])
def test_a_checkpoint_that_cannot_be_carried_on_is_dropped(runner, monkeypatch, over, objectives):
    w = _worker(runner)
    runner._ckpt().save("p1|Muse", _ck(runner, **over))
    monkeypatch.setattr(runner, "request", lambda base, path, *a, **k: {"objectives": objectives})
    assert w._resumable() is None
    assert runner._ckpt().load("p1|Muse") is None


def test_a_console_that_cannot_answer_keeps_the_checkpoint_for_later(runner, monkeypatch):
    w = _worker(runner)
    runner._ckpt().save("p1|Muse", _ck(runner))

    def down(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(runner, "request", down)
    assert w._resumable() is None and runner._ckpt().load("p1|Muse") is not None


def test_a_submission_evaluated_across_the_restart_is_found_and_counted(runner, monkeypatch):
    w = _worker(runner)
    w.report_eval = MagicMock()
    sent = "def main():\n    ft.report_positions(x)\n"

    def fake(base, path, *a, **k):
        if path.endswith("/agent-view"):
            return {"candidate_id": "c9", "seq": 9, "status": "ok", "in_sample_score": 1.2, "model": "Muse",
                    "created_at": 105.0, "code_sha": runner._sha(sent)}
        return {"candidates": [{"id": "c8", "model": "Other", "created_at": 106.0},
                               {"id": "c9", "model": "Muse", "created_at": 105.0}]}

    monkeypatch.setattr(runner, "request", fake)
    runner._act(w).begin("explore", {"id": "o1", "title": "T"})
    submits, parents = [], []
    note = w._recover_pending({"id": "o1"}, {"name": "submit_candidate", "at": 100.0, "code_sha": runner._sha(sent)},
                              submits, parents)
    assert [s["seq"] for s in submits] == [9] and parents == [None]
    assert "code_sha" not in submits[0] and "#9" in note
    assert runner._act(w).rec["submissions"][0]["seq"] == 9
    w.report_eval.assert_called_once()


def test_a_lost_call_is_named_so_the_model_can_repeat_it(runner, monkeypatch):
    w = _worker(runner)
    note = w._recover_pending({"id": "o1"}, {"name": "run_python", "at": 1.0}, [], [])
    assert "run_python" in note and "lost" in note


# ---------------------------------------------------------------------------------------
# Control plane: the steering queue
# ---------------------------------------------------------------------------------------
def test_a_direct_message_is_delivered_once():
    S.add("p1", "Muse", "try the 30-minute bars")
    got = S.take("p1", "Muse", model="Muse", objective_id="o1", since=time.time())
    assert [s["text"] for s in got] == ["try the 30-minute bars"]
    assert S.take("p1", "Muse", model="Muse", objective_id="o1", since=time.time()) == []
    assert S.take("p1", "Qwen", model="Qwen", objective_id="o1") == []          # not for Qwen


def test_a_broadcast_reaches_only_iterations_already_running():
    started_before = time.time() - 60
    S.add("p1", "*", "new champion", objective_id="o1")
    assert len(S.take("p1", "Muse", objective_id="o1", since=started_before)) == 1
    started_after = time.time() + 1
    assert S.take("p1", "Qwen", objective_id="o1", since=started_after) == []     # its prompt had the news
    assert S.take("p1", "Qwen", objective_id="o1", since=started_before) == []    # and it is not re-offered
    assert S.take("p1", "Gemma", objective_id="o2", since=started_before) == []   # another objective


def test_the_champion_news_skips_its_own_model():
    obj = {"id": "o1", "project_id": "p1", "title": "Best strategy"}
    S.broadcast_champion(obj, {"seq": 41, "model": "Muse", "score": 1.234, "rationale": "skew reversal at 10:30"})
    since = time.time() - 60
    assert S.take("p1", "Muse", model="Muse", objective_id="o1", since=since) == []
    got = S.take("p1", "Qwen", model="Qwen", objective_id="o1", since=since)
    assert "#41" in got[0]["text"] and "1.234" in got[0]["text"] and got[0]["source"] == "harness"


def test_a_cancelled_or_empty_message_is_not_delivered():
    s = S.add("p1", "Muse", "never mind")
    S.cancel(s["id"])
    assert S.take("p1", "Muse") == []
    with pytest.raises(ValueError):
        S.add("p1", "Muse", "  ")
    assert S.add("p1", "Muse", "", kind="wrap_up")["kind"] == "wrap_up"


def test_recent_lists_who_received_what():
    s = S.add("p1", "*", "heads up")
    S.take("p1", "Muse", record_id="r1", round_=4, since=time.time() - 5)
    row = next(r for r in S.recent("p1") if r["id"] == s["id"])
    assert row["deliveries"] == [{"agent": "Muse", "record_id": "r1", "round": 4,
                                  "delivered_at": row["deliveries"][0]["delivered_at"]}]


# ---------------------------------------------------------------------------------------
# Control plane: events, resumed records, look-ahead tests across a restart
# ---------------------------------------------------------------------------------------
def test_events_reach_the_subscribers_of_their_project():
    async def main():
        gen = events.stream("p1", keepalive_s=5)
        hello = await gen.__anext__()
        events.publish("p2", "activity", {"x": 1})
        events.publish("p1", "candidate", {"seq": 3})
        frame = await asyncio.wait_for(gen.__anext__(), 2)
        await gen.aclose()
        return hello, frame

    hello, frame = asyncio.run(main())
    assert b"event: hello" in hello
    assert b"event: candidate" in frame and b'"seq": 3' in frame
    assert events.subscribers() == 0


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "DB_PATH", tmp_path / "agent_activity.sqlite3")
    monkeypatch.setattr(A, "_conn", None)
    monkeypatch.setattr(W, "DB_PATH", tmp_path / "work.sqlite3", raising=False)
    monkeypatch.setattr(W, "_conn", None, raising=False)
    A.reset()
    yield A
    A.reset()


def _rec(rid, started, **extra):
    return {"id": rid, "mode": "explore", "status": "running", "started_at": started,
            "objective": {"id": "o1", "title": "T"}, "asked": [], "timeline": [], "chats": [], "submissions": [],
            **extra}


def test_the_record_a_resume_continues_is_resumed_not_interrupted(store):
    store.record({"agent": "Muse", "model": "Muse", "project_id": "p1", "record": _rec("r1", 100.0)})
    store.record({"agent": "Muse", "model": "Muse", "project_id": "p1",
                  "record": _rec("r2", 200.0, resumed_from="r1")})
    recs = {r["id"]: r for r in store.snapshot()[0]["records"]}
    assert recs["r1"]["status"] == "resumed" and recs["r1"]["resumed_by"] == "r2"
    assert recs["r2"]["status"] == "running"
    assert W._outcome(recs["r1"]) == "resumed"


def test_pending_look_ahead_tests_resume_after_a_control_plane_restart(monkeypatch, tmp_path):
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute("CREATE TABLE candidates (id TEXT, objective_id TEXT, seq INT, code TEXT, returns TEXT, "
                "status TEXT, lookahead TEXT)")
    con.executemany("INSERT INTO candidates VALUES (?,?,?,?,?,?,?)", [
        ("c1", "o1", 1, "code", json.dumps([["2026-01-02", 0.01]]), "ok", "pending"),
        ("c2", "o1", 2, "code", "[]", "ok", "pending"),           # its positions file is gone
    ])
    (tmp_path / "o1" / "pending").mkdir(parents=True)
    (tmp_path / "o1" / "pending" / "c1.parquet").write_bytes(b"x")
    monkeypatch.setattr(O, "db", lambda: con)
    monkeypatch.setattr(O, "WORK_ROOT", tmp_path)
    monkeypatch.setattr(O, "get_objective", lambda oid: {
        "id": oid, "project_id": "p", "metric": {"kind": "sharpe", "price_column": "px"}, "dataset": "d",
        "time_column": "t", "lookahead_check": True})
    monkeypatch.setattr(O, "cuts", lambda obj: ["2026-01-01"])
    monkeypatch.setattr(O.projects, "get", lambda pid: {"data_dir": "C:/data"})
    monkeypatch.setattr(O.datasource, "catalog", lambda d: [])
    updates, spawned = {}, []
    monkeypatch.setattr(O, "_update_candidate", lambda cid, f: updates.setdefault(cid, {}).update(f))

    def spawn(coro, what):
        spawned.append(what)
        coro.close()

    monkeypatch.setattr(O, "_spawn", spawn)
    assert asyncio.run(O.resume_pending_lookaheads()) == 1
    assert spawned == ["resumed look-ahead test of #1"]
    assert "resumed" in updates["c1"]["lookahead_detail"] and "lookahead" not in updates["c1"]
    assert updates["c2"] == {"lookahead": "error", "lookahead_detail": O.LOOKAHEAD_LOST}


# ---------------------------------------------------------------------------------------
# End to end: a drain pauses an iteration, the next runner carries it on
# ---------------------------------------------------------------------------------------
SCRIPT = "import ft\n\ndef main():\n    ft.report_positions(x)\n\nmain()\n"
OBJ = {"id": "o1", "title": "T", "metric": {"kind": "sharpe"}, "cooldown_s": 0}


class _Control:
    """The control plane and board, stubbed; `drain` flips on during the first experiment."""

    def __init__(self, runner, monkeypatch):
        self.python_runs, self.candidates, self.steers = [], [], []
        self.drain = False
        self.ctx = {"mode": "explore", "parent": None, "objective": {"title": "T", "metric": {"kind": "sharpe"}},
                    "metric_label": "Sharpe ratio", "library": [], "datasets": ["bars"], "playbook": {}}
        monkeypatch.setattr(runner, "request", self.request)
        monkeypatch.setattr(runner, "drain_requested", lambda: self.drain)
        monkeypatch.setattr(runner, "_mcp_tools", lambda pid: [])
        monkeypatch.setattr(runner, "AUTO_REPAIR", False)
        monkeypatch.setattr(runner, "OBJECTIVE_TOOL_ROUNDS", 4)

    def request(self, base, path, payload=None, **kw):
        if "/context?" in path:
            return self.ctx
        if path.endswith("/objectives?status=running"):
            return {"objectives": [OBJ]}
        if path.endswith("/python"):
            self.python_runs.append(payload["code"])
            self.drain = True                       # restart-swarm.cmd, while the experiment runs
            return {"ok": True, "stdout": "quick_score {'sharpe': 1.0}", "stderr": ""}
        if path.endswith("/candidates") and payload is not None:
            self.candidates.append(payload)
            return {"seq": 300, "candidate_id": "c300", "status": "ok", "score": 1.0}
        if path == "/api/steer/take":
            self.steers.append(payload)
            return {"steers": []}
        return {"files": [], "docs": [], "modules": [], "candidates": [], "entries": []}


def _iteration_worker(runner, replies, seen):
    w = runner.Worker.__new__(runner.Worker)
    w.model, w.agent_name, w.agent_id, w.role, w.slot = "Muse", "Muse #2", "a2", "search", 1
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
    w._objectives, w._obj_checked, w._obj_turn = [], 0.0, -1

    def generate(payload):
        seen.append(json.loads(json.dumps(payload)))
        act = runner._act(w)
        act.chat_start(payload)
        r = _stop("KEEP: lesson.") if not payload.get("tools") and "Write ONE lesson" in str(
            payload["messages"][-1].get("content")) else replies.pop(0)
        act.chat_done(payload, r)
        return r

    w._generate = generate
    return w


def test_an_iteration_paused_by_a_drain_is_carried_on_by_the_next_runner(runner, monkeypatch):
    cp = _Control(runner, monkeypatch)
    seen1: list[dict] = []
    w1 = _iteration_worker(runner, [_call("run_python", {"code": "print(1)"})], seen1)
    w1.iterate(OBJ)
    first = runner._act(w1).rec
    assert w1._paused and first["status"] == "interrupted" and len(cp.python_runs) == 1
    assert [s["round"] for s in cp.steers] == [1]           # steering was read at the top of round 1
    ck = runner._ckpt().load("p1|Muse #2")
    assert ck["converse"]["round"] == 1 and ck["world"]["experiments"] == 1 and ck["record_id"] == first["id"]

    # The next runner: a new worker for the same agent picks the iteration up.
    cp.drain = False
    seen2: list[dict] = []
    w2 = _iteration_worker(runner, [_call("submit_candidate", {"code": SCRIPT, "rationale": "r"})], seen2)
    w2._turn()
    second = runner._act(w2).rec
    assert len(cp.python_runs) == 1                         # the experiment was not run again
    assert [c["code"] for c in cp.candidates] == [SCRIPT]
    tool_msgs = [m for m in seen2[0]["messages"] if m.get("role") == "tool"]
    assert tool_msgs and "quick_score" in tool_msgs[0]["content"]   # the model saw round 1's result
    assert second["resumed_from"] == first["id"] and second["resumed_round"] == 2
    assert second["status"] == "submitted"
    assert runner._ckpt().load("p1|Muse #2") is None       # finished: nothing left to resume
