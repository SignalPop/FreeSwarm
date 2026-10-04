"""Answering feedback before the work: at the start of an explore/build/improve iteration the
agent's own model answers each message in its inbox (accept: the change it will make / reject:
why, with evidence), the replies go on the board to the sender, and what it accepted becomes the
iteration's commitments.

Evidence: by 10-01 Muse-Glimmer had 67 unanswered inbox messages and 0 answered -- mostly the
mentor's per-candidate coaching -- and rarely acted on them.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock
from urllib.parse import unquote

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "swarm_runner.py"
MUSE, QWEN, MENTOR = "Muse-Glimmer-30B-NVFP4", "Qwen3.6-35B-A3B", "DeepSeek-V4-Flash-0731@lambda999"


@pytest.fixture
def runner(monkeypatch):
    spec = importlib.util.spec_from_file_location("swarm_runner_answer_feedback_test", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # The activity record is posted by a background thread: keep it in memory here.
    pushed = []
    monkeypatch.setattr(mod, "_poster", lambda: type("P", (), {"put": lambda self, k, d: pushed.append(d)})())
    mod._pushed = pushed
    return mod


def _worker(runner, monkeypatch, reply=None, *, posts=None, fail_post=()):
    """A worker whose model answers `reply` (a string, or an exception to raise) and whose
    board records what is posted to it."""
    posts = [] if posts is None else posts

    def request(base, path, payload=None, **kw):
        if path == "/mb/messages" and payload is not None:
            if payload.get("reply_to") in fail_post:
                raise RuntimeError("board down")
            posts.append(payload)
            return {"seq": 90000 + len(posts)}
        return {}

    monkeypatch.setattr(runner, "request", request)
    w = runner.Worker.__new__(runner.Worker)
    w.model, w.agent_name, w.agent_id, w.role, w.slot = MUSE, f"{MUSE} #2", "a2", "search", 1
    w.project = {"id": "p1", "slug": "t"}
    w._stop, w.retired = threading.Event(), threading.Event()
    w._budget_block = None
    w.say = MagicMock()
    w.asked = []

    def chat(prompt, max_tokens=2048, system=None):
        w.asked.append({"prompt": prompt, "max_tokens": max_tokens, "deadline": getattr(w, "_gen_deadline", None)})
        if isinstance(reply, Exception):
            raise reply
        return reply(w) if callable(reply) else reply

    w._chat = chat
    w.posts = posts
    runner._act(w).begin("explore", {"id": "o1", "title": "T"})
    return w


def _world():
    return type("W", (), {"saved": [], "sent": []})()


INBOX = [
    {"seq": 501, "from": MENTOR, "channel": "team", "minutes_ago": 40, "candidate": 131,
     "text": "#131 is one-sided: add the mirrored short entry"},
    {"seq": 502, "from": QWEN, "channel": "team", "minutes_ago": 10, "text": "@muse which module did you use?"},
]
OBJ = {"id": "o1", "title": "SPX intraday"}
CTX = {"mode": "explore", "leaderboard": [{"rank": 2, "seq": 131, "model": MUSE, "in_sample_score": 1.04,
                                           "status": "ok", "lookahead": "pass", "rationale": "breakout"}]}
REPLY = ("Thinking... I'll mirror it.\n```json\n"
         + json.dumps([{"reply_to": 501, "verdict": "accept", "text": "accept: add the mirrored short entry to #131 "
                        "with the same contraction threshold"},
                       {"reply_to": 502, "verdict": "reject", "text": "None -- #131 uses no library module; the "
                        "signal is inline (see its code)."}]) + "\n```")


# ---------------------------------------------------------------------------------------
# Replies on the board
# ---------------------------------------------------------------------------------------
def test_each_message_gets_a_reply_to_its_sender_linked_as_an_answer(runner, monkeypatch):
    w = _worker(runner, monkeypatch, REPLY)
    world = _world()
    got = w.answer_feedback(OBJ, CTX, INBOX, world)
    assert [(c["reply_to"], c["verdict"]) for c in got] == [(501, "accept"), (502, "reject")]
    assert got[0]["text"].startswith("add the mirrored short entry") and got[0]["candidate"] == 131
    a, b = w.posts
    assert a["channel"] == "team" and a["author"] == MUSE and a["author_id"] == "a2" and a["kind"] == "chat"
    assert a["reply_to"] == 501 and a["meta"]["reply_to"] == 501 and a["meta"]["answers"] == [501]
    assert a["meta"]["to"] == MENTOR and a["meta"]["agent"] == f"{MUSE} #2" and a["meta"]["feedback_reply"] == "accept"
    assert a["meta"]["objective_id"] == "o1"
    assert a["content"] == (f"@{MENTOR} re #501 [candidate #131] -- accept: add the mirrored short entry to #131 "
                            "with the same contraction threshold")
    assert b["meta"]["to"] == QWEN and b["meta"]["feedback_reply"] == "reject" and b["reply_to"] == 502
    assert [s["answers"] for s in world.sent] == [[501], [502]] and world.sent[0]["to"] == MENTOR


def test_the_answers_count_in_the_collaboration_record_and_reach_the_senders_inbox(runner, monkeypatch):
    w = _worker(runner, monkeypatch, REPLY)
    world = _world()
    w.answer_feedback(OBJ, CTX, INBOX, world)
    posted = list(w.posts)
    monkeypatch.setattr(runner, "request", lambda base, path, payload=None, **kw: {})
    w.record_collaboration({"id": "o1"}, {"mode": "explore", "parent": None}, world,
                           {"seq": 140, "status": "ok", "in_sample_score": 0.5, "lookahead": "pass", "rank": 3},
                           "", INBOX)
    _, _, text, meta = w.say.call_args[0]
    assert [m["seq"] for m in meta["collab"]["answered"]] == [501, 502]
    assert "answered DeepSeek-V4-Flash-0731@lambda999 (#501), Qwen3.6-35B-A3B (#502)" in text
    # On the board: the mentor's inbox shows the reply; Muse's no longer shows #501/#502.
    now = time.time()
    board = [{"seq": 501, "author": MENTOR, "ts": now - 2400, "channel": "team", "content": "x",
              "meta": {"to": MUSE, "candidate_seq": 131}},
             {"seq": 502, "author": QWEN, "ts": now - 600, "channel": "team", "content": "@muse-glimmer-30b-nvfp4 ?"}]
    board += [{**p, "seq": 600 + i, "ts": now - 5} for i, p in enumerate(posted)]
    mentor_inbox = runner._inbox_entries(board, MENTOR, since=0, now=now)
    assert [(m["seq"], m.get("reply_to")) for m in mentor_inbox] == [(600, 501)]
    assert runner._inbox_entries(board, MUSE, since=0, now=now) == []


# ---------------------------------------------------------------------------------------
# The prompts
# ---------------------------------------------------------------------------------------
def test_the_answering_prompt_lists_each_message_its_candidate_and_asks_for_json(runner, monkeypatch):
    w = _worker(runner, monkeypatch, REPLY)
    w.answer_feedback(OBJ, CTX, INBOX, _world())
    p = w.asked[0]["prompt"]
    assert f"#501 from {MENTOR} (40 min ago, about your candidate 131 [candidate 131 by {MUSE}, ok, " \
           "in-sample 1.040, rank 2, look-ahead pass]): #131 is one-sided" in p
    assert f"#502 from {QWEN} (10 min ago): @muse which module" in p
    assert "accept: the concrete change you will make THIS iteration" in p and "reject: why not, with evidence" in p
    assert '[{"reply_to": <message number>, "verdict": "accept"|"reject"|"question", "text": "..."}]' in p
    assert runner.MIN_OUTPUT <= w.asked[0]["max_tokens"] <= runner.MAX_TOKENS
    # The call runs under the wall-time cap, which is cleared afterwards.
    assert abs(w.asked[0]["deadline"] - time.time() - runner.FEEDBACK_MAX_S) < 5 and w._gen_deadline is None


def _ctx(**kw):
    ctx = {"objective": {"title": "T", "metric": {"kind": "sharpe", "price_column": "Close"}, "dataset": "bars"},
           "metric_label": "Sharpe ratio", "mode": "explore", "parent": None, "library": [], "datasets": ["bars"]}
    ctx.update(kw)
    return ctx


def test_the_iteration_prompt_carries_the_commitments_accepted_first(runner):
    commitments = [{"reply_to": 502, "verdict": "reject", "text": "#131 uses no module", "from": QWEN},
                   {"reply_to": 501, "verdict": "accept", "text": "add the mirrored short entry", "from": MENTOR,
                    "candidate": 131}]
    p = runner.iteration_prompt(_ctx(commitments=commitments))
    head = "YOUR COMMITMENTS THIS ITERATION (from the feedback you just answered"
    assert head in p and p.index(head) < p.index("YOUR ASSIGNMENT THIS ITERATION")
    section = p[p.index(head):p.index("YOUR ASSIGNMENT THIS ITERATION")]
    assert section.index("- ACCEPTED #501 (DeepSeek-V4-Flash-0731@lambda999, candidate 131): add the mirrored") \
        < section.index("- REJECTED #502 (Qwen3.6-35B-A3B): #131 uses no module")
    assert "how it carries out YOUR COMMITMENTS" in p
    assert "YOUR COMMITMENTS" not in runner.iteration_prompt(_ctx())


def test_the_commitments_section_stays_compact(runner):
    many = [{"reply_to": n, "verdict": "accept", "text": "x" * 600, "from": MENTOR} for n in range(10)]
    lines = runner._commitment_lines(many)
    assert len("\n".join(lines)) <= runner.COMMITMENTS_CHARS + 40
    assert lines[-1].startswith("- (+") and "more on the board" in lines[-1]


# ---------------------------------------------------------------------------------------
# Never blocking the iteration
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("reply,why", [
    (RuntimeError("POST /v1/chat/completions -> 500: engine down"), "the answering call failed"),
    ("I think the mentor is right about most of it.", "no usable JSON list"),
    ('[{"reply_to": 999, "verdict": "accept", "text": "not a message of mine"}]', "no usable JSON list"),
])
def test_a_failed_or_unparseable_answer_posts_nothing_and_is_recorded(runner, monkeypatch, reply, why):
    from app import work as W
    w = _worker(runner, monkeypatch, reply)
    assert w.answer_feedback(OBJ, CTX, INBOX, _world()) == [] and w.posts == []
    tl = runner._act(w).rec["timeline"]
    step = [e for e in tl if e.get("name") == "answer_feedback"][-1]
    # A soft step: recorded ok, the reason under "soft_error" -- not an unrecovered tool failure.
    assert step["kind"] == "tool" and step["ok"] is True and step["soft"] is True and why in step["result"]
    res = json.loads(step["result"])
    assert res["soft"] is True and why in res["soft_error"] and res["answered"] == 0 and "error" not in res
    assert not W.tool_failed(step) and W.recovery(tl) == {}


def test_a_cut_off_answer_is_named_as_such(runner, monkeypatch):
    def cut(w):
        w._last_finish = "length"
        return '[{"reply_to": 501, "verdict": "acc'
    w = _worker(runner, monkeypatch, cut)
    assert w.answer_feedback(OBJ, CTX, INBOX, _world()) == []
    step = runner._act(w).rec["timeline"][-1]
    assert "cut off before a complete JSON list" in step["result"]


def test_a_reply_that_cannot_be_posted_is_not_counted_and_the_rest_go_on(runner, monkeypatch):
    w = _worker(runner, monkeypatch, REPLY, fail_post=(501,))
    world = _world()
    got = w.answer_feedback(OBJ, CTX, INBOX, world)
    assert [c["reply_to"] for c in got] == [502] and [s["reply_to"] for s in world.sent] == [502]


def test_the_step_shows_on_the_work_timeline(runner, monkeypatch):
    w = _worker(runner, monkeypatch, REPLY)
    w.answer_feedback(OBJ, CTX, INBOX, _world())
    step = runner._act(w).rec["timeline"][-1]
    assert step["kind"] == "tool" and step["name"] == "answer_feedback" and step["ok"] is True
    assert step["args"]["messages"][0].startswith(f"#501 from {MENTOR}")
    res = json.loads(step["result"])
    assert res["answered"] == 2 and res["accept"] == 1 and res["reject"] == 1
    assert res["replies"][0].startswith("#501 accept: add the mirrored")


def test_a_spending_limit_refusal_is_left_to_the_backoff(runner, monkeypatch):
    def refused(w):
        w._budget_block = "429: spending limit reached for today"
        raise RuntimeError(w._budget_block)
    w = _worker(runner, monkeypatch, refused)
    assert w.answer_feedback(OBJ, CTX, INBOX, _world()) == [] and w._budget_block
    # ... and while it stands, no answering call is made at all.
    w.asked.clear()
    assert w.answer_feedback(OBJ, CTX, INBOX, _world()) == [] and w.asked == []


def test_the_deadline_stops_a_request_that_has_no_time_left(runner, monkeypatch):
    w = _worker(runner, monkeypatch, REPLY)
    monkeypatch.setattr(runner, "request", lambda *a, **k: pytest.fail("no request after the deadline"))
    w._gen_deadline = time.time() + 1
    with pytest.raises(RuntimeError, match="no time left"):
        w._generate({"model": MUSE, "messages": []})


# ---------------------------------------------------------------------------------------
# The switch and the caps
# ---------------------------------------------------------------------------------------
def test_the_switch_turns_it_off(runner, monkeypatch):
    monkeypatch.setattr(runner, "ANSWER_FEEDBACK", False)
    w = _worker(runner, monkeypatch, REPLY)
    assert w.answer_feedback(OBJ, CTX, INBOX, _world()) == [] and w.asked == [] and w.posts == []


def test_nothing_to_answer_makes_no_call(runner, monkeypatch):
    w = _worker(runner, monkeypatch, REPLY)
    assert w.answer_feedback(OBJ, CTX, [], _world()) == [] and w.asked == []
    w._stop.set()
    assert w.answer_feedback(OBJ, CTX, INBOX, _world()) == [] and w.asked == []


def test_the_switch_reads_the_environment(monkeypatch):
    def load(value):
        monkeypatch.setenv("FREESWARM_ANSWER_FEEDBACK", value)
        spec = importlib.util.spec_from_file_location("swarm_runner_answer_feedback_env", RUNNER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.ANSWER_FEEDBACK
    assert load("1") is True and load("0") is False and load("off") is False


def test_only_the_newest_four_within_a_day_are_answered(runner, monkeypatch):
    assert runner.FEEDBACK_MAX_MESSAGES == 4 and runner.FEEDBACK_MAX_S >= 480
    inbox = ([{"seq": 100 + i, "from": MENTOR, "minutes_ago": 26 * 60, "text": f"old {i}"} for i in range(2)]
             + [{"seq": 200 + i, "from": MENTOR, "minutes_ago": 60 - i, "text": f"new {i}"} for i in range(8)])
    due, old = runner._feedback_due(inbox)
    assert [m["seq"] for m in due] == list(range(204, 208)) and [m["seq"] for m in old] == [100, 101]

    answers = json.dumps([{"reply_to": n, "verdict": "accept", "text": f"will do {n}"} for n in (100, 200, 203, 205, 207)])
    w = _worker(runner, monkeypatch, answers)
    got = w.answer_feedback(OBJ, CTX, inbox, _world())
    assert [c["reply_to"] for c in got] == [205, 207]           # an old or over-the-cap message is not answered
    p = w.asked[0]["prompt"]
    assert "#207 from" in p and "#204 from" in p and "#203 from" not in p and "#100 from" not in p
    assert p.index("#204 from") < p.index("#207 from")         # oldest first in the prompt


@pytest.mark.parametrize("item,verdict,text", [
    ({"reply_to": 5, "verdict": "Accepted", "text": "do it"}, "accept", "do it"),
    ({"reply_to": "#5", "text": "reject: the drawdown doubled in #140"}, "reject", "the drawdown doubled in #140"),
    ({"reply_to": 5, "verdict": "question", "text": "Which horizon?"}, "question", "Which horizon?"),
    ({"reply_to": 5, "verdict": "maybe", "text": "hmm"}, None, None),
    ({"reply_to": 5, "verdict": "accept", "text": "  "}, None, None),
])
def test_reply_parsing(runner, item, verdict, text):
    got = runner._parse_feedback_replies("draft [1, 2] then\n" + json.dumps([item]), {5})
    assert got == ([{"reply_to": 5, "verdict": verdict, "text": text}] if verdict else [])


# ---------------------------------------------------------------------------------------
# Wired into the iteration
# ---------------------------------------------------------------------------------------
def _iterate(runner, monkeypatch, reply, inbox):
    """Run Worker.iterate up to the model's first turn; return (posts, the iteration prompt, worker)."""
    posts, seen = [], {}
    ctx = {"mode": "explore", "parent": None, "objective": {"title": "T", "metric": {"kind": "sharpe"}},
           "metric_label": "Sharpe ratio", "library": [], "datasets": ["bars"], "playbook": {}}

    w = _worker(runner, monkeypatch, reply, posts=posts)
    w.project = {"id": "p1", "slug": "t", "name": "P"}

    def request(base, path, payload=None, **kw):
        if "/context?" in path:
            return ctx
        if path == "/mb/messages" and payload is not None:
            posts.append(payload)
            return {"seq": 1}
        return {"files": [], "docs": [], "modules": [], "candidates": [], "entries": []}

    monkeypatch.setattr(runner, "request", request)
    monkeypatch.setattr(runner, "_mcp_tools", lambda pid: [])
    w._sync = lambda: ([], [])
    w._inbox_since = 0.0
    w.beat = lambda *a, **k: None
    w.teammates = lambda: []
    w.mentor_coaching = lambda: None
    w.inbox = lambda: list(inbox)

    def converse(messages, tools, call, **kw):
        seen["prompt"] = messages[1]["content"]
        return False, "", {"tool_calls": 0}

    w.converse = converse
    w.iterate({"id": "o1", "title": "T", "metric": {"kind": "sharpe"}})
    return posts, seen.get("prompt"), w


def test_iterate_answers_first_and_hands_the_commitments_to_the_work(runner, monkeypatch):
    extra = {"seq": 503, "from": QWEN, "channel": "team", "minutes_ago": 5, "text": "left for later"}
    reply = json.dumps([{"reply_to": 501, "verdict": "accept", "text": "add the mirrored short entry"}])
    posts, prompt, _ = _iterate(runner, monkeypatch, reply, INBOX[:1] + [extra])
    assert [p["reply_to"] for p in posts if p.get("meta", {}).get("feedback_reply")] == [501]
    assert "- ACCEPTED #501" in prompt and "add the mirrored short entry" in prompt
    # The answered message leaves MESSAGES TO YOU; the unanswered one stays there.
    msgs = prompt[prompt.index("MESSAGES TO YOU"):]
    assert "#503 from" in msgs and "#501 from" not in msgs


def test_iterate_goes_on_when_answering_fails(runner, monkeypatch):
    posts, prompt, _ = _iterate(runner, monkeypatch, RuntimeError("engine down"), INBOX)
    assert prompt is not None and "YOUR COMMITMENTS" not in prompt and "#501 from" in prompt
    assert not [p for p in posts if p.get("meta", {}).get("feedback_reply")]


def test_iterate_stops_for_the_spending_limit(runner, monkeypatch):
    def refused(w):
        w._budget_block = "spending limit"
        raise RuntimeError("spending limit")
    _, prompt, w = _iterate(runner, monkeypatch, refused, INBOX)
    assert prompt is None and w._budget_block


# ---------------------------------------------------------------------------------------
# One agent answers a message (10-01 19:02 / 19:18: both agents of one model answered the
# same messages within a minute, and the mentor then replied to both answers)
# ---------------------------------------------------------------------------------------
class Board:
    """The message board's blackboard (/mb/state, compare-and-set on the version, as in
    app/msgboard.py) and its message log, shared by several workers."""

    def __init__(self):
        self.state, self.posts, self.stale = {}, [], 0

    def request(self, base, path, payload=None, method=None, **kw):
        if path == "/mb/messages" and payload is not None:
            self.posts.append(payload)
            return {"seq": 90000 + len(self.posts)}
        if path.startswith("/mb/state/"):
            key = unquote(path[len("/mb/state/"):].split("?")[0])
            row = self.state.get(key)
            if method == "PUT":
                current = row["version"] if row else 0
                if payload.get("expect_version") is not None and payload["expect_version"] != current:
                    raise RuntimeError(f"/mb/state/{key} -> 409: version conflict")
                self.state[key] = {"key": key, "version": current + 1, "value": payload["value"]}
                return {"key": key, "version": current + 1}
            if self.stale:                      # a read that another agent's write overtakes
                self.stale -= 1
                return {}
            if row is None:
                raise RuntimeError(f"/mb/state/{key} -> 404: no such key")
            return dict(row)
        return {}


def _twin(runner, monkeypatch, board, reply, *, name, agent_id):
    w = _worker(runner, monkeypatch, reply)
    monkeypatch.setattr(runner, "request", board.request)
    w.agent_name, w.agent_id = name, agent_id
    return w


def test_two_agents_of_one_model_answer_each_message_once(runner, monkeypatch):
    board = Board()
    a = _twin(runner, monkeypatch, board, REPLY, name=MUSE, agent_id="a1")
    b = _twin(runner, monkeypatch, board, REPLY, name=f"{MUSE} #2", agent_id="a2")
    assert [c["reply_to"] for c in a.answer_feedback(OBJ, CTX, INBOX, _world())] == [501, 502]
    assert b.answer_feedback(OBJ, CTX, INBOX, _world()) == [] and b.asked == []
    assert len(board.posts) == 2 and {p["meta"]["agent"] for p in board.posts} == {MUSE}
    assert b.feedback_skipped == {501: f"being answered by {MUSE}", 502: f"being answered by {MUSE}"}
    claim = board.state["fbclaim:501"]["value"]
    assert claim["agent"] == MUSE and claim["agent_id"] == "a1"
    assert abs(claim["until"] - time.time() - runner.FEEDBACK_CLAIM_TTL_S) < 5 and runner.FEEDBACK_CLAIM_TTL_S == 1800


def test_a_claim_lost_between_read_and_write_is_left_to_the_winner(runner, monkeypatch):
    board = Board()
    a = _twin(runner, monkeypatch, board, REPLY, name=MUSE, agent_id="a1")
    b = _twin(runner, monkeypatch, board, REPLY, name=f"{MUSE} #2", agent_id="a2")
    assert a._claim_feedback(501)[0] is None
    board.stale = 1                             # b read before a's write landed
    assert b._claim_feedback(501) == (MUSE, None)
    assert board.state["fbclaim:501"]["value"]["agent"] == MUSE


def test_an_expired_claim_is_taken_over_and_an_own_claim_renewed(runner, monkeypatch):
    board = Board()
    a = _twin(runner, monkeypatch, board, REPLY, name=MUSE, agent_id="a1")
    b = _twin(runner, monkeypatch, board, REPLY, name=f"{MUSE} #2", agent_id="a2")
    board.state["fbclaim:501"] = {"version": 3, "value": {"agent": MUSE, "agent_id": "a1", "until": time.time() - 1}}
    assert b._claim_feedback(501) == (None, 4) and board.state["fbclaim:501"]["value"]["agent"] == f"{MUSE} #2"
    assert a._claim_feedback(501) == (f"{MUSE} #2", None)
    assert b._claim_feedback(501) == (None, 5)  # its own claim: renewed, not refused


def test_a_failed_answer_releases_its_claims_for_either_agent(runner, monkeypatch):
    board = Board()
    a = _twin(runner, monkeypatch, board, RuntimeError("timed out"), name=MUSE, agent_id="a1")
    b = _twin(runner, monkeypatch, board, REPLY, name=f"{MUSE} #2", agent_id="a2")
    assert a.answer_feedback(OBJ, CTX, INBOX, _world()) == []
    assert board.state["fbclaim:501"]["value"]["until"] == 0
    step = runner._act(a).rec["timeline"][-1]
    assert json.loads(step["result"])["released"] == [501, 502]
    assert [c["reply_to"] for c in b.answer_feedback(OBJ, CTX, INBOX, _world())] == [501, 502]


def test_an_unreachable_board_does_not_stop_the_answer(runner, monkeypatch):
    w = _worker(runner, monkeypatch, REPLY)
    posts = []

    def request(base, path, payload=None, **kw):
        if path.startswith("/mb/state/"):
            raise RuntimeError("cannot reach http://127.0.0.1:8510/mb/state: refused")
        posts.append(payload)
        return {}
    monkeypatch.setattr(runner, "request", request)
    assert [c["reply_to"] for c in w.answer_feedback(OBJ, CTX, INBOX, _world())] == [501, 502] and len(posts) == 2


def test_feedback_on_a_candidate_goes_to_the_agent_that_ran_it(runner, monkeypatch):
    now = time.time()
    # Muse #1 ran #131 (its collaboration record says so); the mentor wrote to the model about it.
    entries = [{"seq": 400, "author": MUSE, "author_id": "a1", "ts": now - 3600, "channel": "team",
                "content": "muse (explore): ...", "meta": {"agent": MUSE, "collab": {"candidate": 131}}},
               {"seq": 501, "author": MENTOR, "ts": now - 300, "channel": "team", "content": "#131 is one-sided",
                "meta": {"to": MUSE, "candidate_seq": 131}},
               {"seq": 502, "author": QWEN, "ts": now - 300, "channel": "team", "content": "@muse-glimmer-30b-nvfp4 ?"}]
    inbox = runner._inbox_entries(entries, MUSE, since=0, now=now)
    assert inbox[0]["candidate_by"] == {"id": "a1", "agent": MUSE} and "candidate_by" not in inbox[1]
    board = Board()
    b = _twin(runner, monkeypatch, board, REPLY, name=f"{MUSE} #2", agent_id="a2")
    assert [c["reply_to"] for c in b.answer_feedback(OBJ, CTX, inbox, _world())] == [502]
    assert b.feedback_skipped == {501: f"left to {MUSE}, who ran candidate 131"} and "#501 from" not in b.asked[0]["prompt"]
    a = _twin(runner, monkeypatch, board, REPLY, name=MUSE, agent_id="a1")
    assert [c["reply_to"] for c in a.answer_feedback(OBJ, CTX, inbox, _world())] == [501]
    # Its own submission counts even before its record is on the board; a stale message is anyone's.
    assert runner._feedback_route(inbox[0], "a2", {131}) is None
    assert runner._feedback_route({**inbox[0], "minutes_ago": 31}, "a2", set()) is None
    assert runner._feedback_route(inbox[0], "a2", set()) == MUSE


def test_the_collaboration_record_names_the_agent(runner, monkeypatch):
    w = _worker(runner, monkeypatch, REPLY)
    monkeypatch.setattr(runner, "request", lambda base, path, payload=None, **kw: {})
    w.record_collaboration({"id": "o1"}, {"mode": "explore", "parent": None}, _world(),
                           {"seq": 140, "status": "ok"}, "", [])
    meta = w.say.call_args[0][3]
    assert meta["agent"] == f"{MUSE} #2" and meta["collab"]["candidate"] == 140


# ---------------------------------------------------------------------------------------
# The dialogue ends
# ---------------------------------------------------------------------------------------
def _thread(now):
    """mentor #1 -> Muse answers #2 -> mentor replies #3 -> Muse answers #4 -> mentor replies #5."""
    def m(seq, author, rt=None, **meta):
        return {"seq": seq, "author": author, "ts": now - 600 + seq, "channel": "team", "content": f"m{seq}",
                "reply_to": rt, "meta": meta}
    return [m(1, MENTOR, to=MUSE, candidate_seq=131),
            m(2, MUSE, 1, to=MENTOR, feedback_reply="reject", feedback_depth=1, answers=[1]),
            m(3, MENTOR, 2, to=MUSE),
            m(4, MUSE, 3, to=MENTOR, feedback_reply="question", feedback_depth=2, answers=[3]),
            m(5, MENTOR, 4, to=MUSE)]


def test_an_agent_answers_a_reply_to_its_answer_once_and_then_the_thread_ends(runner, monkeypatch):
    now = time.time()
    board = _thread(now)
    got = runner._inbox_entries(board[:3], MUSE, since=0, now=now)
    assert [(m["seq"], m.get("thread_depth")) for m in got] == [(3, 1)]
    # Answering #3 is a depth-2 answer ...
    w = _worker(runner, monkeypatch, json.dumps([{"reply_to": 3, "verdict": "reject", "text": "no: see #140"}]))
    w.answer_feedback(OBJ, CTX, got, _world())
    assert w.posts[0]["meta"]["feedback_depth"] == 2
    # ... and the mentor's reply to it is not answered.
    last = runner._inbox_entries(board, MUSE, since=0, now=now)
    assert [(m["seq"], m.get("thread_depth")) for m in last] == [(5, 2)]
    w = _worker(runner, monkeypatch, REPLY)
    assert w.answer_feedback(OBJ, CTX, last, _world()) == [] and w.asked == []
    assert "closed" in w.feedback_skipped[5]
    # A first answer is depth 1; answers from before the depth was recorded count as 1.
    w = _worker(runner, monkeypatch, REPLY)
    w.answer_feedback(OBJ, CTX, INBOX, _world())
    assert {p["meta"]["feedback_depth"] for p in w.posts} == {1}
    assert runner._feedback_depth({"feedback_reply": "accept"}) == 1 and runner._feedback_depth({}) == 0


def test_the_mentor_does_not_reply_to_plain_acknowledgements(runner, monkeypatch):
    now = time.time()
    board = [{"seq": 2, "author": MUSE, "ts": now - 60, "channel": "team", "content": "accept: will mirror",
              "reply_to": 1, "meta": {"to": MENTOR, "feedback_reply": "accept", "answers": [1]}},
             {"seq": 3, "author": QWEN, "ts": now - 50, "channel": "team", "content": "reject: the drawdown doubled",
              "reply_to": 1, "meta": {"to": MENTOR, "feedback_reply": "reject", "answers": [1]}},
             {"seq": 4, "author": QWEN, "ts": now - 40, "channel": "team", "content": "question: which horizon?",
              "meta": {"to": MENTOR, "feedback_reply": "question"}}]
    assert [m["seq"] for m in runner._inbox_entries(board, MENTOR, since=0, now=now, acks=False)] == [3, 4]
    assert [m["seq"] for m in runner._inbox_entries(board, MENTOR, since=0, now=now)] == [2, 3, 4]
    w = _worker(runner, monkeypatch, REPLY)
    w.model, w.role, w._inbox_since = MENTOR, "mentor", 0.0
    monkeypatch.setattr(runner, "request", lambda base, path, payload=None, **kw: {"entries": board})
    assert [m["seq"] for m in w.inbox()] == [3, 4]
    p = runner.mentor_prompt({"objective": {"title": "T", "metric": {"kind": "sharpe"}}, "metric_label": "Sharpe"},
                             w.inbox())
    assert f"- [3] from {QWEN} (REJECT of your feedback): reject: the drawdown doubled" in p
    assert "accept: will mirror" not in p and "otherwise leave it, the thread is done" in p
    # An agent does not answer a teammate's acknowledgement either.
    w = _worker(runner, monkeypatch, REPLY)
    ack = {"seq": 9, "from": QWEN, "minutes_ago": 1, "text": "accept: will do", "feedback_reply": "accept"}
    assert w.answer_feedback(OBJ, CTX, [ack], _world()) == [] and w.asked == []
    assert "acknowledgement" in w.feedback_skipped[9]


def test_a_slow_model_gets_more_time_and_fewer_messages(runner, monkeypatch):
    runner._speed[MUSE] = 5.0                   # measured: even one answer needs the hard cap
    w = _worker(runner, monkeypatch, REPLY)
    got = w.answer_feedback(OBJ, CTX, INBOX, _world())
    assert [c["reply_to"] for c in got] == [502]                 # the newest only
    assert abs(w.asked[0]["deadline"] - time.time() - runner.FEEDBACK_HARD_MAX_S) < 5
    assert json.loads(json.dumps(runner._act(w).rec["timeline"][-1]["args"]))["time_limit_s"] == 900


def test_messages_not_answered_leave_the_iteration_brief(runner, monkeypatch):
    ack = {"seq": 504, "from": QWEN, "channel": "team", "minutes_ago": 2, "text": "accept: thanks",
           "feedback_reply": "accept"}
    reply = json.dumps([{"reply_to": 501, "verdict": "accept", "text": "add the mirrored short entry"}])
    _, prompt, w = _iterate(runner, monkeypatch, reply, INBOX[:1] + [ack])
    assert 504 in w.feedback_skipped and "#504 from" not in prompt and "- ACCEPTED #501" in prompt
