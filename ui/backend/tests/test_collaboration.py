"""Collaboration plumbing: agents build on each other's candidates, reuse the library, read and
answer their messages, and get the mentor's coaching.

Evidence these guard against (Gex2, 09-30 .. 10-01): 0 of 27 candidates in a day had a parent
(every iteration was an explore once no ranked candidate made money); #132/#133 "Building on
candidate 75" were stored as starting from nothing; the mentor's replies went "to" agent / team /
ok / lowk with candidate numbers as reply_to, so nobody's inbox matched them; plans acted on the
mentor's notes without reply_to and every message counted as unanswered; the reuse detector
missed `from lib import x as y`; the mentor's coaching was clipped away from every brief.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import random
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app import external, library
from app import objectives as O

RUNNER = Path(__file__).resolve().parents[1] / "swarm_runner.py"
MUSE, QWEN, MENTOR = "Muse-Glimmer-30B-NVFP4", "Qwen3.6-35B-A3B", "DeepSeek-V4-Flash-0731@lambda999"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("swarm_runner_collaboration_test", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def msg(seq, author, content, *, ts=None, channel="team", meta=None, reply_to=None, author_id=None):
    return {"seq": seq, "author": author, "content": content, "ts": ts if ts is not None else time.time() - 60,
            "channel": channel, "meta": meta or {}, "reply_to": reply_to, "author_id": author_id}


# ---------------------------------------------------------------------------------------
# The inbox
# ---------------------------------------------------------------------------------------
def test_inbox_takes_to_short_names_agent_names_mentions_and_replies_to_my_messages(runner):
    board = [
        msg(1, MUSE, "Plan: contraction breakout", channel="planning"),
        msg(2, MENTOR, "make it symmetric", meta={"to": MUSE, "candidate_seq": 131}),
        msg(3, MENTOR, "try slope", meta={"to": "muse-glimmer-30b-nvfp4"}),
        msg(4, MENTOR, "and you", meta={"to": f"{MUSE} #2"}),
        msg(5, QWEN, "@Muse-Glimmer-30B-NVFP4 which module?"),
        msg(6, QWEN, "good plan", reply_to=1),                     # a reply to Muse's own post
        msg(7, MENTOR, "for Qwen", meta={"to": QWEN}),
        msg(8, MENTOR, "to nobody", meta={"to": "team member who ran candidate 131"}),
    ]
    got = runner._inbox_entries(board, MUSE, since=0)
    assert [m["seq"] for m in got] == [2, 3, 4, 5, 6]
    assert got[0]["candidate"] == 131 and got[-1]["reply_to"] == 1


def test_inbox_skips_what_the_model_already_answered_and_old_or_own_messages(runner):
    now = time.time()
    board = [
        msg(10, MENTOR, "old", meta={"to": MUSE}, ts=now - 7 * 3600),
        msg(11, MENTOR, "answered by the twin", meta={"to": MUSE}),
        msg(12, MENTOR, "cited by a plan", meta={"to": MUSE}),
        msg(13, MENTOR, "still open", meta={"to": MUSE}),
        msg(14, MUSE, "on it", reply_to=11),
        msg(15, MUSE, "Plan: acting on #12", channel="planning", meta={"answers": [12]}),
        msg(16, MUSE, "@Muse-Glimmer-30B-NVFP4 note to self"),
    ]
    got = runner._inbox_entries(board, MUSE, since=now - runner.INBOX_WINDOW_S, now=now)
    assert [m["seq"] for m in got] == [13]


def test_inbox_keeps_the_newest_eight(runner):
    board = [msg(i, MENTOR, f"n{i}", meta={"to": MUSE}) for i in range(1, 13)]
    assert [m["seq"] for m in runner._inbox_entries(board, MUSE, 0)] == list(range(5, 13))


def test_a_restarted_agent_reads_hours_back_not_one(runner):
    assert runner.INBOX_WINDOW_S >= 6 * 3600 and runner.INBOX_TAIL == 500


# ---------------------------------------------------------------------------------------
# team_post: replies link themselves
# ---------------------------------------------------------------------------------------
def _world(runner, monkeypatch, handler):
    def request(base, path, payload=None, **kw):
        out = handler(path, payload)
        return {"files": [], "docs": [], "modules": [], "candidates": []} if out is None else out

    monkeypatch.setattr(runner, "request", request)
    monkeypatch.setattr(runner, "_mcp_tools", lambda pid: [])
    return runner.ObjectiveWorld({"id": "p1", "name": "P"}, MUSE, [], [],
                                 {"id": "o1", "metric": {"kind": "sharpe", "price_column": "Close"}, "dataset": "bars"},
                                 on_submit=lambda args: {"status": "ok"})


def test_a_plan_citing_an_inbox_message_answers_it(runner, monkeypatch):
    posted = []
    world = _world(runner, monkeypatch, lambda path, payload: posted.append(payload) or {"seq": 99})
    world.inbox_seqs, world.agent_name, world.author_id = {75840, 75841}, f"{MUSE} #2", "a2"
    out = world.call("team_post", {"text": "Plan: symmetric breakout, acting on #75840 and #75841 (not #131)",
                                   "channel": "planning"})
    body = posted[-1]
    assert body["reply_to"] == 75840 and body["meta"]["answers"] == [75840, 75841]
    assert body["meta"]["agent"] == f"{MUSE} #2" and body["author_id"] == "a2" and body["channel"] == "planning"
    assert world.sent[-1]["answers"] == [75840, 75841] and out["answers"] == ["#75840", "#75841"]
    # An explicit reply_to stays as given; a number that is not in the inbox is not an answer.
    world.call("team_post", {"text": "re #131", "reply_to": "75841", "to": "all"})
    assert posted[-1]["reply_to"] == 75841 and posted[-1]["meta"]["answers"] == [75841]
    world.call("team_post", {"text": "candidate #131 is one-sided"})
    assert "reply_to" not in posted[-1] and "answers" not in posted[-1]["meta"]


# ---------------------------------------------------------------------------------------
# Building on candidates
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,seq", [
    ("Building on candidate 75 (Sharpe -0.808), which uses composite SkewRR", 75),
    ("Improvement over candidate 75 (-0.808 Sharpe): Uses the same composite", 75),
    ("Improve candidate 94 by filtering entries", 94),
    ("Mirror #131 for shorts", 131),
    ("A fundamentally different family from candidate 75", None),
    ("Pressure_Below under short gamma (candidates 111, 113 failed)", None),
])
def test_a_rationale_names_the_candidate_it_builds_on(runner, text, seq):
    assert runner._built_on_ref(text) == seq


def _worker(runner, model=MUSE):
    w = runner.Worker.__new__(runner.Worker)
    w.model = model
    w.agent_name = model
    w.agent_id = "a1"
    w.project = {"id": "p1", "slug": "t"}
    w._stop, w.retired = threading.Event(), threading.Event()
    w.say = MagicMock()
    return w


CANDS = [{"id": "c75", "seq": 75, "model": MUSE}, {"id": "c131", "seq": 131, "model": MUSE},
         {"id": "c133", "seq": 133, "model": QWEN}]


def test_the_submission_parent_named_assigned_or_in_the_rationale(runner, monkeypatch):
    world = _world(runner, monkeypatch, lambda path, payload: {"candidates": CANDS})
    w = _worker(runner, QWEN)
    explore = {"mode": "explore", "parent": None}
    assert w._submission_parent(world, {"parent": "#131", "rationale": "x"}, explore) == {"id": "c131", "seq": 131}
    assert w._submission_parent(world, {"parent": 75}, {"mode": "improve", "parent": {"id": "c131", "seq": 131}}) \
        == {"id": "c75", "seq": 75}                                     # what it names wins
    assert w._submission_parent(world, {}, {"mode": "improve", "parent": {"id": "c131", "seq": 131}}) \
        == {"id": "c131", "seq": 131}
    assert w._submission_parent(world, {"rationale": "Building on candidate 75 ..."}, explore) == {"id": "c75", "seq": 75}
    assert w._submission_parent(world, {"rationale": "a new idea"}, explore) is None
    assert w._submission_parent(world, {"parent": "999"}, explore) is None          # no such candidate
    assert w._submission_parent(world, {"parent": "999"}, {"mode": "improve", "parent": {"id": "c131", "seq": 131}})         == {"id": "c131", "seq": 131}                                   # ... then the assigned one


def test_the_collaboration_record_counts_the_real_parent_reuse_and_cited_answers(runner, monkeypatch):
    code = ("import ft\nfrom lib import composite_skew_oinet_vwap as csv, vol_contraction_breakout\n"
            "from lib import (gex_vol_regime,\n    day_type_regime)\nfrom lib.trailing_stop_signal_v2 import signal\n")
    mods = [{"name": "composite_skew_oinet_vwap", "author": MUSE}, {"name": "vol_contraction_breakout", "author": MUSE},
            {"name": "gex_vol_regime", "author": MUSE}, {"name": "day_type_regime", "author": QWEN},
            {"name": "trailing_stop_signal_v2", "author": QWEN}]

    def handler(path, payload):
        if path.endswith("/candidates/c75"):
            return {"model": MUSE}
        if path.endswith("/candidates/c140"):
            return {"code": code}
        if path.endswith("/library"):
            return {"modules": mods}
        return {}

    monkeypatch.setattr(runner, "request", lambda base, path, payload=None, **kw: handler(path, payload))
    w = _worker(runner, QWEN)
    world = type("W", (), {"saved": [], "sent": [{"to": "all", "channel": "planning", "reply_to": 501,
                                                    "text": "Plan", "answers": [501, 502]}]})()
    inbox = [{"seq": 501, "from": MENTOR, "text": "a"}, {"seq": 502, "from": MENTOR, "text": "b"},
             {"seq": 503, "from": MENTOR, "text": "c"}]
    last = {"candidate_id": "c140", "seq": 140, "status": "ok", "in_sample_score": 0.5, "lookahead": "pass", "rank": 3}
    w.record_collaboration({"id": "o1"}, {"mode": "explore", "parent": None}, world, last, "", inbox,
                           built_on={"id": "c75", "seq": 75})
    _, _, text, meta = w.say.call_args[0]
    c = meta["collab"]
    assert c["built_on"] == {"seq": 75, "by": MUSE} and "built on Muse-Glimmer-30B-NVFP4's candidate #75" in text
    assert {r["module"] for r in c["reused"]} == {"composite_skew_oinet_vwap", "vol_contraction_breakout",
                                                  "gex_vol_regime", "day_type_regime", "trailing_stop_signal_v2"}
    assert "reused lib.composite_skew_oinet_vwap (Muse-Glimmer-30B-NVFP4)" in text
    assert [m["seq"] for m in c["answered"]] == [501, 502] and c["inbox"] == 3


def test_lib_imports_reads_what_the_library_reads(runner):
    code = "from lib import a_mod as x, b_mod\nfrom lib import (c_mod,\n d_mod)\nimport lib.e_mod\nfrom lib.f_mod import g\n"
    assert runner._lib_imports(code) == library.imported_modules(code) == ["a_mod", "b_mod", "c_mod", "d_mod", "e_mod", "f_mod"]


# ---------------------------------------------------------------------------------------
# The mentor's replies reach the author
# ---------------------------------------------------------------------------------------
BRIEF = {"leaderboard": [{"seq": 75, "model": MUSE}],
         "recent": [{"seq": 131, "model": MUSE}, {"seq": 133, "model": QWEN}]}


def test_mentor_feedback_on_a_candidate_goes_to_its_author(runner):
    b = runner._mentor_reply("p1", MENTOR, "o1", {"candidate": 133, "to": MUSE, "text": "fix the dtype"}, BRIEF, [])
    assert b["meta"]["to"] == QWEN and b["meta"]["candidate_seq"] == 133 and "reply_to" not in b
    assert b["content"].startswith(f"@{QWEN} [candidate #133] fix the dtype") and b["channel"] == "team"


def test_a_candidate_number_in_reply_to_is_read_as_the_candidate(runner):
    # 09-30: {"reply_to": 106, "to": "agent"} -- 106 was a candidate; board message #106 was unrelated.
    b = runner._mentor_reply("p1", MENTOR, "o1", {"reply_to": 131, "to": "agent", "text": "mirror it"}, BRIEF, [])
    assert b["meta"]["to"] == MUSE and b["meta"]["candidate_seq"] == 131 and "reply_to" not in b


def test_a_reply_to_an_inbox_message_goes_to_its_sender(runner):
    inbox = [{"seq": 9001, "from": QWEN, "text": "why slope?"}]
    b = runner._mentor_reply("p1", MENTOR, "o1", {"reply_to": 9001, "to": "team", "text": "because"}, BRIEF, inbox)
    assert b["reply_to"] == b["meta"]["reply_to"] == 9001 and b["meta"]["to"] == QWEN


def test_invented_names_reach_nobody_and_known_short_names_resolve(runner):
    b = runner._mentor_reply("p1", MENTOR, "o1", {"to": "team member who ran candidate 131", "text": "x"}, BRIEF, [])
    assert "to" not in b["meta"] and not b["content"].startswith("@")
    b = runner._mentor_reply("p1", MENTOR, "o1", {"to": "qwen3.6-35b-a3b", "text": "x"}, BRIEF, [])
    assert b["meta"]["to"] == QWEN
    assert runner._mentor_reply("p1", MENTOR, "o1", {"candidate": 1, "text": " "}, BRIEF, []) is None


def test_the_mentor_sees_who_wrote_each_attempt_and_how_to_address_feedback(runner):
    brief = {"objective": {"title": "T", "description": "", "metric": {"price_column": "Close", "cost_bps": 2}},
             "metric_label": "Sharpe ratio", "habits": {}, "leaderboard": [],
             "recent": [{"seq": 131, "model": MUSE, "status": "ok", "in_sample": 1.04, "rationale": "breakout"}],
             "lessons": [], "ideas": [], "forecasts": [], "forecasters": []}
    p = runner.mentor_prompt(brief, [])
    assert f"#131 by {MUSE} ok" in p and '"candidate": <candidate number>' in p and "Do not invent names" in p


# ---------------------------------------------------------------------------------------
# Coaching and teammates in the brief
# ---------------------------------------------------------------------------------------
def test_coaching_is_read_whole_from_the_mentor_note(runner):
    long = "Stop filtering entries. " * 40
    entries = [msg(1, MENTOR, "Mentor notes -- ...\n\n[idea 5] IDEA: x\n\nCOACHING:\n" + long, channel="planning",
                   meta={"mentor": True}),
               msg(2, MENTOR, "Updated the team practices", channel="planning", meta={"practices": True})]
    got = runner._coaching_from(entries)
    assert got["by"] == MENTOR and got["text"] == long.strip()
    entries.append(msg(3, MENTOR, "Mentor notes", channel="planning", meta={"mentor": True, "coaching": "Keep #131."}))
    assert runner._coaching_from(entries)["text"] == "Keep #131."
    raw = [msg(4, MENTOR, 'Mentor notes\n\nCOACHING:\n{"directions": []}', channel="planning", meta={"mentor": True})]
    assert runner._coaching_from(raw) is None                                   # a failed JSON reply
    stale = [msg(5, MENTOR, "x", channel="planning", meta={"mentor": True, "coaching": "old"},
                 ts=time.time() - runner.COACHING_FRESH_S - 60)]
    assert runner._coaching_from(stale) is None


def test_teammates_include_the_twin_on_the_same_model_but_not_me_or_the_mentor(runner, monkeypatch):
    entries = [msg(1, MUSE, "Plan A (me)", channel="planning", meta={"agent": MUSE}),
               msg(2, MUSE, "Plan B (twin)", channel="planning", meta={"agent": f"{MUSE} #2"}),
               msg(3, MUSE, "Iteration note", channel="planning", author_id="a1"),
               msg(4, MENTOR, "Mentor notes", channel="planning", meta={"mentor": True}),
               msg(5, MENTOR, "Updated the team practices", channel="planning", meta={"practices": True}),
               msg(6, QWEN, "Plan C", channel="planning")]
    monkeypatch.setattr(runner, "request", lambda base, path, payload=None, **kw: {"entries": entries})
    got = _worker(runner).teammates()
    assert [(m["who"], m["text"]) for m in got] == [(f"{MUSE} #2", "Plan B (twin)"), (QWEN, "Plan C")]


def _ctx(**kw):
    ctx = {"objective": {"title": "T", "metric": {"kind": "sharpe", "price_column": "Close"}, "dataset": "bars"},
           "metric_label": "Sharpe ratio", "mode": "explore", "parent": None, "library": [], "datasets": ["bars"]}
    ctx.update(kw)
    return ctx


def test_the_brief_asks_for_links_and_offers_parents(runner):
    p = runner.iteration_prompt(_ctx(
        inbox=[{"seq": 75840, "from": MENTOR, "minutes_ago": 26, "text": "mirror it", "candidate": 131}],
        coaching={"by": MENTOR, "minutes_ago": 5, "text": "Stop filtering entries; build on #131."},
        promising=[{"seq": 114, "model": MUSE, "in_sample_score": 3.4, "rank": None, "rationale": "structural",
                    "problem": "one-sided: 21 long and 5 short trades in-sample."}],
        library=[{"name": "composite_skew_oinet_vwap", "kind": "signal", "version": 1, "author": MUSE,
                  "description": "composite", "used_by": 0, "ok": 0, "errors": 0, "lookahead_fails": 0,
                  "champions": 0, "best_in_sample": None, "comments": []}]))
    assert "#75840 from DeepSeek" in p and "about your candidate 131" in p and "reply_to=<the number>" in p
    assert "MENTOR COACHING (5 min ago" in p and "build on #131" in p
    assert "PROMISING" in p and "candidate 114 by Muse-Glimmer-30B-NVFP4: in-sample 3.400 -- not ranked: one-sided" in p
    assert "composite_skew_oinet_vwap [signal v1] by muse-glimmer-30b-nvfp4" in p
    assert "genuinely different" not in p and "`parent` to submit_candidate" in p
    assert "and the message(s) above you act on (reply_to=<number>)" in p


def test_an_unranked_parent_is_told_what_keeps_it_off_the_board(runner):
    p = runner.iteration_prompt(_ctx(mode="improve", parent={
        "id": "c114", "seq": 114, "model": MUSE, "rank": None, "in_sample_score": 3.4, "rationale": "r", "code": "x=1",
        "problem": "one-sided: 21 long and 5 short trades in-sample -- add the mirrored short entry."}))
    assert "IMPROVE candidate 114 by Muse-Glimmer-30B-NVFP4 (not ranked yet, in-sample 3.400)" in p
    assert "NOT RANKED: one-sided" in p


def test_submit_candidate_takes_a_parent(runner, monkeypatch):
    world = _world(runner, monkeypatch, lambda path, payload: None)
    sub = next(t for t in world.tools() if t["function"]["name"] == "submit_candidate")
    assert "parent" in sub["function"]["parameters"]["properties"]


# ---------------------------------------------------------------------------------------
# The control plane: parents while nothing ranked makes money; the library brief
# ---------------------------------------------------------------------------------------
@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "DB_PATH", tmp_path / "objectives.sqlite3")
    monkeypatch.setattr(O, "_conn", None)
    monkeypatch.setattr(O, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(external, "CONFIG_PATH", tmp_path / "external.json")
    monkeypatch.setattr(library, "_ready", False)
    now = time.time()
    O.db().execute(
        "INSERT INTO objectives (id, project_id, title, metric, split_date, created_at, updated_at) "
        "VALUES ('o1', 'p1', 'T', ?, '2024-07-19', ?, ?)",
        (json.dumps({"kind": "sharpe", "higher_is_better": True, "price_column": "Close", "cost_bps": 2}), now, now))
    O.db().commit()
    yield O.db()
    O.db().close()


def _cand(seq, score, is_score, note="", lookahead="pass", status="ok", audit="none", model=MUSE):
    O.db().execute(
        "INSERT INTO candidates (id, objective_id, seq, created_at, model, status, score, is_score, score_note, "
        "lookahead, audit, metrics, code, rationale) VALUES (?, 'o1', ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', 'x = 1', 'r')",
        (f"c{seq}", seq, time.time(), model, status, score, is_score, note, lookahead, audit))
    O.db().commit()


def test_promising_parents_are_positive_in_sample_runs_worth_fixing(db):
    _cand(75, -0.8, -0.8)                                       # the losing leader
    _cand(114, None, 3.4, "one-sided: 21 long and 5 short trades in-sample")
    _cand(131, None, 1.04, "one-sided", lookahead="fail")       # leaks: never a parent
    _cand(12, None, 0.43, "holdout: no score (the task server could not score that period)")
    _cand(106, -1.25, 0.71)
    _cand(107, None, 2.0, status="error")
    _cand(108, None, 5.0, audit="fail")                          # disqualified
    _cand(109, None, 3.4, "one-sided")                           # a clone of #114's in-sample score
    assert [c["seq"] for c in O._promising("o1")] == [114, 106]


def test_a_losing_board_still_hands_out_improve_iterations_on_promising_parents(db, monkeypatch):
    _cand(75, -0.8, -0.8)
    _cand(114, None, 3.4, "one-sided: 21 long and 5 short trades in-sample")
    monkeypatch.setattr(random, "random", lambda: 0.99)          # past the build and explore rolls
    ctx = asyncio.run(O.context("o1", QWEN))
    assert ctx["mode"] == "improve" and ctx["parent"]["seq"] == 114 and ctx["parent"]["rank"] is None
    assert ctx["parent"]["problem"].startswith("one-sided") and ctx["parent"]["model"] == MUSE
    assert [c["seq"] for c in ctx["promising"]] == [114]


def test_a_losing_board_with_nothing_promising_explores(db, monkeypatch):
    _cand(75, -0.8, -0.8)
    monkeypatch.setattr(random, "random", lambda: 0.99)
    ctx = asyncio.run(O.context("o1", QWEN))
    assert ctx["mode"] == "explore" and ctx["parent"] is None and ctx["promising"] == []


def _module(name, author, comments=(), best=None, uses=0):
    now = time.time()
    conn = library._db()
    conn.execute("INSERT INTO lib_modules (project_id, name, description, version, kind, author, created_at, updated_at) "
                 "VALUES ('p1', ?, 'd', 1, 'signal', ?, ?, ?)", (name, author, now, now))
    conn.execute("INSERT INTO lib_versions (project_id, name, version, code, author, ts, test_ok) "
                 "VALUES ('p1', ?, 1, 'x = 1', ?, ?, 1)", (name, author, now))
    for verdict in comments:
        conn.execute("INSERT INTO lib_comments (project_id, name, version, ts, author, verdict, text) "
                     "VALUES ('p1', ?, 1, ?, 'm', ?, ?)", (name, now, verdict, f"{verdict} {random.random()}"))
    for i in range(uses):
        _cand(500 + len(name) * 10 + i, None, best, "")
        conn.execute("INSERT INTO lib_usage (candidate_id, project_id, name, version) VALUES (?, 'p1', ?, 1)",
                     (f"c{500 + len(name) * 10 + i}", name))
    conn.commit()


def test_the_library_brief_lists_working_modules_first_with_their_author(db):
    _module("pivot_leg_signal_v3", MUSE, comments=["broken"], best=-3.0, uses=1)
    _module("composite_skew_oinet_vwap", MUSE, comments=["works"] * 3 + ["broken"])
    _module("vwap_gap_reversion", QWEN)
    got = library.brief("p1")
    assert [m["name"] for m in got] == ["composite_skew_oinet_vwap", "vwap_gap_reversion", "pivot_leg_signal_v3"]
    assert got[0]["author"] == MUSE


def test_a_resubmitted_strategy_is_named_as_a_clone(db):
    """#149/#151/#153 (10-02) all scored 4.616659 in-sample -- #142's sparse strategy resubmitted from three
    different parents, each told only its score."""
    _cand(142, None, 4.616659, "holdout: no score")
    _cand(140, -1.598, 2.598)
    assert O._clone_of("o1", "c153", 4.616659, None) == 142
    assert O._clone_of("o1", "c153", 4.616659, -1.0) is None        # same in-sample, different holdout: not a clone
    assert O._clone_of("o1", "c153", 2.5, -1.598) is None
    assert O._clone_of("o1", "c142", 4.616659, None) is None        # not a clone of itself
