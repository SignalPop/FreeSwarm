"""/mb/team/threads and /mb/team/counts: the messages behind the Team panel's sent / answered /
unanswered / expired counts (definitions: app/team_threads.py)."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from app import msgboard
from app import team_threads as tt

A = "org/Qwen-X"          # the model under inspection (short name "Qwen-X")
B = "DeepSeek-Y"
C = "Muse-Z"
PID = "p1"


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setattr(msgboard, "DB_PATH", tmp_path / "mb.sqlite3")
    monkeypatch.setattr(msgboard, "_conn", None)
    monkeypatch.setattr(msgboard, "resolve_project", lambda explicit=None: explicit or PID)
    msgboard.app.dependency_overrides[msgboard.require_agent] = lambda: None
    state = {"ts": time.time() - 3600}   # an hour ago: inside every answering window

    def post(channel, author, content, meta=None, author_id=None, reply_to=None, project=PID):
        state["ts"] += 10
        with msgboard._db_lock:
            cur = msgboard.db().execute(
                "INSERT INTO messages (channel,author,author_id,kind,content,meta,reply_to,ts,project_id) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (channel, author, author_id, "chat", content, json.dumps(meta or {}), reply_to, state["ts"], project))
            msgboard.db().commit()
        return cur.lastrowid

    yield post
    msgboard.app.dependency_overrides.pop(msgboard.require_agent, None)
    if msgboard._conn is not None:
        msgboard._conn.close()


def _collab(sent, answered, inbox, **extra):
    return {"agent": A, "mode": "improve", "candidate": 7, "built_on": None, "reused": [], "contributed": [],
            "messages_sent": sent, "answered": answered, "inbox": inbox, "teammates_seen": [], "note": "", **extra}


def _scenario(post):
    s = {}
    # ---- iteration 1: one question in the inbox, answered in the same iteration
    s["q0"] = post("team", B, "@Qwen-X what threshold did you use?", {"to": A, "team": True, "objective_id": "o1"})
    post("general", A, "Iteration on \"obj\": exploring", {"objective_id": "o1"}, author_id="aid-a")
    s["r0"] = post("team", A, "@DeepSeek-Y z>=1.0 with hold 48", {"to": B, "team": True, "reply_to": s["q0"]},
                   reply_to=s["q0"])
    s["plan1"] = post("planning", A, "Plan: try the Imb forecast with a 3-bar hold", {"team": True})
    s["rec1"] = post("team", A, "record 1", {"objective_id": "o1", "candidate_id": "cid7", "collab": _collab(
        [{"to": B, "channel": "team", "reply_to": s["q0"], "text": "z>=1.0 with hold 48"},
         {"to": "all", "channel": "planning", "reply_to": None, "text": "Plan: try the Imb forecast with a 3-bar hold"}],
        [{"seq": s["q0"], "from": B, "channel": "team", "minutes_ago": 1, "text": "what threshold"}], 1)}, author_id="aid-a")
    # ---- a mentor pass reads (and so consumes) this one
    post("team", C, "@qwen-x consumed by the mentor pass", {"team": True})
    post("general", A, "Mentoring: reading the team's results (due)", {}, author_id="aid-a")
    # ---- iteration 2: two questions, neither answered; a second worker of the same model
    #      posts its own marker in between, which must not shorten this worker's window
    s["q1"] = post("team", B, "@Qwen-X can you retest candidate #7 with costs?", {"team": True, "objective_id": "o1"})
    post("general", A, "Iteration on \"obj\": other worker", {}, author_id="aid-b")
    s["q2"] = post("results", C, "hand-over for you", {"to": A, "objective_id": "o1"})
    post("team", C, "@DeepSeek-Y not for Qwen", {"to": B})
    post("general", A, "Iteration on \"obj\": improving #7", {}, author_id="aid-a")
    s["rec2"] = post("team", A, "record 2", {"objective_id": "o1", "collab": _collab([], [], 2)}, author_id="aid-a")
    s["late"] = post("team", A, "@DeepSeek-Y done, see #7", {"to": B, "reply_to": s["q1"]}, reply_to=s["q1"])
    # ---- iteration 3: a newer runner record naming its inbox
    s["q3"] = post("team", B, "@Qwen-X ping", {"to": A, "team": True})
    post("general", A, "Iteration on \"obj\": exploring", {}, author_id="aid-a")
    s["dm"] = post("team", A, "@DeepSeek-Y trying your idea next", {"to": B, "team": True})
    s["rec3"] = post("team", A, "record 3", {"objective_id": "o1", "collab": _collab(
        [{"to": B, "channel": "team", "reply_to": None, "text": "trying your idea next"}], [], 1,
        inbox_seqs=[s["q3"]])}, author_id="aid-a")
    # another model's record and another project's noise must not count
    post("team", B, "B's record", {"collab": {**_collab([{"to": "all"}], [], 3), "agent": B}})
    post("team", B, "@Qwen-X other project", {"to": A}, project="p2")
    return s


KINDS = ("sent", "answered", "unanswered", "expired")


def _check(doc):
    for k in KINDS:
        assert doc["counts"][k] == len(doc[k]), k
    assert doc["unlocated"] == {k: 0 for k in KINDS}


def test_counts_equal_list_lengths(board):
    s = _scenario(board)
    doc = TestClient(msgboard.app).get("/mb/team/threads", params={"agent": A}).json()
    assert doc["counts"] == {"iterations": 3, "sent": 3, "answered": 2, "unanswered": 2, "expired": 0}
    _check(doc)
    # unanswered = messages to A that some iteration read and no agent of A has answered, newest first
    assert [m["seq"] for m in doc["unanswered"]] == [s["q3"], s["q2"]]
    assert all(m["reply"] is None for m in doc["unanswered"])
    q2 = doc["unanswered"][1]
    assert q2["from"] == C and q2["channel"] == "results" and q2["iteration"]["record_seq"] == s["rec2"]
    # answered: in the iteration that read it (q0), or later -- q1 was answered after its record
    assert [m["seq"] for m in doc["answered"]] == [s["q1"], s["q0"]]
    q1, q0 = doc["answered"]
    assert q1["reply"]["seq"] == s["late"] and "retest" in q1["text"] and q1["from"] == B
    assert q0["reply"]["seq"] == s["r0"] and q0["iteration"]["record_seq"] == s["rec1"]
    # sent: matched to the board posts, full text
    assert [m["seq"] for m in doc["sent"]] == [s["dm"], s["plan1"], s["r0"]]
    assert doc["sent"][-1]["to"] == B


def test_through_limits_the_window(board):
    s = _scenario(board)
    doc = TestClient(msgboard.app).get("/mb/team/threads", params={"agent": A, "through": s["rec2"]}).json()
    assert doc["counts"] == {"iterations": 2, "sent": 2, "answered": 2, "unanswered": 1, "expired": 0}
    _check(doc)


def test_counts_route_matches_threads(board):
    s = _scenario(board)
    client = TestClient(msgboard.app)
    got = client.get("/mb/team/counts").json()
    assert set(got["agents"]) == {A, B}
    assert got["agents"][A] == client.get("/mb/team/threads", params={"agent": A}).json()["counts"]
    assert got["agents"][B]["iterations"] == 1 and got["agents"][B]["sent"] == 1
    assert got["through"] > s["rec3"]
    capped = client.get("/mb/team/counts", params={"through": s["rec2"]}).json()
    assert set(capped["agents"]) == {A} and capped["agents"][A]["iterations"] == 2


def test_unknown_agent_is_empty(board):
    _scenario(board)
    doc = TestClient(msgboard.app).get("/mb/team/threads", params={"agent": "nobody"}).json()
    assert doc["counts"] == {"iterations": 0, "sent": 0, "answered": 0, "unanswered": 0, "expired": 0}
    assert doc["sent"] == doc["answered"] == doc["unanswered"] == doc["expired"] == []


# ---- the counting rules, over hand-built rows ------------------------------------------------
NOW = 1_000_000.0
H = 3600.0


def _row(seq, author, content="", meta=None, *, ago=H, channel="team", reply_to=None, author_id=None):
    return {"seq": seq, "ts": NOW - ago, "author": author, "author_id": author_id, "channel": channel,
            "content": content, "meta": meta or {}, "reply_to": reply_to}


def _rec(seq, inbox_seqs, answered=(), agent_id="aid-a", ago=0.5 * H, agent=None):
    return _row(seq, A, "record", {"agent": agent or A, "collab": _collab(
        [], [{"seq": n} for n in answered], len(inbox_seqs), inbox_seqs=list(inbox_seqs))},
        ago=ago, author_id=agent_id)


def _doc(rows):
    team = [r for r in rows if r["channel"] == "team"]
    return tt.threads(team, rows, A, now=NOW)


def test_a_message_read_by_both_agents_counts_once():
    rows = [_row(1, B, "@Qwen-X look", {"to": A}),
            _rec(2, [1], agent_id="aid-a"), _rec(3, [1], agent_id="aid-b", agent=f"{A} #2")]
    doc = _doc(rows)
    assert doc["counts"]["unanswered"] == 1
    (m,) = doc["unanswered"]
    assert m["read_in"] == 2 and m["iteration"]["record_seq"] == 3 and m["iteration"]["agent"] == f"{A} #2"


def test_an_answer_by_the_other_agent_anywhere_on_the_board_counts():
    # agent 1 read #1 and left it; agent #2 answered it later through meta.answers (no reply_to),
    # and #3 by reply_to -- both outside the iteration that read them.
    rows = [_row(1, B, "@Qwen-X a", {"to": A}), _row(2, C, "@Qwen-X b", {"to": A}),
            _rec(3, [1, 2]),
            _row(4, A, "re #1 -- accept: will do", {"agent": f"{A} #2", "answers": [1], "feedback_reply": "accept"},
                 ago=0.2 * H),
            _row(5, A, "@Muse-Z re b", {"agent": A, "reply_to": 2}, reply_to=2, ago=0.1 * H)]
    doc = _doc(rows)
    assert doc["counts"]["answered"] == 2 and doc["counts"]["unanswered"] == 0
    by = {m["seq"]: m for m in doc["answered"]}
    assert by[1]["reply"]["seq"] == 4 and by[1]["reply"]["agent"] == f"{A} #2"
    assert by[2]["reply"]["seq"] == 5


def test_a_teammates_reply_does_not_answer_it():
    rows = [_row(1, B, "@Qwen-X q", {"to": A}), _rec(2, [1]),
            _row(3, C, "I can answer that", {"reply_to": 1}, reply_to=1, ago=0.1 * H)]
    doc = _doc(rows)
    assert doc["counts"]["unanswered"] == 1 and doc["unanswered"][0]["reply"] is None


@pytest.mark.parametrize("ago,reason", [(25 * H, "age"), (7 * H, "reread"), (5 * H, None)])
def test_old_messages_expire(ago, reason):
    rows = [_row(1, B, "@Qwen-X old", {"to": A}, ago=ago), _rec(2, [1], ago=min(ago, 4 * H) - 0.1 * H)]
    doc = _doc(rows)
    if reason is None:
        assert doc["counts"]["unanswered"] == 1 and doc["counts"]["expired"] == 0
    else:
        assert doc["counts"]["unanswered"] == 0 and doc["counts"]["expired"] == 1
        assert doc["expired"][0]["expired"]["reason"] == reason
        assert doc["expired"][0]["expired"]["text"] == tt.EXPIRED_WHY[reason]


def test_an_answer_still_counts_after_the_window():
    rows = [_row(1, B, "@Qwen-X q", {"to": A}, ago=30 * H), _rec(2, [1], ago=29 * H),
            _row(3, A, "re #1", {"reply_to": 1}, reply_to=1, ago=28 * H)]
    assert _doc(rows)["counts"] == {"iterations": 1, "sent": 0, "answered": 1, "unanswered": 0, "expired": 0}


def test_acknowledgements_and_closed_threads_expire():
    rows = [_row(1, A, "re #0 -- accept: ok", {"feedback_reply": "accept", "feedback_depth": 1}, ago=2 * H),
            _row(2, A, "re #9 -- reject: no", {"feedback_reply": "reject", "feedback_depth": 2}, ago=2 * H),
            _row(3, B, "accept: thanks", {"to": A, "feedback_reply": "accept"}, reply_to=1),
            _row(4, B, "but why?", {"to": A}, reply_to=2),           # replies to an answer at depth 2
            _row(5, B, "and this?", {"to": A}, reply_to=1),          # replies to an answer at depth 1: open
            _rec(6, [3, 4, 5])]
    doc = _doc(rows)
    assert {m["seq"]: m["expired"]["reason"] for m in doc["expired"]} == {3: "ack", 4: "depth"}
    assert [m["seq"] for m in doc["unanswered"]] == [5]


def test_unrebuildable_old_records_are_counted_by_age():
    # Records from an older runner whose iteration note is gone: counted, reported as unlocated --
    # unanswered while the record is recent, expired once it is older than the re-read window.
    team = [{"seq": 5, "ts": NOW - 30 * H, "author": A, "channel": "team", "content": "r",
             "meta": {"collab": _collab([], [], 2)}},
            {"seq": 6, "ts": NOW - 1 * H, "author": A, "channel": "team", "content": "r",
             "meta": {"collab": _collab([], [], 1)}}]
    doc = tt.threads(team, [], A, now=NOW)
    assert doc["counts"]["expired"] == 2 and doc["counts"]["unanswered"] == 1
    assert doc["expired"] == doc["unanswered"] == []
    assert doc["unlocated"]["expired"] == 2 and doc["unlocated"]["unanswered"] == 1


def test_summary_counts_every_model():
    rows = [_row(1, B, "@Qwen-X q", {"to": A}), _rec(2, [1]),
            _row(3, B, "B's record", {"collab": {**_collab([{"to": "all"}], [], 0, inbox_seqs=[]), "agent": B}})]
    got = tt.summary([r for r in rows if r["channel"] == "team"], rows, now=NOW)
    assert got[A] == {"iterations": 1, "sent": 0, "answered": 0, "unanswered": 1, "expired": 0}
    assert got[B] == {"iterations": 1, "sent": 1, "answered": 0, "unanswered": 0, "expired": 0}


def test_rules_match_the_runner():
    import swarm_runner as R

    assert tt.FEEDBACK_MAX_AGE_S == R.FEEDBACK_MAX_AGE_S
    assert tt.FEEDBACK_MAX_DEPTH == R.FEEDBACK_MAX_DEPTH
    assert tt.INBOX_WINDOW_S == R.INBOX_WINDOW_S
