"""/mb/team/threads: the messages behind the Team panel's sent / answered / unanswered counts."""

from __future__ import annotations

import json

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
    state = {"ts": 10_000.0}

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


def _check(doc):
    assert doc["counts"]["sent"] == len(doc["sent"])
    assert doc["counts"]["answered"] == len(doc["answered"])
    assert doc["counts"]["unanswered"] == len(doc["unanswered"])
    assert doc["unlocated"] == {"sent": 0, "answered": 0, "unanswered": 0}


def test_counts_equal_list_lengths(board):
    s = _scenario(board)
    doc = TestClient(msgboard.app).get("/mb/team/threads", params={"agent": A}).json()
    assert doc["counts"] == {"iterations": 3, "sent": 3, "answered": 1, "unanswered": 3}
    _check(doc)
    # unanswered = teammates' messages to A that A read and did not reply to that iteration, newest first
    assert [m["seq"] for m in doc["unanswered"]] == [s["q3"], s["q2"], s["q1"]]
    assert all(m["reply"] is None for m in doc["unanswered"])
    q1 = next(m for m in doc["unanswered"] if m["seq"] == s["q1"])
    assert q1["later_reply"]["seq"] == s["late"]          # answered later, outside the iteration
    assert q1["from"] == B and q1["channel"] == "team" and "retest" in q1["text"]
    assert q1["iteration"]["record_seq"] == s["rec2"]
    # answered: the question with the reply under it
    (a,) = doc["answered"]
    assert a["seq"] == s["q0"] and a["reply"]["seq"] == s["r0"]
    # sent: matched to the board posts, full text
    assert [m["seq"] for m in doc["sent"]] == [s["dm"], s["plan1"], s["r0"]]
    assert doc["sent"][-1]["to"] == B


def test_through_limits_the_window(board):
    s = _scenario(board)
    doc = TestClient(msgboard.app).get("/mb/team/threads", params={"agent": A, "through": s["rec2"]}).json()
    assert doc["counts"] == {"iterations": 2, "sent": 2, "answered": 1, "unanswered": 2}
    _check(doc)


def test_unknown_agent_is_empty(board):
    _scenario(board)
    doc = TestClient(msgboard.app).get("/mb/team/threads", params={"agent": "nobody"}).json()
    assert doc["counts"] == {"iterations": 0, "sent": 0, "answered": 0, "unanswered": 0}
    assert doc["sent"] == doc["answered"] == doc["unanswered"] == []


def test_unrebuildable_inbox_is_reported_not_invented():
    # A record whose iteration note is gone: the count stands, the gap is reported.
    team = [{"seq": 5, "ts": 1.0, "author": A, "channel": "team", "content": "r",
             "meta": {"collab": _collab([], [], 2)}}]
    doc = tt.threads(team, [], A)
    assert doc["counts"]["unanswered"] == 2
    assert doc["unanswered"] == [] and doc["unlocated"]["unanswered"] == 2


def test_counts_match_the_panel_arithmetic():
    # max(0, inbox - answered) per record, as TeamPanel.tsx totals it
    recs = [{"meta": {"collab": _collab([{}], [{"seq": 1}, {"seq": 2}], 1)}},
            {"meta": {"collab": _collab([], [], 3)}}]
    assert tt.counts(recs) == {"iterations": 2, "sent": 1, "answered": 2, "unanswered": 3}
