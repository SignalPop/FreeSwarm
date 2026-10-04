"""The work log: iteration records are archived as the inspector flushes them (and backfilled
once from its file), pruned after the retention window, and served with the window's
candidates -- summarized, grouped by error, filtered -- for the Work page."""

from __future__ import annotations

import json
import sqlite3
import time

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from app import agent_activity as A
from app import work as W

TB = ('Traceback (most recent call last):\n  File "script.py", line 9, in <module>\n    exec(...)\n'
      '  File "candidate.py", line 12, in <module>\n    ex = trend_exits(df, atr_lookback=14)\n'
      "TypeError: trend_exits() got an unexpected keyword argument 'atr_lookback'\n")
LINE = "TypeError: trend_exits() got an unexpected keyword argument 'atr_lookback'"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "DB_PATH", tmp_path / "agent_activity.sqlite3")
    monkeypatch.setattr(A, "_conn", None)
    monkeypatch.setattr(W, "DB_PATH", None)        # follows A.DB_PATH: tmp_path / work.sqlite3
    monkeypatch.setattr(W, "OBJECTIVES_DB", tmp_path / "objectives.sqlite3")
    A.reset()
    W.reset()
    yield tmp_path
    W.reset()
    if A._conn is not None:
        A._conn.close()
    A.reset()


@pytest.fixture
def client(store):
    app = FastAPI()
    api = APIRouter(prefix="/api")
    api.include_router(A.router)
    api.include_router(W.router)
    app.include_router(api)
    return TestClient(app)


def tool(name, ok=True, result=None, at=None, **args):
    return {"kind": "tool", "at": at or time.time(), "name": name, "args": args, "ok": ok, "seconds": 1.2,
            "result": result if result is not None else '{"ok": true, "stdout": "fine"}'}


def rec(rid, started, status="running", timeline=(), submissions=(), **extra):
    return {"id": rid, "mode": "explore", "status": status, "started_at": started,
            "ended_at": None if status == "running" else started + 60, "objective": {"id": "o1", "title": "Best strategy"},
            "asked": [{"at": started, "model": "m1", "system": "S" * 50_000, "system_chars": 50_000, "prompt": "P" * 20_000,
                       "prompt_chars": 20_000, "tools": ["run_python"]}],
            "timeline": list(timeline), "chats": [], "submissions": list(submissions),
            "tokens": {"prompt": 10, "completion": 5, "chats": 1}, **extra}


def post(agent, record, project="p1", model=None):
    return A.record({"agent": agent, "model": model or agent, "project_id": project, "record": record})


def archived(store) -> dict[str, sqlite3.Row]:
    with W._lock:
        return {r["id"]: r for r in W.db().execute("SELECT * FROM iterations").fetchall()}


# ---------------------------------------------------------------------------------------
# Reading errors
# ---------------------------------------------------------------------------------------
def test_error_line_is_the_last_exception_of_the_last_traceback():
    assert W.error_line(TB) == LINE
    # A run_python result the runner cut head-and-tail no longer parses as JSON: still found.
    cut = json.dumps({"ok": False, "stdout": "x" * 50, "stderr": TB})[:40] + " ...[cut]... " + json.dumps(TB)[-200:]
    assert W.error_line(W._as_text(cut)) == LINE
    assert W.error_line("polars.exceptions.ColumnNotFoundError: unable to find column \"High\"") .startswith(
        "polars.exceptions.ColumnNotFoundError")
    assert W.error_line("experiment budget used (8 runs this iteration).\nmore") == "experiment budget used (8 runs this iteration)."


def test_tool_failures_include_crashed_scripts_and_rejected_saves():
    assert W.tool_failed(tool("run_python", result=json.dumps({"ok": False, "stderr": TB})))
    assert W.tool_failed(tool("library_save", result=json.dumps({"saved": False, "test_output": TB})))
    assert W.tool_failed(tool("query_data", ok=False, result='{"error": "no such column"}'))
    assert not W.tool_failed(tool("run_python"))


# ---------------------------------------------------------------------------------------
# Archiving
# ---------------------------------------------------------------------------------------
def test_flush_archives_every_iteration_beyond_what_the_inspector_keeps(store, monkeypatch):
    monkeypatch.setattr(A, "KEEP_RECORDS", 2)
    now = time.time()
    for i in range(4):
        post("m1", rec(f"r{i}", now - 400 + i * 60, status="no submission"))
    A.flush()
    rows = archived(store)
    assert set(rows) >= {"r2", "r3"}
    # r0 and r1 left the inspector before this flush; later ones are kept once archived.
    post("m1", rec("r4", now - 100, status="no submission"))
    post("m1", rec("r5", now - 50, status="no submission"))
    A.flush()
    assert {"r2", "r3", "r4", "r5"} <= set(archived(store))
    assert len(A.snapshot()[0]["records"]) == 2


def test_upsert_replaces_by_record_id_and_clips_prompts(store):
    now = time.time()
    post("m1", rec("r1", now - 60))
    A.flush()
    post("m1", rec("r1", now - 60, status="submitted", outcome="#7 ok",
                   submissions=[{"candidate_id": "c7", "seq": 7, "status": "ok", "at": now, "rationale": "gex trend"}]))
    A.flush()
    rows = archived(store)
    assert len(rows) == 1
    r = rows["r1"]
    assert (r["status"], r["outcome"], r["candidate_ids"]) == ("submitted", "ok", "|c7|")
    doc = W.iteration("r1")
    asked = doc["record"]["asked"][0]
    assert len(asked["system"]) <= W.SYSTEM_CHARS and asked["system_chars"] == 50_000
    assert len(asked["prompt"]) <= W.PROMPT_CHARS
    assert doc["summary"]["hypothesis"] == "gex trend"


def test_archive_failure_never_fails_the_post(store, client, monkeypatch):
    def boom():
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(W, "db", boom)
    monkeypatch.setattr(A, "_last_flush", 0.0)     # this post flushes (and so archives)
    r = client.post("/api/agents/activity", json={"agent": "m1", "model": "m1", "project_id": "p1",
                                                    "record": rec("r1", time.time())})
    assert r.status_code == 200 and r.json()["ok"]
    assert A.archive_changed() == 0                  # already marked; and a fresh change...
    post("m1", rec("r1", time.time(), status="no submission"))
    assert A.archive_changed() == 0                  # ...is swallowed too


def test_backfill_copies_the_inspector_file_once(store):
    now = time.time()
    con = sqlite3.connect(A.DB_PATH)
    con.execute("CREATE TABLE agents (key TEXT PRIMARY KEY, doc TEXT NOT NULL, updated_at REAL NOT NULL)")
    doc = {"agent": "old", "model": "old", "role": "search", "project_id": "p1", "updated_at": now,
           "records": [rec("b1", now - 3000, status="no submission"), rec("b2", now - 2000, status="interrupted")]}
    con.execute("INSERT INTO agents VALUES (?, ?, ?)", ("p1|old", json.dumps(doc), now))
    con.commit()
    con.close()
    rows = archived(store)
    assert set(rows) == {"b1", "b2"} and rows["b2"]["outcome"] == "interrupted"
    assert json.loads(W.db().execute("SELECT value FROM meta WHERE key='backfilled'").fetchone()[0])["records"] == 2


def test_old_iterations_are_pruned(store):
    old = time.time() - W.RETENTION_S - 86400
    W.archive([({"agent": "m1", "model": "m1", "project_id": "p1"}, [rec("ancient", old, status="no submission")])])
    assert "ancient" not in archived(store)
    W.archive([({"agent": "m1", "model": "m1", "project_id": "p1"}, [rec("fresh", time.time() - 60, status="done")])])
    assert "fresh" in archived(store)


# ---------------------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------------------
def make_objectives(path, now):
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE objectives (id TEXT PRIMARY KEY, project_id TEXT, title TEXT, metric TEXT);
        CREATE TABLE candidates (id TEXT PRIMARY KEY, objective_id TEXT, seq INTEGER, created_at REAL, model TEXT,
            mode TEXT, parent_id TEXT, rationale TEXT, code TEXT, status TEXT, score REAL, is_score REAL,
            score_note TEXT, metrics TEXT, lookahead TEXT, lookahead_detail TEXT, audit TEXT, audit_notes TEXT,
            stdout TEXT, stderr TEXT, eval_seconds REAL, idea_id INTEGER);
    """)
    con.executemany("INSERT INTO objectives VALUES (?, ?, ?, ?)", [
        ("o1", "p1", "Best strategy", json.dumps({"kind": "sharpe", "higher_is_better": True})),
        ("o2", "p2", "Other project", json.dumps({"kind": "sharpe"}))])
    cand = "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, 'explore', ?, ?, 'import ft', ?, ?, ?, ?, ?, ?, '', NULL, '', ?, ?, ?, NULL)"
    metrics = json.dumps({"in_sample": {"sharpe": 1.5}, "holdout": {"sharpe": 2.1}, "rank": {"holdout": 2.1}})
    con.executemany(cand, [
        ("c1", "o1", 1, now - 3000, "m1", None, "momentum", "ok", 1.2, 1.5, "", metrics, "pass", "out", "", 3.0),
        ("c2", "o1", 2, now - 2000, "m1", "c1", "trend exits", "error", None, None, "the script failed -- see stderr",
         "{}", "skipped", "", TB, 2.5),
        ("c3", "o1", 3, now - 1000, "m2", None, "leaky", "ok", 9.9, 9.9, "", metrics, "fail", "", "", 4.0),
        ("c4", "o1", 4, now - 90_000, "m1", None, "yesterday", "ok", 5.0, 5.0, "", metrics, "pass", "", "", 1.0),
        ("c9", "o2", 1, now - 500, "m1", None, "elsewhere", "ok", 7.0, 7.0, "", metrics, "pass", "", "", 1.0)])
    con.commit()
    con.close()


@pytest.fixture
def seeded(store, client):
    now = time.time()
    make_objectives(W.OBJECTIVES_DB, now)
    crash = json.dumps({"ok": False, "stdout": "", "stderr": TB})
    post("m1", rec("i1", now - 2100, status="submitted", outcome="#2 error",
                   # the crash is its last experiment, so not one a later run fixed (recovered)
                   timeline=[tool("run_python", at=now - 2090, code="print(1)"),
                             tool("run_python", result=crash, at=now - 2080, code="x = trend_exits(df, atr_lookback=14)"),
                             # an errored submission's result is the runner's JSON text, not a dict
                             tool("submit_candidate", result=json.dumps({"candidate_id": "c2", "seq": 2, "status": "error",
                                                                         "stderr_tail": TB}), at=now - 2050),
                             {"kind": "message", "at": now - 2040, "text": "Call submit_candidate NOW"}],
                   submissions=[{"candidate_id": "c2", "seq": 2, "status": "error", "at": now - 2050, "rationale": "trend exits"}]))
    post("m2 #2", rec("i2", now - 1500, status="no submission",
                      timeline=[tool("library_save", result=json.dumps({"saved": False, "test_output": TB}), at=now - 1490)],
                      chats=[{"at": now - 1480, "model": "m2", "seconds": 3, "finish": "length", "error": None},
                             {"at": now - 1470, "model": "m2", "seconds": 1, "finish": None,
                              "error": "/v1/chat/completions -> 429: spending cap"}]), model="m2")
    post("m1", rec("i3", now - 100))                       # running now, not flushed: the endpoint copies it in
    post("x", rec("i9", now - 100, status="done"), project="p2")
    A.flush()
    post("m1", rec("i3", now - 100, timeline=[tool("team_board", at=now - 90)]))
    return now


def test_window_summary_groups_errors_and_finds_the_best(seeded, client):
    doc = client.get("/api/projects/p1/work", params={"hours": 24}).json()
    s = doc["summary"]
    assert s["iterations"] == 3 and s["by_outcome"] == {"error": 1, "no_submission": 1, "running": 1}
    assert (s["candidates"], s["candidates_ok"], s["candidates_error"]) == (3, 2, 1)
    assert (s["experiments"], s["experiments_failed"]) == (2, 1)
    assert s["chat_errors"] == 1 and s["truncations"] == 1
    top = s["errors"][0]
    # One error, three places: a run_python experiment, a library_save test, the candidate itself.
    assert top["line"] == LINE and top["count"] == 3
    assert top["sources"] == {"run_python": 1, "library_save": 1, "candidate": 1}
    # The leaky #3 scored higher but failed the look-ahead test; #4 is outside the window.
    (best,) = s["best"]
    assert (best["seq"], best["score"], best["holdout"]) == (1, 1.2, 2.1)
    assert doc["counts"] == {"all": 6, "iteration": 3, "candidate": 3}
    items = doc["items"]
    assert [i["at"] for i in items] == sorted((i["at"] for i in items), reverse=True)
    assert items[0]["id"] == "i3" and items[0]["tool_calls"] == 1    # the live update, archived on read
    c2 = next(i for i in items if i["id"] == "c2")
    assert c2["error_line"] == LINE and c2["agent"] == "m1" and c2["parent_seq"] == 1
    i1 = next(i for i in items if i["id"] == "i1")
    assert i1["followups"] == 1 and i1["outcome_text"] == "#2 error"
    # submit_candidate's own failure is the candidate row's, not counted twice as a tool error line.
    assert [e["tool"] for e in i1["errors"]] == ["run_python"]


def test_filters_and_paging(seeded, client):
    get = lambda **p: client.get("/api/projects/p1/work", params={"hours": 24, **p}).json()  # noqa: E731
    assert {i["id"] for i in get(outcome="error")["items"]} == {"i1", "c2"}
    assert {i["id"] for i in get(outcome="no_submission")["items"]} == {"i2"}
    assert {i["id"] for i in get(outcome="ok")["items"]} == {"c1", "c3"}
    assert {i["id"] for i in get(outcome="tool_errors")["items"]} == {"i1", "i2", "c2"}
    assert {i["id"] for i in get(kind="candidate")["items"]} == {"c1", "c2", "c3"}
    assert {i["id"] for i in get(agent="m2 #2")["items"]} == {"i2"}
    assert {i["id"] for i in get(agent="m2")["items"]} == {"i2", "c3"}           # a model matches its agents' rows
    assert {i["id"] for i in get(q="ATR_LOOKBACK")["items"]} == {"i1", "i2", "c2"}
    assert {i["id"] for i in get(q="#3")["items"]} == {"c3"}
    page = get(limit=2)
    assert page["total"] == 6 and len(page["items"]) == 2
    assert get(hours=200)["counts"]["candidate"] == 4                              # yesterday's #4 too
    assert client.get("/api/projects/p1/work", params={"outcome": "bogus"}).status_code == 422
    other = client.get("/api/projects/p2/work").json()
    assert {i["id"] for i in other["items"]} == {"i9", "c9"}


def test_detail_endpoints(seeded, client):
    it = client.get("/api/work/iterations/i1").json()
    tools = [e for e in it["record"]["timeline"] if e["kind"] == "tool"]
    assert [bool(e.get("failed")) for e in tools] == [False, True, True]
    assert [e.get("recovered") for e in tools] == [None, False, False]
    assert tools[1]["error_line"] == LINE and "atr_lookback=14" in tools[1]["error_tail"]
    c = client.get("/api/work/candidates/c2").json()
    assert c["error_line"] == LINE and c["stderr"].endswith("'atr_lookback'\n") and c["iterations"] == ["i1"]
    assert c["code"] == "import ft" and c["parent_seq"] == 1
    ok = client.get("/api/work/candidates/c1").json()
    assert ok["metrics"]["holdout"]["sharpe"] == 2.1 and ok["holdout"] == 2.1
    assert client.get("/api/work/iterations/nope").status_code == 404
    assert client.get("/api/work/candidates/nope").status_code == 404


# ---------------------------------------------------------------------------------------
# Clearing: a view marker, nothing deleted
# ---------------------------------------------------------------------------------------
def test_clear_hides_earlier_work_but_keeps_running_iterations(seeded, client):
    get = lambda **p: client.get("/api/projects/p1/work", params={"hours": 24, **p}).json()  # noqa: E731
    assert get()["cleared_at"] is None and not get()["hiding_cleared"]
    r = client.post("/api/projects/p1/work/clear")
    assert r.status_code == 200
    at = r.json()["cleared_at"]
    assert abs(at - time.time()) < 5
    doc = get()
    assert doc["cleared_at"] == at and doc["hiding_cleared"] and doc["since"] == at
    # i3 started before the clear but is still running: ongoing work, so it stays.
    assert {i["id"] for i in doc["items"]} == {"i3"}
    assert doc["counts"] == {"all": 1, "iteration": 1, "candidate": 0}
    s = doc["summary"]
    assert s["iterations"] == 1 and s["by_outcome"] == {"running": 1} and s["candidates"] == 0 and s["errors"] == []
    # New work after the clear shows up.
    post("m3", rec("i4", time.time() + 1, status="no submission"))
    A.flush()
    assert {i["id"] for i in get()["items"]} == {"i3", "i4"}
    # Nothing was deleted: the archive and "show all" still have everything.
    assert {"i1", "i2", "i3", "i4"} <= set(archived(seeded))
    full = get(include_cleared=1)
    assert full["cleared_at"] == at and not full["hiding_cleared"] and full["counts"]["all"] == 7
    # The marker is per project, and survives a reconnect (it lives in work.sqlite3).
    assert client.get("/api/projects/p2/work").json()["counts"]["all"] == 2
    W.reset()
    assert W.cleared_at("p1") == at and W.cleared_at("p2") is None
    # Undo.
    assert client.delete("/api/projects/p1/work/clear").json() == {"cleared_at": None}
    doc = get()
    assert doc["cleared_at"] is None and doc["counts"]["all"] == 7


def test_clear_keeps_iterations_that_ended_after_it_and_newer_candidates(seeded, client):
    W.set_cleared("p1", seeded - 1470)       # i2 ran from -1500 to -1440: it straddles the marker
    doc = client.get("/api/projects/p1/work", params={"hours": 24}).json()
    assert {i["id"] for i in doc["items"]} == {"i2", "i3", "c3"}
    s = doc["summary"]
    assert (s["candidates"], s["candidates_ok"], s["candidates_error"]) == (1, 1, 0)
    # The error groups count only what is shown: i2's library_save failure, not i1's or c2's.
    assert {(g["line"], g["count"]) for g in s["errors"]} == {(LINE, 1), ("/v1/chat/completions -> 429: spending cap", 1)}
    # A marker older than the window changes nothing.
    W.set_cleared("p1", seeded - 30 * 3600)
    doc = client.get("/api/projects/p1/work", params={"hours": 24}).json()
    assert not doc["hiding_cleared"] and doc["counts"]["all"] == 6


# ---------------------------------------------------------------------------------------
# Iterations left "running": closed on a runner restart, and when their agent goes silent
# ---------------------------------------------------------------------------------------
class Clock:
    def __init__(self, t):
        self.t = t

    def time(self):
        return self.t


def row(store, rid):
    r = archived(store)[rid]
    return r["status"], r["outcome"], json.loads(r["summary"])


def test_a_restart_closes_what_the_work_log_has_newer_than_the_inspectors_file(store, client, monkeypatch):
    """The stuck-'running' bug: the work page copied a record's later posts into the work log
    before the inspector flushed them to its own file; the control plane restarted, reloaded
    the older copy, and closed THAT -- which the work log refused as older. The iteration
    showed "running" for hours after its runner was gone."""
    clock = Clock(time.time() - 600)
    monkeypatch.setattr(A, "time", clock)
    post("Muse", rec("stuck", clock.t))
    A.flush()                                                   # the inspector's file: version 1
    clock.t += 30
    post("Muse", rec("stuck", clock.t - 30, timeline=[tool("run_python", at=clock.t)]))
    assert A.archive_changed() == 1                             # the work log: version 2, newer
    A.reset()                                                   # the control plane stops (no flush) ...
    A._conn.close()
    monkeypatch.setattr(A, "_conn", None)
    r = client.post("/api/agents/activity/close-stale",         # ... and the new runner starts
                    json={"before": clock.t + 60, "reason": "the swarm runner was restarted"})
    assert r.json() == {"closed": 1}
    status, outcome, summary = row(store, "stuck")
    assert (status, outcome) == ("interrupted", "interrupted") and "restarted" in summary["reason"]
    assert summary["tool_calls"] == 1                           # the newer content is kept
    rec_ = W.iteration("stuck")["record"]
    assert rec_["ended_at"] == archived(store)["stuck"]["updated_at"] and rec_["pending"] is None
    assert A.snapshot()[0]["records"][0]["status"] == "interrupted"   # and the inspector agrees


def test_a_restart_closes_iterations_the_inspector_no_longer_holds(store, client):
    """The inspector keeps the last few records per agent; the work log keeps them all, so a
    restart has to close the work log's running rows itself."""
    now = time.time()
    W.archive([({"agent": "Muse", "model": "Muse", "project_id": "p1"},
                [rec("evicted", now - 7200, received_at=now - 7000)])])
    post("Qwen", rec("live", now - 30))                         # the new runner's own work
    A.flush()
    r = client.post("/api/agents/activity/close-stale", json={"before": now - 60})
    assert r.json() == {"closed": 1}
    assert row(store, "evicted")[:2] == ("interrupted", "interrupted")
    assert row(store, "live")[:2] == ("running", "running")
    assert W.iteration("evicted")["record"]["ended_at"] == now - 7000
    # Archiving the same running version again (same last-heard time) does not reopen it ...
    W.archive([({"agent": "Muse", "model": "Muse", "project_id": "p1"},
                [rec("evicted", now - 7200, received_at=now - 7000)])])
    assert row(store, "evicted")[0] == "interrupted"
    # ... but a genuinely newer post does: the agent was alive after all.
    W.archive([({"agent": "Muse", "model": "Muse", "project_id": "p1"},
                [rec("evicted", now - 7200, received_at=now - 10)])])
    assert row(store, "evicted")[0] == "running"


def test_an_iteration_whose_agent_went_silent_shows_as_interrupted(store, client):
    now = time.time()
    meta = {"agent": "Muse", "model": "Muse", "project_id": "p1"}
    W.archive([(meta, [rec("silent", now - A.SILENT_AFTER_S - 1800, received_at=now - A.SILENT_AFTER_S - 60),
                       # a long call (a submit with forecast builds) but still within the limit
                       rec("slow", now - 5400, received_at=now - 4400)])])
    post("Qwen", rec("busy", now - 7200))                       # long-running, posting now
    items = {i["id"]: i for i in client.get("/api/projects/p1/work").json()["items"]}
    assert (items["silent"]["status"], items["silent"]["outcome"]) == ("interrupted", "interrupted")
    assert items["silent"]["reason"] == A.SILENT_REASON
    assert items["slow"]["outcome"] == "running" and items["busy"]["outcome"] == "running"
    assert client.get("/api/projects/p1/work", params={"outcome": "running"}).json()["total"] == 2


def test_the_inspector_closes_a_record_whose_agent_went_silent(store, monkeypatch):
    clock = Clock(time.time())
    monkeypatch.setattr(A, "time", clock)
    post("Muse", rec("quiet", clock.t))
    post("Qwen", rec("chatty", clock.t))
    clock.t += A.SILENT_AFTER_S - 60
    A.flush()
    assert {a["agent"]: a["records"][0]["status"] for a in A.snapshot()} == {"Muse": "running", "Qwen": "running"}
    post("Qwen", rec("chatty", clock.t - A.SILENT_AFTER_S + 60))
    clock.t += 120
    A.flush()
    by_agent = {a["agent"]: a["records"][0] for a in A.snapshot()}
    assert by_agent["Muse"]["status"] == "interrupted" and by_agent["Muse"]["end_reason"] == A.SILENT_REASON
    assert by_agent["Qwen"]["status"] == "running"
    assert row(store, "quiet")[:2] == ("interrupted", "interrupted") and row(store, "chatty")[0] == "running"
    # ... and the inspector's file has it closed too, so a restart does not reopen it.
    doc = json.loads(sqlite3.connect(A.DB_PATH).execute("SELECT doc FROM agents WHERE key='p1|Muse'").fetchone()[0])
    assert doc["records"][0]["status"] == "interrupted"


def test_a_failed_copy_to_the_work_log_is_retried(store, monkeypatch):
    calls = []
    real = W.archive
    monkeypatch.setattr(W, "archive", lambda items: calls.append(1) or (0 if len(calls) == 1 else real(items)))
    post("Muse", rec("r1", time.time(), status="interrupted"))
    assert A.archive_changed() == 0                             # failed: not marked as copied ...
    assert A.archive_changed() == 1                             # ... so the next flush copies it
    assert row(store, "r1")[0] == "interrupted"


# ---------------------------------------------------------------------------------------
# Recovered errors: failures the agent fixed later in the same iteration
# ---------------------------------------------------------------------------------------
NAME_TB = ('Traceback (most recent call last):\n  File "candidate.py", line 1, in <module>\n    np.zeros(3)\n'
           "NameError: name 'np' is not defined\n")
CRASH = json.dumps({"ok": False, "stdout": "", "stderr": TB})


def test_a_failure_is_recovered_only_by_a_later_success_of_the_same_tool():
    t = 1000.0
    ok_py, bad_py = tool("run_python", at=t, code="print(1)"), tool("run_python", result=CRASH, at=t, code="x(")
    # run_python / submit_candidate / library_save: any later success of the tool is the fix ...
    assert W.recovery([bad_py, bad_py, ok_py]) == {0: "recovered", 1: "recovered"}
    # ... and the iteration's last try at it, or one before only other tools worked, is not.
    assert W.recovery([ok_py, bad_py, tool("team_board")]) == {1: "unrecovered"}
    assert W.recovery([ok_py, bad_py], running=True) == {1: "pending"}
    # Other tools: the later success must ask for much the same thing.
    bad_q = tool("query_data", ok=False, result='{"error": "no column hgh"}', sql="select hgh, low from gex_bars")
    fixed_q = tool("query_data", sql="select high, low from gex_bars")
    other_q = tool("query_data", sql="select count(*) from trades where day > '2024-01-01' group by symbol")
    assert W.recovery([bad_q, fixed_q]) == {0: "recovered"}
    assert W.recovery([bad_q, other_q]) == {0: "unrecovered"}
    assert W.recovery([bad_q, tool("describe_data", sql="select hgh, low from gex_bars")]) == {0: "unrecovered"}
    # The runner's auto-repair: the call is ok, and marked -- on the event, or only in a cut result.
    repaired = {**ok_py, "auto_repaired": {"attempts": 1, "errors": [LINE]}}
    cut = {**ok_py, "result": '{"auto_repaired": {"attempts": 2, "errors": ["NameError: x"]}, "ok": true, "stdout": "...'}
    assert W.recovery([repaired, cut]) == {0: "auto_repaired", 1: "auto_repaired"}
    assert W.auto_repair(repaired) == {"attempts": 1, "errors": [LINE]}
    assert W.repaired_errors(W.auto_repair(repaired)) == [LINE]
    assert W.auto_repair({**ok_py, "result": '{"ok": true, "auto_repaired": null}'}) is None
    assert W.auto_repair(ok_py) is None


@pytest.fixture
def recovering(store, client):
    """An iteration whose failures were nearly all fixed: a crashed run fixed by the next, a bad
    query fixed by a similar one, a run and a submission the runner auto-repaired (the crashed
    original submission, c5, stays a candidate row of its own) -- and one lookup never fixed."""
    now = time.time()
    make_objectives(W.OBJECTIVES_DB, now)
    t = now - 600
    con = sqlite3.connect(W.OBJECTIVES_DB)
    cand = ("INSERT INTO candidates VALUES (?, 'o1', ?, ?, 'm1', 'explore', NULL, ?, 'import ft', ?, ?, ?, ?, '{}', "
            "'pass', '', NULL, '', '', ?, 1.0, NULL)")
    con.executemany(cand, [("c5", 5, t + 12, "repaired idea: gex flip", "error", None, None, "the script failed", NAME_TB),
                           ("c6", 6, t + 35, "repaired idea: gex flip", "ok", 1.0, 1.0, "", "")])
    con.commit()
    con.close()
    auto = {"attempts": 1, "errors": ["NameError: name 'np' is not defined"], "original_code_lines": 3}
    post("m1", rec("a1", t, status="submitted", outcome="#6 ok", timeline=[
        tool("run_python", result=CRASH, at=t + 1, code="x = trend_exits(df, atr_lookback=14)"),
        tool("run_python", at=t + 2, code="x = trend_exits(df)"),
        tool("query_data", ok=False, result='{"error": "no column hgh"}', at=t + 3, sql="select hgh, low from gex_bars"),
        tool("query_data", at=t + 4, sql="select high, low from gex_bars"),
        tool("library_get", ok=False, result='{"error": "no library entry gex_flip"}', at=t + 5, entry="gex_flip"),
        {**tool("run_python", at=t + 6, code="np.zeros(3)"), "auto_repaired": auto},
        {**tool("submit_candidate", at=t + 10, code="import ft", rationale="repaired idea: gex flip",
                result={"auto_repaired": auto, "candidate_id": "c6", "seq": 6, "status": "ok"}), "seconds": 30.0}],
        submissions=[{"candidate_id": "c6", "seq": 6, "status": "ok", "at": t + 40, "rationale": "repaired idea"}]))
    post("m2", rec("a2", t + 100, status="no submission", timeline=[
        tool("run_python", result=CRASH, at=t + 101, code="x("), tool("run_python", at=t + 102, code="x()")]), model="m2")
    A.flush()
    return now


def test_the_summary_counts_only_errors_that_stayed_broken(recovering, client):
    doc = client.get("/api/projects/p1/work", params={"hours": 24}).json()
    s = doc["summary"]
    assert (s["tool_errors"], s["tool_errors_recovered"], s["tool_auto_repaired"]) == (1, 5, 2)
    assert (s["experiments"], s["experiments_failed"], s["experiments_recovered"]) == (5, 0, 3)
    # The crashed original of the auto-repaired submission is recovered; c2 (from no iteration here) is not.
    assert (s["candidates_error"], s["candidates_error_recovered"]) == (1, 1)
    groups = {g["line"]: g for g in s["errors"]}
    assert set(groups) == {"no library entry gex_flip", LINE}
    assert groups[LINE]["sources"] == {"candidate": 1}               # c2's, not the run a1 fixed
    rows = {i["id"]: i for i in doc["items"]}
    a1 = rows["a1"]
    assert (a1["tool_errors"], a1["tool_errors_recovered"], a1["error_count"]) == (1, 4, 1)
    assert [e["tool"] for e in a1["errors"]] == ["library_get"] and a1["recovered_count"] >= 2
    assert (rows["a2"]["tool_errors"], rows["a2"]["tool_errors_recovered"], rows["a2"]["errors"]) == (0, 1, [])
    assert rows["c5"]["recovered"] is True and rows["c2"]["recovered"] is False and rows["c6"]["recovered"] is False
    # "any failure" is what stayed broken; a search still finds the recovered lines.
    get = lambda **p: {i["id"] for i in client.get("/api/projects/p1/work", params={"hours": 24, **p}).json()["items"]}  # noqa: E731
    assert get(outcome="tool_errors") == {"a1", "c2"}
    assert get(q="atr_lookback") >= {"a1", "a2"}


def test_the_iteration_detail_keeps_recovered_failures_marked(recovering, client):
    tl = client.get("/api/work/iterations/a1").json()["record"]["timeline"]
    marks = [(e["name"], bool(e.get("failed")), e.get("recovered"), bool(e.get("auto_repaired"))) for e in tl]
    assert marks == [("run_python", True, True, False), ("run_python", False, None, False),
                     ("query_data", True, True, False), ("query_data", False, None, False),
                     ("library_get", True, False, False), ("run_python", False, None, True),
                     ("submit_candidate", False, None, True)]
    assert tl[0]["error_line"] == LINE
    assert tl[5]["repair_attempts"] == 1 and tl[5]["repair_errors"] == ["NameError: name 'np' is not defined"]


def test_summaries_archived_before_are_recomputed_once(store):
    now = time.time()
    W.archive([({"agent": "m1", "model": "m1", "project_id": "p1"},
                [rec("old", now - 60, status="no submission",
                     timeline=[tool("run_python", result=CRASH, at=now - 50), tool("run_python", at=now - 40)])])])
    stale = {**W.summarize(rec("old", now - 60, status="no submission")), "tool_errors": 1, "experiments_failed": 1}
    for k in ("tool_errors_recovered", "experiments_recovered"):
        stale.pop(k)
    with W._lock:
        W.db().execute("UPDATE iterations SET summary=? WHERE id='old'", (json.dumps(stale),))
        W.db().execute("UPDATE meta SET value='1' WHERE key='summary_version'")
        W.db().commit()
    W.reset()
    s = json.loads(archived(store)["old"]["summary"])
    assert (s["tool_errors"], s["tool_errors_recovered"], s["experiments_failed"]) == (0, 1, 0)
    assert W.db().execute("SELECT value FROM meta WHERE key='summary_version'").fetchone()[0] == str(W.SUMMARY_VERSION)


# ---------------------------------------------------------------------------------------
# Not errors: failed side requests, soft steps, policy refusals
# ---------------------------------------------------------------------------------------
TIMEOUT = "/v1/chat/completions failed: TimeoutError: timed out"
BUDGET = json.dumps({"error": "experiment budget used (8 runs this iteration). Turn what works into a library "
                              "module with library_save (with a test) and call submit_candidate with a script that "
                              "imports it.", "best_working_code": "print(1)"})
TRUNC = json.dumps({"error": "your run_python call arrived TRUNCATED (the code did not compile, and either the reply's "
                             "finish_reason was 'length' or the arguments were cut mid-token). This experiment has NOT "
                             "been consumed. Resend the tool call with a SHORTER script -- ... Last chunk received: "
                             "...print(g.quantile"})


def side_record(now, status="done", **extra):
    """What the runner recorded on 10-01 between 19:25 and 20:33: the auto-repair request (capped at
    359 s) timing out inside a run_python call, and the feedback answer (299 s / 479 s) timing out
    inside the answer_feedback step -- once by the old runner (ok=False, "error"), once by the new
    one (ok=True, "soft": true, "soft_error")."""
    repair = {**tool("run_python", result=json.dumps({"ok": False, "stderr": TB, "auto_repair_failed": "..."}),
                     at=now - 1000, code="x"), "seconds": 361.1}
    old_fb = {**tool("answer_feedback", ok=False, at=now - 600,
                     result=json.dumps({"error": f"the answering call failed: {TIMEOUT}"})), "seconds": 299.0}
    new_fb = {**tool("answer_feedback", at=now - 300,
                     result={"answered": 0, "soft": True, "soft_error": f"the answering call failed: {TIMEOUT}",
                             "released": [5]}), "seconds": 479.0, "soft": True}
    chats = [{"at": now - 997.9, "model": "m1", "seconds": 359.0, "finish": None, "error": TIMEOUT},
             {"at": now - 599.99, "model": "m1", "seconds": 299.0, "finish": None, "error": TIMEOUT},
             {"at": now - 299.99, "model": "m1", "seconds": 479.0, "finish": None, "error": TIMEOUT},
             # the agent's own conversation failing: still a chat error
             {"at": now + 200, "model": "m1", "seconds": 600.0, "finish": None, "error": TIMEOUT}]
    tl = [{"kind": "chat", **chats[0]}, repair, tool("run_python", at=now - 620, code="y"),
          {"kind": "chat", **chats[1]}, old_fb, {"kind": "chat", **chats[2]}, new_fb, {"kind": "chat", **chats[3]}]
    return rec("side", now - 1100, status=status, timeline=tl, chats=chats, **extra)


def test_failed_side_requests_and_soft_steps_are_not_chat_errors_or_failures():
    now = time.time()
    s = W.summarize(side_record(now))
    assert (s["chat_errors"], s["chat_errors_soft"]) == (1, 3)
    # the crashed run was recovered by the next run; the answer_feedback steps never failed
    assert (s["tool_errors"], s["tool_errors_recovered"]) == (0, 1)
    soft = [e for e in s["errors"] if e["state"] == "soft"]
    assert [e["purpose"] for e in soft] == ["auto_repair", "answer_feedback", "answer_feedback"]
    assert [e["state"] for e in s["errors"] if e["tool"] == "chat"][-1] == "failed"
    assert W.recovery(side_record(now)["timeline"]) == {1: "recovered"}


def test_a_side_request_is_told_by_the_runner_marker_or_a_running_call():
    now = time.time()
    c = {"at": now, "seconds": 359.0, "error": TIMEOUT}
    # The runner's explicit marker wins over the timeline, both ways.
    assert W.side_purpose({**c, "side": "repair"}, []) == "repair"
    assert W.side_purpose({**c, "side": None}, [(now - 5, now + 400, "auto_repair")]) is None
    # A chat that started before a call (the agent's turn that made it) is not inside it.
    assert W.side_purpose({**c, "at": now - 10, "seconds": 9.0}, [(now - 1, now + 400, "auto_repair")]) is None
    # A tool call still running (the record's "pending") holds its repair's request too.
    r = rec("p", now - 60, timeline=[], chats=[c], pending={"kind": "tool", "name": "submit_candidate", "since": now - 3})
    s = W.summarize(r)
    assert (s["chat_errors"], s["chat_errors_soft"]) == (0, 1)


def test_the_detail_shows_failed_side_requests_as_skipped(store):
    now = time.time()
    W.archive([({"agent": "m1", "model": "m1", "project_id": "p1"}, [side_record(now)])])
    doc = W.iteration("side")
    chats = doc["record"]["chats"]
    assert [bool(c.get("soft")) for c in chats] == [True, True, True, False]
    assert chats[0]["purpose"] == "auto_repair" and chats[0]["soft_note"] == W.SOFT_CHAT_NOTE
    assert "error" not in chats[0] and chats[0]["soft_error"] == TIMEOUT and chats[3]["error"] == TIMEOUT
    tl = doc["record"]["timeline"]
    assert [bool(e.get("soft")) for e in tl if e["kind"] == "chat"] == [True, True, True, False]
    fb = [e for e in tl if e.get("name") == "answer_feedback"]
    assert all(e.get("soft") and not e.get("failed") for e in fb)
    assert all("timed out" in e["soft_error"] for e in fb)
    assert doc["summary"]["chat_errors"] == 1


def test_the_experiment_budget_refusal_is_a_refusal_not_a_failed_experiment(store):
    now = time.time()
    runs = [tool("run_python", at=now - 100 + k, code=f"print({k})") for k in range(8)]
    r = rec("b", now - 200, status="no submission",
            timeline=runs + [tool("run_python", ok=False, result=BUDGET, at=now - 50)])
    assert W.refusal(r["timeline"][-1]) == "experiment_budget"
    assert W.recovery(r["timeline"]) == {8: "refused"}
    s = W.summarize(r)
    assert (s["tool_errors"], s["experiments"], s["experiments_failed"], s["experiments_refused"]) == (0, 8, 0, 1)
    assert (s["refusals"], s["refusals_by_kind"]) == (1, {"experiment_budget": 1})
    assert [(e["state"], e["refusal"]) for e in s["errors"]] == [("refused", "experiment_budget")]
    W.archive([({"agent": "m1", "model": "m1", "project_id": "p1"}, [r])])
    doc = W.work("p1", now - 3600)
    assert doc["summary"]["refusals"] == 1 and doc["summary"]["tool_errors"] == 0 and doc["summary"]["errors"] == []
    row = doc["items"][0]
    assert (row["error_count"], row["refused_count"], row["recovered_count"]) == (0, 1, 0)
    e = W.iteration("b")["record"]["timeline"][-1]
    assert e["refused"] and e["refusal"] == "experiment_budget" and not e.get("failed")
    assert e["error_line"].startswith("experiment budget used")


def test_a_truncated_call_is_a_refusal_the_next_successful_call_recovers():
    now = time.time()
    trunc = tool("run_python", ok=False, result=TRUNC, at=now - 30)
    ok = tool("run_python", at=now - 20, code="print(1)")
    assert W.refusal(trunc) == "truncated"
    # Resent and run: recovered.
    assert W.recovery([trunc, ok]) == {0: "recovered"}
    s = W.summarize(rec("t", now - 60, status="no submission", timeline=[trunc, ok]))
    assert (s["tool_errors"], s["tool_errors_recovered"], s["refusals"], s["refusals_recovered"]) == (0, 0, 1, 1)
    assert (s["experiments"], s["experiments_failed"], s["experiments_refused"]) == (1, 0, 1)
    assert [(e["state"], e["recovered"]) for e in s["errors"]] == [("refused", True)]
    # Not resent (yet): a refusal, while running and after -- never a tool error.
    assert W.recovery([trunc], running=True) == {0: "refused"}
    assert W.recovery([trunc]) == {0: "refused"}
    s = W.summarize(rec("t", now - 60, timeline=[trunc]))
    assert (s["tool_errors"], s["refusals"], s["refusals_recovered"]) == (0, 1, 0)
    # A refusal is no success: it does not recover a crash before it.
    crash = tool("run_python", ok=True, result=CRASH, at=now - 40)
    assert W.recovery([crash, trunc]) == {0: "unrecovered", 1: "refused"}
