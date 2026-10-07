"""The monitoring agent and the bug list: platform faults are told from agents' own mistakes,
occurrences of one problem become one bug, rescans never double count, and closed bugs reopen
only when seen again."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import agent_activity as A
from app import bugs as B
from app import monitor as M
from app import prefs as P
from app import work as W

NO_DATASET = ("{\"ok\": false, \"stdout\": \"\", \"stderr\": \"Traceback (most recent call last):\\n  File \\\"script.py\\\", "
              "line 9, in <module>\\n    exec(compile(_src, \\\"candidate.py\\\", \\\"exec\\\"))\\n  File \\\"candidate.py\\\", "
              "line 5, in <module>\\n    trades = ft.load_pl('trade_book_trades_19f971fff6')\\n  File \\\"/work/.ft/ft.py\\\", "
              "line 148, in load_pl\\n    item = _find(name)\\n  File \\\"/work/.ft/ft.py\\\", line 83, in _find\\n    raise "
              "KeyError(...)\\nKeyError: \\\"no dataset 'trade_book_trades_19f971fff6'; available: (none)\\\"\\n\", "
              "\"artifacts\": [], \"duration_s\": 3.25, \"experiments_left\": 5}")


def _run_result(stderr: str) -> str:
    return json.dumps({"ok": False, "stdout": "", "stderr": stderr, "artifacts": []})


def _tb(file: str, line: int, err: str) -> str:
    return (f'Traceback (most recent call last):\n  File "script.py", line 4, in <module>\n    exec(...)\n'
            f'  File "{file}", line {line}\n    x\n    ^\n{err}\n')


def _agent(timeline: list[dict], status: str = "done", model: str = "m1", rid: str = "r1", **rec) -> dict:
    now = time.time()
    return {"agent": model, "model": model, "role": "search", "slot": 0, "project_id": "p1", "updated_at": now,
            "records": [{"id": rid, "mode": "explore", "status": status, "objective": {"id": "o1", "title": "obj"},
                         "started_at": now - 100, "ended_at": now - 1, "pending": None, "timeline": timeline, **rec}]}


def _tool(result, code="print(1)\n", name="run_python", ok=True, at=None) -> dict:
    return {"kind": "tool", "at": at or time.time(), "name": name, "args": {"code": code}, "ok": ok, "seconds": 1.0,
            "result": result}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "DB_PATH", tmp_path / "bugs.sqlite3")
    B.reset()
    monkeypatch.setattr(P, "PREFS_PATH", tmp_path / "prefs.json")
    monkeypatch.setattr(P, "_cache", None)
    monkeypatch.setattr(A, "DB_PATH", tmp_path / "activity.sqlite3")
    monkeypatch.setattr(A, "_conn", None)
    monkeypatch.setattr(M, "_state", {**M._state, "scans": 0, "llm_calls": 0, "last_error": None})
    monkeypatch.setattr(M, "_triage_task", None)
    monkeypatch.setattr(M, "_judged", {})
    monkeypatch.setattr(W, "DB_PATH", tmp_path / "work.sqlite3")
    A.reset()
    W.reset()
    yield
    B.reset()
    A.reset()
    W.reset()


def _file(findings):
    return [B.sight(f) for f in findings]


def test_an_empty_sandbox_catalog_is_a_platform_fault_filed_at_once(store):
    f = M.scan([_agent([_tool(NO_DATASET)])], [], P.MONITOR_DEFAULTS)
    assert len(f) == 1 and f[0]["priority"] == "P1" and f[0]["category"] == "bad_data"
    assert f[0]["script"] == "print(1)\n" and "available: (none)" in f[0]["evidence"]
    _file(f)
    (bug,) = B.list_bugs("open")
    assert bug["title"].startswith("Sandbox has no datasets")
    # Another dataset name is the same problem.
    other = NO_DATASET.replace("trade_book_trades_19f971fff6", "trade_book/trades_x")
    _file(M.scan([_agent([_tool(other)], rid="r2")], [], P.MONITOR_DEFAULTS))
    assert [b["occurrences"] for b in B.list_bugs("open")] == [2]


def test_a_wrong_name_with_the_right_ones_listed_is_the_agents_mistake(store):
    stderr = _tb("/work/.ft/ft.py", 83, "KeyError: \"no dataset 'bars'; available: gex_bars, trades\"")
    stderr = stderr.replace('line 83\n', 'line 83, in _find\n')
    f = M.scan([_agent([_tool(_run_result(stderr))])], [], P.MONITOR_DEFAULTS)
    assert [x["category"] for x in f] == ["agent_error"] and f[0]["min_occurrences"] == 5


def test_code_cut_off_mid_token_is_blamed_on_the_model_that_sent_it(store):
    code = "import ft\nbar = ft.load('sql_exports_db"
    stderr = _tb("candidate.py", 2, "SyntaxError: unterminated string literal (detected at line 2)")
    chat = {"kind": "chat", "at": time.time(), "model": "qwen@groq", "seconds": 3, "finish": "tool_calls"}
    f = M.scan([_agent([chat, _tool(_run_result(stderr), code=code)])], [], P.MONITOR_DEFAULTS)
    assert [x["fingerprint"] for x in f] == ["truncated:qwen@groq"]
    # ... also when it stops inside a name: df.sort_valu -> no attribute 'sort_valu'
    code2 = "import ft\ndf = ft.load('x')\ndf.sort_valu"
    stderr2 = _tb("candidate.py", 3, "AttributeError: 'DataFrame' object has no attribute 'sort_valu'").replace(
        "line 3\n", "line 3, in <module>\n")
    f2 = M.scan([_agent([chat, _tool(_run_result(stderr2), code=code2)])], [], P.MONITOR_DEFAULTS)
    assert [x["fingerprint"] for x in f2] == ["truncated:qwen@groq"]


def test_an_agents_own_error_stays_hidden_until_it_recurs(store):
    stderr = _tb("candidate.py", 1, "NameError: name 'np' is not defined").replace("line 1\n", "line 1, in <module>\n")
    one = lambda rid: M.scan([_agent([_tool(_run_result(stderr), code="np.zeros(3)\nprint(1)\n")], rid=rid)], [],
                             P.MONITOR_DEFAULTS)
    for i in range(4):
        _file(one(f"r{i}"))
    assert B.list_bugs("all") == []
    _file(one("r9"))
    (bug,) = B.list_bugs("open")
    assert bug["occurrences"] == 5 and bug["priority"] == "P4"


def test_a_rescan_counts_nothing_twice_and_a_closed_bug_reopens_only_when_seen_again(store):
    agents = [_agent([_tool(NO_DATASET, at=time.time() - 50)])]
    _file(M.scan(agents, [], P.MONITOR_DEFAULTS))
    _file(M.scan(agents, [], P.MONITOR_DEFAULTS))
    (bug,) = B.list_bugs("open")
    assert bug["occurrences"] == 1
    B.patch_bug(bug["id"], B.BugPatch(status="closed"))
    _file(M.scan(agents, [], P.MONITOR_DEFAULTS))                      # the old occurrence again: stays closed
    assert B.list_bugs("open") == [] and B.counts()["closed"] == 1
    _file(M.scan([_agent([_tool(NO_DATASET)], rid="r2")], [], P.MONITOR_DEFAULTS))
    (bug,) = B.list_bugs("open")
    assert bug["occurrences"] == 2 and "reopened" in B.get_bug(bug["id"])["notes"]


def test_chat_failures_stalls_and_engine_crashes(store):
    now = time.time()
    chat = lambda err: {"kind": "chat", "at": now, "model": "m1", "seconds": 1, "error": err}
    f = M.scan([_agent([chat("/v1/chat/completions -> 429: today's external-model spending limit for search is reached"),
                        chat("/v1/chat/completions -> 502: engine unreachable: ")])], [], P.MONITOR_DEFAULTS)
    assert {x["fingerprint"] for x in f} == {"chat:spending-limit", "chat:unreachable:m1"}
    # A reply that ran into the proxy's timeout is a stall, not a dead engine -- also as older
    # builds reported it ("engine unreachable: " with no reason, after exactly 600 s).
    slow = {"kind": "chat", "at": now, "model": "m1", "seconds": 600.0,
            "error": "/v1/chat/completions -> 502: engine unreachable: "}
    new = {"kind": "chat", "at": now, "model": "m1", "seconds": 1800.0,
           "error": "/v1/chat/completions -> 504: engine did not finish the reply within 1800s: it is busy"}
    f = [x for x in M.scan([_agent([slow, new])], [], P.MONITOR_DEFAULTS) if x["fingerprint"].startswith("chat:")]
    assert [x["fingerprint"] for x in f] == ["chat:timeout:m1", "chat:timeout:m1"] and f[0]["category"] == "stall"
    stuck = _agent([], status="running", pending={"kind": "chat", "model": "m1", "since": now - 1200})
    stuck["updated_at"] = now - 1200
    (s,) = M.scan([stuck], [], P.MONITOR_DEFAULTS, now)
    assert s["fingerprint"] == "stall:chat:m1" and s["severity"] == "high"
    engine = {"state": "error", "model_id": "gemma-4-26B-A4B-it", "started_at": now, "command": "python -m freetoken ...",
              "error": "KeyError: 'model.embed_tokens.weight'",
              "diagnosis": {"summary": "KeyError: 'model.embed_tokens.weight'", "tail": ["Traceback", "KeyError"],
                            "exit_code": 15}}
    (e,) = M.scan([], [engine, {"state": "running", "model_id": "ok"}], P.MONITOR_DEFAULTS)
    assert e["severity"] == "critical" and "gemma-4" in e["title"] and e["script"].startswith("python")


def test_a_bug_is_closed_as_fixed_only_after_enough_clean_chances_and_reopens_if_it_returns(store):
    t0 = time.time() - 3 * 3600
    _file(M.scan([_agent([_tool(NO_DATASET, at=t0)])], [], P.MONITOR_DEFAULTS))
    (bug,) = B.list_bugs("open")
    cfg = {**P.MONITOR_DEFAULTS, "fixed_after": 3, "fixed_quiet_minutes": 30}
    clean = lambda n, start: [_tool(json.dumps({"ok": True, "stdout": "5671 trades"}), at=start + i) for i in range(n)]
    # Two clean run_python calls since: not enough evidence yet.
    agents = [_agent(clean(2, t0 + 60), rid="r2")]
    assert M.fixed_bugs(B.watched(), agents, [], cfg, time.time()) == []
    # Three, but the last occurrence was only 10 minutes ago: not quiet long enough.
    assert M.fixed_bugs(B.watched(), [_agent(clean(3, time.time() - 500), rid="r3")], [], cfg,
                        time.time() - 3 * 3600 + 600) == []
    # Three clean calls and hours of quiet: fixed.
    agents = [_agent(clean(3, t0 + 60), rid="r3")]
    ((bid, why),) = M.fixed_bugs(B.watched(), agents, [], cfg, time.time())
    assert bid == bug["id"] and "not seen in 3 run_python calls" in why
    B.close_fixed(bid, why)
    got = B.get_bug(bid)
    assert got["status"] == "closed" and got["closed_by"] == "monitor" and "closed by the monitor as fixed" in got["notes"]
    assert B.watched() == []
    # It comes back: reopened, and no longer marked closed by the monitor.
    _file(M.scan([_agent([_tool(NO_DATASET)], rid="r9")], [], P.MONITOR_DEFAULTS))
    got = B.get_bug(bid)
    assert got["status"] == "open" and got["closed_by"] is None


def test_an_engine_crash_is_fixed_when_that_model_runs_again_and_reviews_are_never_auto_closed(store):
    now = time.time()
    crash = {"state": "error", "model_id": "g4", "started_at": now - 600, "diagnosis": {"summary": "KeyError: 'x'"}}
    _file(M.scan([], [crash], P.MONITOR_DEFAULTS))
    B.sight({"fingerprint": "review:odd data", "key": "k", "title": "Odd data in the rows export", "at": now - 7200})
    watched = B.watched()
    assert len(watched) == 2
    assert M.fixed_bugs(watched, [], [crash], P.MONITOR_DEFAULTS, now) == []
    running = {"state": "running", "model_id": "g4", "started_at": now - 60}
    ((bid, why),) = M.fixed_bugs(watched, [], [running], P.MONITOR_DEFAULTS, now)
    assert B.get_bug(bid)["title"].startswith("Engine failed") and "running again" in why


def test_recheck_closes_a_bug_once_the_logs_show_it_fixed(store, monkeypatch):
    monkeypatch.setattr(M, "_complete", None)
    monkeypatch.setattr(M, "_engines", None)
    P.set_monitor({"fixed_after": 3})
    now = time.time()
    A.record({"agent": "m1", "model": "m1", "project_id": "p1",
              "record": {"id": "r1", "status": "done", "started_at": now - 60, "timeline": [_tool(NO_DATASET, at=now - 60)]}})
    asyncio.run(M.scan_once(force=True))
    (bug,) = B.list_bugs("open")
    out = asyncio.run(M.recheck(bug["id"]))
    assert out["verdict"] == "not_yet" and out["chances"] == 0 and "0 of 3 run_python calls" in out["message"]
    A.record({"agent": "m1", "model": "m1", "project_id": "p1",
              "record": {"id": "r2", "status": "done", "started_at": now - 30,
                         "timeline": [_tool(json.dumps({"ok": True}), at=now - 30 + i) for i in range(3)]}})
    out = asyncio.run(M.recheck(bug["id"]))                  # no quiet period on an operator's recheck
    assert out["verdict"] == "fixed" and out["bug"]["status"] == "closed" and out["bug"]["closed_by"] == "monitor"
    # The history is in the notes, one timestamped line per event, oldest first.
    events = [ln.split("] ", 1)[1].split(":")[0] for ln in out["bug"]["notes"].splitlines()]
    assert events == ["filed by the monitor", "recheck", "closed by the monitor as fixed"]
    made = B.sight({"fingerprint": "manual:1", "key": "k", "title": "Something I noticed", "source": "manual"})
    assert asyncio.run(M.recheck(made["id"]))["verdict"] == "unknown"


def test_a_slow_model_never_holds_up_a_scan_or_a_recheck(store, monkeypatch):
    """Triage used to run inside the scan, under its lock: on a busy engine a Recheck waited
    minutes behind the model's replies. Now triage runs beside the scan, one pass at a time."""
    calls = []

    async def slow(model, messages, max_tokens, purpose):
        calls.append(model)
        await asyncio.sleep(30)
        return "{}"

    monkeypatch.setattr(M, "_complete", slow)
    monkeypatch.setattr(M, "_loaded", lambda: [{"model": "m1", "ready": True}])
    monkeypatch.setattr(M, "_engines", None)
    P.set_monitor({"enabled": True, "llm_triage": True})
    A.record({"agent": "m1", "model": "m1", "project_id": "p1",
              "record": {"id": "r1", "status": "done", "started_at": time.time(), "timeline": [_tool(NO_DATASET)]}})

    async def main():
        t0 = time.time()
        first = await M.scan_once()
        assert first["new_bugs"] == 1 and first["triage"] == "started" and time.time() - t0 < 5
        await asyncio.sleep(0.05)                                # the pass is now waiting on the model
        assert M._state["triage_running"] and calls == ["m1"]
        assert (await M.scan_once())["triage"] == "busy"          # never two passes at once
        (bug,) = B.list_bugs("open")
        t0 = time.time()
        out = await M.recheck(bug["id"])
        assert out["verdict"] == "not_yet" and time.time() - t0 < 5
        M._triage_task.cancel()

    asyncio.run(main())


def test_model_written_titles_that_echo_the_prompt_are_dropped():
    assert not M._good_title("error|bad_data|stall")
    assert not M._good_title("stall")
    assert M._good_title("Task sandboxes mount no datasets")
    assert M._good_title("Model doesn't submit after its budget runs out")


def test_the_proxy_tells_a_slow_engine_from_a_dead_one():
    import httpx

    from app import main

    slow = main._engine_http_error(httpx.ReadTimeout(""))
    assert slow.status_code == 504 and "did not finish the reply within 1800s" in slow.detail
    dead = main._engine_http_error(httpx.ConnectError(""))
    assert dead.status_code == 502 and dead.detail == "engine unreachable: ConnectError"
    assert main._engine_http_error(httpx.ConnectTimeout("")).status_code == 502   # never reached it


def test_triage_writes_the_bug_up_but_never_over_an_operators_edit(store, monkeypatch):
    _file(M.scan([_agent([_tool(NO_DATASET)])], [], P.MONITOR_DEFAULTS))
    (bug,) = B.list_bugs("open")

    async def complete(model, messages, max_tokens, purpose):
        assert purpose == "monitor" and "no dataset" in messages[0]["content"]
        return ('<think>hmm</think>```json\n{"title": "Task sandboxes mount no datasets", "description": "Every load '
                'fails.", "likely_cause": "task runs mount only /task", "suggestion": "mount the trade book", '
                '"severity": "critical", "priority": "P1"}\n```')

    monkeypatch.setattr(M, "_complete", complete)
    bug = B.get_bug(bug["id"])
    asyncio.run(M._triage("m1", bug))
    got = B.get_bug(bug["id"])
    # The model adds its reading and a fix; a rule-filed bug keeps the detector's title and
    # description (the 0.6B model's rewrites were wrong, #332), and its severity and priority.
    assert got["title"] == bug["title"] and got["triaged"]
    assert got["description"].startswith(bug["description"]) and "Reading by m1: Every load fails." in got["description"]
    assert (got["severity"], got["priority"]) == ("high", "P1")
    assert "Likely cause: task runs mount only /task" in got["description"] and got["suggestion"] == "mount the trade book"
    B.patch_bug(bug["id"], B.BugPatch(title="mine"))
    B.enrich(bug["id"], {"title": "theirs", "suggestion": "better fix"})
    got = B.get_bug(bug["id"])
    assert got["title"] == "mine" and got["suggestion"] == "better fix"


def test_the_routes_and_the_switch(store, monkeypatch):
    app = FastAPI()
    app.include_router(B.router, prefix="/api")
    app.include_router(M.router, prefix="/api")
    c = TestClient(app)
    monkeypatch.setattr(M, "_loaded", lambda: [{"model": "m1", "ready": True}])
    monkeypatch.setattr(M, "_complete", None)
    assert c.get("/api/monitor").json()["config"]["enabled"] is False
    assert asyncio.run(M.scan_once()) == {"skipped": "the monitoring agent is off"}
    doc = c.put("/api/monitor", json={"enabled": True, "stall_minutes": 5}).json()
    assert doc["config"]["enabled"] and doc["config"]["stall_minutes"] == 5 and doc["models"] == ["m1"]
    A.record({"agent": "m1", "model": "m1", "project_id": "p1",
              "record": {"id": "r1", "status": "done", "started_at": time.time(), "timeline": [_tool(NO_DATASET)]}})
    out = c.post("/api/monitor/scan").json()
    assert out["new_bugs"] == 1 and out["state"]["scans"] == 1
    listed = c.get("/api/bugs?status=open").json()
    assert listed["counts"] == {"open": 1, "pending": 0, "closed": 0}
    bid = listed["bugs"][0]["id"]
    assert c.patch(f"/api/bugs/{bid}", json={"status": "pending", "priority": "P2"}).json()["status"] == "pending"
    full = c.get(f"/api/bugs/{bid}").json()
    assert full["sightings"][0]["record_id"] == "r1" and full["script"] == "print(1)\n" and full["priority"] == "P2"
    assert full["notes"].splitlines()[-1].endswith("status open -> pending (by the operator)")
    made = c.post("/api/bugs", json={"title": "by hand", "severity": "low"}).json()
    assert made["source"] == "manual" and c.get("/api/bugs/counts").json()["open"] == 1
    assert c.delete(f"/api/bugs/{made['id']}").json() == {"ok": True}
    assert c.get(f"/api/bugs/{made['id']}").status_code == 404


def test_a_record_left_open_by_a_stopped_or_retired_agent_is_not_a_stall(store):
    """Bugs #69/#70/#72: after the engines were unloaded, retired agents' records kept a chat
    'pending' for an hour, and the monitor called the models stuck."""
    now = time.time()

    def waiting(minutes, model="m1"):
        a = _agent([], status="running", model=model, pending={"kind": "chat", "model": model,
                                                                "since": now - minutes * 60})
        a["updated_at"] = now - minutes * 60
        return a

    cfg = P.MONITOR_DEFAULTS
    # 20 min waiting on a loaded model: a real stall.
    assert [f["fingerprint"] for f in M.scan([waiting(20)], [], cfg, now, {"m1"})] == ["stall:chat:m1"]
    # 60 min: no request lives that long -- abandoned, reported as an agent gone silent, not a stall.
    assert [f["fingerprint"] for f in M.scan([waiting(60)], [], cfg, now, {"m1"})] == ["silent:p1"]
    # Its model is not loaded any more: the worker was retired with it -- nothing to report.
    assert M.scan([waiting(20)], [], cfg, now, {"other"}) == []
    assert M.scan([waiting(60)], [], cfg, now, set()) == []


def test_a_review_is_filed_only_with_evidence_quoted_from_the_log_and_at_a_moderate_level(store, monkeypatch):
    rec = {"id": "r1", "status": "done", "started_at": time.time() - 60, "ended_at": time.time(),
           "timeline": [_tool(json.dumps({"ok": True, "stdout": "rows 0 -- every Close value is NaN after 2024-05-01"}))]}
    agent = {"agent": "m1", "model": "m1", "project_id": "p1", "records": [rec]}
    reply = {"issues": [
        {"title": "Sharpe ratio not optimal", "category": "error|bad_data|stall", "severity": "high",
         "priority": "P2", "description": "the strategy loses", "evidence": "tool_run_python", "tool": "run_python"},
        {"title": "Close column is all NaN after May 2024", "category": "bad_data", "severity": "critical",
         "priority": "P1", "description": "the price is missing", "evidence": "every Close value is NaN after 2024-05-01",
         "tool": "run_python"}]}

    async def complete(model, messages, max_tokens, purpose):
        return json.dumps(reply)

    monkeypatch.setattr(M, "_complete", complete)
    assert asyncio.run(M._review("m1", agent, rec)) == 1
    (bug,) = B.list_bugs("open")
    assert bug["title"] == "Close column is all NaN after May 2024"
    assert (bug["severity"], bug["priority"], bug["category"]) == ("medium", "P3", "bad_data")


def test_the_detectors_severity_comes_back_after_a_model_inflated_it(store):
    f = M.scan([_agent([_tool(NO_DATASET)])], [], P.MONITOR_DEFAULTS)
    _file(f)
    (bug,) = B.list_bugs("open")
    with B._lock:                                   # what an older triage build did to it
        B.db().execute("UPDATE bugs SET severity='critical', priority='P4' WHERE id=?", (bug["id"],))
        B.db().commit()
    _file(f)                                        # a rescan of the same, already-counted occurrence
    got = B.get_bug(bug["id"])
    assert (got["severity"], got["priority"], got["occurrences"]) == ("high", "P1", 1)
    B.patch_bug(bug["id"], B.BugPatch(severity="low"))     # the operator's choice is kept
    _file(f)
    assert B.get_bug(bug["id"])["severity"] == "low"


_PY = ('Traceback (most recent call last):\n  File "script.py", line 9, in <module>\n    exec(compile(_src, "candidate.py", '
       '"exec"))\n')
_AGENT_TB = _PY + ('  File "candidate.py", line 193, in <module>\n    print(int((positions!=0).sum()))\n'
                   "AttributeError: 'bool' object has no attribute 'sum'\n")
# ft refusing a call on purpose: the traceback ends on a `raise` in ft.py ...
_FT_REFUSAL = _PY + ('  File "candidate.py", line 7, in <module>\n    ft.report_positions(positions)\n'
                     '  File "/work/.ft/ft.py", line 1182, in wrapper\n    return fn(*[_to_pandas(a) for a in args])\n'
                     '  File "/work/.ft/ft.py", line 1288, in report_positions\n    "t": _position_times(s.index),\n'
                     '  File "/work/.ft/ft.py", line 1083, in _position_times\n    raise ValueError(\n'
                     "ValueError: report_positions: the series must be indexed by the bar timestamp; got a RangeIndex\n")
# ... and ft breaking: the traceback runs on through ft.py into a library.
_FT_BROKEN = _PY + ('  File "candidate.py", line 5, in <module>\n    print(ft.size(1, inv, base=2.0))\n'
                    '  File "/work/.ft/ft.py", line 583, in size\n    s = np.broadcast_to(np.asarray(scale), (len(d),))\n'
                    '  File "/usr/local/lib/python3.12/site-packages/numpy/lib/_stride_tricks_impl.py", line 456, in '
                    "_broadcast_to\n    it = np.nditer(\n"
                    "ValueError: operands could not be broadcast together with remapped shapes [original->remapped]: "
                    "(3,)  and requested shape (1,)\n")


def _submit(stderr: str) -> dict:
    return _tool(json.dumps({"candidate_id": "c1", "seq": 7, "status": "error", "error": "the script failed -- see stderr",
                             "stderr_tail": stderr, "stdout_tail": ""}), name="submit_candidate", ok=False)


def _per_call(findings):
    """The findings about each call, without the one about the iteration's outcome (submitfail)."""
    return [x for x in findings if not x["fingerprint"].startswith("submitfail:")]


def test_a_failed_submission_or_save_is_judged_by_the_traceback_it_carries(store):
    """Bugs #41 and #16: every failed submission was one bug, "submit_candidate keeps failing: the
    script failed -- see stderr" -- unrelated agent mistakes that no fix could close, with any
    harness fault at submission hidden among them as a P4 agent error."""
    other = _AGENT_TB.replace("AttributeError: 'bool' object has no attribute 'sum'",
                              "KeyError: \"['position'] not in index\"")
    f = [x for x in M.scan([_agent([_submit(_AGENT_TB), _submit(other)])], [], P.MONITOR_DEFAULTS)
         if not x["fingerprint"].startswith("submitfail:")]
    assert [x["fingerprint"] for x in f] == ["agent:data-object-mixup", "agent:KeyError:\"['position'] not in index\""]
    assert {(x["category"], x["min_occurrences"]) for x in f} == {("agent_error", 5)}
    # The platform's own code failing at submission is a platform bug, at its real priority.
    (h,) = _per_call(M.scan([_agent([_submit(_FT_BROKEN)])], [], P.MONITOR_DEFAULTS))
    assert h["fingerprint"].startswith("harness:submit_candidate:ValueError:") and h["priority"] == "P2"
    # A library save: the smoke test's output says whose failure it was ...
    smoke = _PY + ('  File "candidate.py", line 1, in <module>\n    from lib import sig\n  File "/work/.ft/lib/sig.py", '
                   'line 56\n    _et = int(x)\n                ^\nIndentationError: unindent does not match any outer '
                   'indentation level')

    def save(**d):
        return _tool(json.dumps({"saved": False, "test_ok": False, **d}), name="library_save", ok=False,
                     code="def signal(df):\n" + "    x = 1\n" * 80)

    (s,) = M.scan([_agent([save(test_output="[lib] imported sig\n\n" + smoke,
                                error="the smoke test failed -- fix the module and save again")])], [], P.MONITOR_DEFAULTS)
    assert s["fingerprint"] == "agent:IndentationError:unindent does not match any outer indentation level"
    # ... and one refused without a traceback (a look-ahead verdict) stays the tool's own answer.
    (t,) = M.scan([_agent([save(test_output="[causality] FAIL", error="look-ahead: sig.signal() output changed")])],
                  [], P.MONITOR_DEFAULTS)
    assert t["fingerprint"].startswith("toolerr:library_save:look-ahead")


def test_a_call_ft_refuses_on_purpose_is_the_agents_mistake_not_a_harness_error(store):
    """With submissions read properly, report_positions refusing a row-numbered series would have
    become "Harness error in submit_candidate" (P2, filed on the second one). A traceback that
    ENDS on a `raise` in ft.py is ft checking its input; one that runs on into numpy is ft breaking."""
    assert M.traceback_of(_FT_REFUSAL)["raised"] and M.traceback_of(_FT_REFUSAL)["origin"] == "harness"
    assert not M.traceback_of(_FT_BROKEN)["raised"] and M.traceback_of(_FT_BROKEN)["origin"] == "harness"
    assert not M.traceback_of(_AGENT_TB)["raised"]
    (f,) = _per_call(M.scan([_agent([_submit(_FT_REFUSAL)])], [], P.MONITOR_DEFAULTS))
    assert f["fingerprint"].startswith("ftcheck:submit_candidate:ValueError:report_positions")
    assert (f["category"], f["priority"], f["min_occurrences"]) == ("agent_error", "P4", 5)
    (g,) = M.scan([_agent([_tool(_run_result(_FT_BROKEN))])], [], P.MONITOR_DEFAULTS)
    assert g["fingerprint"].startswith("harness:run_python:ValueError:operands could not be broadcast")
    assert (g["priority"], g["min_occurrences"]) == ("P2", 2)
    # It is watched like any tool bug: closed once enough calls of that tool pass without it.
    _file([{**f, "at": time.time() - 7200, "key": f"k{i}"} for i in range(5)])
    (bug,) = B.watched()
    assert M.fix_chances(bug, [_agent([_submit(_AGENT_TB)], rid="r2")], [])[2] == "submit_candidate calls"


def test_a_review_cannot_file_the_agents_own_mistakes_or_other_agents_words(store, monkeypatch):
    """The open list of 2026-09-30: thirteen bugs, eleven filed by a small reviewing model from
    quotes that were real but were not the platform's fault. Each kind, from that list."""
    board = [{"when": "06:49", "who": "m2", "channel": "errors", "kind": "error",
              "text": "library_save failed: the smoke test failed -- fix the module and save again"}]
    shown = json.dumps({"seq": 1459, "status": "ok",
                        "code": "print(f\"Continuous soft dampener, trades {trades}, active days {active_days}\")"})
    refused = json.dumps({"error": "/api/objectives/o1/data/query -> 400: Catalog Error: Table with name "
                                   "sql_exports_dbo_gex_bar10s does not exist!"})
    ranked = {"candidate_id": "c1", "seq": 3, "status": "ok", "rank": "unranked",
              "not_ranked": "too few trades: 1.937 trades/day in-sample -- at least 2 a day are required."}
    caught = json.dumps({"ok": True, "stdout": "error operands could not be broadcast together with remapped shapes "
                                               "[original->remapped]: (5677,)  and requested shape (1,)\n", "stderr": ""})
    rec = {"id": "r1", "status": "done", "started_at": time.time() - 60, "ended_at": time.time(), "timeline": [
        _tool(board, name="team_board"), _tool(shown, name="get_candidate"),
        _tool(_run_result(_AGENT_TB)), _tool(refused, name="query_data", ok=False),
        _tool(ranked, name="submit_candidate"), _tool(_run_result(_FT_BROKEN))]}
    agent = {"agent": "m1", "model": "m1", "project_id": "p1", "records": [rec]}

    def issue(title, evidence, tool=None):
        return {"title": title, "category": "error", "description": "it failed", "evidence": evidence, "tool": tool}

    reply = {"issues": [
        issue("tool failure in library_save", "library_save failed: the smoke test failed -- fix the module and save again"),
        issue("Invalid data format in candidate", "from get_candidate tool output: 'Continuous soft dampener, trades "
                                                  "{trades}, active days {active_days}'"),
        issue("Missing or corrupt data", "from run_python tool output: AttributeError: 'bool' object has no attribute 'sum'")]}

    async def complete(model, messages, max_tokens, purpose):
        return json.dumps(reply)

    monkeypatch.setattr(M, "_complete", complete)
    assert asyncio.run(M._review("m1", agent, rec)) == 0
    reply["issues"] = [
        issue("query_data_tool_error", "Catalog Error: Table with name sql_exports_dbo_gex_bar10s does not exist!",
              "query_data"),
        issue("Trade Frequency Constraint", "The tool response indicates 'too few trades: 1.937 trades/day in-sample -- "
                                            "at least 2 a day are required.'"),
        # A failed call the detectors already filed by rule (the ft.size fault, #88): not filed twice.
        issue("shape mismatch in a failed ft.size call", "ValueError: operands could not be broadcast together with "
                                                         "remapped shapes [original->remapped]: (3,)  and requested shape (1,)")]
    assert asyncio.run(M._review("m1", agent, rec)) == 0
    assert B.list_bugs("all") == []
    # What is left is what reviews are for: something odd in the output of a call that worked
    # (here the agent caught ft's exception and printed it -- no rule reads a successful run).
    rec["timeline"].append(_tool(caught))
    reply["issues"] = [issue("ft.size cannot take one direction for every bar",
                             "error operands could not be broadcast together with remapped shapes [original->remapped]: "
                             "(5677,)  and requested shape (1,)", "run_python")]
    assert asyncio.run(M._review("m1", agent, rec)) == 1
    assert [b["title"] for b in B.list_bugs("open")] == ["ft.size cannot take one direction for every bar"]


def test_time_the_control_plane_was_down_is_not_counted_as_a_stall_or_silence(store):
    """Bugs #69, #70, #128, #156: the whole stack was down 12:08-15:39; the monitor's first scan
    after the restart called every agent the old runner left open 'stuck' or 'silent' for hours."""
    now = time.time()

    def waiting(minutes):
        a = _agent([], status="running", pending={"kind": "chat", "model": "m1", "since": now - minutes * 60})
        a["updated_at"] = now - minutes * 60
        return a

    cfg = P.MONITOR_DEFAULTS
    started = now - 5 * 60                                   # the control plane came up 5 min ago
    # A chat pending from before the restart died with it: not a stall, and silence counts
    # only the 5 minutes watched.
    assert M.scan([waiting(20)], [], cfg, now, {"m1"}, started) == []
    assert M.scan([waiting(200)], [], cfg, now, {"m1"}, started) == []
    # Watched long enough, a runner that really is gone is still reported.
    assert [f["fingerprint"] for f in M.scan([waiting(200)], [], cfg, now, {"m1"}, now - 40 * 60)] == ["silent:p1"]
    # A request that started after the restart and hangs is a stall as before.
    assert [f["fingerprint"] for f in M.scan([waiting(20)], [], cfg, now, {"m1"}, now - 60 * 60)] == ["stall:chat:m1"]


_POLARS_OVER = _PY + ('  File "candidate.py", line 12, in expand_z\n    cum_sum = s_float.cum_sum().over(\'session\')\n'
                      "AttributeError: 'Series' object has no attribute 'over'\n")
_TREND_KW = _PY + ('  File "candidate.py", line 73, in main\n    positions = ft.trend_exits(\n'
                   '  File "/work/.ft/ft.py", line 1202, in wrapper\n    return fn(*[_to_pandas(a) for a in args], '
                   '**{k: _to_pandas(v) for k, v in kw.items()})\n'
                   "TypeError: trend_exits() got an unexpected keyword argument 'atr_lookback'\n")
_TREND_KW_NOW = _PY + ('  File "candidate.py", line 73, in main\n    positions = ft.trend_exits(\n'
                       '  File "/work/.ft/ft.py", line 1240, in wrapper\n    raise bad from None\n'
                       "TypeError: ft.trend_exits: got an unexpected keyword argument 'start_time'.\n"
                       "  'start_time' is not a parameter -- did you mean 'no_entry_before'?\n")


def test_one_mistake_under_many_messages_is_one_bug_that_shows(store):
    """The night of 2026-09-30: 19 data-object mix-ups made 11 bugs and 7 wrong trend_exits keywords
    made 3 (one per tool, filed as "Harness error"), nearly all hidden under min_occurrences."""
    mixups = [_AGENT_TB, _POLARS_OVER, _AGENT_TB.replace("'bool' object has no attribute 'sum'",
                                                         "'Expr' object has no attribute 'to_numpy'")]
    fps = {x["fingerprint"] for s in mixups for x in _per_call(M.scan([_agent([_submit(s)])], [], P.MONITOR_DEFAULTS))}
    assert fps == {"agent:data-object-mixup"}
    for stderr, tool in ((_TREND_KW, _submit), (_TREND_KW_NOW, lambda s: _tool(_run_result(s)))):
        (f,) = _per_call(M.scan([_agent([tool(stderr)])], [], P.MONITOR_DEFAULTS))
        assert f["fingerprint"] == "ftcall:trend_exits" and (f["category"], f["min_occurrences"]) == ("agent_error", 2)
    # an agent's own function called wrongly is still the agent's own error
    own = _PY + ('  File "candidate.py", line 3, in <module>\n    sig(x=1)\n'
                 "TypeError: sig() got an unexpected keyword argument 'x'\n")
    (g,) = _per_call(M.scan([_agent([_submit(own)])], [], P.MONITOR_DEFAULTS))
    assert g["fingerprint"].startswith("agent:TypeError:")


def test_an_iteration_whose_every_submission_failed_is_a_bug(store):
    """Qwen3.6-35B submitted in 6 of 7 iterations that night, every try crashed, and since it DID
    submit, the no-submission bug never counted it."""
    def ok_submit():
        return _tool(json.dumps({"candidate_id": "c2", "seq": 8, "status": "ok", "in_sample_score": 1.0}),
                     name="submit_candidate")

    (f,) = [x for x in M.scan([_agent([_submit(_POLARS_OVER), _submit(_TREND_KW)], status="submitted", model="q")],
                              [], P.MONITOR_DEFAULTS) if x["fingerprint"].startswith("submitfail:")]
    assert f["fingerprint"] == "submitfail:q" and (f["priority"], f["min_occurrences"]) == ("P2", 2)
    assert "#7: AttributeError: 'Series' object has no attribute 'over'" in f["description"]
    # a fix that ran, a run still going, or no submission at all: not this bug
    for agent in (_agent([_submit(_POLARS_OVER), ok_submit()], status="submitted"),
                  _agent([_submit(_POLARS_OVER)], status="running"), _agent([], status="no submission")):
        assert not [x for x in M.scan([agent], [], P.MONITOR_DEFAULTS) if x["fingerprint"].startswith("submitfail:")]
    # closed as fixed once that model's iterations get scored runs again
    _file([{**f, "at": time.time() - 7200, "key": f"k{i}"} for i in range(2)])
    (bug,) = B.watched()
    later = [_agent([ok_submit()], status="submitted", model="q", rid=f"r{i}") for i in range(5)]
    assert M.fix_chances(bug, [{**later[0], "records": [a["records"][0] for a in later]}], [])[:2] == (5, 5)


def test_a_review_cannot_quote_a_run_of_the_agents_code_that_failed(store, monkeypatch):
    """#309, #299, #275, #273, #230, #184 (2026-10-01): a small reviewing model quoted the agent's
    own errors out of calls that WORKED -- run_python's {"ok": false}, a library save whose module
    raised in the causality test, a run_python result cut too long to parse as JSON (its
    escaped traceback was never judged)."""
    raised = json.dumps({"saved": True, "name": "sig", "version": 1, "test_ok": True, "test_output": "[lib] imported sig",
                         "causality": {"verdict": "error", "detail": "sig.signal() raised on data cut at 2024-02-06 "
                                                                     "16:09:59: AttributeError: 'DataFrame' object has "
                                                                     "no attribute 'with_columns'"}})
    cut = json.dumps({"ok": False, "stdout": "x" * 3000, "stderr": _PY + '  File "candidate.py", line 16, in <module>\n'
                      "    df['d'] = pd.qcut(df['x'], 10)\n              ^^\nNameError: name 'pd' is not defined\n"})
    cut = cut[:1500] + "\n\n...[559 characters omitted from the middle]...\n\n" + cut[2100:]
    assert M._parse(cut) is None
    (f,) = M.scan([_agent([_tool(cut)])], [], P.MONITOR_DEFAULTS)              # judged by rule now
    assert f["fingerprint"] == "agent:NameError:name 'pd' is not defined"
    rec = {"id": "r1", "status": "done", "started_at": time.time() - 60, "ended_at": time.time(),
           "timeline": [_tool(raised, name="library_save"), _tool(cut)]}
    agent = {"agent": "m1", "model": "m1", "project_id": "p1", "records": [rec]}
    reply = {"issues": [
        {"title": "Missing 'with_columns' method in ft.rows_pl", "category": "error", "description": "it failed",
         "evidence": "AttributeError: 'DataFrame' object has no attribute 'with_columns'"},
        {"title": "Incorrect column name in ft.rows_pl parameters", "category": "error", "description": "it failed",
         "evidence": "df['d'] = pd.qcut(df['x'], 10)"}]}

    async def complete(model, messages, max_tokens, purpose):
        return json.dumps(reply)

    monkeypatch.setattr(M, "_complete", complete)
    assert asyncio.run(M._review("m1", agent, rec)) == 0
    assert B.list_bugs("all") == []
    # A save whose module passed is still the platform speaking.
    assert not M._reports_failure(json.dumps({"saved": True, "test_ok": True, "causality": {"verdict": "pass"}}))


def test_a_submission_killed_for_time_is_not_filed_with_every_other_failure(store):
    """#41: "the script failed -- see stderr" with no traceback in stderr was one catch-all bug;
    what stderr ends on says what happened."""
    killed = _tool(json.dumps({"candidate_id": "c1", "seq": 117, "status": "error", "error": "the script failed -- see stderr",
                               "stderr_tail": "[killed: exceeded the 300s limit]", "stdout_tail": ""}),
                   name="submit_candidate", ok=False)
    (f,) = [x for x in M.scan([_agent([killed])], [], P.MONITOR_DEFAULTS) if not x["fingerprint"].startswith("submitfail:")]
    assert f["fingerprint"] == "toolerr:submit_candidate:the script failed -- see stderr: [killed: exceeded the <n>s limit]"
    assert "killed" in f["title"]
    other = _tool(json.dumps({"error": "no candidate 1198"}), name="get_candidate", ok=False)
    (g,) = M.scan([_agent([other])], [], P.MONITOR_DEFAULTS)
    assert g["fingerprint"] == "toolerr:get_candidate:no candidate <n>"


# ---------------------------------------------------------------------------------------------
# Recovered errors: an agent's mistake it fixed itself later in the same iteration is no bug
# ---------------------------------------------------------------------------------------------
_NAME_ERR = _tb("candidate.py", 1, "NameError: name 'np' is not defined").replace("line 1\n", "line 1, in <module>\n")


def _crash(at: float) -> dict:
    return _tool(_run_result(_NAME_ERR), code="np.zeros(3)\nprint(1)\n", at=at)


def _ran(at: float) -> dict:
    return _tool(json.dumps({"ok": True, "stdout": "1"}), code="import numpy as np\nnp.zeros(3)\n", at=at)


def _with(e: dict, **args) -> dict:
    return {**e, "args": args}


def _one(records: list[dict]) -> dict:
    """One agent holding these records."""
    return {"agent": "m1", "model": "m1", "role": "search", "slot": 0, "project_id": "p1", "updated_at": time.time(),
            "records": records}


def test_an_error_the_agent_fixed_later_in_its_iteration_is_not_filed(store):
    now, cfg = time.time(), P.MONITOR_DEFAULTS
    assert M.scan([_agent([_crash(now - 50), _ran(now - 40)])], [], cfg) == []
    # Its last try, or an iteration that ended before a fix: filed.
    (f,) = M.scan([_agent([_ran(now - 50), _crash(now - 40)])], [], cfg)
    assert f["fingerprint"] == "agent:NameError:name 'np' is not defined"
    assert len(M.scan([_agent([_crash(now - 40)], status="interrupted")], [], cfg)) == 1
    # Still running: not yet -- a later scan sees whether it got fixed.
    assert M.scan([_agent([_crash(now - 40)], status="running")], [], cfg) == []
    # Another tool: only a call asking for much the same thing fixes it.
    bad = _with(_tool(json.dumps({"error": "no candidate 1198"}), name="get_candidate", ok=False, at=now - 30), seq=1198)
    assert M.scan([_agent([bad, _with(_tool('{"seq": 118}', name="get_candidate", at=now - 20), seq=118)])], [], cfg) == []
    assert len(M.scan([_agent([bad, _with(_tool("{}", name="team_board", at=now - 20))])], [], cfg)) == 1
    # A platform fault is filed even when a retry worked: the fix is the platform's.
    (h,) = M.scan([_agent([_tool(NO_DATASET, at=now - 50), _ran(now - 40)])], [], cfg)
    assert h["category"] == "bad_data"


def test_a_bug_made_only_of_errors_the_agents_fixed_is_closed(store):
    """Filed before recovered errors were left out (or from a sighting whose iteration fixed it
    afterwards): five crashes, each one fixed by the agent's next run."""
    now = time.time()
    first = [_agent([_crash(now - 300 + i)], rid=f"r{i}") for i in range(5)]
    for a in first:
        r = a["records"][0]
        _file([f for i, e in enumerate(r["timeline"]) for f in M.tool_findings(a, r, i, e, None)])
    (bug,) = B.list_bugs("open")
    assert bug["occurrences"] == 5
    fixed = [_agent([_crash(now - 300 + i), _ran(now - 200 + i)], rid=f"r{i}")["records"][0] for i in range(5)]
    running = [*fixed[:4], {**first[4]["records"][0], "status": "running"}]
    assert M.resolve_recovered([_one(running)]) == 0               # one may still be fixed: wait
    assert M.resolve_recovered([_one(fixed)]) == 1
    closed = B.get_bug(bug["id"])
    assert closed["status"] == "closed" and "recovered" in closed["notes"]
    # An unrecovered one brings it back.
    _file(M.scan([_agent([_crash(time.time() + 1)], rid="r9")], [], P.MONITOR_DEFAULTS))
    assert B.get_bug(bug["id"])["status"] == "open"


def test_a_bug_with_errors_that_stayed_broken_stands_and_the_work_log_judges_old_records(store):
    now = time.time()
    broken = [_agent([_crash(now - 300 + i)], rid=f"r{i}") for i in range(5)]
    for a in broken:
        _file(M.scan([a], [], P.MONITOR_DEFAULTS))
    (bug,) = B.list_bugs("open")
    assert M.resolve_recovered([_one([a["records"][0] for a in broken])]) == 0
    assert M.resolve_recovered([]) == 0                            # judged once: not again until sighted again
    M._judged.clear()
    assert M.resolve_recovered([]) == 0                            # records nowhere to be found: it stands
    assert B.get_bug(bug["id"])["status"] == "open"
    # The inspector no longer holds the records, the work log does -- and there they were fixed.
    W.archive([({"agent": "m1", "model": "m1", "project_id": "p1"},
                [{**a["records"][0], "timeline": [_crash(now - 300 + i), _ran(now - 200 + i)]}
                 for i, a in enumerate(broken)])])
    M._judged.clear()
    assert M.resolve_recovered([]) == 1 and B.get_bug(bug["id"])["status"] == "closed"


def test_a_bug_left_under_its_threshold_without_its_recovered_sightings_is_closed():
    t = 1000.0
    recs = {f"r{i}": {"id": f"r{i}", "status": "done", "started_at": t,
                      "timeline": [_crash(t + i)] + ([_ran(t + 50)] if i else [])} for i in range(5)}
    bug = {"source": "monitor", "fingerprint": "agent:NameError:x", "occurrences": 5, "min_occurrences": 5,
           "sightings": [{"record_id": f"r{i}", "at": t + i} for i in range(5)]}
    verdict, why = M.recovered_only(bug, recs)
    assert verdict == "close" and "4 of its 5" in why
    # ... unless it was sighted more often than the sightings it keeps tell
    assert M.recovered_only({**bug, "occurrences": 80}, recs) == ("keep", None)
    # ... and platform faults are never judged so
    assert M.recovered_only({**bug, "fingerprint": "harness:run_python:KeyError:x"}, recs) == ("keep", None)


# ---------------------------------------------------------------------------------------
# Not bugs: failed side requests, soft steps, policy refusals (see work.py)
# ---------------------------------------------------------------------------------------
_TIMEOUT = "/v1/chat/completions failed: TimeoutError: timed out"
_BUDGET = json.dumps({"error": "experiment budget used (8 runs this iteration). Turn what works into a library "
                               "module with library_save (with a test) and call submit_candidate.",
                      "best_working_code": "print(1)"})
_TRUNC = json.dumps({"error": "your run_python call arrived TRUNCATED (the code did not compile, and either the "
                              "reply's finish_reason was 'length' or the arguments were cut mid-token). This "
                              "experiment has NOT been consumed. Resend ... Last chunk received: ...print(g.quantile"})


def test_failed_side_requests_and_soft_steps_file_no_bugs(store):
    now = time.time()
    crash = _run_result(_tb("candidate.py", 3, "KeyError: 'close'"))
    repair = {**_tool(crash, at=now - 1000), "seconds": 361.1}
    old_fb = {**_tool(json.dumps({"error": f"the answering call failed: {_TIMEOUT}"}), name="answer_feedback",
                      ok=False, at=now - 600), "seconds": 299.0}
    new_fb = {**_tool({"answered": 0, "soft": True, "soft_error": f"the answering call failed: {_TIMEOUT}"},
                      name="answer_feedback", at=now - 300), "seconds": 479.0, "soft": True}
    side = lambda at, s: {"kind": "chat", "at": at, "model": "m1", "seconds": s, "error": _TIMEOUT}  # noqa: E731
    tl = [side(now - 997.9, 359.0), repair, side(now - 599.99, 299.0), old_fb, side(now - 299.99, 479.0), new_fb]
    f = M.scan([_agent(tl)], [], P.MONITOR_DEFAULTS)
    assert not [x for x in f if x["fingerprint"].startswith(("chat:", "slow:", "toolerr:answer_feedback"))]
    # The agent's own conversation timing out is still one.
    f = M.scan([_agent(tl + [side(now + 10, 600.0)])], [], P.MONITOR_DEFAULTS)
    assert [x["fingerprint"].split(":")[0] for x in f if x["fingerprint"].startswith("chat:")] == ["chat"]
    # A repair request timing out while its call still runs (the record's "pending"): no bug either.
    rec = _agent([side(now - 400, 359.0)], status="running",
                 pending={"kind": "tool", "name": "run_python", "since": now - 402})
    assert not [x for x in M.scan([rec], [], P.MONITOR_DEFAULTS) if x["fingerprint"].startswith("chat:")]


def test_policy_refusals_file_no_bugs(store):
    now = time.time()
    budget = [_tool(_BUDGET, ok=False, at=now - 50 + k) for k in range(5)]
    assert M.scan([_agent(budget)], [], P.MONITOR_DEFAULTS) == []
    # A truncated call is refused unrun: no bug whether or not it was resent.
    trunc = [_tool(_TRUNC, ok=False, at=now - 40 + k) for k in range(6)]
    assert M.scan([_agent(trunc)], [], P.MONITOR_DEFAULTS) == []
    assert M.scan([_agent(trunc, status="running")], [], P.MONITOR_DEFAULTS) == []
    assert M.scan([_agent(trunc + [_tool('{"ok": true}', at=now)])], [], P.MONITOR_DEFAULTS) == []


def test_an_open_bug_made_of_refusals_is_closed():
    now = time.time()
    tl = [_tool(_TRUNC, ok=False, at=now - 40), _tool(_BUDGET, ok=False, at=now - 30)]
    r = {"id": "r1", "status": "done", "timeline": tl}
    bug = {"source": "monitor", "fingerprint": "toolerr:run_python:your run_python call arrived truncated",
           "occurrences": 2, "min_occurrences": 5,
           "sightings": [{"record_id": "r1", "at": now - 40}, {"record_id": "r1", "at": now - 30}]}
    verdict, why = M.recovered_only(bug, {"r1": r})
    assert verdict == "close" and "refusals" in why


# #409 / #410 / #411 (2026-10-04): Qwen3-0.6B reviewed an iteration whose run_python WORKED and filed
# "empty tables", "constant columns" and "impossible values", all three on ft's compat note in its stderr.
_FT_SESSION = "[ft] the frame had no 'session' column -- added as the New York session date of 't' (what ft.clock gives)"
_WORKED = json.dumps({
    "ok": True, "stdout": "both_any_gex_109: active_days=90, sharpe=2.141\n  long_pct=N/A, short_pct=N/A\n",
    "stderr": _FT_SESSION + "\ncandidate.py:31: ConstantInputWarning: An input array is constant; the correlation "
                            "coefficient is not defined.\n",
    "artifacts": [], "duration_s": 7.59, "data": "in-sample only (rows before 2024-07-19)", "experiments_left": 2})
_FT_REVIEW = {"issues": [
    {"title": "empty tables", "category": "bad_data", "tool": "run_python",
     "description": "The agent's code references a dataset with an empty table.",
     "evidence": "The tool's stderr shows an error about the session column having no 'session' column -- added as "
                 "the New York session date of 't'."},
    {"title": "constant columns", "category": "bad_data", "tool": "run_python",
     "description": "a constant or invalid column definition",
     "evidence": "ConstantInputWarning: An input array is constant; the correlation coefficient is not defined."},
    {"title": "impossible values", "category": "bad_data", "tool": "run_python",
     "description": "the columns have impossible values",
     "evidence": "[ft] the frame had no 'session' column -- added as the New York session date of 't'"}]}


def _reviewed(monkeypatch, rec: dict, reply: dict) -> tuple[int, list[str]]:
    prompts = []

    async def complete(model, messages, max_tokens, purpose):
        prompts.append(messages[0]["content"])
        return json.dumps(reply)

    monkeypatch.setattr(M, "_complete", complete)
    agent = {"agent": "m1", "model": "m1", "project_id": "p1", "records": [rec]}
    return asyncio.run(M._review("Qwen/Qwen3-0.6B", agent, rec)), prompts


def test_a_review_never_files_fts_notes_or_the_stderr_of_a_run_that_worked(store, monkeypatch):
    rec = {"id": "r1", "status": "done", "started_at": time.time() - 60, "ended_at": time.time(),
           "timeline": [_tool(_WORKED)]}
    n, (prompt,) = _reviewed(monkeypatch, rec, _FT_REVIEW)
    assert n == 0 and B.list_bugs("all") == []
    # the reviewer is not even shown them -- its three findings are not spent on the helper's notes
    assert "[ft] the frame" not in prompt and "New York session date" not in prompt and "ConstantInputWarning" not in prompt
    assert "both_any_gex_109: active_days=90, sharpe=2.141" in prompt and "in-sample only" in prompt
    # what the platform said in the same result is still evidence
    n, _ = _reviewed(monkeypatch, rec, {"issues": [
        {"title": "Sharpe disagrees with the scoreboard", "category": "bad_data", "description": "it disagrees",
         "evidence": "both_any_gex_109: active_days=90, sharpe=2.141"}]})
    assert n == 1


def test_what_the_reviewer_is_shown_of_a_result():
    # a note that is the last line of its JSON string hides itself only, not the rest of the result
    last = json.dumps({"saved": True, "test_output": "[lib] imported sig\n\n" + _FT_SESSION, "rows": 4182})
    shown = M._shown(last)
    assert "[ft]" not in shown and "New York" not in shown and '"rows": 4182' in shown and "[lib] imported sig" in shown
    # ...and raw text the same, line by line
    assert M._shown("a\n" + _FT_SESSION + "\nb") == "a\n\nb"
    # a run that failed keeps its stderr: the traceback is what the detectors judge it by
    failed = json.dumps({"ok": False, "stdout": "", "stderr": _FT_SESSION + "\nKeyError: 'no dataset x'"})
    assert "KeyError: 'no dataset x'" in M._shown(failed) and "[ft]" not in M._shown(failed)
    # a result that is not JSON, or cut too long to parse, still loses the notes
    assert "[ft]" not in M._shown('{"ok": true, "stderr": "' + _FT_SESSION + '\n", "stdout": "x' + "y" * 5000)


def test_the_iteration_behind_409_410_411_files_nothing_from_the_work_log():
    """Replays the stored iteration (record 1791151300071-988) and the three evidence quotes the
    reviewer gave, read-only; skipped once the record has left the work log."""
    import sqlite3
    import zlib
    from pathlib import Path

    db = Path(__file__).resolve().parents[1] / "work.sqlite3"
    if not db.is_file():
        pytest.skip("no work log")
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT agent, project_id, record FROM iterations WHERE id='1791151300071-988'").fetchone()
    finally:
        con.close()
    if row is None:
        pytest.skip("the record has left the work log")
    rec = json.loads(zlib.decompress(row[2]))
    agent = {"agent": row[0], "model": row[0], "project_id": row[1], "records": [rec]}
    assert any("[ft] the frame had no 'session' column" in str(e.get("result")) for e in rec["timeline"])
    assert "[ft]" not in M.condense(rec)
    for ev in ("The tool's stderr shows an error about the session column having no 'session' column -- added as "
               "the New York session date of 't'.",
               "The tool's stderr includes a line about the session column having no 'session' column -- added as "
               "the New York session date of 't'."):
        assert not M._platform_evidence(ev, agent, rec)
