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
    A.reset()
    yield
    B.reset()
    A.reset()


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
    asyncio.run(M._triage("m1", B.get_bug(bug["id"])))
    got = B.get_bug(bug["id"])
    # The model writes it up; severity and priority stay the detector's calibrated ones.
    assert got["title"] == "Task sandboxes mount no datasets" and got["triaged"]
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
