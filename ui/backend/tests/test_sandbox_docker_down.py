"""Bug #417: Docker Desktop going down made `docker run` print its own connection error, and the
sandbox handed that back as the script's stderr. 26 submissions from three models were scored
"the script failed -- see stderr" without running, and the monitor blamed each model. Now the
sandbox waits for the daemon and retries, then refuses with a 503; the never-run candidate is
dropped; the monitor files one sandbox outage."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi import HTTPException

from app import monitor as M
from app import objectives as O
from app import sandbox as S

DOWN = ("failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine; check if the path is "
        "correct and if the daemon is running: open //./pipe/dockerDesktopLinuxEngine: The system cannot find the "
        "file specified.\n")


def test_docker_failure_is_told_from_a_script_failure():
    assert S.docker_failure(1, "", DOWN)
    assert S.docker_failure(1, "", "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. "
                                   "Is the docker daemon running?")
    assert S.docker_failure(125, "", "docker: Error response from daemon: Docker Desktop is shutting down.")
    # the script's own failures
    assert S.docker_failure(1, "", 'Traceback (most recent call last):\n  File "script.py"\nZeroDivisionError: x') is None
    assert S.docker_failure(125, "", "my script chose exit code 125") is None
    # output means the container ran
    assert S.docker_failure(1, "hello\n", DOWN) is None


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(S, "_docker", lambda: "docker")
    return tmp_path / "runs"


def test_a_run_waits_for_the_daemon_and_retries(runs, monkeypatch):
    starts = []

    async def start(docker, argv, container, timeout_s):
        starts.append(container)
        return (b"", DOWN.encode(), 1, False) if len(starts) == 1 else (b"42\n", b"", 0, False)

    async def daemon_back(docker, deadline):
        return True

    monkeypatch.setattr(S, "_start", start)
    monkeypatch.setattr(S, "_wait_for_daemon", daemon_back)
    rep = asyncio.run(S.execute("print(42)"))
    assert rep["ok"] and rep["stdout"] == "42\n"
    assert len(starts) == 2 and starts[0] != starts[1]


def test_a_daemon_that_stays_down_is_a_503_not_a_failed_script(runs, monkeypatch):
    async def start(docker, argv, container, timeout_s):
        return b"", DOWN.encode(), 1, False

    async def still_down(docker, deadline):
        return False

    monkeypatch.setattr(S, "_start", start)
    monkeypatch.setattr(S, "_wait_for_daemon", still_down)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(S.execute("print(42)"))
    assert exc.value.status_code == 503
    assert exc.value.detail.startswith(S.SANDBOX_DOWN) and "NOT run" in exc.value.detail
    assert not any(runs.iterdir())                       # no run folder left behind


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "DB_PATH", tmp_path / "objectives.sqlite3")
    monkeypatch.setattr(O, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(O, "_conn", None)
    import app.library as L
    monkeypatch.setattr(L, "_ready", False)
    yield
    if O._conn is not None:
        O._conn.close()
    O._conn = None


def test_a_submission_the_sandbox_never_ran_leaves_no_candidate(temp_db, tmp_path, monkeypatch):
    async def sandbox_down(*a, **k):
        raise HTTPException(status_code=503, detail=f"{S.SANDBOX_DOWN}: Docker could not start the container")

    monkeypatch.setattr(O.projects, "get", lambda pid: {"data_dir": str(tmp_path)})
    monkeypatch.setattr(O.datasource, "catalog", lambda d: [])
    monkeypatch.setattr(O, "_run_forecasting", sandbox_down)
    obj = {"id": "o1", "project_id": "p1", "title": "t", "status": "running", "dataset": "px", "time_column": "t",
           "lookahead_check": False, "require_audit": False, "eval_timeout_s": 60, "best_id": None,
           "metric": {"kind": "sharpe", "price_column": "Close"}}
    with pytest.raises(HTTPException) as exc:
        asyncio.run(O.evaluate(obj, O.Submit(code="import ft\nft.report_positions(x)", model="m", mode="explore")))
    assert exc.value.status_code == 503
    assert O.db().execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0


def test_the_monitor_files_one_sandbox_outage_not_a_model_bug():
    now = time.time()
    err = f"/api/objectives/o1/candidates -> 503: {S.SANDBOX_DOWN}: Docker could not start the container"
    timeline = [{"kind": "tool", "at": now, "name": name, "args": {"code": "print(1)"}, "ok": False, "seconds": 180.0,
                 "result": json.dumps({"error": err})} for name in ("run_python", "submit_candidate")]
    a = {"agent": "m1", "model": "m1", "role": "search", "slot": 0, "project_id": "p1", "updated_at": now,
         "records": [{"id": "r1", "mode": "explore", "status": "no submission", "objective": {"id": "o1", "title": "o"},
                      "started_at": now - 100, "ended_at": now - 1, "pending": None, "timeline": timeline}]}
    r = a["records"][0]
    found = [f for i, e in enumerate(timeline) for f in M.tool_findings(a, r, i, e, None)]
    assert {f["fingerprint"] for f in found} == {"sandbox:down"}
    assert not [f for f in M.record_findings(a, r, now, {}) if f["fingerprint"].startswith("submitfail:")]
