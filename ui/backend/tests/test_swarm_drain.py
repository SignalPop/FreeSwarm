"""Graceful drain of the swarm runner (ui/restart-swarm.cmd, stop-services.cmd --drain).

A restart used to Stop-Process -Force the runner and with it every iteration in flight (median
45 min each; ~50 lost in one day). While the drain file exists no agent starts new work, what
runs finishes normally, and the runner exits 0 once nothing runs -- or at the max wait,
naming what it cut. The worker loop is stubbed: iterate() is a gate the test opens.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "swarm_runner.py"


@pytest.fixture()
def runner(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("swarm_runner_under_test_drain", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "DRAIN_FILE", str(tmp_path / ".swarm_drain"))
    monkeypatch.setattr(mod, "POLL_IDLE_S", 0.01)
    monkeypatch.setattr(mod, "DRAIN_POLL_S", 0.02)
    mod.board_puts = []

    def fake_request(base, path, payload=None, **kw):
        if path.startswith("/mb/state/"):
            mod.board_puts.append((path, kw.get("method"), payload))
        return {}

    monkeypatch.setattr(mod, "request", fake_request)
    mod.logs = []
    monkeypatch.setattr(mod, "log", lambda msg: mod.logs.append(msg))
    return mod


def _until(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def _stub(mod, w, release: threading.Event):
    """Stub the board and the work: each iterate() blocks until `release` is set."""
    w.started = []
    w.register = lambda: True
    w.beat = lambda *a, **k: None
    w.claim = lambda: None
    w.next_objective = lambda: {"id": "o1", "title": "objective one", "cooldown_s": 0}

    def iterate(obj):
        w.started.append(time.time())
        release.wait(10)

    w.iterate = iterate
    w.agent_id = "a1"
    return w


def _worker(mod, stop, release, model="m1"):
    w = mod.Worker({"id": "p1", "slug": "proj"}, model, stop, lambda: ([], []))
    return _stub(mod, w, release)


def _drain_on(mod):
    Path(mod.DRAIN_FILE).write_text("restart\n", encoding="utf-8")


def test_no_new_iteration_starts_while_the_drain_file_exists(runner):
    stop, release = threading.Event(), threading.Event()
    release.set()  # iterations would finish at once, so the loop would start many
    _drain_on(runner)
    w = _worker(runner, stop, release)
    w.start()
    try:
        time.sleep(0.3)
        # busy flickers to "starting" for the moment the loop checks the drain file (by design): poll for idle
        assert w.started == [] and _until(lambda: w.busy is None)
        Path(runner.DRAIN_FILE).unlink()  # cancelling the drain lets work start again
        assert _until(lambda: len(w.started) > 0)
    finally:
        stop.set()
        w.join(2)


def test_start_work_marks_busy_before_it_looks_for_a_drain(runner):
    stop = threading.Event()
    w = _worker(runner, stop, threading.Event())
    assert w._start_work("starting") is True and w.busy == "starting"
    w.busy = None
    _drain_on(runner)
    assert w._start_work("starting") is False and w.busy is None
    stop.set()


def test_running_iteration_finishes_then_the_runner_exits_when_idle(runner):
    stop, release = threading.Event(), threading.Event()
    w = _worker(runner, stop, release)
    w.start()
    try:
        assert _until(lambda: len(w.started) == 1 and w.busy)
        assert w.busy == "iteration on objective one"
        drain = runner.Drain(max_s=3600, report_s=3600)
        _drain_on(runner)
        assert drain.tick([w]) is False  # still running: keep waiting
        status = json.loads(Path(runner.DRAIN_FILE + ".status.json").read_text(encoding="utf-8"))
        assert status["draining"] is True and status["running"] == 1
        assert status["iterations"][0]["work"] == "iteration on objective one"
        assert any(p == "/mb/state/swarm_drain" and m == "PUT" and body["value"]["draining"]
                   for p, m, body in runner.board_puts)
        assert any("DRAIN requested" in m for m in runner.logs)

        release.set()  # the running iteration reaches its normal end
        assert _until(lambda: w.busy is None)
        time.sleep(0.1)
        assert len(w.started) == 1  # ...and nothing new started after it
        # the worker marks itself busy for a moment while it checks the drain file (deliberate: the
        # supervisor must never see a false idle); the real supervisor ticks every 5 s, so poll here
        assert _until(lambda: drain.tick([w]))
        assert drain.outcome == "idle" and drain.cut == []
        assert not Path(runner.DRAIN_FILE).exists()
        assert not Path(runner.DRAIN_FILE + ".status.json").exists()
        done = next(m for m in runner.logs if "DRAIN complete" in m)
        assert "Waited for 1" in done and "iteration on objective one" in done
    finally:
        release.set()
        stop.set()
        w.join(2)


def test_max_wait_exits_anyway_and_names_what_it_cut(runner):
    stop, release = threading.Event(), threading.Event()
    w = _worker(runner, stop, release)
    w.start()
    try:
        assert _until(lambda: w.busy)
        _drain_on(runner)
        drain = runner.Drain(max_s=60, report_s=3600)
        t0 = time.time()
        assert drain.tick([w], now=t0) is False
        assert drain.tick([w], now=t0 + 30) is False
        assert drain.tick([w], now=t0 + 61) is True
        assert drain.outcome == "timeout" and drain.cut == [w]
        assert not Path(runner.DRAIN_FILE).exists()
        cut = next(m for m in runner.logs if "max wait" in m)
        assert "cutting" in cut and "proj/m1: iteration on objective one" in cut
    finally:
        release.set()
        stop.set()
        w.join(2)


def test_removing_the_file_cancels_the_drain(runner):
    stop, release = threading.Event(), threading.Event()
    w = _worker(runner, stop, release)
    w.start()
    try:
        assert _until(lambda: w.busy)
        _drain_on(runner)
        drain = runner.Drain(max_s=3600, report_s=3600)
        assert drain.tick([w]) is False and drain.active
        Path(runner.DRAIN_FILE).unlink()
        assert drain.tick([w]) is False and not drain.active
        assert any("drain cancelled" in m for m in runner.logs)
    finally:
        release.set()
        stop.set()
        w.join(2)


def test_main_drains_then_returns_0(runner, monkeypatch):
    """The supervisor end to end: one agent mid-iteration, a drain request, the iteration ends,
    main() returns 0 and the request file is gone."""
    release = threading.Event()
    made = []

    class StubWorker(runner.Worker):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            _stub(runner, self, release)
            made.append(self)

    def fake_request(base, path, payload=None, **kw):
        if path == "/api/projects":
            return {"projects": [{"id": "p1", "slug": "proj"}]}
        if path == "/api/engines":
            return {"loaded": [{"model": "m1"}]}
        if path.endswith("/swarm/plan"):
            return {"search": [{"model": "m1", "agents": 1}]}
        return {}

    monkeypatch.setattr(runner, "Worker", StubWorker)
    monkeypatch.setattr(runner, "request", fake_request)
    monkeypatch.setattr(runner, "ENGINE_RESYNC_S", 0.2)
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("rc", runner.main()), daemon=True)
    t.start()
    try:
        assert _until(lambda: made and made[0].busy == "iteration on objective one")
        _drain_on(runner)
        time.sleep(0.3)
        assert t.is_alive()  # waiting for the iteration, not cutting it
        release.set()
        t.join(5)
        assert not t.is_alive() and result["rc"] == 0
        assert len(made) == 1 and len(made[0].started) == 1
        assert not Path(runner.DRAIN_FILE).exists()
    finally:
        release.set()
        if t.is_alive():
            _drain_on(runner)  # never leave a supervisor running past the test
        t.join(5)


def test_a_drain_file_left_from_before_the_start_is_ignored(runner, monkeypatch):
    _drain_on(runner)
    monkeypatch.setattr(runner, "ENGINE_RESYNC_S", 0.05)
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("rc", runner.main()), daemon=True)
    t.start()
    try:
        time.sleep(0.3)
        assert t.is_alive()  # it did not take the stale file as a request to exit
        assert any("removing a drain request left from before" in m for m in runner.logs)
    finally:
        # End it the supported way (no agents: it leaves at once), so no supervisor outlives
        # the test and reaches the live services once monkeypatch restores `request`.
        _drain_on(runner)
        t.join(5)
    assert not t.is_alive() and result["rc"] == 0
