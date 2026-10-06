"""The remembered model setup and its restore job, with fake managers -- nothing launches."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import last_setup, main
from app.engine import EngineSupervisor


# ---------------------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------------------
class _Proc:
    def __init__(self, alive: bool = True):
        self.alive, self.pid = alive, 4242

    def poll(self):
        return None if self.alive else 1


class FakeEngine:
    def __init__(self, inst_id, model, gpus, state="starting", alive=True, error=None):
        self.instance_id, self.model_id, self.gpus = inst_id, model, gpus
        self.state, self._alive, self.error = state, alive, error
        self.options: dict = {}
        self.ever_ready = state == "running"
        self.polls = 0

    def is_alive(self):
        return self._alive

    def status(self):
        return {"instance_id": self.instance_id, "gpus": self.gpus, "state": self.state,
                "error": self.error, "port": 1919, "model_id": self.model_id}


class FakeLLM:
    """Engines become ready (or fail) after a couple of polls; records the order and overlap."""

    def __init__(self, fail: dict[str, str] | None = None, crash: dict[str, str] | None = None):
        self.instances: dict[str, FakeEngine] = {}
        self.calls: list[tuple[str, dict, str | None]] = []
        self.fail, self.crash = fail or {}, crash or {}
        self.max_loading = 0

    def all(self):
        return list(self.instances.values())

    def get(self, inst_id):
        inst = self.instances.get(inst_id)
        if inst is not None and inst.state == "starting":
            inst.polls += 1
            if inst.polls >= 2:
                if inst.model_id in self.crash:
                    inst.state, inst._alive, inst.error = "error", False, self.crash[inst.model_id]
                else:
                    inst.state, inst.ever_ready = "running", True
        return inst

    async def start(self, model, options=None, gpus=None):
        self.calls.append((model, dict(options or {}), gpus))
        if model in self.fail:
            raise ValueError(self.fail[model])
        loading = sum(1 for i in self.instances.values() if i.state == "starting")
        self.max_loading = max(self.max_loading, loading + 1)
        inst_id = f"eng{len(self.instances)}"
        inst = FakeEngine(inst_id, model, gpus or "9")
        inst.options = dict(options or {})
        self.instances[inst_id] = inst
        return inst.status()


class FakeTS:
    def __init__(self, running: list[str] | None = None, refuse_gpu: str | None = None):
        self.running = running or []
        self.calls: list[tuple[str, str | None]] = []
        self.refuse_gpu = refuse_gpu
        self.llm_loading_at_start: list[int] = []
        self.llm: FakeLLM | None = None

    def statuses(self):
        return [{"id": f"ts{i}", "model_id": m, "state": "running"} for i, m in enumerate(self.running)]

    async def start(self, model, path, gpu=None):
        self.calls.append((model, gpu))
        if self.llm is not None:
            self.llm_loading_at_start.append(sum(1 for i in self.llm.all() if i.state == "starting"))
        if self.refuse_gpu is not None and gpu == self.refuse_gpu:
            raise ValueError(f"GPU {gpu} has no room")
        self.running.append(model)
        return {"id": "ts1960", "model_id": model, "gpu": gpu or "2", "port": 1960, "state": "running"}

    async def stop(self, inst_id):
        self.running = [m for i, m in enumerate(self.running) if f"ts{i}" != inst_id]
        return {"id": inst_id, "state": "stopped"}


CATALOG = [
    {"id": "small", "path": "/m/small", "size_bytes": 1 * 2**30, "expert_bytes": 0},
    {"id": "big", "path": "/m/big", "size_bytes": 70 * 2**30, "expert_bytes": 60 * 2**30},
    {"id": "mid", "path": "/m/mid", "size_bytes": 20 * 2**30, "expert_bytes": 0},
    {"id": "chronos", "path": "/m/chronos", "size_bytes": 2**29, "expert_bytes": 0},
    {"id": "huge-ts", "path": "/m/huge-ts", "size_bytes": 200 * 2**30, "expert_bytes": 0},
]


def _resolve(model: str) -> Path:
    for m in CATALOG:
        if m["id"] == model:
            return Path(m["path"])
    raise ValueError(f"unknown model: {model!r}")


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(last_setup, "PATH", tmp_path / "last_setup.json")
    monkeypatch.setattr(last_setup, "_job", None)
    monkeypatch.setattr(last_setup, "_task", None)
    return tmp_path / "last_setup.json"


def _restore(llm, ts, gpus=(0, 1, 2), **kw):
    async def go():
        job = await last_setup.start_restore(
            llm, ts, catalog=lambda: CATALOG, resolve=_resolve, gpu_indices=lambda: set(gpus),
            poll_s=0, **kw,
        )
        await last_setup.wait()
        return job

    return asyncio.run(go())


def _by_model(job):
    return {i["model"]: i for i in job["items"]}


# ---------------------------------------------------------------------------------------
# Persistence through the real routes (manager / ts_manager faked)
# ---------------------------------------------------------------------------------------
class RouteLLM:
    def __init__(self):
        self.instances: dict[str, EngineSupervisor] = {}

    def get(self, inst_id):
        return self.instances.get(inst_id)

    async def start(self, model, options=None, gpus=None):
        e = EngineSupervisor(f"eng{len(self.instances)}", 1919, gpus or "1")
        e.model_id, e.state, e._proc, e.options = model, "starting", _Proc(True), dict(options or {})
        e.model_path = f"/m/{model}"
        self.instances[e.instance_id] = e
        return {**e.status(), "gpus": e.gpus}

    async def stop(self, inst_id):
        e = self.instances.pop(inst_id)
        e.state, e._proc = "stopped", None
        return {"instance_id": inst_id, "state": "stopped"}


@pytest.fixture
def client(monkeypatch):
    fake = RouteLLM()
    monkeypatch.setattr(main, "manager", fake)
    monkeypatch.setattr(main.prefs, "set_launch_options", lambda *a, **k: None)

    async def _no_sample():
        return None

    monkeypatch.setattr(main, "_sample_engines", _no_sample)
    return TestClient(main.app), fake


def test_user_start_is_remembered_with_everything_needed(client):
    c, _ = client
    opts = {"moe_backend": "offload", "num_tokens": 131072, "memory_ratio": 0.8}
    r = c.post("/api/engines", json={"model": "big", "options": opts, "gpus": "2"})
    assert r.status_code == 200, r.text
    [e] = last_setup.entries()
    assert e["kind"] == "llm" and e["model"] == "big"
    assert e["options"] == opts and e["requested_gpus"] == "2" and e["gpus"] == "2"
    assert e["model_path"] == "/m/big" and e["port"] == 1919


def test_auto_gpu_keeps_requested_auto_and_assigned_card(client):
    c, _ = client
    c.post("/api/engines", json={"model": "mid", "options": {}})
    [e] = last_setup.entries()
    assert e["requested_gpus"] is None and e["gpus"] == "1"


def test_user_stop_of_live_engine_forgets_it(client):
    c, fake = client
    c.post("/api/engines", json={"model": "big", "options": {}, "gpus": "2"})
    c.post("/api/engines", json={"model": "mid", "options": {}, "gpus": "1"})
    r = c.post("/api/engines/eng0/stop")
    assert r.status_code == 200, r.text
    assert [e["model"] for e in last_setup.entries()] == ["mid"]


def test_crash_does_not_touch_the_set(client):
    c, fake = client
    c.post("/api/engines", json={"model": "big", "options": {}, "gpus": "2"})
    eng = fake.instances["eng0"]
    eng.ever_ready, eng.state = True, "running"
    # The engine dies on its own: nothing calls into last_setup.
    eng._proc = _Proc(False)
    eng.status()
    assert [e["model"] for e in last_setup.entries()] == ["big"]
    # ... and dismissing the crashed card afterwards keeps it too.
    c.post("/api/engines/eng0/stop")
    assert [e["model"] for e in last_setup.entries()] == ["big"]


def test_dismissing_a_load_that_never_got_ready_forgets_it(client):
    c, fake = client
    c.post("/api/engines", json={"model": "big", "options": {}, "gpus": "2"})
    eng = fake.instances["eng0"]
    eng._proc, eng.state = _Proc(False), "error"  # died while loading; never ready
    c.post("/api/engines/eng0/stop")
    assert last_setup.entries() == []


def test_refused_launch_is_not_remembered(client, monkeypatch):
    c, fake = client
    from app.engine import LaunchError

    async def refuse(*a, **k):
        raise LaunchError("does not fit")

    monkeypatch.setattr(fake, "start", refuse)
    r = c.post("/api/engines", json={"model": "big", "options": {}})
    assert r.status_code == 400
    assert last_setup.entries() == []


def test_timeseries_start_and_stop(client, monkeypatch):
    c, _ = client
    ts = FakeTS()
    monkeypatch.setattr(main, "ts_manager", ts)
    monkeypatch.setattr("app.catalog.resolve_model_path", _resolve)
    r = c.post("/api/ts", json={"model": "chronos"})
    assert r.status_code == 200, r.text
    [e] = last_setup.entries()
    assert e["kind"] == "ts" and e["requested_gpus"] is None and e["gpus"] == "2"
    assert c.post("/api/ts/ts0/stop").status_code == 200
    assert last_setup.entries() == []


def test_corrupt_file_reads_as_empty(store):
    store.write_text("{not json", encoding="utf-8")
    assert last_setup.entries() == []
    last_setup.record_llm("mid", {}, None, {"gpus": "1"})
    assert [e["model"] for e in last_setup.entries()] == ["mid"]


# ---------------------------------------------------------------------------------------
# Restore: order, skipping, failures
# ---------------------------------------------------------------------------------------
def _remember():
    last_setup.record_ts("chronos", None, {"gpu": "1"})
    last_setup.record_llm("small", {"memory_ratio": 0.6}, "0", {"gpus": "0"})
    last_setup.record_ts("huge-ts", None, {"gpu": "2"})
    last_setup.record_llm("big", {"moe_backend": "offload"}, "2", {"gpus": "2"})
    last_setup.record_llm("mid", {"num_tokens": 65536}, "1", {"gpus": "1"})


def test_plan_orders_largest_llm_first_then_forecasters():
    _remember()
    items = last_setup.plan(FakeLLM(), FakeTS(), catalog=lambda: CATALOG, resolve=_resolve,
                            gpu_indices=lambda: {0, 1, 2})
    # A 200 GiB forecaster still goes after every LLM.
    assert [i["model"] for i in items] == ["big", "mid", "small", "huge-ts", "chronos"]
    assert all(i["action"] == "start" for i in items)


def test_restore_starts_one_at_a_time_in_order_with_exact_settings():
    _remember()
    llm, ts = FakeLLM(), FakeTS()
    ts.llm = llm
    job = _restore(llm, ts)
    assert [c[0] for c in llm.calls] == ["big", "mid", "small"]
    assert llm.calls[0] == ("big", {"moe_backend": "offload"}, "2")
    assert llm.calls[1] == ("mid", {"num_tokens": 65536}, "1")
    assert llm.max_loading == 1  # never two loading at once
    assert [c[0] for c in ts.calls] == ["huge-ts", "chronos"]
    assert ts.llm_loading_at_start == [0, 0]  # forecasters only after every LLM is ready
    assert job["state"] == "done"
    assert {i["status"] for i in job["items"]} == {"ready"}


def test_restore_skips_models_already_running():
    _remember()
    llm = FakeLLM()
    llm.instances["x"] = FakeEngine("x", "mid", "1", state="running")
    ts = FakeTS(running=["chronos"])
    job = _restore(llm, ts)
    items = _by_model(job)
    assert items["mid"]["status"] == "skipped" and items["chronos"]["status"] == "skipped"
    assert [c[0] for c in llm.calls] == ["big", "small"]
    assert [c[0] for c in ts.calls] == ["huge-ts"]


def test_one_failure_does_not_stop_the_rest():
    _remember()
    llm = FakeLLM(fail={"big": "does not fit"}, crash={"mid": "CUDA out of memory"})
    job = _restore(llm, FakeTS())
    items = _by_model(job)
    assert items["big"]["status"] == "failed" and "does not fit" in items["big"]["error"]
    assert items["mid"]["status"] == "failed" and "out of memory" in items["mid"]["error"]
    assert items["small"]["status"] == "ready"
    assert items["chronos"]["status"] == "ready" and items["huge-ts"]["status"] == "ready"
    # The failures are still remembered -- a failed restore is not a user stop.
    assert {e["model"] for e in last_setup.entries()} == {"big", "mid", "small", "chronos", "huge-ts"}


def test_missing_checkpoint_and_missing_gpu_are_reported_not_launched():
    last_setup.record_llm("gone-model", {}, "1", {"gpus": "1"})
    last_setup.record_llm("big", {}, "5", {"gpus": "5"})      # explicit card that is gone
    last_setup.record_llm("mid", {}, None, {"gpus": "7"})     # auto: falls back to auto
    llm = FakeLLM()
    job = _restore(llm, FakeTS(), gpus=(0, 1, 2))
    items = _by_model(job)
    assert items["gone-model"]["status"] == "failed" and "no longer found" in items["gone-model"]["error"]
    assert items["big"]["status"] == "failed" and "GPU 5" in items["big"]["error"]
    assert items["mid"]["status"] == "ready"
    assert llm.calls == [("mid", {}, None)]


def test_occupied_card_is_shared_if_pinned_and_left_for_auto_otherwise():
    last_setup.record_llm("big", {}, "1", {"gpus": "1"})
    last_setup.record_llm("mid", {}, None, {"gpus": "2"})
    llm = FakeLLM()
    llm.instances["o1"] = FakeEngine("o1", "other", "1", state="running")
    llm.instances["o2"] = FakeEngine("o2", "other2", "2", state="running")
    job = _restore(llm, FakeTS())
    items = _by_model(job)
    # Pinned to GPU 1: launched there beside "other" -- the manager's share check decides.
    assert items["big"]["status"] == "ready" and "sharing GPU 1 with other" in items["big"]["note"]
    assert items["mid"]["status"] == "ready"
    assert llm.calls == [("big", {}, "1"), ("mid", {}, None)]


def test_pinned_share_refused_by_the_manager_is_reported():
    last_setup.record_llm("big", {}, "1", {"gpus": "1"})
    llm = FakeLLM(fail={"big": "GPU 1 is shared with other: use a memory ratio of at most 0.20"})
    llm.instances["o1"] = FakeEngine("o1", "other", "1", state="running")
    items = _by_model(_restore(llm, FakeTS()))
    assert items["big"]["status"] == "failed" and "at most 0.20" in items["big"]["error"]


def test_forecaster_refused_on_old_card_is_placed_automatically():
    last_setup.record_ts("chronos", None, {"gpu": "1"})
    ts = FakeTS(refuse_gpu="1")
    job = _restore(FakeLLM(), ts)
    assert ts.calls == [("chronos", "1"), ("chronos", None)]
    assert _by_model(job)["chronos"]["status"] == "ready"


def test_load_timeout_moves_on():
    last_setup.record_llm("big", {}, "2", {"gpus": "2"})
    last_setup.record_llm("mid", {}, "1", {"gpus": "1"})

    class Slow(FakeLLM):
        def get(self, inst_id):
            inst = self.instances.get(inst_id)
            if inst is not None and inst.model_id == "mid":
                return super().get(inst_id)
            return inst  # "big" never finishes loading

    llm = Slow()
    job = _restore(llm, FakeTS(), ready_timeout_s=0.05)
    items = _by_model(job)
    assert items["big"]["status"] == "failed" and "still loading" in items["big"]["error"]
    assert items["mid"]["status"] == "ready"


def test_second_restore_while_running_is_refused():
    last_setup.record_llm("big", {}, "2", {"gpus": "2"})

    async def go():
        class Never(FakeLLM):
            def get(self, inst_id):
                return self.instances.get(inst_id)

        await last_setup.start_restore(Never(), FakeTS(), catalog=lambda: CATALOG, resolve=_resolve,
                                       gpu_indices=lambda: {2}, poll_s=0.01)
        with pytest.raises(RuntimeError):
            await last_setup.start_restore(FakeLLM(), FakeTS(), catalog=lambda: CATALOG,
                                           resolve=_resolve, gpu_indices=lambda: {2})
        await asyncio.sleep(0.05)
        last_setup._task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await last_setup._task

    asyncio.run(go())
    assert last_setup.current_job()["state"] == "done"


def test_forget_route(client):
    c, _ = client
    last_setup.record_llm("big", {}, "2", {"gpus": "2"})
    r = c.post("/api/engines/restore-last/forget", json={"kind": "llm", "model": "big"})
    assert r.status_code == 200 and r.json()["removed"] is True
    assert last_setup.entries() == []
