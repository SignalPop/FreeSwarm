"""A second engine on a card already running one: allowed only beside the first one's headroom."""

from __future__ import annotations

import asyncio

import pytest

from app import engine as eng
from app import gpu
from app.engine import EngineManager, LaunchError, preflight_share

GIB = 2**30


class Occupant:
    def __init__(self, model="Muse-Glimmer-30B", gpus="1", state="running", ratio=0.8):
        self.model_id, self.instance_id, self.gpus, self.state = model, "eng1", gpus, state
        self.options = {"memory_ratio": ratio} if ratio is not None else {}

    def is_alive(self):
        return True


def _card(monkeypatch, free_gib, total_gib=48.0, index=1):
    monkeypatch.setattr(gpu, "_query_sync", lambda: [{
        "index": index,
        "memory_total_bytes": int(total_gib * GIB),
        "memory_used_bytes": int((total_gib - free_gib) * GIB),
    }])


def test_small_model_beside_glimmer_with_a_low_ratio_is_allowed(monkeypatch):
    # The A6000 running Glimmer at ratio 0.8 had 24.5 GiB free; Glimmer keeps ~9.6 GiB.
    _card(monkeypatch, 24.5)
    preflight_share("1", {"memory_ratio": 0.3}, [Occupant()])


def test_default_ratio_would_eat_the_occupants_headroom(monkeypatch):
    _card(monkeypatch, 24.5)
    with pytest.raises(LaunchError) as exc:
        preflight_share("1", {}, [Occupant()])
    msg = str(exc.value)
    # (24.5 - 9.6 kept by Glimmer - 2 for its own activations) / 24.5 = 0.526
    assert "shared with Muse-Glimmer-30B" in msg and "at most 0.52" in msg


def test_card_with_no_room_beyond_the_headroom_is_refused(monkeypatch):
    # GPU 2 running offloaded Qwen3.6 had 4.5 GiB free -- less than the 9.6 GiB it keeps.
    _card(monkeypatch, 4.5, index=2)
    with pytest.raises(LaunchError) as exc:
        preflight_share("2", {"memory_ratio": 0.1}, [Occupant("Qwen3.6-35B-A3B", "2")])
    assert "no room for a second engine" in str(exc.value)


def test_occupant_without_a_ratio_is_judged_at_the_engine_default(monkeypatch):
    # Default 0.9 keeps 4.8 GiB of a 48 GiB card; 0.5 of 24.5 free leaves 12.25 GiB.
    _card(monkeypatch, 24.5)
    preflight_share("1", {"memory_ratio": 0.5}, [Occupant(ratio=None)])


def test_occupant_still_loading_is_refused(monkeypatch):
    _card(monkeypatch, 40.0)
    with pytest.raises(LaunchError) as exc:
        preflight_share("1", {"memory_ratio": 0.2}, [Occupant(state="starting")])
    assert "still loading" in str(exc.value)


def test_manager_checks_an_explicit_shared_card_and_leaves_empty_ones_alone(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(eng, "preflight_share", lambda g, o, occ: calls.append(g))

    async def fake_start(self, model, options=None):
        return {"instance_id": self.instance_id, "gpus": self.gpus}

    monkeypatch.setattr(eng.EngineSupervisor, "start", fake_start)
    monkeypatch.setattr(EngineManager, "_free_port", lambda self: 30000)
    m = EngineManager()
    m._instances["eng1"] = Occupant()  # noqa: SLF001

    asyncio.run(m.start("Qwen3-0.6B", {"memory_ratio": 0.2}, "1"))
    assert calls == ["1"]
    asyncio.run(m.start("tiny", {}, "0"))
    assert calls == ["1"]  # GPU 0 is empty: nothing to share
