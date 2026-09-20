"""Unload everything: every engine, every forecaster, and any orphan still holding memory.

The ordinary per-engine Unload only reaches engines the control plane is tracking. That is
not always all of them: when an engine's front process exits but its scheduler/tokenizer
workers survive (a crash, a killed window, a restarted control plane), the workers keep their
VRAM and pinned RAM while the Console lists nothing -- observed with 60 GiB held across two
cards under an empty engine list. Those orphans still listen on the engine port range (each
engine binds its HTTP port and the next one up for its torch.distributed store), which is how
this finds them without guessing.

Safety: an orphan is only killed if it is a python.exe listening on a FreeToken engine port.
Command-line matching alone would be too broad -- `spawn_main` is how ANY Python
multiprocessing child looks, including the operator's own unrelated work.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

from .config import settings

# Engines take port pairs upward from engine_port (1919/1920, 1921/1922, ...).
ENGINE_PORT_SPAN = 80


def _listening_pids(lo: int, hi: int) -> dict[int, list[int]]:
    """pid -> engine ports it listens on, from one netstat call."""
    if sys.platform != "win32":
        return {}
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True,
                             timeout=20, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    found: dict[int, list[int]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[3] != "LISTENING":
            continue
        try:
            port = int(parts[1].rsplit(":", 1)[1])
            pid = int(parts[4])
        except (ValueError, IndexError):
            continue
        if lo <= port <= hi and pid > 0:
            found.setdefault(pid, []).append(port)
    return found


def _image_name(pid: int) -> str | None:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, timeout=10, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out or out.startswith("INFO:"):
        return None
    return out.split(",")[0].strip('"').lower()


def _kill_tree(pid: int) -> bool:
    try:
        r = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                           timeout=30, check=False)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def sweep_orphans(tracked_pids: set[int]) -> list[dict]:
    """Kill python processes still listening on engine ports that no tracked engine owns."""
    lo = settings.engine_port
    hi = settings.engine_port + ENGINE_PORT_SPAN
    me = os.getpid()
    killed = []
    for pid, ports in _listening_pids(lo, hi).items():
        if pid == me or pid in tracked_pids:
            continue
        name = _image_name(pid)
        if name != "python.exe":
            continue  # something else happens to use a port in our range: not ours to kill
        killed.append({"pid": pid, "ports": sorted(ports), "killed": _kill_tree(pid)})
    return killed


async def unload_all(manager, ts_manager) -> dict:
    """Stop every engine and forecaster concurrently, then sweep orphans. Returns a report."""
    from . import gpu

    started = time.time()
    try:
        before = {d["index"]: d["memory_used_bytes"] for d in gpu._query_sync()}  # noqa: SLF001
    except Exception:  # noqa: BLE001
        before = {}

    engines = [i for i in manager.all() if i.model_id is not None or i.is_alive()]
    names = {i.instance_id: i.model_id for i in engines}

    async def stop_engine(inst) -> dict:
        try:
            await manager.stop(inst.instance_id)
            return {"instance": inst.instance_id, "model": names.get(inst.instance_id), "ok": True}
        except Exception as exc:  # noqa: BLE001 -- keep going; report it
            return {"instance": inst.instance_id, "model": names.get(inst.instance_id),
                    "ok": False, "error": str(exc)}

    # Concurrently: a 60 GiB offloaded engine takes a while to release pinned memory, and
    # there is no reason for the second one to wait behind it.
    stopped = await asyncio.gather(*(stop_engine(i) for i in engines))

    forecasters = [s["model_id"] for s in ts_manager.statuses()]
    await ts_manager.shutdown()

    tracked = {i.pid for i in manager.all() if getattr(i, "pid", None)}
    orphans = await asyncio.to_thread(sweep_orphans, tracked)

    return {
        "engines": stopped,
        "forecasters": forecasters,
        "orphans": orphans,
        "vram_before": before,
        "seconds": round(time.time() - started, 1),
    }
