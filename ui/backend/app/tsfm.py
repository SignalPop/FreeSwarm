"""Time-series forecasting models: a second category of model, beside the LLM engines.

They are not served by the FreeToken engine (which is for causal LMs). Each loaded one is a
`tsfm_server.py` process in `.venv-ts`, pinned to one GPU, and this module owns their
lifecycle the way `engine.py` owns the LLM engines.

**The one rule that shapes everything here: never starve a running model.** Unlike LLM
engines there is no one-per-GPU restriction -- a 783 MiB forecaster is meant to sit on a card
beside a 48 GiB-class LLM, in the headroom the engine left. So placement is decided purely on
measured free VRAM, with a reserve kept back:

* the card must have (estimated need + RESERVE) free *now* -- the reserve is what the LLM
  engine on that card still has for transient activations and CUDA graph replays;
* a card whose LLM engine is still *starting* is refused outright. An engine sizes its KV
  cache from the free memory it sees at startup, so a forecaster landing mid-load could leave
  it with less than it planned for.

Loading is a deliberate human action (the Models page); agents only use what is loaded.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .config import settings

_GIB = 1 << 30

TS_PYTHON = Path(
    os.getenv("FREESWARM_TS_PYTHON")
    or (Path(settings.repo_root) / ".venv-ts" / "Scripts" / "python.exe")
)
SERVER = Path(settings.repo_root) / "ui" / "backend" / "tsfm_server.py"

PORT_LO, PORT_HI = 1960, 1999
# Kept free on the card after a forecaster loads, for the LLM engine already there.
RESERVE_BYTES = int(float(os.getenv("FREESWARM_TS_VRAM_RESERVE_GB", "1.5")) * _GIB)
# CUDA context + the forecaster's activations, on top of its weights.
OVERHEAD_BYTES = int(0.9 * _GIB)
LOAD_TIMEOUT_S = 180.0

# Architectures that are time-series models, and whether this build can serve each one.
TS_ARCHITECTURES: dict[str, bool] = {
    "Chronos2Model": True,
    "ChronosBoltModelForForecasting": True,
    "ChronosModelForForecasting": True,
    "PatchTSTForPrediction": True,
    # Recognised as time-series (so they are not offered as LLMs) but with no adapter yet.
    "PatchTSMixerForPrediction": False,
    "TimesFmModelForPrediction": False,
    "TimesFm2_5ModelForPrediction": False,
    "TinyTimeMixerForPrediction": False,
}

_NOTES = {
    "Chronos2Model": (
        "Chronos-2: multivariate and covariate-aware. Forecast one or several series jointly while "
        "reading other series (GEX, Greeks, flows...) as inputs. Quantiles; context up to 8192 steps."
    ),
    "ChronosBoltModelForForecasting": (
        "Zero-shot, single series, returns quantiles (a distribution). The general-purpose one."
    ),
    "ChronosModelForForecasting": "Zero-shot, single series, returns quantiles.",
    "PatchTSTForPrediction": (
        "Fixed to the channel count and context it was trained on -- this IBM checkpoint is "
        "7 channels x 512 steps (ETTh1). Not a general forecaster; point forecasts only."
    ),
}


def classify(cfg: dict) -> dict | None:
    """Category info for a checkpoint config, or None if it is not a time-series model."""
    archs = cfg.get("architectures") or []
    arch = archs[0] if archs else None
    if arch in TS_ARCHITECTURES:
        return {
            "category": "timeseries",
            "ts_servable": TS_ARCHITECTURES[arch],
            "ts_note": _NOTES.get(arch)
            or "Recognised time-series architecture; no adapter in this build yet.",
        }
    if "chronos_config" in cfg:
        return {"category": "timeseries", "ts_servable": True, "ts_note": _NOTES["ChronosModelForForecasting"]}
    # Kronos (NeoQuasar) has no `architectures`; its configs are recognisable by their keys.
    if {"s1_bits", "s2_bits", "d_in", "group_size"} <= cfg.keys():
        return {"category": "timeseries", "ts_servable": False,
                "ts_note": "Kronos tokenizer -- loaded automatically with a Kronos model; not a model on its own."}
    if {"s1_bits", "s2_bits", "n_layers", "learn_te"} <= cfg.keys():
        return {"category": "timeseries", "ts_servable": True,
                "ts_note": ("Kronos: foundation model for financial candlesticks (OHLCV), trained on 45 exchanges. "
                            "Forecasts whole candles from up to 512 bars; probabilistic (sampled paths). Needs its "
                            "tokenizer (NeoQuasar/Kronos-Tokenizer-base, or -2k for Kronos-mini) downloaded too.")}
    return None


class TsError(ValueError):
    """A request that should become a 400."""


@dataclass
class TsInstance:
    id: str
    model_id: str
    path: str
    gpu: str
    port: int
    state: str = "starting"  # starting|running|error|stopped
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    health: dict[str, Any] | None = None
    proc: subprocess.Popen | None = None
    log: list[str] = field(default_factory=list)
    # Live activity, for the Swarm page's LEDs: forecasts are proxied through here, so this
    # is the one place that sees every call.
    in_flight: int = 0
    calls: int = 0
    last_call_at: float | None = None
    last_seconds: float | None = None

    def status(self) -> dict:
        return {
            "id": self.id,
            "model_id": self.model_id,
            "gpu": self.gpu,
            "port": self.port,
            "state": self.state,
            "error": self.error,
            "uptime_s": int(time.time() - self.started_at),
            "health": self.health,
            "log_tail": self.log[-12:],
            "in_flight": self.in_flight,
            "calls": self.calls,
            "last_call_at": self.last_call_at,
            "last_seconds": self.last_seconds,
        }


def _weights_bytes(path: Path) -> int:
    total = 0
    for child in path.iterdir():
        if child.is_file() and child.suffix in (".safetensors", ".bin"):
            total += child.stat().st_size
    return total


def _port_free(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


class TsManager:
    def __init__(self) -> None:
        self._instances: dict[str, TsInstance] = {}
        self._lock = asyncio.Lock()

    def statuses(self) -> list[dict]:
        return [i.status() for i in self._instances.values()]

    def running(self) -> list[TsInstance]:
        return [i for i in self._instances.values() if i.state == "running"]

    def for_model(self, model_id: str) -> TsInstance | None:
        return next((i for i in self.running() if i.model_id == model_id), None)

    # -- placement -------------------------------------------------------------------
    def _plan(self, path: Path, gpu: str | None) -> tuple[str, int]:
        """(gpu index, bytes needed) or raise TsError explaining exactly why not."""
        from . import gpu as gpumod
        from .engine import manager as llm_manager
        from .prefs import get_visible_devices

        need = int(_weights_bytes(path) * 1.0) + OVERHEAD_BYTES
        try:
            devices = gpumod._query_sync()  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001
            raise TsError(f"cannot read GPU memory ({exc}); refusing to guess") from None

        pool = {x.strip() for x in (get_visible_devices() or "").split(",") if x.strip()}
        loading = {
            str(e.gpus)
            for e in llm_manager.all()
            if e.state == "starting" and e.gpus is not None
        }

        def free(d: dict) -> int:
            return d["memory_total_bytes"] - d["memory_used_bytes"]

        candidates = [d for d in devices if not pool or str(d["index"]) in pool]
        if gpu is not None:
            candidates = [d for d in candidates if str(d["index"]) == str(gpu)]
            if not candidates:
                raise TsError(f"GPU {gpu} is not in the pool")

        refusals = []
        best = None
        for d in sorted(candidates, key=free, reverse=True):
            idx = str(d["index"])
            if idx in loading:
                refusals.append(
                    f"GPU {idx}: an LLM engine is still loading there -- it sizes its KV cache "
                    "from the memory it sees at startup, so wait until it is running"
                )
                continue
            if free(d) < need + RESERVE_BYTES:
                refusals.append(
                    f"GPU {idx}: {free(d) / _GIB:.1f} GiB free, needs "
                    f"{need / _GIB:.1f} GiB + {RESERVE_BYTES / _GIB:.1f} GiB kept for the "
                    "model already there"
                )
                continue
            best = idx
            break
        if best is None:
            raise TsError("no GPU can take this model without crowding a running one. " + "; ".join(refusals))
        return best, need

    # -- lifecycle -------------------------------------------------------------------
    async def start(self, model_id: str, path: Path, gpu: str | None = None) -> dict:
        if not TS_PYTHON.is_file():
            raise TsError(f"time-series interpreter not found at {TS_PYTHON}")
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
        info = classify(cfg)
        if info is None:
            raise TsError(f"{model_id} is not a time-series model")
        if not info["ts_servable"]:
            raise TsError(f"{model_id}: {info['ts_note']}")

        async with self._lock:
            for key in [k for k, v in self._instances.items() if v.state in ("stopped", "error")]:
                self._instances.pop(key)
            if self.for_model(model_id) or any(
                i.model_id == model_id and i.state == "starting" for i in self._instances.values()
            ):
                raise TsError(f"{model_id} is already loaded")

            chosen, need = self._plan(path, gpu)
            used = {i.port for i in self._instances.values()}
            port = next(
                (p for p in range(PORT_LO, PORT_HI + 1) if p not in used and _port_free(p)), None
            )
            if port is None:
                raise TsError("no free port for a time-series server")

            inst = TsInstance(id=f"ts{port}", model_id=model_id, path=str(path), gpu=chosen, port=port)
            env = {
                **os.environ,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",  # indices match nvidia-smi, like the engines
                "CUDA_VISIBLE_DEVICES": chosen,
                "PYTHONUNBUFFERED": "1",
            }
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            inst.proc = subprocess.Popen(
                [str(TS_PYTHON), str(SERVER), "--model", str(path), "--name", model_id, "--port", str(port)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                creationflags=flags,
            )
            inst.log.append(f"placed on GPU {chosen} (needs ~{need / _GIB:.1f} GiB)")
            self._instances[inst.id] = inst

        asyncio.create_task(self._pump_log(inst))
        await self._await_ready(inst)
        return inst.status()

    async def _pump_log(self, inst: TsInstance) -> None:
        proc = inst.proc
        if proc is None or proc.stdout is None:
            return
        while True:
            line = await asyncio.to_thread(proc.stdout.readline)
            if not line:
                break
            inst.log.append(line.rstrip())
            del inst.log[:-200]
        code = proc.poll()
        if inst.state in ("starting", "running"):
            inst.state = "error"
            # The real error is almost always the last traceback line the server printed.
            tail = next((l for l in reversed(inst.log) if l.strip()), "")
            inst.error = f"exited with code {code}: {tail}"

    async def _await_ready(self, inst: TsInstance) -> None:
        deadline = time.monotonic() + LOAD_TIMEOUT_S
        async with httpx.AsyncClient(timeout=3.0) as client:
            while time.monotonic() < deadline:
                if inst.state == "error":
                    raise TsError(f"{inst.model_id} failed to load: {inst.error}")
                try:
                    r = await client.get(f"http://127.0.0.1:{inst.port}/health")
                    if r.status_code == 200:
                        inst.health = r.json()
                        inst.state = "running"
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.5)
        await self.stop(inst.id)
        raise TsError(f"{inst.model_id} did not become ready within {LOAD_TIMEOUT_S:.0f}s")

    async def stop(self, inst_id: str) -> dict:
        inst = self._instances.get(inst_id)
        if inst is None:
            raise TsError(f"no time-series instance {inst_id!r}")
        proc = inst.proc
        inst.state = "stopped"
        if proc is not None and proc.poll() is None:
            if sys.platform == "win32":
                # /T: take any child the server spawned with it, or its VRAM stays held.
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=20
                )
            else:
                proc.kill()
            await asyncio.to_thread(proc.wait, 10)
        self._instances.pop(inst_id, None)
        return inst.status()

    async def shutdown(self) -> None:
        for key in list(self._instances):
            try:
                await self.stop(key)
            except Exception:  # noqa: BLE001 -- best effort on the way down
                pass

    async def refresh_health(self) -> None:
        """Update VRAM figures for the console; mark dead ones."""
        async with httpx.AsyncClient(timeout=2.0) as client:
            for inst in self.running():
                try:
                    r = await client.get(f"http://127.0.0.1:{inst.port}/health")
                    inst.health = r.json()
                except httpx.HTTPError:
                    pass

    async def forecast(self, model_id: str | None, payload: dict) -> dict:
        running = self.running()
        if not running:
            raise TsError("no time-series model is loaded -- load one from the Models page")
        inst = self.for_model(model_id) if model_id else running[0]
        if inst is None:
            names = ", ".join(i.model_id for i in running)
            raise TsError(f"{model_id} is not loaded. Loaded: {names}")
        inst.in_flight += 1
        started = time.time()
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(f"http://127.0.0.1:{inst.port}/forecast", json=payload)
        finally:
            inst.in_flight -= 1
            inst.calls += 1
            inst.last_call_at = time.time()
            inst.last_seconds = round(time.time() - started, 3)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise TsError(f"{inst.model_id}: {detail}")
        return r.json()


ts_manager = TsManager()
