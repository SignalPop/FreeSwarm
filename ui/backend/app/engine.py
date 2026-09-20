"""Lifecycle supervisor for the FreeToken engine subprocess.

The engine is `python -m freetoken ...`: a uvicorn frontend plus spawned scheduler and
tokenizer workers, all talking over ZMQ loopback. This module owns exactly one of them at a
time and exposes start / stop / status to the HTTP layer.

Three Windows specifics shape the implementation:

* The engine must inherit a **vcvars64 environment** so nvcc can find cl.exe when it JITs
  kernels on first use (see winenv.py).
* It is spawned in its **own process group** (CREATE_NEW_PROCESS_GROUP) so a Ctrl-C in the
  terminal that launched the UI does not also tear the engine down.
* Stopping needs `taskkill /T`: the scheduler and tokenizer workers are separate processes,
  and terminating only the parent orphans them while they still hold VRAM.

Argument handling is deliberately an allow-list (`_FLAG_SPEC`). The control plane has no
authentication -- it is loopback-only -- but "POST a JSON body that becomes argv" is a bad
shape to leave open even on loopback, so unknown keys are rejected rather than forwarded.
"""

from __future__ import annotations

import asyncio
import itertools
import shlex
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .catalog import resolve_model_path
from .diagnose import diagnose
from .config import settings
from .prefs import get_visible_devices
from .winenv import engine_environment

# ---------------------------------------------------------------------------------------
# Launch-argument allow-list.
#
# Each entry maps a JSON key to (cli flag, kind). "flag" is a valueless switch emitted when
# the value is truthy; the others coerce and validate. Anything not listed here is refused
# by build_argv.
# ---------------------------------------------------------------------------------------
_FLAG_SPEC: dict[str, tuple[str, str]] = {
    "tp_size": ("--tp-size", "int"),
    "dtype": ("--dtype", "choice:auto,bfloat16,float16,float32"),
    "attention_backend": (
        "--attention-backend",
        "choice:auto,triton,fi,fa,trtllm,dsv4_sparse,dsa,m3_sparse",
    ),
    "moe_backend": ("--moe-backend", "choice:fused,offload,cpu,hybrid,triton"),
    "expert_load": ("--expert-load", "choice:serial,parallel"),
    "num_pages": ("--num-pages", "int"),
    "num_tokens": ("--num-tokens", "int"),
    "page_size": ("--page-size", "int"),
    "max_running_requests": ("--max-running-requests", "int"),
    "max_seq_len_override": ("--max-seq-len-override", "int"),
    "max_output_tokens": ("--max-output-tokens", "int"),
    "max_prefill_length": ("--max-prefill-length", "int"),
    "memory_ratio": ("--memory-ratio", "float"),
    "moe_cache_size": ("--moe-cache-size", "int"),
    "moe_cache_rate": ("--moe-cache-rate", "float"),
    "moe_cache_policy": ("--moe-cache-policy", "choice:lru"),
    "moe_cpu_threads": ("--moe-cpu-threads", "int"),
    "moe_cpu_layers": ("--moe-cpu-layers", "int"),
    "kv_reserve_tokens": ("--kv-reserve-tokens", "int"),
    "cuda_graph_max_bs": ("--cuda-graph-max-bs", "int"),
    "served_model_name": ("--served-model-name", "token"),
    "tool_call_parser": ("--tool-call-parser", "token"),
    "reasoning_parser": ("--reasoning-parser", "token"),
    "cache_type": ("--cache-type", "token"),
    "moe_cache_auto": ("--moe-cache-auto", "flag"),
    "enable_cache_report": ("--enable-cache-report", "flag"),
    "disable_moe_prefill_overlap": ("--disable-moe-prefill-overlap", "flag"),
    "disable_pynccl": ("--disable-pynccl", "flag"),
}

# Conservative: model names and parser ids are identifiers, not shell fragments.
_TOKEN_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/+")


def _checkpoint_bytes(path: Path) -> int:
    """Total weight bytes in a checkpoint directory -- what the load has to read."""
    total = 0
    try:
        for child in path.iterdir():
            if child.is_file() and child.suffix in (".safetensors", ".gguf", ".bin"):
                total += child.stat().st_size
    except OSError:
        pass
    return total


def _memory_baseline() -> tuple[int, int]:
    """(VRAM used, host RAM used) right now, across the whole machine.

    Machine-wide rather than per-process on purpose: under TCC there is no per-process
    accounting available through nvidia-smi, and the engine's workers are separate
    processes anyway. The baseline subtracts whatever was already in use.
    """
    from . import gpu

    try:
        devices = gpu._query_sync()  # noqa: SLF001 - sync path; this runs before the spawn
        vram = sum(d["memory_used_bytes"] for d in devices)
    except Exception:  # noqa: BLE001 - a missing nvidia-smi must not block a launch
        vram = 0
    host = gpu.host_memory()
    ram = max(0, host["total_bytes"] - host["available_bytes"])
    return vram, ram


class LaunchError(ValueError):
    """A launch request that should become a 400, not a 500."""


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """True if something is already listening on `port`.

    SO_REUSEADDR is deliberately NOT set: on Windows it would let the bind succeed against
    a live listener and report the port free.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        try:
            sock.bind((host, port))
        except OSError:
            return True
    return False


def _find_port_holders(port: int) -> list[str]:
    """Best-effort 'what is holding this port', for the error message."""
    if sys.platform != "win32":
        return []
    try:
        out = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return []
    pids: set[str] = set()
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[3] == "LISTENING" and parts[1].endswith(f":{port}"):
            pids.add(parts[4])
    return sorted(pids)


def _expert_bytes(model_path: Path) -> int:
    """Bytes of MoE expert weights in a checkpoint -- what an offloaded engine pins in RAM.

    Read from the safetensors headers only (no tensor data), so this costs a few file opens
    even for a 67 GiB checkpoint. Falls back to the whole checkpoint size when the layout
    cannot be parsed: experts dominate an offloaded MoE model, so that over-estimates
    slightly rather than waving through a launch that cannot fit.
    """
    import json
    import struct

    total = experts = 0
    try:
        shards = sorted(model_path.glob("*.safetensors"))
    except OSError:
        return 0
    if not shards:
        return 0
    for shard in shards:
        try:
            with open(shard, "rb") as fh:
                length = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(length))
        except (OSError, ValueError, struct.error):
            return _checkpoint_bytes(model_path)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            try:
                begin, end = meta["data_offsets"]
            except (KeyError, TypeError, ValueError):
                continue
            size = end - begin
            total += size
            if ".experts." in name or ".mlp.expert" in name:
                experts += size
    if total <= 0:
        return _checkpoint_bytes(model_path)
    return experts or total


# Backends that keep expert weights in HOST RAM rather than VRAM.
_OFFLOAD_BACKENDS = {"offload", "cpu", "hybrid"}


def host_pin_budget_bytes(total_bytes: int) -> int:
    """Return the configured shared page-locked host-memory budget."""
    import os

    try:
        absolute = float(
            os.getenv("FREETOKEN_HOST_RAM_GB")
            or os.getenv("FREETOKEN_HOST_PIN_GB")
            or 0
        ) * 2**30
    except ValueError:
        absolute = 0.0
    try:
        fraction = float(
            os.getenv("FREETOKEN_HOST_RAM_FRACTION")
            or os.getenv("FREETOKEN_HOST_PIN_FRACTION")
            or 0.52
        )
    except ValueError:
        fraction = 0.52
    return int(absolute if absolute > 0 else total_bytes * fraction)


def preflight_host_ram(
    model_path: Path,
    options: dict[str, Any],
    display_name: str | None = None,
    others: list[tuple[str, Path, dict[str, Any]]] | None = None,
) -> None:
    """Refuse an offloaded launch whose expert banks cannot fit in host RAM ALONGSIDE the
    engines already running.

    Host RAM is the one resource every engine shares. VRAM is per-card and the manager
    already keeps one engine per GPU, but expert banks are pinned (``cudaHostRegister``) and
    non-pageable, so two large offloaded models on two different GPUs still compete for the
    same RAM. On a 128 GiB box Qwen3.6 (61.5 GiB of experts) and gpt-oss-120b (56.8 GiB)
    cannot both be resident, however many GPUs are free.

    Without this the second launch reads its whole checkpoint off disk, pins as much as it
    can, and dies minutes in with ``cudaHostRegister failed for 0.5 GiB`` -- a message about
    the last half-gigabyte that says nothing about the 118 GiB already committed.
    """
    from . import gpu

    backend = str(options.get("moe_backend") or "fused")
    if backend not in _OFFLOAD_BACKENDS:
        return  # fused keeps experts in VRAM; host RAM is not the constraint

    need = _expert_bytes(model_path)
    if need <= 0:
        return

    host = gpu.host_memory()
    total = host.get("total_bytes") or 0
    if total <= 0:
        return  # unknown (non-Windows) -- do not block on a number we do not have

    # NOT "total RAM minus headroom". There is a hard ceiling on how much host memory can be
    # cudaHostRegister'd, well below physical RAM, and free RAM does not predict it.
    #
    # Measured on this machine (127.9 GiB RAM, A6000s in TCC mode -- so this is NOT the WDDM
    # shared-memory limit, which does not apply to a TCC device):
    #
    #   solo, 1 GiB chunks   -> 69 GiB, then `invalid argument`, ~58 GiB still free
    #   solo, 4 GiB chunks   -> 68 GiB  (same bytes, 17 regions vs 69: a BYTE ceiling,
    #                                    not a limit on the number of registrations)
    #   10 GiB held by another process -> the next process reached only 62 GiB
    #
    # That last line is the case that matters: it reproduces the real failure exactly.
    # gpt-oss-20b pins 9.5 GiB, Qwen3.6 needs 61.5, and the load died on its final 1.0 GiB
    # bank. So the ceiling is shared enough between processes to matter, even across two
    # different GPUs, and budgeting the SUM against ~52% of RAM gives the right verdict for
    # every combination measured:
    #
    #   Qwen3.6 alone 61.5 <= 66.5  allowed  (solo ceiling 68-69: loads)
    #   Qwen3.6 + gpt-oss-20b 71.0 > 66.5  refused (measured: fails)
    #   gpt-oss-120b + gpt-oss-20b 66.3 <= 66.5  allowed (measured: 62 GiB was reachable)
    #
    # The exact mechanism is a driver/OS pinned-memory limit rather than anything this code
    # can query, so 0.52 is an empirical fit, not a derived constant. RE-MEASURE on different
    # hardware; override with the engine's canonical FREETOKEN_HOST_RAM_FRACTION or
    # FREETOKEN_HOST_RAM_GB names. Keep the old PIN aliases for existing UI deployments.
    budget = host_pin_budget_bytes(total)

    committed = 0
    holders: list[str] = []
    for name, path, opts in others or []:
        if str(opts.get("moe_backend") or "fused") not in _OFFLOAD_BACKENDS:
            continue
        size = _expert_bytes(path)
        if size <= 0:
            continue
        committed += size
        holders.append(f"{name} ({size / 2**30:.1f} GiB)")

    if committed + need <= budget:
        return

    label = display_name or model_path.name
    gib = 2**30
    detail = (
        f" Already pinned by {', '.join(holders)}." if holders else ""
    )
    raise LaunchError(
        f"{label} needs ~{need / gib:.1f} GiB of page-locked host RAM for its experts, but "
        f"only ~{max(0.0, budget - committed) / gib:.1f} GiB of this machine's "
        f"~{budget / gib:.0f} GiB pinning limit is left.{detail} "
        f"That ceiling is a driver limit on page-locked memory, not free RAM -- measured on "
        f"this machine, registration stops well below total RAM and fails with "
        f"'cudaHostRegister failed' partway through the load however much RAM is free. "
        f"Unload one of those models, or serve a quantized checkpoint (FP8/NVFP4 experts "
        f"are half or a quarter the size)."
    )


def preflight_fit(
    model_path: Path,
    options: dict[str, Any],
    gpus: str | None = None,
    display_name: str | None = None,
) -> None:
    """Refuse a configuration that cannot physically work on this machine.

    Tensor parallelism is unavailable on native Windows (pynccl links -lnccl, and NCCL has
    no Windows build), so the weights cannot be split across the two cards. That leaves:

      * MoE checkpoints -- `--moe-backend offload` streams the experts from host RAM, so
        only the non-expert weights need to fit in VRAM.
      * dense checkpoints -- must fit on a single GPU, full stop.
    """
    from . import gpu
    from .catalog import model_meta_for
    from .prefs import get_visible_devices

    # A hub checkpoint's directory is its snapshot hash, so `model_path.name` renders as
    # "6cee5e81ee8391..." -- unrecognisable in an error the user has to act on. The catalog
    # id the launch was requested with is the name they actually know it by.
    label = display_name or model_path.name

    size = _checkpoint_bytes(model_path)
    if size <= 0:
        return  # nothing measurable (GGUF single file, odd layout) -- let the engine try

    try:
        devices = gpu._query_sync()  # noqa: SLF001
    except Exception:  # noqa: BLE001
        return
    if not devices:
        return

    selected = gpus if gpus is not None else get_visible_devices()
    if selected:
        wanted = {int(x) for x in selected.split(",") if x.strip().isdigit()}
        devices = [d for d in devices if d["index"] in wanted] or devices

    # Largest single card, because that is the unit a model must fit into.
    best = max(devices, key=lambda d: d["memory_total_bytes"] - d["memory_used_bytes"])
    free = best["memory_total_bytes"] - best["memory_used_bytes"]

    backend = str(options.get("moe_backend") or "fused")
    meta = model_meta_for(model_path)
    is_moe = bool(meta.get("is_moe"))

    if meta.get("category") == "timeseries":
        raise LaunchError(
            f"{label} is a time-series forecasting model, not an LLM -- load it from the "
            "Time series section of the Models page, which places it on a GPU with room."
        )
    if not meta.get("supported", True):
        raise LaunchError(meta.get("unsupported_reason") or "unsupported architecture")

    if int(options.get("tp_size") or 1) > 1:
        raise LaunchError(
            "tensor parallel size > 1 cannot work on Windows: it needs NCCL, which has no "
            "Windows build. Use --tp-size 1, and --moe-backend offload for a model larger "
            "than one card."
        )

    if backend != "fused":
        return  # offload / cpu / hybrid keep the experts out of VRAM

    # Weights plus room for KV cache, CUDA graphs and activations. 0.88 matches what an
    # A6000 actually had left after loading gpt-oss-20b (13.5 GiB used for 12.8 GiB of
    # weights).
    needed = size * 1.06
    if needed <= free * 0.95:
        return

    gib = 2 ** 30
    if is_moe:
        raise LaunchError(
            f"{label} is {size / gib:.1f} GiB but the largest selected GPU has "
            f"{free / gib:.1f} GiB free, and Windows cannot split a model across GPUs "
            f"(no NCCL). Use MoE backend 'offload' -- it pins the experts in host RAM and "
            f"streams them over PCIe, so only the non-expert weights need VRAM."
        )
    raise LaunchError(
        f"{label} is {size / gib:.1f} GiB but the largest selected GPU has "
        f"{free / gib:.1f} GiB free. It is a dense model, so there is nothing to offload "
        f"and Windows cannot split it across GPUs (no NCCL). This checkpoint cannot be "
        f"served on this machine."
    )


def preflight_ports(base_port: int | None = None) -> None:
    """Refuse to launch when the engine's ports are already taken.

    The engine needs two: `server_port` for its HTTP API and `server_port + 1` for the
    torch.distributed rendezvous store. A stale worker holding the second one makes the
    engine die several seconds into startup with a bare
    `DistNetworkError: ... failed to bind ... (system error: 10048)`, which says nothing
    about the actual cause. Checking up front turns that into an actionable message.

    Orphans are the usual cause: the engine spawns scheduler/tokenizer workers as separate
    processes, so killing only the parent leaves them holding both the port and their VRAM.
    """
    base = base_port or settings.engine_port
    for port, role in (
        (base, "engine HTTP API"),
        (base + 1, "torch.distributed rendezvous"),
    ):
        if not _port_in_use(port):
            continue
        holders = _find_port_holders(port)
        who = f" (held by PID {', '.join(holders)})" if holders else ""
        raise LaunchError(
            f"port {port} ({role}) is already in use{who}. This is usually an orphaned "
            "engine worker from a previous run that was killed without its process tree. "
            f"Stop it with: taskkill /PID {holders[0] if holders else '<pid>'} /T /F"
        )


def _coerce(key: str, flag: str, kind: str, value: Any) -> list[str]:
    if kind == "flag":
        return [flag] if value else []
    if value is None or value == "":
        return []
    if kind == "int":
        try:
            n = int(value)
        except (TypeError, ValueError):
            raise LaunchError(f"{key} must be an integer, got {value!r}") from None
        if not 0 <= n <= 1 << 31:
            raise LaunchError(f"{key} out of range: {n}")
        return [flag, str(n)]
    if kind == "float":
        try:
            f = float(value)
        except (TypeError, ValueError):
            raise LaunchError(f"{key} must be a number, got {value!r}") from None
        if not 0.0 <= f <= 1e6:
            raise LaunchError(f"{key} out of range: {f}")
        return [flag, repr(f)]
    if kind.startswith("choice:"):
        allowed = kind.split(":", 1)[1].split(",")
        if str(value) not in allowed:
            raise LaunchError(f"{key} must be one of {', '.join(allowed)}; got {value!r}")
        return [flag, str(value)]
    if kind == "token":
        s = str(value)
        if not s or set(s) - _TOKEN_OK:
            raise LaunchError(f"{key} contains characters that are not allowed: {value!r}")
        return [flag, s]
    raise LaunchError(f"internal: unknown kind {kind} for {key}")


def _served_name_for(model_path: Path) -> str:
    """A readable name for a checkpoint, for --served-model-name.

    HuggingFace cache layout is models--org--name/snapshots/<sha>, so the directory name
    is a hash; walk up to the repo directory and rebuild org/name from it.
    """
    for part in model_path.parts:
        if part.startswith("models--"):
            return part.removeprefix("models--").replace("--", "/")
    return model_path.name


def build_argv(
    model_path: Path, options: dict[str, Any], port: int | None = None
) -> list[str]:
    """argv for `python -m freetoken`, from a validated option dict."""
    argv = [
        str(settings.venv_python),
        "-m", "freetoken",
        "--model-path", str(model_path),
        "--host", settings.engine_host,
        "--port", str(port or settings.engine_port),
        # The browser reaches the engine through this control plane, but the allow-list is
        # set anyway so a direct fetch from the dev server also works.
        "--cors-origins", ",".join(settings.frontend_origins),
    ]
    unknown = sorted(set(options) - set(_FLAG_SPEC))
    if unknown:
        raise LaunchError(f"unsupported option(s): {', '.join(unknown)}")
    # Without this the engine names itself after the checkpoint DIRECTORY, which for a
    # HuggingFace cache entry is a snapshot hash like 6cee5e81ee83.... That is what shows
    # up in /v1/models and what an agent would have to type to route to it.
    options = dict(options)
    options.setdefault("served_model_name", _served_name_for(model_path))
    for key, value in options.items():
        flag, kind = _FLAG_SPEC[key]
        argv.extend(_coerce(key, flag, kind, value))
    return argv


@dataclass
class LogRing:
    """Bounded, cursor-addressable line buffer shared by the reader threads and HTTP."""

    capacity: int
    _lines: deque = field(default_factory=deque)
    _counter: itertools.count = field(default_factory=itertools.count)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, stream: str, text: str) -> None:
        with self._lock:
            if len(self._lines) >= self.capacity:
                self._lines.popleft()
            self._lines.append(
                {"seq": next(self._counter), "ts": time.time(), "stream": stream, "text": text}
            )

    def since(self, cursor: int, limit: int = 500) -> tuple[list[dict], int]:
        with self._lock:
            out = [e for e in self._lines if e["seq"] >= cursor][:limit]
            nxt = out[-1]["seq"] + 1 if out else cursor
            return out, nxt

    def texts(self) -> list[str]:
        """Captured lines as plain strings, oldest first -- what diagnose() reads."""
        with self._lock:
            return [e["text"] for e in self._lines]

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()


class EngineSupervisor:
    """Owns at most one engine process. Public methods are safe to call concurrently."""

    def __init__(
        self,
        instance_id: str = "default",
        port: int | None = None,
        gpus: str | None = None,
    ) -> None:
        self.instance_id = instance_id
        # Each engine needs two consecutive ports; the manager spaces instances by two.
        self.port = port or settings.engine_port
        # None means "whatever the global GPU preference says"; the manager sets this to a
        # single card so two instances cannot both claim the same GPU.
        self.gpus = gpus
        self._proc: subprocess.Popen | None = None
        self._lock = asyncio.Lock()
        self.logs = LogRing(settings.log_ring_size)
        self.state: str = "stopped"  # stopped|starting|running|stopping|error
        self.error: str | None = None
        # Populated once when a process dies, from its captured output.
        self.diagnosis: dict | None = None
        self.model_id: str | None = None
        self.model_path: str | None = None
        # What the engine calls itself (derived from the checkpoint dir). Routing accepts
        # either this or the catalog id, because clients know it by different names.
        self.served_name: str | None = None
        self.options: dict[str, Any] = {}
        self.argv: list[str] = []
        self.started_at: float | None = None
        # Baselines captured at launch, used to estimate load progress from memory growth
        # when the engine's own /health reports no byte counts.
        self.model_size_bytes: int = 0
        self.baseline_vram_bytes: int = 0
        self.baseline_ram_bytes: int = 0
        self._readers: list[threading.Thread] = []

    # -- introspection -------------------------------------------------------------
    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc and self._proc.poll() is None else None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict:
        # A process that died on its own (bad flags, OOM, driver fault) must surface as an
        # error rather than reading "running" forever.
        if self.state in {"starting", "running"} and self._proc is not None:
            code = self._proc.poll()
            if code is not None:
                self.state = "error"
                # Give the reader threads a moment to drain the pipes: the process is
                # already gone, but its final traceback may still be in flight, and that
                # traceback is the entire point of the diagnosis.
                for _ in range(20):
                    if any(t.is_alive() for t in self._readers):
                        time.sleep(0.05)
                    else:
                        break
                result = diagnose(self.logs.texts(), code)
                self.diagnosis = {**result.as_dict(), "exit_code": code}
                self.error = result.summary
                self.logs.append("ui", f"engine exited with code {code}: {result.summary}")
                if result.hint:
                    self.logs.append("ui", f"hint: {result.hint}")
        return {
            "instance_id": self.instance_id,
            "port": self.port,
            "gpus": self.gpus if self.gpus is not None else get_visible_devices(),
            "state": self.state,
            "error": self.error,
            "diagnosis": self.diagnosis,
            "pid": self.pid,
            "model_id": self.model_id,
            "served_name": self.served_name,
            "model_path": self.model_path,
            "options": self.options,
            "command": " ".join(shlex.quote(a) for a in self.argv),
            "started_at": self.started_at,
            "model_size_bytes": self.model_size_bytes,
            "baseline_vram_bytes": self.baseline_vram_bytes,
            "baseline_ram_bytes": self.baseline_ram_bytes,
            "uptime_s": int(time.time() - self.started_at) if self.started_at else 0,
            "engine_url": f"http://{settings.engine_host}:{self.port}",
        }

    def mark_running(self) -> None:
        """Called by the health poller once the engine reports ready."""
        if self.state == "starting" and self.is_alive():
            self.state = "running"

    # -- lifecycle -----------------------------------------------------------------
    def _pump(self, stream: Any, name: str) -> None:
        try:
            for line in iter(stream.readline, ""):
                self.logs.append(name, line.rstrip("\r\n"))
        except (ValueError, OSError):
            pass
        finally:
            try:
                stream.close()
            except OSError:
                pass

    async def start(self, model: str, options: dict[str, Any] | None = None) -> dict:
        async with self._lock:
            if self.is_alive():
                raise LaunchError(
                    f"engine already running (pid {self.pid}); stop it before starting another model"
                )
            if not settings.venv_python.is_file():
                raise LaunchError(
                    f"engine interpreter not found at {settings.venv_python}. "
                    "Create the venv or set FREESWARM_UI_PYTHON."
                )

            preflight_ports(self.port)

            model_path = resolve_model_path(model)
            opts = dict(options or {})
            # Catch "cannot fit" here rather than as a CUDA OOM several minutes into the
            # load, once the whole checkpoint has been read off disk.
            preflight_fit(model_path, opts, self.gpus, display_name=model)
            # Host RAM is shared by every engine, so this one has to be judged against what
            # the others have already pinned -- the manager is the only thing that knows.
            preflight_host_ram(
                model_path,
                opts,
                display_name=model,
                others=[
                    (inst.model_id or inst.instance_id, Path(inst.model_path), inst.options)
                    for inst in manager.all()
                    if inst is not self and inst.model_path and inst.state != "stopped"
                ],
            )
            argv = build_argv(model_path, opts, self.port)

            # Read at launch time, not import time: the GPU selection is an operator
            # preference that can change while the control plane is running. It only ever
            # applies to a NEW engine process, because CUDA reads the variable once at
            # context init.
            visible = self.gpus if self.gpus is not None else get_visible_devices()
            extra_env: dict[str, str] = {}
            if visible:
                extra_env["CUDA_VISIBLE_DEVICES"] = visible
            env = engine_environment(extra_env)

            # Checkpoint size drives the progress estimate; measure memory before the
            # engine touches it so growth is attributable to the load.
            self.model_size_bytes = _checkpoint_bytes(model_path)
            self.baseline_vram_bytes, self.baseline_ram_bytes = _memory_baseline()

            self.logs.clear()
            self.logs.append("ui", "$ " + " ".join(shlex.quote(a) for a in argv))
            self.logs.append(
                "ui",
                f"CUDA_VISIBLE_DEVICES={visible}" if visible else "CUDA_VISIBLE_DEVICES=(all GPUs)",
            )

            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0

            try:
                proc = subprocess.Popen(
                    argv,
                    cwd=str(settings.repo_root),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creationflags,
                )
            except OSError as exc:
                self.state = "error"
                self.error = f"failed to spawn engine: {exc}"
                raise LaunchError(self.error) from exc

            self._proc = proc
            self.state = "starting"
            self.error = None
            self.diagnosis = None
            self.model_id = model
            # Must match --served-model-name above, since that is the name the engine
            # answers to and therefore the one routing has to match on.
            self.served_name = opts.get("served_model_name") or _served_name_for(model_path)
            self.model_path = str(model_path)
            self.options = opts
            self.argv = argv
            self.started_at = time.time()

            self._readers = [
                threading.Thread(target=self._pump, args=(proc.stdout, "stdout"), daemon=True),
                threading.Thread(target=self._pump, args=(proc.stderr, "stderr"), daemon=True),
            ]
            for t in self._readers:
                t.start()

            return self.status()

    def _kill_tree(self, pid: int) -> None:
        """taskkill the whole tree -- the scheduler/tokenizer workers are child processes
        and would otherwise survive the parent, still holding VRAM."""
        if sys.platform != "win32":
            return
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True, timeout=30, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass

    async def stop(self, timeout_s: float = 20.0) -> dict:
        async with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                self.state = "stopped"
                self._proc = None
                self.started_at = None
                return self.status()

            self.state = "stopping"
            pid = proc.pid
            self.logs.append("ui", f"stopping engine (pid {pid})")

            # Ask nicely first: uvicorn's lifespan shutdown reaps the backend workers and
            # frees VRAM cleanly. Escalate to the tree kill if that does not land.
            try:
                proc.terminate()
            except OSError:
                pass

            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline and proc.poll() is None:
                await asyncio.sleep(0.25)

            if proc.poll() is None:
                self.logs.append("ui", "graceful stop timed out; killing process tree")
                self._kill_tree(pid)
                await asyncio.to_thread(proc.wait, 10)
            else:
                # Reap stragglers even on the clean path: a worker that missed the shutdown
                # would keep its CUDA context -- and its VRAM -- alive.
                self._kill_tree(pid)

            self._proc = None
            self.state = "stopped"
            self.error = None
            self.diagnosis = None
            self.started_at = None
            self.logs.append("ui", "engine stopped")
            return self.status()

    async def shutdown(self) -> None:
        """Called from the app lifespan so closing the UI never orphans an engine."""
        if self.is_alive():
            await self.stop()


class EngineManager:
    """Several engines at once, one per GPU.

    Port allocation spaces instances by two because each engine binds `port` and
    `port + 1`. GPU allocation hands each instance one card from the configured pool, so
    two models never contend for the same VRAM -- the failure mode there is the second
    load OOMing halfway through, minutes in.
    """

    def __init__(self) -> None:
        self._instances: dict[str, EngineSupervisor] = {}
        self._lock = asyncio.Lock()

    # -- introspection -----------------------------------------------------------
    def all(self) -> list[EngineSupervisor]:
        return list(self._instances.values())

    def get(self, instance_id: str) -> EngineSupervisor | None:
        return self._instances.get(instance_id)

    def statuses(self) -> list[dict]:
        """Engines worth showing. A never-used placeholder is not one.

        `primary()` lazily creates a default supervisor so the single-engine endpoints
        always have something to address. It has no model and no process, but it was being
        listed like a real engine -- the console showed "no model · GPU 0,1,2 · stopped"
        with an Unload button, and counted it in "0 of 1 engine running". While waiting for
        VRAM to drop after a real unload, that placeholder is the one thing that makes it
        impossible to tell whether anything is still resident.
        """
        return [
            inst.status()
            for inst in self._instances.values()
            if inst.model_id is not None or inst.is_alive() or inst.state == "error"
        ]

    def running(self) -> list[EngineSupervisor]:
        return [i for i in self._instances.values() if i.is_alive()]

    def primary(self) -> EngineSupervisor:
        """The instance the single-engine endpoints act on.

        Prefers a live one so /api/console keeps showing something useful after an
        instance is stopped; falls back to the default slot.
        """
        live = self.running()
        if live:
            return live[0]
        if self._instances:
            return next(iter(self._instances.values()))
        inst = EngineSupervisor("default", settings.engine_port, None)
        self._instances["default"] = inst
        return inst

    def for_model(self, name: str) -> EngineSupervisor | None:
        """The running instance serving `name`, matched on either the catalog id it was
        started with or the name the engine reports."""
        for inst in self.running():
            if inst.model_id == name or inst.served_name == name:
                return inst
        return None

    def loaded_models(self) -> list[dict]:
        """What is resident right now -- the menu a routing agent chooses from."""
        out: list[dict] = []
        for inst in self.running():
            out.append(
                {
                    "instance_id": inst.instance_id,
                    "model": inst.model_id,
                    "served_name": inst.served_name,
                    "state": inst.state,
                    "port": inst.port,
                    "gpus": inst.gpus,
                    "ready": inst.state == "running",
                }
            )
        return out

    # -- allocation --------------------------------------------------------------
    def _free_port(self) -> int:
        used = {i.port for i in self._instances.values() if i.is_alive()}
        port = settings.engine_port
        while port in used or _port_in_use(port) or _port_in_use(port + 1):
            port += 2
            if port > settings.engine_port + 200:
                raise LaunchError("no free engine port in range")
        return port

    def _free_gpu(self, requested: str | None) -> str | None:
        if requested:
            return requested
        pool = [x.strip() for x in (get_visible_devices() or "").split(",") if x.strip()]
        if not pool:
            return None  # no restriction configured; let the engine see everything
        taken = {i.gpus for i in self._instances.values() if i.is_alive() and i.gpus}
        for gpu_id in pool:
            if gpu_id not in taken:
                return gpu_id
        raise LaunchError(
            f"every GPU in the pool ({', '.join(pool)}) already has an engine. "
            "Stop one first, or add a GPU in Settings."
        )

    # -- lifecycle ---------------------------------------------------------------
    async def start(
        self,
        model: str,
        options: dict[str, Any] | None = None,
        gpus: str | None = None,
        instance_id: str | None = None,
    ) -> dict:
        async with self._lock:
            # Reap finished instances so their port and GPU become available again.
            for key in [k for k, v in self._instances.items() if not v.is_alive()]:
                if self._instances[key].state in {"stopped", "error"}:
                    self._instances.pop(key)

            if any(i.model_id == model for i in self.running()):
                raise LaunchError(f"{model} is already loaded")

            chosen_gpu = self._free_gpu(gpus)
            port = self._free_port()
            key = instance_id or f"eng{port}"
            inst = EngineSupervisor(key, port, chosen_gpu)
            self._instances[key] = inst

        # Outside the manager lock: start() takes the instance's own lock and can block
        # for a moment, and holding both would serialise unrelated launches.
        try:
            return await inst.start(model, options)
        except Exception:
            async with self._lock:
                self._instances.pop(key, None)
            raise

    async def stop(self, instance_id: str) -> dict:
        inst = self._instances.get(instance_id)
        if inst is None:
            raise LaunchError(f"no engine instance {instance_id!r}")
        result = await inst.stop()
        async with self._lock:
            self._instances.pop(instance_id, None)
        return result

    async def move(self, instance_id: str, gpus: str) -> dict:
        """Unload a model from its GPU and reload it on another, keeping its options.

        Doing this by hand means Stop, then Load, then remembering every option the engine
        was launched with -- and getting it wrong means a model that silently comes back
        with a different cache size. This carries `options` across.

        The two halves cannot overlap: CUDA reads its device assignment once at process
        start, so the only way to change a running engine's GPU is to stop it and launch
        again. The old engine's VRAM must actually be released before the new one starts,
        or the target may fail to allocate -- so the stop is awaited, not fired off.
        """
        inst = self._instances.get(instance_id)
        if inst is None:
            raise LaunchError(f"no engine instance {instance_id!r}")

        target = (gpus or "").strip()
        if not target:
            raise LaunchError("no target GPU given")
        if inst.gpus == target:
            raise LaunchError(f"{inst.model_id or 'this engine'} is already on GPU {target}")

        model = inst.model_id
        if not model:
            raise LaunchError("that engine has no model loaded to move")
        options = dict(inst.options)

        pool = [x.strip() for x in (get_visible_devices() or "").split(",") if x.strip()]
        if pool and target not in pool:
            raise LaunchError(
                f"GPU {target} is not in the pool ({', '.join(pool)}). Add it in Settings."
            )
        occupant = next(
            (
                i
                for i in self._instances.values()
                if i is not inst and i.is_alive() and i.gpus == target
            ),
            None,
        )
        if occupant is not None:
            raise LaunchError(
                f"GPU {target} already runs {occupant.model_id or 'an engine'}. "
                "Stop or move that one first."
            )

        await inst.stop()
        async with self._lock:
            self._instances.pop(instance_id, None)

        try:
            return await self.start(model, options, target)
        except Exception as exc:
            # The model is now loaded nowhere. Say so plainly -- the alternative is a user
            # staring at an empty Models page wondering what happened to their engine.
            raise LaunchError(
                f"unloaded {model} from GPU {inst.gpus}, but starting it on GPU {target} "
                f"failed: {exc}. Nothing is loaded now; start it again from the Models page."
            ) from exc

    async def shutdown(self) -> None:
        """Stop every engine. Called from the app lifespan so exiting the UI never leaves
        a model holding VRAM."""
        for inst in list(self._instances.values()):
            if inst.is_alive():
                await inst.stop()
        self._instances.clear()


manager = EngineManager()

# Single-engine endpoints still address one instance; this keeps them working while the
# manager handles the rest.
supervisor = manager.primary()
