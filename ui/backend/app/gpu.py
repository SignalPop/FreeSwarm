"""GPU + host telemetry for the console's resource gauges.

Everything comes from `nvidia-smi --query-gpu`, which works identically for WDDM and TCC
cards -- with one caveat this box actually hits: under WDDM the driver, not CUDA, owns
memory, so `memory.used` is reported per-GPU but `--query-compute-apps` returns
"Not available in WDDM driver model". The TCC cards report both. We therefore only read the
per-GPU totals, which are reliable on both.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time

_FIELDS = [
    "index",
    "name",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "compute_cap",
]

_CACHE_TTL_S = 1.0
_cache: tuple[float, list[dict]] = (0.0, [])
_lock = asyncio.Lock()


def _nvidia_smi() -> str | None:
    return shutil.which("nvidia-smi")


def _parse_number(raw: str) -> float | None:
    raw = raw.strip()
    if not raw or raw.startswith("[") or raw in {"N/A", "Not Supported"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _query_sync() -> list[dict]:
    exe = _nvidia_smi()
    if exe is None:
        return []
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={','.join(_FIELDS)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    gpus: list[dict] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(_FIELDS):
            continue
        total_mib = _parse_number(parts[2]) or 0.0
        used_mib = _parse_number(parts[3]) or 0.0
        gpus.append(
            {
                "index": int(_parse_number(parts[0]) or 0),
                "name": parts[1],
                "memory_total_bytes": int(total_mib * 1024 * 1024),
                "memory_used_bytes": int(used_mib * 1024 * 1024),
                "utilization_pct": _parse_number(parts[4]),
                "temperature_c": _parse_number(parts[5]),
                "power_draw_w": _parse_number(parts[6]),
                "power_limit_w": _parse_number(parts[7]),
                "compute_cap": parts[8],
            }
        )
    return gpus


async def query_gpus() -> list[dict]:
    """Cached, non-blocking GPU snapshot. nvidia-smi costs ~40ms; the console polls at 1 Hz
    and several widgets read it, so one subprocess per second is plenty."""
    global _cache
    async with _lock:
        stamp, cached = _cache
        now = time.monotonic()
        if now - stamp < _CACHE_TTL_S and cached:
            return cached
        gpus = await asyncio.to_thread(_query_sync)
        _cache = (now, gpus)
        return gpus


def host_memory() -> dict:
    """Total/available physical RAM via the Win32 GlobalMemoryStatusEx struct.

    Avoids a psutil dependency for the one number the sidebar needs; falls back to zeros
    off Windows or if the call fails.
    """
    import ctypes
    import sys

    if sys.platform != "win32":
        return {"total_bytes": 0, "available_bytes": 0}

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    try:
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {"total_bytes": 0, "available_bytes": 0}
    except (AttributeError, OSError):
        return {"total_bytes": 0, "available_bytes": 0}
    return {
        "total_bytes": int(status.ullTotalPhys),
        "available_bytes": int(status.ullAvailPhys),
    }
