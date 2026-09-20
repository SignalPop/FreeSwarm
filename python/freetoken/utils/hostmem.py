"""Host RAM accounting: how much there is, and how much FreeToken may take.

The engine holds the MoE expert banks in host RAM for the process lifetime, page-locked
(``cudaHostRegister``) so the GPU can dereference them. That memory is *not* reclaimable:
the OS cannot page it out under pressure, so a bank set that is merely "large" rather than
"too large" still starves every other process on the box. Before this module nothing in the
engine knew the size of host RAM at all -- the only probe, ``expert_banks._host_ram_fits_parallel``,
read ``/proc/meminfo`` and silently answered "plenty" on every non-Linux platform.

Three numbers, all in bytes:

* :func:`available` -- what the OS says can be handed out right now without paging.
* :func:`total` -- installed physical RAM.
* :func:`budget` -- the ceiling FreeToken holds itself to, from the environment:

  ``FREETOKEN_HOST_RAM_GB``
      Absolute cap in GiB. Wins over the fraction.
  ``FREETOKEN_HOST_RAM_FRACTION``
      Fraction of *total* RAM (default ``0.52`` -- see the constant below; this tracks the
      WDDM page-locking ceiling, not RAM pressure). Sized against total rather than
      available because the banks outlive the load: what matters is the steady state the
      machine is left in, not how much happened to be free at launch.
  ``FREETOKEN_HOST_RAM_HEADROOM_GB``
      Absolute RAM (default ``8``) to leave for the OS and everything else. The budget is
      additionally clamped to ``total - headroom``.

Every probe is best-effort: an unknown platform reports ``None`` and callers keep their
previous behaviour rather than guessing.
"""

from __future__ import annotations

import ctypes
import os
import sys

_GIB = 1 << 30

# Not "most of RAM". There is a hard driver ceiling on cudaHostRegister'd memory well below
# physical RAM: measured at 68-69 GiB on a 127.9 GiB machine (same bytes whether registered
# as 69 x 1 GiB or 17 x 4 GiB, so a byte limit, not a region-count one), failing while ~58
# GiB was still free. Not the WDDM shared-memory limit -- the cards measured were in TCC
# mode. A budget above that ceiling is not a budget: it passes loads that cannot physically
# pin. 0.52 is an empirical fit; RE-MEASURE on different hardware.
DEFAULT_FRACTION = 0.52
DEFAULT_HEADROOM_GB = 8.0


def _win_memory_status() -> tuple[int, int] | None:
    """``(total, available)`` from ``GlobalMemoryStatusEx``. ``ullAvailPhys`` is the
    Windows analog of ``MemAvailable``: free + zeroed + standby (reclaimable cache)."""

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
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def _linux_memory_status() -> tuple[int | None, int | None]:
    """``(total, available)`` from ``/proc/meminfo``; either may be ``None``."""
    total = avail = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
                if total is not None and avail is not None:
                    break
    except (OSError, ValueError, IndexError):
        pass
    return total, avail


def _psutil_memory_status() -> tuple[int, int] | None:
    """Last resort for platforms with neither probe (macOS, BSD). Optional dependency."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        vm = psutil.virtual_memory()
    except Exception:  # noqa: BLE001 -- best-effort probe
        return None
    return int(vm.total), int(vm.available)


def _status() -> tuple[int | None, int | None]:
    if sys.platform == "win32":
        win = _win_memory_status()
        if win is not None:
            return win
    elif sys.platform.startswith("linux"):
        total, avail = _linux_memory_status()
        if total is not None or avail is not None:
            return total, avail
    other = _psutil_memory_status()
    if other is not None:
        return other
    return None, None


def total() -> int | None:
    """Installed physical RAM in bytes, or ``None`` if it cannot be determined."""
    return _status()[0]


def available() -> int | None:
    """RAM in bytes the OS can hand out now without paging, or ``None`` if unknown."""
    return _status()[1]


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def budget() -> int | None:
    """Bytes of host RAM FreeToken may hold resident, or ``None`` if RAM size is unknown.

    ``FREETOKEN_HOST_RAM_GB`` sets it outright (still clamped by the headroom below, so an
    oversized absolute value cannot promise RAM the box does not have). Otherwise it is
    ``FREETOKEN_HOST_RAM_FRACTION`` (default 0.52) of total, and in either case at most
    ``total - FREETOKEN_HOST_RAM_HEADROOM_GB``.
    """
    phys = total()
    if phys is None:
        return None
    headroom_gb = _env_float("FREETOKEN_HOST_RAM_HEADROOM_GB")
    headroom = int((DEFAULT_HEADROOM_GB if headroom_gb is None else headroom_gb) * _GIB)

    absolute = _env_float("FREETOKEN_HOST_RAM_GB")
    if absolute is not None:
        cap = int(absolute * _GIB)
    else:
        fraction = _env_float("FREETOKEN_HOST_RAM_FRACTION") or DEFAULT_FRACTION
        cap = int(min(fraction, 1.0) * phys)
    return max(0, min(cap, phys - headroom))


def fmt(nbytes: int | None) -> str:
    """``94.1 GiB`` / ``unknown`` -- for log lines and error messages."""
    return "unknown" if nbytes is None else f"{nbytes / _GIB:.1f} GiB"


def describe() -> str:
    """One-line summary of the host memory situation, for startup logs."""
    return (
        f"host RAM: total={fmt(total())} available={fmt(available())} "
        f"freetoken budget={fmt(budget())}"
    )


__all__ = ["available", "budget", "describe", "fmt", "total"]
