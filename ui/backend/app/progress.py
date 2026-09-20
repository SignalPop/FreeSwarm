"""Reconstruct what the engine is doing during a model load.

The engine's own `/health` reports a phase slug (`weights`, `expert_banks`, `warmup`) plus
byte counts, which it learns from progress messages the workers push over the ack queue.
That covers weight loading well and everything after it badly: KV allocation, CUDA-graph
capture and prefill warmup all arrive as `other` with no numbers, and on a 61 GiB offloaded
model that stretch is minutes of an apparently frozen bar.

So this module merges two sources:

  * **`/health`** -- authoritative byte progress while it is reporting any.
  * **The captured stdout** -- the engine logs a distinctive line at every phase boundary,
    and tqdm writes `NN%|...| done/total` for the bars. Parsing those recovers the phases
    the ack stream never describes.

The result is an ordered, weighted stepper: which phase, how far through it, how long it
has taken, and one line of detail. Weights are guesses at relative duration, tuned against
a measured gpt-oss-120b offload load (~12 min, almost all of it weights + expert pinning).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Ordered phases with a rough share of total load time. The share only drives the overall
# bar; each phase reports its own exact percentage when it has one.
PHASES: list[tuple[str, str, float]] = [
    ("starting",    "Starting engine process",      0.02),
    ("weights",     "Loading weight shards",        0.45),
    ("expert_banks", "Pinning experts in host RAM", 0.35),
    ("kv_cache",    "Allocating KV cache",          0.03),
    ("cuda_graphs", "Capturing CUDA graphs",        0.10),
    ("warmup",      "Prefill warmup",               0.05),
    ("ready",       "Ready",                        0.00),
]
_ORDER = {slug: i for i, (slug, _, _) in enumerate(PHASES)}
_LABEL = {slug: label for slug, label, _ in PHASES}


# --- log markers -----------------------------------------------------------------------
# Each entry: (regex, phase it proves the engine has REACHED).
_MARKERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Free memory before loading model", re.I), "weights"),
    (re.compile(r"Loading expert|expert bank|pinning expert", re.I), "expert_banks"),
    (re.compile(r"Allocating\s+[\d,]+\s+tokens for KV cache", re.I), "kv_cache"),
    (re.compile(r"Free memory after initialization", re.I), "kv_cache"),
    (re.compile(r"Start capturing CUDA graphs|Preparing for capturing CUDA graphs", re.I), "cuda_graphs"),
    (re.compile(r"Capturing graphs:", re.I), "cuda_graphs"),
    (re.compile(r"Free GPU memory after capturing CUDA graphs", re.I), "warmup"),
    (re.compile(r"Prefill warmup|warmup complete", re.I), "warmup"),
    (re.compile(r"API server is ready to serve|Scheduler is idle", re.I), "ready"),
]

# tqdm: "Capturing graphs: bs = 4 | avail_mem = 33.90 GiB:  67%|######   | 2/3 [00:34<00:17]"
_TQDM_FRACTION = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")
_TQDM_PERCENT = re.compile(r"(\d{1,3})%\|")

# "Allocating 8192 tokens for KV cache, K + V = 0.45 GiB"
_KV_LINE = re.compile(r"Allocating\s+([\d,]+)\s+tokens for KV cache[^0-9]*([\d.]+\s*\w+)?", re.I)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _clean(line: str) -> str:
    return _ANSI.sub("", line).strip()


@dataclass
class LoadStatus:
    phase: str = "starting"
    label: str = "Starting engine process"
    detail: str = ""
    phase_pct: float | None = None      # progress within the current phase, 0-100
    overall_pct: float = 0.0            # weighted across all phases
    done_bytes: int = 0
    total_bytes: int = 0
    # True when phase_pct came from the memory-growth heuristic rather than the engine.
    estimated: bool = False
    steps: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "phase": self.phase,
            "label": self.label,
            "detail": self.detail,
            "phase_pct": self.phase_pct,
            "overall_pct": round(self.overall_pct, 1),
            "done_bytes": self.done_bytes,
            "total_bytes": self.total_bytes,
            "estimated": self.estimated,
            "steps": self.steps,
        }


def _overall(phase: str, phase_pct: float | None) -> float:
    """Weighted completion across phases: everything before `phase` counts as done."""
    idx = _ORDER.get(phase, 0)
    total = sum(w for _, _, w in PHASES) or 1.0
    before = sum(w for i, (_, _, w) in enumerate(PHASES) if i < idx)
    current_weight = PHASES[idx][2]
    fraction = (phase_pct or 0.0) / 100.0
    return min(100.0, ((before + current_weight * fraction) / total) * 100.0)


def _latest_phase(lines: list[str]) -> str:
    """Furthest phase any log line proves we reached.

    Max rather than last-match: tqdm redraws and interleaved worker output mean a later
    line can belong to an earlier phase, and the bar must never travel backwards.
    """
    reached = "starting"
    for raw in lines:
        line = _clean(raw)
        if not line:
            continue
        for pattern, phase in _MARKERS:
            if pattern.search(line) and _ORDER[phase] > _ORDER[reached]:
                reached = phase
    return reached


def _tqdm_progress(lines: list[str], phase: str) -> tuple[float | None, str]:
    """(percent, detail) from the most recent tqdm bar belonging to `phase`."""
    for raw in reversed(lines[-80:]):
        line = _clean(raw)
        if "%|" not in line and "/" not in line:
            continue
        if phase == "cuda_graphs" and not (
            "Capturing graphs" in line or "capturing CUDA graphs" in line
        ):
            continue
        frac = _TQDM_FRACTION.search(line)
        if frac:
            done, total = int(frac.group(1)), int(frac.group(2))
            if total > 0:
                head = line.split(":")[0][:60]
                return (done / total) * 100.0, f"{head} - {done} of {total}"
        pct = _TQDM_PERCENT.search(line)
        if pct:
            return float(pct.group(1)), line.split(":")[0][:60]
    return None, ""


def _memory_estimate(mem: dict | None) -> tuple[float | None, str]:
    """(percent, detail) estimated from how much memory the load has consumed so far.

    The engine reports byte counts only for the phases its workers push progress for; on
    this port that frequently leaves weight loading with no number at all, which is the
    longest phase. Resident bytes against the checkpoint size is a genuine signal for it:
    fused loads grow VRAM, offloaded loads grow host RAM.

    It IS an estimate -- other processes move memory too, and a quantised checkpoint does
    not expand 1:1 -- so the caller marks it as approximate rather than presenting it as
    exact.
    """
    if not mem:
        return None, ""
    size = int(mem.get("model_size_bytes") or 0)
    if size <= 0:
        return None, ""

    grown_vram = max(0, int(mem.get("vram_bytes") or 0) - int(mem.get("baseline_vram") or 0))
    grown_ram = max(0, int(mem.get("ram_bytes") or 0) - int(mem.get("baseline_ram") or 0))
    # Whichever pool the weights are actually landing in dominates; offload grows RAM,
    # fused grows VRAM.
    grown = max(grown_vram, grown_ram)
    if grown <= 0:
        return None, ""

    pct = min(99.0, (grown / size) * 100.0)
    where = "host RAM" if grown_ram >= grown_vram else "VRAM"
    return pct, f"~{grown / 2**30:.1f} of {size / 2**30:.1f} GiB into {where}"


def build(
    log_lines: list[str],
    health: dict | None,
    elapsed_s: float,
    mem: dict | None = None,
) -> LoadStatus:
    """Merge the engine's /health with parsed log output into one progress view."""
    status = LoadStatus()

    log_phase = _latest_phase(log_lines)

    # /health's phase is authoritative while it reports bytes -- those come straight from
    # the worker ack stream. Outside that, the logs know more.
    health_phase = (health or {}).get("phase") or "other"
    progress = (health or {}).get("progress") or {}
    done = int(progress.get("done_bytes") or 0)
    total = int(progress.get("total_bytes") or 0)

    phase = log_phase
    if health_phase in _ORDER and total > 0 and _ORDER[health_phase] > _ORDER[log_phase]:
        phase = health_phase

    if (health or {}).get("status") == "ok":
        phase = "ready"

    status.phase = phase
    status.label = _LABEL.get(phase, "Loading")
    status.done_bytes, status.total_bytes = done, total

    # Percent within the phase, best source first.
    if phase == "ready":
        status.phase_pct = 100.0
    elif total > 0 and phase in {"weights", "expert_banks"}:
        status.phase_pct = min(100.0, (done / total) * 100.0)
        status.detail = f"{done / 2**30:.1f} of {total / 2**30:.1f} GiB"
    else:
        pct, detail = _tqdm_progress(log_lines, phase)
        if pct is None and phase in {"weights", "expert_banks"}:
            pct, detail = _memory_estimate(mem)
            status.estimated = pct is not None
        status.phase_pct = pct
        status.detail = detail

    # Phase-specific colour the logs can supply.
    if not status.detail:
        for raw in reversed(log_lines[-60:]):
            line = _clean(raw)
            kv = _KV_LINE.search(line)
            if kv and phase == "kv_cache":
                status.detail = f"{kv.group(1)} tokens"
                break
            if phase == "expert_banks" and "expert" in line.lower():
                status.detail = line[:90]
                break

    status.overall_pct = 100.0 if phase == "ready" else _overall(phase, status.phase_pct)

    # The stepper: every phase with its state, so the UI can show what is done, what is
    # running and what is still ahead.
    current_idx = _ORDER[phase]
    status.steps = [
        {
            "slug": slug,
            "label": label,
            "state": (
                "done" if i < current_idx
                else "active" if i == current_idx
                else "pending"
            ),
            "pct": (
                100.0 if i < current_idx
                else status.phase_pct if i == current_idx
                else None
            ),
        }
        for i, (slug, label, _) in enumerate(PHASES)
        if slug != "ready"
    ]
    return status
