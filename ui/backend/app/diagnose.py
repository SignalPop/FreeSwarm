"""Turn a dead engine's output into something a person can act on.

`engine exited with code 1` is true and useless. The engine writes the real reason to
stderr -- a Python traceback, an nvcc failure, a CUDA OOM -- and then the process is gone.
This module reads the captured lines and answers two questions:

  1. **What went wrong?**  `extract_error` finds the most informative line, preferring a
     traceback's final `SomeError: message` over surrounding noise.
  2. **What should I do about it?**  `match_known_issue` recognises the failures this port
     actually hits on Windows and returns a fix, because most of them are not guessable
     from the raw message (the NCCL one in particular reads like a network error).

Pattern order matters: the list is scanned top to bottom and the first hit wins, so more
specific patterns precede general ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A Python exception's last line: "RuntimeError: nvcc 12.3 would build ..."
_EXC_RE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt)):\s*(.+)$")

# Lines that are noise rather than the failure.
_NOISE = (
    "FutureWarning",
    "UserWarning",
    "DeprecationWarning",
    "warnings.warn",
    "Traceback (most recent call last)",
)


@dataclass
class Diagnosis:
    """What the UI shows in place of a bare exit code."""

    summary: str                 # one line, the actual error
    hint: str | None = None      # what to do about it
    doc: str | None = None       # where it is written up
    tail: list[str] | None = None  # last raw lines, for the details panel

    def as_dict(self) -> dict:
        return {
            "summary": self.summary,
            "hint": self.hint,
            "doc": self.doc,
            "tail": self.tail or [],
        }


# (regex, hint, doc-anchor). Matched against the whole captured output, case-insensitive.
_KNOWN_ISSUES: list[tuple[re.Pattern[str], str, str]] = [
    (
        # Any mention of NCCL in a Windows failure is this issue: torch says
        # "Distributed package doesn't have NCCL built in", the JIT link says "-lnccl",
        # and there is no Windows NCCL build for either to find.
        # No word boundaries: "-lnccl" (the JIT linker flag) has none before "nccl".
        re.compile(r"nccl", re.I),
        "Tensor parallelism needs NCCL, which has no Windows build. Set tensor parallel "
        "size to 1. For a model too large for one card, use MoE backend 'offload' so the "
        "experts stream from host RAM instead of being sharded across GPUs.",
        "ui/README.md - Things that will bite you",
    ),
    (
        re.compile(r"nvcc\s+\d+\.\d+\s+would build kernels linking|libcudart\.so\.\d+", re.I),
        "The CUDA toolkit's major version does not match the torch wheel's. The control "
        "plane should be picking the venv's cu13 toolchain; check Settings -> Build "
        "toolchain, and make sure no system CUDA_PATH is overriding it.",
        "ui/README.md - Things that will bite you",
    ),
    (
        re.compile(r"ninja exited with status|error C\d{4}|FAILED: \[code=", re.I),
        "A CUDA kernel failed to compile. Re-run scripts\\patch_deps_windows.py -- the "
        "MSVC build of tvm-ffi needs two patches, and the second (a FunctionInfo "
        "specialisation) is required for the MoE offload path.",
        "scripts/patch_deps_windows.py",
    ),
    (
        # libzmq reports a Windows reserved-port bind (WSAEACCES) as "Permission denied",
        # which reads like a security problem rather than a port clash.
        re.compile(r"ZMQError.*Permission denied|Permission denied.*tcp://", re.I),
        "The engine's internal ZMQ port lands inside a Windows reserved port range -- "
        "Hyper-V, Docker and WSL carve blocks out of the dynamic range (49152+), and "
        "binding one fails with 'Permission denied' rather than 'in use'. List them with: "
        "netsh int ipv4 show excludedportrange protocol=tcp. Fixed by moving the port band "
        "in python/freetoken/utils/mp.py below the dynamic range; restart the engine.",
        "python/freetoken/utils/mp.py - zmq_endpoint",
    ),
    (
        re.compile(r"only one usage of each socket address|10048|failed to bind", re.I),
        "A port the engine needs (1919, or 1920 for its rendezvous store) is already held, "
        "almost always by an orphaned worker from a previous run. Run stop-services.cmd, "
        "or taskkill /PID <pid> /T /F.",
        "ui/README.md - Orphaned workers hold ports and VRAM",
    ),
    (
        re.compile(r"out of memory|OutOfMemoryError|CUDA error: out of memory", re.I),
        "The GPU ran out of VRAM. Lower KV pages or the MoE cache size, reduce memory "
        "ratio, or switch MoE backend to 'offload' to move the experts into host RAM.",
        None,
    ),
    (
        re.compile(r"cudaHostRegister|pinned|MemoryError|page ?file|commit limit", re.I),
        "Host memory ran out while pinning experts. The offload path needs the experts to "
        "fit pinned in RAM. Use --expert-load serial, or pick a smaller checkpoint.",
        "ui/README.md - Hardware constraints",
    ),
    (
        re.compile(r"No module named|ImportError|ModuleNotFoundError", re.I),
        "A Python dependency is missing from the engine venv. Check the module name below "
        "and pip install it into .venv.",
        None,
    ),
    (
        re.compile(r"cl\.exe|Microsoft Visual|vcvars|host compiler", re.I),
        "The MSVC host compiler could not be found. The control plane resolves vcvars64.bat "
        "itself -- check Settings -> Build toolchain; set FREETOKEN_VCVARS to override.",
        None,
    ),
    (
        re.compile(r"is not a (valid )?(checkpoint|directory)|No such file or directory|"
                   r"does not exist|safetensors", re.I),
        "The checkpoint could not be read. Verify the model directory still holds its "
        "config.json and all its weight shards.",
        None,
    ),
]


# Startup banners that mention half the feature set. Matching hints against these makes
# every failure look like whichever keyword appears first.
_CONFIG_DUMP = ("ServerArgs(", "Parsed arguments", "Resolved config:", "EngineConfig(")


def _is_noise(line: str) -> bool:
    return any(n in line for n in _NOISE)


def _is_config_dump(line: str) -> bool:
    return any(c in line for c in _CONFIG_DUMP)


def _failure_region(lines: list[str]) -> list[str]:
    """The lines that actually describe the failure.

    Scoped to everything after the last traceback/ERROR marker, so a hint is matched
    against the crash rather than against the startup banner hundreds of lines earlier.
    Config dumps are dropped outright.
    """
    marker = -1
    for i, line in enumerate(lines):
        if "Traceback (most recent call last)" in line or "ERROR" in line:
            marker = i
    region = lines[marker:] if marker >= 0 else lines[-40:]
    return [ln for ln in region if not _is_config_dump(ln)]


def extract_error(lines: list[str]) -> str | None:
    """The most informative line in the captured output.

    Searched backwards: the failure that killed the process is at the end, and an engine
    that got a long way in will have many earlier lines that merely look alarming.
    """
    for line in reversed(lines):
        text = line.strip()
        if not text or _is_noise(text):
            continue
        m = _EXC_RE.match(text)
        if m:
            return f"{m.group(1)}: {m.group(2)}"[:600]

    # No exception line: fall back to the last ERROR the engine logged.
    for line in reversed(lines):
        if "ERROR" in line and not _is_noise(line):
            # Strip the engine's ANSI colour codes and log prefix.
            cleaned = re.sub(r"\x1b\[[0-9;]*m", "", line)
            cleaned = re.sub(r"^\[[^\]]*\]\s*", "", cleaned.strip())
            return cleaned[:600]
    return None


def match_known_issue(lines: list[str]) -> tuple[str, str | None] | None:
    """(hint, doc) for the first known failure mode present in the output."""
    blob = chr(10).join(_failure_region(lines))
    for pattern, hint, doc in _KNOWN_ISSUES:
        if pattern.search(blob):
            return hint, doc
    return None


def diagnose(lines: list[str], exit_code: int | None) -> Diagnosis:
    """Build the full diagnosis shown on the Console when an engine dies."""
    summary = extract_error(lines)
    if summary is None:
        summary = (
            f"engine exited with code {exit_code}"
            if exit_code is not None
            else "engine stopped unexpectedly"
        )
        if not lines:
            summary += " (no output captured -- it died before writing anything)"

    known = match_known_issue(lines)
    hint, doc = known if known else (None, None)

    # Keep the tail meaningful: drop warning spam so the panel shows the failure, not
    # thirty lines of FutureWarning.
    tail = [ln for ln in lines[-60:] if ln.strip() and not _is_noise(ln)][-25:]

    return Diagnosis(summary=summary, hint=hint, doc=doc, tail=tail)
