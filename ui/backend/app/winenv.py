"""Build the environment the FreeToken engine subprocess needs on native Windows.

Two things have to be in place before the engine can serve:

1. **cl.exe on PATH.** FreeToken JIT-compiles CUDA kernels on first use through tvm-ffi,
   and nvcc shells out to the MSVC host compiler. Upstream's Windows notes say to launch
   from a `vcvars64` prompt; we do the same thing programmatically by running
   `vcvars64.bat` once and capturing the environment it produces, so the UI can be started
   from any ordinary shell.

2. **A CUDA toolkit whose major matches torch's.** `freetoken.kernel._toolchain` refuses to
   compile when `nvcc --version` disagrees with `torch.version.cuda` on the major, because
   the resulting kernel DLL would link a cudart the torch wheel does not ship.

Both results are cached for the lifetime of the process -- `vcvars64.bat` takes a second or
two and its output never changes while the machine is up.
"""

from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path

_VSWHERE = Path(
    os.getenv("ProgramFiles(x86)", r"C:\Program Files (x86)")
) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"


@functools.lru_cache(maxsize=1)
def find_vcvars64() -> Path | None:
    """Locate vcvars64.bat via vswhere, preferring an install that has the C++ toolset."""
    override = os.getenv("FREETOKEN_VCVARS")
    if override:
        p = Path(override)
        return p if p.is_file() else None
    if not _VSWHERE.is_file():
        return None
    try:
        out = subprocess.run(
            [
                str(_VSWHERE),
                "-latest",
                "-products", "*",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath",
                "-format", "value",
            ],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.stdout.splitlines():
        root = Path(line.strip())
        if not line.strip():
            continue
        candidate = root / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if candidate.is_file():
            return candidate
    return None


@functools.lru_cache(maxsize=1)
def _vcvars_environment() -> dict[str, str]:
    """Run vcvars64.bat in a throwaway cmd and capture the environment it leaves behind."""
    vcvars = find_vcvars64()
    if vcvars is None:
        return {}
    try:
        # `set` after the batch file; the `&&` keeps us from printing a stale env if
        # vcvars itself fails. Output is the local ANSI codepage, hence errors="replace".
        out = subprocess.run(
            ["cmd.exe", "/s", "/c", f'"{vcvars}" >nul && set'],
            capture_output=True, text=True, errors="replace", timeout=180, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    env: dict[str, str] = {}
    for line in out.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep and key:
            env[key] = value
    return env


def _venv_cuda_toolchains() -> list[Path]:
    """CUDA toolchains installed as pip wheels inside the engine's venv.

    `nvidia-cuda-nvcc` / `nvidia-nvvm` / `nvidia-cuda-crt` / `nvidia-cuda-runtime` unpack
    to `site-packages/nvidia/cu<major>/` in exactly the layout a Windows CUDA toolkit uses
    (`bin/nvcc.exe`, `include/`, `lib/x64/cudart.lib`), so the directory can be used as
    CUDA_PATH directly -- no system-wide toolkit install, no admin rights.

    Newest major first, so a box with both cu12 and cu13 wheels prefers cu13.
    """
    from .config import settings

    # <venv>/Scripts/python.exe -> <venv>/Lib/site-packages/nvidia
    nvidia_root = settings.venv_python.parent.parent / "Lib" / "site-packages" / "nvidia"
    if not nvidia_root.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    try:
        for child in nvidia_root.iterdir():
            if not child.name.startswith("cu") or not child.is_dir():
                continue
            if not (child / "bin" / "nvcc.exe").is_file():
                continue
            try:
                major = int(child.name[2:])
            except ValueError:
                continue
            found.append((major, child))
    except OSError:
        return []
    return [p for _, p in sorted(found, key=lambda t: -t[0])]


@functools.lru_cache(maxsize=1)
def find_cuda_home() -> Path | None:
    """The CUDA toolkit the engine should build kernels with.

    Order matters, and the venv comes before the ambient environment on purpose. This
    machine has CUDA 10.0 through 12.4 installed system-wide with `CUDA_PATH` pointing at
    12.3, while the engine's torch wheel is cu130. Trusting `CUDA_PATH` first makes the
    engine load the whole model and then die at first kernel JIT with:

        nvcc 12.3 would build kernels linking libcudart.so.12, but torch 2.11.0+cu130
        ships CUDA 13.0 (libcudart.so.13)

    So: an explicit override wins, then the venv-local wheel toolchain (which is matched to
    the venv's torch by construction), then whatever the ambient environment offers.
    """
    override = os.getenv("FREETOKEN_CUDA_HOME")
    if override and (Path(override) / "bin" / "nvcc.exe").is_file():
        return Path(override)

    for candidate in _venv_cuda_toolchains():
        return candidate

    for raw in (os.getenv("CUDA_PATH"), os.getenv("CUDA_HOME")):
        if raw and (Path(raw) / "bin" / "nvcc.exe").is_file():
            return Path(raw)
    return None


def engine_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The full env for a FreeToken engine subprocess: vcvars + CUDA + caller overrides."""
    env = dict(os.environ)
    env.update(_vcvars_environment())

    # Older control-plane builds documented HOST_PIN_* while the engine has always read
    # HOST_RAM_*. Translate the aliases so a launch accepted by the preflight receives the
    # same budget in the engine subprocess.
    if not env.get("FREETOKEN_HOST_RAM_GB") and env.get("FREETOKEN_HOST_PIN_GB"):
        env["FREETOKEN_HOST_RAM_GB"] = env["FREETOKEN_HOST_PIN_GB"]
    if (
        not env.get("FREETOKEN_HOST_RAM_FRACTION")
        and env.get("FREETOKEN_HOST_PIN_FRACTION")
    ):
        env["FREETOKEN_HOST_RAM_FRACTION"] = env["FREETOKEN_HOST_PIN_FRACTION"]

    cuda_home = find_cuda_home()
    if cuda_home is not None:
        env["CUDA_PATH"] = str(cuda_home)
        env["CUDA_HOME"] = str(cuda_home)
        env["PATH"] = os.pathsep.join(
            [str(cuda_home / "bin"), env.get("PATH", "")]
        )

    # Unbuffered so the log ring sees engine output as it happens rather than in 8 KiB
    # bursts, and UTF-8 so the engine's box-drawing status output does not raise on the
    # legacy console codepage.
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    # CUDA's default enumeration is FASTEST_FIRST, which does NOT match nvidia-smi: on a box
    # with a small display GPU and large compute cards, CUDA index 0 is a compute card while
    # nvidia-smi index 0 is the display one. CUDA_VISIBLE_DEVICES is interpreted in the
    # *CUDA* order, so without this pin a device list copied from nvidia-smi selects the
    # wrong GPUs. PCI_BUS_ID makes the two orders agree.
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    if extra:
        env.update({k: v for k, v in extra.items() if v is not None})
    return env


def toolchain_report() -> dict:
    """Diagnostics for the Settings view: is this box able to build kernels at all?"""
    vcvars = find_vcvars64()
    cuda_home = find_cuda_home()
    nvcc_version = None
    if cuda_home is not None:
        try:
            out = subprocess.run(
                [str(cuda_home / "bin" / "nvcc.exe"), "--version"],
                capture_output=True, text=True, timeout=30, check=False,
            )
            for line in out.stdout.splitlines():
                if "release" in line:
                    nvcc_version = line.strip()
                    break
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "vcvars": str(vcvars) if vcvars else None,
        "cuda_home": str(cuda_home) if cuda_home else None,
        "nvcc": nvcc_version,
    }
