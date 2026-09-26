"""Runs the Python that chat writes, in a throwaway container, and collects what it made.

**One container per run, `--network none`.** Nobody reviews this code before it executes --
a model writes it and a button runs it -- so the container is the trust boundary, and the
isolation has to be real rather than nominal.

Two arrangements were measured on this stack (Docker Desktop, WSL2 backend) before settling
on this one:

* a long-lived service on an ``--internal`` network -- published ports stop working, so the
  control plane cannot reach it at all;
* a long-lived service on a bridge with ``enable_ip_masquerade=false`` -- the port works,
  but the VM's own NAT still carries traffic out: ``1.1.1.1:53`` and ``8.8.8.8:443`` were
  both reachable from inside.

``--network none`` blocks egress outright (verified: ``OSError`` on connect), and having no
listening socket at all removes the question. The cost is a container start per run, which
measures around a second against a 60-second default budget.

Each run gets a host directory bind-mounted at ``/work`` and executes with that as its
working directory, so the forms a model actually writes -- ``plt.savefig("chart.png")``,
``df.to_csv("data.csv")``, ``wb.save("book.xlsx")`` -- land as artifacts with no output-path
convention to get wrong. Artifacts stay on the host, so serving them back is a file read.
"""

from __future__ import annotations

import asyncio
import mimetypes
import shutil
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import settings

IMAGE = "freeswarm-sandbox:latest"
RUNS_ROOT = Path(settings.repo_root) / "ui" / "sandbox" / ".runs"

DEFAULT_TIMEOUT_S = 60
MAX_TIMEOUT_S = 600
# Enough to read a traceback or a table dump; past that the signal is at the two ends.
MAX_STREAM_CHARS = 200_000
MAX_ARTIFACT_BYTES = 64 << 20
MAX_ARTIFACTS = 64
# Runs are kept so their artifacts stay downloadable from the chat transcript, but the
# directory cannot grow without bound.
MAX_RETAINED_RUNS = 200

# Resource ceilings. A runaway script must not be able to take the machine down -- the
# whole reason this work started.
MEMORY = "4g"
# Files a run may be handed (script helpers, library modules, config): enough for a large code
# library; past it the run is refused rather than silently missing files.
MAX_FILES = 1000
MAX_FILES_BYTES = 64 << 20
CPUS = "2"
PIDS = "256"

router = APIRouter(prefix="/sandbox", tags=["sandbox"])


class RunRequest(BaseModel):
    code: str = Field(..., max_length=1_000_000)
    timeout_s: int = Field(DEFAULT_TIMEOUT_S, ge=1, le=MAX_TIMEOUT_S)
    # Extra files placed beside the script, so a follow-up run can build on a previous
    # one's CSV without the model having to re-emit it.
    files: dict[str, str] = Field(default_factory=dict)


# `mimetypes` consults the platform registry, so the same .csv is text/csv on Linux and
# application/vnd.ms-excel on Windows. The console decides "render inline vs offer as a
# download" from this string, so it is pinned for the formats the sandbox exists to produce.
_MIME_OVERRIDES = {
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".html": "text/html",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".ods": "application/vnd.oasis.opendocument.spreadsheet",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".parquet": "application/vnd.apache.parquet",
    ".zip": "application/zip",
}


def guess_mime(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix in _MIME_OVERRIDES:
        return _MIME_OVERRIDES[suffix]
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


def _docker() -> str | None:
    return shutil.which("docker")


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_STREAM_CHARS:
        return text, False
    half = MAX_STREAM_CHARS // 2
    omitted = len(text) - MAX_STREAM_CHARS
    return f"{text[:half]}\n\n... [{omitted} chars omitted] ...\n\n{text[-half:]}", True


def _sweep_old_runs() -> None:
    try:
        dirs = sorted(
            (d for d in RUNS_ROOT.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime
        )
    except OSError:
        return
    if len(dirs) <= MAX_RETAINED_RUNS:
        return
    for stale in dirs[:-MAX_RETAINED_RUNS]:
        shutil.rmtree(stale, ignore_errors=True)


def _collect_artifacts(run_dir: Path, script: Path) -> list[dict]:
    """Every file the script left behind, as {name, size, mime}."""
    found: list[dict] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path == script:
            continue
        # Dot-files and anything under a dot-directory are tool residue (caches, configs),
        # not output the user asked for -- listing them as artifacts is just noise.
        rel_parts = path.relative_to(run_dir).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > MAX_ARTIFACT_BYTES:
            continue
        rel = path.relative_to(run_dir).as_posix()
        found.append({"name": rel, "size": size, "mime": guess_mime(rel)})
        if len(found) >= MAX_ARTIFACTS:
            break
    return found


async def _image_present(docker: str) -> bool:
    proc = await asyncio.create_subprocess_exec(
        docker, "image", "inspect", IMAGE,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    return proc.returncode == 0


@router.get("/status")
async def status() -> dict:
    """Whether a run could succeed right now -- drives the Run button's enabled state."""
    docker = _docker()
    if docker is None:
        return {"available": False, "reason": "Docker is not installed or not on PATH."}
    try:
        proc = await asyncio.create_subprocess_exec(
            docker, "info", "--format", "{{.ServerVersion}}",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
    except (OSError, asyncio.TimeoutError):
        return {"available": False, "reason": "Could not talk to the Docker daemon."}
    if proc.returncode != 0:
        return {"available": False, "reason": "Docker is installed but not running."}
    if not await _image_present(docker):
        return {
            "available": False,
            "reason": f"The {IMAGE} image is not built yet. Run ui\\run-sandbox.bat.",
        }
    return {"available": True, "docker": (out or b"").decode().strip(), "image": IMAGE}


async def execute(
    code: str,
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    files: dict[str, str] | None = None,
    mounts: list[tuple[str, str]] | None = None,
) -> dict:
    """Run `code` in a fresh container and return the run report.

    `mounts` are extra READ-ONLY bind mounts as (host path, container path) -- how a swarm
    objective hands a candidate the project's data. Later mounts may sit inside earlier ones
    (a truncated copy of one dataset laid over the full data folder); Docker applies them in
    order. Nothing mounted this way is writable: output goes to /work only.
    """
    docker = _docker()
    if docker is None:
        raise HTTPException(
            status_code=503,
            detail="Docker is not installed or not on PATH, so the sandbox cannot run.",
        )

    run_id = uuid.uuid4().hex
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    script = run_dir / "script.py"
    script.write_text(code, encoding="utf-8")

    # This used to keep the first 32 files and silently drop the rest. Once the project's code
    # library passed ~29 modules, the config files callers add LAST (field_scan's and
    # regime_map's .ft/*_cfg.json) were dropped and those tools crashed with no clue why. A
    # generous limit that fails loudly instead.
    files = files or {}
    size = sum(len(t) for t in files.values())
    if len(files) > MAX_FILES or size > MAX_FILES_BYTES:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise HTTPException(status_code=413, detail=f"sandbox run given {len(files)} files ({size:,} chars); the limit "
                                                    f"is {MAX_FILES} files / {MAX_FILES_BYTES:,} chars")
    for name, text in files.items():
        # A supplied name like ../../etc/x must not escape the run directory.
        candidate = (run_dir / name).resolve()
        if not str(candidate).startswith(str(run_dir.resolve())):
            continue
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(text, encoding="utf-8")

    extra: list[str] = []
    for host, target in mounts or []:
        extra += ["-v", f"{Path(host).resolve()}:{target}:ro"]

    container = f"ft-sandbox-{run_id[:12]}"
    argv = [
        docker, "run", "--rm", "--name", container,
        # The isolation that matters. See the module docstring for what was measured.
        "--network", "none",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--memory", MEMORY, "--memory-swap", MEMORY,
        "--cpus", CPUS, "--pids-limit", PIDS,
        "-v", f"{run_dir.resolve()}:/work",
        *extra,
        "-w", "/work",
        IMAGE,
        # Through the bootstrap, not directly: it turns plt.show() into a saved PNG.
        # Without it the commonest form of generated plotting code produces nothing at all,
        # silently -- no window, no file, no message.
        "python", "-I", "/opt/ft/bootstrap.py", "script.py",
    ]

    started = time.time()
    timed_out = False
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"could not start docker: {exc}") from None

    try:
        raw_out, raw_err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s + 20)
    except asyncio.CancelledError:
        # The caller gave up (e.g. a cancelled re-test): stop the container too, or it keeps
        # running with the run directory mounted.
        killer = await asyncio.create_subprocess_exec(
            docker, "kill", container,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await killer.wait()
        raise
    except asyncio.TimeoutError:
        timed_out = True
        # Kill the CONTAINER, not just the docker client: killing the client would leave the
        # container running with the run directory mounted.
        killer = await asyncio.create_subprocess_exec(
            docker, "kill", container,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await killer.wait()
        try:
            raw_out, raw_err = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            raw_out, raw_err = b"", b""

    duration = time.time() - started
    out, out_cut = _truncate((raw_out or b"").decode("utf-8", "replace"))
    err, err_cut = _truncate((raw_err or b"").decode("utf-8", "replace"))
    exit_code = proc.returncode

    if timed_out:
        err = (err + f"\n\n[killed: exceeded the {timeout_s}s limit]").lstrip()
    elif exit_code == 137:
        # 128+9: the kernel OOM-killed it, which otherwise surfaces as a bare exit code.
        err = (err + f"\n\n[killed: exceeded the {MEMORY} memory limit]").lstrip()

    artifacts = _collect_artifacts(run_dir, script)
    _sweep_old_runs()

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "ok": (not timed_out) and exit_code == 0,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_s": round(duration, 2),
        "stdout": out,
        "stderr": err,
        "truncated": out_cut or err_cut,
        "artifacts": artifacts,
    }


@router.post("/run")
async def run(req: RunRequest) -> dict:
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="no code supplied")
    report = await execute(req.code, timeout_s=req.timeout_s, files=req.files)
    report.pop("run_dir", None)  # a host path; the browser has no use for it
    return report


@router.get("/artifacts/{run_id}/{name:path}")
def artifact(run_id: str, name: str) -> FileResponse:
    run_dir = (RUNS_ROOT / run_id).resolve()
    if not str(run_dir).startswith(str(RUNS_ROOT.resolve())) or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="unknown run")
    path = (run_dir / name).resolve()
    # Traversal guard: the resolved file must still sit inside its own run directory.
    if not str(path).startswith(str(run_dir)) or not path.is_file():
        raise HTTPException(status_code=404, detail="unknown artifact")
    return FileResponse(path, media_type=guess_mime(path.name), filename=path.name)


@router.delete("/runs/{run_id}")
def delete_run(run_id: str) -> dict:
    run_dir = (RUNS_ROOT / run_id).resolve()
    if not str(run_dir).startswith(str(RUNS_ROOT.resolve())):
        raise HTTPException(status_code=404, detail="unknown run")
    shutil.rmtree(run_dir, ignore_errors=True)
    return {"ok": True}
