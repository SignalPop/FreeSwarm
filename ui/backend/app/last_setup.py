"""The last working model setup, remembered so one click brings it back.

After a crash, a reboot or a control-plane restart every engine and forecaster is gone, and
bringing the swarm back meant re-picking each checkpoint on the Models page with its GPU and
options, one by one. This keeps the set that was running in ``ui/backend/last_setup.json``
and restores it as a background job.

What updates the remembered set, and what deliberately does not:

* A user-initiated start that the manager accepts records the model with everything the
  launch carried (options, the GPU asked for and the GPU it got, the port it got).
* A user's Unload of a live engine forgets it. So does unloading an engine that died before
  it ever became ready -- that was a configuration that does not load, not a crash.
* An engine exiting on its own (crash, OOM) changes nothing, and neither does dismissing the
  card of an engine that crashed after it had been serving: restoring after a crash is the
  point. The control plane's own shutdown and "Unload all" change nothing either -- the
  former is a restart, the latter is how memory is cleared before bringing things back.

Restoring starts LLM engines ONE AT A TIME, waiting for each to be ready or failed before the
next -- concurrent loads contend for host RAM, pinned memory and disk -- largest checkpoint
first (it needs the most free VRAM / pinned RAM), then time-series forecasters last (their
placement refuses a card whose LLM is still loading). One failure never stops the rest.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("freetoken.last_setup")

PATH = Path(__file__).resolve().parent.parent / "last_setup.json"

# Big offloaded checkpoints spend most of a load pinning experts: ~12 min for 61 GiB, so a
# 146 GiB one can pass half an hour. Past this the job stops waiting and moves on.
LLM_READY_TIMEOUT_S = 60 * 60
POLL_S = 1.0

_lock = threading.Lock()
_job: dict | None = None
_task: asyncio.Task | None = None


# ---------------------------------------------------------------------------------------
# The remembered set
# ---------------------------------------------------------------------------------------
def _read() -> dict:
    try:
        data = json.loads(PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"entries": []}
    except (OSError, ValueError) as exc:
        logger.warning("last_setup.json unreadable (%s); treating as empty", exc)
        return {"entries": []}
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        return {"entries": []}
    data["entries"] = [
        e
        for e in data["entries"]
        if isinstance(e, dict) and e.get("kind") in ("llm", "ts") and isinstance(e.get("model"), str) and e["model"]
    ]
    return data


def _write(data: dict) -> None:
    data["version"] = 1
    data["updated_at"] = time.time()
    tmp = PATH.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(PATH)
    except OSError as exc:  # non-fatal: the launch itself already happened
        logger.error("could not persist last_setup.json: %s", exc)


def entries() -> list[dict]:
    with _lock:
        return [dict(e) for e in _read()["entries"]]


def updated_at() -> float | None:
    with _lock:
        return _read().get("updated_at")


def _upsert(entry: dict) -> None:
    with _lock:
        data = _read()
        data["entries"] = [
            e for e in data["entries"] if not (e["kind"] == entry["kind"] and e["model"] == entry["model"])
        ] + [entry]
        _write(data)


def _gpu_str(value: Any) -> str | None:
    s = str(value).strip() if value is not None else ""
    return s or None


def record_llm(model: str, options: dict | None, requested_gpus: str | None, status: dict | None) -> None:
    """A user-initiated LLM start the manager accepted. `status` is what it returned."""
    status = status or {}
    requested = _gpu_str(requested_gpus)
    _upsert(
        {
            "kind": "llm",
            "model": model,
            "options": dict(options or {}),
            # Asked for (None = auto) vs got: restore puts it back on the card it had, and
            # falls back to auto only when auto is what the user asked for.
            "requested_gpus": requested,
            "gpus": _gpu_str(status.get("gpus")) or requested,
            "port": status.get("port"),
            "served_name": status.get("served_name"),
            "model_path": status.get("model_path"),
            "saved_at": time.time(),
        }
    )


def record_ts(model: str, requested_gpu: str | None, status: dict | None, model_path: str | None = None) -> None:
    """A time-series forecaster that started and answered its health check."""
    status = status or {}
    requested = _gpu_str(requested_gpu)
    _upsert(
        {
            "kind": "ts",
            "model": model,
            "requested_gpus": requested,
            "gpus": _gpu_str(status.get("gpu")) or requested,
            "port": status.get("port"),
            "model_path": model_path,
            "saved_at": time.time(),
        }
    )


def forget(kind: str, model: str | None) -> bool:
    if not model:
        return False
    with _lock:
        data = _read()
        kept = [e for e in data["entries"] if not (e["kind"] == kind and e["model"] == model)]
        if len(kept) == len(data["entries"]):
            return False
        data["entries"] = kept
        _write(data)
        return True


def llm_stop_forgets(inst: Any) -> str | None:
    """The model a user's Unload of `inst` should forget, or None to keep it.

    Called BEFORE the stop, while the instance still says whether it is alive. A live engine:
    forget. A dead one that had been serving (it crashed) -- keep, dismissing its card must not
    drop it from what the button restores. A dead one that never got ready -- forget, that is a
    configuration that does not load.
    """
    model = getattr(inst, "model_id", None)
    if inst is None or not model:
        return None
    if inst.is_alive():
        return model
    return None if getattr(inst, "ever_ready", False) else model


def ts_stop_forgets(ts_manager: Any, inst_id: str) -> str | None:
    """Same for a forecaster: forget a live one, keep one that died on its own."""
    for s in ts_manager.statuses():
        if s.get("id") == inst_id:
            return s.get("model_id") if s.get("state") in ("starting", "running") else None
    return None


# ---------------------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------------------
def _gpu_indices() -> set[int]:
    from . import gpu

    return {int(d["index"]) for d in gpu._query_sync()}  # noqa: SLF001


def _running_llms(llm: Any) -> set[str]:
    return {i.model_id for i in llm.all() if i.model_id and i.is_alive()}


def _running_ts(ts: Any) -> set[str]:
    return {s.get("model_id") for s in ts.statuses() if s.get("state") in ("starting", "running")}


def _split(gpus: str | None) -> list[str]:
    return [x.strip() for x in (gpus or "").split(",") if x.strip()]


def plan(
    llm: Any,
    ts: Any,
    *,
    catalog: Callable[[], list[dict]] | None = None,
    resolve: Callable[[str], Path] | None = None,
    gpu_indices: Callable[[], set[int]] | None = None,
) -> list[dict]:
    """The remembered set in start order, each judged: start / running / invalid."""
    if catalog is None:
        from .catalog import list_models as catalog
    if resolve is None:
        from .catalog import resolve_model_path as resolve
    gpu_indices = gpu_indices or _gpu_indices

    try:
        cat = catalog()
    except Exception:  # noqa: BLE001 -- sizes only order the list
        cat = []
    sizes: dict[str, int] = {}
    for m in cat:
        size = int(m.get("size_bytes") or 0) or int(m.get("expert_bytes") or 0)
        sizes[str(m.get("id"))] = size
        sizes[str(m.get("path"))] = size
    try:
        present: set[int] | None = set(gpu_indices()) or None
    except Exception:  # noqa: BLE001 -- no nvidia-smi: do not refuse on a number we do not have
        present = None
    running = {"llm": _running_llms(llm), "ts": _running_ts(ts)}

    items: list[dict] = []
    for e in entries():
        item = {
            **e,
            "size_bytes": sizes.get(e["model"]) or sizes.get(str(e.get("model_path"))) or 0,
            "action": "start",
            "reason": None,
            "launch_gpus": e.get("gpus"),
        }
        if e["model"] in running[e["kind"]]:
            item["action"], item["reason"] = "running", "already running"
            items.append(item)
            continue
        try:
            path = resolve(e["model"])
        except (ValueError, OSError) as exc:
            item["action"], item["reason"] = "invalid", f"checkpoint no longer found: {exc}"
            items.append(item)
            continue
        item["resolved_path"] = str(path)
        item["size_bytes"] = item["size_bytes"] or sizes.get(str(path)) or 0
        if present is not None:
            gone = [g for g in _split(e.get("gpus")) if not g.isdigit() or int(g) not in present]
            if gone:
                if e.get("requested_gpus") is None:
                    # It was on auto: let the manager pick again.
                    item["launch_gpus"] = None
                    item["reason"] = f"GPU {','.join(gone)} is gone; will take a free card"
                else:
                    item["action"] = "invalid"
                    item["reason"] = f"GPU {','.join(gone)} no longer exists on this machine"
        items.append(item)

    # LLMs before forecasters; within each, biggest first (it needs the most free VRAM and
    # pinned RAM, which only gets scarcer as others load).
    items.sort(key=lambda i: (i["kind"] != "llm", -int(i["size_bytes"] or 0), i["model"].lower()))
    return items


# ---------------------------------------------------------------------------------------
# The restore job
# ---------------------------------------------------------------------------------------
def current_job() -> dict | None:
    return _job


def job_running() -> bool:
    return _job is not None and _job.get("state") == "running"


def _finish(item: dict, status: str, error: str | None = None) -> None:
    item["status"] = status
    item["error"] = error
    item["finished_at"] = time.time()


async def _start_llm(item: dict, llm: Any, ready_timeout_s: float, poll_s: float) -> None:
    model = item["model"]
    gpus = item.get("launch_gpus")
    if gpus:
        occupant = next((i for i in llm.all() if i.is_alive() and str(i.gpus) == str(gpus)), None)
        if occupant is not None:
            if item.get("requested_gpus") is None:
                item["note"] = f"GPU {gpus} is held by {occupant.model_id}; taking a free card"
                gpus = None
            else:
                # Pinned to a card that already runs an engine: share it. The plan starts
                # the largest model first, and the manager refuses if there is no room.
                item["note"] = f"sharing GPU {gpus} with {occupant.model_id or 'another engine'}"
    options = dict(item.get("options") or {})
    status = await llm.start(model, options, gpus)
    record_llm(model, options, item.get("requested_gpus"), status)
    inst_id = status.get("instance_id")
    item["instance_id"] = inst_id
    item["gpus"] = status.get("gpus")
    deadline = time.monotonic() + ready_timeout_s
    while True:
        inst = llm.get(inst_id)
        if inst is None:
            _finish(item, "failed", "the engine was unloaded while it was loading")
            return
        st = inst.status()  # also reaps a process that died on its own
        if st.get("state") == "running":
            _finish(item, "ready")
            return
        if st.get("state") in ("error", "stopped"):
            _finish(item, "failed", st.get("error") or "the engine exited while loading")
            return
        if time.monotonic() > deadline:
            _finish(
                item, "failed",
                f"still loading after {ready_timeout_s / 60:.0f} min; left it loading and moved on",
            )
            return
        await asyncio.sleep(poll_s)


async def _start_ts(item: dict, ts: Any, resolve: Callable[[str], Path]) -> None:
    model = item["model"]
    path = resolve(model)
    gpu = item.get("launch_gpus")
    try:
        status = await ts.start(model, path, gpu)
    except Exception as exc:  # noqa: BLE001
        if not gpu or item.get("requested_gpus") is not None:
            raise
        # It was placed automatically last time; the old card may simply be fuller now.
        item["note"] = f"GPU {gpu} refused ({exc}); placed automatically"
        status = await ts.start(model, path, None)
    record_ts(model, item.get("requested_gpus"), status, str(path))
    item["gpus"] = status.get("gpu")
    _finish(item, "ready")


async def _execute(
    job: dict,
    llm: Any,
    ts: Any,
    resolve: Callable[[str], Path],
    ready_timeout_s: float,
    poll_s: float,
) -> None:
    try:
        for item in job["items"]:
            if item["status"] != "queued":
                continue
            item["status"] = "loading"
            item["started_at"] = time.time()
            try:
                if item["kind"] == "llm":
                    await _start_llm(item, llm, ready_timeout_s, poll_s)
                else:
                    await _start_ts(item, ts, resolve)
            except asyncio.CancelledError:
                _finish(item, "failed", "cancelled")
                raise
            except Exception as exc:  # noqa: BLE001 -- one failure must not stop the rest
                _finish(item, "failed", str(exc) or type(exc).__name__)
    finally:
        for item in job["items"]:
            if item["status"] in ("queued", "loading"):
                _finish(item, "failed", "the restore job ended before this one started")
        job["state"] = "done"
        job["finished_at"] = time.time()


async def start_restore(
    llm: Any = None,
    ts: Any = None,
    *,
    catalog: Callable[[], list[dict]] | None = None,
    resolve: Callable[[str], Path] | None = None,
    gpu_indices: Callable[[], set[int]] | None = None,
    ready_timeout_s: float = LLM_READY_TIMEOUT_S,
    poll_s: float = POLL_S,
) -> dict:
    """Plan and launch the restore in the background; returns the job to poll."""
    global _job, _task
    if llm is None:
        from .engine import manager as llm
    if ts is None:
        from .tsfm import ts_manager as ts
    if resolve is None:
        from .catalog import resolve_model_path as resolve
    if job_running():
        raise RuntimeError("a restore is already running")

    items = await asyncio.to_thread(plan, llm, ts, catalog=catalog, resolve=resolve, gpu_indices=gpu_indices)
    job_items = []
    for it in items:
        status = {"start": "queued", "running": "skipped", "invalid": "failed"}[it["action"]]
        job_items.append(
            {
                "kind": it["kind"],
                "model": it["model"],
                "size_bytes": it["size_bytes"],
                "options": it.get("options") or {},
                "requested_gpus": it.get("requested_gpus"),
                "launch_gpus": it.get("launch_gpus"),
                "gpus": it.get("launch_gpus"),
                "status": status,
                "error": it["reason"] if status == "failed" else None,
                "note": it["reason"] if status != "failed" else None,
                "instance_id": None,
                "started_at": None,
                "finished_at": None if status == "queued" else time.time(),
            }
        )
    _job = {
        "id": uuid.uuid4().hex[:12],
        "state": "running",
        "started_at": time.time(),
        "finished_at": None,
        "items": job_items,
    }
    _task = asyncio.create_task(_execute(_job, llm, ts, resolve, ready_timeout_s, poll_s))
    return _job


async def wait() -> None:
    """Await the running job (tests)."""
    if _task is not None:
        await _task


# ---------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------
router = APIRouter()


def _public(item: dict) -> dict:
    return {k: v for k, v in item.items() if k != "resolved_path"}


@router.get("/engines/restore-last")
async def restore_preview() -> dict:
    """What "Start last setup" would do right now, in the order it would do it."""
    from .engine import manager
    from .tsfm import ts_manager

    items = await asyncio.to_thread(plan, manager, ts_manager)
    return {
        "entries": [_public(i) for i in items],
        "to_start": sum(1 for i in items if i["action"] == "start"),
        "updated_at": updated_at(),
        "job": current_job(),
    }


@router.get("/engines/restore-last/job")
async def restore_job() -> dict:
    """The current or last restore job, for polling progress cheaply."""
    return {"job": current_job()}


@router.post("/engines/restore-last")
async def restore_start() -> dict:
    """Start the remembered set: LLMs one at a time, largest first, then forecasters."""
    try:
        job = await start_restore()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    return {"job": job}


class ForgetRequest(BaseModel):
    kind: str = Field(..., pattern="^(llm|ts)$")
    model: str


@router.post("/engines/restore-last/forget")
async def restore_forget(req: ForgetRequest) -> dict:
    """Drop one model from the remembered set without touching anything running."""
    return {"removed": forget(req.kind, req.model)}
