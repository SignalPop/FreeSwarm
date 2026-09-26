"""FreeSwarm control-plane API.

The FreeToken engine already exposes a FastAPI surface of its own (OpenAI / Anthropic
chat, plus `/health`, `/v1/stats`, `/v1/requests`, `/v1/cache/*`). What it does not have is
anything above a single running model: no model discovery, no start/stop, no GPU telemetry,
no log capture, and no authentication. That is what this server adds, and it proxies the
engine so the browser and any agent clients talk to exactly one origin.

Route groups:

* `/api/auth/*`   -- OAuth2 password grant, unauthenticated by necessity.
* `/api/*`        -- console control plane, requires a bearer token once any account exists.
* `/v1/*`         -- OpenAI-compatible passthrough for agent frameworks. Same bearer token,
                     supplied as the client's `api_key`, so an agent swarm needs nothing but
                     `OPENAI_BASE_URL` and `OPENAI_API_KEY`.

Security posture -- deliberate and worth stating plainly:

* The engine itself always binds **127.0.0.1**. It has no auth, so it must never be the
  thing listening on a network interface; this server is the only front door.
* This server binds loopback by default. `Settings.validate()` refuses a non-loopback bind
  unless accounts exist AND TLS is configured (or the insecure override is explicit).
* Model launches are constrained two ways: the path must resolve inside a configured model
  root (catalog.resolve_model_path) and the flags must be in engine._FLAG_SPEC.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

import httpx
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from . import auth, gpu, mcp_oauth, mcp_registry, prefs, progress, projects, remotes, tokens
from .catalog import list_models
from .config import settings
from .engine import LaunchError, host_pin_budget_bytes, manager
from .winenv import toolchain_report

logger = logging.getLogger("freetoken.ui")

_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    if _client is None:  # pragma: no cover - lifespan guarantees this
        raise RuntimeError("HTTP client not initialised")
    return _client


async def _health_poller() -> None:
    """Flip the supervisor from 'starting' to 'running' once the engine answers /health.

    The engine spends minutes in model load; its uvicorn frontend answers `/health` with
    `status: loading` the whole time, then `ok`. Polling here means the UI's own status
    endpoint is instant instead of every client racing the engine.
    """
    while True:
        try:
            for inst in manager.all():
                if inst.state == "starting":
                    with contextlib.suppress(httpx.HTTPError):
                        r = await client().get(
                            f"http://{settings.engine_host}:{inst.port}/health", timeout=5.0
                        )
                        if r.status_code == 200 and r.json().get("status") == "ok":
                            inst.mark_running()
                inst.status()  # reaps a process that died on its own
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a poller must never die
            logger.debug("health poll failed", exc_info=True)
        await asyncio.sleep(1.0)


def _federation_models() -> list[dict]:
    """Models this computer could share: the live ones, plus any that died on their own.

    A shared model that crashes (out of memory, say) is reported with its diagnosis instead
    of vanishing, so a computer using it can say what happened. It stays reported until it is
    loaded again or another load reaps it; one that was unloaded on purpose just goes.
    """
    out: dict[str, dict] = {}
    for inst in manager.all():
        if not inst.model_id:
            continue
        if inst.is_alive():
            out[inst.model_id] = {"name": inst.model_id, "served_name": inst.served_name, "port": inst.port,
                                  "ready": inst.state == "running"}
        elif inst.state == "error" and inst.model_id not in out:
            hint = (inst.diagnosis or {}).get("hint") or ""
            out[inst.model_id] = {"name": inst.model_id, "served_name": inst.served_name, "port": None,
                                  "ready": False, "error": (inst.error or "the engine exited")[:300],
                                  "hint": hint.split(". ")[0].rstrip(".")[:200] or None}
    return list(out.values())


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    global _client
    _client = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=3.0))
    poller = asyncio.create_task(_health_poller())
    # LAN federation: discovery always listens; the gateway starts only if sharing is on.
    from . import federation

    federation.configure(_federation_models)
    try:
        federation.start()
    except Exception:  # noqa: BLE001 -- federation must never keep the console from starting
        logger.exception("federation failed to start")
    # External providers: read the catalogs (prices, windows) in the background, and start the
    # loop that asks stronger models for ideas when a search is stuck.
    from . import escalation, external, objectives

    try:
        await asyncio.to_thread(objectives.migrate_ranking)
    except Exception:  # noqa: BLE001 -- a failed migration must never keep the console from starting
        logger.exception("ranking migration failed")
    warm = asyncio.create_task(external.warm(_client))
    escalator = asyncio.create_task(escalation.run(complete_text, all_loaded))
    # Tokens processed per model (app/tokens.py): samples the engines' lifetime counters.
    token_sampler = asyncio.create_task(tokens.run(_sample_engines))
    try:
        yield
    finally:
        for t in (warm, escalator):
            t.cancel()
        with contextlib.suppress(Exception):
            await external.shutdown()
        with contextlib.suppress(Exception):
            federation.stop()
        poller.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poller
        # The engines' counters die with them: take a last reading before they go.
        with contextlib.suppress(Exception):
            await _sample_engines()
        token_sampler.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await token_sampler
        with contextlib.suppress(Exception):
            await asyncio.to_thread(tokens.close)
        # Never leave an engine -- and 96 GB of VRAM -- behind when the UI exits.
        await manager.shutdown()
        # Time-series servers are separate processes too; reap them the same way.
        from .tsfm import ts_manager

        await ts_manager.shutdown()
        await _client.aclose()
        _client = None


from .version import APP_VERSION, info as version_info  # noqa: E402

app = FastAPI(title="FreeSwarm Control Plane", version=APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.exception_handler(LaunchError)
async def _launch_error_handler(_: Request, exc: LaunchError) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.exception_handler(Exception)
async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Any crash inside a route: say WHAT failed and WHERE, not "500 Internal Server Error".
    The console shows `detail`; the traceback's last frames say which line to look at. This is
    a local operator's console, so the trace is shown rather than hidden."""
    import traceback

    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    frames = traceback.extract_tb(exc.__traceback__)[-4:]
    where = [f"{Path(f.filename).name}:{f.lineno} in {f.name}" for f in frames]
    return JSONResponse({"detail": f"{type(exc).__name__}: {exc}" + (f" (at {where[-1]})" if where else ""),
                         "error_type": type(exc).__name__, "path": request.url.path,
                         "traceback": "".join(traceback.format_exception(exc))[-4000:], "where": where},
                        status_code=500)


# =======================================================================================
# Auth  (unauthenticated by necessity)
# =======================================================================================
auth_router = APIRouter(prefix="/api/auth", tags=["auth"])


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


@auth_router.post("/token", response_model=TokenResponse)
async def login(form: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    """OAuth2 password grant. Returns a short-lived access token + a refresh token."""
    if not auth.auth_enabled():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="authentication is not configured; no user accounts exist",
        )
    # scrypt verification is intentionally slow, so keep it off the event loop.
    username = await asyncio.to_thread(auth.authenticate, form.username, form.password)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    pair = auth.issue_tokens(username)
    return TokenResponse(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        expires_in=pair.expires_in,
    )


class RefreshRequest(BaseModel):
    refresh_token: str


@auth_router.post("/refresh", response_model=TokenResponse)
async def refresh(req: RefreshRequest) -> TokenResponse:
    username = auth.decode_token(req.refresh_token, "refresh")
    pair = auth.issue_tokens(username)
    return TokenResponse(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        expires_in=pair.expires_in,
    )


@auth_router.get("/status")
async def auth_status() -> dict:
    """Lets the login page know whether to show a form at all."""
    return {
        "auth_required": auth.auth_enabled(),
        "user_count": auth.user_count(),
        "tls": settings.tls_enabled,
        "bind": settings.host,
    }


@auth_router.get("/me")
async def me(user: str | None = Depends(auth.require_user)) -> dict:
    return {"username": user, "authenticated": user is not None}


app.include_router(auth_router)


# =======================================================================================
# Control plane  (bearer-protected once any account exists)
# =======================================================================================
api = APIRouter(prefix="/api", tags=["control"], dependencies=[Depends(auth.require_user)])


async def _engine_get(path: str, inst: Any | None = None) -> Any | None:
    """GET a JSON doc from the engine, or None when it is not reachable/ready.

    None rather than an exception: every console widget polls these, and an engine that is
    stopped or mid-load is an ordinary state, not an error.
    """
    target = inst or manager.primary()
    if not target.is_alive():
        return None
    try:
        r = await client().get(f"http://{settings.engine_host}:{target.port}{path}")
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except ValueError:
        return None


def _engine_identity(inst: Any) -> str:
    # The engine's counters are per process, so a restart must start a new baseline.
    return f"{inst.instance_id}:{inst.started_at}"


async def _sample_engines() -> None:
    """Fold every live engine's lifetime token counters into the per-model totals. Called by
    the sampler, and before anything that stops an engine, whose counters die with it."""
    insts = manager.running()
    # Identities are taken before the await, so a reply is never credited to a newer process.
    idents = [(_engine_identity(i), i.model_id or i.served_name or "") for i in insts]
    stats = await asyncio.gather(*(_engine_get("/v1/stats", i) for i in insts))
    for (ident, model), s in zip(idents, stats):
        tokens.observe_engine(ident, model, s)
    tokens.forget_engines({ident for ident, _ in idents})


def _enabled_indices() -> set[int] | None:
    """GPU indices in the engine pool, or None when every GPU is in it.

    One place decides this so /api/system, /api/gpus and /api/console cannot disagree
    about which cards are in play.
    """
    selected = prefs.get_visible_devices()
    if not selected:
        return None
    return {int(x) for x in selected.split(",") if x.strip().isdigit()}


def _annotate(devices: list[dict], chosen: set[int] | None) -> list[dict]:
    return [{**g, "enabled": chosen is None or g["index"] in chosen} for g in devices]


def _host_memory_doc() -> dict:
    host = gpu.host_memory()
    return {
        **host,
        "pin_budget_bytes": host_pin_budget_bytes(host.get("total_bytes") or 0),
    }


@api.get("/system")
async def system() -> dict:
    # Every GPU, each flagged: Settings needs the full list in order to offer the ones
    # that are currently switched off.
    gpus = _annotate(await gpu.query_gpus(), _enabled_indices())
    return {
        "gpus": gpus,
        "host_memory": _host_memory_doc(),
        "toolchain": toolchain_report(),
        "config": {
            "engine_url": settings.engine_base_url,
            "visible_devices": prefs.get_visible_devices(),
            "model_roots": [str(p) for p in settings.model_roots],
            "python": str(settings.venv_python),
        },
        "ts": time.time(),
    }


class GpuSelection(BaseModel):
    """Which GPUs the next engine launch may use, in nvidia-smi (PCI bus) order."""

    indices: list[int] = Field(default_factory=list)
    all_gpus: bool = False


@api.get("/gpus")
async def gpus() -> dict:
    """Every GPU, each flagged with whether it is currently in the engine's pool."""
    devices = await gpu.query_gpus()
    chosen = _enabled_indices()
    return {
        "gpus": _annotate(devices, chosen),
        "visible_devices": prefs.get_visible_devices(),
        "all_gpus": chosen is None,
        # A change cannot affect a process that has already initialised its CUDA context.
        "applies_to_next_launch": bool(manager.running()),
    }


@api.post("/gpus")
async def set_gpus(req: GpuSelection) -> dict:
    if not req.all_gpus and not req.indices:
        raise HTTPException(
            status_code=400,
            detail="select at least one GPU, or set all_gpus to use every device",
        )
    known = {g["index"] for g in await gpu.query_gpus()}
    unknown = sorted(set(req.indices) - known) if not req.all_gpus else []
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"no such GPU index: {', '.join(map(str, unknown))}. Present: "
                   f"{', '.join(map(str, sorted(known)))}",
        )
    try:
        value = await asyncio.to_thread(prefs.set_visible_devices, req.indices, req.all_gpus)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "visible_devices": value,
        "all_gpus": value == "",
        "applies_to_next_launch": bool(manager.running()),
    }


# =======================================================================================
# Projects
#
# A project scopes three things: its message board, which connectors its agents may use,
# and a data directory for everything that is not coordination traffic. The connector list
# is an intersection -- a server must be enabled globally (Settings / mcp_servers.json)
# AND selected by the project.
# =======================================================================================
class ProjectCreate(BaseModel):
    name: str
    data_dir: str | None = None
    connectors: list[str] = Field(default_factory=list)


class ProjectUpdate(BaseModel):
    name: str | None = None
    data_dir: str | None = None
    connectors: list[str] | None = None


def _project_view(project: dict) -> dict:
    """A project plus the derived facts the UI needs."""
    data_dir = Path(project["data_dir"])
    try:
        available = {sp.name for sp in mcp_registry.load_config() if sp.enabled}
    except ValueError:
        available = set()
    selected = set(project.get("connectors") or [])
    return {
        **project,
        "data_dir_exists": data_dir.is_dir(),
        # Selected here but not enabled globally: shown so a silently-inactive connector
        # is visible rather than mysterious.
        "connectors_unavailable": sorted(selected - available),
        "connectors_active": sorted(selected & available),
        "models": project.get("models"),
        "sql": project.get("sql"),
        "swarm_enabled": project.get("swarm_enabled", True),
        "model_roles": project.get("model_roles") or {},
    }


@api.get("/projects")
async def list_projects() -> dict:
    items = await asyncio.to_thread(projects.list_projects)
    return {
        "projects": [_project_view(p) for p in items],
        "active": projects.active_id(),
        "default_root": str(projects.DEFAULT_ROOT),
    }


@api.post("/projects")
async def create_project(req: ProjectCreate) -> dict:
    try:
        project = await asyncio.to_thread(
            projects.create, req.name, req.data_dir, req.connectors
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _project_view(project)


@api.post("/projects/{project_id}/activate")
async def activate_project(project_id: str) -> dict:
    try:
        return _project_view(await asyncio.to_thread(projects.set_active, project_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.post("/projects/{project_id}")
async def update_project(project_id: str, req: ProjectUpdate) -> dict:
    fields = req.model_dump(exclude_none=True)
    if not fields:
        raise HTTPException(status_code=400, detail="nothing to update")
    try:
        return _project_view(await asyncio.to_thread(projects.update, project_id, **fields))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.post("/projects/{project_id}/delete")
async def delete_project(project_id: str, remove_files: bool = False) -> dict:
    """Remove a project. `remove_files` only ever deletes inside the managed projects
    root -- a data directory pointed at your Documents folder is left alone."""
    removed = await asyncio.to_thread(projects.delete, project_id, remove_files)
    if not removed:
        raise HTTPException(status_code=404, detail=f"no project {project_id!r}")
    return {"deleted": True, "active": projects.active_id()}


# ---- data directory -------------------------------------------------------------------
@api.get("/projects/{project_id}/files")
async def project_files(project_id: str, subdir: str = "") -> dict:
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"no project {project_id!r}")
    try:
        return await asyncio.to_thread(projects.list_files, project, subdir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api.get("/projects/{project_id}/file")
async def project_file(project_id: str, path: str, max_bytes: int = 200_000):
    """Read one text file out of the data directory."""
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"no project {project_id!r}")
    try:
        target = projects.resolve_in_data_dir(project, path, must_exist=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target.is_file():
        raise HTTPException(status_code=400, detail="not a file")

    cap = max(1, min(int(max_bytes), 2_000_000))
    raw = await asyncio.to_thread(lambda: target.read_bytes()[:cap])
    size = target.stat().st_size
    return {
        "path": path,
        "size_bytes": size,
        "truncated": size > cap,
        "text": raw.decode("utf-8", errors="replace"),
    }


class ProjectMkdir(BaseModel):
    path: str


@api.post("/projects/{project_id}/mkdir")
async def project_mkdir(project_id: str, req: ProjectMkdir) -> dict:
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"no project {project_id!r}")
    try:
        target = projects.resolve_in_data_dir(project, req.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        await asyncio.to_thread(lambda: target.mkdir(parents=True, exist_ok=True))
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"created": req.path}


@api.post("/projects/{project_id}/upload")
async def project_upload(project_id: str, request: Request) -> dict:
    """Upload a file into the data directory.

    Multipart, with the destination in the `path` form field. The filename supplied by the
    client is never trusted directly -- it goes through the same containment check as every
    other path, because `../../autorun.inf` is a filename a browser will happily send.
    """
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"no project {project_id!r}")

    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "filename"):
        raise HTTPException(status_code=400, detail="no file part in the request")

    subdir = str(form.get("path") or "")
    name = Path(str(upload.filename or "upload.bin")).name  # strip any directory component
    try:
        target = projects.resolve_in_data_dir(project, f"{subdir}/{name}" if subdir else name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("wb") as fh:
        while chunk := await upload.read(1024 * 1024):
            fh.write(chunk)
            written += len(chunk)
    return {"path": str(target.relative_to(Path(project["data_dir"]))).replace("\\", "/"),
            "size_bytes": written}


class ProjectDelete(BaseModel):
    path: str


@api.post("/projects/{project_id}/files/delete")
async def project_file_delete(project_id: str, req: ProjectDelete) -> dict:
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"no project {project_id!r}")
    try:
        target = projects.resolve_in_data_dir(project, req.path, must_exist=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if target == Path(project["data_dir"]).resolve():
        raise HTTPException(status_code=400, detail="refusing to delete the data root")

    import shutil as _shutil

    if target.is_dir():
        await asyncio.to_thread(_shutil.rmtree, target, True)
    else:
        await asyncio.to_thread(target.unlink)
    return {"deleted": req.path}


class ModelRole(BaseModel):
    model: str
    role: str = ""


@api.get("/model-roles")
async def model_roles() -> dict:
    """What each model is good at. Read by the model-router connector so one model can
    decide whether another is better suited to a task."""
    return {"roles": prefs.get_model_roles()}


@api.post("/model-roles")
async def set_model_role(req: ModelRole) -> dict:
    roles = await asyncio.to_thread(prefs.set_model_role, req.model, req.role)
    return {"roles": roles}


class AutoQuarantine(BaseModel):
    enabled: bool


@api.get("/auto-quarantine")
async def auto_quarantine() -> dict:
    """Whether a disqualified result retires the library modules it was built on."""
    return {"enabled": prefs.get_auto_quarantine()}


@api.put("/auto-quarantine")
async def set_auto_quarantine(req: AutoQuarantine) -> dict:
    return {"enabled": await asyncio.to_thread(prefs.set_auto_quarantine, req.enabled)}


@api.get("/models")
async def models() -> dict:
    return {"models": await asyncio.to_thread(list_models)}


class StartRequest(BaseModel):
    model: str = Field(..., description="Model id or path from /api/models")
    options: dict[str, Any] = Field(default_factory=dict)


@api.post("/engine/start")
async def engine_start(req: StartRequest) -> dict:
    """Start a model. Kept for the single-engine console flow; it allocates through the
    manager, so a second call loads a second model rather than failing."""
    result = await manager.start(req.model, req.options)
    await asyncio.to_thread(prefs.set_launch_options, req.model, req.options, None)
    return result


class EnginesStartRequest(BaseModel):
    model: str = Field(..., description="Model id or path from /api/models")
    options: dict[str, Any] = Field(default_factory=dict)
    # One GPU index, in nvidia-smi order. Omitted, the manager picks a free card from the
    # configured pool so two engines never land on the same one.
    gpus: str | None = None


def all_loaded() -> list[dict]:
    """Resident models: this computer's engines, models shared by paired computers (named
    ``model@computer``, with a `remote` block) and enabled external models (``id@groq`` /
    ``id@openrouter``, with an `external` block), in one list. Every entry carries its
    published scores (`swe`, `aa`) for the model lists and the swarm policy."""
    from . import external, federation, ratings

    out = manager.loaded_models() + federation.remote_loaded()
    for m in out:
        sc = ratings.scores(m.get("model") or m.get("served_name"))
        m["swe"], m["aa"] = sc["swe"], sc["aa"]
    return out + external.loaded()


async def complete_text(model: str, messages: list[dict], max_tokens: int, purpose: str) -> str:
    """One non-streaming completion from any model the console can reach -- a local engine,
    a paired computer or an external provider. Used by background jobs (escalation)."""
    import json as _json

    from . import external, federation

    payload: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens, "stream": False}
    if external.is_external(model):
        return await external.complete_text(client(), model, messages, max_tokens, purpose)
    fed = federation.route(model)
    if fed is not None:
        resp = await federation.relay(fed[0], fed[1], "chat/completions", payload)
        data, code = _json.loads(resp.body), resp.status_code
    else:
        inst, served = await _require_ready_model(model)
        try:
            r = await client().post(f"http://{settings.engine_host}:{inst.port}/v1/chat/completions",
                                    json={**payload, "model": served}, timeout=1800.0)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"engine unreachable: {exc}") from exc
        data, code = r.json(), r.status_code
    if code != 200:
        raise HTTPException(status_code=502, detail=f"{model} returned {code}: {str(data)[:300]}")
    msg = ((data.get("choices") or [{}])[0].get("message") or {})
    return (msg.get("content") or msg.get("reasoning_content") or "").strip()


@api.get("/engines")
async def list_engines() -> dict:
    """Every engine instance, plus what is resident and ready to answer."""
    return {"engines": manager.statuses(), "loaded": all_loaded()}


@api.post("/engines")
async def start_engine(req: EnginesStartRequest) -> dict:
    """Load a model into a NEW engine, alongside anything already running."""
    result = await manager.start(req.model, req.options, req.gpus)
    # Saved only once the launch is accepted, so a refused configuration is not preset
    # next time. The requested GPU is kept, not the one assigned: auto stays auto.
    await asyncio.to_thread(prefs.set_launch_options, req.model, req.options, req.gpus)
    return result


@api.get("/launch-options")
async def launch_options() -> dict:
    """Options each model was last launched with, for presetting the Models page."""
    return {"models": await asyncio.to_thread(prefs.get_launch_options)}


@api.post("/engines/unload-all")
async def unload_all_engines() -> dict:
    """Stop every engine and forecaster, then kill orphaned engine workers still holding
    memory. The per-engine Unload cannot reach an orphan, because the control plane is no
    longer tracking it."""
    from .tsfm import ts_manager
    from .unload import unload_all

    await _sample_engines()
    return await unload_all(manager, ts_manager)


@api.post("/engines/{instance_id}/stop")
async def stop_engine(instance_id: str) -> dict:
    """Unload this engine's model and free its GPU."""
    await _sample_engines()
    return await manager.stop(instance_id)


class MoveRequest(BaseModel):
    gpus: str = Field(..., description="Target GPU index, e.g. \"2\"")


@api.post("/engines/{instance_id}/move")
async def move_engine(instance_id: str, req: MoveRequest) -> dict:
    """Unload a model from its GPU and reload it on another, keeping its launch options.

    A running engine cannot change card: CUDA fixes the device when the process starts. So
    this is a stop followed by a start, sequenced so the old engine's VRAM is released
    before the new one tries to allocate.
    """
    await _sample_engines()
    return await manager.move(instance_id, req.gpus)


@api.post("/engine/stop")
async def engine_stop(instance_id: str | None = None) -> dict:
    """Stop one engine. Without an id, stops the primary -- what the console's Stop
    button means when only one is running."""
    target = manager.get(instance_id) if instance_id else manager.primary()
    if target is None:
        raise HTTPException(status_code=404, detail=f"no engine {instance_id!r}")
    await _sample_engines()
    return await manager.stop(target.instance_id)


@api.get("/engine")
async def engine_status() -> dict:
    status_doc = manager.primary().status()
    status_doc["health"] = await _engine_get("/health")
    return status_doc


async def _engine_detail(inst: Any, gpus: list[dict]) -> dict:
    """One engine's full picture: status, health, stats and load progress.

    Gathered concurrently per engine because the console polls at 1 Hz and a serial walk
    over several engines would spend most of a second waiting on loopback round-trips.
    """
    status_doc = inst.status()
    ident, model = _engine_identity(inst), inst.model_id or inst.served_name or ""
    health, stats = await asyncio.gather(
        _engine_get("/health", inst),
        _engine_get("/v1/stats", inst),
    )
    # The console polls at 1 Hz; its readings keep the token totals current for free.
    tokens.observe_engine(ident, model, stats)

    load = None
    if status_doc["state"] in {"starting", "running"}:
        host = gpu.host_memory()
        # Attribute memory growth to the card this engine actually holds, otherwise the
        # other engine's VRAM would inflate this one's progress estimate.
        own = [g for g in gpus if str(g["index"]) == str(inst.gpus)] or gpus
        load = progress.build(
            inst.logs.texts(),
            health,
            float(status_doc.get("uptime_s") or 0),
            mem={
                "model_size_bytes": status_doc.get("model_size_bytes") or 0,
                "baseline_vram": status_doc.get("baseline_vram_bytes") or 0,
                "baseline_ram": status_doc.get("baseline_ram_bytes") or 0,
                "vram_bytes": sum(g["memory_used_bytes"] for g in own),
                "ram_bytes": max(0, host["total_bytes"] - host["available_bytes"]),
            },
        ).as_dict()

    return {**status_doc, "health": health, "stats": stats, "load": load}


@api.get("/console")
async def console() -> dict:
    """One poll drives the whole dashboard."""
    primary = manager.primary()
    status_doc = primary.status()
    health, stats, cache, gpus = await asyncio.gather(
        _engine_get("/health"),
        _engine_get("/v1/stats"),
        _engine_get("/v1/cache/status"),
        gpu.query_gpus(),
    )
    # The console shows the engine's GPU pool, not every card in the machine: a display
    # adapter's usage would otherwise inflate the VRAM gauge and read as engine load.
    chosen = _enabled_indices()
    total_gpus = len(gpus)
    if chosen is not None:
        gpus = [item for item in gpus if item["index"] in chosen]
    # Reconstruct what the load is doing. The engine's own /health goes quiet between
    # phases, and on a big offloaded model that silence is several minutes long.
    load = None
    if status_doc["state"] in {"starting", "running"}:
        host = gpu.host_memory()
        load = progress.build(
            primary.logs.texts(),
            health,
            float(status_doc.get("uptime_s") or 0),
            mem={
                "model_size_bytes": status_doc.get("model_size_bytes") or 0,
                "baseline_vram": status_doc.get("baseline_vram_bytes") or 0,
                "baseline_ram": status_doc.get("baseline_ram_bytes") or 0,
                "vram_bytes": sum(g["memory_used_bytes"] for g in gpus),
                "ram_bytes": max(0, host["total_bytes"] - host["available_bytes"]),
            },
        ).as_dict()

    # Every engine, each with its own health/stats/progress.
    details = await asyncio.gather(
        *(_engine_detail(inst, gpus) for inst in manager.all())
    )

    return {
        "engine": status_doc,
        "engines": list(details),
        "loaded": all_loaded(),
        "health": health,
        "stats": stats,
        "cache": cache,
        "load": load,
        "gpus": _annotate(gpus, chosen),
        # So the UI can say "1 GPU hidden" instead of silently omitting a card.
        "gpus_hidden": total_gpus - len(gpus),
        "host_memory": _host_memory_doc(),
        "ts": time.time(),
    }


@api.get("/engine/stats")
async def engine_stats() -> dict:
    return {"stats": await _engine_get("/v1/stats")}


@api.get("/engine/requests")
async def engine_requests(since: int = 0, limit: int = 100) -> dict:
    limit = max(1, min(limit, 512))
    doc = await _engine_get(f"/v1/requests?since={since}&limit={limit}")
    return doc or {"entries": [], "next_cursor": since}


@api.get("/engine/cache")
async def engine_cache() -> dict:
    return {"cache": await _engine_get("/v1/cache/status")}


class CacheRebuildRequest(BaseModel):
    """Mirrors the engine's own /v1/cache/rebuild body; all fields optional."""

    num_pages: int | None = None
    moe_cache_size: int | None = None
    num_mamba_slots: int | None = None
    num_swa_pages: int | None = None
    swa_full_tokens_ratio: float | None = None
    # Which engine to rebuild; defaults to the primary (first live) one. Not forwarded.
    instance_id: str | None = None


@api.post("/engine/cache/rebuild")
async def engine_cache_rebuild(req: CacheRebuildRequest) -> Any:
    # Resolved per request: real loads create their own instances, so a reference captured
    # at startup would point at the never-started placeholder.
    inst = manager.get(req.instance_id) if req.instance_id else manager.primary()
    if inst is None or not inst.is_alive():
        raise HTTPException(status_code=409, detail="engine is not running")
    payload = req.model_dump(exclude_none=True, exclude={"instance_id"})
    if not payload:
        raise HTTPException(status_code=400, detail="no cache parameters supplied")
    try:
        # A rebuild drains in-flight work and reallocates pools; it is slow by nature.
        r = await client().post(
            f"http://{settings.engine_host}:{inst.port}/v1/cache/rebuild",
            json=payload,
            timeout=180.0,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"engine unreachable: {exc}") from exc
    return JSONResponse(r.json(), status_code=r.status_code)


# ---------------------------------------------------------------------------------------
# MCP connectors
#
# Registration is filesystem-only by design (see mcp_registry's module docstring): a stdio
# MCP server is an arbitrary command line, so allowing an HTTP caller to define one would
# turn this into a remote-code-execution endpoint. These routes read, probe, and invoke --
# never define.
# ---------------------------------------------------------------------------------------
@api.get("/mcp/servers")
async def mcp_servers() -> dict:
    try:
        specs = mcp_registry.load_config()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "config_path": str(mcp_registry.CONFIG_PATH),
        "servers": [
            {**spec.model_dump(), "auth": mcp_oauth.token_summary(spec.name)}
            for spec in specs
        ],
    }


@api.post("/mcp/servers/init")
async def mcp_init_config() -> dict:
    """Write a starter mcp_servers.json so there is something concrete to edit."""
    path = await asyncio.to_thread(mcp_registry.write_example_config)
    return {"config_path": str(path)}


def _project_connectors(project_id: str | None) -> set[str] | None:
    """Connector names the project allows, or None when it has no restriction.

    None (rather than "everything") is returned when no project exists yet, so a fresh
    install still works before anyone has made one.
    """
    pid = project_id or projects.active_id()
    if not pid:
        return None
    project = projects.get(pid)
    if project is None:
        return None
    return set(project.get("connectors") or [])


# The whole probe of one server, spawn included. The registry bounds only the MCP
# handshake; a server whose command itself blocks (e.g. `docker run` while the Docker daemon
# is down) was unbounded, and because every server was gathered into ONE response, that one
# server kept the Connectors page on "probing..." for all of them, indefinitely.
PROBE_TIMEOUT_S = 45.0


async def _bounded_probe(spec) -> dict:
    try:
        return await asyncio.wait_for(mcp_registry.probe(spec), PROBE_TIMEOUT_S)
    except asyncio.TimeoutError:
        return {
            "name": spec.name, "ok": False, "tools": [],
            "error": f"no answer within {PROBE_TIMEOUT_S:.0f}s -- does its command start on its own?",
        }


@api.get("/mcp/probe")
async def mcp_probe(project_id: str | None = None, server: str | None = None) -> dict:
    """Connect to every server this project may use and enumerate its tools.

    With `server`, probe just that one -- the Connectors page asks per server so each
    result appears as soon as it is ready instead of waiting for the slowest.
    """
    allowed = _project_connectors(project_id)
    specs = [
        s for s in mcp_registry.load_config()
        if s.enabled and (allowed is None or s.name in allowed) and (server is None or s.name == server)
    ]
    if not specs:
        return {"servers": []}
    results = await asyncio.gather(*(_bounded_probe(s) for s in specs))
    return {"servers": list(results)}


@api.get("/mcp/tools")
async def mcp_tools(project_id: str | None = None) -> dict:
    """Every enabled connector's tools, already in OpenAI function-tool schema.

    An agent passes this straight into a chat completion's `tools` array; the engine's
    tool-call parser turns the model's output back into calls for /api/mcp/call.
    """
    try:
        allowed = _project_connectors(project_id)
        specs = [
            s for s in mcp_registry.load_config()
            if s.enabled and (allowed is None or s.name in allowed)
        ]
    except ValueError as exc:
        # A malformed config is the operator's typo, not a server fault -- report it in the
        # errors channel so the Connectors page can show the parse error verbatim.
        return {"tools": [], "errors": [{"server": "(config)", "error": str(exc)}]}
    results = await asyncio.gather(*(mcp_registry.probe(s) for s in specs))
    tools: list[dict] = []
    errors: list[dict] = []
    for r in results:
        tools.extend(r["tools"])
        if not r["ok"]:
            errors.append({"server": r["name"], "error": r["error"]})
    return {"tools": tools, "errors": errors}


# --- OAuth for remote MCP servers ------------------------------------------------------
# Three-step browser round-trip:
#   1. POST /api/mcp/oauth/start   -> returns an authorization URL to open
#   2. the provider redirects to  /api/mcp/oauth/callback  (loopback, RFC 8252)
#   3. GET  /api/mcp/oauth/status  -> reports whether the token landed
# The tokens themselves never leave the server; the UI only learns authorised yes/no.
class McpAuthRequest(BaseModel):
    server: str


def _redirect_uri() -> str:
    return f"http://{settings.host}:{settings.port}{mcp_oauth.DEFAULT_REDIRECT_PATH}"


@api.post("/mcp/oauth/start")
async def mcp_oauth_start(req: McpAuthRequest) -> dict:
    spec = next((x for x in mcp_registry.load_config() if x.name == req.server), None)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"no MCP server named {req.server!r}")
    if spec.transport == "stdio":
        raise HTTPException(
            status_code=400,
            detail="stdio servers run as local processes and have nothing to authorise",
        )
    if not spec.oauth:
        raise HTTPException(
            status_code=400,
            detail=f'set "oauth": true on {req.server!r} in mcp_servers.json first',
        )

    flow = mcp_oauth.register_flow(spec.name)
    provider = mcp_oauth.build_provider(spec.name, spec.url or "", _redirect_uri(), flow, spec)

    async def drive() -> None:
        """Run the handshake in the background.

        It cannot complete until the operator approves in their browser, so it parks on
        the callback event. Awaiting it here would deadlock this request -- the browser
        redirect is served by *this* app.
        """
        try:
            await mcp_registry.list_tools(spec, provider)
        except Exception as exc:  # noqa: BLE001 - reported through /status
            flow.error = flow.error or mcp_registry.describe_exception(exc)
            flow.url_ready.set()
            flow.code_ready.set()

    asyncio.create_task(drive())

    # Wait only for the URL, which the SDK produces after discovery + registration.
    try:
        await asyncio.wait_for(flow.url_ready.wait(), timeout=45.0)
    except asyncio.TimeoutError:
        mcp_oauth.drop_flow(flow.state)
        raise HTTPException(
            status_code=504,
            detail="timed out discovering the authorization server (is the URL right?)",
        ) from None

    if flow.authorization_url is None:
        detail = flow.error or "the server did not return an authorization URL"
        mcp_oauth.drop_flow(flow.state)
        raise HTTPException(status_code=502, detail=detail)

    return {
        "server": spec.name,
        "authorization_url": flow.authorization_url,
        "state": flow.state,
        "redirect_uri": _redirect_uri(),
    }


@api.get("/mcp/oauth/status")
async def mcp_oauth_status(server: str, state: str | None = None) -> dict:
    flow = mcp_oauth.get_flow(state) if state else None
    return {
        "server": server,
        **mcp_oauth.token_summary(server),
        "pending": bool(flow and not flow.code_ready.is_set()),
        "error": flow.error if flow else None,
    }


@api.post("/mcp/oauth/disconnect")
async def mcp_oauth_disconnect(req: McpAuthRequest) -> dict:
    return {"forgotten": await asyncio.to_thread(mcp_oauth.forget, req.server)}


class McpCallRequest(BaseModel):
    tool: str = Field(..., description="Qualified name: <server>__<tool>")
    arguments: dict[str, Any] = Field(default_factory=dict)


@api.post("/mcp/call")
async def mcp_call(req: McpCallRequest, project_id: str | None = None) -> dict:
    allowed = _project_connectors(project_id)
    specs = [
        s for s in mcp_registry.load_config()
        if allowed is None or s.name in allowed
    ]
    try:
        return await mcp_registry.call_tool(specs, req.tool, req.arguments)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="MCP call timed out") from None
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc


# ---------------------------------------------------------------------------------------
# Remote backends -- other FreeSwarm control planes this one can route to by model name.
# ---------------------------------------------------------------------------------------
@api.get("/backends")
async def backends() -> dict:
    try:
        specs = remotes.load_backends()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    results = await asyncio.gather(*(remotes.status(client(), b) for b in specs))
    return {"config_path": str(remotes.CONFIG_PATH), "backends": list(results)}


@api.post("/backends/init")
async def backends_init() -> dict:
    path = await asyncio.to_thread(remotes.write_example_config)
    return {"config_path": str(path)}


@api.get("/logs")
async def logs(cursor: int = 0, limit: int = 500, instance_id: str | None = None) -> dict:
    limit = max(1, min(limit, 2000))
    target = manager.get(instance_id) if instance_id else manager.primary()
    if target is None:
        raise HTTPException(status_code=404, detail=f"no engine {instance_id!r}")
    entries, next_cursor = target.logs.since(cursor, limit)
    return {"entries": entries, "next_cursor": next_cursor}


# ---------------------------------------------------------------------------------------
# Chat -- streaming passthrough used by the console's own Chat view
# ---------------------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = True


async def _require_ready_model(explicit: str | None):
    """Resolve a request to (instance, served name), or explain why it cannot be served."""
    if not manager.running():
        raise HTTPException(status_code=409, detail="no engine is running")

    inst, _ = await _instance_for(explicit)
    if inst is None:
        loaded = sorted(await _local_model_names())
        raise HTTPException(
            status_code=404,
            detail=(
                f"{explicit!r} is not loaded. Currently resident: "
                f"{', '.join(loaded) if loaded else '(none)'}. "
                "Load it from the Models page, or ask for one of these."
            ),
        )

    health = await _engine_get("/health", inst)
    if health is None:
        raise HTTPException(status_code=503, detail="engine is not reachable yet")
    if health.get("status") == "loading":
        raise HTTPException(
            status_code=503,
            detail=f"{inst.model_id} is still loading",
        )
    # The engine insists on the name IT knows; a catalog id would be rejected.
    served = health.get("model") or inst.served_name or inst.model_id
    if not served:
        raise HTTPException(status_code=503, detail="engine has not reported a served model")
    return inst, str(served)


def _sse_relay(url: str, payload: dict[str, Any]):
    async def relay():
        # No read timeout: the gap between SSE chunks is bounded by decode speed, not by
        # anything this layer should second-guess. connect stays short so a dead engine
        # fails fast.
        timeout = httpx.Timeout(None, connect=5.0)
        try:
            async with client().stream("POST", url, json=payload, timeout=timeout) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", "replace")
                    yield f'data: {{"error": {body!r}}}\n\n'.encode()
                    return
                async for chunk in response.aiter_raw():
                    yield chunk
        except httpx.HTTPError as exc:
            yield f'data: {{"error": "engine stream failed: {exc}"}}\n\n'.encode()

    return StreamingResponse(
        relay(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@api.post("/chat")
async def chat(req: ChatRequest) -> Any:
    sampling = {k: v for k in ("temperature", "top_p", "max_tokens")
                if (v := getattr(req, k)) is not None}
    messages = [m.model_dump() for m in req.messages]

    # `model@computer`: served by a paired computer, not by an engine here. Checked before
    # _require_ready_model, which would reject it for the wrong reason -- "no engine is
    # running" -- on a machine that only uses other people's models.
    from . import external, federation

    if external.is_external(req.model):
        return await external.complete(client(), req.model, {"messages": messages, "stream": req.stream, **sampling},
                                       purpose="chat")
    fed = federation.route(req.model) if req.model else None
    if fed is not None:
        return await federation.relay(fed[0], fed[1], "chat/completions",
                                      {"messages": messages, "stream": req.stream, **sampling})

    inst, model = await _require_ready_model(req.model)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": req.stream,
        **sampling,
    }

    url = f"http://{settings.engine_host}:{inst.port}/v1/chat/completions"
    if not req.stream:
        try:
            r = await client().post(url, json=payload, timeout=600.0)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"engine unreachable: {exc}") from exc
        return JSONResponse(r.json(), status_code=r.status_code)
    return _sse_relay(url, payload)


# Python sandbox (ui/sandbox container): /api/sandbox/run, /api/sandbox/artifacts/...
# Mounted on `api`, so it inherits the same auth dependency as every other control route.
from .sandbox import router as sandbox_router  # noqa: E402 -- avoids a circular import

api.include_router(sandbox_router)

# Standing objectives (continuous swarm improvement): /api/projects/{id}/objectives, /api/objectives/...
from .objectives import router as objectives_router  # noqa: E402

api.include_router(objectives_router)

# The project code library (modules, versions, comments, regime maps): /api/projects/{id}/library...
from .library import router as library_router  # noqa: E402

api.include_router(library_router)

# The swarm playbook (operator charter + agent-maintained team practices).
from .playbook import router as playbook_router  # noqa: E402

api.include_router(playbook_router)

# External review: a frontier Claude model checks a result the local harness passed.
from .review import router as review_router  # noqa: E402

api.include_router(review_router)

# LAN federation console: sharing, pairing requests, paired computers (/api/federation...).
from .federation import router as federation_router  # noqa: E402

api.include_router(federation_router)

# Model downloads from Hugging Face (known models pinned; custom repos planned first).
from .downloads import router as downloads_router  # noqa: E402

api.include_router(downloads_router)

# Forecast Lab: test input combinations and find which inputs improve forecasts.
from .tslab import router as tslab_router  # noqa: E402

api.include_router(tslab_router)

# Decile studies ("deci-plots"): per-signal decile tables on in-sample data, stored for the team.
from .deciplot import router as deciplot_router  # noqa: E402

api.include_router(deciplot_router)

# Ensembles: verified candidates combined into one weighted portfolio candidate, and the
# in-sample correlations to choose them by (/api/objectives/{oid}/ensembles, .../correlations).
from .ensembles import router as ensembles_router  # noqa: E402

api.include_router(ensembles_router)

# External providers (Groq, OpenRouter), published model ratings, and the stuck-search escalation.
from .external import router as external_router  # noqa: E402
from .ratings import router as ratings_router  # noqa: E402
from .escalation import router as escalation_router  # noqa: E402
from .mentor import router as mentor_router  # noqa: E402

api.include_router(external_router)
# Tokens processed per model, every source (/api/usage/tokens).
api.include_router(tokens.router)
api.include_router(ratings_router)
api.include_router(escalation_router)
api.include_router(mentor_router)

# Agent inspector: what each agent is working on and was asked, and who asked each forecaster
# for what (/api/agents/activity...). The middleware only tags requests with their caller.
from .agent_activity import CallerMiddleware, router as agent_activity_router  # noqa: E402

api.include_router(agent_activity_router)
app.add_middleware(CallerMiddleware)


# =======================================================================================
# Project resources for the swarm: data folder (DuckDB), SQL Server (read-only login),
# and which models the project's agents may use.
# =======================================================================================
from . import datasource, sqlsource  # noqa: E402


def _require_project(project_id: str) -> dict:
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"no project {project_id!r}")
    return project


class DataQuery(BaseModel):
    sql: str = Field(..., max_length=100_000)
    max_rows: int = Field(200, ge=1, le=1000)


@api.get("/projects/{project_id}/data/catalog")
async def data_catalog(project_id: str) -> dict:
    project = _require_project(project_id)
    files = await asyncio.to_thread(datasource.catalog, project["data_dir"])
    return {"data_dir": project["data_dir"], "files": files}


@api.get("/projects/{project_id}/data/describe")
async def data_describe(project_id: str, view: str) -> dict:
    project = _require_project(project_id)
    try:
        return await asyncio.to_thread(datasource.describe, project["data_dir"], view)
    except datasource.DataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@api.post("/projects/{project_id}/data/query")
async def data_query(project_id: str, req: DataQuery) -> dict:
    project = _require_project(project_id)
    try:
        return await asyncio.to_thread(datasource.query, project["data_dir"], req.sql, req.max_rows)
    except datasource.DataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


# -- SQL Server. Listing and setup use the OPERATOR's Windows login and run only when the
#    Projects page asks; every agent query uses the project's restricted reader login.
@api.get("/sql/databases")
async def sql_databases(server: str = "localhost") -> dict:
    try:
        return {"databases": await asyncio.to_thread(sqlsource.list_databases, server)}
    except sqlsource.SqlError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@api.get("/sql/tables")
async def sql_tables(database: str, server: str = "localhost") -> dict:
    try:
        return {"tables": await asyncio.to_thread(sqlsource.list_tables, server, database)}
    except sqlsource.SqlError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


class SqlSetup(BaseModel):
    server: str = "localhost"
    database: str
    tables: list[str] = Field(..., min_length=1, max_length=500)


@api.post("/projects/{project_id}/sql/setup")
async def sql_setup(project_id: str, req: SqlSetup) -> dict:
    """Create/refresh this project's read-only login: SELECT on exactly these tables."""
    project = _require_project(project_id)
    try:
        out = await asyncio.to_thread(
            sqlsource.provision, project["slug"], req.server, req.database, req.tables
        )
    except sqlsource.SqlError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    await asyncio.to_thread(projects.update, project_id, sql=out["config"])
    return out


def _sql_cfg(project: dict) -> dict:
    cfg = project.get("sql")
    if not cfg:
        raise HTTPException(status_code=400, detail="SQL is not set up for this project")
    return cfg


@api.post("/projects/{project_id}/sql/verify")
async def sql_verify(project_id: str) -> dict:
    cfg = _sql_cfg(_require_project(project_id))
    try:
        return await asyncio.to_thread(sqlsource.verify, cfg)
    except sqlsource.SqlError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@api.post("/projects/{project_id}/sql/remove")
async def sql_remove(project_id: str) -> dict:
    project = _require_project(project_id)
    cfg = project.get("sql")
    if cfg:
        try:
            await asyncio.to_thread(sqlsource.remove, project["slug"], cfg["server"])
        except sqlsource.SqlError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
    await asyncio.to_thread(projects.update, project_id, sql=None)
    return {"ok": True}


@api.post("/projects/{project_id}/sql/query")
async def sql_query(project_id: str, req: DataQuery) -> dict:
    cfg = _sql_cfg(_require_project(project_id))
    try:
        return await asyncio.to_thread(sqlsource.query, cfg, req.sql, req.max_rows)
    except sqlsource.SqlError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@api.get("/projects/{project_id}/sql/describe")
async def sql_describe(project_id: str, table: str) -> dict:
    cfg = _sql_cfg(_require_project(project_id))
    try:
        return await asyncio.to_thread(sqlsource.describe, cfg, table)
    except sqlsource.SqlError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


# -- Export an allowed SQL table to parquet in the project's data folder -----------------
# A background job: a large table takes minutes, and holding an HTTP request open that long
# invites a proxy timeout that would look like a failure while the export carried on.
_export_jobs: dict[str, dict] = {}


class SqlExport(BaseModel):
    table: str = Field(..., description="schema.table -- must be one of the project's allowed tables")
    folder: str = Field("sql_exports", max_length=200, description="folder inside the data directory")
    where: str | None = Field(None, max_length=4000, description="optional WHERE filter (T-SQL)")
    rows_per_file: int = Field(1_000_000, ge=10_000, le=50_000_000)


@api.post("/projects/{project_id}/sql/export")
async def sql_export(project_id: str, req: SqlExport) -> dict:
    import re as _re
    import uuid as _uuid

    project = _require_project(project_id)
    cfg = _sql_cfg(project)
    try:
        table = sqlsource._normalize_table(req.table)
    except sqlsource.SqlError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    if table.lower() not in {t.lower() for t in cfg["tables"]}:
        raise HTTPException(status_code=400, detail=f"{table} is not one of this project's tables")
    # schema_table, no dots: the catalog derives the view name from the folder name, and a
    # dot would be read as a file extension.
    leaf = _re.sub(r"[^A-Za-z0-9_-]+", "_", table.replace(".", "_"))
    folder = (req.folder or "sql_exports").strip().strip("/\\")
    try:
        dest = projects.resolve_in_data_dir(project, f"{folder}/{leaf}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    for job in _export_jobs.values():
        if job["project_id"] == project_id and job["table"] == table and job["state"] == "running":
            return job  # one export of a table at a time; hand back the one in flight

    job = {
        "id": _uuid.uuid4().hex[:12], "project_id": project_id, "table": table,
        "dest": str(dest), "relative": f"{folder}/{leaf}", "where": req.where,
        "state": "running", "rows": 0, "total": None, "files": 0,
        "started_at": time.time(), "finished_at": None, "error": None, "result": None,
    }
    _export_jobs[job["id"]] = job

    def progress(rows: int, total: int | None, files: int) -> None:
        job.update(rows=rows, total=total, files=files)

    async def run() -> None:
        try:
            job["result"] = await asyncio.to_thread(
                sqlsource.export_table, cfg, table, dest, req.where, req.rows_per_file, progress
            )
            job["state"] = "done"
        except Exception as exc:  # noqa: BLE001 -- surfaced on the job, not a 500
            job["state"] = "failed"
            job["error"] = str(exc)
        finally:
            job["finished_at"] = time.time()

    asyncio.create_task(run())
    return job


@api.get("/projects/{project_id}/sql/exports")
async def sql_exports(project_id: str) -> dict:
    jobs = [j for j in _export_jobs.values() if j["project_id"] == project_id]
    return {"jobs": sorted(jobs, key=lambda j: j["started_at"], reverse=True)[:20]}


# -- The project's swarm: on/off, and exactly what its agents can reach ------------------
class SwarmToggle(BaseModel):
    enabled: bool


@api.post("/projects/{project_id}/swarm")
async def project_swarm(project_id: str, req: SwarmToggle) -> dict:
    _require_project(project_id)
    return _project_view(await asyncio.to_thread(projects.update, project_id, swarm_enabled=req.enabled))


@api.get("/projects/{project_id}/swarm/resources")
async def project_swarm_resources(project_id: str) -> dict:
    """Everything this project's agents can use, computed the way the swarm runner does.

    One place that answers "what does this swarm have access to?" -- models (and whether each
    is loaded, since only loaded models become agents), forecasters, data files, SQL tables
    and connectors -- so the Swarm page shows the same world the agents are given.
    """
    from .tsfm import ts_manager

    from . import swarm_policy

    project = _require_project(project_id)
    allowed = project.get("models")

    def ok(model: str) -> bool:
        return swarm_policy.permitted(project, model)

    loaded = {m["model"]: m for m in all_loaded() if m.get("model")}
    llms = [
        {"model": name, "loaded": True, "ready": bool(m.get("ready")), "gpu": m.get("gpus"),
         "remote": m.get("remote"), "external": m.get("external"), "swe": m.get("swe"), "aa": m.get("aa")}
        for name, m in loaded.items() if ok(name)
    ]
    # Allowed explicitly but not loaded: shown so a missing agent is explained, not mysterious.
    # A network model is named `model@computer`, and that name is only meaningful while that
    # computer is paired and sharing it. A saved entry naming a computer that is gone -- or
    # renamed, which is the same thing here -- can never load, so listing it as "allowed, not
    # loaded" is noise the operator cannot act on. Only network models actually found are shown.
    for name in allowed or []:
        if name in loaded or any(i["model_id"] == name for i in ts_manager.statuses()):
            continue
        if "@" in name:
            continue
        llms.append({"model": name, "loaded": False, "ready": False, "gpu": None})

    forecasters = [
        {"model": i["model_id"], "gpu": i["gpu"], "state": i["state"],
         "family": (i.get("health") or {}).get("family"),
         "in_flight": i.get("in_flight", 0), "calls": i.get("calls", 0),
         "last_call_at": i.get("last_call_at"), "last_seconds": i.get("last_seconds")}
        for i in ts_manager.statuses() if ok(i["model_id"])
    ]
    files = await asyncio.to_thread(datasource.catalog, project["data_dir"])
    view = _project_view(project)
    return {
        "project": {"id": project["id"], "name": project["name"], "slug": project["slug"]},
        "enabled": project.get("swarm_enabled", True),
        "models_filter": allowed,
        "llms": llms,
        "forecasters": forecasters,
        "data": {"data_dir": project["data_dir"], "count": len(files), "files": files[:25]},
        "sql": project.get("sql"),
        "connectors": {"active": view["connectors_active"], "unavailable": view["connectors_unavailable"]},
    }


class ModelRole(BaseModel):
    model: str = Field(..., max_length=300)
    role: str = Field("auto", pattern="^(auto|search|ideas|both)$")


@api.post("/projects/{project_id}/model-role")
async def project_model_role(project_id: str, req: ModelRole) -> dict:
    """Set what one model does in this project's swarm; "auto" returns it to the automatic rule."""
    project = _require_project(project_id)
    roles = dict(project.get("model_roles") or {})
    if req.role == "auto":
        roles.pop(req.model, None)
    else:
        roles[req.model] = req.role
    return _project_view(await asyncio.to_thread(projects.update, project_id, model_roles=roles))


class ProjectModels(BaseModel):
    models: list[str] | None = Field(None, description="None = every loaded model")


@api.post("/projects/{project_id}/models")
async def project_models(project_id: str, req: ProjectModels) -> dict:
    _require_project(project_id)
    return _project_view(await asyncio.to_thread(projects.update, project_id, models=req.models))


# =======================================================================================
# Time-series models -- a second category beside the LLM engines (see app/tsfm.py).
# =======================================================================================
from .tsfm import TsError, ts_manager  # noqa: E402


class TsStartRequest(BaseModel):
    model: str = Field(..., description="Model id from /api/models (category 'timeseries')")
    gpu: str | None = Field(None, description="GPU index; omitted = the one with most room")


class TsForecastRequest(BaseModel):
    model: str | None = Field(None, description="A loaded time-series model; omitted = any")
    series: list[float] | list[list[float]] | None = None
    horizon: int = Field(24, ge=1, le=1024)
    quantiles: list[float] | None = None
    # Candle models (Kronos): one or more [bars x (open, high, low, close[, volume])] histories,
    # their bar timestamps (ISO) and the bar spacing, and how many sample paths to draw.
    candles: list[list[list[float]]] | None = None
    timestamps: list[list[str]] | None = None
    freq_seconds: int | None = Field(None, ge=1, le=86_400 * 31)
    samples: int | None = Field(None, ge=1, le=200)
    # Covariate models (Chronos-2): [{target, past_covariates, future_covariates}].
    inputs: list[dict[str, Any]] | None = None


def _ts_http(exc: TsError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@api.get("/ts")
async def ts_status() -> dict:
    await ts_manager.refresh_health()
    return {"instances": ts_manager.statuses()}


@api.post("/ts")
async def ts_start(req: TsStartRequest) -> dict:
    from .catalog import resolve_model_path

    try:
        path = resolve_model_path(req.model)
        return await ts_manager.start(req.model, path, req.gpu)
    except TsError as exc:
        raise _ts_http(exc) from None


@api.post("/ts/{inst_id}/stop")
async def ts_stop(inst_id: str) -> dict:
    try:
        return await ts_manager.stop(inst_id)
    except TsError as exc:
        raise _ts_http(exc) from None


@api.post("/ts/forecast")
async def ts_forecast(req: TsForecastRequest) -> dict:
    try:
        return await ts_manager.forecast(req.model, req.model_dump(exclude={"model"}, exclude_none=True))
    except TsError as exc:
        raise _ts_http(exc) from None

app.include_router(api)


# =======================================================================================
# OpenAI-compatible passthrough for agent clients
#
# An agent framework sets OPENAI_BASE_URL=http(s)://host:8000/v1 and OPENAI_API_KEY=<access
# token>; the standard `Authorization: Bearer` header the SDK already sends is exactly what
# require_user validates. Concurrency is the engine's to manage -- raise
# --max-running-requests (default 4) and size --num-pages for the number of agents.
# =======================================================================================
openai_router = APIRouter(prefix="/v1", tags=["openai"], dependencies=[Depends(auth.require_user)])

# Forwarded verbatim. Deliberately a list, not a wildcard: the engine also serves control
# routes under /v1 (stats, requests, cache/rebuild) that must not be reachable through the
# unprivileged agent surface.
_PASSTHROUGH_POST = {"chat/completions", "completions", "messages", "responses"}


async def _local_model_names() -> set[str]:
    """Every model name any local engine answers to.

    Both the catalog id and the engine's own served name count: the engine derives the
    latter from the checkpoint directory, while clients usually know the former.
    """
    names: set[str] = set()
    for inst in manager.running():
        if inst.model_id:
            names.add(inst.model_id)
        if inst.served_name:
            names.add(inst.served_name)
    return names


async def _instance_for(model: str | None):
    """(instance, resolved model name) for a request, or (None, None) if unroutable."""
    if model:
        inst = manager.for_model(model)
        if inst is not None:
            return inst, model
        return None, None
    # No model named: use the primary, which is what a single-engine client expects.
    primary = manager.primary()
    if primary.is_alive():
        return primary, (primary.model_id or primary.served_name)
    return None, None


@openai_router.get("/models")
async def openai_models() -> dict:
    """Every model reachable from here -- local plus every enabled remote backend.

    An agent router reads this to decide what it can send where; the `owned_by` field
    names the machine so a caller can tell local from remote.
    """
    data = [
        {
            "id": name,
            "object": "model",
            "owned_by": "freetoken",
            "created": int(inst.started_at or time.time()),
        }
        for inst in manager.running()
        for name in sorted({n for n in (inst.model_id, inst.served_name) if n})
    ]
    created = int(time.time())
    from . import federation

    for m in federation.remote_loaded():
        if m["ready"]:
            data.append({"id": m["model"], "object": "model", "owned_by": m["remote"]["node"], "created": created})
    from . import external

    for m in external.loaded():
        data.append({"id": m["model"], "object": "model", "owned_by": m["external"]["provider"], "created": created})
    try:
        backends = [b for b in remotes.load_backends() if b.enabled]
    except ValueError:
        backends = []
    for backend in backends:
        for name in await remotes.remote_models(client(), backend):
            data.append(
                {"id": name, "object": "model", "owned_by": backend.name, "created": created}
            )
    return {"object": "list", "data": data}


def _remote_relay(backend: remotes.RemoteBackend, path: str, payload: dict):
    """Stream an SSE response from a remote backend, adding its auth header."""
    model = str(payload.get("model") or "")
    payload = tokens.with_stream_usage(payload)

    async def relay():
        timeout = httpx.Timeout(None, connect=8.0)
        try:
            async with client().stream(
                "POST",
                f"{backend.base}/v1/{path}",
                json=payload,
                headers=backend.headers,
                timeout=timeout,
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", "replace")
                    yield f'data: {{"error": {body!r}}}\n\n'.encode()
                    return
                async for chunk in tokens.metered(
                        response.aiter_raw(), lambda u: tokens.record_usage("network", model, u)):
                    yield chunk
        except httpx.HTTPError as exc:
            yield f'data: {{"error": "backend {backend.name} failed: {exc}"}}\n\n'.encode()

    return StreamingResponse(
        relay(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@openai_router.post("/{path:path}")
async def openai_passthrough(path: str, request: Request) -> Any:
    if path not in _PASSTHROUGH_POST:
        raise HTTPException(status_code=404, detail=f"unsupported path /v1/{path}")

    try:
        payload = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="body must be JSON") from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    requested = payload.get("model")

    # `id@groq` / `id@openrouter`: an enabled external model, metered against the daily limit.
    from . import external, federation

    if external.is_external(str(requested) if requested else None):
        if path != "chat/completions":
            raise HTTPException(status_code=400, detail="external models serve chat/completions only")
        return await external.complete(client(), str(requested), payload,
                                       purpose=request.headers.get("x-freeswarm-purpose", "swarm"))

    # `model@computer`: a model a paired computer shares over the LAN federation.
    fed = federation.route(str(requested)) if requested else None
    if fed is not None:
        return await federation.relay(fed[0], fed[1], path, payload)

    # Route by model name: a name this machine does not serve may belong to a remote
    # backend. Checked before _require_ready_model so a request for a remote model is not
    # rejected merely because the local engine is stopped.
    backend = None
    if requested:
        try:
            backend = await remotes.resolve(client(), str(requested), await _local_model_names())
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    if backend is not None:
        if payload.get("stream"):
            return _remote_relay(backend, path, payload)
        try:
            r = await client().post(
                f"{backend.base}/v1/{path}",
                json=payload,
                headers=backend.headers,
                timeout=600.0,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502, detail=f"backend {backend.name} unreachable: {exc}"
            ) from exc
        data = r.json()
        if r.status_code == 200:
            tokens.record_usage("network", str(requested), tokens.usage_from_body(data))
        return JSONResponse(data, status_code=r.status_code)

    # The engine requires the name IT knows; clients may use the catalog id, and with
    # several engines resident the name also decides WHICH engine answers.
    inst, served = await _require_ready_model(requested)
    payload["model"] = served

    url = f"http://{settings.engine_host}:{inst.port}/v1/{path}"
    if payload.get("stream"):
        return _sse_relay(url, payload)
    try:
        r = await client().post(url, json=payload, timeout=600.0)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"engine unreachable: {exc}") from exc
    return JSONResponse(r.json(), status_code=r.status_code)


app.include_router(openai_router)


@app.get("/api/mcp/oauth/callback", include_in_schema=False)
async def mcp_oauth_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
):
    """Loopback redirect target for the MCP OAuth flow.

    Deliberately NOT behind `require_user`: the authorization server redirects a plain
    browser here, and that redirect cannot carry our bearer token. The protection is the
    `state` parameter -- 24 bytes of urlsafe randomness minted per flow and matched here,
    which is precisely the CSRF defence OAuth specifies. An unknown or replayed state is
    rejected, so this endpoint cannot be driven by anyone who did not start the flow.
    """
    from fastapi.responses import HTMLResponse

    def page(title: str, detail: str, ok: bool) -> HTMLResponse:
        colour = "#4ade80" if ok else "#f85149"
        return HTMLResponse(
            "<!doctype html><meta charset='utf-8'>"
            "<title>FreeSwarm - MCP authorisation</title>"
            "<body style=\"margin:0;display:grid;place-items:center;height:100vh;"
            "background:#0b0d10;color:#e8eaed;"
            "font-family:ui-sans-serif,system-ui,'Segoe UI',sans-serif\">"
            "<div style='text-align:center;max-width:30rem;padding:2rem'>"
            f"<div style='font-size:1.25rem;color:{colour};margin-bottom:.5rem'>{title}</div>"
            f"<div style='font-size:.9rem;color:#9aa4b2;line-height:1.6'>{detail}</div>"
            "<div style='font-size:.8rem;color:#6b7583;margin-top:1.5rem'>"
            "You can close this tab and return to the console.</div>"
            "</div></body>",
            status_code=200 if ok else 400,
        )

    if error:
        if state:
            mcp_oauth.finish_flow(state, None, error_description or error)
        return page("Authorisation declined", error_description or error, False)

    if not state or not code:
        return page("Invalid callback", "The redirect was missing its code or state.", False)

    if not mcp_oauth.finish_flow(state, code):
        # Unknown state: expired (5 min), already used, or not started by us.
        return page(
            "Unrecognised authorisation",
            "This request did not match a pending authorisation. It may have expired - "
            "start the connection again from the Connectors page.",
            False,
        )

    return page("Connected", "The MCP server has been authorised.", True)


@app.get("/api/health")
async def health() -> dict:
    """Liveness of the control plane itself (not the engine). Never authenticated, so a
    monitor can poll it without a credential."""
    return {"status": "ok", "engine_state": manager.primary().state, **version_info()}
