"""Projects -- the unit of isolation for a piece of work.

A project owns three things:

  * **A message board.** Sessions, messages, tasks and blackboard entries are scoped to a
    project, so two pieces of work do not read each other's coordination traffic.
  * **A set of enabled connectors.** MCP servers are declared once, globally, in
    `mcp_servers.json` (Settings). A project then chooses which of them its agents may
    use. A connector must be enabled in *both* places to be reachable -- the global flag
    is "this machine may run it at all", the project flag is "this work may use it".
  * **A data directory.** Everything that is not coordination traffic: documents, inputs,
    generated artefacts.

Registry lives in `ui/backend/projects.json`. Data directories default to
`<repo>/projects/<slug>/data` but may point anywhere, which is why every file operation
goes through `resolve_in_data_dir` -- an agent asking for `../../.ssh/id_rsa` is a
realistic failure mode, not a hypothetical one.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .config import REPO_ROOT

logger = logging.getLogger("freetoken.projects")

PROJECTS_PATH = Path(__file__).resolve().parent.parent / "projects.json"
DEFAULT_ROOT = REPO_ROOT / "projects"

_lock = threading.RLock()

_SLUG_RE = re.compile(r"[^a-z0-9-]+")

# Windows reserves these as device names; a directory called CON or AUX cannot be created.
_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")[:48]
    if not slug:
        slug = "project"
    if slug in _RESERVED:
        slug = f"{slug}-project"
    return slug


def _read() -> dict[str, Any]:
    if not PROJECTS_PATH.is_file():
        return {"projects": [], "active": None}
    try:
        data = json.loads(PROJECTS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.error("projects.json unreadable: %s", exc)
        return {"projects": [], "active": None}
    if not isinstance(data, dict):
        return {"projects": [], "active": None}
    data.setdefault("projects", [])
    data.setdefault("active", None)
    return data


def _write(data: dict[str, Any]) -> None:
    PROJECTS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def list_projects() -> list[dict]:
    with _lock:
        return list(_read()["projects"])


def get(project_id: str) -> dict | None:
    with _lock:
        return next((p for p in _read()["projects"] if p["id"] == project_id), None)


def active_id() -> str | None:
    """The project the console and unscoped agent calls operate on."""
    with _lock:
        data = _read()
        current = data.get("active")
        ids = {p["id"] for p in data["projects"]}
        if current in ids:
            return current
        # Self-heal: the active project was deleted, or this is a fresh install.
        return data["projects"][0]["id"] if data["projects"] else None


def set_active(project_id: str) -> dict:
    with _lock:
        data = _read()
        if not any(p["id"] == project_id for p in data["projects"]):
            raise ValueError(f"no project {project_id!r}")
        data["active"] = project_id
        _write(data)
        return next(p for p in data["projects"] if p["id"] == project_id)


def create(name: str, data_dir: str | None = None, connectors: list[str] | None = None) -> dict:
    name = (name or "").strip()
    if not name:
        raise ValueError("project name must not be empty")

    with _lock:
        data = _read()
        base_slug = slugify(name)
        # Slugs are directory names, so they have to be unique even when names collide.
        taken = {p["slug"] for p in data["projects"]}
        slug, n = base_slug, 2
        while slug in taken:
            slug, n = f"{base_slug}-{n}", n + 1

        if data_dir:
            path = Path(data_dir).expanduser()
        else:
            path = DEFAULT_ROOT / slug / "data"
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"cannot create data directory {path}: {exc}") from exc

        project = {
            "id": uuid.uuid4().hex,
            "name": name,
            "slug": slug,
            "data_dir": str(path.resolve()),
            # Empty means no connectors: a new project starts with no tool access rather
            # than inheriting everything, so enabling one is a deliberate act.
            "connectors": list(connectors or []),
            "created_at": time.time(),
        }
        data["projects"].append(project)
        if not data.get("active"):
            data["active"] = project["id"]
        _write(data)
        return project


ROLES = ("search", "ideas", "both")


def update(project_id: str, **fields: Any) -> dict:
    allowed = {"name", "data_dir", "connectors", "models", "sql", "swarm_enabled", "model_roles"}
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"cannot update: {', '.join(sorted(unknown))}")

    with _lock:
        data = _read()
        project = next((p for p in data["projects"] if p["id"] == project_id), None)
        if project is None:
            raise ValueError(f"no project {project_id!r}")

        if "name" in fields:
            name = (fields["name"] or "").strip()
            if not name:
                raise ValueError("project name must not be empty")
            project["name"] = name
        if "connectors" in fields:
            project["connectors"] = sorted({str(c) for c in (fields["connectors"] or [])})
        if "models" in fields:
            # Which loaded models this project's swarm may use. None = all of them; an
            # explicit list narrows it (e.g. keep the 120B for one project only).
            models = fields["models"]
            project["models"] = None if models is None else sorted({str(m) for m in models})
        if "model_roles" in fields:
            # What each model does in this project's swarm: search (write and test candidates),
            # ideas (asked for new directions when the search is stuck) or both. A model with
            # no entry follows the automatic rule (swarm_policy).
            roles = {str(k): str(v) for k, v in (fields["model_roles"] or {}).items()}
            bad = {v for v in roles.values() if v not in ROLES}
            if bad:
                raise ValueError(f"unknown role(s): {', '.join(sorted(bad))}")
            project["model_roles"] = dict(sorted(roles.items()))
        if "swarm_enabled" in fields:
            # Whether this project has a swarm at all. Absent = on, so every existing project
            # keeps the agents it already had.
            project["swarm_enabled"] = bool(fields["swarm_enabled"])
        if "sql" in fields:
            # Set only via sqlsource.provision, never free-form from a request body, and it
            # holds no secret -- the reader's password lives in auth/secrets.json.
            project["sql"] = fields["sql"]
        if "data_dir" in fields and fields["data_dir"]:
            path = Path(str(fields["data_dir"])).expanduser()
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ValueError(f"cannot use data directory {path}: {exc}") from exc
            # Existing files are NOT moved: the old directory may be shared or huge, and
            # silently relocating someone's documents is not a decision to make for them.
            project["data_dir"] = str(path.resolve())

        _write(data)
        return project


def delete(project_id: str, remove_files: bool = False) -> bool:
    with _lock:
        data = _read()
        project = next((p for p in data["projects"] if p["id"] == project_id), None)
        if project is None:
            return False
        data["projects"] = [p for p in data["projects"] if p["id"] != project_id]
        if data.get("active") == project_id:
            data["active"] = data["projects"][0]["id"] if data["projects"] else None
        _write(data)

    if remove_files:
        # Only ever delete a directory this app created under its own root. A project
        # pointed at C:\Users\me\Documents must not be removable by a button in a web UI.
        path = Path(project["data_dir"])
        try:
            inside_default = path.resolve().is_relative_to(DEFAULT_ROOT.resolve())
        except (OSError, ValueError):
            inside_default = False
        if inside_default:
            shutil.rmtree(path, ignore_errors=True)
            # The default layout is <root>/<slug>/data, so removing just the data
            # directory leaves an empty <slug>/ behind. Clean it up too -- but only when
            # it is genuinely empty, in case someone put something beside the data dir.
            parent = path.parent
            try:
                if (
                    parent != DEFAULT_ROOT.resolve()
                    and parent.is_relative_to(DEFAULT_ROOT.resolve())
                    and not any(parent.iterdir())
                ):
                    parent.rmdir()
            except OSError:
                pass
        else:
            logger.warning(
                "not deleting %s: outside the managed projects root", path
            )
    return True


def ensure_default() -> dict:
    """Guarantee at least one project exists, so nothing has to handle 'no project'."""
    with _lock:
        data = _read()
        if data["projects"]:
            return next(p for p in data["projects"] if p["id"] == active_id())
    return create("Default")


# =======================================================================================
# Data directory
# =======================================================================================
def resolve_in_data_dir(project: dict, relative: str, must_exist: bool = False) -> Path:
    """Resolve `relative` inside the project's data directory, or raise.

    The containment check runs *after* `resolve()` so that `..` segments and symlinks are
    collapsed first -- a path that merely looks contained but points elsewhere is still
    rejected. This is the boundary between "a project's documents" and "the filesystem".
    """
    root = Path(project["data_dir"]).resolve()
    candidate = (root / (relative or "")).resolve()
    if candidate != root and not candidate.is_relative_to(root):
        raise ValueError(f"path escapes the project data directory: {relative!r}")
    if must_exist and not candidate.exists():
        raise ValueError(f"no such path: {relative!r}")
    return candidate


def list_files(project: dict, subdir: str = "") -> dict:
    root = Path(project["data_dir"])
    try:
        target = resolve_in_data_dir(project, subdir)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if not target.is_dir():
        raise ValueError(f"not a directory: {subdir!r}")

    entries: list[dict] = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            try:
                stat = child.stat()
            except OSError:
                continue
            entries.append(
                {
                    "name": child.name,
                    "path": str(child.relative_to(root)).replace("\\", "/"),
                    "is_dir": child.is_dir(),
                    "size_bytes": stat.st_size if child.is_file() else 0,
                    "modified_at": stat.st_mtime,
                }
            )
    except OSError as exc:
        raise ValueError(f"cannot read {subdir!r}: {exc}") from exc

    total = 0
    try:
        total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    except OSError:
        pass
    return {
        "root": str(root),
        "subdir": subdir,
        "entries": entries,
        "total_bytes": total,
        "exists": root.is_dir(),
    }
