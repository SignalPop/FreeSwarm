"""Operator preferences that outlive a process restart.

Small, deliberately separate from `Settings` (which is immutable, env-derived startup
config). Anything here can be changed from the UI at runtime and is persisted to
`ui/backend/prefs.json`.

Today that is the GPU selection. It is a preference rather than a setting because the
useful workflow -- "take card 2 out of the pool, I want it for something else" -- is a
thing you do while the console is open, not something you restart the server to change.

Note on when it takes effect: `CUDA_VISIBLE_DEVICES` is read by the CUDA driver when a
process initialises, so a change only applies to the **next** engine launch. The API says
so in its response rather than pretending otherwise.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger("freetoken.prefs")

PREFS_PATH = Path(__file__).resolve().parent.parent / "prefs.json"

_lock = threading.Lock()
_cache: dict | None = None

# Env var is the seed for the first run only; after that prefs.json wins, so a change made
# in the UI is not silently reverted by a stale variable in a launcher script.
_ENV_DEFAULT = os.getenv("FREETOKEN_VISIBLE_DEVICES", "1,2")


def _load() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    data: dict = {}
    if PREFS_PATH.is_file():
        try:
            loaded = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError) as exc:
            logger.warning("prefs.json unreadable (%s); using defaults", exc)
    _cache = data
    return data


def _save(data: dict) -> None:
    global _cache
    _cache = data
    try:
        PREFS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        # Non-fatal: the preference still applies for this process lifetime.
        logger.error("could not persist prefs.json: %s", exc)


def get_visible_devices() -> str:
    """CUDA_VISIBLE_DEVICES for the next engine launch, in PCI-bus order.

    Empty string means "all GPUs" -- the engine then sees every card.
    """
    with _lock:
        data = _load()
        value = data.get("visible_devices")
        return _ENV_DEFAULT if value is None else str(value)


def get_model_roles() -> dict[str, str]:
    """What each model is good at, keyed by model id.

    Surfaced to other models through the model-router connector: a generalist deciding
    whether to delegate needs to know that one of its neighbours is a coding specialist,
    and nothing in a checkpoint says so. Operator-authored, because only the operator
    knows why they loaded a particular model.
    """
    with _lock:
        roles = _load().get("model_roles")
        return dict(roles) if isinstance(roles, dict) else {}


def get_auto_quarantine() -> bool:
    """Retire a library module automatically when a result built on it is disqualified.

    On by default, and it is the difference between a swarm that learns and one that does
    not. The mechanical look-ahead check runs on every candidate; without this, a signal
    module proven to leak stays `active`, every agent keeps importing it, and the night is
    spent producing better-scoring versions of the same invalid result. Turn it off only
    to triage by hand.
    """
    with _lock:
        v = _load().get("auto_quarantine")
        return True if v is None else bool(v)


def set_auto_quarantine(enabled: bool) -> bool:
    with _lock:
        data = dict(_load())
        data["auto_quarantine"] = bool(enabled)
        _save(data)
        return bool(enabled)


def set_model_role(model: str, role: str) -> dict[str, str]:
    with _lock:
        data = dict(_load())
        roles = dict(data.get("model_roles") or {})
        if role.strip():
            roles[model] = role.strip()[:500]
        else:
            roles.pop(model, None)
        data["model_roles"] = roles
        _save(data)
        return roles


def get_launch_options() -> dict[str, dict]:
    """The options each model was last launched with, keyed by model id.

    The Models page presets its form from these, so reloading a model does not mean
    re-entering the backend, context and GPU that worked last time.
    Shape: ``{model: {"options": {...}, "gpus": "1" | None}}``.
    """
    with _lock:
        saved = _load().get("launch_options")
        return dict(saved) if isinstance(saved, dict) else {}


def set_launch_options(model: str, options: dict, gpus: str | None) -> None:
    with _lock:
        data = dict(_load())
        saved = dict(data.get("launch_options") or {})
        saved[model] = {"options": dict(options), "gpus": gpus or None}
        data["launch_options"] = saved
        _save(data)


def set_visible_devices(indices: list[int] | None, all_gpus: bool = False) -> str:
    """Persist a GPU selection. `all_gpus` clears the restriction entirely."""
    with _lock:
        data = dict(_load())
        if all_gpus:
            data["visible_devices"] = ""
        else:
            # Sorted + de-duplicated so the stored value is canonical, and the engine's
            # device 0 is predictable (the lowest selected PCI index).
            clean = sorted({int(i) for i in (indices or [])})
            if any(i < 0 for i in clean):
                raise ValueError("GPU indices must be non-negative")
            data["visible_devices"] = ",".join(str(i) for i in clean)
        _save(data)
        return data["visible_devices"]
