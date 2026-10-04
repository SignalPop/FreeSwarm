"""Test task server: tasks defined by JSON files -- any table becomes a swarm problem, no code.

Serves every *.json task in this folder (and in any folder listed in FREESWARM_TASK_DIRS,
';'-separated) with the optional taskkit helpers: TableTask loads and as-of joins the sources
with polars, and the ready "trading" / "forecast" valuations score the actions. The format is in
mcp/taskkit/table.py; bike_rentals.json is the example.

Folders are rescanned on every call: a new or edited task file is picked up without a restart.

Register (launched on demand over stdio):
    {"name": "test-tables", "transport": "stdio",
     "command": "<repo>/.venv/Scripts/python.exe",
     "args": ["<repo>/mcp/test/tables/server.py"], "enabled": true}
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))                  # mcp/, for taskkit

from taskkit.server import TasksProvider, serve  # noqa: E402
from taskkit.table import TableTask, load_table_tasks  # noqa: E402


def folders() -> list[Path]:
    out = [HERE]
    for extra in (os.environ.get("FREESWARM_TASK_DIRS") or "").split(";"):
        if extra.strip() and Path(extra.strip()).is_dir():
            out.append(Path(extra.strip()))
    return out


_lock = threading.Lock()
_cache: dict[str, tuple[str, TableTask]] = {}   # name -> (fingerprint, task holding its rows)
_errors: list[str] = []


def tasks() -> list[TableTask]:
    """The current tasks, reusing a task object (and its rows) while its file is unchanged."""
    os.environ.setdefault("TASKKIT_CACHE", str(HERE / ".cache"))
    fresh, errors = load_table_tasks(folders())
    with _lock:
        out = []
        for t in fresh:
            fp = t.version()
            kept = _cache.get(t.name)
            if kept and kept[0] == fp:
                out.append(kept[1])
            else:
                _cache[t.name] = (fp, t)
                out.append(t)
        for gone in set(_cache) - {t.name for t in fresh}:
            del _cache[gone]
        _errors[:] = errors
    return out


if __name__ == "__main__":
    serve(TasksProvider(tasks, errors=lambda: list(_errors)), name="test-tables", default_port=8522,
          oauth_dir=HERE / ".oauth")
