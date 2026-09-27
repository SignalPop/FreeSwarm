"""taskkit -- OPTIONAL helpers for writing a FreeSwarm task server (a data/action MCP).

The task interface is the MCP tools and file formats in mcp/README.md -- not this package. A
server may implement those tools any way it likes; taskkit only saves work:

* `taskkit.server`     -- maps the interface's tools onto any object with methods named after them
                          (no data library); `serve(provider)` runs it over stdio or HTTP.
* `taskkit.task`       -- `Task`, a polars base class that implements the data half of the
                          interface (rows, exports, samples, SQL, stats, leak scan) for you.
* `taskkit.table`      -- `TableTask`: a task from a JSON file, no code.
* `taskkit.evaluators` -- ready valuations: trading positions, forecasts (numpy).
* `taskkit.metrics`    -- return statistics and swing capture (numpy).

Imports are lazy, so using one part never loads the others' dependencies.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "Task": "task", "KEY": "task", "align_actions": "task", "cache_dir": "task", "read_actions": "task",
    "ts_ns": "task", "iso": "task", "keys_ns": "task", "records": "task",
    "TableTask": "table", "load_table_tasks": "table",
    "build_server": "server", "serve": "server", "TasksProvider": "server", "Provider": "server",
    "evaluators": None, "metrics": None,
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'taskkit' has no attribute {name!r}")
    mod = _EXPORTS[name]
    if mod is None:
        return importlib.import_module(f"{__name__}.{name}")
    return getattr(importlib.import_module(f"{__name__}.{mod}"), name)
