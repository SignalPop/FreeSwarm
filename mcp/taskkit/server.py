"""Serve a task provider as an MCP server -- the FreeSwarm task interface, and nothing else.

The interface is the set of MCP tools below: their names, JSON arguments and JSON results, plus
two file formats (the rows parquet a server writes, the actions parquet it reads). The full spec
is mcp/README.md. HOW a server loads, caches, queries, manages actions and values them is its
own business -- polars, numpy, pandas, SQL, a simulator, another language -- and never shows here.

This module only maps the tools onto a *provider*: any object with methods named after the tools
(duck-typed; see `Provider`). It imports no data library. Two ways to build a provider:

* implement the methods yourself (mcp/gex does: its own polars data layer and numpy valuation), or
* use `TasksProvider([...Task objects...])` with the optional polars helper in taskkit/task.py
  (mcp/test does).

    from taskkit.server import serve
    serve(MyProvider(), name="my-tasks")      # stdio; `--http --port 8201` for streamable HTTP
"""

from __future__ import annotations

import argparse
import logging
import traceback
from typing import Any, Callable, Protocol

logger = logging.getLogger("taskkit")

INSTRUCTIONS = (
    "A FreeSwarm task server. Each task is a table of time-aligned rows (signals), a target column, "
    "an action per row that a strategy decides and the server manages, and a valuation of the result. "
    "Call task_list, then task_describe(task) to learn the rows, the target and what an action means; "
    "task_sample_rows, task_query and task_column_stats show in-sample data. Strategy code reads the rows "
    "with ft.rows() and reports ft.report_actions(series indexed by the rows' t column); the harness "
    "values it through the harness_* tools."
)

#: The tools of the interface, in the order they are registered. harness_* are the control
#: plane's alone (they touch the holdout); FreeSwarm hides and refuses them on agents' routes.
AGENT_TOOLS = ("task_list", "task_describe", "task_sample_rows", "task_column_stats", "task_query")
HARNESS_TOOLS = ("harness_export_rows", "harness_evaluate", "harness_actions", "harness_leak_scan")


class Provider(Protocol):
    """What a task server implements. Every method returns a JSON-able dict; raising an exception
    (or returning {"error": "..."}) reports a failure to the caller."""

    def task_list(self) -> dict[str, Any]: ...
    def task_describe(self, task: str, target: str | None) -> dict[str, Any]: ...
    def task_sample_rows(self, task: str, limit: int, offset: int, columns: list[str] | None) -> dict[str, Any]: ...
    def task_column_stats(self, task: str) -> dict[str, Any]: ...
    def task_query(self, task: str, sql: str, limit: int) -> dict[str, Any]: ...
    def harness_export_rows(self, task: str, path: str, until: str | None, target: str | None) -> dict[str, Any]: ...
    def harness_evaluate(self, task: str, actions_path: str, target: str | None) -> dict[str, Any]: ...
    def harness_actions(self, task: str, actions_path: str, start: str | None, end: str | None,
                        limit: int, target: str | None) -> dict[str, Any]: ...
    def harness_leak_scan(self, task: str, top: int, target: str | None) -> dict[str, Any]: ...


def build_server(provider: Any, name: str = "tasks", oauth: tuple[Any, str] | None = None):
    """An MCPServer exposing the interface's tools over `provider`. `oauth` = (the server's
    .oauth folder, its base URL) switches OAuth 2.1 on (taskkit.oauth): every /mcp request then
    needs a bearer token the control plane obtained with the operator's approval."""
    from mcp.server.mcpserver import MCPServer

    kwargs: dict[str, Any] = {}
    auth_provider = None
    if oauth is not None:
        from .oauth import auth_kwargs

        kwargs, auth_provider = auth_kwargs(oauth[0], oauth[1], name)
    mcp = MCPServer(name, instructions=INSTRUCTIONS, **kwargs)
    if auth_provider is not None:
        from .oauth import mount

        mount(mcp, auth_provider)

    def guarded(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            out = fn()
            return out if isinstance(out, dict) else {"result": out}
        except Exception as exc:  # noqa: BLE001 -- the caller gets the reason, not a dead session
            logger.error("%s", traceback.format_exc())
            return {"error": f"{type(exc).__name__}: {exc}"}

    @mcp.tool()
    def task_list() -> dict[str, Any]:
        """The tasks this server offers: name, title, target, score and size. Start here."""
        return guarded(provider.task_list)

    @mcp.tool()
    def task_describe(task: str, target: str | None = None) -> dict[str, Any]:
        """Everything about one task: description and rules for agents, the target column (and the
        `target_options` it may be switched to with `target`), the data's shape, what an action means
        and its bounds, the value function, every column with its role, the holdout and the cuts."""
        return guarded(lambda: provider.task_describe(task, target))

    @mcp.tool()
    def task_sample_rows(task: str, limit: int = 20, offset: int = 0, columns: list[str] | None = None) -> dict[str, Any]:
        """In-sample rows of a task as records (at most 500 per call; `offset` pages through them;
        `columns` limits the columns). Holdout rows are never returned."""
        return guarded(lambda: provider.task_sample_rows(task, limit, offset, columns))

    @mcp.tool()
    def task_column_stats(task: str) -> dict[str, Any]:
        """Per-column summary of a task's in-sample rows: count, missing, mean, std, min, quartiles,
        max (numeric) or distinct values (text)."""
        return guarded(lambda: provider.task_column_stats(task))

    @mcp.tool()
    def task_query(task: str, sql: str, limit: int = 200) -> dict[str, Any]:
        """A read-only SQL SELECT over the task's IN-SAMPLE rows, as the table `rows` -- e.g.
        `SELECT hour, avg(price) FROM rows GROUP BY hour ORDER BY hour`. At most `limit` rows."""
        return guarded(lambda: provider.task_query(task, sql, limit))

    @mcp.tool()
    def harness_export_rows(task: str, path: str, until: str | None = None, target: str | None = None) -> dict[str, Any]:
        """HARNESS ONLY. Write the task's rows with t < `until` (every row without it) to the
        parquet file `path`, and task.json (the description, without the cuts) beside it."""
        return guarded(lambda: provider.harness_export_rows(task, path, until, target))

    @mcp.tool()
    def harness_evaluate(task: str, actions_path: str, target: str | None = None) -> dict[str, Any]:
        """HARNESS ONLY. Manage the actions in the parquet file `actions_path` (columns t, pos) and
        value the result: segments (in_sample / holdout) with their scores, the per-period curve,
        diagnostics and in-sample notes for the agent."""
        return guarded(lambda: provider.harness_evaluate(task, actions_path, target))

    @mcp.tool()
    def harness_actions(task: str, actions_path: str, start: str | None = None, end: str | None = None,
                        limit: int = 500, target: str | None = None) -> dict[str, Any]:
        """HARNESS ONLY. The drill-down of a window (e.g. one day of the result curve): `bars` (the
        target as OHLC candles or a line), `state` (the managed state, e.g. the position) and
        `events` (what the actions did: trades, a charge schedule, ...)."""
        return guarded(lambda: provider.harness_actions(task, actions_path, start, end, limit, target))

    @mcp.tool()
    def harness_leak_scan(task: str, top: int = 15, target: str | None = None) -> dict[str, Any]:
        """HARNESS / OPERATOR ONLY. In-sample check for columns that look like they know the future:
        how each column's change correlates with the target's move over the current and the NEXT
        row. `suspects` predict the next move clearly better -- probably filed before they were known."""
        return guarded(lambda: provider.harness_leak_scan(task, top, target))

    return mcp


def serve(provider: Any, name: str = "tasks", default_port: int = 8201, argv: list[str] | None = None,
          oauth_dir: Any = None) -> None:
    """Run a task server: stdio by default, `--http [--host H] [--port P]` for streamable HTTP
    (the endpoint is http://H:P/mcp).

    Over HTTP the server requires OAuth: its secrets must exist in `oauth_dir` (made by the
    server's make_oauth_secrets.py). `--no-auth` serves without it, on a loopback host only --
    for local testing. stdio needs no OAuth: the server is a child process of the control plane
    with no network surface."""
    import sys
    from pathlib import Path

    ap = argparse.ArgumentParser(description=f"FreeSwarm task server '{name}'")
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=default_port)
    ap.add_argument("--no-auth", action="store_true", help="HTTP without OAuth (loopback hosts only; testing)")
    ap.add_argument("--public-url", help="the URL clients use, if not http://HOST:PORT (e.g. behind a TLS proxy)")
    args = ap.parse_args(argv)
    oauth = None
    if args.http:
        secrets_file = Path(oauth_dir) / "server.json" if oauth_dir else None
        if args.no_auth:
            if args.host not in ("127.0.0.1", "localhost", "::1"):
                sys.exit("--no-auth is only allowed on a loopback host; use OAuth (make_oauth_secrets.py)")
            print(f"[{name}] WARNING: serving HTTP WITHOUT OAuth on {args.host} (testing only)", file=sys.stderr)
        elif secrets_file is None or not secrets_file.is_file():
            sys.exit(f"[{name}] no OAuth secrets ({secrets_file}): run make_oauth_secrets.py next to the server first "
                     "(or --no-auth for loopback testing)")
        else:
            oauth = (Path(oauth_dir), (args.public_url or f"http://{args.host}:{args.port}").rstrip("/"))
    mcp = build_server(provider, name, oauth)
    if args.http:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        mcp.run("streamable-http", host=args.host, port=args.port)
    else:
        mcp.run()


class TasksProvider:
    """A provider over Task objects (the optional polars helper in taskkit/task.py). `tasks` is a
    callable returning the current list, so a registry can add or reload tasks without a restart."""

    def __init__(self, tasks: Callable[[], list[Any]], errors: Callable[[], list[str]] | None = None):
        self._tasks, self._errors = tasks, errors

    def _get(self, task: str):
        found = {t.name: t for t in self._tasks()}
        if task not in found:
            raise ValueError(f"no task {task!r}; this server has: {sorted(found) or 'none'}")
        return found[task]

    def task_list(self) -> dict[str, Any]:
        out = []
        for t in self._tasks():
            try:
                d = t.describe()
                out.append({k: d[k] for k in ("name", "title", "target", "score", "rows", "first", "holdout_from")}
                           | {"action": d["action"].get("kind")})
            except Exception as exc:  # noqa: BLE001
                out.append({"name": t.name, "title": t.title, "error": f"{type(exc).__name__}: {exc}"})
        return {"tasks": out, "errors": self._errors() if self._errors else []}

    def task_describe(self, task, target=None):
        return self._get(task).with_target(target).describe()

    def task_sample_rows(self, task, limit, offset, columns):
        return self._get(task).sample(limit, offset, columns)

    def task_column_stats(self, task):
        return self._get(task).column_stats()

    def task_query(self, task, sql, limit):
        return self._get(task).query(sql, limit)

    def harness_export_rows(self, task, path, until, target=None):
        return self._get(task).with_target(target).export(path, until)

    def harness_evaluate(self, task, actions_path, target=None):
        return self._get(task).with_target(target).evaluate_file(actions_path)

    def harness_actions(self, task, actions_path, start, end, limit, target=None):
        return self._get(task).with_target(target).action_log_file(actions_path, start, end, limit)

    def harness_leak_scan(self, task, top, target=None):
        return self._get(task).with_target(target).leak_scan(top)
