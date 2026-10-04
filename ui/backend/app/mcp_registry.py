"""MCP connector registry -- lets local models actually *do* things.

FreeToken can emit tool calls (`--tool-call-parser`), and MCP is the standard way to expose
tools to a model. This module bridges the two: it connects to MCP servers, converts their
tool schemas into the OpenAI function-tool shape the engine's parsers already understand,
and executes the calls a model asks for.

**Registration is file-based on purpose.** An MCP stdio server is an arbitrary command line;
letting a *network* endpoint define one would make this control plane a remote-code-execution
service. So servers are declared in `ui/backend/mcp_servers.json`, which only someone with
filesystem access can write, and the HTTP API can enumerate/connect/call but never define.
That is the single most important security property in this file.

Config format (`mcp_servers.json`):

    {
      "servers": [
        {"name": "files", "transport": "stdio",
         "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\\\work"],
         "env": {}, "enabled": true},
        {"name": "internal", "transport": "http",
         "url": "http://127.0.0.1:9000/mcp", "enabled": true}
      ]
    }

Sessions are opened per call rather than held open. MCP sessions are stateful and the SDK
binds them to the async context that created them, so caching one across FastAPI requests
invites cross-task cancellation bugs; a fresh stdio process per call is slower but cannot
leak state between agents.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger("freetoken.mcp")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "mcp_servers.json"

# A single MCP call should not be able to wedge a request handler forever.
CALL_TIMEOUT_S = 120.0
CONNECT_TIMEOUT_S = 60.0


class ServerSpec(BaseModel):
    name: str
    transport: Literal["stdio", "http", "sse"] = "stdio"
    # stdio
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    # http / sse
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    # What the server is: "tool" -- a connector agents and chat call for tools (the default) --
    # or "task" -- a data/action MCP (mcp/README.md) that serves a project's data, manages the
    # actions strategies take and values them. Only "task" servers can be a project's
    # data/action MCP or score an objective.
    kind: Literal["tool", "task"] = "tool"
    # Remote servers usually sit behind OAuth 2.1. When true, the session attaches a
    # bearer token obtained through the browser flow in mcp_oauth.py and refreshes it
    # automatically. Ignored for stdio, which has no HTTP layer to authorise.
    oauth: bool = False
    # Pre-registered OAuth client credentials. The MCP spec prefers Dynamic Client
    # Registration (RFC 7591), and the SDK tries it first -- but most real
    # providers do not implement it. GitHub, for instance, answers the registration
    # endpoint with a 404 and expects an OAuth App you created by hand. Set these to skip
    # registration entirely.
    client_id: str | None = None
    client_secret: str | None = None
    # Read the secret from an environment variable instead of storing it in this file.
    # Preferred: mcp_servers.json is ordinary config and not ACL-restricted.
    client_secret_env: str | None = None
    # ...or from a file holding only the secret (owner-only), as a task server's
    # make_oauth_secrets.py writes to ui/backend/auth/mcp_clients/<name>.secret.
    client_secret_file: str | None = None
    # Scopes to request. Providers that do not advertise defaults need this.
    scopes: list[str] = Field(default_factory=list)

    def resolved_client_secret(self) -> str | None:
        if self.client_secret_env:
            return os.environ.get(self.client_secret_env) or None
        if self.client_secret_file:
            try:
                return Path(self.client_secret_file).read_text(encoding="utf-8").strip() or None
            except OSError:
                return None
        return self.client_secret

    def validate_runnable(self) -> None:
        if self.transport == "stdio":
            if not self.command:
                raise ValueError(f"{self.name}: stdio transport needs a 'command'")
            if self.oauth:
                raise ValueError(f"{self.name}: oauth applies to http/sse, not stdio")
        elif not self.url:
            raise ValueError(f"{self.name}: {self.transport} transport needs a 'url'")


def _normalise_entries(raw: dict) -> list[dict]:
    """Accept both config shapes.

    Native (ordered, explicit):      {"servers": [{"name": "x", ...}, ...]}
    Claude-Desktop style (common):   {"mcpServers": {"x": {"command": ..., "args": [...]}}}

    The second form is what tool docs and READMEs hand you, so pasting one straight in
    should work. A server listed under `mcpServers` has no `enabled` key in that
    convention -- being listed *is* being enabled -- so it defaults to true there, while
    the native form keeps its explicit flag.
    """
    if isinstance(raw.get("servers"), list):
        return [e for e in raw["servers"] if isinstance(e, dict)]

    mapping = raw.get("mcpServers")
    if isinstance(mapping, dict):
        entries: list[dict] = []
        for name, spec in mapping.items():
            if not isinstance(spec, dict):
                continue
            entry = {"name": name, **spec}
            entry.setdefault("enabled", True)
            # Claude Desktop infers stdio from the presence of `command`.
            if "transport" not in entry:
                entry["transport"] = "stdio" if entry.get("command") else "http"
            entries.append(entry)
        return entries
    return []


def load_config() -> list[ServerSpec]:
    """Read the on-disk registry. A malformed file is reported, not silently ignored."""
    if not CONFIG_PATH.is_file():
        return []
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.error("mcp_servers.json is unreadable: %s", exc)
        raise ValueError(f"mcp_servers.json is invalid: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("mcp_servers.json must be a JSON object")

    out: list[ServerSpec] = []
    for entry in _normalise_entries(raw):
        # Keys the Claude Desktop format carries that we do not model (e.g. "type",
        # "disabled", "autoApprove") would otherwise trip pydantic; drop them rather than
        # rejecting an otherwise valid server.
        known = {k: v for k, v in entry.items() if k in ServerSpec.model_fields}
        try:
            out.append(ServerSpec(**known))
        except Exception as exc:  # noqa: BLE001 - one bad entry must not hide the rest
            logger.error("skipping malformed MCP server entry %r: %s", entry, exc)
    return out


def write_example_config() -> Path:
    """Create a commented starter file if none exists, so the UI can point at something."""
    if CONFIG_PATH.is_file():
        return CONFIG_PATH
    example = {
        "_comment": (
            "MCP servers available to agents. Edited on disk only -- the HTTP API "
            "deliberately cannot add entries, because a stdio server is an arbitrary "
            "command line. Set enabled:true to activate."
        ),
        "servers": [
            {
                "name": "filesystem",
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", str(Path.home())],
                "env": {},
                "enabled": False,
            },
            {
                "name": "example-http",
                "transport": "http",
                "url": "http://127.0.0.1:9000/mcp",
                "headers": {},
                "enabled": False,
            },
        ],
    }
    CONFIG_PATH.write_text(json.dumps(example, indent=2), encoding="utf-8")
    return CONFIG_PATH


def _auth_for(spec: ServerSpec, provider: Any | None):
    """The httpx auth object for this server, if any.

    `provider` is passed in rather than built here so the interactive connect flow can
    supply one wired to a browser round-trip, while ordinary calls get the silent,
    refresh-only variant.
    """
    if not spec.oauth or spec.transport == "stdio":
        return None
    if provider is not None:
        return provider
    from . import mcp_oauth
    from .config import settings

    redirect = f"http://{settings.host}:{settings.port}{mcp_oauth.DEFAULT_REDIRECT_PATH}"
    return mcp_oauth.build_stored_provider(spec.name, spec.url or "", redirect, spec)


@contextlib.asynccontextmanager
async def _session(spec: ServerSpec, auth_provider: Any | None = None):
    """Open an MCP session for `spec`, initialise it, and tear it down on exit."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    spec.validate_runnable()

    if spec.transport == "stdio":
        command = spec.command or ""
        # On Windows an npx/npm entry point is a .cmd shim that CreateProcess will not run
        # unless it is resolved to its full path first.
        resolved = shutil.which(command)
        if resolved is None:
            raise ValueError(f"{spec.name}: command not found on PATH: {command!r}")
        params = StdioServerParameters(
            command=resolved,
            args=list(spec.args),
            env={**os.environ, **spec.env},
            cwd=spec.cwd,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), CONNECT_TIMEOUT_S)
                yield session
        return

    auth = _auth_for(spec, auth_provider)

    if spec.transport == "sse":
        from mcp.client.sse import sse_client

        async with sse_client(spec.url or "", headers=spec.headers or None, auth=auth) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), CONNECT_TIMEOUT_S)
                yield session
        return

    # mcp 2.x: streamable_http_client takes a prepared client, not headers/auth. The
    # helper applies the transports' own timeouts (300s read, because a server may hold a
    # response stream open), which a hand-rolled AsyncClient would not.
    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

    http_client = create_mcp_http_client(headers=spec.headers or None, auth=auth)
    async with http_client, streamable_http_client(
        spec.url or "", http_client=http_client
    ) as streams:
        # The streamable-HTTP client yields (read, write) plus, in some SDK versions, a
        # session-id callback. Take the first two either way.
        read, write = streams[0], streams[1]
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), CONNECT_TIMEOUT_S)
            yield session


# Task servers (docs/task-servers.md) expose `harness_*` tools that only the control plane may
# call: they export rows including the holdout and score actions on it. The HTTP routes agents
# use (/api/mcp/tools, /api/mcp/call) hide and refuse them; the harness calls call_tool directly.
HARNESS_PREFIX = "harness_"


TASK_TOOLS = {"task_list", "task_describe", "harness_export_rows", "harness_evaluate"}


def _write_config(servers: list[dict]) -> None:
    """Rewrite mcp_servers.json with `servers`, keeping its other keys (the comment)."""
    raw: dict = {}
    if CONFIG_PATH.is_file():
        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except ValueError:
            raw = {}
    raw.pop("mcpServers", None)
    raw["servers"] = servers
    tmp = CONFIG_PATH.with_name(CONFIG_PATH.name + ".tmp")
    tmp.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    tmp.replace(CONFIG_PATH)


def _raw_servers() -> list[dict]:
    if not CONFIG_PATH.is_file():
        return []
    return _normalise_entries(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))


NAME_RX = r"^[a-z0-9][a-z0-9_-]{0,40}$"


async def register(name: str, source: str, kind: str = "auto", client_id: str | None = None,
                   client_secret: str | None = None, trust_code: bool = False) -> dict:
    """Register an MCP server from the console: a URL (http transport; OAuth detected from its
    discovery metadata) or a local Python script / a folder holding server.py (stdio, run with
    the control plane's own interpreter). `kind` "auto" probes for the task interface's tools.

    A local script is code the control plane will RUN, so it needs `trust_code` -- and only a
    Python file is accepted, never a free-form command line. Returns the stored entry and what
    the probe found."""
    import re
    import sys

    import httpx

    name = (name or "").strip().lower()
    if not re.match(NAME_RX, name):
        raise ValueError("name: lowercase letters, digits, '-' or '_' (up to 41 characters)")
    servers = _raw_servers()
    if any(s.get("name") == name for s in servers):
        raise ValueError(f"an MCP server named {name!r} is already registered")
    if kind not in ("auto", "task", "tool"):
        raise ValueError("kind must be auto, task or tool")
    src = (source or "").strip().strip('"')
    entry: dict[str, Any] = {"name": name, "enabled": True}
    if re.match(r"^https?://", src, re.I):
        from urllib.parse import urlparse

        u = urlparse(src)
        if not u.netloc:
            raise ValueError("that URL has no host")
        entry.update(transport="http", url=src)
        # OAuth? The MCP authorization spec publishes protected-resource metadata; a 401 with a
        # WWW-Authenticate header says the same.
        oauth = False
        try:
            async with httpx.AsyncClient(timeout=8) as c:
                meta = await c.get(f"{u.scheme}://{u.netloc}/.well-known/oauth-protected-resource{u.path}")
                if meta.status_code == 200:
                    oauth = True
                else:
                    r = await c.post(src, json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
                    oauth = r.status_code == 401
        except httpx.HTTPError as exc:
            raise ValueError(f"could not reach {src}: {exc}") from None
        if oauth:
            entry.update(oauth=True, scopes=["tasks"] if kind == "task" else [])
            if client_id:
                entry["client_id"] = client_id.strip()
                if client_secret:
                    from . import mcp_oauth

                    path = CONFIG_PATH.parent / "auth" / "mcp_clients" / f"{name}.secret"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(client_secret.strip(), encoding="utf-8")
                    mcp_oauth._restrict(path)
                    entry["client_secret_file"] = str(path)
    else:
        p = Path(src).expanduser()
        if p.is_dir():
            p = p / "server.py"
        if not p.is_file():
            raise ValueError(f"no such file: {p} (give a URL, a Python server script, or a folder holding server.py)")
        if p.suffix.lower() != ".py":
            raise ValueError("a local MCP server must be a Python script (.py) -- it is run with the control plane's Python")
        if not trust_code:
            raise ValueError("a local server is code the control plane will run: confirm you trust it (trust_code)")
        entry.update(transport="stdio", command=sys.executable, args=[str(p.resolve())], cwd=str(p.resolve().parent),
                     env={})
    spec = ServerSpec.model_validate({**entry, "kind": "tool" if kind == "auto" else kind})
    spec.validate_runnable()
    found: dict[str, Any] = {"ok": None, "tools": 0, "task_server": None, "error": None}
    if not (spec.oauth and spec.transport != "stdio"):
        try:
            tools = await asyncio.wait_for(list_tools(spec), 60)
            names = {t["function"]["name"].split("__", 1)[-1] for t in tools}
            found.update(ok=True, tools=len(tools), task_server=TASK_TOOLS <= names)
        except Exception as exc:  # noqa: BLE001 -- reported, the operator decides
            found.update(ok=False, error=describe_exception(exc))
    if kind == "auto":
        if found["task_server"] is None:
            why = found["error"] or "it needs sign-in before it can be probed"
            raise ValueError(f"could not probe the server to tell its kind ({why}) -- choose "
                             "'data/action MCP' or 'tool connector'")
        kind = "task" if found["task_server"] else "tool"
    entry["kind"] = kind
    _write_config(servers + [entry])
    return {"server": {k: v for k, v in entry.items() if k != "env"}, "probe": found}


def unregister(name: str) -> bool:
    servers = _raw_servers()
    kept = [s for s in servers if s.get("name") != name]
    if len(kept) == len(servers):
        return False
    _write_config(kept)
    return True


def task_server_names() -> set[str]:
    """Registered, enabled data/action MCPs (kind "task")."""
    return {s.name for s in load_config() if s.enabled and s.kind == "task"}


def harness_only(qualified: str) -> bool:
    """Whether a qualified tool name (server__tool) is reserved for the control plane."""
    return qualified.split("__", 1)[-1].lower().startswith(HARNESS_PREFIX)


def _field(obj: Any, name: str, legacy: str) -> Any:
    """An MCP type's field under its current name or the camelCase one of older SDKs. mcp 2.x
    renamed them (input_schema, is_error, ...): reading only the old names sent every connector
    tool to the models WITHOUT its parameters -- agents called task_describe({}) over and over
    (bug #130) -- and never saw a tool's own error flag."""
    v = getattr(obj, name, None)
    return v if v is not None else getattr(obj, legacy, None)


def _to_openai_tool(server: str, tool: Any) -> dict:
    """Convert an MCP tool descriptor into an OpenAI function-tool schema.

    The name is namespaced `server__tool` so two servers exposing `read_file` do not
    collide in a single tool list, and so `call_tool` can route back from the name the
    model emitted. `__` is used because the OpenAI tool-name grammar allows it but it is
    vanishingly rare inside a real MCP tool name.
    """
    schema = _field(tool, "input_schema", "inputSchema") or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": f"{server}__{tool.name}",
            "description": (getattr(tool, "description", "") or "")[:1024],
            "parameters": schema,
        },
    }


async def list_tools(spec: ServerSpec, auth_provider: Any | None = None) -> list[dict]:
    async with _session(spec, auth_provider) as session:
        result = await asyncio.wait_for(session.list_tools(), CONNECT_TIMEOUT_S)
        return [_to_openai_tool(spec.name, t) for t in result.tools]


def describe_exception(exc: BaseException, depth: int = 0) -> str:
    """Flatten an exception chain into one readable line.

    The MCP SDK runs its transport in an anyio TaskGroup, so a server that fails to start
    surfaces as `ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)` -- which
    says nothing at all. Walking `.exceptions` and `.__cause__` recovers the real message
    (a missing command, an import error inside the server, a protocol timeout).
    """
    if depth > 6:
        return type(exc).__name__
    subs = list(getattr(exc, "exceptions", None) or [])
    if subs:
        inner = "; ".join(describe_exception(s, depth + 1) for s in subs[:3])
        return inner or type(exc).__name__
    text = str(exc).strip()
    label = f"{type(exc).__name__}: {text}" if text else type(exc).__name__
    cause = exc.__cause__ or exc.__context__
    if not text and cause is not None:
        return f"{label} <- {describe_exception(cause, depth + 1)}"
    return label


async def probe(spec: ServerSpec, auth_provider: Any | None = None) -> dict:
    """Connect, enumerate, disconnect -- the health check behind the Connectors UI."""
    try:
        tools = await list_tools(spec, auth_provider)
        return {"name": spec.name, "ok": True, "tools": tools, "error": None}
    except asyncio.TimeoutError:
        return {
            "name": spec.name, "ok": False, "tools": [],
            "error": f"timed out after {CONNECT_TIMEOUT_S:.0f}s -- does the server start on its own?",
        }
    except Exception as exc:  # noqa: BLE001 - any failure is a connector status, not a 500
        return {"name": spec.name, "ok": False, "tools": [], "error": describe_exception(exc)}


def _flatten_content(content: Any) -> str:
    """MCP returns a list of typed content blocks; agents want text."""
    parts: list[str] = []
    for block in content or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
            continue
        data = getattr(block, "data", None)
        if data is not None:
            mime = _field(block, "mime_type", "mimeType") or "application/octet-stream"
            parts.append(f"[{mime}: {len(data)} bytes]")
            continue
        parts.append(str(block))
    return "\n".join(parts)


async def call_tool(specs: list[ServerSpec], qualified: str, arguments: dict) -> dict:
    """Execute `server__tool`. Returns a dict shaped for a tool-result message."""
    server_name, sep, tool_name = qualified.partition("__")
    if not sep:
        raise ValueError(f"tool name must be '<server>__<tool>', got {qualified!r}")

    spec = next((s for s in specs if s.name == server_name and s.enabled), None)
    if spec is None:
        raise ValueError(f"no enabled MCP server named {server_name!r}")

    async with _session(spec) as session:
        result = await asyncio.wait_for(
            session.call_tool(tool_name, arguments or {}), CALL_TIMEOUT_S
        )
    content = _flatten_content(getattr(result, "content", None))
    structured = _field(result, "structured_content", "structuredContent")
    return {
        "server": server_name,
        "tool": tool_name,
        # MCP reports tool-level failures in-band via isError rather than by raising, so a
        # model can see and recover from them; surface that faithfully. FastMCP's argument
        # validation ("Error executing tool task_describe: 1 validation error ... task Field
        # required") comes back with isError=False, so an agent calling task_describe({})
        # looked successful and repeated it 13 times an iteration (bug #130). A taskkit server
        # reports its own failures as a returned {"error": "..."} (the Provider contract), also
        # with isError=False: "ValueError: unknown columns ['TotalAbsGex']" (10-01) looked like
        # a successful sample.
        "is_error": bool(_field(result, "is_error", "isError")) or content.startswith(_FASTMCP_ERROR)
        or _error_only(structured),
        "content": content,
        "structured": structured,
    }


def _error_only(structured: Any) -> bool:
    """A structured result that is nothing but an error report: {"error": "..."}."""
    return isinstance(structured, dict) and bool(structured.get("error")) and set(structured) <= {"error", "hint", "detail"}


_FASTMCP_ERROR = "Error executing tool "
