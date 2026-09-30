"""A connector's in-band failure must reach the agent as a failure (bug #130)."""

from __future__ import annotations

import asyncio
import contextlib
import types

from app import mcp_registry as M


def _call(text: str, is_error: bool, monkeypatch) -> dict:
    result = types.SimpleNamespace(isError=is_error, content=[types.SimpleNamespace(text=text)],
                                   structuredContent=None)

    class Session:
        async def call_tool(self, name, args):
            return result

    @contextlib.asynccontextmanager
    async def session(spec):
        yield Session()

    monkeypatch.setattr(M, "_session", session)
    spec = types.SimpleNamespace(name="gex", enabled=True)
    return asyncio.run(M.call_tool([spec], "gex__task_describe", {}))


def test_fastmcp_argument_errors_are_errors_even_when_the_server_says_they_are_not(monkeypatch):
    """Muse-Glimmer sent task_describe({}) 13 times an iteration: FastMCP answered 'Error executing
    tool ... Field required' with isError=False, so the runner's repeat guard never saw a failure."""
    out = _call("Error executing tool task_describe: 1 validation error for task_describeArguments\ntask\n"
                "  Field required", False, monkeypatch)
    assert out["is_error"] is True
    assert _call('{"name": "gex_intraday"}', False, monkeypatch)["is_error"] is False
    assert _call("anything", True, monkeypatch)["is_error"] is True


def test_the_runner_hands_a_connector_error_to_the_agent_as_a_failed_call(monkeypatch):
    """Only a result with "error" counts as failed in the runner: that is what arms the
    identical-repeat guard which ends an iteration stuck re-sending the same bad call."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "swarm_runner.py"
    spec = importlib.util.spec_from_file_location("swarm_runner_mcp_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    replies = {"gex__task_describe": {"server": "gex", "tool": "task_describe", "is_error": True,
                                      "content": "Error executing tool task_describe: task Field required"},
               "gex__task_list": {"server": "gex", "tool": "task_list", "is_error": False, "content": "{}"}}

    def request(base, path, payload=None, **kw):
        if path.startswith("/api/mcp/call"):
            return replies[payload["tool"]]
        return {"files": [], "docs": []}

    monkeypatch.setattr(runner, "request", request)
    monkeypatch.setattr(runner, "_mcp_tools", lambda pid: [{"function": {"name": n}} for n in replies])
    world = runner.ProjectWorld({"id": "p1", "name": "P"}, "me", [], [])
    assert world.call("gex__task_describe", {}) == {"error": "Error executing tool task_describe: task Field required"}
    assert world.call("gex__task_list", {})["content"] == "{}"


def test_tool_schemas_and_error_flags_are_read_under_the_mcp_2_names():
    """mcp 2.2 renamed inputSchema -> input_schema and isError -> is_error: every connector tool
    reached the models with an empty parameter list."""
    schema = {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]}
    new = types.SimpleNamespace(name="task_describe", description="d", input_schema=schema)
    old = types.SimpleNamespace(name="task_describe", description="d", inputSchema=schema)
    assert M._to_openai_tool("gex", new)["function"]["parameters"] == schema
    assert M._to_openai_tool("gex", old)["function"]["parameters"] == schema
