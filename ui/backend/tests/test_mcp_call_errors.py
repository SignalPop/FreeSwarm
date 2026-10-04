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


def test_a_task_objective_fills_in_its_own_task_for_its_task_server(monkeypatch):
    """Qwen sent gex__task_sample_rows without `task` (10-01 12:09, "task: Field required"): an
    objective iteration has exactly one task, so the runner supplies it -- and only to tools of the
    objective's task server that take one, never over a task the agent named."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "swarm_runner.py"
    spec = importlib.util.spec_from_file_location("swarm_runner_task_arg_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    sent = []

    def request(base, path, payload=None, **kw):
        if path.startswith("/api/mcp/call"):
            sent.append(payload)
            return {"is_error": False, "content": "{}"}
        return {"files": [], "docs": []}

    task_props = {"type": "object", "properties": {"task": {"type": "string"}, "limit": {"type": "integer"}}}
    tools = [{"function": {"name": "gex__task_sample_rows", "parameters": task_props}},
             {"function": {"name": "gex__task_list", "parameters": {"type": "object", "properties": {}}}},
             {"function": {"name": "other__task_sample_rows", "parameters": task_props}}]
    monkeypatch.setattr(runner, "request", request)
    monkeypatch.setattr(runner, "_mcp_tools", lambda pid: tools)
    world = runner.ObjectiveWorld(
        {"id": "p1", "name": "P"}, "me", [], [],
        {"id": "o1", "metric": {"kind": "task", "task_server": "gex", "task": "gex_intraday"}},
        on_submit=lambda *a, **kw: None)
    world.call("gex__task_sample_rows", {"limit": 10})
    world.call("gex__task_sample_rows", {"task": "other_task"})
    world.call("gex__task_list", {})
    world.call("other__task_sample_rows", {})
    assert [p["arguments"] for p in sent] == [{"limit": 10, "task": "gex_intraday"}, {"task": "other_task"}, {}, {}]
    sent.clear()
    # Without the server prefix (Qwen 10-01 12:53, "task_describe"): the one tool of that name is meant.
    world.call("task_list", {})
    assert [(p["tool"], p["arguments"]) for p in sent] == [("gex__task_list", {})]
    out = world.call("task_sample_rows", {})                        # two servers have it: still refused
    assert "unknown tool 'task_sample_rows'" in out["error"] and len(sent) == 1


def test_ask_model_falls_back_to_a_free_model_when_the_external_budget_is_spent(monkeypatch):
    """Qwen asked qwen3.8@groq at 14:09 and got the 429 'spending limit' refusal: a free peer answers."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "swarm_runner.py"
    spec = importlib.util.spec_from_file_location("swarm_runner_ask_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    asked = []

    def request(base, path, payload=None, **kw):
        if path == "/v1/chat/completions":
            asked.append(payload["model"])
            if payload["model"].endswith("@groq"):
                raise RuntimeError("/v1/chat/completions -> 429: today's external-model spending limit for search is reached")
            return {"choices": [{"message": {"content": "an answer"}, "finish_reason": "stop"}]}
        return {"files": [], "docs": []}

    monkeypatch.setattr(runner, "request", request)
    monkeypatch.setattr(runner, "_mcp_tools", lambda pid: [])
    monkeypatch.setattr(runner, "permitted", lambda project, m: True)
    monkeypatch.setattr(runner, "is_external", lambda m: m.endswith("@groq"))
    world = runner.ProjectWorld({"id": "p1", "name": "P"}, "me", ["me", "qwen/qwen3.8-27b@groq", "Muse"], [])
    out = world.call("ask_model", {"model": "qwen/qwen3.8-27b@groq", "prompt": "q"})
    assert out["answer"] == "an answer" and out["model"] == "Muse" and "out of today's external budget" in out["note"]
    assert asked == ["qwen/qwen3.8-27b@groq", "Muse"]


def test_a_tool_name_with_look_alike_letters_is_the_latin_one():
    """Qwen called "decі_plot" (Cyrillic і) twice on 10-01 15:10 and got 'unknown tool'."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "swarm_runner.py"
    spec = importlib.util.spec_from_file_location("swarm_runner_lookalike_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner._plain_name("dec\u0456_plot") == "deci_plot"
    assert runner._plain_name("\uff44\uff45\uff43\uff49_plot") == "deci_plot"          # full-width letters
    assert runner._plain_name(" run_python ") == "run_python"
    assert runner._plain_name("\u65e5\u672c") == "\u65e5\u672c"                         # not a look-alike: unchanged
