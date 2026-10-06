"""Data sources of a task server (e.g. the GEX bars' Source NULL / Source 2): the project setting,
the agents' calls pinned to the objective's own source, and a candidate's replay on another one."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from app import objectives as O
from app import task_objectives as T


def test_the_project_keeps_a_chosen_source_per_task(tmp_path, monkeypatch):
    from app import projects

    monkeypatch.setattr(projects, "PROJECTS_PATH", tmp_path / "projects.json")
    monkeypatch.setattr(projects, "DEFAULT_ROOT", tmp_path / "projects")
    p = projects.create("Gex2", task_server="gex")
    projects.update(p["id"], task_options={"gex_intraday": {"source": "2"}})
    projects.update(p["id"], task_options={"gex_intraday": {"target": "Close"}})          # merged, not replaced
    assert projects.get(p["id"])["task_options"] == {"gex_intraday": {"source": "2", "target": "Close"}}
    projects.update(p["id"], task_options={"gex_intraday": {"source": ""}})               # "" clears it
    assert projects.get(p["id"])["task_options"] == {"gex_intraday": {"target": "Close"}}


def test_valuing_calls_carry_the_objectives_source_and_only_when_it_has_one():
    obj = {"metric": {"kind": "task", "target": "Close", "source": "2"}}
    assert T._choices(obj) == {"target": "Close", "source": "2"}
    assert T._choices({"metric": {"kind": "task", "target": "Close"}}) == {"target": "Close"}


def test_agents_read_the_rows_of_their_objectives_own_source(monkeypatch):
    """Another source's in-sample rows can span the objective's holdout dates (the same SPY prices),
    so the runner sets `source` to the objective's own whatever the agent sent -- and strips it on
    an objective without one, where the server's default is its source."""
    path = Path(__file__).resolve().parents[1] / "swarm_runner.py"
    spec = importlib.util.spec_from_file_location("swarm_runner_source_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    sent = []

    def request(base, path, payload=None, **kw):
        if path.startswith("/api/mcp/call"):
            sent.append(payload["arguments"])
            return {"is_error": False, "content": "{}"}
        return {"files": [], "docs": []}

    props = {"type": "object", "properties": {"task": {"type": "string"}, "sql": {"type": "string"},
                                              "source": {"type": "string"}}}
    monkeypatch.setattr(runner, "request", request)
    monkeypatch.setattr(runner, "_mcp_tools", lambda pid: [{"function": {"name": "gex__task_query", "parameters": props}}])

    def world(metric):
        return runner.ObjectiveWorld({"id": "p1", "name": "P"}, "me", [], [],
                                     {"id": "o1", "metric": {"kind": "task", "task_server": "gex", "task": "t", **metric}},
                                     on_submit=lambda *a, **kw: None)

    world({"source": "2"}).call("gex__task_query", {"sql": "SELECT 1"})
    world({"source": "2"}).call("gex__task_query", {"sql": "SELECT 1", "source": "original"})
    world({}).call("gex__task_query", {"sql": "SELECT 1", "source": "2"})
    assert sent == [{"sql": "SELECT 1", "source": "2", "task": "t"},
                    {"sql": "SELECT 1", "source": "2", "task": "t"},
                    {"sql": "SELECT 1", "task": "t"}]


def test_a_replay_is_compared_with_the_scored_run_period_by_period():
    obj = {"split_date": "2024-07-19", "metric": {"task_info": {"shape": {"last": "2024-12-17T21:01:50"}}}}
    scored = [["2024-07-01", 0.01], ["2024-07-02", -0.005], ["2024-08-01", 0.02], ["2024-12-17", 0.0]]
    replay = scored + [["2025-01-02", -0.01], ["2025-01-03", 0.03]]
    c = O._source_comparison(obj, scored, replay, additive=False)
    by = {p["name"]: p for p in c["periods"]}
    assert list(by) == ["built on", "objective holdout", "never seen", "all"]
    assert by["built on"]["replay"]["days"] == 2 and by["objective holdout"]["replay"]["days"] == 2
    assert by["never seen"]["scored"] is None and by["never seen"]["replay"]["days"] == 2
    assert abs(by["never seen"]["replay"]["total_return"] - (0.99 * 1.03 - 1)) < 1e-9
    assert by["all"]["replay"]["max_drawdown"] < 0 and c["shared_days"] == 4
    add = O._source_comparison(obj, scored, replay, additive=True)
    assert abs([p for p in add["periods"] if p["name"] == "all"][0]["replay"]["total_return"] - 0.045) < 1e-9
