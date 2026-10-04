"""Recurring failed tool calls of the swarm's agents (work archive + #errors board, 09-20..10-01),
each made to work when the intent is plain, or answered with the fix when it is not."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

RUNNER = Path(__file__).resolve().parents[1] / "swarm_runner.py"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("swarm_runner_tool_call_fixes", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _objective_world(runner, monkeypatch, handler, mcp=(), metric=None):
    """An ObjectiveWorld whose control-plane calls go to `handler(path, payload)`."""
    def request(base, path, payload=None, **kw):
        out = handler(path, payload)
        if out is None:
            return {"files": [], "docs": [], "modules": [], "candidates": []}
        return out

    monkeypatch.setattr(runner, "request", request)
    monkeypatch.setattr(runner, "_mcp_tools", lambda pid: list(mcp))
    return runner.ObjectiveWorld({"id": "p1", "name": "P"}, "me", [], [],
                                 {"id": "o1", "metric": metric or {"kind": "sharpe", "price_column": "Close"},
                                  "dataset": "bars"},
                                 on_submit=lambda args: {"status": "ok"})


# ---- the control plane's 422s read as one line ------------------------------------------------

def test_a_pydantic_422_reads_as_the_field_the_limit_and_what_was_sent(runner):
    """forecast(horizon=360) came back as "[{'type': 'less_than_equal', 'loc': ['body', 'horizon'], ...}]"."""
    detail = [{"type": "less_than_equal", "loc": ["body", "horizon"], "msg": "Input should be less than or equal to 256",
               "input": 360, "ctx": {"le": 256}},
              {"type": "missing", "loc": ["body", "column"], "msg": "Field required", "input": None}]
    assert runner._validation_text(detail) == ("horizon: Input should be less than or equal to 256 (you sent 360); "
                                               "column: Field required")
    assert runner._validation_text("plain text") == "plain text"


# ---- tool names -------------------------------------------------------------------------------

NAMES = {"deci_plot", "describe_data", "team_post", "team_board", "library_get", "library_list", "library_save",
         "regime_map", "regime_lab", "run_python", "submit_candidate", "gex__task_describe"}


@pytest.mark.parametrize("typed", ["deciplot", "decie_plot", "decide_plot", "decpi_plot", "Deci-Plot"])
def test_a_tool_name_one_typo_from_exactly_one_tool_is_that_tool(runner, typed):
    """10 'unknown tool' calls on the board (09-29..10-01) were deci_plot misspelt."""
    name, note = runner._resolve_tool_name(typed, NAMES)
    assert name == "deci_plot" and typed in note


@pytest.mark.parametrize("typed", ["decipic", "regime_detector", "signal", "task_describe", "run_python"])
def test_other_names_are_left_for_the_unknown_tool_reply(runner, typed):
    assert runner._resolve_tool_name(typed, NAMES) == (typed, None)


# ---- arguments read as the schema declares them -----------------------------------------------

DECI_PROPS = {"signal": {"type": "string"},
              "timeframes": {"type": "array", "items": {"type": "string"}},
              "horizons": {"type": "array", "items": {"type": "integer"}},
              "window_days": {"type": "integer"}}


def test_string_encoded_lists_and_numbers_are_decoded(runner):
    """deci_plot(timeframes='["5min", "15min"]', horizons='[1, 3, 6, 12, 0]') -- Qwen, 10-01."""
    args, notes = runner._coerce_args({"signal": "SkewRR_Value", "timeframes": '["5min", "15min"]',
                                       "horizons": "[1, 3, 6, 12, 0]", "window_days": "20"}, DECI_PROPS)
    assert args == {"signal": "SkewRR_Value", "timeframes": ["5min", "15min"], "horizons": [1, 3, 6, 12, 0],
                    "window_days": 20}
    assert len(notes) == 2 and "timeframes" in notes[0]
    assert runner._coerce_args({"timeframes": "5min, 15min", "horizons": "['1', '3']"}, DECI_PROPS)[0] == \
        {"timeframes": ["5min", "15min"], "horizons": [1, 3]}
    assert runner._coerce_args({"horizons": 12}, DECI_PROPS)[0] == {"horizons": [12]}
    assert runner._coerce_args({"horizons": ["1", 3]}, DECI_PROPS) == ({"horizons": [1, 3]}, [])
    assert runner._coerce_args({"other": "[1]"}, DECI_PROPS) == ({"other": "[1]"}, [])   # undeclared: as sent
    # A connector's optional list (gex__task_sample_rows(columns='["t", "Close"]'), 10-01).
    mcp = {"columns": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]}}
    assert runner._coerce_args({"columns": '["t", "Close"]'}, mcp)[0] == {"columns": ["t", "Close"]}


def _worker(runner, monkeypatch):
    class _NoopPoster:
        def put(self, key, doc):
            pass

    monkeypatch.setattr(runner, "_poster", lambda: _NoopPoster())
    w = runner.Worker.__new__(runner.Worker)
    w.model = w.agent_name = "Qwen3.6-35B-A3B"
    w.role, w.slot = "search", 0
    w.project = {"id": "p1", "slug": "test"}
    w._stop, w.retired = threading.Event(), threading.Event()
    w._budget_streak = 0
    w.say = MagicMock()
    return w


def test_converse_runs_a_misspelt_tool_with_decoded_arguments_and_says_so(runner, monkeypatch):
    tools = [{"type": "function", "function": {"name": "deci_plot", "description": "d",
                                               "parameters": {"type": "object", "properties": DECI_PROPS}}}]
    replies = [
        {"choices": [{"finish_reason": "tool_calls", "message": {"content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "decіe_plot", "arguments": json.dumps(
                {"signal": "SkewRR_Value", "horizons": "[1, 3, 0]"})}}]}}], "usage": {}},
        {"choices": [{"finish_reason": "stop", "message": {"content": "done", "tool_calls": []}}], "usage": {}},
    ]
    seen = []

    def call(name, args):
        seen.append((name, args))
        return {"verdict": "flat"}

    w = _worker(runner, monkeypatch)
    w._generate = lambda payload: replies.pop(0)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    ok, text, _ = w.converse(messages, tools, call, tag={"objective_id": "o1"}, max_rounds=3)
    assert ok and seen == [("deci_plot", {"signal": "SkewRR_Value", "horizons": [1, 3, 0]})]
    result = json.loads(next(m["content"] for m in messages if m["role"] == "tool"))
    assert result["verdict"] == "flat" and "decie_plot" in result["note"] and "horizons" in result["note"]


# ---- SQL cut off mid-statement ----------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "SELECT CASE WHEN gex < 0 THEN 'a' ELSE 'b' END AS r,\n  CASE WHEN",
    'SELECT "Close", "HistVol" FROM sql_exports_dbo_gexbar10s ORDER BY .',
    'SELECT "Pressure_Below", "SkewRR_Value", "Imb_OINet_D',
    "SELECT avg(x FROM t",
    "SELECT * FROM t WHERE d = '2024-01",
])
def test_a_query_that_ran_off_the_end_is_not_run(runner, sql):
    assert runner._sql_cut_off(sql) is not None


@pytest.mark.parametrize("sql", [
    "SELECT count(*) FROM t",
    "SELECT a, b FROM t WHERE x = 'in' ORDER BY a DESC LIMIT 5;",
    "-- don't forget the split\nSELECT max(t) AS last_in FROM t WHERE s = 'a--b'",
    "WITH d AS (SELECT 1 AS x) SELECT x FROM d /* done, */",
    "SELECT \"Imb_OINet_D0\" FROM bars GROUP BY 1",
])
def test_complete_queries_pass(runner, sql):
    assert runner._sql_cut_off(sql) is None


def test_query_data_refuses_a_cut_off_query_without_running_it(runner, monkeypatch):
    sent = []
    world = _objective_world(runner, monkeypatch, lambda path, payload: sent.append(path))
    out = world.call("query_data", {"sql": "SELECT a,\n  CASE WHEN"})
    assert "cut off" in out["error"] and "CASE WHEN" in out["error"]
    assert not any("/data/query" in p for p in sent)


# ---- unknown tools: what the arguments fit, and library modules --------------------------------

def test_an_unknown_tool_whose_arguments_fit_one_tool_suggests_that_tool(runner, monkeypatch):
    """Qwen3-0.6B called regime_detector(name='regime', version=1): library_get's arguments."""
    world = _objective_world(runner, monkeypatch, lambda path, payload: None)
    out = world.call("regime_detector", {"name": "regime", "version": 1})
    assert out["error"].startswith("unknown tool 'regime_detector' -- did you mean library_get")


def test_a_library_module_called_as_a_tool_is_told_how_modules_are_used(runner, monkeypatch):
    def handler(path, payload):
        if path.endswith("/library"):
            return {"modules": [{"name": "regime_detector", "kind": "regime"}]}
        return None

    world = _objective_world(runner, monkeypatch, handler)
    out = world.call("regime_detector", {"name": "regime"})
    assert "is a library module, not a tool" in out["error"]
    assert "from lib import regime_detector" in out["error"] and "regime_detector.detect(df)" in out["error"]
    assert "regime_map(regime='regime_detector')" in out["error"]


# ---- a connector tool called without `task` ---------------------------------------------------

def _task_tools():
    req = {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]}
    return [{"function": {"name": "gex__task_describe", "parameters": req}},
            {"function": {"name": "gex__task_list", "parameters": {"type": "object", "properties": {}}}}]


@pytest.mark.parametrize("tasks, filled", [(["gex_intraday"], True), (["a", "b"], False)])
def test_a_missing_task_is_the_servers_only_task(runner, monkeypatch, tasks, filled):
    """gex__task_describe({}) in a dataset objective (09-30): 'task: Field required'."""
    sent = []

    def handler(path, payload):
        if path.startswith("/api/mcp/call"):
            sent.append(payload)
            if payload["tool"] == "gex__task_list":
                return {"is_error": False, "content": json.dumps({"tasks": [{"name": t} for t in tasks]}),
                        "structured": None}
            return {"is_error": False, "content": "{}"}
        return None

    world = _objective_world(runner, monkeypatch, handler, mcp=_task_tools())
    out = world.call("gex__task_describe", {})
    if filled:
        assert sent[-1] == {"tool": "gex__task_describe", "arguments": {"task": "gex_intraday"}}
        assert "gex_intraday" in out["note"]
        world.call("gex__task_describe", {})
        assert [p["tool"] for p in sent].count("gex__task_list") == 1          # asked once per world
    else:
        assert out == {"error": "gex__task_describe needs `task` -- one of: a, b (see gex__task_list)"}
        assert [p["tool"] for p in sent] == ["gex__task_list"]
    world.call("gex__task_describe", {"task": "mine"})
    assert sent[-1]["arguments"] == {"task": "mine"}


# ---- candidates by number or id ---------------------------------------------------------------

def test_candidates_are_found_by_any_number_or_id(runner, monkeypatch):
    """The lookup read only the 500 newest of 1,545 candidates (#1233 was 'no candidate') and
    stripped a leading 'c' off ids."""
    cands = [{"id": f"{n:010x}", "seq": n} for n in range(1792, 0, -1)] + [{"id": "c0ffee1234", "seq": 5000}]
    paths = []

    def handler(path, payload):
        paths.append(path)
        if "/candidates?" in path:
            return {"candidates": cands if "limit=5000" in path else cands[:500]}
        return None

    world = _objective_world(runner, monkeypatch, handler)
    assert world._candidate("1233")[0]["seq"] == 1233
    assert world._candidate("#1233")[0]["seq"] == 1233
    assert world._candidate("c42")[0]["seq"] == 42
    assert world._candidate("c0ffee1234")[0]["seq"] == 5000
    assert world._candidate("c0ffee")[0]["seq"] == 5000                      # a unique id prefix
    hit, miss = world._candidate("9999")
    assert hit is None and "up to #5000" in miss["error"] and "idea" in miss["error"]


# ---- library_save -----------------------------------------------------------------------------

def test_library_save_takes_the_name_it_was_meant_to_have(runner, monkeypatch):
    """'signal_charm_Imb_adaptive_gate' was refused: names are lowercase identifiers."""
    sent = []

    def handler(path, payload):
        if path.endswith("/library") and payload:
            sent.append(payload)
            return {"saved": True, "name": payload["name"], "version": 1}
        return None

    world = _objective_world(runner, monkeypatch, handler)
    out = world.call("library_save", {"name": "signal_charm_Imb_adaptive_gate", "kind": "signal",
                                      "source": "def signal(df):\n    return df['x']\n",
                                      "test_code": "print(signal_charm_Imb_adaptive_gate.signal(df))"})
    assert sent[0]["name"] == "signal_charm_imb_adaptive_gate"
    assert sent[0]["code"].startswith("def signal(df)")
    assert sent[0]["test_code"] == "print(signal_charm_imb_adaptive_gate.signal(df))"
    assert out["saved"] and "from lib import signal_charm_imb_adaptive_gate" in out["note"]
    assert runner._module_name("x" * 60) == ("x" * 48, True)
    assert runner._module_name("2-step Regime") == ("m_2_step_regime", True)
    assert runner._module_name("ok_name") == ("ok_name", False)
    out = world.call("library_save", {"name": "a", "kind": "util"})
    assert "needs the module's source in `code`" in out["error"] and len(sent) == 1


# ---- connector errors returned as data --------------------------------------------------------

def test_a_taskkit_error_result_is_an_error(monkeypatch):
    """task_sample_rows answered {"error": "ValueError: unknown columns ['TotalAbsGex']"} with
    isError=False (10-01): the agent saw a success."""
    import contextlib

    from app import mcp_registry as M

    def call(structured):
        result = types.SimpleNamespace(isError=False, content=[types.SimpleNamespace(text=json.dumps(structured))],
                                       structuredContent=structured)

        class Session:
            async def call_tool(self, name, args):
                return result

        @contextlib.asynccontextmanager
        async def session(spec):
            yield Session()

        monkeypatch.setattr(M, "_session", session)
        return asyncio.run(M.call_tool([types.SimpleNamespace(name="gex", enabled=True)], "gex__task_sample_rows", {}))

    assert call({"error": "ValueError: unknown columns ['TotalAbsGex']; see task_describe"})["is_error"] is True
    assert call({"rows": [], "error": None})["is_error"] is False
    assert call({"columns": ["t"], "error": "partial"})["is_error"] is False   # data with a remark: not a failure


# ---- deci_plot expressions --------------------------------------------------------------------

@pytest.fixture
def deci(monkeypatch):
    from app import deciplot

    monkeypatch.setattr(deciplot, "dataset_columns", lambda data_dir, ds: [
        ("SlotUtc", "TIMESTAMP"), ("Close", "DOUBLE"), ("GEX", "DOUBLE"), ("SkewRR_Value", "DOUBLE"),
        ("FC_Pressure_Total", "DOUBLE"), ("Imb_OINet_D0", "DOUBLE")])
    return deciplot


def test_typeset_operators_mean_their_ascii_ones(deci):
    """'FC_Pressure_Total × 1e3 / Imb_OINet_D0' (10-01) could not be parsed."""
    spec = deci.resolve_signal({"id": "o1", "dataset": "bars"}, "", "FC_Pressure_Total × 1e3 / Imb_OINet_D0",
                               "GEX ≤ 0")
    assert spec["signal"] == "FC_Pressure_Total * 1e3 / Imb_OINet_D0 | when GEX <= 0"
    assert spec["columns"] == ["FC_Pressure_Total", "Imb_OINet_D0"]


def test_a_comparison_given_as_the_signal_is_sent_to_condition(deci):
    with pytest.raises(HTTPException) as exc:
        deci.resolve_signal({"id": "o1", "dataset": "bars"}, "", "GEX > 0")
    assert "must be NUMERIC" in exc.value.detail and "condition='GEX > 0'" in exc.value.detail


def test_an_unknown_column_names_the_close_ones():
    from app.objectives import series_expression

    with pytest.raises(HTTPException) as exc:
        series_expression("SkewRR_Valu * 2", ["SkewRR_Value", "GEX"])
    assert "did you mean SkewRR_Value" in exc.value.detail
    with pytest.raises(HTTPException) as exc:
        series_expression("GEX_change", ["SkewRR_Value", "GEX", "Close"])
    assert "a change or lag is not a column" in exc.value.detail


# ---- library errors ---------------------------------------------------------------------------

def test_a_missing_module_names_the_likely_one_or_the_ft_helper():
    from app import library as L

    names = ["signal_imb_forecast_soft_dampener", "regime_gex_vol_tercile"]
    assert "did you mean signal_imb_forecast_soft_dampener" in L.missing_module("signal_imb_forecast_soft_dampener_v2", names)
    for helper in ("align", "ft.align", "ft"):
        assert "`ft` helper module" in L.missing_module(helper, names)
    assert "library_list shows them" in L.missing_module("zzz", names)


def test_regime_map_on_a_non_regime_names_the_regime_modules():
    from app import library as L

    mods = {"regime_gex": {"kind": "regime"}, "sig_a": {"kind": "signal"}}
    assert L.not_a_regime("sig_a", mods) == ("'sig_a' is not an active regime module (it is a signal module) -- "
                                            "the regime modules are: regime_gex")
    assert "no regime module yet" in L.not_a_regime("signal", {})


def test_a_failed_scan_says_why():
    from app import library as L

    rep = {"stderr": "Traceback (most recent call last):\n  File x\nKeyError: 'Close'\n"}
    assert L._why(rep) == ": KeyError: 'Close'"
    assert L._why({"stderr": ""}) == ""


# ---- forecast horizon -------------------------------------------------------------------------

def test_a_forecast_past_the_longest_horizon_forecasts_the_longest(monkeypatch):
    """forecast(horizon=360) was a 422 (le=256); the intent -- as far ahead as possible -- is plain."""
    from app import objectives as O

    asked = {}

    class Mgr:
        async def forecast(self, model, body):
            asked.update(body)
            return {"forecasts": [{"median": [1.0] * body["horizon"], "quantiles": {}}]}

    class Cap:
        def __getattr__(self, name):
            return lambda *a, **k: None

    monkeypatch.setattr(O, "get_objective", lambda oid: {"id": oid, "project_id": "p1", "dataset": "bars",
                                                         "split_date": None})
    monkeypatch.setattr(O.projects, "get", lambda pid: {"id": pid, "data_dir": "."})
    monkeypatch.setattr(O, "_forecaster", lambda model: (Mgr(), {"model": "chronos", "context_length": 512}))
    monkeypatch.setattr(O, "_load_series", lambda *a: (list(range(100)), [float(i) for i in range(100)], "t"))
    monkeypatch.setattr(O, "input_streams", lambda *a, **k: {})
    monkeypatch.setattr(O, "_note_inputs", lambda *a: None)
    monkeypatch.setattr(O, "_capture", lambda *a, **k: Cap())
    out = asyncio.run(O.forecast_by_name("o1", O.ForecastByName(column="Close", horizon=360)))
    assert asked["horizon"] == 256 and out["horizon"] == 256 and "horizon 360" in out["note"]
    out = asyncio.run(O.forecast_by_name("o1", O.ForecastByName(column="Close", horizon=12)))
    assert asked["horizon"] == 12 and "horizon" not in out["note"].split(".")[0]


def test_the_runner_can_save_the_practices_with_post():
    """The mentor's practices rewrite POSTs (the runner's request() has no PUT); the route took PUT only
    and answered 405 -- the practices never changed (10-01 16:18)."""
    from app import playbook

    methods = {m for r in playbook.router.routes if getattr(r, "path", "") == "/projects/{project_id}/playbook"
               for m in r.methods}
    assert {"GET", "PUT", "POST"} <= methods
