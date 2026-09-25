"""Tool calls a model wrote as text instead of making them (swarm_runner._text_tool_calls)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "swarm_runner.py"
NAMES = {"submit_candidate", "query_data"}


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("swarm_runner_under_test", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_xml_invoke_mixed_with_harmony_prefix(runner):
    # What gpt-oss actually posted: its channel prefix, then an XML-ish call.
    text = ('<|start|> to=submit_candidate<|message|><atem:function_calls>\n'
            '<atem:invoke name="submit_candidate">\n'
            '<atem:parameter name="code">\nimport ft\nx = 1 < 2\n</atem:parameter>\n'
            '<atem:parameter name="rationale">flip it</atem:parameter>\n'
            '</atem:invoke>\n</atem:function_calls>')
    assert runner._text_tool_calls(text, NAMES) == [
        ("submit_candidate", {"code": "import ft\nx = 1 < 2", "rationale": "flip it"})]


def test_harmony_json(runner):
    text = ('<|start|>assistant<|channel|>commentary to=functions.submit_candidate <|constrain|>json'
            '<|message|>{"code": "print({1: 2})", "rationale": "r"}<|call|>')
    assert runner._text_tool_calls(text, NAMES) == [("submit_candidate", {"code": "print({1: 2})", "rationale": "r"})]


def test_bare_json_with_string_arguments(runner):
    text = 'Calling now: {"name": "query_data", "arguments": "{\\"sql\\": \\"SELECT 1\\"}"} ok'
    assert runner._text_tool_calls(text, NAMES) == [("query_data", {"sql": "SELECT 1"})]


def test_truncated_script_is_not_submitted(runner):
    text = '<invoke name="submit_candidate"><parameter name="code">import ft\ndf = ft.load('
    assert runner._text_tool_calls(text, NAMES) == []


def test_unknown_tool_and_prose_ignored(runner):
    assert runner._text_tool_calls('<invoke name="rm_rf"><parameter name="x">1</parameter></invoke>', NAMES) == []
    assert runner._text_tool_calls("I think the {signal} is noisy; next I will test it.", NAMES) == []
    assert runner._text_tool_calls("", NAMES) == []


def _ctx():
    return {
        "objective": {"title": "T", "description": "", "split_date": "2024-07-19", "dataset": "d", "lookahead_check": True,
                      "metric": {"kind": "sharpe", "price_column": "Close", "cost_bps": 2, "max_leverage": 3}},
        "metric_label": "Sharpe ratio", "datasets": ["d"], "mode": "explore",
        "forecasters": [{"model": "amazon/chronos-2", "family": "chronos2", "context_length": 8192,
                         "native_horizon": 64, "supports_covariates": True}],
        "forecast_board": [{"view": "fc_imb", "model": "amazon/chronos-2", "series": ["Imb"], "inputs": ["GEX"],
                            "horizon": 6, "skill": {"Imb": {"skill": 0.09}}, "used_by": 4, "helped": 0.8}],
        "fields": {"GEX": {"about": "gamma", "columns": ["GEX"]}},
        "ideas": [{"id": 3, "model": "DeepSeek", "text": "IDEA: fade imbalance", "tried": 1}],
        "lessons": ["KEEP: 15-min bars"],
        "leaderboard": [{"rank": 1, "seq": 5, "model": "m", "in_sample_score": 0.1, "rationale": "r"}],
    }


def test_iteration_prompt_puts_stable_sections_first(runner):
    p = runner.iteration_prompt(_ctx())
    order = ["HOW \"BETTER\" IS MEASURED", "FIELD GUIDE", "FORECAST SCOREBOARD", "DIRECTIONS FROM THE MENTOR",
             "LEADERBOARD", "TEAMMATES RIGHT NOW", "YOUR ASSIGNMENT THIS ITERATION"]
    at = [p.index(x) for x in order]
    assert at == sorted(at)


def test_iteration_prompt_offers_ideas_by_number_and_ft_forecast(runner):
    p = runner.iteration_prompt(_ctx())
    assert "[idea 3] from DeepSeek, tried 1 times" in p
    assert "pass its number as `idea`" in p
    assert "ft.forecast(" in p and "reading GEX" in p and "helped 0.8" in p
    assert "COSTS DECIDE MOST RESULTS HERE" in p
