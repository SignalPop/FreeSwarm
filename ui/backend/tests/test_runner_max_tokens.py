"""Regression tests for the max_tokens floor and the truncated-code guard.

Two bugs from qwen/qwen3.8-27b@groq:

  #4 -- swarm replies came back cut off mid-sentence. The runner's max_tokens formula,
        ``max(256, min(budget, ctx - est - 128))``, floored at 256 whenever the real prompt
        exceeded the window it thought it had. External models had no entry in ``_context``,
        so ``context_for`` returned DEFAULT_CONTEXT=8192 while Groq's real Qwen3.8 window is
        131,072. A 25,720-token prompt (well under Groq's real window) then went out with
        ``max_tokens=256`` and Groq happily returned 256 tokens with finish "length".

  #1 -- 43 ``run_python`` calls arrived with code cut off mid-token
        (``bar = ft.load('sql_exports_db``, ``print("GEX:", g.quantile``). Same root cause:
        max_tokens floored at 256 truncated the tool_call arguments. The extra guard here is
        a safety net: even after the floor is raised, code that does not compile AND was
        obviously cut off no longer burns an experiment -- the runner refuses it and asks the
        model to resend a shorter script.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "swarm_runner.py"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("swarm_runner_under_test_max_tokens", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------
# max_tokens: what the runner asks for
# ------------------------------------------------------------------------------------------
def _max_tokens(runner, *, ctx: int, est: int, budget: int | None = None,
                cap: int | None = None) -> int:
    """The formula the runner uses to size max_tokens (kept in sync with swarm_runner.py)."""
    budget = budget if budget is not None else runner.MAX_TOKENS
    provider_cap = cap if cap is not None else budget
    available = ctx - est - 128
    return min(provider_cap, budget, max(runner.MIN_OUTPUT, available))


def test_min_output_is_at_least_1024(runner):
    """A floor of 256 truncated tool calls and answers on external models; 1024 leaves room
    for a plausible tool call plus a short reply."""
    assert runner.MIN_OUTPUT >= 1024


def test_correct_ctx_uses_full_budget(runner):
    """The fixed case: with Groq's real 128K window, a 26K prompt leaves plenty of room for
    the runner's own budget."""
    got = _max_tokens(runner, ctx=131_072, est=25_720, budget=16_384, cap=16_384)
    assert got == 16_384


def test_wrong_ctx_no_longer_truncates_to_256(runner):
    """Bug #4 exact numbers: ctx thought to be DEFAULT (8192), est ~25,720. The OLD formula
    ``max(256, min(budget, ctx-est-128))`` returned 256. The NEW formula never dips below
    MIN_OUTPUT, so the reply has room to complete even when the window guess is wrong."""
    got = _max_tokens(runner, ctx=8_192, est=25_720, budget=16_384, cap=16_384)
    assert got >= runner.MIN_OUTPUT
    assert got >= 1024  # explicit -- the old bug was 256


def test_provider_cap_is_respected(runner):
    """Groq caps Qwen3.8-27b's output at 16,384 tokens per request. Even with plenty of window
    room, we never ask for more than the provider will generate."""
    got = _max_tokens(runner, ctx=131_072, est=1_000, budget=32_768, cap=16_384)
    assert got == 16_384


def test_env_budget_is_respected(runner):
    """FREESWARM_SWARM_MAX_TOKENS caps what we ask for even when the provider and window would
    allow more."""
    got = _max_tokens(runner, ctx=131_072, est=1_000, budget=4_096, cap=65_536)
    assert got == 4_096


def test_tiny_available_still_gets_min_output(runner):
    """A wrong-small ctx that makes ``ctx-est-128`` come out near zero must not silently
    truncate to 256; MIN_OUTPUT is the floor. If the provider then overflows, the existing
    retry path (learn ctx from the refusal) corrects it."""
    got = _max_tokens(runner, ctx=8_192, est=8_000, budget=16_384, cap=16_384)
    assert got == runner.MIN_OUTPUT


# ------------------------------------------------------------------------------------------
# context registry: external models must not fall back to DEFAULT_CONTEXT
# ------------------------------------------------------------------------------------------
def test_external_context_lookup_when_populated(runner):
    """``context_for`` reads from ``_context``; the sync loop populates it for external models
    from the catalog. If the runner ever regresses to using DEFAULT for hosted models, this
    fails."""
    name = "qwen/qwen3.8-27b@groq"
    runner._context[name] = 131_072
    try:
        assert runner.context_for(name) == 131_072
    finally:
        runner._context.pop(name, None)


def test_external_context_default_when_unpopulated(runner):
    """Sanity: if nothing populates _context, we fall back to DEFAULT_CONTEXT (the bug's
    starting condition)."""
    name = "some/uncatalogued-model@groq"
    runner._context.pop(name, None)
    assert runner.context_for(name) == runner.DEFAULT_CONTEXT


def test_max_output_lookup_when_populated(runner):
    """``max_output_for`` reads from ``_max_output``, populated for external models from the
    catalog's provider-published cap."""
    name = "qwen/qwen3.8-27b@groq"
    runner._max_output[name] = 16_384
    try:
        assert runner.max_output_for(name) == 16_384
    finally:
        runner._max_output.pop(name, None)


def test_max_output_none_when_unknown(runner):
    name = "some/local-engine"
    runner._max_output.pop(name, None)
    assert runner.max_output_for(name) is None


# ------------------------------------------------------------------------------------------
# truncated-code guard: run_python with cut-off arguments
# ------------------------------------------------------------------------------------------
def test_truncated_load_call_flagged(runner):
    """Bug #1 exact example: ``bar = ft.load('sql_exports_db`` -- open string literal, reply
    finish 'tool_calls' (Groq's quirk when max_tokens hits mid-argument). Compile fails with
    a cut-off marker, so the guard fires."""
    assert runner._truncated_code_call(
        "run_python", {"code": "bar = ft.load('sql_exports_db"}, True, "tool_calls") is True


def test_truncated_print_call_flagged(runner):
    """Second Bug #1 example: ``print("GEX quantiles:", g.quantile`` -- open paren."""
    assert runner._truncated_code_call(
        "run_python", {"code": "print(\"GEX quantiles:\", g.quantile"}, True, "tool_calls") is True


def test_truncated_code_with_finish_length_flagged(runner):
    """finish 'length' alone is enough evidence when the code also does not compile."""
    assert runner._truncated_code_call(
        "run_python", {"code": "x = 1 + "}, True, "length") is True


def test_bad_json_envelope_flagged(runner):
    """If Groq's JSON envelope itself would not parse, we saw ``args = {}`` in the caller.
    Even an empty-code run_python call here means the tool call arrived cut off."""
    assert runner._truncated_code_call("run_python", {}, False, "tool_calls") is True


def test_valid_code_is_not_flagged(runner):
    """Compileable code is never flagged -- the guard must never delay a successful call."""
    assert runner._truncated_code_call(
        "run_python", {"code": "import ft\nprint(ft.__name__)"}, True, "tool_calls") is False


def test_missing_colon_is_not_flagged(runner):
    """A genuine SyntaxError the model wrote (missing colon) with a normal 'stop' finish is NOT
    flagged -- the sandbox should surface the error so the model learns, not the guard."""
    assert runner._truncated_code_call(
        "run_python", {"code": "def foo()\n    return 1"}, True, "stop") is False


def test_missing_colon_with_length_finish_still_flagged(runner):
    """Even a model-written bug: if the reply's finish is 'length', we treat it as truncated
    and refuse -- there is no way to tell if the missing colon is the model's or the cutoff's,
    and re-asking costs nothing while running a broken script costs an experiment."""
    assert runner._truncated_code_call(
        "run_python", {"code": "def foo()\n    return 1"}, True, "length") is True


def test_non_run_python_not_flagged(runner):
    """The guard only applies to run_python -- other tools have their own validation."""
    assert runner._truncated_code_call(
        "query_data", {"sql": "SELECT * FROM foo WHERE bar = '"}, True, "length") is False


def test_unclosed_bracket_flagged(runner):
    """`[` never closed -- Python 3.10+ says "'[' was never closed", which we match."""
    assert runner._truncated_code_call(
        "run_python", {"code": "xs = [1, 2, 3,\ny = 4"}, True, "tool_calls") is True


# ------------------------------------------------------------------------------------------
# Truncated-code guard: cover the other tools that carry Python source
# ------------------------------------------------------------------------------------------
def test_library_save_truncated_code_flagged(runner):
    """Bug #1 also appeared as a library_save whose signal() body ran off the end mid-index
    expression (``src = df[``). Groq returned finish "tool_calls" and only 205 completion
    tokens. Before this fix the guard only looked at run_python; the truncated save reached
    the library API, which rejected it -- wasting a save slot the mentor still counts."""
    truncated_code = (
        "def signal(df, lookback=20, min_bars=8):\n"
        "    import ft\n"
        "    cols = ['SlotUtc', 'Close']\n"
        "    avail = [c for c in df.columns if c in cols]\n"
        "    src = df["
    )
    assert runner._truncated_code_call(
        "library_save",
        {"name": "iv_gate", "kind": "signal", "code": truncated_code},
        True, "tool_calls") is True


def test_library_save_truncated_test_code_flagged(runner):
    """The library_save `test_code` argument is Python too; a cut-off test also proves the
    reply was truncated. The guard concatenates both source arguments before compiling."""
    good_code = "def signal(df):\n    return df['Close'] * 0\n"
    truncated_test = "import ft\nrows = ft.load('sql_exports_db"
    assert runner._truncated_code_call(
        "library_save",
        {"name": "flat", "kind": "signal", "code": good_code, "test_code": truncated_test},
        True, "tool_calls") is True


def test_submit_candidate_truncated_code_flagged(runner):
    """submit_candidate carries the whole strategy; a truncated tool_calls reply here burns a
    candidate slot on a script that cannot compile. Same trailing-token cutoff pattern as
    Bug #1's run_python failures."""
    assert runner._truncated_code_call(
        "submit_candidate",
        {"code": "import ft\nrows = ft.load('sql_exports_db",
         "rationale": "gex signal"},
        True, "tool_calls") is True


def test_submit_candidate_valid_code_not_flagged(runner):
    """A submit_candidate whose script compiles must never be blocked -- that would refuse
    every candidate silently."""
    good = (
        "import ft\n"
        "rows = ft.load('sql_exports_dbo_gexbar10s', columns=['SlotUtc', 'Close'])\n"
        "ft.report_positions([0] * len(rows))\n"
    )
    assert runner._truncated_code_call(
        "submit_candidate", {"code": good, "rationale": "flat baseline"}, True, "tool_calls") is False


# ------------------------------------------------------------------------------------------
# Trailing-token heuristic: catch the SyntaxError-shaped cutoffs whose message is only
# "invalid syntax". These slipped through the marker list even after the max_tokens floor
# was raised, because Groq's finish stays "tool_calls" and the specific SyntaxError we get
# is not one of the classic cutoff messages.
# ------------------------------------------------------------------------------------------
def test_trailing_dot_flagged(runner):
    """``rows['SkewRR_Value'].to_np.`` -- the reply was cut just before the attribute name,
    Python raises a generic "invalid syntax", but a well-formed statement never ends on a
    dot. Seen live on qwen/qwen3.8-27b@groq (completion 106-215 tokens)."""
    assert runner._truncated_code_call(
        "run_python",
        {"code": "import ft\nrows = ft.rows_pl()\nx = rows['SkewRR_Value'].to_np."},
        True, "tool_calls") is True


def test_trailing_equals_flagged(runner):
    """``hv    =`` -- cut just after an assignment target. Live example: 215 completion
    tokens ended on ``hv    =``. A missing right-hand side is truncation, not a bug the model
    wrote on purpose."""
    assert runner._truncated_code_call(
        "run_python", {"code": "import numpy as np\nclose = np.zeros(10)\nhv    ="},
        True, "tool_calls") is True


def test_trailing_comma_flagged(runner):
    """A tuple / call whose last argument is missing: ``ft.load('gex', columns=['a', 'b',``."""
    assert runner._truncated_code_call(
        "run_python", {"code": "import ft\nrows = ft.load('gex', columns=['a', 'b',"},
        True, "tool_calls") is True


def test_trailing_open_paren_flagged(runner):
    """The classic cutoff: ``print("GEX quantiles:", g.quantile(`` -- ends on an opener."""
    assert runner._truncated_code_call(
        "run_python", {"code": "g = 1\nprint(\"GEX quantiles:\", g.quantile("},
        True, "tool_calls") is True


def test_unterminated_triple_quote_flagged(runner):
    """library_save Bug #1: the model's signal() opened a triple-quoted docstring and the
    reply was cut before the closing ``\"\"\"``. Python 3.12+ says "unterminated triple-quoted
    string literal", which is not word-identical to the earlier "unterminated string literal"
    -- broadening the marker to "unterminated" catches both."""
    code = (
        "import pandas as pd\n"
        "def signal(df):\n"
        '    """df: 15-min bars.\n'
        "    Returns"
    )
    assert runner._truncated_code_call(
        "library_save", {"name": "iv", "kind": "signal", "code": code},
        True, "tool_calls") is True


def test_trailing_letter_not_flagged(runner):
    """Compileable code whose last char is a letter is never treated as truncated by the
    trailing-char rule. A ``def foo()\\n    return 1`` (missing colon) still ends on a digit
    -- compile fails, but the trailing char is not one that only appears mid-expression."""
    # This is a genuine model-written bug with a normal 'stop' finish.
    assert runner._truncated_code_call(
        "run_python", {"code": "def foo()\n    return 1"}, True, "tool_calls") is False
