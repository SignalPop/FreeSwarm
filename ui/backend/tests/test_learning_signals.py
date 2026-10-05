"""What the swarm learns from a candidate that ran but was not ranked: the iteration goes on to fix
it, the brief states the activity floor up front, lessons carry how their candidate fared, and the
brief does not repeat itself."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from app import objectives as O

RUNNER = Path(__file__).resolve().parents[1] / "swarm_runner.py"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("swarm_runner_learning", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_unranked_reason_only_for_fixable_misses(runner):
    sparse = {"status": "ok", "not_ranked": "holdout: only 5 active days (need 20)"}
    assert runner._unranked_reason(sparse) == "holdout: only 5 active days (need 20)"
    one_sided = {"status": "ok", "not_ranked_because": "one-sided: 2 long and 0 short trades in-sample"}
    assert runner._unranked_reason(one_sided).startswith("one-sided")
    none = {"status": "ok", "not_ranked": "no trades: the strategy never opened a position in-sample"}
    assert runner._unranked_reason(none).startswith("no trades")
    assert runner._unranked_reason({"status": "ok", "rank": "3 of 40"}) is None
    assert runner._unranked_reason({"status": "error", "not_ranked": "only 5 active days"}) is None
    # Nothing the script can change: no next try for it.
    assert runner._unranked_reason({"status": "ok", "not_ranked": "holdout: no score (the task server could "
                                                                  "not score that period)"}) is None


def test_a_timed_out_repair_teaches_the_estimate_and_is_forgotten_later(runner, monkeypatch):
    """10-04: every Qwen3.6 crash sent a repair that waited the whole 360s and timed out; the
    failure was never recorded, so the next crash did the same."""
    m = "Qwen3.6-test"
    runner._note_reply_failed("repair", m, 359.0)
    assert runner._expected_reply_s(m, 1000, kind="repair") >= 500        # > the 360s limit: not asked
    runner._note_reply_s("repair", m, 60.0)                              # a later success pulls it back down
    assert runner._expected_reply_s(m, 1000, kind="repair") < 400
    key = f"repair|{m}"
    monkeypatch.setitem(runner._reply_at, key, runner._reply_at[key] - runner.REPLY_S_STALE_S - 1)
    assert runner._expected_reply_s(m, 1000, kind="repair") is None       # stale: asked again


def test_code_cut_at_a_statement_boundary_is_refused_as_truncated(runner):
    cut = "import ft\nMIN_HOLD = 32; COOLDOWN = 3; EV"                 # #1851, 10-04
    assert runner._truncated_code_call("submit_candidate", {"code": cut}, True, "tool_calls")
    # A name the script defines, or a builtin, is a (useless) statement, not a cut.
    assert not runner._truncated_code_call("run_python", {"code": "x = 1\nx"}, True, "tool_calls")
    assert not runner._truncated_code_call("run_python", {"code": "import ft\nprint"}, True, "tool_calls")
    assert not runner._truncated_code_call("run_python", {"code": "import ft\nft.report_positions(p)"}, True,
                                           "tool_calls")


def test_doubly_escaped_quotes_are_unescaped_only_when_that_fixes_the_code(runner):
    sent = '\\"\\"\\"Day-level risk scale.\\"\\"\\"\nimport numpy as np\nx = np.zeros(3)\n'
    assert runner._unescaped_code(sent) == '"""Day-level risk scale."""\nimport numpy as np\nx = np.zeros(3)\n'
    fine = 'print("say \\"hi\\"")\n'
    assert runner._unescaped_code(fine) is None                        # compiles as written
    assert runner._unescaped_code("def f(:\n  pass\n") is None          # broken for another reason


def test_monitor_does_not_take_an_auto_repaired_script_as_platform_evidence():
    from app import monitor as M

    repaired = json.dumps({"auto_repaired": {"attempts": 1, "errors": ["AttributeError: 'list' object has no "
                                                                       "attribute 'to_list'"]}, "ok": True,
                           "stdout": "[Your script failed with AttributeError ...; it was repaired automatically]"})
    assert M._reports_failure(repaired)
    assert M._reports_failure("{\"ok\": true, \"stdout\": \"[... it was repaired automatically -- the code" + "x" * 5000)
    assert not M._reports_failure(json.dumps({"ok": True, "stdout": "rows 100"}))


def test_monitor_does_not_take_the_agents_reply_to_the_mentor_as_platform_evidence():
    """#401: "I accept. I will keep candidate 1855's core..." -- the agent's own words -- filed as
    incorrect data."""
    from app import monitor as M

    said = "I accept. I will keep candidate 1855's core and participation intact"
    r = {"timeline": [{"kind": "tool", "name": "answer_feedback", "ok": True, "result": said}]}
    assert not M._platform_evidence(said, {"agent": "Muse"}, r)


def _ctx(**extra):
    return {
        "objective": {"title": "T", "description": "", "split_date": "2024-07-19", "dataset": "d", "lookahead_check": True,
                      "metric": {"kind": "sharpe", "price_column": "Close", "cost_bps": 2, "max_leverage": 3}},
        "metric_label": "Sharpe ratio", "datasets": ["d"], "mode": "explore",
        "fields": {"GEX": {"about": "gamma", "columns": ["GEX"]}},
        "recent": [{"seq": 9, "model": "m", "status": "ok", "problem": "holdout: only 5 active days (need 20)",
                    "lookahead": "pass", "rationale": "r"}],
        **extra,
    }


def test_brief_states_the_activity_floor_and_recent_misses(runner):
    act = {"need": 20, "in_sample_days": 223, "holdout_days": 97, "share": 0.31, "in_sample_active": 69,
           "recent": 40, "recent_sparse": 12}
    p = runner.iteration_prompt(_ctx(activity=act))
    assert "ACTIVITY FLOOR (REQUIRED" in p and "69 or more of its 223" in p and "31% of sessions" in p
    assert "12 of the last 40 candidates are NOT RANKED" in p
    # The floor is stable text: it sits with the rules, ahead of the per-iteration sections.
    assert p.index("ACTIVITY FLOOR (REQUIRED") < p.index("FIELD GUIDE") < p.index("RECENT ATTEMPTS")
    assert "ACTIVITY FLOOR" not in runner.iteration_prompt(_ctx())


def test_decile_studies_list_each_study_once(runner):
    row = {"signal": "Pressure_Below", "h": 1, "spread_bps": 0.03, "t": 3.54, "rho": 0.939, "verdict": "monotone",
           "consistency": 1.0}
    shape = {"signal": "Pressure_Below", "timeframe": "10s", "h": 1, "means": [-0.04, 0.02], "shape": "rising",
             "spread_bps": 0.06, "t": 3.51}
    deci = {"studied": 2, "best_by_timeframe": {"10s": [row, row, row]}, "shapes": [shape] * 5,
            "flat": ["GEX", "GEX", "Pinning_Composite"]}
    text = "\n".join(runner._knowledge_lines(deci, None))
    assert text.count("Pressure_Below h1 top-bottom") == 1
    assert text.count("Pressure_Below 10s h1: [") == 1
    assert "GEX, Pinning_Composite" in text and "GEX, GEX" not in text


def test_task_columns_keep_only_key_and_target_beside_the_field_guide(runner):
    cols = [{"name": "t", "role": "key", "description": "bar time"},
            {"name": "Close", "role": "target", "description": "SPY last"},
            {"name": "GEX", "role": "signal", "description": "net dealer gamma exposure"}]
    ctx = {"objective": {"metric": {"kind": "task", "task_server": "gex", "task": "x"}, "dataset": None},
           "task": {"columns": cols}}
    full = "\n".join(runner._task_lines(ctx))
    assert "GEX = net dealer gamma exposure" in full
    short = "\n".join(runner._task_lines({**ctx, "fields": {"(base)": {"about": "", "columns": ["GEX"]}}}))
    assert "t (key) = bar time" in short and "Close (target)" in short and "GEX = net dealer" not in short


def test_lessons_are_tagged_with_their_candidates_standing_and_deduped():
    held = json.dumps({"rank": {"holdout": 1.0}})
    rows = [
        ("KEEP: GEX filter 0.5", "a", 10, "ok", 1.2, "", "pass", None, 1.2, held),
        ("KEEP: GEX filter 0.7", "a2", 11, "ok", 1.1, "", "pass", None, 1.1, held),   # the same lesson, other numbers
        ("KEEP: sparse entries on DABS", "b", 12, "ok", None, "holdout: only 5 active days (need 20)", "pass", None,
         2.0, "{}"),
        ("AVOID: shift(-1)", "c", 13, "ok", 0.9, "", "fail", None, 0.9, "{}"),
        ("TRY: vanna fade", "d", 14, "error", None, "Traceback", None, None, None, None),
        ("KEEP: consolidated rule " + "x" * 600, None, None, None, None, None, None, None, None, None),
    ]
    out = O._lesson_lines(rows, {"a": 1})
    assert out[0] == "[#10: rank 1, holds up on unseen data] KEEP: GEX filter 0.5"
    assert len(out) == 5
    assert out[1].startswith("[#12: NOT RANKED -- holdout: only 5 active days")
    assert out[2].startswith("[#13: DISQUALIFIED -- look-ahead]")
    assert out[3].startswith("[#14: failed to run]")
    assert out[4].startswith("KEEP: consolidated rule") and len(out[4]) == O.LESSON_CHARS


def test_holdout_check_is_a_word_never_a_number(runner):
    def cand(is_score, ho, score=0.1):
        return {"score": score, "is_score": is_score, "metrics": {"rank": {"holdout": ho}}}

    assert O.holdout_check(cand(4.695, 0.147)) == "collapses"            # #157 of 19f971
    assert O.holdout_check(cand(2.0, 1.0)) == "weakens"
    assert O.holdout_check(cand(1.5, 1.6)) == "holds up"
    assert O.holdout_check(cand(1.5, 1.6, score=None)) is None           # unranked: nothing to say
    assert O.holdout_check(cand(-1.0, 0.5)) is None                      # nothing in-sample to carry over
    assert O.holdout_check(cand(1.5, 1.6), higher=False) is None
    ctx = _ctx(leaderboard=[{"rank": 1, "seq": 157, "model": "m", "in_sample_score": 4.695, "rationale": "r",
                             "holdout_check": "collapses"}],
               mode="improve", parent={"seq": 157, "rank": 1, "in_sample_score": 4.695, "rationale": "r",
                                       "holdout_check": "collapses"})
    p = runner.iteration_prompt(ctx)
    assert "in-sample 4.695; collapses on unseen data" in p and "UNSEEN-DATA CHECK" in p
    assert "Its in-sample result COLLAPSES on unseen data" in p
    assert "0.147" not in p


def test_holdout_too_sparse_to_score_says_so_from_in_sample_numbers():
    days = [f"2024-{m:02d}-{d:02d}" for m in range(1, 13) for d in range(1, 29)]
    # Active on every 10th day: ~10% of sessions in-sample, so far too few in the holdout.
    returns = [[d, (0.001 if i % 10 == 0 else 0.0)] for i, d in enumerate(days)]
    obj = {"metric": {"kind": "sharpe", "min_active_days": 20}, "split_date": "2024-09-01"}
    score, _, note, m = O._score_returns_unsided(obj, returns)
    assert score is None and note.startswith("holdout: only") and "the strategy is SPARSE" in note
    assert f"on {m['in_sample']['active_days']} of {m['in_sample']['days']} days in-sample" in note


def test_activity_brief_counts_recent_sparse_misses(monkeypatch):
    con = sqlite3.connect(":memory:", check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("CREATE TABLE candidates (objective_id TEXT, seq INT, status TEXT, score REAL, score_note TEXT, metrics TEXT)")
    m = json.dumps({"in_sample": {"days": 223, "active_days": 40}, "holdout": {"days": 97, "active_days": 9}})
    con.executemany("INSERT INTO candidates VALUES (?,?,?,?,?,?)", [
        ("o", 1, "ok", None, "holdout: only 9 active days (need 20)", m),
        ("o", 2, "ok", 0.5, "", m),
        ("o", 3, "error", None, "Traceback", None),
        ("o", 4, "ok", None, "one-sided: 3 long and 0 short", m),
    ])
    monkeypatch.setattr(O, "db", lambda: con)
    obj = {"id": "o", "split_date": "2024-07-19", "metric": {"kind": "sharpe", "min_active_days": 20}}
    a = O._activity_brief(obj)
    assert a["need"] == 20 and a["holdout_days"] == 97 and a["in_sample_days"] == 223
    assert a["in_sample_active"] == 69 and a["share"] == 0.31
    assert a["recent"] == 4 and a["recent_sparse"] == 1
    assert O._activity_brief({**obj, "split_date": None}) is None
