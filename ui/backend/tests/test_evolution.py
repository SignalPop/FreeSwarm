"""The search's memory: islands of distinct strategies, niches, lenses, instruction experiments and the two-tier checks.

Evidence (b27d52cb27, 11-01): the top 20 candidates are one strategy and the top eight islands share one niche, so 27 "new
bests" reshuffled the same trades. Candidates that changed 20-50% of their parent beat it 44% of the time, those
under 20% only 12-17%."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import pytest

from app import evolution as E
from app import external, library, mentor
from app import objectives as O

RUNNER = Path(__file__).resolve().parents[1] / "swarm_runner.py"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("swarm_runner_evolution_test", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "DB_PATH", tmp_path / "objectives.sqlite3")
    monkeypatch.setattr(O, "_conn", None)
    monkeypatch.setattr(O, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(external, "CONFIG_PATH", tmp_path / "external.json")
    monkeypatch.setattr(library, "_ready", False)
    E._cache.clear()
    now = time.time()
    O.db().execute("INSERT INTO objectives (id, project_id, title, metric, dataset, split_date, created_at, updated_at, require_audit) "
                   "VALUES ('o1', 'p1', 't', ?, 'px', '2024-03-01', ?, ?, 1)",
                   (json.dumps({"kind": "sharpe", "price_column": "Close", "higher_is_better": True}), now, now))
    O.db().commit()
    yield O.db()
    O.db().close()


DAYS = [f"2024-01-{d:02d}" for d in range(1, 31)] + [f"2024-02-{d:02d}" for d in range(1, 29)]


def _series(seed: int, noise: float = 0.0, base: int | None = None) -> list[list]:
    rng = np.random.default_rng(seed if base is None else base)
    r = rng.normal(0.001, 0.01, len(DAYS))
    if noise:
        r = r + np.random.default_rng(seed).normal(0, noise, len(DAYS))
    return [[d, float(v)] for d, v in zip(DAYS, r)]


def _cand(seq, is_score, score=None, *, returns=None, parent=None, status="ok", lookahead="pass", audit="none", long=20, short=20,
          active=30, smooth=0.8, calmar=3.0, change=None, assignment=None, novelty=None, mode="improve", code="x = 1"):
    metrics = {"in_sample": {"days": 58, "active_days": active, "smoothness": smooth, "calmar": calmar, "sharpe": is_score},
               "execution": {"sides": {"long": long, "short": short}}, "rank": {"holdout": is_score}}
    if change:
        metrics["change"] = change
    if novelty:
        metrics["novelty"] = novelty
    O.db().execute(
        "INSERT INTO candidates (id, objective_id, seq, created_at, model, mode, parent_id, rationale, code, status, score, is_score, "
        "metrics, returns, lookahead, audit, assignment) VALUES (?, 'o1', ?, ?, 'M', ?, ?, 'why', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (f"c{seq}", seq, time.time(), mode, parent, code, status, score, is_score, json.dumps(metrics),
         json.dumps(returns if returns is not None else _series(seq)), lookahead, audit, json.dumps(assignment or {})))
    O.db().commit()


def _obj():
    return O.get_objective("o1")


# ---------------------------------------------------------------------------------------
# Islands
# ---------------------------------------------------------------------------------------
def test_copies_of_one_strategy_are_one_island_however_their_code_differs(db):
    for seq, score in ((1, 2.0), (2, 1.9), (3, 1.8)):                 # the same return stream, a little noise
        _cand(seq, score, score, returns=_series(seq, 1e-5, base=7))
    _cand(4, 1.5, 1.5, returns=_series(40))
    a = E.archive(_obj(), fresh=True)
    assert a["members"] == 4 and len(a["islands"]) == 2
    assert sorted(i["size"] for i in a["islands"]) == [1, 3]


def test_the_archive_keeps_out_what_cannot_be_a_parent(db):
    _cand(1, 2.0, 2.0)
    _cand(2, 2.0, 2.0, lookahead="fail")
    _cand(3, 2.0, 2.0, audit="fail")
    _cand(4, -1.0, -1.0)
    _cand(5, 2.0, None, status="error")
    assert E.archive(_obj(), fresh=True)["members"] == 1


def test_islands_read_in_sample_days_only(db):
    """Two strategies that differ only after the split are one strategy to everything agents act on."""
    head = _series(5)
    tail_a, tail_b = [["2024-03-05", 0.02], ["2024-03-06", -0.01]], [["2024-03-05", -0.02], ["2024-03-06", 0.03]]
    _cand(1, 2.0, 2.0, returns=head + tail_a)
    _cand(2, 1.9, 1.9, returns=head + tail_b)
    assert len(E.archive(_obj(), fresh=True)["islands"]) == 1


def test_a_parent_comes_from_any_island_and_the_busy_one_gives_way(db):
    for i in range(4):
        _cand(i + 1, 2.0 - i / 10, 2.0 - i / 10)
    for j in range(30):                                                 # the team has only worked on #1
        _cand(100 + j, None, None, parent="c1", status="error")
    rng = random.Random(0)
    picks = [E.pick_parent(_obj(), rng) for _ in range(300)]
    counts = {i: sum(1 for p in picks if p["id"] == f"c{i}") for i in (1, 2, 3, 4)}
    assert all(counts.values()) and counts[1] < counts[2]


def test_no_archive_parent_while_there_is_only_one_strategy(db):
    _cand(1, 2.0, 2.0)
    assert E.pick_parent(_obj()) is None


def test_objectives_without_a_daily_series_have_no_archive(db):
    O.db().execute("UPDATE objectives SET metric=? WHERE id='o1'", (json.dumps({"kind": "judge"}),))
    O.db().commit()
    assert E.pick_parent(_obj()) is None and E.scorecard(_obj()) is None and E.view(_obj()) == {"applicable": False}


# ---------------------------------------------------------------------------------------
# Lenses, niches, scorecard
# ---------------------------------------------------------------------------------------
def test_a_lens_rewards_its_own_kind_of_good_on_in_sample_numbers():
    c = {"is_score": 2.0, "metrics": {"in_sample": {"days": 100, "active_days": 50, "smoothness": 0.5, "calmar": 7.0},
                                      "holdout": {"sharpe": 99.0}, "execution": {"sides": {"long": 10, "short": 10}}}}
    assert E.lens_value(c, "primary") == 2.0
    assert E.lens_value(c, "smooth") == 1.0
    assert E.lens_value(c, "drawdown") == 7.0
    assert E.lens_value(c, "breadth") == pytest.approx(2.0 * np.log1p(20))
    assert E.lens_value(c, "novel", 0.25) == 1.5
    assert E.lens_value({**c, "is_score": -1.0}, "smooth") is None


def test_a_task_candidate_has_a_place_on_the_map_too():
    d = {"trades": {"long": 90, "short": 10}, "trades_per_day": 2.0}
    c = {"metrics": {"in_sample": {"days": 50, "active_days": 40}, "task": {"diagnostics": {"in_sample": d}}}}
    b = E.behaviour(c)
    assert b["trades"] == 100 and b["long_share"] == 0.9
    assert E.niche(b) == (2, 2, "long")


def test_explorers_are_sent_where_nobody_has_been(db):
    _cand(1, 2.0, 2.0, long=4, short=0, active=5)            # the rare, long-only corner
    spot = E.empty_niche(_obj(), random.Random(0))
    assert spot["empty"] and spot["cell"][2] == "both" and spot["cell"][0] >= 1


def test_the_scorecard_says_how_concentrated_the_search_is(db):
    for i in range(6):
        _cand(i + 1, 2.0 - i / 100, 2.0, returns=_series(i, 1e-5, base=3))
    _cand(10, 1.0, 1.0, returns=_series(11))
    card = E.scorecard(_obj())
    assert card["strategies"] == 2 and card["top"] == 7 and card["top_same_strategy"] == 6
    assert card["leaders"]["primary"]["seq"] == 1


def test_inspirations_come_from_other_islands_with_their_code(db):
    _cand(1, 2.0, 2.0, code="parent")
    _cand(2, 1.5, 1.5, code="other")
    got = E.inspirations(_obj(), "c1", random.Random(0))
    assert [i["seq"] for i in got] == [2] and got[0]["code"] == "other"


# ---------------------------------------------------------------------------------------
# Instruction experiments
# ---------------------------------------------------------------------------------------
def _won(seq, variant, win, parent="c0"):
    _cand(seq, 3.0, 3.0 if win else 1.0, parent=parent, returns=_series(seq, assignment_noise := 1e-4, base=seq),
          assignment={"variant": variant, "parent_via": "archive", "lens": "smooth"})


def test_seeds_are_written_once_and_the_first_of_a_mode_is_the_control(db):
    E._seed("o1")
    E._seed("o1")
    improve = E.variants("o1", "improve")
    assert [v["name"] for v in improve] == ["focused", "structural", "searcher"]
    assert improve[0]["control"] == 1 and improve[0]["text"] == "" and improve[1]["control"] == 0


def test_a_win_is_a_jump_over_the_parent_and_a_small_gain_is_not(db):
    _cand(1, 2.0, 2.0)
    _cand(2, 2.1, 2.1, parent="c1", assignment={"variant": 1})      # under a tenth of the leader's 2.5
    _cand(3, 2.5, 2.5, parent="c1", assignment={"variant": 1})
    _cand(4, 0.5, 0.5, parent="c1", assignment={"variant": 1})
    got = {o["seq"]: o for o in E._outcomes(_obj())}
    assert not got[2]["win"] and got[2]["beat_parent"]
    assert got[3]["win"] and not got[4]["win"]


def test_an_explorer_wins_with_a_new_strategy_not_a_copy_of_the_leader(db):
    _cand(1, 2.0, 2.0, mode="explore", assignment={"variant": 1}, novelty={"rho_leader": 0.1})
    _cand(2, 2.0, 2.0, mode="explore", assignment={"variant": 1}, novelty={"rho_leader": 0.95})
    got = {o["seq"]: o["win"] for o in E._outcomes(_obj())}
    assert got == {1: True, 2: False}


def test_a_disqualified_or_unfinished_candidate_earns_nothing(db):
    _cand(1, 2.0, 2.0)
    _cand(2, 5.0, 5.0, parent="c1", lookahead="fail", assignment={"variant": 1})
    _cand(3, 5.0, 5.0, parent="c1", lookahead="pending", assignment={"variant": 1})
    got = E._outcomes(_obj())
    assert [(o["seq"], o["win"]) for o in got] == [(2, False)]


def test_thompson_sampling_prefers_what_wins_and_still_tries_the_rest(db):
    _cand(0, 1.0, 1.0)
    E._seed("o1")
    ids = {v["name"]: v["id"] for v in E.variants("o1", "improve")}
    seq = 1
    for i in range(12):
        _won(seq, ids["structural"], True)
        seq += 1
    for i in range(12):
        _won(seq, ids["focused"], False)
        seq += 1
    rng = random.Random(1)
    picks = [E.pick_variant(_obj(), "improve", rng)["name"] for _ in range(100)]
    assert picks.count("structural") > picks.count("focused")
    assert picks.count("searcher") > 0                                # untried, so it draws widely


def test_a_clear_loser_is_retired_but_the_control_never_is(db):
    _cand(0, 1.0, 1.0)
    E._seed("o1")
    ids = {v["name"]: v["id"] for v in E.variants("o1", "improve")}
    seq = 1
    for _ in range(20):
        _won(seq, ids["structural"], True)
        seq += 1
    for _ in range(20):
        _won(seq, ids["searcher"], False)
        seq += 1
    for _ in range(20):
        _won(seq, ids["focused"], False)
        seq += 1
    assert E.settle_experiments(_obj()) == ["searcher"]
    assert sorted(v["name"] for v in E.variants("o1", "improve")) == ["focused", "structural"]


def test_new_instructions_are_refused_when_full_or_repeated(db):
    E._seed("o1")
    text = "Rewrite the exit system and keep the entries exactly as they are."
    assert E.add_variant("o1", "improve", "exit", text, "mentor")["name"] == "exit"
    with pytest.raises(E.HTTPException, match="already under test"):
        E.add_variant("o1", "improve", "exit two", "  " + text, "mentor")
    for i in range(E.MAX_ACTIVE_VARIANTS - 4):
        E.add_variant("o1", "improve", f"v{i}", f"A different instruction number {i} for the team.", "mentor")
    with pytest.raises(E.HTTPException, match="under test"):
        E.add_variant("o1", "improve", "more", "One more instruction that does not fit in.", "mentor")


def test_the_control_cannot_be_retired_by_hand(db):
    E._seed("o1")
    control = E.variants("o1", "improve")[0]
    with pytest.raises(E.HTTPException, match="control"):
        asyncio.run(E.retire_variant("o1", control["id"]))


def test_the_assignment_carries_the_treatments(db):
    _cand(1, 2.0, 2.0)
    a = E.assign(_obj(), "improve", "archive", "c1", "smooth", 4, random.Random(0))
    assert a["parent_via"] == "archive" and a["lens"] == "smooth" and a["practices_version"] == 4 and a["variant"]
    assert "niche" not in a
    assert "niche" in E.assign(_obj(), "explore", None, None, None, 4, random.Random(0))
    assert "variant" not in E.assign(_obj(), "build", None, None, None, 4, random.Random(0))


def test_the_mentor_section_has_win_counts_and_no_ranked_gains(db):
    _cand(1, 2.0, 2.0)
    section = E.mentor_section(_obj())
    assert section["variants"] and not any("median_gain" in v or "beat_parent" in v for v in section["variants"])
    assert section["scorecard"]["strategies"] == 1


# ---------------------------------------------------------------------------------------
# The brief and the submission
# ---------------------------------------------------------------------------------------
def _brief_ctx(**kw):
    return {"objective": {"title": "T", "metric": {"kind": "sharpe"}}, "metric_label": "Sharpe", "mode": "improve", **kw}


def test_the_brief_carries_the_instruction_the_lens_and_the_other_strategies(runner):
    parent = {"seq": 5, "rationale": "r", "code": "print(1)", "model": "M"}
    p = runner.iteration_prompt(_brief_ctx(
        parent=parent, assignment={"instruction": "Replace one whole component.", "lens": "smooth"},
        inspirations=[{"seq": 9, "model": "Q", "in_sample_score": 1.2, "niche": "daily", "rho_leader": 0.1, "rationale": "rr", "code": "pass"}],
        scorecard={"candidates": 300, "strategies": 2, "top": 20, "top_same_strategy": 20,
                   "leaders": {"smooth": {"about": "the smoothest curve", "seq": 4, "model": "M", "in_sample_score": 1.0, "trades": 40}}}))
    assert "Replace one whole component." in p and "Make ONE focused change" not in p
    assert "a smoother equity curve" in p
    assert "OTHER STRATEGIES ON THE TEAM" in p and "Candidate 9 by Q" in p
    assert "20 of the top 20 are ONE strategy" in p and "the smoothest curve: candidate 4" in p


def test_the_control_keeps_the_brief_as_it_was_and_explorers_get_a_niche(runner):
    p = runner.iteration_prompt(_brief_ctx(parent={"seq": 5, "rationale": "r", "code": "x"}, assignment={"instruction": ""}))
    assert "Make ONE focused change" in p
    e = runner.iteration_prompt(_brief_ctx(mode="explore", assignment={
        "niche": {"label": "daily, most days, both sides", "empty": True}, "instruction": "Write a family."}))
    assert "never reached: daily, most days, both sides" in e and "Write a family." in e


def test_the_mentor_sees_the_experiments_and_may_propose_one(runner):
    brief = {"objective": {"title": "T", "metric": {"kind": "sharpe"}}, "metric_label": "Sharpe",
             "evolution": {"scorecard": {"candidates": 300, "strategies": 2, "top": 20, "top_same_strategy": 20},
                           "variants": [{"id": 3, "mode": "improve", "name": "structural", "wins": 4, "n": 9, "failed": 1,
                                         "status": "active", "control": 0, "text": "Replace a part"}]}}
    p = runner.mentor_prompt(brief, [])
    assert "INSTRUCTIONS UNDER TEST" in p and "[3] improve 'structural'" in p and '"instruction"' in p
    task = {"objective": {"title": "T", "metric": {"kind": "task"}}, "metric_label": "Sharpe"}
    t = runner.mentor_prompt(task, [])
    assert '"forecasts"' not in t and '"instruction"' in t


def test_parent_comes_from_the_archive_and_stays_recorded(db, monkeypatch):
    _cand(1, 2.0, 2.0)
    _cand(2, 1.5, 1.5)
    monkeypatch.setattr(O, "get_candidate", O.get_candidate)
    monkeypatch.setattr(random, "random", lambda: 0.99)            # past the build and explore rolls...
    monkeypatch.setattr(E, "ARCHIVE_SHARE", 1.1)                   # ...and into the archive
    from app import projects

    monkeypatch.setattr(projects, "get", lambda pid: {"id": pid, "data_dir": ""})
    ctx = asyncio.run(O.context("o1", "M"))
    a = ctx["assignment"]
    assert ctx["mode"] == "improve" and a["parent_via"] == "archive" and a["lens"] in E.LENSES and a.get("variant")
    assert ctx["scorecard"]["strategies"] == 2
    assert ctx["inspirations"] and ctx["inspirations"][0]["seq"] != ctx["parent"]["seq"]


# ---------------------------------------------------------------------------------------
# Change size and the stages
# ---------------------------------------------------------------------------------------
def test_change_size_counts_code_lines_not_comments():
    a = "\n".join(f"x{i} = {i}" for i in range(10))
    assert O._change_size(a, a + "\n# note\n") == 0
    assert 0 < O._change_size(a, a.replace("x3 = 3", "x3 = 99")) < 0.2
    assert O._change_size(a, "completely = 'new'") == 1
    assert O._change_size(None, a) is None


def test_the_static_stage_stops_broken_syntax_and_unknown_library_names(monkeypatch):
    import app.library as L

    monkeypatch.setattr(L, "module_files", lambda pid: {".ft/lib/good.py": "x = 1", ".ft/lib/__init__.py": ""})
    obj = {"project_id": "p1"}
    syntax = O._static_problem(obj, "def f(:\n  pass\n")
    assert "Traceback" in syntax and "SyntaxError" in syntax
    assert "ImportError" in O._static_problem(obj, "from lib import missing\n")
    assert O._static_problem(obj, "from lib import good\nimport ft\n") is None


def test_the_smoke_cut_is_early_in_the_in_sample_period():
    cut = O._smoke_cut({"metric": {"mid_cut": "2024-01-19 18:17:00"}, "split_date": "2024-07-19"})
    assert "2023-08-01" < cut < "2023-11-01"
    assert O._smoke_cut({"metric": {}, "split_date": "2024-07-19"}) is None


def _smoked(stderr, monkeypatch, db):
    async def fake_run(*a, **k):
        return {"ok": False, "stderr": stderr, "stdout": ""}

    async def fake_off(fn, *a):
        return {"items": []}

    monkeypatch.setattr(O, "_run_forecasting", fake_run)
    monkeypatch.setattr(O, "_off", fake_off)
    _cand(1, 2.0, 2.0)
    O.db().execute("UPDATE candidates SET eval_seconds=90 WHERE id='c1'")
    obj = {**_obj(), "eval_timeout_s": 300}
    obj["metric"]["mid_cut"] = "2024-01-19 18:17:00"
    return asyncio.run(O._smoke_stop(obj, "c1", "code", "d", []))


@pytest.mark.parametrize("stderr", [
    "Traceback...\nTypeError: unsupported operand type(s) for *: 'NoneType' and 'int'",     # a mean of no rows yet
    "Traceback...\nNameError: name 'state' is not defined",                                  # set in a later branch
    "Traceback...\nUnboundLocalError: cannot access local variable 'x'",
    "Traceback...\nAttributeError: 'NoneType' object has no attribute 'x'",
    "Traceback...\nValueError: not enough rows",
    "ImportError: this one was handled\nTraceback...\nValueError: then fewer rows ended it",  # only the final one counts
])
def test_a_smoke_failure_fewer_rows_could_explain_goes_on_to_the_full_run(stderr, db, monkeypatch):
    assert _smoked(stderr, monkeypatch, db) is None


@pytest.mark.parametrize("stderr", ["Traceback...\nImportError: cannot import name 'x'", "SyntaxError: invalid syntax",
                                    "Traceback...\nModuleNotFoundError: No module named 'y'"])
def test_a_smoke_failure_no_row_count_explains_ends_the_evaluation(stderr, db, monkeypatch):
    assert _smoked(stderr, monkeypatch, db)["ok"] is False


def test_smoke_runs_only_for_a_slow_lineage(db, monkeypatch):
    calls = []

    async def fake_run(*a, **k):
        calls.append(a)
        return {"ok": False, "stderr": "ImportError: cannot import name 'x'", "stdout": ""}

    async def fake_off(fn, *a):
        return {"items": []}

    monkeypatch.setattr(O, "_run_forecasting", fake_run)
    monkeypatch.setattr(O, "_off", fake_off)
    _cand(1, 2.0, 2.0)
    obj = {**_obj(), "eval_timeout_s": 300}
    obj["metric"]["mid_cut"] = "2024-01-19 18:17:00"
    O.db().execute("UPDATE candidates SET eval_seconds=10 WHERE id='c1'")
    assert asyncio.run(O._smoke_stop(obj, "c1", "code", "d", [])) is None and not calls
    O.db().execute("UPDATE candidates SET eval_seconds=90 WHERE id='c1'")
    assert asyncio.run(O._smoke_stop(obj, "c1", "code", "d", []))["ok"] is False and calls


def test_a_look_ahead_near_the_top_gets_every_cut_and_a_weak_one_stays_short(db, monkeypatch):
    for i in range(30):
        _cand(i + 1, 5.0 - i / 10, 5.0 - i / 10)
    O.db().execute("UPDATE objectives SET best_id='c1' WHERE id='o1'")
    O.db().commit()
    obj = _obj()
    _cand(100, 1.0, 1.0, parent="c1")                              # weak, and worse than its parent
    _cand(101, 9.0, 9.0, parent="c1")                              # would take the title
    _cand(102, 2.0, None)                                          # cannot rank at all
    assert not O._worth_full_lookahead(obj, "c100")
    assert O._worth_full_lookahead(obj, "c101")
    assert not O._worth_full_lookahead(obj, "c102")

    calls = []

    async def fake_lookahead(*a, **k):
        calls.append(k)
        return "pass", "fine"

    monkeypatch.setattr(O, "_lookahead", fake_lookahead)
    worst, detail = asyncio.run(O._two_tier_lookahead(obj, "c100", "x", "d", [], True, None, []))
    assert (worst, detail) == ("short", O.LOOKAHEAD_SHORT + "fine") and len(calls) == 1
    assert calls[0]["active_k"] == O.LOOKAHEAD_FIRST_CUTS
    calls.clear()
    asyncio.run(O._two_tier_lookahead(obj, "c101", "x", "d", [], True, None, []))
    assert len(calls) == 2 and calls[1]["with_fixed"] is False


def test_a_leak_found_in_the_first_tier_ends_the_test(db, monkeypatch):
    _cand(1, 2.0, 2.0)
    calls = []

    async def fake_lookahead(*a, **k):
        calls.append(k)
        return "fail", "leak"

    monkeypatch.setattr(O, "_lookahead", fake_lookahead)
    assert asyncio.run(O._two_tier_lookahead(_obj(), "c1", "x", "d", [], True, None, [])) == ("fail", "leak")
    assert len(calls) == 1


def test_nothing_short_tested_can_take_the_title(db):
    """The verdict of the first tier is not a pass: not at submission, not by a rescore, not as an ensemble member."""
    from app import ensembles

    O.db().execute("UPDATE objectives SET require_audit=0, lookahead_check=1, metric=? WHERE id='o1'",
                   (json.dumps({"kind": "sharpe", "price_column": "Close", "higher_is_better": True, "mid_cut": "2024-01-19"}),))
    O.db().commit()
    _cand(1, 9.0, 9.0, lookahead="short", returns=_series(1))
    obj = _obj()
    assert O._settle(obj, "c1", 1, "x", {"status": "ok", "score": 9.0, "lookahead": "short"}) is False
    assert _obj()["best_id"] is None
    assert O._recrown(_obj())["champion"] is None and _obj()["best_id"] is None
    assert "look-ahead" in ensembles.ineligible(O.get_candidate("c1"))


def test_a_candidate_that_can_take_the_title_never_stops_at_the_short_test(db):
    _cand(1, 2.0, 2.0)
    assert O._worth_full_lookahead(_obj(), "c1")                 # no champion yet: anything scored could be one
    O.db().execute("UPDATE objectives SET best_id='c1' WHERE id='o1'")
    O.db().commit()
    for i in range(30):
        _cand(i + 10, 1.9 - i / 100, 1.9 - i / 100)
    _cand(100, 2.5, 2.5, parent="c10")
    assert O._worth_full_lookahead(_obj(), "c100")


def test_the_top_ranks_are_counted_without_clones(db):
    """19f971: the top 25 rows held 7 distinct scores, so rank 26 was really 7th and got only the short test."""
    _cand(1, 5.0, 5.0)
    O.db().execute("UPDATE objectives SET best_id='c1' WHERE id='o1'")
    O.db().commit()
    for i in range(30):
        _cand(i + 2, 4.0, 4.0)                                   # clones: one distinct rank
    _cand(100, 3.0, 3.0, parent="c1")
    assert O._worth_full_lookahead(_obj(), "c100")


def test_a_short_candidate_that_climbs_into_the_top_gets_the_rest_of_the_test(db, monkeypatch):
    _cand(1, 2.0, 2.0, lookahead="short")
    _cand(2, 1.0, 1.0)
    obj = _obj()
    runs, cuts = [], []
    pos = Path(O.WORK_ROOT) / "p.parquet"

    async def fake_run(*a, **k):
        runs.append(a[0])
        return {"ok": True, "stderr": "", "stdout": "", "run_dir": "x"}

    async def fake_lookahead(*a, **k):
        cuts.append(k)
        return "pass", "the other cuts"

    from app import projects

    monkeypatch.setattr(projects, "get", lambda pid: {"id": pid, "data_dir": ""})
    monkeypatch.setattr(O, "_run_forecasting", fake_run)
    monkeypatch.setattr(O, "_positions_file", lambda rep: pos)
    monkeypatch.setattr(O, "_lookahead", fake_lookahead)

    async def go():
        O._finish_short_tests(obj, O._distinct(O._ranked("o1", True)))
        O._finish_short_tests(obj, O._distinct(O._ranked("o1", True)))       # not twice
        await asyncio.gather(*list(O._bg))

    asyncio.run(go())
    assert len(runs) == 1 and cuts[0]["with_fixed"] is False
    c = O.get_candidate("c1", light=True)
    assert c["lookahead"] == "pass" and c["lookahead_detail"] == "the other cuts"
    assert not O._finishing


def test_a_leak_found_by_the_rest_of_the_test_disqualifies(db, monkeypatch):
    _cand(1, 2.0, 2.0, lookahead="short")

    async def fake_run(*a, **k):
        return {"ok": True, "stderr": "", "stdout": "", "run_dir": "x"}

    async def fake_lookahead(*a, **k):
        return "fail", "leak"

    from app import projects

    monkeypatch.setattr(projects, "get", lambda pid: {"id": pid, "data_dir": ""})
    monkeypatch.setattr(O, "_run_forecasting", fake_run)
    monkeypatch.setattr(O, "_positions_file", lambda rep: Path("p"))
    monkeypatch.setattr(O, "_lookahead", fake_lookahead)
    monkeypatch.setattr(O, "_auto_quarantine", lambda *a: [])
    monkeypatch.setattr(O, "_publish_candidate", lambda *a: None)
    asyncio.run(O._finish_short_test(_obj(), "c1"))
    assert O.get_candidate("c1", light=True)["lookahead"] == "fail"


def test_a_candidate_with_no_trades_to_cut_keeps_its_first_tier_pass(db, monkeypatch):
    """With no in-sample position change the second tier has no cuts: that is not an error."""
    obj = _obj()
    obj["metric"]["price_column"] = "Close"
    got = asyncio.run(O._lookahead({**obj, "lookahead_check": True}, "x", "d", [], False, None, [], active_k=0, with_fixed=False))
    assert got[0] == "pass"


def test_a_guarded_library_import_is_left_to_the_run(monkeypatch):
    import app.library as L

    monkeypatch.setattr(L, "module_files", lambda pid: {".ft/lib/good.py": "x = 1"})
    code = "try:\n    from lib import maybe\nexcept ImportError:\n    maybe = None\n"
    assert O._static_problem({"project_id": "p1"}, code) is None


def test_one_iteration_is_one_trial_however_many_times_it_submitted(db):
    """A crash, its repair and its retries were told the same thing: 1 win in 1, not 1 in 3."""
    _cand(1, 1.0, 1.0)
    told = {"variant": 1, "iteration": "it1"}
    _cand(2, None, None, parent="c1", status="error", assignment=told)
    _cand(3, 1.0, 1.0, parent="c1", assignment=told)
    _cand(4, 3.0, 3.0, parent="c1", assignment=told)
    _cand(5, None, None, parent="c1", status="error", assignment={"variant": 1, "iteration": "it2"})
    got = E._outcomes(_obj())
    assert len(got) == 2
    won = next(o for o in got if o["assignment"]["iteration"] == "it1")
    assert won["win"] and not won["failed"]
    assert next(o for o in got if o["assignment"]["iteration"] == "it2")["failed"]


def test_an_iteration_waits_for_all_its_submissions_to_settle(db):
    _cand(1, 1.0, 1.0)
    told = {"variant": 1, "iteration": "it1"}
    _cand(2, 1.0, 1.0, parent="c1", assignment=told)
    _cand(3, 3.0, 3.0, parent="c1", lookahead="pending", assignment=told)
    assert E._outcomes(_obj()) == []


def test_every_context_gets_its_own_iteration_id(db):
    _cand(1, 2.0, 2.0)
    a, b = (E.assign(_obj(), "improve", None, None, None, 1) for _ in range(2))
    assert a["iteration"] != b["iteration"]


def test_an_untried_instruction_does_not_get_a_proven_one_retired(db):
    """Control 2/30, structural 3/30, two new arms: the new arms' wide draws left structural 1.6% and it was retired."""
    _cand(0, 1.0, 1.0)
    E._seed("o1")
    ids = {v["name"]: v["id"] for v in E.variants("o1", "improve")}
    seq = 1
    for name, wins in (("focused", 2), ("structural", 3)):
        for i in range(30):
            _won(seq, ids[name], i < wins)
            seq += 1
    E.add_variant("o1", "improve", "new one", "A new instruction with nothing tried yet.", "mentor")
    E.add_variant("o1", "improve", "new two", "Another new instruction with nothing tried yet.", "mentor")
    assert E.settle_experiments(_obj()) == []


def test_an_objective_with_no_positions_gets_no_instruction(db):
    O.db().execute("UPDATE objectives SET metric=? WHERE id='o1'", (json.dumps({"kind": "judge"}),))
    O.db().commit()
    a = E.assign(_obj(), "improve", None, None, None, 1)
    assert "instruction" not in a and "variant" not in a
    O.db().execute("UPDATE objectives SET metric=? WHERE id='o1'", (json.dumps({"kind": "sharpe"}),))    # self-reported
    O.db().commit()
    assert "instruction" not in E.assign(_obj(), "improve", None, None, None, 1)


def test_where_lower_is_better_only_the_score_itself_is_a_lens(db):
    O.db().execute("UPDATE objectives SET metric=? WHERE id='o1'", (json.dumps({"kind": "task", "higher_is_better": False}),))
    O.db().commit()
    _cand(1, 0.5, 0.5, smooth=0.1, returns=_series(1))
    _cand(2, 3.0, 3.0, smooth=0.9, returns=_series(2))
    obj = _obj()
    assert E.lens_value(O.get_candidate("c2"), "smooth", 0.1, False) is None
    card = E.scorecard(obj)
    assert list(card["leaders"]) == ["primary"] and card["leaders"]["primary"]["seq"] == 1
    rng = random.Random(0)
    assert {E.pick_parent(obj, rng)["id"] for _ in range(50)} == {"c1", "c2"}
    O.db().execute("UPDATE candidates SET assignment=? WHERE id='c1'", (json.dumps({"variant": 1}),))
    O.db().commit()
    assert E._outcomes(obj)[0]["win"]                            # an explorer can win an error-measure objective


def test_a_deleted_or_disqualified_candidate_leaves_the_archive_at_once(db):
    _cand(1, 2.0, 2.0)
    _cand(2, 1.9, 1.9)
    _cand(3, 1.8, 1.8)
    assert E.archive(_obj())["members"] == 3
    O.db().execute("DELETE FROM candidates WHERE id='c2'")
    O.db().commit()
    assert E.archive(_obj())["members"] == 2
    O.db().execute("UPDATE candidates SET lookahead='fail' WHERE id='c3'")
    O.db().commit()
    assert E.archive(_obj())["members"] == 1


def test_a_broken_archive_costs_the_iteration_nothing(db, monkeypatch):
    _cand(1, 2.0, 2.0)
    _cand(2, 1.5, 1.5)
    from app import projects

    def boom(obj):
        raise RuntimeError("archive down")

    monkeypatch.setattr(projects, "get", lambda pid: {"id": pid, "data_dir": ""})
    monkeypatch.setattr(random, "random", lambda: 0.99)
    monkeypatch.setattr(E, "ARCHIVE_SHARE", 1.1)
    monkeypatch.setattr(E, "pick_parent", boom)
    ctx = asyncio.run(O.context("o1", "M"))
    assert ctx["mode"] == "improve" and ctx["assignment"]["parent_via"] == "tournament" and ctx["parent"]


def test_a_child_of_an_unranked_parent_faces_the_same_bar(db):
    """Otherwise the archive, which allows unranked parents, wins by a child merely being positive."""
    _cand(1, 2.0, None)
    _cand(2, 2.05, 2.05, parent="c1", assignment={"variant": 1, "parent_via": "archive"}, novelty={"rho_leader": 0.1})
    _cand(3, 2.9, 2.9, parent="c1", assignment={"variant": 1, "parent_via": "archive"}, novelty={"rho_leader": 0.1})
    got = {o["seq"]: o["win"] for o in E._outcomes(_obj())}
    assert got == {2: False, 3: True}


def test_a_disqualified_score_sets_no_bar(db):
    _cand(1, 1.0, 1.0)
    _cand(2, 50.0, 50.0, lookahead="fail")
    _cand(3, 1.5, 1.5, parent="c1", assignment={"variant": 1})
    assert E._outcomes(_obj())[-1]["win"]
    assert E.view(_obj())["jump"] == pytest.approx(0.15)


def test_the_view_for_the_console_has_every_section(db):
    _cand(1, 2.0, 2.0)
    _cand(2, 1.5, 1.5, change={"kind": "logic", "size": 0.3, "parent_seq": 1}, parent="c1", assignment={"variant": 1, "parent_via": "archive", "lens": "smooth"})
    v = E.view(_obj())
    assert v["applicable"] and v["strategies"] == 2 and v["experiments"] and v["niches"]
    assert v["change_sizes"][0]["arm"] == "medium (20-50%)"
    assert [r["arm"] for r in v["parent_arms"]] == ["archive"]
    assert set(v["stages"]) >= {"static", "smoke", "full", "ok", "cuts_saved"}


def test_mentor_brief_includes_the_evolution_section(db, monkeypatch):
    monkeypatch.setattr(mentor.external, "config", lambda: {"mentor": {"every_candidates": 5, "every_minutes": 30}})
    from app import projects

    monkeypatch.setattr(projects, "get", lambda pid: {"id": pid, "data_dir": ""})
    for name in ("_field_scan_brief", "_deci_brief", "_combo_brief", "_library_brief", "_playbook", "_forecasters_brief",
                 "_practices_due", "field_guide"):
        monkeypatch.setattr(O, name, lambda *a, **k: None)
    monkeypatch.setattr(mentor, "forecast_scoreboard", lambda obj: [])
    _cand(1, 2.0, 2.0)
    assert mentor.brief("o1")["evolution"]["variants"]
