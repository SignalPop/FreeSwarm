"""The mentor and the team's memory: scoreboards, cadence, idea tracking, and the plan role."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

from app import escalation, mentor, swarm_policy
from app import objectives as O


# ---------------------------------------------------------------------------------------
# A throwaway objectives database
# ---------------------------------------------------------------------------------------
@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "DB_PATH", tmp_path / "objectives.sqlite3")
    monkeypatch.setattr(O, "_conn", None)
    monkeypatch.setattr(O, "WORK_ROOT", tmp_path / "work")
    now = time.time()
    O.db().execute(
        "INSERT INTO objectives (id, project_id, title, metric, split_date, created_at, updated_at) "
        "VALUES ('o1', 'p1', 'Best strategy', ?, '2024-07-19', ?, ?)",
        (json.dumps({"kind": "sharpe", "higher_is_better": True, "price_column": "Close", "cost_bps": 2}), now, now))
    O.db().commit()
    escalation._ensure_table()
    yield O.db()
    O.db().close()


_seq = iter(range(1, 10_000))


def add_candidate(*, is_score=None, status="ok", idea_id=None, created=None, champion_at=None, metrics=None,
                  code="x = 1\n"):
    seq = next(_seq)
    O.db().execute(
        "INSERT INTO candidates (id, objective_id, seq, created_at, model, status, is_score, metrics, idea_id, "
        "champion_at, code) VALUES (?, 'o1', ?, ?, 'm', ?, ?, ?, ?, ?, ?)",
        (f"c{seq}", seq, created or time.time(), status, is_score, json.dumps(metrics or {}), idea_id,
         champion_at, code))
    O.db().commit()
    return f"c{seq}"


# ---------------------------------------------------------------------------------------
# Brute force: a candidate that only changed its parent's numbers
# ---------------------------------------------------------------------------------------
PARENT = "import ft\nx = ft.load('d')\npos = (x['GEX'] > 0.2) * 1.5  # long in positive gamma\n"


def test_change_kind_parameters_only():
    child = "import ft\nx = ft.load('d')\n\npos = (x['GEX'] > 0.35) * 2.0   # tuned\n"
    assert O._change_kind(PARENT, child) == "parameters only"


def test_change_kind_logic():
    child = "import ft\nx = ft.load('d')\npos = (x['GEX'] > 0.2) * 1.5 * (x['IntrVol'] < 0.3)\n"
    assert O._change_kind(PARENT, child) == "logic"


def test_change_kind_identical_and_unknown():
    assert O._change_kind(PARENT, PARENT + "\n") == "identical"
    assert O._change_kind(None, PARENT) is None
    assert O._change_kind(PARENT, "def broken(:\n") is None


def test_agent_is_told_when_it_only_tuned_numbers(monkeypatch):
    monkeypatch.setattr(O, "_ranked", lambda *a, **k: [])
    c = {"id": "c1", "seq": 9, "status": "ok", "is_score": 0.1, "lookahead": "pass", "stdout": "",
         "metrics": {"in_sample": {}, "change": {"kind": "parameters only", "parent_seq": 4}}}
    view = O.agent_view({"id": "o1", "metric": {"kind": "sharpe", "higher_is_better": True}}, c)
    assert "#4 with only its numbers changed" in view["change"]


# ---------------------------------------------------------------------------------------
# Ideas: stored, tracked, scored -- and kept out of the stuck ladder
# ---------------------------------------------------------------------------------------
def test_idea_scoreboard_counts_candidates_that_tested_it(db):
    iid = mentor.add_idea("o1", "DeepSeek", "Trade gamma flips only at the open: dealer hedging peaks then.")
    add_candidate(is_score=-1.0, idea_id=iid)
    add_candidate(is_score=0.5, idea_id=iid, metrics={"costs": {"verdict": "... edge before costs ..."}})
    add_candidate(status="error", idea_id=iid)
    add_candidate(is_score=3.0)  # not testing the idea
    [row] = mentor.idea_scoreboard("o1")
    assert row["id"] == iid and row["tried"] == 3 and row["ran"] == 2 and row["failed"] == 1
    assert row["best_in_sample"] == 0.5 and row["best_diagnosis"] == "edge given away by costs"


def test_mentor_ideas_do_not_reset_the_stuck_ladder(db):
    old = time.time() - 5000
    add_candidate(is_score=1.0, created=old, champion_at=old + 30)
    mentor.add_idea("o1", "DeepSeek", "A regular mentor direction about volatility regimes and exits.")
    a = escalation.assess(O.get_objective("o1"))
    assert a["ideas"] == []  # only answers to being stuck count here


def test_recrown_after_delete_is_not_an_improvement(db):
    created = time.time() - 3 * 3600
    add_candidate(is_score=1.0, created=created, champion_at=created + 60)       # crowned when evaluated
    add_candidate(is_score=0.9, created=created, champion_at=time.time() - 10)   # re-crowned hours later
    a = escalation.assess(O.get_objective("o1"))
    assert abs(a["since"] - (created + 60)) < 1


def test_ideas_for_context_lists_mentor_ideas_with_ids(db):
    iid = mentor.add_idea("o1", "DeepSeek", "Fade extreme put/call OI imbalance near expiry; exit on normalisation.")
    add_candidate(is_score=0.1, idea_id=iid)
    [i] = escalation.ideas_for_context("o1")
    assert i["id"] == iid and i["tried"] == 1 and i["trigger"] == "mentor"


# ---------------------------------------------------------------------------------------
# Cadence
# ---------------------------------------------------------------------------------------
def test_mentor_due_first_time_then_after_enough_candidates(db, monkeypatch):
    assert mentor.due("o1")[0]
    mentor.add_idea("o1", "DeepSeek", "An idea long enough to be stored as a direction.")
    assert not mentor.due("o1")[0]
    for _ in range(mentor.MENTOR_EVERY_CANDIDATES):
        add_candidate(is_score=0.0)
    assert mentor.due("o1")[0]


def test_mentor_due_after_time_with_one_candidate(db):
    mentor.add_idea("o1", "DeepSeek", "An idea long enough to be stored as a direction.")
    O.db().execute("UPDATE ideas SET ts = ts - ?", (mentor.MENTOR_EVERY_MINUTES * 60 + 5,))
    O.db().commit()
    assert not mentor.due("o1")[0]  # time alone is not enough: nothing new to read
    add_candidate(is_score=0.0)
    assert mentor.due("o1")[0]


def test_mentor_active_window(monkeypatch):
    monkeypatch.setitem(mentor._seen, "p9", time.time())
    assert mentor.mentor_active("p9")
    monkeypatch.setitem(mentor._seen, "p9", time.time() - mentor.MENTOR_ACTIVE_S - 1)
    assert not mentor.mentor_active("p9")


# ---------------------------------------------------------------------------------------
# Forecast scoreboard and habits
# ---------------------------------------------------------------------------------------
def test_forecast_scoreboard_compares_users_with_non_users(db, monkeypatch):
    feats = [{"view": "fc_imb", "params": {"model": "amazon/chronos-2", "series": ["Imb"], "covariates": ["GEX"],
                                           "horizon": 6},
              "skill": {"Imb": {"with_inputs": {"skill": 0.09, "direction": 0.64}, "lift_skill": 0.01}}},
             {"view": "fc_gex", "params": {"model": "chronos-bolt", "series": ["GEX"], "horizon": 1},
              "skill": {"GEX": {"skill_vs_no_change": -0.1, "direction_accuracy": 0.51}}}]
    monkeypatch.setattr(O, "list_features", lambda oid: feats)
    for s in (1.0, 2.0):
        add_candidate(is_score=s, metrics={"features_used": ["fc_imb"]})
    for s in (-1.0, 0.0):
        add_candidate(is_score=s)
    board = {f["view"]: f for f in mentor.forecast_scoreboard(O.get_objective("o1"))}
    imb = board["fc_imb"]
    assert imb["used_by"] == 2 and imb["helped"] == 2.0  # 1.5 vs -0.5
    assert imb["inputs"] == ["GEX"] and imb["skill"]["Imb"]["lift_from_inputs"] == 0.01
    assert board["fc_gex"]["used_by"] == 0 and board["fc_gex"]["skill"]["GEX"]["skill"] == -0.1


def test_diagnosis_mix_counts_habits(db):
    add_candidate(is_score=-1, metrics={"costs": {"verdict": "The signal has an edge before costs"},
                                        "change": {"kind": "parameters only"}})
    add_candidate(is_score=-2, metrics={"costs": {"verdict": "No clear edge before costs"}})
    add_candidate(status="error")
    mix = mentor.diagnosis_mix("o1")
    assert mix["failed_to_run"] == 1
    assert mix["results"] == {"edge given away by costs": 1, "no edge": 1}
    assert mix["changes_vs_parent"] == {"parameters only": 1}


# ---------------------------------------------------------------------------------------
# The plan: who mentors
# ---------------------------------------------------------------------------------------
LOADED = [{"model": "Muse", "ready": True, "aa": 17, "swe": 76},
          {"model": "Qwen", "ready": True, "aa": 18, "swe": 73.4},
          {"model": "DeepSeek@lambda999", "ready": True, "remote": {"node": "x"}, "aa": 40, "swe": None}]


def test_best_free_model_mentors_when_two_others_search():
    p = swarm_policy.plan({"models": None}, LOADED)
    assert [m["model"] for m in p["search"]] == ["Muse", "Qwen"]
    assert [m["model"] for m in p["mentors"]] == ["DeepSeek@lambda999"]
    assert next(m for m in p["models"] if m["model"] == "DeepSeek@lambda999")["mentor"]


def test_both_role_searches_and_mentors():
    p = swarm_policy.plan({"models": None, "model_roles": {"DeepSeek@lambda999": "both"}}, LOADED)
    assert "DeepSeek@lambda999" in [m["model"] for m in p["search"]]
    assert [m["model"] for m in p["mentors"]] == ["DeepSeek@lambda999"]


def test_no_mentor_when_too_few_searchers():
    p = swarm_policy.plan({"models": None}, LOADED[1:])
    assert p["mentors"] == [] and len(p["search"]) == 2


def test_search_role_keeps_it_searching():
    p = swarm_policy.plan({"models": None, "model_roles": {"DeepSeek@lambda999": "search"}}, LOADED)
    assert p["mentors"] == [] and "DeepSeek@lambda999" in [m["model"] for m in p["search"]]


# ---------------------------------------------------------------------------------------
# The runner's mentor: prompt and parsing
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def runner():
    path = Path(__file__).resolve().parents[1] / "swarm_runner.py"
    spec = importlib.util.spec_from_file_location("swarm_runner_mentor_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_mentor_from_fenced_reply(runner):
    text = 'Here you go:\n```json\n{"directions": [{"idea": "A", "test": "B"}], "coaching": "stop tuning"}\n```'
    notes = runner.parse_mentor(text)
    assert notes["directions"][0]["idea"] == "A" and notes["coaching"] == "stop tuning"


def test_parse_mentor_falls_back_to_coaching(runner):
    notes = runner.parse_mentor("Just prose, no JSON {here}.")
    assert notes["directions"] == [] and "Just prose" in notes["coaching"]


def test_mentor_prompt_carries_the_evidence(runner):
    brief = {"objective": {"title": "T", "description": "", "metric": {"price_column": "Close", "cost_bps": 2}},
             "metric_label": "Sharpe ratio", "habits": {"changes_vs_parent": {"parameters only": 7}},
             "leaderboard": [], "recent": [], "lessons": ["KEEP: 15-min bars"],
             "ideas": [{"id": 3, "minutes_ago": 5, "tried": 2, "ran": 2, "best_in_sample": 0.4, "best_seq": 9,
                        "best_diagnosis": "no edge", "median_in_sample": 0.1, "idea": "gamma flips"}],
             "forecasts": [{"view": "fc_imb", "model": "chronos-2", "series": ["Imb"], "inputs": ["GEX"],
                            "horizon": 6, "skill": {}, "used_by": 4, "helped": 0.8}],
             "forecasters": [{"model": "amazon/chronos-2", "supports_covariates": True}]}
    p = runner.mentor_prompt(brief, [{"seq": 77, "from": "Qwen", "text": "should we flip #532?"}])
    for needle in ('"parameters only": 7', "idea 3", "fc_imb", "helped 0.8", "[77] from Qwen", "KEEP: 15-min bars",
                   '"directions"', '"forecasts"'):
        assert needle in p, needle
