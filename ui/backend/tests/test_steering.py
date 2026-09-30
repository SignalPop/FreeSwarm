"""Steering the search toward better strategies (09-30): who searches, what the leaderboard shows
agents, and what the shared memory lets through."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

from app import library as L
from app import objectives as O
from app import swarm_policy

RATED = [{"model": "Muse", "ready": True, "aa": 17, "swe": 76},
         {"model": "Qwen", "ready": True, "aa": 18, "swe": 73.4},
         {"model": "DeepSeek@lambda999", "ready": True, "remote": {"node": "x"}, "aa": 40, "swe": None}]
TINY = {"model": "Qwen/Qwen3-0.6B", "ready": True}


def test_an_unrated_model_does_not_search_beside_rated_ones():
    p = swarm_policy.plan({"models": None}, RATED + [TINY])
    assert "Qwen/Qwen3-0.6B" not in [m["model"] for m in p["search"]]
    assert "Qwen/Qwen3-0.6B" in [m["model"] for m in p["reserved"]]
    assert [m["model"] for m in p["mentors"]] == ["DeepSeek@lambda999"]     # still two rated searchers


def test_an_unrated_model_searches_when_it_is_all_there_is_or_when_set_to_search():
    assert [m["model"] for m in swarm_policy.plan({"models": None}, [TINY])["search"]] == ["Qwen/Qwen3-0.6B"]
    p = swarm_policy.plan({"models": None, "model_roles": {"Qwen/Qwen3-0.6B": "search"}}, RATED + [TINY])
    assert "Qwen/Qwen3-0.6B" in [m["model"] for m in p["search"]]


@pytest.fixture(scope="module")
def runner():
    path = Path(__file__).resolve().parents[1] / "swarm_runner.py"
    spec = importlib.util.spec_from_file_location("swarm_runner_steering_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_mentor_to_be_gets_no_search_agents_during_the_startup_grace(runner):
    """Bug #8: right after a start DeepSeek was the only model ready, so it searched -- and each of
    those iterations was retired unfinished minutes later when it became the mentor."""
    alone = swarm_policy.plan({"models": None}, RATED[2:])            # only DeepSeek is ready yet
    assert [m["model"] for m in alone["search"]] == ["DeepSeek@lambda999"]
    assert runner._mentor_to_be(alone, "DeepSeek@lambda999")
    assert not runner._mentor_to_be(alone, "Muse")
    forced = swarm_policy.plan({"models": None, "model_roles": {"DeepSeek@lambda999": "search"}}, RATED[2:])
    assert not runner._mentor_to_be(forced, "DeepSeek@lambda999")      # the operator's Search is kept
    assert runner.STARTUP_GRACE_S > 0


def test_clones_leave_the_ranking_agents_see():
    ranked = [{"id": "a", "score": -2.657, "is_score": -2.2}, {"id": "b", "score": -2.657, "is_score": -2.2},
              {"id": "c", "score": -2.657, "is_score": -1.0}, {"id": "d", "score": -3.0, "is_score": None}]
    assert [c["id"] for c in O._distinct(ranked)] == ["a", "c", "d"]


def test_the_lesson_parser_ignores_markdown_and_keeps_only_lessons(runner):
    """Qwen3-0.6B's '**KEEP:** ... **TEAM:** I built on the mentor's idea' and 'LIB x: works'
    lines went into the team lessons verbatim."""
    kept, team, verdicts = runner._reflection_lines(
        "- **KEEP:** SkewRR at 30 min, t 3.1\n**TEAM:** built on idea 902\n* LIB comp_x: works -- #59 +0.4\n"
        "1. AVOID: 10s deciles, below costs\nSome prose the model added\nTRY: GexFlip distance at 10:30")
    assert kept == ["KEEP: SkewRR at 30 min, t 3.1", "AVOID: 10s deciles, below costs", "TRY: GexFlip distance at 10:30"]
    assert team == "built on idea 902"
    assert verdicts == [("comp_x", "works", "#59 +0.4")]


# ---------------------------------------------------------------------------------------
# "works" must be earned
# ---------------------------------------------------------------------------------------
@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "DB_PATH", tmp_path / "objectives.sqlite3")
    monkeypatch.setattr(O, "_conn", None)
    now = time.time()
    O.db().execute("INSERT INTO objectives (id, project_id, title, metric, split_date, created_at, updated_at) "
                   "VALUES ('o1', 'p1', 't', ?, '2024-07-19', ?, ?)",
                   (json.dumps({"kind": "sharpe", "higher_is_better": True}), now, now))
    for cid, seq, score in (("lose", 1, -3.1), ("win", 2, 0.8)):
        O.db().execute("INSERT INTO candidates (id, objective_id, seq, created_at, model, status, is_score, metrics) "
                       "VALUES (?, 'o1', ?, ?, 'm', 'ok', ?, '{}')", (cid, seq, now, score))
    O.db().commit()
    yield
    O.db().close()


def test_works_needs_a_candidate_that_makes_money(db):
    """Agents wrote WORKS under -3.10 and -5.14: it ran, it did not work."""
    c = lambda **kw: L.Comment(text="evidence here", author="Qwen", **{"verdict": "works", **kw})  # noqa: E731
    assert L._earned_works(c(candidate_id="win")) is None
    assert "loses in-sample" in L._earned_works(c(candidate_id="lose"))
    assert "no candidate" in L._earned_works(c())
    assert L._earned_works(L.Comment(text="mine", verdict="works")) is None      # the operator's word stands
