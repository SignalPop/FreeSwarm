"""New ideas on a schedule: the first rung is asked regularly, stuck or not, without climbing
or resetting the stuck ladder -- and the cadences round-trip through Settings."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from app import escalation, external, mentor
from app import objectives as O


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "DB_PATH", tmp_path / "objectives.sqlite3")
    monkeypatch.setattr(O, "_conn", None)
    monkeypatch.setattr(O, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(external, "CONFIG_PATH", tmp_path / "external.json")
    monkeypatch.setattr(escalation, "_busy", set())
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


def add_candidate(*, created=None, champion_at=None):
    seq = next(_seq)
    O.db().execute(
        "INSERT INTO candidates (id, objective_id, seq, created_at, model, status, is_score, metrics, code, champion_at) "
        "VALUES (?, 'o1', ?, ?, 'm', 'ok', 0.0, '{}', 'x = 1\n', ?)",
        (f"c{seq}", seq, created or time.time(), champion_at))
    O.db().commit()


def add_idea(trigger: str, ago_s: float = 0.0) -> int:
    cur = O.db().execute(
        "INSERT INTO ideas (objective_id, ts, model, rung, text, candidates_at, trigger) VALUES ('o1',?,?,0,?,0,?)",
        (time.time() - ago_s, "DeepSeek", f"a {trigger} idea about volatility regimes", trigger))
    O.db().commit()
    return int(cur.lastrowid)


def cfg(**kw) -> dict:
    return {**external.DEFAULT_CONFIG["escalation"], **kw}


def age_objective(seconds: float) -> None:
    O.db().execute("UPDATE objectives SET created_at = created_at - ?", (seconds,))
    O.db().commit()


# ---------------------------------------------------------------------------------------
# periodic_due
# ---------------------------------------------------------------------------------------
def test_scheduled_due_after_enough_candidates(db):
    c = cfg(scheduled_candidates=3, scheduled_minutes=999)
    for _ in range(2):
        add_candidate()
    assert not escalation.periodic_due(escalation.scheduled_assess(O.get_objective("o1")), c)
    add_candidate()
    assert escalation.periodic_due(escalation.scheduled_assess(O.get_objective("o1")), c)


def test_scheduled_due_after_minutes_with_one_candidate(db):
    c = cfg(scheduled_candidates=99, scheduled_minutes=20)
    age_objective(21 * 60)
    s = escalation.scheduled_assess(O.get_objective("o1"))
    assert not escalation.periodic_due(s, c)  # nobody is working: no ideas for nobody
    add_candidate()
    assert escalation.periodic_due(escalation.scheduled_assess(O.get_objective("o1")), c)


def test_scheduled_clock_restarts_after_any_ladder_ask(db):
    c = cfg(scheduled_candidates=99, scheduled_minutes=20)
    age_objective(3600)
    add_candidate(created=time.time() - 1800)
    add_idea("stuck", ago_s=60)  # a stuck ask a minute ago
    add_candidate()
    assert not escalation.periodic_due(escalation.scheduled_assess(O.get_objective("o1")), c)
    O.db().execute("UPDATE ideas SET ts = ts - ?", (25 * 60,))
    O.db().commit()
    assert escalation.periodic_due(escalation.scheduled_assess(O.get_objective("o1")), c)


def test_mentor_ideas_do_not_restart_the_schedule(db):
    c = cfg(scheduled_candidates=99, scheduled_minutes=20)
    age_objective(3600)
    add_idea("mentor", ago_s=60)
    add_candidate()
    assert escalation.periodic_due(escalation.scheduled_assess(O.get_objective("o1")), c)


def test_scheduled_can_be_switched_off(db):
    age_objective(3600)
    for _ in range(50):
        add_candidate()
    s = escalation.scheduled_assess(O.get_objective("o1"))
    assert escalation.periodic_due(s, cfg())
    assert not escalation.periodic_due(s, cfg(scheduled=False))


# ---------------------------------------------------------------------------------------
# Kept out of the stuck climb, handed to agents
# ---------------------------------------------------------------------------------------
def test_scheduled_ideas_do_not_climb_or_reset_the_ladder(db):
    old = time.time() - 5000
    add_candidate(created=old, champion_at=old + 30)
    add_idea("scheduled")
    add_idea("mentor")
    assert escalation.assess(O.get_objective("o1"))["ideas"] == []
    stuck = add_idea("stuck")
    assert [i["id"] for i in escalation.assess(O.get_objective("o1"))["ideas"]] == [stuck]


def test_scheduled_ideas_reach_agents_while_fresh(db):
    iid = add_idea("scheduled")
    add_idea("scheduled", ago_s=escalation.MENTOR_IDEAS_FRESH_S + 60)  # too old
    ctx = escalation.ideas_for_context("o1")
    assert [i["id"] for i in ctx] == [iid] and ctx[0]["trigger"] == "scheduled"
    assert [i["id"] for i in escalation.regular_ideas("o1")] == [iid]


# ---------------------------------------------------------------------------------------
# escalate(trigger="scheduled") and the tick
# ---------------------------------------------------------------------------------------
LADDER = {"ladder": [{"model": "DeepSeek@lambda999"}, {"model": "claude@openrouter"}]}


@pytest.fixture
def asked(db, monkeypatch):
    calls: list[dict] = []

    async def complete(model, messages, max_tokens, purpose):
        calls.append({"model": model, "prompt": messages[0]["content"], "purpose": purpose})
        return "1. Size by inverse volatility at entry. 2. ... 3. ..."

    monkeypatch.setattr(escalation, "_complete", complete)
    monkeypatch.setattr(escalation, "_loaded", lambda: [])
    monkeypatch.setattr(escalation.swarm_policy, "plan", lambda project, loaded: LADDER)
    monkeypatch.setattr(O, "_ranked", lambda *a, **k: [])
    return calls


def test_scheduled_ask_goes_to_the_first_rung_without_the_reserve(asked):
    add_idea("stuck")  # the stuck climb is on rung 2 now -- a scheduled ask stays on rung 1
    got = asyncio.run(escalation.escalate(O.get_objective("o1"), {"id": "p1"}, trigger="scheduled"))
    assert got["rung"] == 0 and got["model"] == "DeepSeek@lambda999"
    [call] = asked
    assert not external.is_ideas(call["purpose"])  # a paid first rung stops where search stops
    assert "GENUINELY NEW" in call["prompt"] and "stuck in a local optimum" not in call["prompt"]
    assert "ft.inverse_vol" in call["prompt"] and "FALSIFY" in call["prompt"]
    row = O.db().execute("SELECT trigger FROM ideas WHERE id=?", (got["id"],)).fetchone()
    assert row[0] == "scheduled"


def test_stuck_ask_still_climbs_with_the_reserve(asked):
    add_idea("stuck")
    got = asyncio.run(escalation.escalate(O.get_objective("o1"), {"id": "p1"}))
    assert got["rung"] == 1 and external.is_ideas(asked[0]["purpose"])
    assert "stuck in a local optimum" in asked[0]["prompt"]


def _tick_with(monkeypatch, busy: set[str]) -> list[str]:
    triggers: list[str] = []

    async def escalate(obj, project, *, trigger="stuck"):
        assert obj["id"] in escalation._busy  # held while asking: no concurrent ask
        triggers.append(trigger)
        return {}

    async def list_objectives(pid, status):
        return {"objectives": [{"id": "o1"}]}

    monkeypatch.setattr(escalation, "escalate", escalate)
    monkeypatch.setattr(escalation, "_busy", busy)
    monkeypatch.setattr(escalation.projects, "list_projects", lambda: [{"id": "p1"}])
    monkeypatch.setattr(O, "list_objectives", list_objectives)
    asyncio.run(escalation._tick())
    return triggers


def test_tick_asks_on_schedule_and_skips_busy_objectives(db, monkeypatch):
    age_objective(3600)
    add_candidate()
    assert _tick_with(monkeypatch, set()) == ["scheduled"]
    assert escalation._busy == set()
    assert _tick_with(monkeypatch, {"o1"}) == []


def test_tick_prefers_stuck_over_scheduled(db, monkeypatch):
    age_objective(3 * 3600)
    for _ in range(external.DEFAULT_CONFIG["escalation"]["stuck_candidates"]):
        add_candidate()
    assert _tick_with(monkeypatch, set()) == ["stuck"]


# ---------------------------------------------------------------------------------------
# Settings round-trip
# ---------------------------------------------------------------------------------------
def test_config_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(external, "CONFIG_PATH", tmp_path / "external.json")

    async def overview():
        return external.config()

    monkeypatch.setattr(external, "overview", overview)
    cfg = external.config()
    assert cfg["escalation"]["scheduled"] is True and cfg["mentor"] == external.DEFAULT_CONFIG["mentor"]

    out = asyncio.run(external.write_config(external.ConfigReq(
        escalation={"scheduled": False, "scheduled_minutes": 0, "scheduled_candidates": "7", "bogus": 1},
        mentor={"every_candidates": 3, "every_minutes": 15, "other": 2})))
    esc = out["escalation"]
    assert esc["scheduled"] is False and esc["scheduled_minutes"] == 1 and esc["scheduled_candidates"] == 7
    assert "bogus" not in esc and esc["stuck_candidates"] == external.DEFAULT_CONFIG["escalation"]["stuck_candidates"]
    assert out["mentor"] == {"every_candidates": 3, "every_minutes": 15}

    saved = json.loads((tmp_path / "external.json").read_text(encoding="utf-8"))
    assert saved["escalation"]["scheduled"] is False and saved["mentor"]["every_minutes"] == 15
    assert external.config()["mentor"] == {"every_candidates": 3, "every_minutes": 15}
    assert mentor.cadence() == (3, 15)


def test_old_config_file_gets_the_new_defaults(tmp_path, monkeypatch):
    path = tmp_path / "external.json"
    path.write_text(json.dumps({"escalation": {"enabled": True, "stuck_candidates": 40}}), encoding="utf-8")
    monkeypatch.setattr(external, "CONFIG_PATH", path)
    cfg = external.config()
    assert cfg["escalation"]["scheduled_minutes"] == external.DEFAULT_CONFIG["escalation"]["scheduled_minutes"]
    assert cfg["mentor"] == external.DEFAULT_CONFIG["mentor"]
