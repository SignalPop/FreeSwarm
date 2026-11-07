"""ft.candidate_positions: which numbers reach a run, what a refusal says, and that a refused
number is never swapped for another one (#497)."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

from app import monitor as M
from app import objectives as O

BACKEND = Path(__file__).resolve().parents[1]
FT_PATH = BACKEND.parent / "sandbox" / "ft.py"
RUNNER = BACKEND / "swarm_runner.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def runner():
    return _load("swarm_runner_members_test", RUNNER)


@pytest.fixture
def ft(monkeypatch, tmp_path):
    mod = _load("ft_members_test", FT_PATH)
    monkeypatch.setattr(mod, "_FT", str(tmp_path))
    return mod


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "DB_PATH", tmp_path / "objectives.sqlite3")
    monkeypatch.setattr(O, "_conn", None)
    now = time.time()
    O.db().execute("INSERT INTO objectives (id, project_id, title, metric, created_at, updated_at) "
                   "VALUES ('o1', 'p1', 't', '{}', ?, ?)", (now, now))
    O.db().commit()


def _cand(seq, code="print('x')", status="ok", mode="improve", lookahead="pass", audit="none", notes=""):
    O.db().execute("INSERT INTO candidates (id, objective_id, seq, created_at, mode, code, status, lookahead, audit, "
                   "audit_notes) VALUES (?, 'o1', ?, 0, ?, ?, ?, ?, ?, ?)",
                   (f"c{seq}", seq, mode, code, status, lookahead, audit, notes))
    O.db().commit()


def _members(files: dict) -> list[int]:
    return sorted(int(k.rsplit("/", 1)[1][:-3]) for k in files if k.endswith(".py"))


LOOP = "for c in [2196, 2235]:\n    p = ft.candidate_positions(c)\n    print(p.tail(390))\n"


# ---------------------------------------------------------------------------------------
# What reaches a run
# ---------------------------------------------------------------------------------------
def test_an_experiment_gets_the_candidates_it_loops_over(db):
    """10-17, qwen: #2196 and #1579 ran when named; the same two in `for c in [...]` were refused."""
    _cand(2196), _cand(2235), _cand(390)
    files = O.member_files("o1", LOOP, loose=True)
    assert {2196, 2235} <= set(_members(files))


def test_a_scored_script_must_still_name_its_members(db):
    """disqualify_dependents reads member_seqs from the text: a scored script that hid its members
    in a variable would outlive their disqualification."""
    _cand(2196), _cand(2235)
    assert _members(O.member_files("o1", LOOP)) == []
    assert O.member_seqs(LOOP) == []


def test_the_loose_scan_applies_only_when_a_call_passes_a_variable(db):
    _cand(7), _cand(390)
    code = "p = ft.candidate_positions(7)\nprint(p.tail(390))\n"
    assert _members(O.member_files("o1", code, loose=True)) == [7]


def test_the_number_scan_reads_whole_numbers_only():
    code = "ids = ['#2196', 2235]\nx = 0.5 + 1e3 + a12 + 3.0\nfor c in ids:\n    ft.candidate_positions(c)\n"
    assert O._numbers_behind_a_variable(code) == [2196, 2235]


def test_window_lengths_do_not_use_up_the_room_for_real_members(db):
    numbers = ", ".join(str(n) for n in range(1, 41))
    _cand(5000)
    code = f"w = [{numbers}]\nfor c in [5000]:\n    ft.candidate_positions(c)\n"
    assert 5000 in _members(O.member_files("o1", code, loose=True))


def test_members_of_a_member_still_come_along(db):
    _cand(1), _cand(2, code="ft.candidate_positions(1)")
    assert _members(O.member_files("o1", "for c in [2]:\n    ft.candidate_positions(c)\n", loose=True)) == [1, 2]


@pytest.mark.parametrize("kw,says", [
    (dict(status="error"), "it has no score (status: error)"),
    (dict(mode="ensemble"), "it is an ensemble"),
    (dict(lookahead="fail"), "look-ahead test: fail"),
    (dict(audit="fail", notes="[auto] Disqualified with the family: quarantined"),
     "it was disqualified (Disqualified with the family: quarantined)"),
])
def test_a_refused_number_comes_with_its_reason(db, kw, says):
    _cand(9, **kw)
    files = O.member_files("o1", "ft.candidate_positions(9)")
    assert _members(files) == []
    assert says in json.loads(files[".ft/members/refused.json"])["9"]


def test_a_number_nobody_has_is_said_to_be_that(db):
    assert "no candidate" in json.loads(O.member_files("o1", "ft.candidate_positions(9)")[".ft/members/refused.json"])["9"]


def test_a_script_with_good_members_writes_no_refusals(db):
    _cand(9)
    assert set(O.member_files("o1", "ft.candidate_positions(9)")) == {".ft/members/9.py"}


# ---------------------------------------------------------------------------------------
# What the agent reads
# ---------------------------------------------------------------------------------------
def test_ft_says_why_when_the_control_plane_gave_a_reason(ft, tmp_path):
    (tmp_path / "members").mkdir()
    (tmp_path / "members" / "refused.json").write_text(json.dumps({"2561": "it was disqualified (a leak)"}), encoding="utf-8")
    with pytest.raises(FileNotFoundError) as exc:
        ft.candidate_positions(2561)
    assert "candidate #2561 cannot be used: it was disqualified (a leak)" in str(exc.value)


def test_ft_without_a_reason_says_to_write_the_number_out(ft):
    """Before a control plane restart, and for a number computed at run time."""
    with pytest.raises(FileNotFoundError) as exc:
        ft.candidate_positions(2196)
    assert "ft.candidate_positions(2196)" in str(exc.value)


def test_no_refusal_offers_a_number_to_copy(ft, tmp_path):
    (tmp_path / "members").mkdir()
    (tmp_path / "members" / "refused.json").write_text(json.dumps({"5": "it was disqualified"}), encoding="utf-8")
    for seq in (5, 6):
        with pytest.raises(FileNotFoundError) as exc:
            ft.candidate_positions(seq)
        assert not [n for n in O._WHOLE_NUMBER.findall(str(exc.value)) if int(n) != seq]


# ---------------------------------------------------------------------------------------
# The repair must not answer a different question
# ---------------------------------------------------------------------------------------
def _refused(seq: int, verb: str = "cannot be used: it was disqualified") -> dict:
    return {"ok": False, "stdout": "", "artifacts": [], "duration_s": 1.0,
            "stderr": "Traceback (most recent call last):\n  File \"/work/.ft/ft.py\", line 2589, in candidate_positions\n"
                      f"FileNotFoundError: candidate #{seq} {verb}\n"}


def test_a_candidate_refused_by_name_is_not_repaired(runner):
    """10-19, Muse: #2561 came back "repaired automatically" with #1233's positions."""
    code = "import ft\npos = ft.candidate_positions(2561)\nprint(pos.head())\n"
    asked = []
    out = runner._auto_repair("run_python", code, _refused(2561), lambda *a: asked.append(a) or code.replace("2561", "1233"),
                              lambda c: pytest.fail(f"ran {c}"))
    assert asked == [] and out["ok"] is False and "auto_repaired" not in out


def test_the_old_wording_of_the_refusal_is_caught_too(runner):
    """ft.py is live at once; a runner restarted later meets either text."""
    code = "pos = ft.candidate_positions(2561)\n"
    crash = _refused(2561, "is not available: only verified candidates of this objective can be used")
    asked = []
    runner._auto_repair("run_python", code, crash, lambda *a: asked.append(a), lambda c: pytest.fail(f"ran {c}"))
    assert asked == []


def test_a_repair_may_write_out_a_number_the_script_already_had(runner):
    """A submitted loop: naming its members in the calls is the real fix."""
    code = "for c in [2196]:\n    p = ft.candidate_positions(c)\n"
    fixed = "p = ft.candidate_positions(2196)\nprint(p)\n"
    out = runner._auto_repair("run_python", code, _refused(2196, "is not available: spell it out"), lambda *a: fixed,
                              lambda c: {"ok": True, "stdout": "", "stderr": ""})
    assert out["code_ran"] == fixed


def test_a_repair_may_not_bring_in_a_candidate_the_script_never_named(runner):
    code = "for c in [2196]:\n    p = ft.candidate_positions(c)\n"
    swapped = "p = ft.candidate_positions(1233)\nprint(p)\n"
    out = runner._auto_repair("run_python", code, _refused(2196, "is not available: spell it out"), lambda *a: swapped,
                              lambda c: pytest.fail(f"ran {c}"))
    assert "different candidate (#1233)" in out["auto_repair_failed"]


def test_a_repair_of_an_unrelated_crash_cannot_swap_a_member_either(runner):
    code = "p = ft.candidate_positions(7)\nprint(zz)\n"
    crash = {"ok": False, "stdout": "", "artifacts": [], "duration_s": 1.0,
             "stderr": "Traceback (most recent call last):\n  File \"candidate.py\", line 2\nNameError: name 'zz' is not defined\n"}
    out = runner._auto_repair("run_python", code, crash, lambda *a: "p = ft.candidate_positions(8)\nprint(p)\n",
                              lambda c: pytest.fail(f"ran {c}"))
    assert "different candidate (#8)" in out["auto_repair_failed"]


def test_get_candidate_says_a_candidate_was_disqualified(runner, monkeypatch):
    """#2561 read "status ok, look-ahead pass, problem ''" while ft refused it."""
    full = {"id": "c1", "seq": 2561, "model": "m", "rationale": "r", "code": "x", "status": "ok", "metrics": {},
            "lookahead": "pass", "score_note": "", "audit": "fail", "audit_notes": "[auto] Disqualified with the family"}
    monkeypatch.setattr(runner, "request", lambda *a, **k: full if "/candidates/" in a[1] else {"files": [], "docs": []})
    monkeypatch.setattr(runner, "_mcp_tools", lambda pid: [])
    world = runner.ObjectiveWorld({"id": "p1", "name": "P"}, "me", [], [],
                                  {"id": "o1", "metric": {"kind": "sharpe"}}, on_submit=lambda a: None)
    monkeypatch.setattr(world, "_candidate", lambda ref, across=False: ({"id": "c1", "_note": ""}, None))
    out = world.call("get_candidate", {"candidate": "2561"})
    assert out["disqualified"].startswith("[auto] Disqualified")


def test_get_candidate_with_a_library_module_name_gives_the_module(runner, monkeypatch):
    """11-05: get_candidate('dual_morning_slot_iv_gate') -- 60+ calls, each "no candidate ...": agents meant to
    read the module. They get it, with a note; a name that is neither still gets the miss."""
    mod = {"name": "dual_morning_slot_iv_gate", "kind": "signal", "shown_version": 1, "version": 1,
           "status": "active", "description": "d", "code": "def signal(df): ...", "evidence": {}}

    def request(base, path, *a, **k):
        if path.endswith("/library/dual_morning_slot_iv_gate"):
            return mod
        if "/library/" in path:
            raise RuntimeError(f"{path} -> 404: no module")
        return {"files": [], "docs": []}

    monkeypatch.setattr(runner, "request", request)
    monkeypatch.setattr(runner, "_mcp_tools", lambda pid: [])
    world = runner.ObjectiveWorld({"id": "p1", "name": "P"}, "me", [], [],
                                  {"id": "o1", "metric": {"kind": "sharpe"}}, on_submit=lambda a: None)
    miss = {"error": "no candidate 'x' in this objective -- candidate numbers here run up to #337"}
    monkeypatch.setattr(world, "_candidate", lambda ref, across=False: (None, miss))
    out = world.call("get_candidate", {"candidate": "dual_morning_slot_iv_gate"})
    assert out["code"] == "def signal(df): ..." and "is a library module, not a candidate" in out["note"]
    assert world.call("get_candidate", {"candidate": "not_a_module"}) == miss
    assert world.call("get_candidate", {"candidate": "#9999"}) == miss


# ---------------------------------------------------------------------------------------
# The monitor
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "error candidate #2561 is not available: only verified candidates of this objective can be used",
    "ERR: candidate #2196 cannot be used: it was disqualified (a leak)",
])
def test_a_refusal_a_script_caught_and_printed_is_not_platform_evidence(text):
    rec = {"timeline": [{"kind": "tool", "name": "run_python", "ok": True, "args": {"code": "print(1)"},
                         "result": json.dumps({"ok": True, "stdout": text + "\n", "stderr": ""})}]}
    assert not M._platform_evidence(text, {"agent": "m"}, rec)
