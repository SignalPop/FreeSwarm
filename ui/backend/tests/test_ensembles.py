"""Ensembles: verified candidates combined into one weighted portfolio candidate.

The weights must be strictly causal (a day's weight reads only earlier returns), the combined
return must be the plain weighted sum of the members' stored net returns, correlations shown to
agents must come from in-sample dates only, and the code-running paths must refuse an ensemble.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import time

import numpy as np
import pytest
from fastapi import HTTPException

from app import ensembles as E
from app import objectives as O

SPLIT = "2024-07-01"


def days(n: int, start: str = "2024-01-01") -> list[str]:
    d0 = dt.date.fromisoformat(start)
    out, d = [], d0
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "DB_PATH", tmp_path / "objectives.sqlite3")
    monkeypatch.setattr(O, "_conn", None)
    monkeypatch.setattr(O, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(O, "_board_post", lambda *a, **k: None)
    now = time.time()
    O.db().execute(
        "INSERT INTO objectives (id, project_id, title, metric, split_date, created_at, updated_at) "
        "VALUES ('o1', 'p1', 'Best strategy', ?, ?, ?, ?)",
        (json.dumps({"kind": "sharpe", "higher_is_better": True, "min_active_days": 5, "rank": "robust"}),
         SPLIT, now, now))
    O.db().commit()
    yield O.db()
    O.db().close()


_seq = iter(range(1, 100_000))


def add(returns: list[list], *, status="ok", lookahead="pass", audit="none", mode="improve", code="x = 1\n") -> dict:
    seq = next(_seq)
    obj = O.get_objective("o1")
    score, is_score, note, metrics = O._score_returns(obj, returns) if returns else (None, None, "", {})
    O.db().execute(
        "INSERT INTO candidates (id, objective_id, seq, created_at, model, mode, rationale, code, status, score, "
        "is_score, metrics, returns, lookahead, audit) VALUES (?, 'o1', ?, ?, 'm', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (f"c{seq}", seq, time.time(), mode, f"strategy {seq}", code, status, score, is_score, json.dumps(metrics),
         json.dumps(returns), lookahead, audit))
    O.db().commit()
    return O.get_candidate(f"c{seq}")


def series(rng: np.random.Generator, n: int, mu: float = 0.001, sd: float = 0.01) -> list[list]:
    return [[d, float(x)] for d, x in zip(days(n), rng.normal(mu, sd, n))]


# ---------------------------------------------------------------------------------------
# Weights and the combined return (pure)
# ---------------------------------------------------------------------------------------
def test_appending_future_returns_never_changes_past_weights_or_returns():
    rng = np.random.default_rng(1)
    full = [series(rng, 120, sd=s) for s in (0.005, 0.01, 0.02)]
    short = [s[:80] for s in full]
    for weighting in ("equal", "inverse_vol"):
        d1, _, W1, r1 = E.combine(short, weighting, 20)
        d2, _, W2, r2 = E.combine(full, weighting, 20)
        assert d2[:80] == d1
        assert np.array_equal(W2[:80], W1)
        assert np.array_equal(r2[:80], r1)


def test_a_days_own_returns_do_not_change_its_weight():
    rng = np.random.default_rng(2)
    R = rng.normal(0, 0.01, (100, 3))
    seen = np.ones_like(R, dtype=bool)
    W = E.weights(R, seen, "inverse_vol", 20)
    R2 = R.copy()
    R2[50] = [0.3, -0.2, 0.25]                      # a wild day 50
    W2 = E.weights(R2, seen, "inverse_vol", 20)
    assert np.array_equal(W[:51], W2[:51])           # weights up to and including day 50 unchanged
    assert not np.array_equal(W[51], W2[51])         # day 51 is the first to see it


def test_weights_sum_to_one_and_equal_is_one_over_n():
    rng = np.random.default_rng(3)
    R = rng.normal(0, 0.01, (60, 4))
    seen = np.ones_like(R, dtype=bool)
    assert np.allclose(E.weights(R, seen, "equal", 20), 0.25)
    assert np.allclose(E.weights(R, seen, "inverse_vol", 20).sum(axis=1), 1.0)


def test_inverse_vol_gives_the_calmer_member_more_weight_after_warm_up():
    rng = np.random.default_rng(4)
    R = np.column_stack([rng.normal(0, 0.005, 80), rng.normal(0, 0.02, 80)])
    seen = np.ones_like(R, dtype=bool)
    W = E.weights(R, seen, "inverse_vol", 20)
    need = E.min_history(20)
    assert np.allclose(W[:need], 0.5)                # warm-up: equal weights
    assert (W[need:, 0] > W[need:, 1]).all()
    assert W[40:, 0].mean() > 0.7                    # ~ 0.02 / (0.005 + 0.02) = 0.8


def test_a_member_with_no_variance_keeps_the_equal_share():
    rng = np.random.default_rng(5)
    R = np.column_stack([np.zeros(40), rng.normal(0, 0.01, 40), rng.normal(0, 0.02, 40)])
    W = E.weights(R, np.ones_like(R, dtype=bool), "inverse_vol", 10)
    assert np.allclose(W[20:, 0], 1 / 3)
    assert np.allclose(W.sum(axis=1), 1.0)


def test_combined_return_is_the_weighted_sum_and_a_missing_day_counts_as_zero():
    a = [["2024-01-02", 0.02], ["2024-01-03", -0.01], ["2024-01-04", 0.03]]
    b = [["2024-01-02", 0.00], ["2024-01-04", 0.01]]              # no return on the 3rd
    dates, R, W, r = E.combine([a, b], "equal", 20)
    assert dates == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert np.allclose(r, [0.01, -0.005, 0.02])


# ---------------------------------------------------------------------------------------
# Creating ensembles
# ---------------------------------------------------------------------------------------
def test_an_ensemble_is_scored_with_score_returns_and_settled_like_any_candidate(db, monkeypatch):
    rng = np.random.default_rng(6)
    a, b = add(series(rng, 250)), add(series(rng, 250))
    calls = []
    real = O._score_returns

    def spy(obj, returns):
        calls.append(returns)
        return real(obj, returns)

    monkeypatch.setattr(O, "_score_returns", spy)
    settled = []
    real_settle = O._settle
    monkeypatch.setattr(O, "_settle", lambda *a_, **k: settled.append(a_[4]) or real_settle(*a_, **k))
    out = E.create(O.get_objective("o1"), [a["seq"], f"#{b['seq']}"], "equal", 20, "", "tester")
    c = O.get_candidate(out["id"])
    assert c["mode"] == "ensemble" and c["status"] == "ok" and c["lookahead"] == "pass"
    assert calls and calls[-1] == c["returns"]
    expect = real(O.get_objective("o1"), c["returns"])
    assert c["score"] == expect[0] and c["is_score"] == expect[1]
    assert settled and settled[0]["lookahead"] == "pass"
    assert "costs" not in c["metrics"]
    assert c["audit"] in ("pending", "none")               # contender -> audit (require_audit is on)
    ens = c["metrics"]["ensemble"]
    assert [m["seq"] for m in ens["members"]] == [a["seq"], b["seq"]]
    assert set(ens["member_returns"]) == {str(a["seq"]), str(b["seq"])}
    assert "#" + str(a["seq"]) in c["code"] and c["code"].lstrip().startswith("#")


def test_the_agent_sees_no_holdout(db):
    rng = np.random.default_rng(7)
    a, b = add(series(rng, 250)), add(series(rng, 250))
    out = E.create(O.get_objective("o1"), [a["seq"], b["seq"]], "inverse_vol", 20, "diversify", "agent")
    text = json.dumps(out)
    assert "holdout_sharpe" not in text and "member_returns" not in text and '"holdout"' not in text
    assert out["ensemble"]["members"][0]["avg_weight_in_sample"] is not None


@pytest.mark.parametrize("kw,why", [
    ({"lookahead": "pending"}, "look-ahead"),
    ({"lookahead": "fail"}, "look-ahead"),
    ({"audit": "fail"}, "audit"),
    ({"status": "error"}, "error"),
    ({"mode": "ensemble"}, "ensemble"),
])
def test_ineligible_members_are_rejected(db, kw, why):
    rng = np.random.default_rng(8)
    good, bad = add(series(rng, 250)), add(series(rng, 250), **kw)
    with pytest.raises(HTTPException) as exc:
        E.create(O.get_objective("o1"), [good["seq"], bad["seq"]], "equal", 20, "", "t")
    assert exc.value.status_code == 400 and why in exc.value.detail and f"#{bad['seq']}" in exc.value.detail


def test_member_count_and_identical_ensembles_are_rejected(db):
    rng = np.random.default_rng(9)
    cs = [add(series(rng, 250)) for _ in range(9)]
    obj = O.get_objective("o1")
    with pytest.raises(HTTPException) as exc:
        E.create(obj, [cs[0]["seq"], cs[0]["id"]], "equal", 20, "", "t")      # one distinct member
    assert exc.value.status_code == 400
    with pytest.raises(HTTPException) as exc:
        E.create(obj, [c["seq"] for c in cs], "equal", 20, "", "t")           # nine
    assert exc.value.status_code == 400
    with pytest.raises(HTTPException):
        E.create(obj, [cs[0]["seq"], 99999], "equal", 20, "", "t")            # unknown
    E.create(obj, [cs[0]["seq"], cs[1]["seq"]], "equal", 20, "", "t")
    with pytest.raises(HTTPException) as exc:
        E.create(obj, [cs[1]["seq"], cs[0]["seq"]], "equal", 20, "", "t")
    assert exc.value.status_code == 409


def test_correlations_use_no_dates_on_or_after_the_split(db):
    ds = days(250)
    rng = np.random.default_rng(10)
    x = rng.normal(0, 0.01, 250)
    ins = [d < SPLIT for d in ds]
    # Identical before the split, exact opposites after it.
    a = add([[d, float(v)] for d, v in zip(ds, x)])
    b = add([[d, float(v if i else -v)] for d, v, i in zip(ds, x, ins)])
    out = asyncio.run(E.correlations("o1", seqs=f"{a['seq']},{b['seq']}"))
    assert out["days"] == sum(ins)
    assert out["matrix"][0][1] == pytest.approx(1.0)
    ens = E.create(O.get_objective("o1"), [a["seq"], b["seq"]], "equal", 20, "", "t")
    stored = O.get_candidate(ens["id"])["metrics"]["ensemble"]["correlation_in_sample"]
    assert stored[0][1] == pytest.approx(1.0)


def test_correlation_suggestions_pick_the_uncorrelated_pair(db):
    rng = np.random.default_rng(11)
    ds = days(250)
    base = rng.normal(0.002, 0.01, 250)
    a = add([[d, float(v)] for d, v in zip(ds, base)])
    b = add([[d, float(v + rng.normal(0, 0.001))] for d, v in zip(ds, base)])       # a clone of a
    c = add([[d, float(v)] for d, v in zip(ds, rng.normal(0.002, 0.01, 250))])      # independent
    out = asyncio.run(E.correlations("o1", seqs=f"{a['seq']},{b['seq']},{c['seq']}"))
    best = out["suggestions"][0]
    assert c["seq"] in best["seqs"] and not {a["seq"], b["seq"]} <= set(best["seqs"])
    assert best["avg_abs_rho"] < 0.3


# ---------------------------------------------------------------------------------------
# Guards on the code-running paths, the audit, and disqualification
# ---------------------------------------------------------------------------------------
def _ensemble(db) -> dict:
    rng = np.random.default_rng(12)
    a, b = add(series(rng, 250)), add(series(rng, 250), code="import ft\nMEMBER_B = 1\n")
    out = E.create(O.get_objective("o1"), [a["seq"], b["seq"]], "equal", 20, "", "t")
    return O.get_candidate(out["id"])


def test_rerun_and_run_refuse_an_ensemble(db):
    e = _ensemble(db)
    O._update_candidate(e["id"], {"score": None})          # would otherwise qualify for a re-run
    for fn in (O.rerun, O.run_candidate):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(fn("o1", e["id"]))
        assert exc.value.status_code == 409


def test_retest_skips_ensembles(db, monkeypatch):
    e = _ensemble(db)
    queued = []

    async def fake_retest(obj, ids):
        queued.extend(ids)

    monkeypatch.setattr(O, "_retest", fake_retest)

    async def go():
        st = await O.start_retest("o1", O.Retest(top=25))
        await asyncio.sleep(0)
        return st

    st = asyncio.run(go())
    assert e["id"] not in queued and e["seq"] not in st["queue"]
    assert [s["seq"] for s in st["skipped"]] == [e["seq"]]
    assert len(queued) == 2


def test_retest_loop_itself_skips_an_ensemble(db, monkeypatch):
    e = _ensemble(db)
    monkeypatch.setattr(O.projects, "get", lambda pid: {"data_dir": ""})
    monkeypatch.setattr(O.datasource, "catalog", lambda d: [])

    async def boom(*a, **k):
        raise AssertionError("an ensemble must not be run")

    monkeypatch.setattr(O, "_run_forecasting", boom)
    O._retests["o1"] = {"results": [], "progress": {}, "cancel": False}
    asyncio.run(O._retest(O.get_objective("o1"), [e["id"]]))
    assert O._retests["o1"]["skipped"][0]["seq"] == e["seq"]


def test_the_audit_prompt_shows_the_spec_and_every_members_code(db):
    e = _ensemble(db)
    p = O._audit_prompt(O.get_objective("o1"), e)
    assert "ENSEMBLE of" in p and "MEMBER_B = 1" in p and "x = 1" in p


def test_disqualifying_a_member_takes_its_ensembles_down(db):
    e = _ensemble(db)
    member = e["metrics"]["ensemble"]["members"][0]
    O.db().execute("UPDATE objectives SET best_id=? WHERE id='o1'", (e["id"],))
    O.db().commit()
    hit = E.disqualify_dependents(O.get_objective("o1"), ids={member["id"]}, why="leak")
    assert hit == [e["seq"]]
    assert O.get_candidate(e["id"])["audit"] == "fail"
    assert O.get_objective("o1")["best_id"] != e["id"]
