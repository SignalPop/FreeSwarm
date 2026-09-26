"""Ensembles: several verified strategies run side by side as one portfolio, weighted and sized.

The ranking rewards a SMOOTH equity curve (the weaker period's metric times the R^2 of log
equity -- see ``objectives._robust``). One strategy's curve is as jagged as its edge; two whose
daily returns are nearly uncorrelated wobble at different times, so their sum is steadier than
either. An ensemble makes that a first-class candidate instead of something every agent has to
re-implement inside one script.

**What an ensemble is.** A candidate row with ``mode='ensemble'``. Nothing is run: its daily
return is a weighted sum of its members' STORED net-of-cost daily returns,

    r_t = sum_i w_{i,t} * r_{i,t}

-- a portfolio of separately run strategies. Positions are not netted across members (two
members long and short the same bar both pay their own costs), and each member's costs are
exactly the ones it paid when it was scored. A member with no return on a date counts as 0.

**Weights are strictly causal.** ``equal``: w_i = 1/n. ``inverse_vol``: w_{i,t} is
proportional to 1/sigma_i, the sample std (ddof 1) of member i's daily returns over the
``lookback_days`` trading days strictly BEFORE t -- day t's own return never enters its own
weight -- normalised so the weights sum to 1. Until every member has at least
max(5, lookback_days // 2) reported days in that window the day falls back to equal weights,
and a member whose sigma is zero or undefined keeps the equal share 1/n (the rest of the budget
is split by inverse volatility among the others).

**Checks.** Members must already be verified: scored ok, a clean look-ahead pass, not failed an
audit, not themselves an ensemble, 2..8 distinct, same objective. Each member's decisions were
proven causal by the look-ahead test and the weights read only earlier returns, so the ensemble
is recorded as passing it. The combined series goes through ``objectives._score_returns`` exactly
like any candidate, then ``objectives._settle`` (contender -> audit -> crown), so the hidden
holdout, the robust score and the title logic are unchanged. There is no ``costs`` breakdown:
costs are already inside the member returns.

**What agents see.** In-sample only, like everything else: the correlation matrix and the
suggestions come from dates before the split, and the combine result is ``agent_view`` (in-sample
metrics) plus in-sample average weights. The operator's console shows the whole period.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import objectives as O

logger = logging.getLogger("freetoken.ensembles")

router = APIRouter(tags=["ensembles"])

MIN_MEMBERS = 2
MAX_MEMBERS = 8
DEFAULT_LOOKBACK = 20
WEIGHTINGS = ("equal", "inverse_vol")
# The audit prompt truncates the code block at 16k characters; member code shares what is
# left after the spec, so every member is shown rather than the first one in full.
AUDIT_CODE_BUDGET = 15_000
# Correlation suggestions: pool size and how many sets come back.
MAX_POOL = 40
SUGGESTIONS = 6


# =======================================================================================
# The arithmetic (pure: tested directly)
# =======================================================================================
def grid(series: list[list[list]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Every member's returns on one date grid (the union of their dates, sorted).

    Returns (dates, R, observed): R[t, i] is member i's return on dates[t] (0 when it reported
    none) and observed[t, i] says whether it did."""
    dates = sorted({str(d)[:10] for s in series for d, _ in s})
    at = {d: k for k, d in enumerate(dates)}
    R = np.zeros((len(dates), len(series)))
    seen = np.zeros((len(dates), len(series)), dtype=bool)
    for i, s in enumerate(series):
        for d, r in s:
            k = at[str(d)[:10]]
            R[k, i] = float(r)
            seen[k, i] = True
    return dates, R, seen


def min_history(lookback: int) -> int:
    """Reported days a member needs inside the window before inverse-vol weighting applies."""
    return max(5, lookback // 2)


def weights(R: np.ndarray, observed: np.ndarray, weighting: str, lookback: int) -> np.ndarray:
    """Daily weights, one row per date, each row summing to 1. Row t reads R[:t] only.

    equal: 1/n every day. inverse_vol: 1/sigma_i over the `lookback` grid days strictly before
    t (ddof 1), normalised. A day on which any member has fewer than `min_history(lookback)`
    reported days in that window is an equal-weight day; a member whose sigma is zero or
    undefined gets 1/n and the others share the rest in proportion to 1/sigma."""
    T, n = R.shape
    W = np.full((T, n), 1.0 / n)
    if weighting == "equal" or n == 0:
        return W
    need = min_history(lookback)
    for t in range(T):
        lo = max(0, t - lookback)
        if t - lo < 2 or (observed[lo:t].sum(axis=0) < need).any():
            continue                                    # warm-up: equal weights
        sd = R[lo:t].std(axis=0, ddof=1)
        ok = np.isfinite(sd) & (sd > 1e-12)
        if not ok.any():
            continue
        inv = np.where(ok, 1.0 / np.where(ok, sd, 1.0), 0.0)
        share = 1.0 - (n - int(ok.sum())) / n           # what is left after the equal shares
        W[t] = np.where(ok, inv / inv.sum() * share, 1.0 / n)
    return W


def combine(series: list[list[list]], weighting: str, lookback: int) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """(dates, per-member returns R, weights W, combined r) with r_t = sum_i W[t,i] * R[t,i]."""
    dates, R, seen = grid(series)
    W = weights(R, seen, weighting, lookback)
    return dates, R, W, (W * R).sum(axis=1)


def correlation(R: np.ndarray) -> list[list[float | None]]:
    """Pearson correlation of the columns; None where a column has no variance."""
    n = R.shape[1]
    out: list[list[float | None]] = [[None] * n for _ in range(n)]
    if R.shape[0] < 3:
        return out
    X = R - R.mean(axis=0)
    sd = np.sqrt((X * X).sum(axis=0))
    for i in range(n):
        for j in range(n):
            if sd[i] > 1e-15 and sd[j] > 1e-15:
                out[i][j] = round(float((X[:, i] * X[:, j]).sum() / (sd[i] * sd[j])), 4)
    return out


def in_sample(series: list[list[list]], split: str | None) -> list[list[list]]:
    """Each series cut to the dates BEFORE the split -- the only part agents may see."""
    return [[[d, r] for d, r in s if not split or str(d)[:10] < split] for s in series]


def _avg_abs(C: list[list[float | None]], idx: list[int]) -> float | None:
    vals = [abs(C[a][b]) for k, a in enumerate(idx) for b in idx[k + 1:] if C[a][b] is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def suggest(C: list[list[float | None]], eligible: list[int], limit: int = SUGGESTIONS) -> list[list[int]]:
    """Greedy low-correlation sets of 2..4 among `eligible` column indices: start from every
    pair, keep adding the member that keeps the average |rho| lowest. Best (lowest) first."""
    seen: set[tuple[int, ...]] = set()
    found: list[tuple[float, list[int]]] = []
    pairs = sorted(((abs(C[a][b]), a, b) for k, a in enumerate(eligible) for b in eligible[k + 1:]
                    if C[a][b] is not None), key=lambda x: x[0])
    for _, a, b in pairs[:12]:
        cur = [a, b]
        while True:
            key = tuple(sorted(cur))
            if key not in seen:
                seen.add(key)
                found.append((_avg_abs(C, cur) or 0.0, list(key)))
            if len(cur) >= 4:
                break
            rest = [(_avg_abs(C, cur + [c]), c) for c in eligible if c not in cur]
            rest = [(v, c) for v, c in rest if v is not None]
            if not rest:
                break
            cur = cur + [min(rest)[1]]
    found.sort(key=lambda x: (x[0], -len(x[1])))
    return [s for _, s in found[:limit]]


# =======================================================================================
# Candidates
# =======================================================================================
def is_ensemble(c: dict) -> bool:
    return (c.get("mode") or "") == "ensemble"


def _returns(c: dict) -> list[list]:
    r = c.get("returns")
    return json.loads(r or "[]") if isinstance(r, str) else (r or [])


def _resolve(oid: str, ref: Any) -> dict:
    """A member by number (12, "12", "#12") or id, within this objective."""
    s = str(ref).strip().lstrip("#").strip()
    with O._lock:
        row = O.db().execute("SELECT id FROM candidates WHERE objective_id=? AND (id=? OR CAST(seq AS TEXT)=?)",
                             (oid, s, s)).fetchone()
    if row is None:
        raise HTTPException(status_code=400, detail=f"no candidate {ref!r} in this objective")
    return O.get_candidate(row["id"])


def ineligible(c: dict) -> str | None:
    """Why a candidate cannot be an ensemble member, or None if it can."""
    if is_ensemble(c):
        return "it is an ensemble itself (ensembles do not nest)"
    if c["status"] != "ok":
        return f"its evaluation is {c['status']}"
    if c.get("lookahead") != "pass":
        return f"its look-ahead verdict is {c.get('lookahead')!r} (members must have passed it)"
    if c.get("audit") == "fail":
        return "it failed its audit (disqualified)"
    if len(_returns(c)) < 10:
        return "it has no daily return stream"
    return None


def audit_body(c: dict) -> str:
    """What an auditor reads for an ensemble: its spec, then each member's code."""
    ens = (c.get("metrics") or {}).get("ensemble") or {}
    spec = (c.get("code") or "").strip()
    members = ens.get("members") or []
    parts = [spec, "",
             "# AUDITOR NOTE: this candidate runs no code of its own. Its daily return is the weighted sum",
             "# of the members' stored net-of-cost daily returns (weights from returns strictly before each",
             "# day). Every member already passed the look-ahead test. Review the MEMBERS' code below for the",
             "# defects on the checklist; a defect in any member is a defect of the ensemble."]
    budget = max(1500, (AUDIT_CODE_BUDGET - len("\n".join(parts))) // max(1, len(members)))
    for m in members:
        try:
            code = O.get_candidate(m["id"]).get("code") or ""
        except HTTPException:
            code = "# (member deleted -- its code is no longer available)"
        cut = code if len(code) <= budget else code[:budget] + f"\n# ... truncated ({len(code) - budget} more characters)"
        parts += ["", f"# ===== member #{m.get('seq')} ({m.get('model')}) =====", cut]
    return "\n".join(parts)


def _spec(members: list[dict], weighting: str, lookback: int) -> str:
    """The human-readable `code` of an ensemble: nothing here runs."""
    lines = [f"# ENSEMBLE of {' + '.join('#' + str(m['seq']) for m in members)}",
             "# A portfolio of separately run strategies: each member's stored net-of-cost daily returns,",
             "# weighted daily. No netting of positions; each member's costs as it paid them. Nothing here runs.",
             f"# weighting: {weighting}" + (f", lookback {lookback} trading days strictly before each day "
                                            f"(equal weights until {min_history(lookback)} reported days)"
                                            if weighting == "inverse_vol" else " (1/n each)"),
             "# r_t = sum_i w_{i,t} * r_{i,t}   (a missing member return counts as 0)",
             "# members:"]
    lines += [f"#   #{m['seq']:<6} {(m.get('model') or '?')[:40]:<40} {m.get('rationale', '')[:120]}" for m in members]
    return "\n".join(lines) + "\n"


def _mean_weights(W: np.ndarray, dates: list[str], split: str | None) -> list[float]:
    """Average weight per member over the in-sample dates (all dates without a split)."""
    rows = [k for k, d in enumerate(dates) if not split or d < split]
    if not rows:
        return []
    return [round(float(x), 4) for x in W[rows].mean(axis=0)]


def build(obj: dict, members: list[dict], weighting: str, lookback: int) -> tuple[list[list], dict]:
    """The combined return series and metrics["ensemble"] for these members (no database writes)."""
    ppy = float(obj["metric"].get("periods_per_year") or 252)
    split = obj.get("split_date")
    series = [_returns(c) for c in members]
    dates, R, W, r = combine(series, weighting, lookback)
    combined = [[d, float(x)] for d, x in zip(dates, r)]
    ins = in_sample(series, split)
    _, R_in, _ = grid(ins)
    stats = []
    for c, s in zip(members, series):
        st_in = O._stats([x for d, x in s if not split or d < split], ppy)
        st_ho = O._stats([x for d, x in s if split and d >= split], ppy) if split else {}
        stats.append({"seq": c["seq"], "in_sample_sharpe": st_in.get("sharpe"), "holdout_sharpe": st_ho.get("sharpe"),
                      "in_sample_score": c.get("is_score")})
    ens = {
        "members": [{"id": c["id"], "seq": c["seq"], "model": c.get("model"),
                     "rationale": (c.get("rationale") or "")[:200]} for c in members],
        "weighting": weighting,
        "lookback_days": lookback,
        "weights": [[d, [round(float(w), 5) for w in row]] for d, row in zip(dates, W)],
        "avg_weights": [round(float(x), 4) for x in W.mean(axis=0)],
        "avg_weights_in_sample": _mean_weights(W, dates, split),
        "member_returns": {str(c["seq"]): [[d, round(float(x), 10)] for d, x in s] for c, s in zip(members, series)},
        "member_stats": stats,
        "correlation_in_sample": correlation(R_in),
        "members_note": ("A portfolio of separately run strategies: each member's stored net-of-cost daily "
                         "returns, weighted daily, with no netting of positions; a missing member return "
                         "counts as 0. Weights use only returns before each day."),
    }
    return combined, ens


def _existing(oid: str, ids: list[str], weighting: str, lookback: int) -> int | None:
    """The seq of an identical ensemble already on record (same members, weighting, lookback)."""
    want = sorted(ids)
    with O._lock:
        rows = O.db().execute("SELECT seq, metrics FROM candidates WHERE objective_id=? AND mode='ensemble'",
                              (oid,)).fetchall()
    for row in rows:
        e = (json.loads(row["metrics"] or "{}") or {}).get("ensemble") or {}
        if (sorted(m["id"] for m in e.get("members") or []) == want and e.get("weighting") == weighting
                and (weighting == "equal" or e.get("lookback_days") == lookback)):
            return row["seq"]
    return None


def _norm_weighting(w: str | None) -> str:
    s = (w or "equal").strip().lower().replace("-", "_").replace(" ", "_")
    s = {"inverse_volatility": "inverse_vol", "invvol": "inverse_vol", "inv_vol": "inverse_vol",
         "equal_weight": "equal", "equal_weighted": "equal"}.get(s, s)
    if s not in WEIGHTINGS:
        raise HTTPException(status_code=400, detail=f"weighting must be one of {', '.join(WEIGHTINGS)}, not {w!r}")
    return s


def create(obj: dict, refs: list[Any], weighting: str, lookback: int, rationale: str, model: str) -> dict:
    """Record a new ensemble candidate, score it, settle it, and return what its author is told."""
    if obj["metric"]["kind"] not in O.RETURN_METRICS:
        raise HTTPException(status_code=400, detail="ensembles need an objective scored on daily returns")
    weighting = _norm_weighting(weighting)
    if not (5 <= int(lookback) <= 250):
        raise HTTPException(status_code=400, detail="lookback_days must be between 5 and 250")
    lookback = int(lookback)
    members: list[dict] = []
    for ref in refs:
        c = _resolve(obj["id"], ref)
        if any(m["id"] == c["id"] for m in members):
            continue                                     # the same member twice is one member
        members.append(c)
    if not (MIN_MEMBERS <= len(members) <= MAX_MEMBERS):
        raise HTTPException(status_code=400, detail=f"an ensemble needs {MIN_MEMBERS} to {MAX_MEMBERS} distinct members "
                                                    f"(got {len(members)})")
    bad = [f"#{c['seq']}: {why}" for c in members if (why := ineligible(c))]
    if bad:
        raise HTTPException(status_code=400, detail="not eligible as ensemble members -- " + "; ".join(bad)
                                                    + ". Members must be scored, have passed the look-ahead test, "
                                                      "not have failed an audit, and not be ensembles.")
    members.sort(key=lambda c: c["seq"])                 # fixed member order (colours, weights)
    dup = _existing(obj["id"], [c["id"] for c in members], weighting, lookback)
    if dup is not None:
        raise HTTPException(status_code=409, detail=f"identical to ensemble #{dup} (same members and weighting)")

    t0 = time.time()
    combined, ens = build(obj, members, weighting, lookback)
    score, is_score, note, metrics = O._score_returns(obj, combined)
    metrics["ensemble"] = ens
    metrics["source"] = "ensemble: weighted sum of the members' stored net-of-cost daily returns (no costs module: " \
                        "costs are already inside each member's returns)"
    code = _spec(members, weighting, lookback)
    rationale = (rationale or "").strip() or f"{weighting} ensemble of " + " + ".join(f"#{c['seq']}" for c in members)

    cid = uuid.uuid4().hex[:10]
    with O._lock:
        seq = max(O.db().execute("SELECT COALESCE(MAX(seq),0) FROM candidates WHERE objective_id=?",
                                 (obj["id"],)).fetchone()[0] or 0,
                  O.db().execute("SELECT COALESCE(MAX(seq),0) FROM seq_hwm WHERE objective_id=?",
                                 (obj["id"],)).fetchone()[0] or 0) + 1
        O.db().execute("INSERT INTO seq_hwm (objective_id, seq) VALUES (?,?) "
                       "ON CONFLICT(objective_id) DO UPDATE SET seq=excluded.seq", (obj["id"], seq))
        O.db().execute(
            "INSERT INTO candidates (id, objective_id, seq, created_at, model, mode, parent_id, rationale, code, "
            "answer, status) VALUES (?,?,?,?,?, 'ensemble', NULL, ?, ?, '', 'evaluating')",
            (cid, obj["id"], seq, time.time(), model, rationale[:8000], code))
        O.db().commit()
    fields = {"status": "ok", "score": score, "is_score": is_score, "score_note": note,
              "metrics": json.dumps(metrics), "returns": json.dumps(combined),
              "lookahead": "pass",
              "lookahead_detail": "every member passed the look-ahead test; weights use only returns before each day",
              "eval_seconds": round(time.time() - t0, 2)}
    contender = O._settle(obj, cid, seq, code, fields)

    ins = metrics.get("in_sample") or {}
    O._board_post(obj["project_id"], "results", model or "operator",
                  f"ENSEMBLE #{seq} = {' + '.join('#' + str(c['seq']) for c in members)} ({weighting}"
                  + (f", {lookback}-day lookback" if weighting == "inverse_vol" else "") + f"): in-sample "
                  f"{obj['metric']['kind']} {ins.get(obj['metric']['kind'])}, smoothness {ins.get('smoothness')}"
                  + (" -- contender for best." if contender else "."),
                  {"objective_id": obj["id"], "candidate_id": cid, "seq": seq, "ensemble": [c["seq"] for c in members]})
    return view(obj, O.get_candidate(cid))


def view(obj: dict, c: dict) -> dict:
    """agent_view plus the ensemble's in-sample shape. Nothing from the holdout."""
    out = O.agent_view(obj, c)
    e = (c.get("metrics") or {}).get("ensemble") or {}
    C = e.get("correlation_in_sample") or []
    out.update({
        "id": c["id"],
        "ensemble": {
            "members": [{"seq": m["seq"], "model": m.get("model"), "avg_weight_in_sample": w}
                        for m, w in zip(e.get("members") or [], e.get("avg_weights_in_sample") or [])],
            "weighting": e.get("weighting"), "lookback_days": e.get("lookback_days"),
            "correlation_in_sample": C,
            "avg_abs_correlation_in_sample": _avg_abs(C, list(range(len(C)))) if C else None,
            "members_in_sample_sharpe": {str(s["seq"]): s.get("in_sample_sharpe") for s in e.get("member_stats") or []},
        },
    })
    return out


def disqualify_dependents(obj: dict, ids: set[str] | None = None, seqs: set[int] | None = None,
                          why: str = "") -> list[int]:
    """A member was disqualified after the ensemble was built (demoted, failed audit, a leak on
    re-test, a quarantined module): every ensemble containing it goes with it, and the title is
    handed on if one of them held it. Never raises -- the member's own disqualification stands."""
    ids, seqs = set(ids or ()), set(seqs or ())
    if not ids and not seqs:
        return []
    try:
        with O._lock:
            rows = O.db().execute("SELECT id, seq, metrics FROM candidates WHERE objective_id=? AND mode='ensemble' "
                                  "AND audit != 'fail'", (obj["id"],)).fetchall()
        hit: list[tuple[str, int]] = []
        for row in rows:
            members = ((json.loads(row["metrics"] or "{}") or {}).get("ensemble") or {}).get("members") or []
            bad = [m for m in members if m.get("id") in ids or m.get("seq") in seqs]
            if bad:
                O._update_candidate(row["id"], {
                    "audit": "fail",
                    "audit_notes": (f"[auto] Disqualified with its member(s) #{', #'.join(str(m['seq']) for m in bad)}. "
                                    f"{why}")[:20_000]})
                hit.append((row["id"], row["seq"]))
        if not hit:
            return []
        fresh = O.get_objective(obj["id"])
        if fresh.get("best_id") in {h[0] for h in hit}:
            with O._lock:
                O.db().execute("UPDATE objectives SET best_id=NULL, updated_at=? WHERE id=?", (time.time(), obj["id"]))
                O.db().commit()
            nxt = O._ranked(obj["id"], O._higher(obj), limit=1)
            if nxt:
                O._crown(obj["id"], nxt[0]["id"])
        return sorted(h[1] for h in hit)
    except Exception:  # noqa: BLE001
        logger.exception("could not disqualify the ensembles built on %s", ids or seqs)
        return []


# =======================================================================================
# Routes
# =======================================================================================
class Combine(BaseModel):
    members: list[int | str] = Field(..., max_length=50)
    weighting: str = Field("equal", max_length=40)
    lookback_days: int = DEFAULT_LOOKBACK
    rationale: str = Field("", max_length=8_000)
    model: str = Field("operator", max_length=200)


@router.post("/objectives/{oid}/ensembles")
async def create_ensemble(oid: str, req: Combine) -> dict:
    """Combine verified candidates into one weighted portfolio candidate (see the module doc)."""
    obj = O.get_objective(oid)
    return create(obj, req.members, req.weighting, req.lookback_days, req.rationale, req.model or "operator")


@router.get("/objectives/{oid}/correlations")
async def correlations(oid: str, seqs: str | None = None, top: int = 12) -> dict:
    """In-sample (dates before the split) daily-return correlations among the given candidates,
    or the top `top` ranked non-ensembles, with low-correlation sets to combine.

    Safe to hand to agents: every number here is computed from in-sample dates only."""
    obj = O.get_objective(oid)
    split = obj.get("split_date")
    if seqs:
        refs = [s for s in seqs.replace(" ", ",").split(",") if s.strip()][:MAX_POOL]
        cands = []
        for ref in refs:
            c = _resolve(oid, ref)
            if all(x["id"] != c["id"] for x in cands):
                cands.append(c)
    else:
        top = max(2, min(int(top or 12), MAX_POOL))
        ranked = [c for c in O._ranked(oid, O._higher(obj), 200) if not is_ensemble(c)][:top]
        cands = [O.get_candidate(c["id"]) for c in ranked]
    cands = [c for c in cands if _returns(c)]
    ppy = float(obj["metric"].get("periods_per_year") or 252)
    series = in_sample([_returns(c) for c in cands], split)
    dates, R, _ = grid(series)
    C = correlation(R) if cands else []
    info = []
    for c, s in zip(cands, series):
        why = ineligible(c)
        info.append({"id": c["id"], "seq": c["seq"], "model": c.get("model"),
                     "rationale": (c.get("rationale") or "")[:160],
                     "in_sample_score": c.get("is_score"),
                     "in_sample_sharpe": O._stats([r for _, r in s], ppy).get("sharpe"),
                     "eligible": why is None, **({"why_not": why} if why else {})})
    pool = [k for k, x in enumerate(info) if x["eligible"] and (x["in_sample_score"] or 0) > 0]
    sugg = []
    for idx in suggest(C, pool):
        # What the equal-weight blend of the set did in-sample: the smoothing, in numbers.
        blend = R[:, idx].mean(axis=1) if len(dates) else np.zeros(0)
        st = O._stats([float(x) for x in blend], ppy)
        sugg.append({"seqs": [info[k]["seq"] for k in idx], "avg_abs_rho": _avg_abs(C, idx),
                     "members_in_sample_sharpe": {str(info[k]["seq"]): info[k]["in_sample_sharpe"] for k in idx},
                     "equal_weight_in_sample_sharpe": st.get("sharpe"),
                     "equal_weight_in_sample_smoothness": st.get("smoothness")})
    return {"period": f"in-sample only (dates before {split})" if split else "all dates (no split)",
            "days": len(dates), "candidates": info, "seqs": [x["seq"] for x in info], "matrix": C,
            "suggestions": sugg,
            "note": ("|rho| < 0.3 is nearly uncorrelated: combining such strategies smooths the equity curve. "
                     "Suggestions use only candidates that are eligible (scored, look-ahead pass, not "
                     "disqualified, not ensembles) with a positive in-sample score.")}
