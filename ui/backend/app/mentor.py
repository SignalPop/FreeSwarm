"""The mentor: the team's strongest free model thinks for the team instead of searching.

A search that only mutates the leader turns into brute force -- thresholds nudged, windows
swapped -- and learns slowly because nobody steps back. The mentor does. On a regular cadence
(every ``every_candidates`` new candidates, or every ``every_minutes`` even without one --
Settings -> External models, external.config()["mentor"]), the runner's mentor agent
(swarm_runner.Worker with role "mentor") reads the evidence pack built here and gives the team:

* **directions** -- conceptually new ideas, each with the hypothesis and the experiment that
  would falsify it, stored in the ``ideas`` table (escalation.py) and shown to every searcher;
  a candidate says which idea it tested (``candidates.idea_id``), so each idea gets a record;
* **coaching** on the board -- what the evidence says the team is doing wrong (parameter
  tweaks, costs, forecasts that do not help) and replies to teammates' questions;
* **forecasts to build** -- recipes for the time-series models, chosen to learn about the data.

It also takes over the team-practices rewrite, which otherwise costs a search iteration.

The evidence is the team's memory made legible: the ideas scoreboard (what each idea's
candidates scored), the forecast scoreboard (each forecast's measured skill, the lift its
inputs gave, and how candidates that used it fared against those that did not), the cost and
direction diagnoses, and how many recent candidates only changed numbers. In-sample only: the
holdout never reaches an agent, the mentor included.
"""

from __future__ import annotations

import statistics
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import external, projects
from . import objectives as obj_mod

router = APIRouter(tags=["mentor"])

MENTOR_ACTIVE_S = 1800  # a mentor that asked for a brief this recently is on duty
IDEA_TEXT_MAX = 8000

_seen: dict[str, float] = {}  # project id -> last time a mentor asked for a brief
# objective id -> when a due brief was last handed out. A pass that posts no direction (only
# coaching, or a failed generation) leaves no idea behind; without this, a time-only cadence
# would call it due again on the mentor's next poll, a minute later.
_passes: dict[str, float] = {}


def cadence() -> tuple[int, int]:
    """(every_candidates, every_minutes) from Settings -> External models."""
    c = external.config()["mentor"]
    return max(1, int(c["every_candidates"])), max(1, int(c["every_minutes"]))


def mentor_active(project_id: str) -> bool:
    """Whether a mentor is working for this project (then it, not a searcher, rewrites the practices)."""
    return time.time() - _seen.get(project_id, 0.0) < MENTOR_ACTIVE_S


def _ensure_tables() -> None:
    from .escalation import _ensure_table

    _ensure_table()


def _median(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def _rows(sql: str, args: tuple) -> list[dict]:
    with obj_mod._lock:
        return [dict(r) for r in obj_mod.db().execute(sql, args).fetchall()]


def due(oid: str) -> tuple[bool, str]:
    _ensure_tables()
    every_n, every_min = cadence()
    last = _rows("SELECT ts FROM ideas WHERE objective_id=? AND trigger='mentor' ORDER BY ts DESC LIMIT 1", (oid,))
    since = max(last[0]["ts"] if last else 0.0, _passes.get(oid, 0.0))
    if not since:
        return True, "no mentor notes yet"
    n = _rows("SELECT count(*) AS n FROM candidates WHERE objective_id=? AND created_at > ?", (oid, since))[0]["n"]
    minutes = (time.time() - since) / 60
    if n >= every_n:
        return True, f"{n} new candidates since the last notes"
    # Time alone is enough: a slow search (or one going in circles) needs fresh thinking most.
    if minutes >= every_min:
        return True, f"{minutes:.0f} minutes and {n} new candidate(s) since the last notes"
    return False, f"{n} new candidate(s), {minutes:.0f} min since the last notes"


def _verdict_kind(verdict: str | None) -> str:
    v = verdict or ""
    # "No clear edge before costs" also contains "edge before costs": test it first.
    if "No clear edge" in v:
        return "no edge"
    if "FLIPPED" in v:
        return "points the wrong way (flip wins)"
    if "points the wrong way" in v:
        return "wrong way and costs"
    if "edge before costs" in v:
        return "edge given away by costs"
    return "healthy" if v else "no diagnosis"


def idea_scoreboard(oid: str, limit: int = 10) -> list[dict]:
    """The latest ideas and what the candidates that tested them scored (in-sample)."""
    _ensure_tables()
    ideas = _rows("SELECT id, ts, model, trigger, text FROM ideas WHERE objective_id=? ORDER BY ts DESC LIMIT ?",
                  (oid, limit))
    out = []
    for i in ideas:
        cands = _rows("SELECT seq, status, is_score, champion_at, metrics FROM candidates "
                      "WHERE objective_id=? AND idea_id=?", (oid, i["id"]))
        ok = [c for c in cands if c["status"] == "ok" and c["is_score"] is not None]
        best = max(ok, key=lambda c: c["is_score"], default=None)
        verdict = None
        if best:
            m = obj_mod.json.loads(best["metrics"] or "{}")
            verdict = _verdict_kind(((m.get("costs") or {}).get("verdict")))
        out.append({"id": i["id"], "model": i["model"], "trigger": i["trigger"],
                    "minutes_ago": round((time.time() - i["ts"]) / 60), "idea": i["text"][:600],
                    "tried": len(cands), "ran": len(ok), "failed": len(cands) - len(ok),
                    "best_in_sample": round(best["is_score"], 3) if best else None,
                    "best_seq": best["seq"] if best else None,
                    "median_in_sample": _median([c["is_score"] for c in ok]),
                    "best_diagnosis": verdict,
                    "champions": sum(1 for c in cands if c["champion_at"])})
    return out


def forecast_scoreboard(obj: dict, recent: int = 300) -> list[dict]:
    """Every forecast feature: how good its forecasts are, and whether using it helped.

    `helped` compares the median in-sample score of recent candidates that loaded the feature
    with that of candidates that loaded no forecast at all -- evidence, not proof (candidates
    differ in more than their forecasts), which is why the count sits beside it."""
    feats = obj_mod.list_features(obj["id"])
    cands = _rows("SELECT is_score, metrics, code FROM candidates WHERE objective_id=? AND status='ok' "
                  "AND is_score IS NOT NULL ORDER BY seq DESC LIMIT ?", (obj["id"], recent))
    views = [f["view"] for f in feats]
    users: dict[str, list[float]] = {}
    none: list[float] = []
    for c in cands:
        m = obj_mod.json.loads(c["metrics"] or "{}")
        # Recorded by the harness since features_used existed; older candidates: the views
        # their code names (ft.forecast recipes are only known from the record).
        used = m["features_used"] if "features_used" in m else [v for v in views if v in (c["code"] or "")]
        if not used:
            none.append(c["is_score"])
        for v in used:
            users.setdefault(v, []).append(c["is_score"])
    base = _median(none)
    out = []
    for f in feats:
        skill = {}
        for series, s in (f.get("skill") or {}).items():
            if "with_inputs" in s:
                skill[series] = {"skill": (s.get("with_inputs") or {}).get("skill"),
                                 "direction": (s.get("with_inputs") or {}).get("direction"),
                                 "lift_from_inputs": s.get("lift_skill")}
            else:
                skill[series] = {"skill": s.get("skill_vs_no_change"), "direction": s.get("direction_accuracy")}
        used_by = users.get(f["view"], [])
        med = _median(used_by)
        p = f.get("params") or {}
        out.append({"view": f["view"], "model": p.get("model"), "series": p.get("series"),
                    "inputs": p.get("covariates") or [], "horizon": p.get("horizon"), "every": p.get("every"),
                    "skill": skill, "used_by": len(used_by), "median_in_sample_users": med,
                    "median_in_sample_no_forecast": base,
                    "helped": (round(med - base, 3) if med is not None and base is not None else None),
                    "auto": f["view"].startswith("fc_auto_"),
                    "requested_by": (f.get("recipe") or {}).get("requested_by")})
    out.sort(key=lambda r: (-(r["used_by"]), r["view"]))
    return out


def diagnosis_mix(oid: str, n: int = 30) -> dict:
    """What the last `n` evaluated candidates' results say, counted: the team's habits."""
    cands = _rows("SELECT status, metrics FROM candidates WHERE objective_id=? ORDER BY seq DESC LIMIT ?", (oid, n))
    kinds: dict[str, int] = {}
    change: dict[str, int] = {}
    failed = 0
    for c in cands:
        if c["status"] != "ok":
            failed += 1
            continue
        m = obj_mod.json.loads(c["metrics"] or "{}")
        k = _verdict_kind((m.get("costs") or {}).get("verdict"))
        kinds[k] = kinds.get(k, 0) + 1
        ck = (m.get("change") or {}).get("kind")
        if ck:
            change[ck] = change.get(ck, 0) + 1
    return {"candidates": len(cands), "failed_to_run": failed, "results": kinds, "changes_vs_parent": change}


def brief(oid: str, model: str = "") -> dict:
    """The mentor's evidence pack. Also marks the mentor on duty for the project."""
    obj = obj_mod.get_objective(oid)
    project = projects.get(obj["project_id"]) or {}
    _seen[obj["project_id"]] = time.time()
    is_due, why = due(oid)
    if is_due:
        _passes[oid] = time.time()
    higher = obj_mod._higher(obj)
    ranked = obj_mod._ranked(oid, higher, 6)

    def line(c: dict) -> dict:
        m = c.get("metrics") or {}
        return {"seq": c["seq"], "model": c["model"], "status": c["status"], "in_sample": c.get("is_score"),
                "idea_id": c.get("idea_id"), "rationale": (c.get("rationale") or "")[:350],
                "diagnosis": ((m.get("costs") or {}).get("verdict") or "")[:400] or None,
                "change": (m.get("change") or {}).get("kind"),
                "forecasts_used": m.get("features_used") or [],
                "problem": (c.get("score_note") or "")[:200] if c["status"] != "ok" else None}

    recent = _rows(f"SELECT {obj_mod._LIGHT} FROM candidates WHERE objective_id=? ORDER BY seq DESC LIMIT 15", (oid,))
    for r in recent:
        r["metrics"] = obj_mod.json.loads(r.get("metrics") or "{}")
    with obj_mod._lock:
        lessons = [r[0] for r in obj_mod.db().execute(
            "SELECT text FROM lessons WHERE objective_id=? AND active=1 ORDER BY ts DESC LIMIT 30", (oid,)).fetchall()]
    return {
        "due": is_due, "why": why,
        "objective": {k: obj[k] for k in ("id", "title", "description", "metric", "split_date", "dataset", "time_column")},
        "metric_label": obj_mod.METRIC_LABEL.get(obj["metric"]["kind"], obj["metric"]["kind"]),
        "leaderboard": [line(c) for c in ranked],
        "recent": [line(c) for c in recent],
        "lessons": lessons,
        "ideas": idea_scoreboard(oid),
        "forecasts": forecast_scoreboard(obj),
        "habits": diagnosis_mix(oid),
        "forecasters": obj_mod._forecasters_brief(obj),
        "field_scan": obj_mod._field_scan_brief(obj["project_id"]),
        # Decile studies and explored forecast-input combinations: what is already known.
        "deci_studies": obj_mod._deci_brief(obj),
        "forecast_inputs": obj_mod._combo_brief(obj),
        "fields": obj_mod.field_guide(obj, project.get("data_dir", "")) if project else {},
        "playbook": obj_mod._playbook(obj["project_id"]),
        "library": obj_mod._library_brief(obj["project_id"]),
        # Leased like the searchers' chore was: only one rewrite at a time.
        "refresh_practices": obj_mod._practices_due(obj["project_id"]) if is_due else False,
    }


class IdeaIn(BaseModel):
    model: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=20, max_length=IDEA_TEXT_MAX)
    trigger: str = Field("mentor", pattern=r"^(mentor|operator)$")


def add_idea(oid: str, model: str, text: str, trigger: str = "mentor") -> int:
    _ensure_tables()
    obj_mod.get_objective(oid)
    with obj_mod._lock:
        total = obj_mod.db().execute("SELECT count(*) FROM candidates WHERE objective_id=?", (oid,)).fetchone()[0]
        cur = obj_mod.db().execute(
            "INSERT INTO ideas (objective_id, ts, model, rung, text, candidates_at, trigger) VALUES (?,?,?,?,?,?,?)",
            (oid, time.time(), model, 0, text[:IDEA_TEXT_MAX], total, trigger))
        obj_mod.db().commit()
    return int(cur.lastrowid)


@router.get("/objectives/{oid}/mentor/brief")
async def mentor_brief(oid: str, model: str = "") -> dict:
    import asyncio

    return await asyncio.to_thread(brief, oid, model)


@router.post("/objectives/{oid}/ideas")
async def post_idea(oid: str, req: IdeaIn) -> dict:
    """A mentor's direction for the team (one idea per call, so each gets its own record)."""
    return {"id": add_idea(oid, req.model, req.text, req.trigger)}


@router.get("/objectives/{oid}/scoreboards")
async def scoreboards(oid: str) -> dict:
    """The team's memory at a glance: ideas, forecasts and habits (for the console)."""
    import asyncio

    obj = obj_mod.get_objective(oid)
    if obj is None:
        raise HTTPException(status_code=404, detail="no such objective")
    ideas, forecasts, habits = await asyncio.gather(
        asyncio.to_thread(idea_scoreboard, oid), asyncio.to_thread(forecast_scoreboard, obj),
        asyncio.to_thread(diagnosis_mix, oid))
    return {"ideas": ideas, "forecasts": forecasts, "habits": habits,
            "mentor_active": mentor_active(obj["project_id"]), "mentor_due": due(oid)[1]}
