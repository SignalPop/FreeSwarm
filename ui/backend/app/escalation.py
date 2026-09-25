"""Getting a stuck search unstuck: ask stronger models for new directions, cheapest first.

Every few minutes each running objective of a swarm-enabled project is checked. It is
**stuck** when no new champion has been crowned for at least ``stuck_candidates`` candidates
*and* ``stuck_minutes`` minutes (both from Settings -> External models). A stuck objective gets
one **idea** from the first rung of its project's ladder (swarm_policy.plan): the free model
with the best AA Intelligence score. If it is still stuck ``step_candidates`` candidates and
``step_minutes`` later, the next rung -- the cheapest external model that out-scores the one
below -- is asked, and so on up to the most expensive. A new champion resets the climb.

An idea is a short set of conceptually new directions. It is stored with the objective and
handed to every agent in the context of its next iterations (``objectives.context`` ->
``ideas``), until the objective improves.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

from fastapi import APIRouter, HTTPException

from . import external, projects, swarm_policy
from . import objectives as obj_mod

logger = logging.getLogger("freetoken.escalation")
router = APIRouter(tags=["escalation"])

TICK_S = 120
IDEA_MAX_TOKENS = 8000

CompleteFn = Callable[[str, list[dict], int, str], Awaitable[str]]
LoadedFn = Callable[[], list[dict]]
_complete: CompleteFn | None = None
_loaded: LoadedFn | None = None
_busy: set[str] = set()
# objective id -> {"ts", "detail", "model"}: why the last due escalation did not produce an
# idea. It was only logged before, so "Claude never fired" (the whole daily budget was gone
# before the first rung came due) was invisible from the console. Cleared by the next idea.
_last_error: dict[str, dict] = {}


def _note_error(oid: str, detail: object, model: str | None = None) -> None:
    _last_error[oid] = {"ts": time.time(), "detail": str(detail)[:1000], "model": model}


def _ensure_table() -> None:
    with obj_mod._lock:
        obj_mod.db().execute(
            """CREATE TABLE IF NOT EXISTS ideas (
                id INTEGER PRIMARY KEY AUTOINCREMENT, objective_id TEXT NOT NULL, ts REAL NOT NULL,
                model TEXT NOT NULL, rung INTEGER NOT NULL, text TEXT NOT NULL,
                candidates_at INTEGER NOT NULL DEFAULT 0, trigger TEXT NOT NULL DEFAULT 'stuck')""")
        obj_mod.db().commit()


def assess(obj: dict) -> dict:
    """How long the objective has gone without improving, and the ideas asked for since."""
    _ensure_table()
    oid = obj["id"]
    with obj_mod._lock:
        last_imp = obj_mod.db().execute(
            "SELECT max(champion_at) FROM candidates WHERE objective_id=? AND champion_at IS NOT NULL",
            (oid,)).fetchone()[0]
        since = last_imp or obj["created_at"]
        n_since = obj_mod.db().execute(
            "SELECT count(*) FROM candidates WHERE objective_id=? AND created_at > ?", (oid, since)).fetchone()[0]
        total = obj_mod.db().execute("SELECT count(*) FROM candidates WHERE objective_id=?", (oid,)).fetchone()[0]
        ideas = [dict(r) for r in obj_mod.db().execute(
            "SELECT * FROM ideas WHERE objective_id=? AND ts > ? ORDER BY ts", (oid, since)).fetchall()]
        after_last = None
        if ideas:
            after_last = obj_mod.db().execute(
                "SELECT count(*) FROM candidates WHERE objective_id=? AND created_at > ?",
                (oid, ideas[-1]["ts"])).fetchone()[0]
    return {"since": since, "improved_before": bool(last_imp), "candidates_since": n_since,
            "minutes_since": (time.time() - since) / 60, "total_candidates": total, "ideas": ideas,
            "candidates_since_idea": after_last,
            "minutes_since_idea": (time.time() - ideas[-1]["ts"]) / 60 if ideas else None}


def due(a: dict, cfg: dict) -> bool:
    if not a["ideas"]:
        return a["candidates_since"] >= cfg["stuck_candidates"] and a["minutes_since"] >= cfg["stuck_minutes"]
    return (a["candidates_since_idea"] >= cfg["step_candidates"]
            and a["minutes_since_idea"] >= cfg["step_minutes"])


def _prompt(obj: dict, a: dict) -> str:
    ctx_ranked = obj_mod._ranked(obj["id"], obj_mod._higher(obj), 8)
    with obj_mod._lock:
        recent = [dict(r) for r in obj_mod.db().execute(
            "SELECT seq, model, status, rationale, score_note, is_score FROM candidates WHERE objective_id=? "
            "ORDER BY seq DESC LIMIT 15", (obj["id"],)).fetchall()]
        lessons = [r[0] for r in obj_mod.db().execute(
            "SELECT text FROM lessons WHERE objective_id=? AND active=1 ORDER BY ts DESC LIMIT 30",
            (obj["id"],)).fetchall()]
    metric = obj_mod.METRIC_LABEL.get(obj["metric"]["kind"], obj["metric"]["kind"])
    lines = [
        "You are advising a team of AI agents that search for the best solution to a quantitative "
        "research objective. Each agent writes a Python candidate, which is scored on a hidden holdout. "
        f"The search has produced {a['candidates_since']} candidates over {a['minutes_since'] / 60:.1f} hours "
        "without beating the current best. They are stuck in a local optimum.",
        "", f"OBJECTIVE: {obj['title']}", obj.get("description") or "", f"Metric: {metric}",
        "", "CURRENT LEADERBOARD (best first; in-sample score and the author's rationale):",
    ]
    for i, c in enumerate(ctx_ranked, 1):
        lines.append(f"{i}. #{c['seq']} ({c['model']}) in-sample {c.get('is_score')}: {(c.get('rationale') or '')[:500]}")
    lines += ["", "MOST RECENT ATTEMPTS (newest first):"]
    for c in recent:
        note = f" -- {c['score_note'][:200]}" if c.get("score_note") else ""
        lines.append(f"#{c['seq']} {c['status']} ({c['model']}): {(c.get('rationale') or '')[:300]}{note}")
    if lessons:
        lines += ["", "WHAT THE TEAM HAS LEARNED:"] + [f"- {x}" for x in lessons]
    if a["ideas"]:
        lines += ["", "IDEAS ALREADY GIVEN SINCE THE LAST IMPROVEMENT (they did not help -- go further):"]
        lines += [f"- from {i['model']}: {i['text'][:1500]}" for i in a["ideas"]]
    lines += [
        "", "Propose 3 CONCEPTUALLY DIFFERENT directions the team has not tried -- a different model "
        "family, signal, data source in the project, framing of the target, or risk treatment; not a "
        "parameter tweak of the leader. For each: (1) the idea in one sentence, (2) why it could beat the "
        "leader, (3) the first concrete experiment an agent should run. Be specific and brief; plain text, "
        "no preamble. Guard against look-ahead bias and overfitting to the in-sample period.",
    ]
    return "\n".join(lines)


async def escalate(obj: dict, project: dict, *, trigger: str = "stuck") -> dict:
    """Ask the next rung for ideas now. Returns the stored idea."""
    if _complete is None or _loaded is None:
        raise HTTPException(status_code=503, detail="escalation is not running")
    p = swarm_policy.plan(project, _loaded())
    if not p["ladder"]:
        detail = "no model to ask: load a rated model, or tick an external model for this project"
        _note_error(obj["id"], detail)
        raise HTTPException(status_code=409, detail=detail)
    a = await asyncio.to_thread(assess, obj)
    rung = min(len(a["ideas"]), len(p["ladder"]) - 1)
    model = p["ladder"][rung]["model"]
    prompt = await asyncio.to_thread(_prompt, obj, a)
    try:
        # The purpose prefix is what lets this call spend the reserve search may not touch.
        text = await _complete(model, [{"role": "user", "content": prompt}], IDEA_MAX_TOKENS,
                               f"{external.IDEAS_PURPOSE}{obj['id']}")
    except HTTPException as exc:
        _note_error(obj["id"], exc.detail, model)
        raise
    except Exception as exc:
        _note_error(obj["id"], f"{type(exc).__name__}: {exc}", model)
        raise
    if not text:
        _note_error(obj["id"], f"{model} returned no ideas", model)
        raise HTTPException(status_code=502, detail=f"{model} returned no ideas")
    _last_error.pop(obj["id"], None)
    with obj_mod._lock:
        cur = obj_mod.db().execute(
            "INSERT INTO ideas (objective_id, ts, model, rung, text, candidates_at, trigger) VALUES (?,?,?,?,?,?,?)",
            (obj["id"], time.time(), model, rung, text[:8000], a["total_candidates"], trigger))
        obj_mod.db().commit()
    logger.info("objective %s stuck for %d candidates: rung %d (%s) gave ideas", obj["id"],
                a["candidates_since"], rung, model)
    return {"id": cur.lastrowid, "model": model, "rung": rung, "text": text}


def ideas_for_context(oid: str) -> list[dict]:
    """Ideas since the last improvement, newest first -- what agents are told to try."""
    obj = obj_mod.get_objective(oid)
    a = assess(obj)
    return [{"model": i["model"], "rung": i["rung"], "text": i["text"]} for i in reversed(a["ideas"])][:3]


async def _tick() -> None:
    cfg = external.config()["escalation"]
    if not cfg["enabled"]:
        return
    for project in await asyncio.to_thread(projects.list_projects):
        if not project.get("swarm_enabled", True):
            continue
        objs = (await obj_mod.list_objectives(project["id"], "running")).get("objectives", [])
        for o in objs:
            if o["id"] in _busy:
                continue
            obj = obj_mod.get_objective(o["id"])
            a = await asyncio.to_thread(assess, obj)
            if not due(a, cfg):
                continue
            _busy.add(o["id"])
            try:
                await escalate(obj, project)
            except HTTPException as exc:
                logger.warning("escalation for %s skipped: %s", o["id"], exc.detail)
            except Exception:  # noqa: BLE001 -- one objective must not stop the others
                logger.exception("escalation for %s failed", o["id"])
            finally:
                _busy.discard(o["id"])


async def run(complete: CompleteFn, loaded: LoadedFn) -> None:
    global _complete, _loaded
    _complete, _loaded = complete, loaded
    await asyncio.to_thread(_ensure_table)
    while True:
        await asyncio.sleep(TICK_S)
        try:
            await _tick()
        except Exception:  # noqa: BLE001
            logger.exception("escalation tick failed")


# =======================================================================================
# API
# =======================================================================================
@router.get("/objectives/{oid}/escalation")
async def status(oid: str) -> dict:
    obj = obj_mod.get_objective(oid)
    project = projects.get(obj["project_id"]) or {}
    a = await asyncio.to_thread(assess, obj)
    cfg = external.config()["escalation"]
    p = swarm_policy.plan(project, _loaded()) if _loaded else {"ladder": []}
    rung = min(len(a["ideas"]), max(len(p["ladder"]) - 1, 0))
    return {"config": cfg, "stuck": a["candidates_since"] >= cfg["stuck_candidates"]
            and a["minutes_since"] >= cfg["stuck_minutes"], "due": due(a, cfg),
            "candidates_since_improvement": a["candidates_since"],
            "minutes_since_improvement": round(a["minutes_since"], 1),
            "ladder": p["ladder"], "next_rung": rung if p["ladder"] else None,
            "next_model": p["ladder"][rung]["model"] if p["ladder"] else None,
            "ideas": [{**i, "text": i["text"]} for i in reversed(a["ideas"])],
            # Why the last due (or operator) escalation produced nothing -- budget refused,
            # provider error, no ladder -- as {ts, detail, model}; None after a good idea.
            "last_error": _last_error.get(oid)}


@router.post("/objectives/{oid}/escalation/run")
async def run_now(oid: str) -> dict:
    """The operator's "get unstuck now": ask the next rung without waiting for the thresholds."""
    obj = obj_mod.get_objective(oid)
    project = projects.get(obj["project_id"])
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    if oid in _busy:
        raise HTTPException(status_code=409, detail="already asking for ideas")
    _busy.add(oid)
    try:
        return await escalate(obj, project, trigger="operator")
    finally:
        _busy.discard(oid)


@router.get("/projects/{project_id}/swarm/plan")
async def project_plan(project_id: str) -> dict:
    project = projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    return swarm_policy.plan(project, _loaded() if _loaded else [])


# =======================================================================================
# External-model usage, per project
# =======================================================================================
def _outcomes(project_id: str, names: list[str]) -> tuple[dict, dict, list[dict]]:
    """From objectives.sqlite3, for this project's objectives: what each model submitted
    (count by status, look-ahead passes, champions) and the ideas each gave."""
    _ensure_table()
    cands: dict[str, dict] = {n: {"total": 0, "by_status": {}, "lookahead_pass": 0, "champions": 0,
                                  "last_ts": None} for n in names}
    ideas: dict[str, dict] = {n: {"count": 0, "last_ts": None} for n in names}
    with obj_mod._lock:
        con = obj_mod.db()
        objs = [dict(r) for r in con.execute(
            "SELECT id, title, status FROM objectives WHERE project_id=?", (project_id,)).fetchall()]
        if not names:
            return cands, ideas, objs
        marks = ",".join("?" * len(names))
        rows = con.execute(
            "SELECT c.model, c.status, COUNT(*), SUM(c.lookahead='pass'), SUM(c.champion_at IS NOT NULL), "
            "MAX(c.created_at) FROM candidates c JOIN objectives o ON o.id=c.objective_id "
            f"WHERE o.project_id=? AND c.model IN ({marks}) GROUP BY c.model, c.status",
            (project_id, *names)).fetchall()
        irows = con.execute(
            "SELECT i.model, COUNT(*), MAX(i.ts) FROM ideas i JOIN objectives o ON o.id=i.objective_id "
            f"WHERE o.project_id=? AND i.model IN ({marks}) GROUP BY i.model", (project_id, *names)).fetchall()
    for model, status, n, la, champ, last in rows:
        c = cands[model]
        c["total"] += n
        c["by_status"][status] = n
        c["lookahead_pass"] += int(la or 0)
        c["champions"] += int(champ or 0)
        c["last_ts"] = max(c["last_ts"] or 0, last or 0) or None
    for model, n, last in irows:
        ideas[model] = {"count": n, "last_ts": last}
    return cands, ideas, objs


_NO_USE = {"calls_total": 0, "usd_total": 0.0, "last_call_ts": None, "calls_today": 0, "usd_today": 0.0,
           "prompt_tokens_today": 0, "completion_tokens_today": 0, "ideas_calls_today": 0}


@router.get("/external/usage")
async def external_usage(project_id: str | None = None) -> dict:
    """What every enabled external model does, costs and is blocked by -- for one project.

    Built for the Swarm page's Resources panel: a hosted model that ran, spent $10 and
    produced nothing (rate limits, malformed tool calls, then an exhausted budget) looked
    exactly like one that was working. Spend and refusals are per model across the console
    (one bill); role, candidates, ideas and escalation errors are this project's. Lives here,
    not in external.py, because it needs the plan, the projects and the objectives database,
    all of which already import external.
    """
    project = projects.get(project_id) if project_id else None
    if project_id and project is None:
        raise HTTPException(status_code=404, detail="project not found")
    loaded = _loaded() if _loaded else []
    loaded_names = {m.get("model") for m in loaded}
    p = swarm_policy.plan(project, loaded) if project else None
    enabled = list(external.config()["enabled"])
    ledger = await asyncio.to_thread(external.usage_by_model)
    if project:
        cands, ideas, objs = await asyncio.to_thread(_outcomes, project["id"], enabled)
    else:
        cands, ideas, objs = {}, {}, []

    esc_errors = [{"objective_id": o["id"], "title": o["title"], **_last_error[o["id"]]}
                  for o in objs if o["id"] in _last_error]
    esc_errors.sort(key=lambda e: -(e.get("ts") or 0))

    rows = []
    for name in enabled:
        sp = external.split(name)
        if sp is None:
            continue
        # A model can be several things at once: "Both" searches and sits on the ladder, and
        # an ideas-only model is also listed as reserved. `loaded` is False without an API key.
        role: dict = {"allowed": bool(project) and swarm_policy.permitted(project, name),
                      "loaded": name in loaded_names, "search": None, "ideas": None, "reserved": None}
        if p is not None:
            s = next((m for m in p["search"] if m["model"] == name), None)
            if s:
                role["search"] = {"agents": s.get("agents", 1), "why": s.get("why")}
            rung = next((i for i, m in enumerate(p["ladder"]) if m["model"] == name), None)
            if rung is not None:
                role["ideas"] = {"rung": rung, "of": len(p["ladder"]), "why": p["ladder"][rung].get("why")}
            r = next((m for m in p["reserved"] if m["model"] == name), None)
            if r:
                role["reserved"] = {"why": r.get("why"), "budget_paused": bool(r.get("budget_paused"))}
        rows.append({"model": name, "provider": sp[0], "provider_label": external.PROVIDERS[sp[0]]["label"],
                     "role": role, **(ledger.get(name) or _NO_USE), "refusals": external.refusals(name),
                     "candidates": cands.get(name), "ideas": ideas.get(name)})

    limit, reserve = external.limits()
    return {"today_usd": round(external.spent_today(), 4), "limit_usd": limit, "ideas_reserve_usd": reserve,
            "search_limit_usd": round(limit - reserve, 4), "search_budget_left": external.search_budget_left(),
            "search_paused": (p or {}).get("search_paused"), "escalation_errors": esc_errors, "models": rows}
