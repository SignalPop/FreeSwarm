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
        raise HTTPException(status_code=409, detail=(
            "no model to ask: load a rated model, or tick an external model for this project"))
    a = await asyncio.to_thread(assess, obj)
    rung = min(len(a["ideas"]), len(p["ladder"]) - 1)
    model = p["ladder"][rung]["model"]
    prompt = await asyncio.to_thread(_prompt, obj, a)
    text = await _complete(model, [{"role": "user", "content": prompt}], IDEA_MAX_TOKENS,
                           f"ideas:{obj['id']}")
    if not text:
        raise HTTPException(status_code=502, detail=f"{model} returned no ideas")
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
            "ideas": [{**i, "text": i["text"]} for i in reversed(a["ideas"])]}


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

