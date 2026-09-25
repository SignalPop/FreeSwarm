"""Which models a project's swarm uses, and for what.

Two jobs, two different scores (see ratings.py):

* **Search** -- the agents that write and test candidates all day. Every *free* model the
  project allows (this computer's engines and models shared by paired computers) searches.
  A paid external model joins the search only when it is ticked for the project **and** it is
  *similar* to the free models -- a peer of the same ability, adding throughput:

  1. it is the same model as a free one (``gpt-oss-120b@groq`` next to a local gpt-oss-120b,
     ``qwen/qwen3.8-27b@groq`` next to a local Qwen3.8-27B), or
  2. its SWE-bench Verified score is within ``SWE_MARGIN`` points of the free models' range, or
  3. when SWE cannot decide (it or the free models are unrated), its AA Intelligence score is
     within ``AA_MARGIN`` points of the free models' AA range.

  Externals that are clearly stronger are kept for the escalation ladder, not routine search.

* **Ideas when stuck** -- the escalation ladder (escalation.py). When the search stops
  improving, the model with the best Artificial Analysis Intelligence Index is asked for new
  conceptual directions: first the best *free* model (e.g. DeepSeek-V4), then, if the search is
  still stuck, the project's external models that out-score it, **cheapest first**, climbing
  toward the most expensive only while the problem stays unsolved.

* **Mentor** -- one free model thinks for the team instead of searching (mentor.py): it reads
  the team's results on a cadence and posts directions, coaching and forecasts to build, and
  rewrites the team practices. On Auto, the free model with the best AA score becomes the
  mentor once at least MENTOR_MIN_SEARCHERS other free models search (a lone model must keep
  searching); a model set to New ideas or Both mentors too (Both also searches). Paid models
  never mentor -- it runs every few candidates, which is what the free models are for.

"Free" here means no per-token bill; paired computers' models count as free.

**Roles set by the operator win.** A project can mark any allowed model Search, New ideas or
Both (``project["model_roles"]``). A marked model does exactly that, whatever the rules above
would say; models left on Auto follow them. Once any model is marked New ideas (or Both), the
ladder is the marked models only: free ones first, best AA first, then paid ones cheapest first.
Every searching model runs at once -- ``parallel_agents`` agents per hosted model and
``parallel_local_agents`` per free one -- so the search uses everything it is given.

**Money stops the search, not the ladder.** Once today's *search* budget is spent (the daily
limit minus the reserve held for ideas, see external.py), hosted models move from ``search``
to ``reserved`` until midnight, whatever their role; free models keep searching, and the
ladder keeps its paid rungs because the reserve is theirs to spend.
"""

from __future__ import annotations

from . import external, ratings

SWE_MARGIN = 5.0
AA_MARGIN = 5.0
MENTOR_MIN_SEARCHERS = 2


def kind(m: dict) -> str:
    if m.get("external"):
        return "external"
    if m.get("remote"):
        return "network"
    return "local"


def permitted(project: dict, name: str) -> bool:
    """Whether a project may use a model. External models must be ticked explicitly: "all
    loaded models" (models == None) never includes something that costs money."""
    allowed = project.get("models")
    if external.is_external(name):
        return allowed is not None and name in allowed
    return allowed is None or name in allowed


def _scored(m: dict) -> dict:
    name = m["model"]
    sc = ratings.scores(name)
    swe = m.get("swe") if m.get("swe") is not None else sc["swe"]
    aa = m.get("aa") if m.get("aa") is not None else sc["aa"]
    ext = m.get("external") or {}
    key = ratings.rating_for(name)
    return {"model": name, "kind": kind(m), "swe": swe, "aa": aa, "rating_label": sc["rating_label"],
            "rating_key": key.key if key else None,
            "price_blended": ext.get("price_blended"), "price_in": ext.get("price_in"),
            "price_out": ext.get("price_out"), "provider": ext.get("provider_label")}


def plan(project: dict, loaded: list[dict]) -> dict:
    models = [_scored(m) for m in loaded
              if m.get("model") and m.get("ready", True) and permitted(project, m["model"])]
    free = [m for m in models if m["kind"] != "external"]
    paid = [m for m in models if m["kind"] == "external"]

    free_swe = [m["swe"] for m in free if m["swe"] is not None]
    lo, hi = (min(free_swe), max(free_swe)) if free_swe else (None, None)
    free_aa = [m["aa"] for m in free if m["aa"] is not None]
    alo, ahi = (min(free_aa), max(free_aa)) if free_aa else (None, None)
    same_as = {m["rating_key"]: m["model"] for m in free if m["rating_key"]}

    search, reserved = [], []
    for m in free:
        search.append({**m, "why": "free model"})
    for m in paid:
        if m["rating_key"] in same_as:
            search.append({**m, "why": f"same model as {same_as[m['rating_key']]}"})
        elif m["swe"] is not None and lo is not None:
            if lo - SWE_MARGIN <= m["swe"] <= hi + SWE_MARGIN:
                search.append({**m, "why": f"SWE {m['swe']} is within {SWE_MARGIN:g} points of the free models' range ({lo}-{hi})"})
            else:
                side = "above" if m["swe"] > hi else "below"
                reserved.append({**m, "why": f"SWE {m['swe']} is {side} the free models' range ({lo}-{hi})"})
        elif m["aa"] is not None and alo is not None:
            if alo - AA_MARGIN <= m["aa"] <= ahi + AA_MARGIN:
                search.append({**m, "why": f"no SWE comparison; AA {m['aa']} is within {AA_MARGIN:g} points of the free models' AA range ({alo}-{ahi})"})
            else:
                side = "above" if m["aa"] > ahi else "below"
                reserved.append({**m, "why": f"no SWE comparison; AA {m['aa']} is {side} the free models' AA range ({alo}-{ahi})"})
        else:
            reserved.append({**m, "why": "no score to compare with the free models"})

    # Ladder: best free model by AA, then stronger paid models by cost, cheapest first.
    ladder = []
    best_free = max((m for m in free if m["aa"] is not None), key=lambda m: m["aa"], default=None)
    floor = -1.0
    if best_free:
        ladder.append({**best_free, "why": "highest AA Intelligence among the free models"})
        floor = best_free["aa"]
    climb = sorted((m for m in paid if m["aa"] is not None and m["aa"] > floor and m["price_blended"] is not None),
                   key=lambda m: (m["price_blended"], -m["aa"]))
    for m in climb:
        ladder.append({**m, "why": f"AA {m['aa']} is above the best free model; ${m['price_blended']}/Mtok blended"
                       if best_free else f"AA {m['aa']}; ${m['price_blended']}/Mtok blended"})
    left_out = [{**m, "why": "AA not above the best free model" if m["aa"] is not None else "no AA score"}
                for m in paid if m not in climb]

    # Operator roles override the automatic choice, model by model.
    roles = project.get("model_roles") or {}
    by_name = {m["model"]: m for m in models}
    for name, role in roles.items():
        m = by_name.get(name)
        if m is None:
            continue
        in_search = any(x["model"] == name for x in search)
        if role in ("search", "both") and not in_search:
            reserved[:] = [x for x in reserved if x["model"] != name]
            search.append({**m, "why": "you set it to Search"})
        elif role == "ideas" and in_search:
            search[:] = [x for x in search if x["model"] != name]
            reserved.append({**m, "why": "you set it to New ideas only"})
        else:
            for x in search:
                if x["model"] == name:
                    x["why"] = "you set it to Search" if role == "search" else "you set it to Both"
    marked = [by_name[n] for n, r in roles.items() if r in ("ideas", "both") and n in by_name]
    if marked:
        free_m = sorted((m for m in marked if m["kind"] != "external"), key=lambda m: -(m["aa"] if m["aa"] is not None else -1))
        paid_m = sorted((m for m in marked if m["kind"] == "external"),
                        key=lambda m: (m["price_blended"] if m["price_blended"] is not None else 1e9, -(m["aa"] or 0)))
        ladder = [{**m, "why": "you set it to New ideas" if roles[m["model"]] == "ideas" else "you set it to Both"}
                  for m in free_m + paid_m]
        left_out = [{**m, "why": "not set to New ideas"} for m in paid if m["model"] not in {x["model"] for x in ladder}]

    # Mentor: the free models set to New ideas / Both, else (on Auto) the best free model by AA
    # -- taken out of the search so its slow, careful generations go to thinking for everyone.
    mentors = []
    marked_free = [by_name[n] for n, r in roles.items() if r in ("ideas", "both") and n in by_name
                   and by_name[n]["kind"] != "external"]
    if marked_free:
        mentors = [{**m, "why": "you set it to New ideas" if roles[m["model"]] == "ideas" else "you set it to Both"}
                   for m in marked_free]
    elif best_free and roles.get(best_free["model"], "auto") == "auto":
        others = [x for x in search if x["kind"] != "external" and x["model"] != best_free["model"]]
        if len(others) >= MENTOR_MIN_SEARCHERS:
            search[:] = [x for x in search if x["model"] != best_free["model"]]
            mentors = [{**best_free, "why": (f"highest AA among the free models, with {len(others)} others searching: "
                                             "it mentors the team instead of searching (set it to Both to also search)")}]

    # Today's search budget spent: hosted models leave the search until midnight. The runner
    # retires agents whose model drops out of `search`, so this is what stops them -- before,
    # they stayed on and every iteration failed at once with the spending-limit 429 and posted
    # an error and a thought (~2000 junk board messages an hour). The ladder is untouched: the
    # ideas reserve (external.ideas_reserve_usd) is exactly for those calls.
    paused = None
    if any(m["kind"] == "external" for m in search) and external.search_budget_left() <= 0:
        limit, reserve = external.limits()
        spent = external.spent_today()
        why = (f"today's search budget is spent (${spent:.2f} of ${limit - reserve:.2f} — ${reserve:.2f} held for "
               "ideas); resumes at midnight" if reserve > 0 else
               f"today's spending limit is reached (${spent:.2f} of ${limit:.2f}); resumes at midnight")
        for m in [m for m in search if m["kind"] == "external"]:
            search.remove(m)
            reserved[:] = [x for x in reserved if x["model"] != m["model"]]
            reserved.append({**m, "why": why, "budget_paused": True})
        paused = why

    cfg = external.config()
    per_external = int(cfg.get("parallel_agents", 3))
    per_free = int(cfg.get("parallel_local_agents", 1))
    for m in search:
        m["agents"] = per_external if m["kind"] == "external" else per_free
    search_names = {m["model"] for m in search}
    ladder_names = {m["model"] for m in ladder}
    mentor_names = {m["model"] for m in mentors}
    table = [{**m, "role": roles.get(m["model"], "auto"),
              "searching": m["model"] in search_names, "ideas": m["model"] in ladder_names,
              "mentor": m["model"] in mentor_names}
             for m in models]
    return {"models": table, "search": search, "reserved": reserved, "ladder": ladder, "not_in_ladder": left_out,
            "mentors": mentors,
            "swe_range": [lo, hi], "swe_margin": SWE_MARGIN, "aa_range": [alo, ahi], "aa_margin": AA_MARGIN,
            "search_paused": paused}
