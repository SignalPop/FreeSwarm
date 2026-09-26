"""External review: a frontier model checks a result this machine's harness already passed.

The local checks are mechanical. ``lookahead`` re-runs the candidate with future rows removed
and compares the positions it takes -- which catches code that READS the future, and misses
code whose positions are computed correctly and then mis-aligned onto earlier bars. The swarm's
own auditor is a second local model given only the candidate's source, with no tools, so it
cannot open a library module the candidate imports and tends to fail anything that does.

This closes both gaps by sending the whole picture -- the hypothesis, the code, **the source of
every project library module the candidate imports**, the metrics, the split, and what the local
checks already concluded -- to a Claude model, and asking specifically about look-ahead,
overfitting and whether the score can be trusted.

Whatever comes back goes on the swarm's message board. If the reviewer says the result should
not stand, what happens next is the operator's setting:

* ``ask``  -- the verdict is returned and the operator confirms the demotion (the default:
  a review that costs money and disqualifies work should be seen before it lands).
* ``auto`` -- the demotion is applied immediately, exactly as if the operator had done it by
  hand, so an unattended swarm stops building on a leak overnight.

Either way the demotion runs through ``objectives.demote``, so the reason becomes a team lesson,
a project pitfall every future agent reads, and a board post -- there is one path for
disqualifying a result, and this is a caller of it.

The API key lives in ui/backend/auth/secrets.json with the other secrets (owner-only ACL,
git-ignored) and is never returned to the browser -- the console only ever learns whether one
is set.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import tokens
from .auth import AUTH_DIR, _restrict_permissions

logger = logging.getLogger("freetoken.review")
router = APIRouter(tags=["review"])

SECRETS_FILE = AUTH_DIR / "secrets.json"
SECRET_KEY = "anthropic:api_key"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "review.json"

_lock = threading.RLock()

# What the operator may choose between. All are frontier models; Opus 5.5 is the default (newer
# and cheaper than Opus 5), Fable 5.1 the most capable and the most expensive. Kept as a list rather than free text so the console cannot send a
# model id that the account has no access to or that this code has not been checked against.
MODELS = [
    {"id": "claude-opus-5-5", "label": "Claude Opus 5.5",
     "note": "$4 / $20 per Mtok · the default: newest Opus, cheaper than Opus 5; thinking is always on"},
    {"id": "claude-opus-5", "label": "Claude Opus 5",
     "note": "$5 / $25 per Mtok · the previous Opus"},
    {"id": "claude-fable-5-1", "label": "Claude Fable 5.1",
     "note": "$10 / $50 per Mtok · most capable; thinking is always on"},
]
MODEL_IDS = {m["id"] for m in MODELS}

# The review reads a lot (code plus every module it imports) and writes little, but it thinks
# hard first, so the ceiling is generous and the timeout long.
MAX_TOKENS = 16_000
TIMEOUT_S = 600.0
MAX_CODE_CHARS = 60_000
MAX_MODULE_CHARS = 20_000

DEFAULTS = {
    "enabled": False,
    "model": "claude-opus-5-5",
    # Set explicitly: Opus 5.5 defaults to medium when effort is omitted, one level below Opus 5.
    "effort": "high",
    # 'ask' -- return the verdict and let the operator confirm; 'auto' -- demote immediately.
    "autonomy": "ask",
    # Review every candidate that takes the title, without being asked. Off by default: it is
    # real money per champion, and a busy objective crowns often.
    "auto_review_champions": False,
}


# =======================================================================================
# Config and key
# =======================================================================================
def _read_secrets() -> dict:
    try:
        return json.loads(SECRETS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def api_key() -> str | None:
    return _read_secrets().get(SECRET_KEY) or None


def _write_key(value: str | None) -> None:
    with _lock:
        data = _read_secrets()
        if value:
            data[SECRET_KEY] = value
        else:
            data.pop(SECRET_KEY, None)
        AUTH_DIR.mkdir(parents=True, exist_ok=True)
        SECRETS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        _restrict_permissions(SECRETS_FILE)


def config() -> dict:
    try:
        saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    cfg = {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}}
    if cfg["model"] not in MODEL_IDS:
        cfg["model"] = DEFAULTS["model"]
    if cfg["autonomy"] not in ("ask", "auto"):
        cfg["autonomy"] = "ask"
    return cfg


def _save_config(cfg: dict) -> None:
    with _lock:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def available() -> bool:
    """Configured well enough to actually run a review."""
    cfg = config()
    return bool(cfg["enabled"] and api_key())


# =======================================================================================
# The review itself
# =======================================================================================
class Finding(BaseModel):
    severity: str = Field(description="one of: critical, major, minor")
    title: str
    detail: str


class Verdict(BaseModel):
    """What the reviewer must answer. Every field is required so no judgement is left implied."""
    trustworthy: bool = Field(description="Can the reported score be believed?")
    look_ahead_found: bool = Field(description="Does the strategy use information it could not have had at the time it trades?")
    look_ahead_detail: str = Field(description="If found, the exact mechanism and the lines responsible. If not, say what you checked and why it is clean.")
    disqualify: bool = Field(description="Should this result be removed from the leaderboard?")
    lesson: str = Field(description="One sentence, imperative, that would stop an agent repeating this. Empty if nothing to learn.")
    findings: list[Finding] = Field(description="Every concrete problem found, worst first. Empty list if none.")
    improvements: list[str] = Field(description="Concrete suggestions that would make the strategy genuinely better.")
    summary: str = Field(description="Two or three sentences an operator can read at a glance.")


SYSTEM = (
    "You are reviewing a quantitative trading strategy that an automated research swarm produced, "
    "before it is allowed to stand as the best known result.\n\n"
    "The harness that produced the score has ALREADY checked, mechanically, that the strategy's "
    "positions do not change when future rows are deleted, and it computes returns itself from "
    "prices rather than trusting the candidate. So do not repeat those checks. Your job is the "
    "class of problem that test cannot see, above all MIS-ALIGNMENT: positions that are computed "
    "correctly from a window and then attached to bars inside that same window. Resampling label "
    "and closed conventions, flooring a timestamp to its bin and merging on it, shifting, "
    "reindexing and forward-filling are where this hides. Work out, concretely, which timestamp's "
    "information each position is acting on, and say so.\n\n"
    "Also judge whether the score is believable: an information coefficient or Sharpe far above "
    "what the horizon plausibly supports usually means the signal and the return are "
    "contemporaneous rather than predictive.\n\n"
    "You are given the source of every project library module the strategy imports. Read them -- "
    "a leak is often in the module, not the candidate.\n\n"
    "Be concrete and quote the responsible lines. Do not disqualify a result for style, for "
    "missing tests, or for anything you have not actually demonstrated. If you cannot find a "
    "problem, say so plainly and set disqualify to false."
)


def _assemble(obj: dict, cand: dict, modules: dict[str, str]) -> str:
    m = cand.get("metrics") or {}
    parts = [
        f"OBJECTIVE: {obj['title']}",
        (obj.get("description") or "").strip(),
        "",
        f"METRIC: {obj['metric'].get('kind')} "
        f"({'higher' if obj['metric'].get('higher_is_better', True) else 'lower'} is better)",
        f"SCORE: {(m.get('holdout') or {}).get(obj['metric'].get('kind'), cand.get('score'))} on the holdout, "
        f"{cand.get('is_score')} in-sample (ranking score {cand.get('score')}: the weaker period times equity-curve "
        f"smoothness)",
        f"HOLDOUT SPLIT: {obj.get('split_date') or '(not set)'}",
        f"LOCAL LOOK-AHEAD CHECK: {cand.get('lookahead')} -- {cand.get('lookahead_detail') or 'no detail'}",
    ]
    if m:
        parts += [f"REPORTED METRICS: {json.dumps(m)[:2000]}"]
    parts += ["", f"THE AGENT'S HYPOTHESIS:\n{cand.get('rationale') or '(none given)'}", ""]
    if modules:
        parts += ["PROJECT LIBRARY MODULES THIS STRATEGY IMPORTS -- read these too, the leak is "
                  "often here rather than in the candidate. This is the complete set it can reach, "
                  "including anything since retired or quarantined:"]
        try:
            from . import library

            status = {n: st for n, (_c, st) in library.module_sources(obj["project_id"]).items()}
        except Exception:  # noqa: BLE001
            status = {}
        for name, src in modules.items():
            mark = f"  [status: {status[name]}]" if status.get(name) and status[name] != "active" else ""
            parts += [f"\n--- lib/{name}.py{mark} ---\n```python\n{src[:MAX_MODULE_CHARS]}\n```"]
        parts += [""]
    else:
        parts += ["(the strategy imports no project library modules)", ""]
    parts += [f"THE CANDIDATE'S CODE:\n```python\n{(cand.get('code') or cand.get('answer') or '')[:MAX_CODE_CHARS]}\n```"]
    return "\n".join(parts)


def _modules_for(project_id: str, code: str) -> dict[str, str]:
    """Source of every project library module the candidate imports. This is the context the
    swarm's own auditor never had, and the reason it failed candidates it could not read."""
    try:
        from . import library

        # The whole reachable set, not just the first hop: a leak one module down is still
        # the reason the score is wrong.
        return library.reachable_modules(project_id, code or "")
    except Exception:  # noqa: BLE001 -- review without them rather than not at all
        logger.exception("could not collect library modules for review")
        return {}


def _kwargs(cfg: dict, prompt: str, system: str | None = None) -> dict:
    kwargs: dict = {
        "model": cfg["model"],
        "max_tokens": MAX_TOKENS,
        "system": system or SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
        "output_format": Verdict,
        "output_config": {"effort": cfg.get("effort") or "high"},
    }
    # Fable 5.1 thinks always and rejects an explicit thinking config; Opus 5 and 5.5 think by
    # default (5.5 cannot be switched off) and accept an explicit adaptive config, which keeps
    # the intent visible. `display` is opt-in: the
    # default returns thinking blocks with empty text, which is most of a review's wall clock
    # with nothing to show for it. Summaries cost no extra tokens and are what the console
    # reports while it works.
    if cfg["model"] != "claude-fable-5-1":
        kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
    return kwargs


def _meter(cfg: dict, r) -> None:
    """Count the reviewer's tokens with every other model's (the Console's token table). Done
    before the verdict is checked: a refusal or an unusable answer was still paid for."""
    u = getattr(r, "usage", None)
    if u is None:
        return
    prompt = sum(getattr(u, k, None) or 0 for k in ("input_tokens", "cache_read_input_tokens",
                                                     "cache_creation_input_tokens"))
    tokens.record_usage("external", f"{cfg['model']}@anthropic",
                        {"input_tokens": prompt, "output_tokens": getattr(u, "output_tokens", None) or 0})


def _call(cfg: dict, key: str, prompt: str) -> tuple[Verdict, dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=key, timeout=TIMEOUT_S)
    kwargs = _kwargs(cfg, prompt)
    try:
        r = client.messages.parse(**kwargs)
        _meter(cfg, r)
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Anthropic rejected the API key") from None
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Anthropic rate limit -- try again shortly") from None
    except anthropic.APIStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic error {exc.status_code}: {exc.message}") from None
    except anthropic.APIConnectionError as exc:
        raise HTTPException(status_code=502, detail=f"could not reach Anthropic: {exc}") from None
    # A frontier model may decline; that is not a pass.
    if getattr(r, "stop_reason", None) == "refusal":
        raise HTTPException(status_code=502, detail="the reviewer declined to answer this request")
    v = r.parsed_output
    if v is None:
        raise HTTPException(status_code=502, detail="the reviewer did not return a usable verdict")
    u = getattr(r, "usage", None)
    usage = {"input_tokens": getattr(u, "input_tokens", None),
             "output_tokens": getattr(u, "output_tokens", None)} if u else {}
    return v, usage


def _finish(r) -> tuple[Verdict, dict]:
    """The checks both the blocking and the streaming call make on a finished message."""
    if getattr(r, "stop_reason", None) == "refusal":
        raise HTTPException(status_code=502, detail="the reviewer declined to answer this request")
    v = getattr(r, "parsed_output", None)
    if v is None:  # a streamed message carries the JSON but not always the parsed object
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        try:
            v = Verdict.model_validate_json(text)
        except Exception:  # noqa: BLE001
            v = None
    if v is None:
        raise HTTPException(status_code=502, detail="the reviewer did not return a usable verdict")
    u = getattr(r, "usage", None)
    return v, ({"input_tokens": getattr(u, "input_tokens", None),
                "output_tokens": getattr(u, "output_tokens", None)} if u else {})


def _call_streamed(cfg: dict, key: str, prompt: str, on_progress,
                   system: str | None = None) -> tuple[Verdict, dict]:
    """Same review, streamed, so the console can show that it is thinking and how far along.

    `on_progress({phase, output_tokens, ...})` is called as blocks arrive. The reviewer spends
    most of the wall clock thinking before it writes a word of the verdict, so a plain spinner
    looks identical to a hang -- the phase and the token count are what make it legible.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=key, timeout=TIMEOUT_S)
    phase, last, note, chars = "sending", 0.0, "", 0
    try:
        with client.messages.stream(**_kwargs(cfg, prompt, system)) as stream:
            for event in stream:
                kind = getattr(event, "type", "")
                if kind == "content_block_start":
                    block = getattr(event, "content_block", None)
                    phase = "thinking" if getattr(block, "type", "") == "thinking" else "writing"
                elif kind == "message_start":
                    phase = "thinking"
                elif kind == "content_block_delta":
                    # The snapshot's usage only lands with the final message_delta, so the
                    # live figure is counted here instead, from what has actually arrived.
                    d = getattr(event, "delta", None)
                    text = getattr(d, "thinking", None) or getattr(d, "text", None) or ""
                    chars += len(text)
                    if getattr(d, "type", "") == "thinking_delta" and text.strip():
                        note = (note + text)[-400:]
                # Throttled: one update every 400ms is enough to look alive and keeps the
                # SSE stream from carrying more traffic than the review itself.
                now = time.time()
                if now - last >= 0.4:
                    last = now
                    on_progress({
                        "phase": phase,
                        # ~4 chars per token: a progress figure, not an invoice. The exact
                        # count comes back with the final message.
                        "output_tokens": chars // 4,
                        "note": note.strip()[-180:],
                    })
            msg = stream.get_final_message()
            _meter(cfg, msg)
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Anthropic rejected the API key") from None
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Anthropic rate limit -- try again shortly") from None
    except anthropic.APIStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic error {exc.status_code}: {exc.message}") from None
    except anthropic.APIConnectionError as exc:
        raise HTTPException(status_code=502, detail=f"could not reach Anthropic: {exc}") from None
    return _finish(msg)


class ReviewReq(BaseModel):
    # Overrides for one run; the saved config is the default.
    model: str | None = None
    autonomy: str | None = None


SYSTEM_MODULE = (
    "You are reviewing a reusable signal module from an automated research swarm's code library, "
    "before more results are built on it.\n\n"
    "This is not one candidate's script -- it is a module that many candidates import, so a defect "
    "here has already contaminated every result that used it and will contaminate every future one. "
    "Judge the module on its own terms.\n\n"
    "Look above all for MIS-ALIGNMENT: a value computed from a window and then attached to bars "
    "inside that same window. Resampling label and closed conventions, flooring a timestamp to its "
    "bin and merging on it, shifting, reindexing and forward-filling are where this hides. Work out, "
    "concretely, which timestamp's information each output value is acting on, and say so. "
    "`df.resample(rule)` in pandas labels a bar with the START of its interval while `.agg('last')` "
    "fills it from the END -- attaching that to bars inside the interval is look-ahead.\n\n"
    "Also judge whether the module's own evidence is plausible: scores far above what the horizon "
    "supports usually mean the signal and the return are contemporaneous rather than predictive.\n\n"
    "You are given the source of every project library module this one imports. Read them -- the "
    "defect may be one level down.\n\n"
    "Be concrete and quote the responsible lines. Do not disqualify for style, for missing tests, or "
    "for anything you have not demonstrated. `disqualify` here means: retire this module and "
    "disqualify every result built on it. If the module is sound, say so plainly and set it false."
)


def _assemble_module(project_id: str, name: str, mod: dict, deps: dict[str, str]) -> str:
    ev = mod.get("evidence") or {}
    parts = [
        f"LIBRARY MODULE: {name}  (kind: {mod.get('kind')}, version {mod.get('version')})",
        (mod.get("description") or "").strip(),
        "",
        f"EVIDENCE FROM THE SWARM: used by {ev.get('uses', 0)} candidates, "
        f"{ev.get('champions', 0)} of which took the lead; "
        f"{ev.get('lookahead_fails', 0)} were caught by the harness's look-ahead test. "
        f"Best in-sample score seen: {ev.get('best_in_sample')}; best holdout: {ev.get('best_holdout')}.",
    ]
    if mod.get("test_output"):
        parts += ["", f"ITS OWN TEST OUTPUT:\n{str(mod['test_output'])[:2000]}"]
    others = {k: v for k, v in deps.items() if k != name}
    if others:
        parts += ["", "MODULES THIS ONE IMPORTS -- read these too, the defect may be here:"]
        for dname, src in others.items():
            parts += [f"\n--- lib/{dname}.py ---\n```python\n{src[:MAX_MODULE_CHARS]}\n```"]
    parts += ["", f"THE MODULE UNDER REVIEW:\n```python\n{(mod.get('code') or '')[:MAX_CODE_CHARS]}\n```"]
    return "\n".join(parts)


class ModuleReviewReq(BaseModel):
    model: str | None = None
    # 'ask' returns the verdict for the operator to confirm; 'auto' retires immediately.
    autonomy: str | None = None


@router.post("/projects/{project_id}/library/{name}/review/stream")
async def review_module_streamed(project_id: str, name: str, req: ModuleReviewReq):
    """Review one library module, and optionally retire it and everything built on it."""
    import asyncio

    from . import library, objectives as O

    cfg = config()
    key = api_key()
    if not key:
        raise HTTPException(status_code=400, detail="no Anthropic API key is set -- add one in Settings")
    if req.model:
        if req.model not in MODEL_IDS:
            raise HTTPException(status_code=400, detail=f"unknown model {req.model!r}")
        cfg = {**cfg, "model": req.model}
    autonomy = req.autonomy or cfg["autonomy"]
    if autonomy not in ("ask", "auto"):
        raise HTTPException(status_code=400, detail="autonomy must be 'ask' or 'auto'")

    mod = library.get_module(project_id, name)
    deps = library.reachable_modules(project_id, mod.get("code") or "")
    prompt = _assemble_module(project_id, name, mod, deps)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_progress(ev: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", **ev})

    async def run() -> None:
        started = time.time()
        try:
            verdict, usage = await asyncio.to_thread(
                _call_streamed, cfg, key, prompt, on_progress, SYSTEM_MODULE)
            body = [f"MODULE REVIEW of lib.{name} by {cfg['model']}: "
                    + ("RETIRE" if verdict.disqualify else
                       "look-ahead found" if verdict.look_ahead_found else "no blocking problem found"),
                    "", verdict.summary.strip()]
            if verdict.look_ahead_detail.strip():
                body += ["", f"Look-ahead: {verdict.look_ahead_detail.strip()}"]
            if verdict.findings:
                body += ["", "Findings:"] + [f"- [{f.severity}] {f.title}: {f.detail}" for f in verdict.findings]
            if verdict.improvements:
                body += ["", "Suggested improvements:"] + [f"- {s}" for s in verdict.improvements]
            text = "\n".join(body)

            # The verdict is a comment on the module either way, so the team reads it.
            await asyncio.to_thread(
                library.add_comment, project_id, name,
                "broken" if verdict.disqualify else "works" if verdict.trustworthy else "note",
                text[:8000], cfg["model"], mod.get("version"))
            O._board_post(project_id, "results", cfg["model"], text,
                          {"module": name, "module_review": True, "disqualify": verdict.disqualify})

            retired = None
            if verdict.disqualify and autonomy == "auto":
                retired = await asyncio.to_thread(
                    O.retire_modules, project_id, [name],
                    verdict.lesson.strip() or verdict.summary.strip()[:600], cfg["model"])

            await queue.put({"type": "done", "result": {
                "model": cfg["model"], "seconds": round(time.time() - started, 1), "usage": usage,
                "module": name, "autonomy": autonomy,
                "verdict": verdict.model_dump(), "board_text": text,
                "modules_reviewed": sorted(deps),
                "retired": retired,
                "awaiting_confirmation": bool(verdict.disqualify and autonomy == "ask"),
            }})
        except HTTPException as exc:
            await queue.put({"type": "error", "error": str(exc.detail)})
        except Exception as exc:  # noqa: BLE001 -- the stream must report, not hang
            logger.exception("module review failed")
            await queue.put({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            await queue.put(None)

    async def events():
        task = asyncio.create_task(run())
        yield _sse({"type": "start", "model": cfg["model"], "module": name,
                    "modules": sorted(k for k in deps if k != name), "prompt_chars": len(prompt)})
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield _sse(item)
        finally:
            task.cancel()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class RetireReq(BaseModel):
    reason: str = Field(..., min_length=1, max_length=8000)
    reviewer: str = Field("operator", max_length=200)


@router.post("/projects/{project_id}/library/{name}/retire")
async def retire_module(project_id: str, name: str, req: RetireReq) -> dict:
    """Quarantine a module and disqualify every result built on it, across the project."""
    import asyncio

    from . import objectives as O

    return await asyncio.to_thread(O.retire_modules, project_id, [name], req.reason, req.reviewer)


def _prepare(oid: str, cid: str, req: ReviewReq):
    """Everything settled before the reviewer is called: config, candidate, prompt."""
    from . import objectives as O

    cfg = config()
    key = api_key()
    if not key:
        raise HTTPException(status_code=400, detail="no Anthropic API key is set -- add one in Settings")
    if req.model:
        if req.model not in MODEL_IDS:
            raise HTTPException(status_code=400, detail=f"unknown model {req.model!r}")
        cfg = {**cfg, "model": req.model}
    autonomy = req.autonomy or cfg["autonomy"]
    if autonomy not in ("ask", "auto"):
        raise HTTPException(status_code=400, detail="autonomy must be 'ask' or 'auto'")

    obj = O.get_objective(oid)
    cand = O.get_candidate(cid)
    if cand["objective_id"] != oid:
        raise HTTPException(status_code=404, detail="candidate belongs to another objective")

    modules = _modules_for(obj["project_id"], cand.get("code") or "")
    return cfg, key, autonomy, obj, cand, modules, _assemble(obj, cand, modules)


@router.post("/objectives/{oid}/candidates/{cid}/review")
async def review_candidate(oid: str, cid: str, req: ReviewReq) -> dict:
    """Send one candidate to Claude and act on what comes back."""
    import asyncio

    cfg, key, autonomy, obj, cand, modules, prompt = _prepare(oid, cid, req)
    started = time.time()
    verdict, usage = await asyncio.to_thread(_call, cfg, key, prompt)
    return await _apply(oid, cid, cfg, autonomy, obj, cand, verdict, usage,
                        time.time() - started, modules)


async def _apply(oid: str, cid: str, cfg: dict, autonomy: str, obj: dict, cand: dict,
                 verdict: Verdict, usage: dict, seconds: float, modules: dict) -> dict:
    """Post the verdict to the board, act on it if allowed, and shape the reply."""
    from . import objectives as O

    head = "DISQUALIFIED" if verdict.disqualify else ("look-ahead found" if verdict.look_ahead_found
                                                      else "no blocking problem found")
    body = [
        f"EXTERNAL REVIEW of #{cand['seq']} by {cfg['model']}: {head}",
        "",
        verdict.summary.strip(),
    ]
    if verdict.look_ahead_detail.strip():
        body += ["", f"Look-ahead: {verdict.look_ahead_detail.strip()}"]
    if verdict.findings:
        body += ["", "Findings:"] + [f"- [{f.severity}] {f.title}: {f.detail}" for f in verdict.findings]
    if verdict.improvements:
        body += ["", "Suggested improvements:"] + [f"- {s}" for s in verdict.improvements]
    O._board_post(obj["project_id"], "results", cfg["model"], "\n".join(body),
                  {"objective_id": oid, "candidate_id": cid, "seq": cand["seq"],
                   "external_review": True, "disqualify": verdict.disqualify})

    demoted = None
    if verdict.disqualify and autonomy == "auto":
        finding = "\n".join(body[2:]).strip()
        demoted = await O.demote(oid, cid, O.Demote(
            finding=finding,
            lesson=verdict.lesson.strip() or f"Disqualified by {cfg['model']}: {verdict.summary.strip()[:300]}",
            reviewer=cfg["model"]))

    return {
        "model": cfg["model"], "seconds": round(seconds, 1), "usage": usage,
        "autonomy": autonomy,
        "verdict": verdict.model_dump(),
        "board_text": "\n".join(body),
        "demoted": demoted,
        # What the reviewer actually read, so the console can say so rather than imply it.
        "modules_reviewed": sorted(modules),
        # 'ask' and a disqualifying verdict: the console offers the demotion pre-filled.
        "awaiting_confirmation": bool(verdict.disqualify and autonomy == "ask"),
    }


@router.post("/objectives/{oid}/candidates/{cid}/review/stream")
async def review_candidate_streamed(oid: str, cid: str, req: ReviewReq):
    """The same review as an SSE stream, so the console can show it working.

    A review is a minute of silence on one POST. Streaming does not make it faster, but it
    turns a frozen button into a phase and a token count, which is the difference between
    'thinking' and 'hung'.
    """
    import asyncio

    cfg, key, autonomy, obj, cand, modules, prompt = _prepare(oid, cid, req)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_progress(ev: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", **ev})

    async def run() -> None:
        started = time.time()
        try:
            verdict, usage = await asyncio.to_thread(_call_streamed, cfg, key, prompt, on_progress)
            done = await _apply(oid, cid, cfg, autonomy, obj, cand, verdict, usage,
                                time.time() - started, modules)
            await queue.put({"type": "done", "result": done})
        except HTTPException as exc:
            await queue.put({"type": "error", "error": str(exc.detail)})
        except Exception as exc:  # noqa: BLE001 -- the stream must report, not hang
            logger.exception("streamed review failed")
            await queue.put({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            await queue.put(None)

    async def events():
        task = asyncio.create_task(run())
        # Tell the console what it is waiting for before the first token arrives.
        yield _sse({"type": "start", "model": cfg["model"], "seq": cand["seq"],
                    "modules": sorted(modules), "prompt_chars": len(prompt)})
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield _sse(item)
        finally:
            task.cancel()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


# =======================================================================================
# Console config API
# =======================================================================================
@router.get("/review/config")
async def read_config() -> dict:
    return {**config(), "key_set": bool(api_key()), "models": MODELS}


class ConfigReq(BaseModel):
    enabled: bool | None = None
    model: str | None = None
    effort: str | None = None
    autonomy: str | None = None
    auto_review_champions: bool | None = None
    # Write-only. None leaves it alone; "" clears it.
    api_key: str | None = Field(None, max_length=400)


@router.put("/review/config")
async def write_config(req: ConfigReq) -> dict:
    cfg = config()
    if req.model is not None:
        if req.model not in MODEL_IDS:
            raise HTTPException(status_code=400, detail=f"unknown model {req.model!r}")
        cfg["model"] = req.model
    if req.autonomy is not None:
        if req.autonomy not in ("ask", "auto"):
            raise HTTPException(status_code=400, detail="autonomy must be 'ask' or 'auto'")
        cfg["autonomy"] = req.autonomy
    if req.effort is not None:
        if req.effort not in ("low", "medium", "high", "xhigh", "max"):
            raise HTTPException(status_code=400, detail="effort must be low, medium, high, xhigh or max")
        cfg["effort"] = req.effort
    if req.enabled is not None:
        cfg["enabled"] = req.enabled
    if req.auto_review_champions is not None:
        cfg["auto_review_champions"] = req.auto_review_champions
    _save_config(cfg)
    if req.api_key is not None:
        _write_key(req.api_key.strip() or None)
    return {**config(), "key_set": bool(api_key()), "models": MODELS}


@router.post("/review/test")
async def test_key() -> dict:
    """One cheap call, to prove the key and model work before a real review depends on them."""
    import asyncio

    import anthropic

    key = api_key()
    if not key:
        raise HTTPException(status_code=400, detail="no API key is set")
    cfg = config()

    def go() -> str:
        client = anthropic.Anthropic(api_key=key, timeout=60.0)
        # Thinking is always on for Opus 5.5 / Fable 5.1 and counts toward max_tokens, so the
        # ceiling leaves room for it; low effort keeps the check cheap and quick.
        r = client.messages.create(
            model=cfg["model"], max_tokens=1024, output_config={"effort": "low"},
            messages=[{"role": "user", "content": "Reply with the single word: ready"}])
        return next((b.text for b in r.content if b.type == "text"), "")

    try:
        text = await asyncio.to_thread(go)
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Anthropic rejected the API key") from None
    except anthropic.APIStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic error {exc.status_code}: {exc.message}") from None
    except anthropic.APIConnectionError as exc:
        raise HTTPException(status_code=502, detail=f"could not reach Anthropic: {exc}") from None
    return {"ok": True, "model": cfg["model"], "reply": text.strip()[:80]}
