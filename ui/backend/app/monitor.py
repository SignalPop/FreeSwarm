"""The monitoring agent: reads every agent's log (the agent inspector's records) and the engines'
state, and files what it finds in the bug list (app/bugs.py).

Two layers:

* Detectors -- plain rules, run every TICK_S over everything the inspector holds. They know the
  difference between the PLATFORM failing and an agent's own mistake, because the difference
  decides what is worth a developer's time:
    - a traceback whose last own frame is the harness (/work/.ft/ft.py) is a platform fault;
      one in the agent's candidate.py is its own bug, filed only when the same error recurs;
    - code that fails to compile because it stops mid-token was cut off on its way from the
      model (a provider truncating tool-call arguments), not written wrong;
    - control-plane errors (5xx), unreachable engines, spending limits, exhausted experiment
      budgets, model replies or tool calls stuck for minutes, agents that stop reporting,
      iterations that keep ending without a submission, engines that crashed.
  Every finding is fingerprinted, and every occurrence keyed by the record and event it came
  from, so a rescan never counts twice (the scan is stateless and survives restarts).

* Triage (optional, `llm_triage`) -- a loaded model writes each new bug up (cause, suggested
  fix, severity, priority) and reads finished iterations for problems no rule catches: bad or
  missing data, tools contradicting the brief, results that look corrupt. A few calls per tick,
  local models only unless one is named in the settings.

Off by default; the toggle is read every tick (prefs.get_monitor), so it applies at once.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Awaitable, Callable

from fastapi import APIRouter
from pydantic import BaseModel, Field

from . import agent_activity, bugs, prefs

logger = logging.getLogger("freetoken.monitor")
router = APIRouter(tags=["monitor"])

CompleteFn = Callable[[str, list[dict], int, str], Awaitable[str]]
LoadedFn = Callable[[], list[dict]]
EnginesFn = Callable[[], list[dict]]

TICK_S = 60.0
RECENT_S = 6 * 3600.0          # stalls: only agents heard from this recently (older ones were stopped)
REVIEW_WITHIN_S = 24 * 3600.0  # triage reviews iterations that ended this recently
TRIAGE_PER_TICK = 2
REVIEWS_PER_TICK = 1
PURPOSE = "monitor"

_complete: CompleteFn | None = None
_loaded: LoadedFn | None = None
_engines: EnginesFn | None = None
_scan_lock = asyncio.Lock()
_state: dict[str, Any] = {"running": False, "last_scan": None, "last_error": None, "last_findings": 0,
                          "last_new": 0, "scans": 0, "llm_calls": 0, "model": None, "llm_note": None}


# =============================================================================================
# Reading tool results
# =============================================================================================
_FRAME = re.compile(r'File "([^"]+)", line (\d+)')
_ERRLINE = re.compile(r"^\s*([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt))(?::\s?(.*))?$", re.M)
_API_ERR = re.compile(r"(/[\w/.\-{}]+) -> (\d{3}): (.*)", re.S)
_TRUNCATED = ("unterminated string literal", "was never closed", "unexpected eof", "unterminated triple-quoted",
              "expected an indented block", "incomplete input")
_AGENT_FILES = ("candidate.py", "<string>", "/.ft/lib/", "/.ft/research/", "/.ft/members/")
_HARNESS_FILES = ("/.ft/ft.py", "/opt/ft/", "bootstrap.py")


def _as_text(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, default=str)


def _parse(v: Any) -> dict | None:
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.lstrip().startswith("{"):
        try:
            d = json.loads(v)
            return d if isinstance(d, dict) else None
        except ValueError:
            return None
    return None


def traceback_of(text: str) -> dict | None:
    """{etype, message, origin: harness | agent | library, frame, line} of the LAST traceback in
    `text`, or None. `origin` is the deepest frame that is not a third-party package."""
    if "Traceback" not in text and "Error:" not in text:
        return None
    errs = list(_ERRLINE.finditer(text))
    if not errs:
        return None
    last = errs[-1]
    etype, msg = last.group(1).rsplit(".", 1)[-1], (last.group(2) or "").strip()
    frames = [(f, int(n)) for f, n in _FRAME.findall(text[: last.start()])]
    own = [f for f in frames if "site-packages" not in f[0] and "/lib/python" not in f[0] and "<frozen" not in f[0]]
    # The deepest frame that is not a third-party package decides: an error raised inside polars
    # from the agent's own call is the agent's; one raised in (or through) ft.py is the harness's.
    frame = own[-1] if own else (frames[-1] if frames else ("", 0))
    path = frame[0].replace("\\", "/")
    if any(h in path for h in _HARNESS_FILES) and not any(a in path for a in _AGENT_FILES):
        origin = "harness"
    elif any(a in path for a in _AGENT_FILES) or path in ("script.py", ""):
        origin = "agent"
    else:
        origin = "library"
    return {"etype": etype, "message": msg[:500], "origin": origin, "frame": path, "line": frame[1]}


def _code_of(args: Any) -> str:
    if isinstance(args, dict):
        for k in ("code", "script", "source"):
            if isinstance(args.get(k), str):
                return args[k]
        return json.dumps(args, default=str, indent=1) if args else ""
    return _as_text(args or "")


def _is_truncated(tb: dict, code: str) -> bool:
    """Code that simply stops: a SyntaxError at its last line saying so, or an error at the last
    line naming the half-written word the code ends on (`df.sort_valu` -> no attribute 'sort_valu')."""
    lines = code.rstrip("\n").count("\n") + 1 if code else 0
    at_end = not lines or tb["line"] >= lines - 1
    if tb["etype"] in ("SyntaxError", "IndentationError"):
        return at_end and any(t in tb["message"].lower() for t in _TRUNCATED)
    if tb["origin"] != "agent" or not code or code.endswith("\n") or not at_end:
        return False
    tail = re.search(r"([A-Za-z_]\w*)$", code)
    return bool(tail) and f"'{tail.group(1)}'" in tb["message"]


# =============================================================================================
# Detectors: one record's events -> findings
# =============================================================================================
def _where(a: dict, r: dict) -> dict:
    obj = r.get("objective") or {}
    return {"project_id": a.get("project_id"), "agent": a.get("agent"), "model": a.get("model"),
            "objective_id": obj.get("id"), "objective_title": obj.get("title"), "record_id": r.get("id")}


def _finding(where: dict, key: str, at: float, **f: Any) -> dict:
    out = {**where, "key": key, "at": at}
    out.update(f)
    ctx = {k: where.get(k) for k in ("agent", "model", "project_id", "objective_title", "record_id")}
    out["context"] = {**ctx, **(f.get("context") or {})}
    return out


def tool_findings(a: dict, r: dict, i: int, e: dict, chat_model: str | None) -> list[dict]:
    name = e.get("name") or "?"
    raw = e.get("result")
    text = _as_text(raw)
    d = _parse(raw) or {}
    at = float(e.get("at") or r.get("started_at") or time.time())
    w = _where(a, r)
    key = f"{r.get('id')}:{i}:{at}"
    code = _code_of(e.get("args"))
    base = dict(tool=name, script=code, evidence=text, context={"tool_ok": e.get("ok"), "seconds": e.get("seconds"),
                                                                "chat_model": chat_model})
    err = d.get("error") if isinstance(d.get("error"), str) else None

    # Control-plane errors surfaced as {"error": "/api/... -> NNN: detail"}.
    if err and not e.get("ok"):
        if "experiment budget used" in err:
            return [_finding(w, key, at, fingerprint="budget:experiments", category="stall", severity="low",
                             priority="P3", min_occurrences=3, title="Agents run out of run_python experiments",
                             description="An agent used its whole experiment budget for one iteration and was told "
                                         "to submit instead. Frequent hits mean experiments fail or repeat (look at the "
                                         "runs before it) or the budget is too small for the task.", **base)]
        m = _API_ERR.search(err)
        if m and m.group(2).startswith("5"):
            return [_finding(w, key, at, fingerprint=f"api5xx:{name}:{bugs.normalize(m.group(1))}:{m.group(2)}",
                             category="error", severity="high", priority="P1", min_occurrences=1,
                             title=f"{name}: control plane error {m.group(2)} on {bugs.normalize(m.group(1), 80)}",
                             description=f"The control plane failed while serving the {name} tool: "
                                         f"{m.group(3)[:400]}", **base)]
        if m:
            return [_finding(w, key, at, fingerprint=f"api4xx:{name}:{bugs.normalize(m.group(3), 90)}",
                             category="error", severity="medium", priority="P3", min_occurrences=3,
                             title=f"{name} rejected ({m.group(2)}): {m.group(3)[:120]}",
                             description="Agents keep calling this tool in a way the control plane refuses. Either "
                                         "the tool's description / the brief misleads them, or the check is too "
                                         f"strict.\n\n{m.group(3)[:600]}", **base)]
        tb = traceback_of(err)
        if tb is None:
            # The tool's own feedback to the agent ("the smoke test failed", "no candidate 1198"):
            # the agent's mistake, worth a look only when it keeps happening.
            return [_finding(w, key, at, fingerprint=f"toolerr:{name}:{bugs.normalize(err, 90, quoted=False)}",
                             category="agent_error", severity="low", priority="P4", min_occurrences=5,
                             title=f"{name} keeps failing: {err[:140]}",
                             description="Agents keep getting this answer from the tool. If it is always the same "
                                         f"mistake, the tool's description or the brief could prevent it.\n\n{err[:1000]}",
                             **base)]

    # Where a traceback can be: a failed run's stderr, a failed submission's error, or -- when the
    # runner cut a long result so it no longer parses -- the raw text.
    if d:
        src = (_as_text(d.get("stderr") or "") if d.get("ok") is False else "") or err or \
              (_as_text(d.get("error") or "") if d.get("status") == "error" else "")
    else:
        src = text if "Traceback (most recent call last)" in text else ""
    tb = traceback_of(src) if src else None
    if tb is None:
        return []

    who = chat_model or a.get("model")
    if _is_truncated(tb, code):
        return [_finding(w, key, at, fingerprint=f"truncated:{who}", category="error", severity="high", priority="P2",
                         min_occurrences=2, title=f"Code from {who} arrives cut off mid-token",
                         description=f"{name} got code that stops in the middle of a statement ({tb['etype']}: "
                                     f"{tb['message']}). The model's tool-call arguments are being truncated on the "
                                     "way (provider / tool-call parsing / max_tokens), so the run is wasted before "
                                     "it starts and the experiment budget drains.", **base)]
    if tb["origin"] == "harness":
        msg = tb["message"].lower()
        if ("available:" in msg or "valid columns" in msg) and "(none)" not in msg:
            pass                                   # a name the agent got wrong, with the right ones listed
        else:
            none = "(none)" in msg
            return [_finding(w, key, at, fingerprint=f"harness:{name}:{tb['etype']}:"
                                                     f"{bugs.normalize(tb['message'], 120, quoted=False)}",
                             category="bad_data" if none else "error", severity="high" if none else "medium",
                             priority="P1" if none else "P2", min_occurrences=1 if none else 2,
                             title=(f"Sandbox has no datasets: {tb['message'][:120]}" if none else
                                    f"Harness error in {name}: {tb['etype']}: {tb['message'][:120]}"),
                             description=f"The error was raised inside the platform's own code ({tb['frame']}, line "
                                         f"{tb['line']}), not the agent's script." +
                                         (" The sandbox's dataset catalog is empty, so every load the brief suggests "
                                          "fails." if none else ""), **base)]
    return [_finding(w, key, at, fingerprint=f"agent:{tb['etype']}:{bugs.normalize(tb['message'], 120)}",
                     category="agent_error", severity="low", priority="P4", min_occurrences=5,
                     title=f"Recurring agent error: {tb['etype']}: {tb['message'][:120]}",
                     description="The same error keeps coming back in agents' own code. One is a mistake; many "
                                 "point at a gap in the brief, the tool docs or the ft helper (an API the models "
                                 "expect that is not there).", **base)]


def chat_findings(a: dict, r: dict, i: int, e: dict, cfg: dict) -> list[dict]:
    model = e.get("model") or a.get("model")
    at = float(e.get("at") or r.get("started_at") or time.time())
    w = {**_where(a, r), "model": model}
    key = f"{r.get('id')}:{i}:{at}"
    out: list[dict] = []
    err = str(e.get("error") or "")
    ctx = {"seconds": e.get("seconds"), "prompt_tokens": e.get("prompt_tokens"),
           "completion_tokens": e.get("completion_tokens"), "finish": e.get("finish")}
    if err:
        low = err.lower()
        if "spending limit" in low or "-> 429" in err:
            out.append(_finding(w, key, at, fingerprint="chat:spending-limit", category="stall", severity="medium",
                                priority="P2", min_occurrences=1, title="External model spending limit reached",
                                description="Requests to an external model are refused because a spending limit is "
                                            "reached; the agents using it stop making progress until it resets or "
                                            "is raised.", evidence=err, context=ctx))
        elif ("-> 504" in err or "did not finish the reply" in low
              or (re.search(r"unreachable:\s*$", err) and float(e.get("seconds") or 0) >= 590)):
            # Reached but too slow: the proxy's reply timeout ran out (older builds reported that
            # as "engine unreachable: " with nothing after it, at exactly 600 s).
            out.append(_finding(w, key, at, fingerprint=f"chat:timeout:{model}", category="stall", severity="high",
                                priority="P2", min_occurrences=1, title=f"Replies from {model} time out",
                                description="A reply took longer than the proxy waits, so it was cut off and the "
                                            "work done on it thrown away. The engine is overloaded: too many agents "
                                            "on it, very long prompts, or long reasoning. Fewer agents per model, a "
                                            "smaller max_tokens or a faster model helps.", evidence=err, context=ctx))
        elif "unreachable" in low or "-> 502" in err or "-> 503" in err:
            out.append(_finding(w, key, at, fingerprint=f"chat:unreachable:{model}", category="error", severity="high",
                                priority="P1", min_occurrences=1, title=f"Model {model} unreachable",
                                description="Chat requests to this model failed: its engine is down or not "
                                            "answering. Every agent on it is idle.", evidence=err, context=ctx))
        elif "failed to call a function" in low or "tool_use_failed" in low:
            out.append(_finding(w, key, at, fingerprint=f"chat:toolcall-format:{model}", category="error",
                                severity="medium", priority="P3", min_occurrences=3,
                                title=f"{model} produces tool calls the server cannot parse",
                                description="The provider rejected the model's tool call as malformed. Each one "
                                            "costs a turn.", evidence=err, context=ctx))
        else:
            m = re.search(r"-> (\d{3})", err)
            out.append(_finding(w, key, at, fingerprint=f"chat:{m.group(1) if m else 'err'}:{model}:{bugs.normalize(err, 80)}",
                                category="error", severity="medium", priority="P2", min_occurrences=2,
                                title=f"Chat request to {model} failed: {err[:120]}", description=err[:1000],
                                evidence=err, context=ctx))
    if e.get("finish") == "length":
        out.append(_finding(w, key + ":len", at, fingerprint=f"chat:length:{model}", category="error", severity="low",
                            priority="P3", min_occurrences=5, title=f"Replies from {model} cut off at max_tokens",
                            description="The model hit its token limit mid-reply; a tool call or answer in it is "
                                        "lost or truncated.", evidence=_as_text(e.get("said") or "")[-2000:],
                            context=ctx))
    secs = float(e.get("seconds") or 0)
    if secs >= float(cfg.get("slow_reply_s") or 600):
        out.append(_finding(w, key + ":slow", at, fingerprint=f"slow:{model}", category="stall", severity="low",
                            priority="P3", min_occurrences=3, title=f"Very slow replies from {model}",
                            description=f"Single replies take {secs:.0f}s or more. With several agents on one engine "
                                        "each waits for the others; fewer agents per model, a smaller context or "
                                        "a faster model would speed the search up.", evidence=json.dumps(ctx),
                            context=ctx))
    return out


# No request can still be in flight after this long: the runner gives up on a reply after 1800 s
# (FREESWARM_SWARM_TIMEOUT_S; the proxy waits as long). A record still "waiting" past it was
# abandoned -- its worker was retired or the runner stopped -- and calling that a stall sent the
# operator after models that were fine (bugs #70, #72: "waited 60 min").
ABANDONED_S = 1800.0 + 300.0


def record_findings(a: dict, r: dict, now: float, cfg: dict, loaded: set[str] | None = None) -> list[dict]:
    """Findings about the record as a whole: stalls and outcomes. `loaded`: the models loaded now
    (None = unknown) -- an agent whose model is no longer loaded was retired, not stalled."""
    out: list[dict] = []
    w = _where(a, r)
    status = r.get("status")
    updated = float(a.get("updated_at") or 0)
    stall_s = 60.0 * float(cfg.get("stall_minutes") or 15)
    newest = (a.get("records") or [{}])[-1].get("id") == r.get("id")
    model_gone = loaded is not None and a.get("model") not in loaded
    if status == "running" and newest and now - updated < RECENT_S and not model_gone:
        p = r.get("pending") or {}
        since = float(p.get("since") or 0)
        if since and now - since > ABANDONED_S:
            since = 0.0                          # not waiting any more: judged as a silent agent below
        if since and now - since > stall_s:
            what = p.get("model") if p.get("kind") == "chat" else p.get("name")
            mins = (now - since) / 60
            out.append(_finding(w, f"{r.get('id')}:pending:{since}", since,
                                fingerprint=f"stall:{p.get('kind')}:{what}", category="stall", severity="high",
                                priority="P2", min_occurrences=1,
                                title=(f"Agents stuck waiting on {what}" if p.get("kind") == "chat"
                                       else f"Tool call {what} hangs"),
                                description=f"{a.get('agent')} has waited {mins:.0f} min for a "
                                            f"{'reply from ' if p.get('kind') == 'chat' else 'result of '}{what} "
                                            f"with no progress.", tool=p.get("name"),
                                script=_code_of(p.get("args")) if p.get("args") else "",
                                evidence=json.dumps(p, default=str), context={"pending_minutes": round(mins, 1)}))
        elif not since and now - updated > max(2 * stall_s, 1800.0):
            out.append(_finding(w, f"{r.get('id')}:silent", updated, fingerprint=f"silent:{a.get('project_id')}",
                                category="stall", severity="medium", priority="P2", min_occurrences=1,
                                title="Agents stop reporting mid-iteration",
                                description=f"{a.get('agent')} has not reported for {(now - updated) / 60:.0f} min "
                                            "while its iteration is still open. The swarm runner may have died or "
                                            "been stopped without closing its work.",
                                evidence=json.dumps({"status": status, "updated_at": updated}), context={}))
    if status == "no submission" and r.get("mode") not in ("mentor", "task"):
        tools = [e.get("name") for e in r.get("timeline") or [] if e.get("kind") == "tool"]
        out.append(_finding(w, f"{r.get('id')}:outcome", float(r.get("ended_at") or r.get("started_at") or now),
                            fingerprint=f"nosubmit:{a.get('model')}", category="stall", severity="low",
                            priority="P3", min_occurrences=5,
                            title=f"Iterations of {a.get('model')} end without a submission",
                            description="Whole iterations finish without a candidate, so their compute buys nothing. "
                                        "Look at what the last tool calls were: failing experiments, a used-up "
                                        "budget or a model that never gets to submit_candidate.",
                            evidence=json.dumps({"tools": tools[-15:]}), context={"tool_calls": len(tools)}))
    return out


def engine_findings(engines: list[dict]) -> list[dict]:
    out = []
    for s in engines:
        if s.get("state") != "error":
            continue
        diag = s.get("diagnosis") or {}
        summary = s.get("error") or diag.get("summary") or "the engine exited"
        model = s.get("model_id") or "?"
        at = float(s.get("started_at") or time.time())
        out.append({"key": f"{model}:{at}", "at": at, "model": model, "project_id": None,
                    "fingerprint": f"engine:{model}:{bugs.normalize(summary, 100)}", "category": "error",
                    "severity": "critical", "priority": "P1", "min_occurrences": 1,
                    "title": f"Engine failed to run {model}: {summary[:140]}",
                    "description": f"The engine for {model} exited (code {diag.get('exit_code')}). {summary}"
                                   + (f"\n\nHint: {diag['hint']}" if diag.get("hint") else ""),
                    "script": s.get("command") or "", "evidence": "\n".join(diag.get("tail") or [])[-8000:],
                    "tool": "engine", "context": {"gpus": s.get("gpus"), "port": s.get("port"),
                                                  "model_path": s.get("model_path"), "exit_code": diag.get("exit_code")}})
    return out


def scan(agents: list[dict], engines: list[dict], cfg: dict, now: float | None = None,
         loaded: set[str] | None = None) -> list[dict]:
    """Every finding in the current logs (pure: no I/O). `loaded`: model names loaded now."""
    now = now or time.time()
    out: list[dict] = []
    for a in agents:
        for r in a.get("records") or []:
            chat_model = None
            for i, e in enumerate(r.get("timeline") or []):
                kind = e.get("kind")
                try:
                    if kind == "chat":
                        chat_model = e.get("model") or chat_model
                        out += chat_findings(a, r, i, e, cfg)
                    elif kind == "tool":
                        out += tool_findings(a, r, i, e, chat_model)
                except Exception:  # noqa: BLE001 -- one odd event must not blind the monitor
                    logger.debug("monitor: event %s of %s skipped", i, r.get("id"), exc_info=True)
            out += record_findings(a, r, now, cfg, loaded)
    out += engine_findings(engines)
    return out


# =============================================================================================
# Seeing a bug fixed
# =============================================================================================
# A bug is closed as fixed only on evidence: since its last occurrence the thing it happened in
# (the same tool, the same model's replies, the same kind of iteration) ran again enough times
# without it, and a while has passed. Numbers per kind: record-level problems get fewer chances
# (an iteration is long), an engine crash is fixed the moment that model runs again.
RECORD_CHANCES = 5
_TOOL_KINDS = {"harness", "api5xx", "api4xx", "toolerr", "agent", "budget"}
_CHAT_KINDS = {"chat", "slow"}


def fix_chances(bug: dict, agents: list[dict], engines: list[dict]) -> tuple[int, int, str] | None:
    """(chances since the last occurrence, chances needed, what they were) -- None when this kind
    of bug cannot be judged from the logs (a model's review, one filed by hand)."""
    fp = bug.get("fingerprint") or ""
    kind = fp.split(":", 1)[0]
    since = float(bug.get("last_seen") or 0)
    model, tool, pid = bug.get("model"), bug.get("tool"), bug.get("project_id")
    mine = [a for a in agents if not pid or a.get("project_id") == pid]
    if kind == "engine":
        up = any(s.get("model_id") == model and s.get("state") == "running" and float(s.get("started_at") or 0) > since
                 for s in engines)
        return (1 if up else 0), 1, f"{model} loaded and running again"
    if kind in _TOOL_KINDS or kind == "truncated" or (kind == "stall" and fp.startswith("stall:tool:")):
        name = "run_python" if kind == "budget" else tool
        n = 0
        for a in mine:
            for r in a.get("records") or []:
                chat_model = None
                for e in r.get("timeline") or []:
                    if e.get("kind") == "chat":
                        chat_model = e.get("model") or chat_model
                    elif e.get("kind") == "tool" and float(e.get("at") or 0) > since:
                        if kind == "truncated":     # code this model sent since, whatever the tool
                            args = e.get("args")
                            n += chat_model == model and isinstance(args, dict) and any(
                                isinstance(args.get(k), str) and args[k].strip() for k in ("code", "script", "source"))
                        else:
                            n += e.get("name") == name
        what = f"tool calls with code from {model}" if kind == "truncated" else f"{name} calls"
        return n, 0, what
    if kind in _CHAT_KINDS or (kind == "stall" and fp.startswith("stall:chat:")):
        who = fp.split(":", 2)[2] if kind == "stall" else model
        n = sum(1 for a in agents for r in a.get("records") or [] for e in r.get("timeline") or []
                if e.get("kind") == "chat" and float(e.get("at") or 0) > since and not e.get("error")
                and (fp == "chat:spending-limit" or e.get("model") == who))
        return n, 0, (f"replies from {who}" if fp != "chat:spending-limit" else "model replies")
    if kind in ("silent", "nosubmit"):
        n = 0
        for a in mine:
            if kind == "nosubmit" and a.get("model") != model:
                continue
            for r in a.get("records") or []:
                ended = float(r.get("ended_at") or 0)
                if ended > since and (r.get("status") == "submitted" if kind == "nosubmit" else r.get("status") != "running"):
                    n += 1
        return n, RECORD_CHANCES, ("iterations that submitted" if kind == "nosubmit" else "iterations closed normally")
    return None


def fixed_bugs(bugs_: list[dict], agents: list[dict], engines: list[dict], cfg: dict, now: float) -> list[tuple[int, str]]:
    """[(bug id, why it counts as fixed)] among `bugs_` (pure: no I/O)."""
    quiet_s = 60.0 * float(cfg.get("fixed_quiet_minutes") or 30)
    out = []
    for b in bugs_:
        got = fix_chances(b, agents, engines)
        if got is None:
            continue
        n, need, what = got
        need = need or int(cfg.get("fixed_after") or 15)
        since = float(b.get("last_seen") or now)
        is_engine = (b.get("fingerprint") or "").startswith("engine:")
        if n >= need and (is_engine or now - since >= quiet_s):
            gap = (now - since) / 60
            span = f"{gap:.0f} min" if gap < 120 else f"{gap / 60:.1f} h"
            out.append((b["id"], what if is_engine else
                        f"not seen in {n} {what} over {span} since its last occurrence. It reopens if it comes back."))
    return out


def _good_title(title: str) -> bool:
    """A model-written title worth keeping: not an echo of the prompt's options (a small model
    answered "error|bad_data|stall"), and long enough to say something."""
    t = title.strip()
    return len(t) >= 12 and "|" not in t and t.lower() not in ("error", "bad_data", "stall", "title")


# =============================================================================================
# Triage by a model
# =============================================================================================
def pick_model(cfg: dict) -> str | None:
    loaded = _loaded() if _loaded else []
    ready = [m for m in loaded if m.get("model") and m.get("ready", True)]
    names = {m["model"] for m in ready}
    if cfg.get("model"):
        return cfg["model"] if cfg["model"] in names else None
    for pool in ([m for m in ready if not m.get("external") and not m.get("remote")],
                 [m for m in ready if not m.get("external")]):
        if pool:
            return max(pool, key=lambda m: m.get("aa") or 0)["model"]
    return None


def _json_from(text: str) -> Any:
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M)
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = t.find(opener), t.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except ValueError:
                continue
    return None


_ABOUT = ("You are the monitoring agent of FreeSwarm: LLM agents search for trading strategies by calling tools "
          "(run_python runs code in a sandbox with `import ft` for data, query_data runs SQL, submit_candidate "
          "scores a script). You file bugs in the PLATFORM for the developer who maintains it.")


def _clip(s: Any, n: int) -> str:
    s = _as_text(s or "")
    return s if len(s) <= n else s[: n // 2] + "\n…\n" + s[-n // 2:]


async def _triage(model: str, bug: dict) -> None:
    prompt = (f"{_ABOUT}\n\nA detector filed this bug from the agents' logs. Write it up.\n\n"
              f"Detector title: {bug['title']}\nCategory: {bug['category']}  Severity: {bug['severity']}  "
              f"Priority: {bug['priority']}  Seen: {bug['occurrences']} times\nTool: {bug.get('tool')}  "
              f"Model: {bug.get('model')}  Objective: {bug.get('objective_title')}\n"
              f"Detector description: {bug['description']}\n\nEvidence (latest):\n{_clip(bug.get('evidence'), 4000)}\n\n"
              f"Script / inputs (latest):\n{_clip(bug.get('script'), 3000)}\n\n"
              "Reply with ONLY a JSON object: {\"title\": short and specific, \"description\": what happened and "
              "how it hurts the search (2-5 sentences), \"likely_cause\": \"...\", \"suggestion\": a concrete fix in "
              "the platform}")
    _state["llm_calls"] += 1
    text = await _complete(model, [{"role": "user", "content": prompt}], 1500, PURPOSE)
    doc = _json_from(text)
    if not isinstance(doc, dict):
        await asyncio.to_thread(bugs.mark_triaged, bug["id"])
        return
    desc = str(doc.get("description") or "").strip()
    if doc.get("likely_cause"):
        desc += f"\n\nLikely cause: {str(doc['likely_cause']).strip()}"
    # Severity and priority stay the detector's: they are calibrated per kind of problem, and a
    # small triage model marked nearly everything "critical" (a P4 critical agent mistake).
    fields = {k: doc.get(k) for k in ("title", "suggestion")}
    if not _good_title(str(fields.get("title") or "")):
        fields.pop("title")                      # keep the detector's title
    await asyncio.to_thread(bugs.enrich, bug["id"], {**fields, "description": desc}, model)


def condense(r: dict, limit: int = 12_000) -> str:
    """One iteration as a short transcript: what the model said, what each tool got and returned."""
    obj = r.get("objective") or {}
    lines = [f"mode={r.get('mode')} status={r.get('status')} objective={obj.get('title')!r}"]
    for e in r.get("timeline") or []:
        if e.get("kind") == "chat":
            lines.append(f"[model {e.get('model')} {e.get('seconds')}s finish={e.get('finish')}"
                         f"{' ERROR ' + str(e.get('error'))[:200] if e.get('error') else ''}] "
                         f"{_clip(e.get('said'), 300)}")
        elif e.get("kind") == "tool":
            lines.append(f"[tool {e.get('name')} ok={e.get('ok')} {e.get('seconds')}s] args: "
                         f"{_clip(_code_of(e.get('args')), 700)}\n  -> {_clip(e.get('result'), 900)}")
    text = "\n".join(lines)
    return text if len(text) <= limit else text[:2000] + "\n…\n" + text[-(limit - 2000):]


def _squash(text: str) -> str:
    return re.sub(r"[\s\"'`\\]+", " ", text.lower()).strip()


def _grounded(evidence: str, transcript: str) -> bool:
    """Whether a review's evidence is really in the iteration: a quote of at least 20 characters
    found in the transcript (spacing, quotes and escapes ignored). A small model filed "Sharpe
    ratio not optimal" with the evidence "tool_run_python" -- nothing the log says."""
    ev = _squash(evidence)
    if len(ev) < 20:
        return False
    hay = _squash(transcript)
    return ev[:60] in hay or any(chunk in hay for chunk in (ev[i:i + 40] for i in range(0, max(1, len(ev) - 40), 20)))


async def _review(model: str, a: dict, r: dict) -> int:
    transcript = condense(r)
    prompt = (f"{_ABOUT}\n\nRead this iteration of agent {a.get('agent')} and report problems in the PLATFORM only: "
              "wrong, missing or corrupt data (empty tables, all-NaN or constant columns, impossible values, dates "
              "out of range), tools that fail or contradict what the brief tells the agent, results that disagree "
              "with each other, and anything that wastes the agent's time. Do NOT report the agent's ordinary "
              "coding mistakes or how good its strategy is.\n\nAlready filed (do not repeat):\n"
              + "\n".join(await asyncio.to_thread(bugs.open_titles)) +
              f"\n\nIteration:\n{transcript}\n\nReply with ONLY JSON: {{\"issues\": [{{\"title\", \"category\": one of "
              "error / bad_data / stall, \"description\", \"evidence\": an EXACT quote copied from the iteration above, "
              "\"tool\"}]}} -- at most 3, or {\"issues\": []} if nothing is wrong. An issue without an exact quote is "
              "discarded.")
    _state["llm_calls"] += 1
    doc = _json_from(await _complete(model, [{"role": "user", "content": prompt}], 2000, PURPOSE))
    issues = doc.get("issues") if isinstance(doc, dict) else doc if isinstance(doc, list) else []
    n = 0
    w = _where(a, r)
    for it in (issues or [])[:3]:
        if not isinstance(it, dict) or not _good_title(str(it.get("title") or "")):
            continue                             # an echo of the prompt, or cut off: not a finding
        if not str(it.get("description") or "").strip() or not _grounded(str(it.get("evidence") or ""), transcript):
            continue                             # a claim with nothing in the log behind it
        title = str(it["title"]).strip()[:200]
        # A model's review is a lead, not a verdict: filed at a fixed, moderate level (a small model
        # rated "Sharpe ratio not optimal" P2 high); the operator raises it if it holds up.
        sev, pri = "medium", "P3"
        category = str(it.get("category") or "")
        category = category if category in ("error", "bad_data", "stall") else "bad_data"
        tool = str(it.get("tool") or "")[:80] or None
        script = next((_code_of(e.get("args")) for e in reversed(r.get("timeline") or [])
                       if e.get("kind") == "tool" and tool and e.get("name") == tool), "")
        res = await asyncio.to_thread(bugs.sight, _finding(
            w, f"{r.get('id')}:review", float(r.get("ended_at") or time.time()),
            fingerprint=f"review:{bugs.normalize(title, 80)}", category=category,
            severity=sev, priority=pri, min_occurrences=1, title=title, description=str(it.get("description") or ""),
            evidence=_as_text(it.get("evidence") or ""), tool=tool, script=script, source="monitor"))
        await asyncio.to_thread(bugs.mark_triaged, res["id"])       # the model wrote it up already
        if res["new"]:
            await asyncio.to_thread(bugs.add_note, res["id"], f"found by {model} reviewing an iteration of {a.get('agent')}")
        n += 1
    return n


def _reviewed_table() -> None:
    with bugs._lock:
        bugs.db().execute("CREATE TABLE IF NOT EXISTS monitor_reviewed (record_key TEXT PRIMARY KEY, at REAL NOT NULL)")
        bugs.db().commit()


def _to_review(agents: list[dict], now: float, limit: int) -> list[tuple[dict, dict, str]]:
    _reviewed_table()
    cands = []
    for a in agents:
        for r in a.get("records") or []:
            if r.get("status") in ("running", None) or now - float(r.get("ended_at") or 0) > REVIEW_WITHIN_S:
                continue
            if not any(e.get("kind") == "tool" for e in r.get("timeline") or []):
                continue
            cands.append((float(r.get("ended_at") or 0), a, r, f"{a.get('project_id')}|{a.get('agent')}|{r.get('id')}"))
    cands.sort(key=lambda c: c[0], reverse=True)
    out = []
    with bugs._lock:
        for _, a, r, key in cands:
            if bugs.db().execute("SELECT 1 FROM monitor_reviewed WHERE record_key=?", (key,)).fetchone() is None:
                out.append((a, r, key))
            if len(out) >= limit:
                break
    return out


def _mark_reviewed(key: str) -> None:
    with bugs._lock:
        bugs.db().execute("INSERT OR REPLACE INTO monitor_reviewed (record_key, at) VALUES (?, ?)", (key, time.time()))
        bugs.db().commit()


# =============================================================================================
# The loop
# =============================================================================================
# One model call may take this long before triage gives up on it for this pass. The models are
# the swarm's own engines, often busy for minutes; triage is a nicety and must never pin anything.
LLM_CALL_TIMEOUT_S = 300.0
_triage_task: asyncio.Task | None = None


def _start_triage(cfg: dict, agents: list[dict], now: float) -> str:
    """Run model triage in the background -- never inside a scan. The scan (and a Recheck, and
    the Scan-now button) used to wait for it, and on a busy engine that meant many minutes, with
    the lock held so every later scan and recheck queued behind it. One pass at a time: while the
    last one is still waiting on the model, this tick's is skipped."""
    global _triage_task
    if _triage_task is not None and not _triage_task.done():
        return "busy"
    _triage_task = asyncio.create_task(_triage_pass(cfg, agents, now))
    return "started"


async def _triage_pass(cfg: dict, agents: list[dict], now: float) -> None:
    model = pick_model(cfg)
    _state["model"] = model
    _state["llm_note"] = None if model else ("the configured model is not loaded" if cfg.get("model")
                                             else "no local model loaded")
    if not model:
        return
    _state["triage_running"] = True

    def failed(what: str, exc: BaseException) -> None:
        if isinstance(exc, asyncio.TimeoutError):
            _state["llm_note"] = f"{what} timed out after {LLM_CALL_TIMEOUT_S:.0f}s -- {model} is busy; next pass retries"
        else:
            _state["llm_note"] = f"{what} failed: {getattr(exc, 'detail', exc)}"[:300]

    try:
        for bug in await asyncio.to_thread(bugs.untriaged, TRIAGE_PER_TICK):
            try:
                await asyncio.wait_for(_triage(model, bug), LLM_CALL_TIMEOUT_S)
            except Exception as exc:  # noqa: BLE001 -- the model is busy or gone: next pass
                failed("triage", exc)
                return
        for a, r, key in await asyncio.to_thread(_to_review, agents, now, REVIEWS_PER_TICK):
            try:
                await asyncio.wait_for(_review(model, a, r), LLM_CALL_TIMEOUT_S)
                await asyncio.to_thread(_mark_reviewed, key)
            except Exception as exc:  # noqa: BLE001
                failed("review", exc)
                return
    except Exception:  # noqa: BLE001 -- a background pass must never take the monitor down
        logger.exception("monitor triage pass failed")
    finally:
        _state["triage_running"] = False
        _state["last_triage"] = time.time()


async def scan_once(force: bool = False, llm: bool = True) -> dict:
    """One pass of the rules over everything (seconds), then -- if on, and `llm` -- model triage
    started in the background, never waited for."""
    cfg = prefs.get_monitor()
    if not cfg["enabled"] and not force:
        return {"skipped": "the monitoring agent is off"}
    async with _scan_lock:
        now = time.time()
        agents = await asyncio.to_thread(agent_activity.snapshot)
        engines = []
        if _engines is not None:
            try:
                engines = await asyncio.to_thread(_engines)
            except Exception:  # noqa: BLE001
                logger.debug("monitor: engine statuses unavailable", exc_info=True)
        loaded = None
        if _loaded is not None:
            try:
                loaded = {m["model"] for m in _loaded() if m.get("model")}
            except Exception:  # noqa: BLE001 -- unknown: judge stalls without it
                logger.debug("monitor: loaded models unavailable", exc_info=True)
        findings = scan(agents, engines, cfg, now, loaded)
        new = counted = reopened = 0

        def file_all() -> tuple[int, int, int]:
            n = c = o = 0
            for f in findings:
                res = bugs.sight(f)
                n += res["new"] and res["visible"]
                c += res["counted"]
                o += res["reopened"]
            return n, c, o

        new, counted, reopened = await asyncio.to_thread(file_all)
        closed = 0
        if cfg.get("auto_close", True):
            watched = await asyncio.to_thread(bugs.watched)
            for bid, why in fixed_bugs(watched, agents, engines, cfg, now):
                await asyncio.to_thread(bugs.close_fixed, bid, why)
                closed += 1
        _state.update(last_scan=time.time(), last_findings=len(findings), last_new=new, last_error=None)
        _state["scans"] += 1
        _state["last_closed"] = closed
    out = {"findings": len(findings), "new_bugs": new, "sightings": counted, "reopened": reopened,
           "closed_fixed": closed}
    if llm and cfg["llm_triage"] and _complete is not None:
        out["triage"] = _start_triage(cfg, agents, now)
    return out


async def run(complete: CompleteFn, loaded: LoadedFn, engines: EnginesFn | None = None) -> None:
    global _complete, _loaded, _engines
    _complete, _loaded, _engines = complete, loaded, engines
    _state["running"] = True
    try:
        while True:
            await asyncio.sleep(TICK_S)
            try:
                await scan_once()
            except Exception as exc:  # noqa: BLE001 -- the monitor must never die
                _state["last_error"] = str(exc)[:500]
                logger.exception("monitor scan failed")
    finally:
        _state["running"] = False


# =============================================================================================
# Routes (under /api)
# =============================================================================================
class MonitorConfig(BaseModel):
    enabled: bool | None = None
    llm_triage: bool | None = None
    model: str | None = Field(None, max_length=300)
    stall_minutes: int | None = Field(None, ge=1, le=24 * 60)
    slow_reply_s: int | None = Field(None, ge=30, le=24 * 3600)
    auto_close: bool | None = None
    fixed_after: int | None = Field(None, ge=1, le=1000)
    fixed_quiet_minutes: int | None = Field(None, ge=0, le=7 * 24 * 60)


def _status() -> dict:
    loaded = _loaded() if _loaded else []
    return {"config": prefs.get_monitor(), "state": dict(_state), "tick_s": TICK_S,
            "models": sorted({m["model"] for m in loaded if m.get("model")})}


@router.get("/monitor")
async def monitor_status() -> dict:
    return _status()


@router.put("/monitor")
async def monitor_config(req: MonitorConfig) -> dict:
    await asyncio.to_thread(prefs.set_monitor, req.model_dump(exclude_none=True))
    return _status()


async def recheck(bid: int) -> dict:
    """Is this bug fixed? Rescan first (a new occurrence must count), then judge it on the same
    evidence as auto-close, minus the quiet period: the operator asked, so enough clean chances
    since the last occurrence are enough. Closes it when fixed. Rules only: never waits on a model."""
    await scan_once(force=True, llm=False)
    bug = await asyncio.to_thread(bugs.get_bug, bid)
    if bug["status"] == "closed":
        return {"verdict": "closed", "message": "Already closed.", "bug": bug}
    agents = await asyncio.to_thread(agent_activity.snapshot)
    engines = []
    if _engines is not None:
        try:
            engines = await asyncio.to_thread(_engines)
        except Exception:  # noqa: BLE001
            logger.debug("monitor: engine statuses unavailable", exc_info=True)
    cfg = prefs.get_monitor()
    got = fix_chances(bug, agents, engines) if bug["source"] == "monitor" else None
    ago_min = (time.time() - float(bug.get("last_seen") or time.time())) / 60
    last = f"last seen {ago_min:.0f} min ago" if ago_min < 120 else f"last seen {ago_min / 60:.1f} h ago"
    if got is None:
        await asyncio.to_thread(bugs.add_note, bid, f"recheck: cannot be judged from the logs ({last})")
        return {"verdict": "unknown", "bug": await asyncio.to_thread(bugs.get_bug, bid),
                "message": "This kind of bug (filed by hand, or found by the model's review) cannot be judged from "
                           f"the logs; {last}. Close it by hand once you have checked."}
    n, need, what = got
    need = need or int(cfg.get("fixed_after") or 15)
    if n >= need:
        why = what if bug["fingerprint"].startswith("engine:") else \
            f"on recheck: not seen in {n} {what} since its last occurrence ({last}). It reopens if it comes back."
        await asyncio.to_thread(bugs.close_fixed, bid, why)
        return {"verdict": "fixed", "message": f"Fixed: {why}", "bug": await asyncio.to_thread(bugs.get_bug, bid)}
    await asyncio.to_thread(bugs.add_note, bid, f"recheck: not confirmed fixed -- {n} of {need} {what} clean since "
                                                f"the last occurrence ({last})")
    return {"verdict": "not_yet", "bug": await asyncio.to_thread(bugs.get_bug, bid), "chances": n, "needed": need,
            "message": (f"Not confirmed yet: {n} of {need} {what} since its last occurrence ({last}) went clean. "
                        "The agents have not given it enough chances to come back -- recheck after they have run "
                        "more." if bug["fingerprint"].split(":", 1)[0] != "engine" else
                        "Not fixed yet: that model has not been loaded and running since the crash.")}


@router.post("/bugs/{bid}/recheck")
async def bug_recheck(bid: int) -> dict:
    return await recheck(bid)


@router.post("/monitor/scan")
async def monitor_scan() -> dict:
    """Scan now, whether or not the monitor is on (model triage only if it is set up)."""
    try:
        out = await scan_once(force=True)
    except Exception as exc:  # noqa: BLE001
        _state["last_error"] = str(exc)[:500]
        raise
    return {**out, **_status()}
