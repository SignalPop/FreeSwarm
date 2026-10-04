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

from . import agent_activity, bugs, prefs, work
from .diagnose import stopped_from_outside

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
_FRAME = re.compile(r'File "([^"]+)", line (\d+)[^\n]*\n?([^\n]*)')      # path, line, the source line under it
_ERRLINE = re.compile(r"^\s*([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt))(?::\s?(.*))?$", re.M)
# "trend_exits() got an unexpected keyword argument 'x'", and as ft words it since 2026-10-01:
# "ft.trend_exits: got an unexpected keyword argument 'x'."
_UNKNOWN_KEYWORD = re.compile(r"(?:ft\.)?(\w+)(?:\(\))?:? got an unexpected keyword argument '(\w+)'")
_DATA_NO_ATTR = re.compile(r"'(DataFrame|Series|LazyFrame|Expr|numpy\.ndarray|float|int|bool|NoneType)' object has "
                           r"no attribute '\w+'")
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


_FT_CALL = re.compile(r"\bft\s*\.|\bft\b\s*\(")
_FRAME_FN = re.compile(r'File "([^"]+)", line \d+, in ([^\n]+)\n?([^\n]*)')     # path, function, source line


def _through_shim(text: str) -> bool:
    """ft.py is in the traceback only as a shim on the agent's own data object: the agent's line
    that entered ft calls no ft function (`rows.with_columns(vwap=vwap)`, `df.sort(reverse=True)`)
    -- ft's pandas/polars compatibility layer (2026-10-01) wraps those methods. An error polars
    raises under it, or the shim's own refusal, is the agent's mistake on its own object, not a
    harness fault ("Harness error in run_python: TypeError: cannot create expression literal for
    value of type Series" was a pandas Series passed to polars' with_columns). A helper called by
    name stays the harness's, `ft.quick_score(...)` or a bare `quick_score(...)` imported from ft."""
    frames = _FRAME_FN.findall(text)
    for k, (path, fn, _src) in enumerate(frames):
        if any(h in path.replace("\\", "/") for h in _HARNESS_FILES):
            if k == 0:
                return False
            cpath, _cfn, line = frames[k - 1]
            if not any(a in cpath.replace("\\", "/") for a in _AGENT_FILES) or not line.strip():
                return False
            called = re.search(rf"(?<![\w.]){re.escape(fn.strip())}\s*\(", line)
            return not called and not _FT_CALL.search(line)
    return False


def traceback_of(text: str) -> dict | None:
    """{etype, message, origin: harness | agent | library, frame, line, raised} of the LAST
    traceback in `text`, or None. `origin` is the deepest frame that is not a third-party
    package; `raised` says that frame is where the traceback ends and its line is a `raise` --
    the code there refused something on purpose, it did not break."""
    if "Traceback" not in text and "Error:" not in text:
        return None
    errs = list(_ERRLINE.finditer(text))
    if not errs:
        return None
    last = errs[-1]
    etype, msg = last.group(1).rsplit(".", 1)[-1], (last.group(2) or "").strip()
    frames = [(f, int(n), src.strip()) for f, n, src in _FRAME.findall(text[: last.start()])]
    own = [f for f in frames if "site-packages" not in f[0] and "/lib/python" not in f[0] and "<frozen" not in f[0]]
    # The deepest frame that is not a third-party package decides: an error raised inside polars
    # from the agent's own call is the agent's; one raised in (or through) ft.py is the harness's.
    frame = own[-1] if own else (frames[-1] if frames else ("", 0, ""))
    path = frame[0].replace("\\", "/")
    if any(h in path for h in _HARNESS_FILES) and not any(a in path for a in _AGENT_FILES):
        origin = "harness"
    elif any(a in path for a in _AGENT_FILES) or path in ("script.py", ""):
        origin = "agent"
    else:
        origin = "library"
    raised = bool(frames) and frames[-1] is frame and frame[2].startswith("raise ")
    if origin == "harness" and (raised or frames[-1] is not frame) and _through_shim(text[: last.start()]):
        origin = "agent"
    return {"etype": etype, "message": msg[:500], "origin": origin, "frame": path, "line": frame[1], "raised": raised}


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
    # A policy refusal (over the experiment budget; code that arrived truncated and was refused
    # unrun, for the agent to resend) is the runner working as designed, and a soft step (answering
    # the agent's feedback) one the iteration goes on without: the Work page counts neither as an
    # error (work.refusal, work.soft_step), and neither is a bug.
    if work.refusal(e) or work.soft_step(e):
        return []
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
    # What a failed submission or library save carries beside its one-line error: the script's
    # stderr, the smoke test's output. The traceback in there says what failed and whose failure
    # it was. Without it every such failure was the same bug -- "submit_candidate keeps failing:
    # the script failed -- see stderr" (#41), "library_save: the smoke test failed" (#16) -- a
    # bug made of unrelated mistakes that no fix could ever close, and a harness fault at
    # submission hid in it as an agent error.
    attached = ""
    for k, failed in (("stderr_tail", d.get("status") == "error"), ("test_output", d.get("test_ok") is False)):
        if failed and traceback_of(_as_text(d.get(k) or "")):
            attached = _as_text(d[k])
            break

    # Control-plane errors surfaced as {"error": "/api/... -> NNN: detail"}.
    if err and not e.get("ok"):
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
        if tb is None and not attached:
            # The tool's own feedback to the agent ("look-ahead: ...", "no candidate 1198"):
            # the agent's mistake, worth a look only when it keeps happening. "see stderr" says
            # nothing by itself; the line stderr ends on does ("[killed: exceeded the 300s
            # limit]"), so a run killed for time is not one bug with every other failure (#41).
            tail = next((ln.strip() for ln in reversed(_as_text(d.get("stderr_tail") or "").splitlines())
                         if ln.strip()), "") if "see stderr" in err else ""
            what = f"{err}: {tail[:200]}" if tail else err
            return [_finding(w, key, at, fingerprint=f"toolerr:{name}:{bugs.normalize(what, 90, quoted=False)}",
                             category="agent_error", severity="low", priority="P4", min_occurrences=5,
                             title=f"{name} keeps failing: {what[:140]}",
                             description="Agents keep getting this answer from the tool. If it is always the same "
                                         f"mistake, the tool's description or the brief could prevent it.\n\n{what[:1000]}",
                             **base)]

    # Where a traceback can be: a failed run's stderr, a failed submission's error, or -- when the
    # runner cut a long result so it no longer parses -- the raw text.
    if d:
        src = (_as_text(d.get("stderr") or "") if d.get("ok") is False else "") or attached or err or \
              (_as_text(d.get("error") or "") if d.get("status") == "error" else "")
    else:
        # Still JSON-escaped: without real newlines no error line is found, and a cut result's
        # traceback went unjudged -- so a model's review could file it as a platform fault (#299).
        src = text.replace("\\n", "\n").replace('\\"', '"') if "Traceback (most recent call last)" in text else ""
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
    kw = _UNKNOWN_KEYWORD.search(tb["message"]) if tb["etype"] == "TypeError" else None
    if kw and tb["origin"] == "harness":
        # An ft helper called with a parameter it does not take (trend_exits(atr_lookback=...)): the
        # agent's call, not ft breaking -- once filed as "Harness error", and once per tool and per
        # wrong name, so seven lost submissions on 2026-09-30 made three bugs, two of them hidden.
        fn = kw.group(1)
        return [_finding(w, key, at, fingerprint=f"ftcall:{fn}", category="agent_error", severity="medium",
                         priority="P3", min_occurrences=2,
                         title=f"Agents call ft.{fn} with parameters it does not take",
                         description=f"ft.{fn} was called with a keyword it does not have ('{kw.group(2)}' this "
                                     "time). Each one kills the run in seconds; at submit_candidate it costs a "
                                     f"candidate. The sightings list every wrong name: make ft.{fn}'s docstring, "
                                     "the brief or ft._PARAM_ALIASES steer them to the right parameter.", **base)]
    if tb["origin"] == "harness":
        msg = tb["message"].lower()
        none = "(none)" in msg
        if ("available:" in msg or "valid columns" in msg) and not none:
            pass                                   # a name the agent got wrong, with the right ones listed
        elif tb["raised"] and not none:
            # ft checked what it was given and said no ("report_positions: the series must be
            # indexed by the bar timestamp", "ft.size: direction has 2 values but scale has 3"):
            # the helper working as written, on a call the agent got wrong. Not a harness error
            # -- but the same refusal again and again means its contract trips agents up.
            return [_finding(w, key, at, fingerprint=f"ftcheck:{name}:{tb['etype']}:"
                                                     f"{bugs.normalize(tb['message'], 120, quoted=False)}",
                             category="agent_error", severity="low", priority="P4", min_occurrences=5,
                             title=f"ft keeps refusing agents' calls: {tb['message'][:120]}",
                             description="The sandbox helper raised this on purpose: it checked what the agent "
                                         f"passed and refused it ({tb['frame']}, line {tb['line']}). One is the "
                                         "agent's mistake; many mean the brief, the helper's docs or its contract "
                                         "lead agents into the call it refuses.", **base)]
        else:
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
    if tb["etype"] == "AttributeError" and _DATA_NO_ATTR.match(tb["message"]):
        # One mistake under many messages: the wrong kind of data object -- pandas vs polars, a polars
        # Series vs an expression, a numpy array, the number a reduction returned. Fingerprinted per
        # message, 19 of them on 2026-09-30 made 11 bugs, each under the 5 that shows one.
        return [_finding(w, key, at, fingerprint="agent:data-object-mixup", category="agent_error",
                         severity="medium", priority="P3", min_occurrences=5,
                         title="Agents call methods their data object does not have (pandas / polars / numpy mix-ups)",
                         description="A script calls a method on the wrong kind of object: a polars method on a pandas "
                                     "frame or back, an expression method (.over, .alias) on a polars Series, "
                                     ".to_numpy on pl.col(...), a Series method on a numpy array or on the number "
                                     ".mean() returned. Each sighting has the exact message; the hint agents get "
                                     "comes from objectives._frame_hint -- a message it says nothing for is the gap "
                                     f"to close.\n\nThis one: {tb['message'][:300]}", **base)]
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
# When this process started. Chats and tool calls go through the control plane, so nothing
# pending from before it started is still in flight, and silence is only measured over time the
# monitor was watching: its first scan after a restart of the whole stack called every agent
# the old runner left open "stuck" or "silent" for the hours it was down (#69, #70, #128, #156).
STARTED_AT = time.time()


def record_findings(a: dict, r: dict, now: float, cfg: dict, loaded: set[str] | None = None,
                    watching_since: float = 0.0) -> list[dict]:
    """Findings about the record as a whole: stalls and outcomes. `loaded`: the models loaded now
    (None = unknown) -- an agent whose model is no longer loaded was retired, not stalled.
    `watching_since`: when the control plane started (see STARTED_AT)."""
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
        if since and (now - since > ABANDONED_S or since < watching_since):
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
        elif not since and now - max(updated, watching_since) > max(2 * stall_s, 1800.0):
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
    tried = _submit_outcomes(r)
    if status not in ("running", "interrupted") and tried and "ok" not in tried:
        # The iteration did submit -- so "nosubmit" stays quiet -- but every try crashed: all its
        # research bought nothing. Qwen3.6-35B ended 6 of 7 iterations so on 2026-09-30 and no bug said so.
        errs = [e for e in _submit_errors(r) if e]
        out.append(_finding(w, f"{r.get('id')}:submitfail", float(r.get("ended_at") or r.get("started_at") or now),
                            fingerprint=f"submitfail:{a.get('model')}", category="stall", severity="high",
                            priority="P2", min_occurrences=2,
                            title=f"Iterations of {a.get('model')} end with every submission failed",
                            description=f"The iteration called submit_candidate {len(tried)} time(s) and every run "
                                        "failed, so it produced no scored candidate. The errors say whether the model keeps making "
                                        "one mistake (a hint or an ft change can stop it) or the scorer refuses "
                                        "something it should not.\n\n" + "\n".join(f"- {e}" for e in errs[:4]),
                            tool="submit_candidate", evidence=json.dumps({"errors": errs}),
                            context={"submissions": len(tried)}))
    return out


def _submit_outcomes(r: dict) -> list[str]:
    """The status of each submit_candidate result in the record ("ok" / "error" / ...), in order;
    a refused call (truncated, over the limit) has no candidate and does not count."""
    out = []
    for e in r.get("timeline") or []:
        if e.get("kind") == "tool" and e.get("name") == "submit_candidate":
            d = _parse(e.get("result")) or {}
            if d.get("seq") or d.get("candidate_id"):
                out.append(str(d.get("status") or ("error" if d.get("error") else "ok")))
    return out


def _submit_errors(r: dict) -> list[str]:
    """The line each failed submission died of: its traceback's last line, or the scorer's reason."""
    out = []
    for e in r.get("timeline") or []:
        if e.get("kind") != "tool" or e.get("name") != "submit_candidate":
            continue
        d = _parse(e.get("result")) or {}
        if (d.get("seq") or d.get("candidate_id")) and d.get("status") == "error":
            tb = traceback_of(_as_text(d.get("stderr_tail") or ""))
            out.append(f"#{d.get('seq')}: " + (f"{tb['etype']}: {tb['message'][:200]}" if tb
                                                else _as_text(d.get("error") or "")[:200]))
    return out


def engine_findings(engines: list[dict]) -> list[dict]:
    out = []
    for s in engines:
        if s.get("state") != "error":
            continue
        diag = s.get("diagnosis") or {}
        # Stopped from outside (a restart closed its console, Ctrl+C), not crashed: every restart
        # used to file a critical "Engine failed to run ..." bug per loaded model.
        if diag.get("stopped") or stopped_from_outside(diag.get("tail") or []):
            continue
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


# Findings that are an agent's own mistake with a tool (its script's error, a call ft or the
# control plane refused, a wrong parameter), as opposed to the platform failing. One the agent
# fixed itself later in the same iteration -- the next run_python / submit_candidate /
# library_save ran, or a later call of another tool with similar arguments worked (see
# work.recovery) -- is not a sighting: the Work page counts it as "recovered", and a bug made of
# them told the operator about mistakes nobody needed to fix. Platform faults (harness errors,
# control-plane 5xx, code cut off on its way from the model) are filed even when a retry worked.
AGENT_KINDS = frozenset({"toolerr", "ftcheck", "ftcall", "agent", "api4xx"})


def _fp_kind(fp: Any) -> str:
    return str(fp or "").split(":", 1)[0]


def _recovery(r: dict) -> dict[int, str]:
    try:
        return work.recovery(r.get("timeline") or [], running=r.get("status") == "running")
    except Exception:  # noqa: BLE001 -- unknown: judge every failure as it stands
        logger.debug("monitor: recovery of %s unknown", r.get("id"), exc_info=True)
        return {}


def _side_windows(r: dict) -> list:
    try:
        return work.side_windows(r)
    except Exception:  # noqa: BLE001 -- unknown: judge every chat as the agent's own
        logger.debug("monitor: side requests of %s unknown", r.get("id"), exc_info=True)
        return []


def scan(agents: list[dict], engines: list[dict], cfg: dict, now: float | None = None,
         loaded: set[str] | None = None, watching_since: float = 0.0) -> list[dict]:
    """Every finding in the current logs (pure: no I/O). `loaded`: model names loaded now.
    An agent's own error that it fixed later in its iteration ("recovered"), or that its still
    running iteration may yet fix ("pending"), is no finding (AGENT_KINDS); nor is a policy refusal,
    a soft step, or a failed side request (see work.refusal / soft_step / side_purpose)."""
    now = now or time.time()
    out: list[dict] = []
    for a in agents:
        for r in a.get("records") or []:
            chat_model = None
            states = _recovery(r)
            windows = _side_windows(r)
            for i, e in enumerate(r.get("timeline") or []):
                kind = e.get("kind")
                try:
                    if kind == "chat":
                        chat_model = e.get("model") or chat_model
                        # A failed side request (an auto-repair or the answer to the agent's feedback,
                        # capped, the iteration going on without it) is soft: no finding.
                        if not (e.get("error") and work.side_purpose(e, windows)):
                            out += chat_findings(a, r, i, e, cfg)
                    elif kind == "tool":
                        found = tool_findings(a, r, i, e, chat_model)
                        if states.get(i) in ("recovered", "pending"):
                            found = [f for f in found if _fp_kind(f.get("fingerprint")) not in AGENT_KINDS]
                        out += found
                except Exception:  # noqa: BLE001 -- one odd event must not blind the monitor
                    logger.debug("monitor: event %s of %s skipped", i, r.get("id"), exc_info=True)
            out += record_findings(a, r, now, cfg, loaded, watching_since)
    out += engine_findings(engines)
    return out


def recovered_only(bug: dict, records: dict[str, dict]) -> tuple[str, str | None]:
    """Is this open bug (with its "sightings") made only of errors the agent fixed itself?
    ("close", why) when every sighting the bug keeps is a recovered error -- or, when it keeps
    them all, too few are unrecovered to have filed it -- ("keep", None) when it stands, and
    ("wait", None) when that cannot be told yet (an iteration still running). A sighting whose
    record or call is gone cannot be judged: the bug stands. Pure: `records` by id."""
    if bug.get("source", "monitor") != "monitor" or _fp_kind(bug.get("fingerprint")) not in AGENT_KINDS:
        return "keep", None
    sightings = bug.get("sightings") or []
    if not sightings:
        return "keep", None
    fixed = broken = 0
    states_of: dict[str, dict[int, str]] = {}
    for s in sightings:
        r = records.get(s.get("record_id") or "")
        if r is None:
            return "keep", None
        tl = r.get("timeline") or []
        at = float(s.get("at") or 0)
        idx = next((i for i, e in enumerate(tl) if isinstance(e, dict) and e.get("kind") == "tool"
                    and abs(float(e.get("at") or r.get("started_at") or 0) - at) < 1e-3), None)
        if idx is None:
            return "keep", None
        if r["id"] not in states_of:
            states_of[r["id"]] = _recovery(r)
        st = states_of[r["id"]].get(idx)
        if st == "pending":
            return "wait", None
        # A policy refusal or a soft step was never an error (work.refusal / soft_step).
        if st in ("recovered", "refused") or (st is None and work.soft_step(tl[idx])):
            fixed += 1
        elif st == "unrecovered":
            broken += 1
        else:
            return "keep", None                  # not a failure the Work page knows: judge it as filed
    total = int(bug.get("occurrences") or len(sightings))
    need = int(bug.get("min_occurrences") or 1)
    if broken == 0:
        why = (f"all {fixed} sighting{'s' if fixed != 1 else ''} kept were errors the agent fixed itself later in "
               "the same iteration (a later call of the tool ran), policy refusals or soft steps -- not a bug.")
    elif total <= len(sightings) and broken < need:
        why = (f"{fixed} of its {total} sightings were errors the agent fixed itself later in the same iteration "
               f"(recovered); the {broken} left are under the {need} that file this kind of bug.")
    else:
        return "keep", None
    return "close", why + " It reopens if an unrecovered one comes."


_judged: dict[int, tuple[str, float | None]] = {}   # bug id -> (fingerprint, last_seen) judged to stand


def resolve_recovered(agents: list[dict]) -> int:
    """Close the open agent-error bugs that recovered_only() says are no bugs (filed before
    recovered errors were left out, or from a sighting its iteration fixed after it was filed).
    Returns how many were closed. Each bug is judged again only when it is sighted again."""
    todo = [b for b in bugs.watched() if _fp_kind(b.get("fingerprint")) in AGENT_KINDS
            and _judged.get(b["id"]) != (b.get("fingerprint"), b.get("last_seen"))]
    if not todo:
        return 0
    full = [bugs.get_bug(b["id"]) for b in todo]
    have = {str(r["id"]): r for a in agents for r in a.get("records") or [] if isinstance(r, dict) and r.get("id")}
    need = {s["record_id"] for b in full for s in b.get("sightings") or []
            if s.get("record_id") and s["record_id"] not in have}
    if need:
        try:
            for rid, r in work.records(sorted(need)).items():
                have.setdefault(rid, {**r, "id": r.get("id") or rid})
        except Exception:  # noqa: BLE001 -- without the archive only the inspector's records judge
            logger.debug("monitor: work log records unavailable", exc_info=True)
    closed = 0
    for b in full:
        verdict, why = recovered_only(b, have)
        if verdict == "close":
            bugs.close_fixed(b["id"], why or "")
            closed += 1
        elif verdict == "keep":
            _judged[b["id"]] = (b.get("fingerprint"), b.get("last_seen"))
    return closed


# =============================================================================================
# Seeing a bug fixed
# =============================================================================================
# A bug is closed as fixed only on evidence: since its last occurrence the thing it happened in
# (the same tool, the same model's replies, the same kind of iteration) ran again enough times
# without it, and a while has passed. Numbers per kind: record-level problems get fewer chances
# (an iteration is long), an engine crash is fixed the moment that model runs again.
RECORD_CHANCES = 5
_TOOL_KINDS = {"harness", "ftcheck", "ftcall", "api5xx", "api4xx", "toolerr", "agent", "budget"}
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
    if kind == "submitfail":
        n = sum(1 for a in mine if a.get("model") == model for r in a.get("records") or []
                if float(r.get("ended_at") or 0) > since and "ok" in _submit_outcomes(r))
        return n, RECORD_CHANCES, "iterations with a submission that ran"
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
    """The configured model when it is loaded; otherwise -- none configured, or the configured one
    not loaded (unloaded, or its engine went with a restart) -- the best local model loaded, the
    ones already serving the swarm first. Never an external model unless it is the configured one.
    A configured model that is not loaded used to stop triage altogether ("the configured model
    is not loaded") until someone loaded it again."""
    loaded = _loaded() if _loaded else []
    ready = [m for m in loaded if m.get("model") and m.get("ready", True)]
    names = {m["model"] for m in ready}
    if cfg.get("model") and cfg["model"] in names:
        return cfg["model"]
    for pool in ([m for m in ready if not m.get("external") and not m.get("remote")],
                 [m for m in ready if not m.get("external")]):
        if pool:
            return max(pool, key=lambda m: m.get("aa") or 0)["model"]
    return None


def triage_note(configured: str, model: str | None) -> str | None:
    """What the Bugs page says about triage's model (None: nothing to say)."""
    if not model:
        return (f"the configured model {configured} is not loaded and no other local model is -- load {configured} "
                "or any local model" if configured else "no local model loaded")
    if configured and model != configured:
        return (f"the configured model {configured} is not loaded -- using {model} instead (load {configured}, "
                "or pick a loaded model for the monitor in Settings)")
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
    if not str(bug.get("fingerprint") or "").startswith("review:"):
        # A rule-filed bug's title and description come from the rule that matched and say exactly
        # what happened; the small model's rewrite made them wrong ("ft.trend_exits called without
        # required parameters" for an unknown keyword, #332). Keep both; its reading goes below.
        fields.pop("title", None)
        if desc:
            desc = f"{bug.get('description') or ''}\n\nReading by {model}: {desc}".strip()
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


# Tools whose results are not the platform speaking. The board, the library and get_candidate
# hand back what AGENTS wrote -- a teammate's error post, a candidate's source -- and
# submit_candidate / combine_candidates return the verdict on the agent's own strategy.
_NOT_EVIDENCE_TOOLS = {"team_board", "team_post", "library_list", "library_get", "library_comment", "get_candidate",
                       "research_search", "research_get", "ask_model", "submit_candidate", "combine_candidates",
                       # the agent's own reply to the mentor (#401: "I accept. I will keep candidate 1855's
                       # core..." filed as incorrect data)
                       "answer_feedback"}


def _platform_evidence(evidence: str, a: dict, r: dict) -> bool:
    """Whether a review's quote comes from where a platform fault no rule catches can show: the
    result of a tool call that WORKED and returns the platform's own data or analysis.

    Everything else a small reviewing model quoted was noise, and eleven of the thirteen open
    bugs on 2026-09-30 were that noise: the agent's own traceback ("Missing signal function
    import", "Function name typo"), a refusal working as designed ("Trade Frequency
    Constraint"), another agent's error post read off the team board ("tool failure"), a print
    statement in a candidate's source ("Invalid data format"). A call that failed is never
    review evidence: the detectors above already judged it -- the agent's mistake, counted until
    it recurs, or a platform fault filed by rule with the right priority."""
    chat_model = None
    for i, e in enumerate(r.get("timeline") or []):
        if e.get("kind") == "chat":
            chat_model = e.get("model") or chat_model
        if e.get("kind") != "tool" or e.get("name") in _NOT_EVIDENCE_TOOLS:
            continue
        if not _grounded(evidence, _clip(e.get("result"), 900)):      # the result as the reviewer was shown it
            continue
        try:
            judged = e.get("ok") is False or _reports_failure(e.get("result")) or \
                bool(tool_findings(a, r, i, e, chat_model))
        except Exception:  # noqa: BLE001 -- an event the detectors cannot read is not evidence either
            judged = True
        if not judged:
            return True
    return False


def _reports_failure(result: Any) -> bool:
    """Whether a tool call that went through hands back a run of the AGENT's code that failed:
    run_python's {"ok": false} (the call worked, the script did not), a library save whose test
    or causality check the module's own code failed. Six of the open review bugs on 2026-10-01
    quoted exactly that -- "'DataFrame' object has no attribute 'with_columns'" from a saved
    module whose signal() raised in the causality test (#309), a traceback in a run_python
    result too long to parse (#299)."""
    d = _parse(result)
    if d is None:
        text = _as_text(result or "")
        return bool(re.match(r'\s*\{\s*"(?:ok|saved|test_ok)":\s*false', text)) or "repaired automatically" in text
    if d.get("ok") is False or d.get("saved") is False or d.get("test_ok") is False or d.get("status") == "error":
        return True
    # The agent's script failed and the runner fixed it in the same call: still the agent's mistake
    # (#400, 10-04: "'list' object has no attribute 'to_list'", filed as a data inconsistency).
    if d.get("auto_repaired") or "repaired automatically" in _as_text(d.get("stdout") or d.get("note") or ""):
        return True
    c = d.get("causality") if isinstance(d.get("causality"), dict) else {}
    return c.get("verdict") == "fail" or bool(re.search(r"\) raised on data cut", str(c.get("detail") or "")))


async def _review(model: str, a: dict, r: dict) -> int:
    transcript = condense(r)
    prompt = (f"{_ABOUT}\n\nRead this iteration of agent {a.get('agent')} and report problems in the PLATFORM only: "
              "wrong, missing or corrupt data (empty tables, all-NaN or constant columns, impossible values, dates "
              "out of range), tools that fail or contradict what the brief tells the agent, results that disagree "
              "with each other, and anything that wastes the agent's time. Do NOT report the agent's ordinary "
              "coding mistakes or how good its strategy is: an error in the agent's own script, a tool refusing a "
              "bad call, a candidate that scores badly or is not ranked, and anything read off the team board, the "
              "library or another candidate's code are not platform problems.\n\nAlready filed (do not repeat):\n"
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
        if not _platform_evidence(str(it.get("evidence") or ""), a, r):
            continue                             # the agent's own mistake, or not the platform's words
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
    _state["llm_note"] = triage_note(cfg.get("model") or "", model)
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
        findings = scan(agents, engines, cfg, now, loaded, STARTED_AT)
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
        # Not a fix judgement but the filing rule applied to what is already filed: a bug made
        # only of errors the agents fixed themselves goes, whether or not auto-close is on.
        try:
            recovered = await asyncio.to_thread(resolve_recovered, agents)
        except Exception:  # noqa: BLE001 -- never fails the scan
            logger.warning("monitor: closing recovered-only bugs failed", exc_info=True)
            recovered = 0
        _state["last_closed_recovered"] = recovered
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
           "closed_fixed": closed, "closed_recovered": recovered}
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
