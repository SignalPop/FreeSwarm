"""The work log: everything the swarm did, iteration by iteration, for reading a long (overnight)
session after the fact.

The agent inspector (app/agent_activity.py) keeps only each agent's last few iteration records,
so by morning most of the night is gone from it. This module keeps them: whenever the inspector
flushes, every record that changed is upserted here (by record id) into a small sqlite file,
with its prompts clipped and the rest compressed, plus a precomputed summary row so the list
never has to open a record. Records older than RETENTION_S are pruned. The first open backfills
from whatever the inspector's own file still holds.

The page (/work) reads two things side by side, newest first: those iterations, and the
candidates the objectives store scored in the same window (status, scores, look-ahead verdict,
and for a failure the last line of its traceback). The list carries summaries only; one
iteration's full tool timeline, or one candidate's code and stderr, is fetched when its row opens.

"Clear" on the page is a view marker, not a deletion: it stores the moment (per project, in
`meta` as `cleared_at:<project_id>`) and from then on the list and its summary cover only work
newer than that -- iterations that started after it, ended after it, or are still running, and
candidates created after it -- unless the request asks for the cleared work too. Undo removes it.

Nothing here may slow down or fail the activity post that feeds it: `archive` swallows its own
errors.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import threading
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from . import agent_activity

logger = logging.getLogger("freetoken.work")
router = APIRouter(tags=["work"])

# None: work.sqlite3 beside the inspector's store, so a test that moves one moves both.
DB_PATH: Path | None = None
OBJECTIVES_DB = Path(__file__).resolve().parent.parent / "objectives.sqlite3"
RETENTION_S = 30 * 86400.0
PRUNE_EVERY_S = 3600.0
MAX_WINDOW_H = 30 * 24
SYSTEM_CHARS = 2_000        # the system prompt is ~100 KB and nearly the same every time
PROMPT_CHARS = 8_000
SUMMARY_ERRORS = 30         # error lines kept per iteration summary (search and grouping)
ROW_ERRORS = 3              # ... of which a list row carries
TEXT_CHARS = 600
# Bumped when summarize() changes what it computes: rows archived before are summarized again
# from their records on the next open (2: recovered and auto-repaired errors counted apart;
# 3: policy refusals, soft steps and failed side requests are not errors).
SUMMARY_VERSION = 3

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None
_conn_path: Path | None = None
_obj_conn: sqlite3.Connection | None = None
_obj_path: Path | None = None
_last_prune = 0.0


def _path() -> Path:
    return DB_PATH or Path(agent_activity.DB_PATH).with_name("work.sqlite3")


def db() -> sqlite3.Connection:
    global _conn, _conn_path
    path = _path()
    if _conn is not None and _conn_path == path:
        return _conn
    if _conn is not None:
        _conn.close()
    _conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
    _conn_path = path
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.executescript("""
        CREATE TABLE IF NOT EXISTS iterations (
            id TEXT PRIMARY KEY,
            project_id TEXT, agent TEXT, model TEXT, role TEXT,
            objective_id TEXT, objective_title TEXT, mode TEXT,
            started REAL, ended REAL, status TEXT, outcome TEXT,
            updated_at REAL NOT NULL,
            candidate_ids TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL,
            record BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS iterations_project ON iterations(project_id, started);
        CREATE INDEX IF NOT EXISTS iterations_running ON iterations(updated_at) WHERE status = 'running';
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
    """)
    _conn.commit()
    if _conn.execute("SELECT 1 FROM meta WHERE key='backfilled'").fetchone() is None:
        _backfill(_conn)
    v = _conn.execute("SELECT value FROM meta WHERE key='summary_version'").fetchone()
    if v is None or v[0] != str(SUMMARY_VERSION):
        _resummarize(_conn)
    return _conn


def _resummarize(con: sqlite3.Connection) -> None:
    """Summarize every archived record again (summarize() changed since they were stored)."""
    n = 0
    try:
        ids = [r[0] for r in con.execute("SELECT id FROM iterations").fetchall()]
        for k in range(0, len(ids), 200):
            chunk = ids[k:k + 200]
            rows = con.execute(f"SELECT id, record FROM iterations WHERE id IN ({','.join('?' * len(chunk))})",
                               chunk).fetchall()
            for r in rows:
                try:
                    s = summarize(json.loads(zlib.decompress(r["record"])))
                except Exception as exc:  # noqa: BLE001 -- one odd record must not block the rest
                    logger.debug("work log: re-summarizing %s failed: %s", r["id"], exc)
                    continue
                con.execute("UPDATE iterations SET summary=?, outcome=? WHERE id=?",
                            (json.dumps(s, default=str), s["outcome"], r["id"]))
                n += 1
    except sqlite3.Error as exc:
        logger.warning("work log: re-summarizing failed: %s", exc)
        return
    con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('summary_version', ?)", (str(SUMMARY_VERSION),))
    con.commit()
    logger.info("work log: %d iteration summaries recomputed (version %d)", n, SUMMARY_VERSION)


def reset() -> None:
    """Close the connections (tests point the paths elsewhere first)."""
    global _conn, _conn_path, _obj_conn, _obj_path, _last_prune
    with _lock:
        for c in (_conn, _obj_conn):
            if c is not None:
                c.close()
        _conn = _obj_conn = None
        _conn_path = _obj_path = None
        _last_prune = 0.0


def _objectives_db() -> sqlite3.Connection | None:
    """The objectives store, read only (its own module owns writing it)."""
    global _obj_conn, _obj_path
    if _obj_conn is not None and _obj_path == OBJECTIVES_DB:
        return _obj_conn
    if not Path(OBJECTIVES_DB).exists():
        return None
    _obj_conn = sqlite3.connect(OBJECTIVES_DB, check_same_thread=False, timeout=30.0)
    _obj_conn.row_factory = sqlite3.Row
    _obj_conn.execute("PRAGMA query_only=ON")
    _obj_path = OBJECTIVES_DB
    return _obj_conn


# =======================================================================================
# Reading errors out of tool results and tracebacks
# =======================================================================================
_EXC_LINE = re.compile(r"^(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Exception|Exit|Interrupt|Warning|Fault|Timeout)"
                       r"(?::.*)?$")
_ADDR = re.compile(r"\b0x[0-9a-fA-F]+\b|\b[0-9a-f]{10,}\b")


def _clip(s: Any, n: int = TEXT_CHARS) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _tail(s: Any, n: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else "…" + s[-n:]


def _as_text(result: Any) -> str:
    """A tool result as plain text. The runner stores most results as a head-and-tail cut of
    their JSON, which no longer parses: unescape the newlines and quotes so a traceback reads
    as one."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            return result.replace("\\n", "\n").replace('\\"', '"')
    if isinstance(result, dict):
        parts = [str(result[k]) for k in ("error", "stderr_tail", "stderr", "test_output", "stdout_tail", "stdout")
                 if result.get(k)]
        causality = result.get("causality")
        if isinstance(causality, dict) and causality.get("detail"):
            parts.append(str(causality["detail"]))
        return "\n".join(parts) if parts else json.dumps(result, default=str)
    return "" if result is None else json.dumps(result, default=str)


def error_line(text: Any) -> str:
    """What went wrong, in one line: the exception line ending the LAST traceback in `text`
    (e.g. "TypeError: f() got an unexpected keyword argument 'x'"), else the first line of
    the message."""
    t = str(text or "")
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    tb = t.rfind("Traceback (most recent call last)")
    if tb >= 0:
        after = [ln.strip() for ln in t[tb:].splitlines() if ln.strip()]
        for ln in reversed(after):
            if _EXC_LINE.match(ln):
                return _clip(ln, 200)
    for ln in reversed(lines):
        if _EXC_LINE.match(ln):
            return _clip(ln, 200)
    return _clip(lines[0], 200) if lines else ""


def error_key(line: str) -> str:
    """Two occurrences of one error group together: memory addresses and long ids go."""
    return _ADDR.sub("<id>", line)


# =======================================================================================
# What is not an error: soft steps, side requests, refusals
# =======================================================================================
# Steps the runner takes for the agent whose failure the iteration goes on without, by design
# (answering its feedback before the work). Newer runners record a failed one ok=True with
# "soft": true and the reason under "soft_error"; older ones recorded it ok=False with "error".
SOFT_STEPS = frozenset({"answer_feedback"})
# Tools whose call can include the runner's auto-repair: a model request made while it ran.
REPAIR_TOOLS = frozenset({"run_python", "submit_candidate", "library_save"})
# A policy refusal is the runner saying no on purpose, not a tool breaking:
#   "experiment_budget" -- run_python over the iteration's experiment budget ("experiment budget used
#                          (8 runs this iteration) ...");
#   "truncated"         -- a call whose code arrived cut off mid-token ("... arrived TRUNCATED ...
#                          This experiment has NOT been consumed"); the agent resends it.
_REFUSALS = (("experiment_budget", re.compile(r"experiment budget used \(\d+ runs? this iteration\)")),
             ("truncated", re.compile(r"call arrived TRUNCATED")))
SOFT_CHAT_NOTE = "skipped (model busy)"


def soft_step(e: dict) -> bool:
    """A runner step whose failure is soft by design (see SOFT_STEPS): never a failed call."""
    if not isinstance(e, dict):
        return False
    r = e.get("result")
    return bool(e.get("soft")) or e.get("name") in SOFT_STEPS or (isinstance(r, dict) and r.get("soft") is True)


def refusal(e: dict) -> str | None:
    """The kind of policy refusal this tool call got ("experiment_budget" / "truncated"), or None."""
    if not isinstance(e, dict) or e.get("kind", "tool") != "tool":
        return None
    r = e.get("result")
    if isinstance(r, dict):
        text = str(r.get("error") or "")
    elif isinstance(r, str):
        text = r[:600]
    else:
        return None
    for kind, rx in _REFUSALS:
        if rx.search(text):
            return kind
    return None


def _side_label(name: Any) -> str:
    if name in SOFT_STEPS:
        return str(name)
    if name in REPAIR_TOOLS:
        return "auto_repair"
    return f"during {name}"


def side_windows(rec: dict) -> list[tuple[float, float, str]]:
    """(from, to, purpose) of every span in which a model request cannot be the agent's own
    conversation: while one of its tool calls (or a runner step) ran, the agent's conversation
    waits for the result, so a chat inside one is a side request the runner made -- the auto-repair
    of a crashed script, the answer to the agent's feedback. A tool call still running (the record's
    "pending") is open-ended."""
    out: list[tuple[float, float, str]] = []
    for e in rec.get("timeline") or []:
        if isinstance(e, dict) and e.get("kind") == "tool" and e.get("at") is not None:
            at = float(e["at"])
            out.append((at, at + float(e.get("seconds") or 0), _side_label(e.get("name"))))
    p = rec.get("pending")
    if isinstance(p, dict) and p.get("kind") == "tool" and p.get("since") is not None:
        out.append((float(p["since"]), float("inf"), _side_label(p.get("name"))))
    return out


def side_purpose(c: dict, windows: list[tuple[float, float, str]]) -> str | None:
    """What a chat request was for when it was not the agent's own conversation ("auto_repair",
    "answer_feedback", ...), else None. The runner's explicit marker ("side") wins when recorded;
    otherwise a chat that started and ended inside a tool call's span is one (side_windows)."""
    if not isinstance(c, dict):
        return None
    if "side" in c:
        return str(c["side"]) if c["side"] else None
    if c.get("at") is None:
        return None
    a = float(c["at"])
    b = a + float(c.get("seconds") or 0)
    for lo, hi, what in windows:
        # seconds are rounded to 0.1 s on both sides
        if lo - 0.05 <= a and b <= hi + 0.15:
            return what
    return None


def tool_failed(e: dict) -> bool:
    """A tool call that did not do its job: the runner's own flag, or a result saying so
    (run_python reports a crashed script as ok=false; library_save as saved=false). A soft step
    (SOFT_STEPS) never failed: the iteration goes on without it by design."""
    if soft_step(e):
        return False
    if not e.get("ok", True):
        return True
    r = e.get("result")
    if isinstance(r, dict):
        return bool(r.get("error")) or r.get("ok") is False or r.get("saved") is False or r.get("status") == "error"
    if isinstance(r, str):
        head = r[:200]
        return head.startswith('{"ok": false') or head.startswith('{"error":') or '"saved": false' in head \
            or '"status": "error"' in head
    return False


def _tool_error(e: dict) -> tuple[str, str]:
    """(one-line error, traceback tail) for a failed tool call."""
    text = _as_text(e.get("result"))
    return error_line(text) or "failed", _tail(text, 3000)


# =======================================================================================
# Recovered errors: a failed call the agent fixed itself later in the same iteration
# =======================================================================================
# For these tools the next successful call of the same tool in the iteration is the fix (the
# agent rewrote its script, its candidate, its module): their arguments are code, so "similar"
# would mean little. For any other tool the later success must be a call with similar arguments
# (the same lookup done right), not just any call of the tool.
RECOVER_BY_TOOL = frozenset({"run_python", "submit_candidate", "library_save"})
SIMILAR_ARGS = 0.6
_ARGS_CHARS = 1_500
_REPAIRED = re.compile(r'"auto_repaired"\s*:(?!\s*(?:null|false|\{\}|0\b))')
_CANDIDATE_ID = re.compile(r'"candidate_id"\s*:\s*"([^"]+)"')


def auto_repair(e: dict) -> dict | None:
    """What the runner's auto-repair did for this call ({"attempts", "errors", ...}), or None.
    A repaired call is recorded ok with "auto_repaired" in its result -- in the result dict, or
    in the runner's head-and-tail cut of its JSON (then only what can still be read of it)."""
    if not isinstance(e, dict):
        return None
    r = e.get("result")
    for v in (e.get("auto_repaired"), r.get("auto_repaired") if isinstance(r, dict) else None):
        if isinstance(v, dict) and v:
            return v
        if v is True or (isinstance(v, int) and not isinstance(v, bool) and v > 0):
            return {"attempts": v if v is not True else None}
    if isinstance(r, str) and _REPAIRED.search(r):
        try:
            d = json.loads(r)
        except ValueError:
            return {}
        a = d.get("auto_repaired") if isinstance(d, dict) else None
        return a if isinstance(a, dict) else {}
    return None


def repaired_errors(info: dict | None) -> list[str]:
    """The error lines an auto-repair fixed, as far as its record says."""
    errs = (info or {}).get("errors") or []
    if not isinstance(errs, list):
        errs = [errs]
    out = []
    for x in errs:
        line = error_line(_as_text(x) if isinstance(x, (dict, str)) else json.dumps(x, default=str))
        if line:
            out.append(line)
    return out


def _args_text(args: Any) -> str:
    try:
        return json.dumps(args if args is not None else {}, sort_keys=True, default=str)[:_ARGS_CHARS]
    except (TypeError, ValueError):
        return str(args)[:_ARGS_CHARS]


def similar_args(a: Any, b: Any) -> bool:
    """Two calls of one tool that ask for the same thing: equal arguments, or arguments whose
    JSON is mostly the same (a fixed name, an added or corrected parameter)."""
    ta, tb = _args_text(a), _args_text(b)
    if ta == tb:
        return True
    if isinstance(a, dict) and isinstance(b, dict) and a and b and not set(a) & set(b):
        return False
    from difflib import SequenceMatcher
    m = SequenceMatcher(None, ta, tb, autojunk=False)
    return m.real_quick_ratio() >= SIMILAR_ARGS and m.quick_ratio() >= SIMILAR_ARGS and m.ratio() >= SIMILAR_ARGS


def recovery(timeline: list[dict], running: bool = False) -> dict[int, str]:
    """For each tool call in `timeline` (by its index there) that failed or was auto-repaired:
      "recovered"     -- it failed, and a LATER call of the same tool in this iteration succeeded
                         (for run_python / submit_candidate / library_save any later success of
                         the tool; for other tools one with similar arguments);
      "auto_repaired" -- the runner repaired it: it is recorded ok with "auto_repaired" in its result;
      "pending"       -- failed, not fixed yet, and the iteration is still running (`running`);
      "unrecovered"   -- failed, and the iteration ended without fixing it;
      "refused"       -- a policy refusal (see refusal()), not an error: over the experiment budget,
                         or code that arrived truncated and was not resent successfully (yet). A
                         truncated call is "recovered" by the next successful call of its tool.
    Calls that simply worked (and soft steps, see soft_step) are not in the result."""
    tools = [(i, e) for i, e in enumerate(timeline or []) if isinstance(e, dict) and e.get("kind") == "tool"]
    failed = [tool_failed(e) for _, e in tools]
    out: dict[int, str] = {}
    for pos, (i, e) in enumerate(tools):
        if not failed[pos]:
            if auto_repair(e) is not None:
                out[i] = "auto_repaired"
            continue
        name = e.get("name")
        later = [x for (_, x), bad in zip(tools[pos + 1:], failed[pos + 1:]) if not bad and x.get("name") == name]
        why = refusal(e)
        if why:
            out[i] = "recovered" if why == "truncated" and later else "refused"
            continue
        if name in RECOVER_BY_TOOL:
            fixed = bool(later)
        else:
            fixed = any(similar_args(e.get("args"), x.get("args")) for x in later)
        out[i] = "recovered" if fixed else ("pending" if running else "unrecovered")
    return out


RATIONALE_MATCH = 120


def _squash(s: Any) -> str:
    return " ".join(str(s or "").split())


def made_by(c: dict, objective_id: str | None, model: str | None, w: dict) -> bool:
    """Candidate row `c` is a run of the submit_candidate call `w` ({from, to, rationale}) made
    in an iteration on `objective_id` by `model`: the same objective, made while the call ran,
    with the call's rationale (or, without one, by the same model)."""
    if c.get("objective_id") != objective_id:
        return False
    at = float(c.get("at") or 0)
    if not (w["from"] - 2.0 <= at <= w["to"] + 2.0):
        return False
    if w.get("rationale"):
        return _squash(c.get("rationale")).rstrip("…").startswith(w["rationale"][:RATIONALE_MATCH].rstrip("…")[:100])
    return not model or not c.get("model") or c.get("model") == model


def _candidate_id(r: Any) -> str | None:
    if isinstance(r, dict):
        return r.get("candidate_id") or None
    if isinstance(r, str):
        m = _CANDIDATE_ID.search(r)
        return m.group(1) if m else None
    return None


# =======================================================================================
# Summaries
# =======================================================================================
def _outcome(rec: dict) -> str:
    subs = rec.get("submissions") or []
    st = rec.get("status")
    if st == "running":
        return "running"
    if subs:
        return "ok" if any(s.get("status") == "ok" for s in subs) else "error"
    if st in ("interrupted", "stopped"):
        return "interrupted"
    if st == "no submission":
        return "no_submission"
    return "done"


def summarize(rec: dict) -> dict:
    """What a list row shows for one iteration record (and what search and grouping read)."""
    tl = rec.get("timeline") or []
    tools = [e for e in tl if e.get("kind") == "tool"]
    chats = rec.get("chats") or [e for e in tl if e.get("kind") == "chat"]
    subs = rec.get("submissions") or []
    errors: list[dict] = []
    # Failed calls the agent fixed later in the iteration ("recovered") and calls the runner
    # auto-repaired are not failures of the iteration: they are counted apart, and their error
    # lines kept (marked) for search, but only what stayed broken counts as failed.
    states = recovery(tl, running=rec.get("status") == "running")
    failed = recovered = repaired = rp = rp_failed = rp_recovered = rp_refused = 0
    refusals: Counter = Counter()
    refusals_recovered = 0
    recovered_cands: list[str] = []
    submit_windows: list[dict] = []
    for i, e in enumerate(tl):
        if e.get("kind") != "tool":
            continue
        is_rp = e.get("name") == "run_python"
        st = states.get(i)
        why = refusal(e) if st in ("refused", "recovered") else None
        if why:
            # A policy refusal is no error and no experiment (nothing ran): counted apart. A
            # truncated call the agent resent successfully is "recovered".
            refusals[why] += 1
            refusals_recovered += st == "recovered"
            rp_refused += is_rp
            errors.append({"tool": e.get("name"), "line": _tool_error(e)[0], "at": e.get("at"), "state": "refused",
                           "refusal": why, "recovered": st == "recovered"})
            continue
        rp += is_rp
        if st is None:
            continue
        if e.get("name") == "submit_candidate" and st in ("recovered", "auto_repaired"):
            # Every run of this call is its own candidate row: the crashed original and each failed
            # repair attempt of an auto-repaired submission stay in the objectives store as errors
            # that its result does not name. work() marks those recovered by when they were made.
            at0 = float(e.get("at") or 0)
            args = e.get("args") if isinstance(e.get("args"), dict) else {}
            submit_windows.append({"from": at0, "to": at0 + float(e.get("seconds") or 0),
                                   "rationale": _squash(args.get("rationale"))[:RATIONALE_MATCH]})
        if st == "auto_repaired":
            repaired += 1
            rp_recovered += is_rp
            for line in repaired_errors(auto_repair(e))[-3:]:
                errors.append({"tool": e.get("name"), "line": line, "at": e.get("at"), "state": "auto_repaired"})
            continue
        fixed = st == "recovered"
        if fixed:
            recovered += 1
            rp_recovered += is_rp
        else:
            failed += 1
            rp_failed += is_rp
        r = e.get("result")
        # A submitted candidate that failed is listed (with its stderr) as a candidate row. Its
        # result is a dict when it scored, the runner's cut of the JSON when it errored.
        cid = _candidate_id(r) if e.get("name") == "submit_candidate" else None
        if cid:
            if fixed:
                recovered_cands.append(cid)
            continue
        errors.append({"tool": e.get("name"), "line": _tool_error(e)[0], "at": e.get("at"),
                       "state": "recovered" if fixed else "failed"})
    # A failed request the runner made on the side (an auto-repair, the answer to the agent's
    # feedback: capped, and the iteration goes on without it) is soft, not a chat error.
    windows = side_windows(rec)
    chat_errors = soft_chats = 0
    for c in chats:
        if c.get("error"):
            side = side_purpose(c, windows)
            if side:
                soft_chats += 1
            else:
                chat_errors += 1
            errors.append({"tool": "chat", "line": error_line(c["error"]) or "chat failed", "at": c.get("at"),
                           "state": "soft" if side else "failed", **({"purpose": side} if side else {})})
    errors.sort(key=lambda x: float(x.get("at") or 0))
    if len(errors) > SUMMARY_ERRORS:
        # What stayed broken is kept first; recovered, refused and soft lines fill what room is left.
        keep = set(id(x) for x in [x for x in errors if x["state"] == "failed"][-SUMMARY_ERRORS:])
        room = SUMMARY_ERRORS - len(keep)
        if room > 0:
            keep |= set(id(x) for x in [x for x in errors if x["state"] != "failed"][-room:])
        errors = [x for x in errors if id(x) in keep]
    hypothesis, source = "", None
    for s in reversed(subs):
        if s.get("rationale"):
            hypothesis, source = s["rationale"], "submission"
            break
    if not hypothesis:
        for key in ("said", "reasoning"):
            said = next((e.get(key) for e in tl if e.get("kind") == "chat" and e.get(key)), None)
            if said:
                hypothesis, source = said, key
                break
    names = Counter(e.get("name") for e in tools)
    parent = rec.get("parent") or {}
    return {
        "outcome": _outcome(rec),
        "outcome_text": rec.get("outcome"),
        "reason": rec.get("reason") or rec.get("end_reason"),
        "tool_calls": len(tools),
        "tools": dict(names.most_common(8)),
        "experiments": rp,
        # Failed and never fixed in the iteration; fixed by a later run or auto-repaired.
        "experiments_failed": rp_failed,
        "experiments_recovered": rp_recovered,
        # Failed calls not fixed later in the iteration (still open while it runs) ...
        "experiments_refused": rp_refused,
        # Policy refusals (not errors): {"experiment_budget": n, "truncated": n}; of them, truncated
        # calls the agent resent successfully.
        "refusals": sum(refusals.values()),
        "refusals_by_kind": dict(refusals),
        "refusals_recovered": refusals_recovered,
        "tool_errors": failed,
        # ... failed calls a later call of the tool fixed, plus calls the runner auto-repaired ...
        "tool_errors_recovered": recovered + repaired,
        # ... of which auto-repaired.
        "tool_auto_repaired": repaired,
        "recovered_candidates": recovered_cands,
        "recovered_submits": submit_windows,
        "chats": len(chats),
        # The agent's own conversation failing; side requests that failed (soft) are counted apart.
        "chat_errors": chat_errors,
        "chat_errors_soft": soft_chats,
        "truncations": sum(1 for c in chats if c.get("finish") == "length"),
        "followups": sum(1 for e in tl if e.get("kind") == "message"),
        "tokens": rec.get("tokens"),
        "submissions": [{k: s.get(k) for k in ("candidate_id", "seq", "status", "in_sample_score", "lookahead", "error")}
                        for s in subs],
        "parent_seq": parent.get("seq"),
        "idea": rec.get("idea"),
        "hypothesis": _clip(hypothesis, 800),
        "hypothesis_from": source,
        "timeline_dropped": rec.get("timeline_dropped") or 0,
        "errors": errors,
    }


def _slim(rec: dict) -> dict:
    """The record as archived: the system prompt (the same ~100 KB every time) and the
    iteration prompt clipped; everything the agent did kept."""
    out = dict(rec)
    out["asked"] = [{**a, "system": _clip(a.get("system"), SYSTEM_CHARS), "prompt": _clip(a.get("prompt"), PROMPT_CHARS)}
                    for a in rec.get("asked") or [] if isinstance(a, dict)]
    return out


# =======================================================================================
# Archiving
# =======================================================================================
def _rows(meta: dict, recs: list[dict]) -> list[tuple]:
    rows = []
    for rec in recs:
        if not isinstance(rec, dict) or not rec.get("id"):
            continue
        obj = rec.get("objective") or {}
        summary = summarize(rec)
        cids = "".join(f"|{s['candidate_id']}" for s in summary["submissions"] if s.get("candidate_id"))
        blob = zlib.compress(json.dumps(_slim(rec), default=str).encode(), 6)
        rows.append((str(rec["id"]), meta.get("project_id"), meta.get("agent"), meta.get("model"), meta.get("role"),
                     obj.get("id"), obj.get("title"), rec.get("mode"), rec.get("started_at"), rec.get("ended_at"),
                     rec.get("status"), summary["outcome"],
                     float(rec.get("received_at") or rec.get("started_at") or time.time()),
                     cids + "|" if cids else "", json.dumps(summary, default=str), blob))
    return rows


_UPSERT = ("INSERT INTO iterations (id, project_id, agent, model, role, objective_id, objective_title, mode, started, "
           "ended, status, outcome, updated_at, candidate_ids, summary, record) "
           "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
           "ON CONFLICT(id) DO UPDATE SET project_id=excluded.project_id, agent=excluded.agent, model=excluded.model, "
           "role=excluded.role, objective_id=excluded.objective_id, objective_title=excluded.objective_title, "
           "mode=excluded.mode, started=excluded.started, ended=excluded.ended, status=excluded.status, "
           "outcome=excluded.outcome, updated_at=excluded.updated_at, candidate_ids=excluded.candidate_ids, "
           "summary=excluded.summary, record=excluded.record "
           # The newest version (by when it was last heard) wins; on a tie a closed copy is not
           # reopened by a running one.
           "WHERE excluded.updated_at > iterations.updated_at OR (excluded.updated_at = iterations.updated_at "
           "AND NOT (excluded.status = 'running' AND COALESCE(iterations.status, '') != 'running'))")


def _close_in_place(con: sqlite3.Connection, rid: str, blob: bytes, status: str, reason: str | None,
                    ended_at: float | None, extra: dict | None = None) -> None:
    """Close a row that is still "running", keeping everything it recorded: its status, end and
    summary change; `updated_at` (when the agent was last heard) does not."""
    rec = json.loads(zlib.decompress(blob))
    rec.update(status=status, ended_at=ended_at, pending=None)
    if reason:
        rec["end_reason"] = reason
    for k, v in (extra or {}).items():
        if v is not None:
            rec[k] = v
    summary = summarize(rec)
    con.execute("UPDATE iterations SET ended=?, status=?, outcome=?, summary=?, record=? WHERE id=?",
                (ended_at, status, summary["outcome"], json.dumps(summary, default=str),
                 zlib.compress(json.dumps(rec, default=str).encode(), 6), rid))


def _close_superseded(con: sqlite3.Connection, recs: list[dict]) -> None:
    """Closed records the upsert refused because the row is a NEWER running version of them:
    the inspector closed an older copy (e.g. one a restarted control plane reloaded from its
    file, while this log already had the later posts). The iteration is over either way, so
    close the row in place and keep its later content."""
    closing = {str(r["id"]): r for r in recs if isinstance(r, dict) and r.get("id") and r.get("status") != "running"}
    ids = list(closing)
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        rows = con.execute(f"SELECT id, record, updated_at FROM iterations WHERE status = 'running' AND id IN "
                           f"({','.join('?' * len(chunk))})", chunk).fetchall()
        for r in rows:
            inc = closing[r["id"]]
            ended = max(float(inc.get("ended_at") or 0), float(r["updated_at"] or 0)) or None
            _close_in_place(con, r["id"], r["record"], str(inc["status"]), inc.get("end_reason"), ended,
                            {k: inc.get(k) for k in ("reason", "outcome")})


def close_running(before: float, reason: str) -> list[str]:
    """Close as interrupted every iteration still "running" whose agent was last heard from
    before `before`; returns their ids. For a swarm runner that just started (what an earlier
    runner left open is over, including iterations the inspector no longer holds), and for
    iterations whose agent went silent. Never raises."""
    try:
        with _lock:
            con = db()
            rows = con.execute("SELECT id, record, updated_at FROM iterations WHERE status = 'running' AND updated_at < ?",
                               (before,)).fetchall()
            for r in rows:
                # It ended when it was last heard from (as the inspector's _abandon has it).
                _close_in_place(con, r["id"], r["record"], "interrupted", reason, r["updated_at"])
            if rows:
                con.commit()
        return [r["id"] for r in rows]
    except Exception as exc:  # noqa: BLE001 -- the work log is a view; it must never fail its caller
        logger.warning("work log: closing running iterations failed: %s", exc)
        return []


def archive(items: list[tuple[dict, list[dict]]]) -> int:
    """Upsert iteration records: `items` is [(agent meta {agent, model, role, project_id}, records)].
    Called by the inspector's flush for the records that changed. Never raises."""
    global _last_prune
    try:
        rows = [r for meta, recs in items for r in _rows(meta, recs)]
        if not rows:
            return 0
        with _lock:
            con = db()
            con.executemany(_UPSERT, rows)
            _close_superseded(con, [r for _, recs in items for r in recs])
            now = time.time()
            if now - _last_prune > PRUNE_EVERY_S:
                _last_prune = now
                con.execute("DELETE FROM iterations WHERE COALESCE(ended, started, updated_at) < ?", (now - RETENTION_S,))
            con.commit()
        return len(rows)
    except Exception as exc:  # noqa: BLE001 -- the work log is a view; it must never fail a post
        logger.warning("work log: archive failed: %s", exc)
        return 0


def _backfill(con: sqlite3.Connection) -> None:
    """Once, on the first open: copy in whatever the inspector's own file still holds."""
    n = 0
    src = Path(agent_activity.DB_PATH)
    if src.exists():
        try:
            ro = sqlite3.connect(f"{src.resolve().as_uri()}?mode=ro", uri=True, timeout=30.0)
            try:
                docs = [d for (d,) in ro.execute("SELECT doc FROM agents").fetchall()]
            finally:
                ro.close()
            for d in docs:
                try:
                    doc = json.loads(d)
                except ValueError:
                    continue
                rows = _rows(doc, doc.get("records") or [])
                con.executemany(_UPSERT, rows)
                n += len(rows)
        except sqlite3.Error as exc:
            logger.warning("work log: backfill from %s failed: %s", src, exc)
    con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('backfilled', ?)", (json.dumps({"at": time.time(), "records": n}),))
    con.commit()


# =======================================================================================
# Reading
# =======================================================================================
def _metric(meta: str | None) -> tuple[bool, str | None]:
    try:
        m = json.loads(meta or "{}")
    except ValueError:
        m = {}
    return bool(m.get("higher_is_better", True)), m.get("kind")


def _holdout(metrics: dict, kind: str | None) -> float | None:
    rank = metrics.get("rank") or {}
    if isinstance(rank, dict) and rank.get("holdout") is not None:
        return rank["holdout"]
    ho = metrics.get("holdout") or {}
    return ho.get(kind) if isinstance(ho, dict) and kind else None


def _candidates(project_id: str, since: float) -> list[dict]:
    con = _objectives_db()
    if con is None:
        return []
    with _lock:
        rows = con.execute(
            "SELECT c.id, c.objective_id, c.seq, c.created_at, c.model, c.mode, c.parent_id, c.rationale, c.status, "
            "c.score, c.is_score, c.score_note, c.lookahead, c.eval_seconds, c.idea_id, "
            "json_extract(c.metrics, '$.rank') AS rank_m, json_extract(c.metrics, '$.holdout') AS holdout_m, "
            "substr(c.stderr, -4000) AS stderr_tail, p.seq AS parent_seq, o.title AS objective_title, o.metric AS metric "
            "FROM candidates c JOIN objectives o ON o.id = c.objective_id LEFT JOIN candidates p ON p.id = c.parent_id "
            "WHERE o.project_id = ? AND c.created_at >= ? ORDER BY c.created_at DESC", (project_id, since)).fetchall()
    out = []
    for r in rows:
        higher, kind = _metric(r["metric"])
        metrics = {}
        for k, col in (("rank", "rank_m"), ("holdout", "holdout_m")):
            try:
                metrics[k] = json.loads(r[col]) if r[col] else {}
            except ValueError:
                metrics[k] = {}
        err = ""
        if r["status"] == "error":
            err = error_line(r["stderr_tail"]) or r["score_note"] or "error"
        out.append({
            "kind": "candidate", "id": r["id"], "at": r["created_at"], "objective_id": r["objective_id"],
            "objective_title": r["objective_title"], "seq": r["seq"], "model": r["model"], "agent": None,
            "mode": r["mode"], "parent_seq": r["parent_seq"], "status": r["status"],
            "outcome": "ok" if r["status"] == "ok" else "error", "score": r["score"], "is_score": r["is_score"],
            "holdout": _holdout(metrics, kind), "higher_is_better": higher, "score_note": r["score_note"] or "",
            "lookahead": r["lookahead"], "eval_seconds": r["eval_seconds"], "idea": r["idea_id"],
            "rationale": _clip(r["rationale"], 800), "error_line": err,
        })
    return out


# =======================================================================================
# The clear marker
# =======================================================================================
def _clear_key(project_id: str) -> str:
    return f"cleared_at:{project_id}"


def cleared_at(project_id: str) -> float | None:
    """When the page was last cleared for this project, or None."""
    with _lock:
        r = db().execute("SELECT value FROM meta WHERE key = ?", (_clear_key(project_id),)).fetchone()
    try:
        return float(r[0]) if r is not None else None
    except (TypeError, ValueError):
        return None


def set_cleared(project_id: str, at: float | None = None) -> float:
    """Hide the work so far from the page (nothing is deleted); returns the marker."""
    at = time.time() if at is None else float(at)
    with _lock:
        con = db()
        con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (_clear_key(project_id), repr(at)))
        con.commit()
    return at


def unclear(project_id: str) -> None:
    """Undo the clear: the page shows the whole window again."""
    with _lock:
        con = db()
        con.execute("DELETE FROM meta WHERE key = ?", (_clear_key(project_id),))
        con.commit()


def _after_clear(it: dict, cleared: float) -> bool:
    """Work that is new since the clear. An iteration started before it but still running, or
    ending after it, is ongoing work and stays."""
    if it["kind"] == "candidate":
        return (it["at"] or 0) > cleared
    return (it["at"] or 0) > cleared or it["status"] == "running" or (it["ended"] or 0) > cleared


def _iterations(project_id: str, since: float) -> list[dict]:
    with _lock:
        rows = db().execute(
            "SELECT id, agent, model, role, objective_id, objective_title, mode, started, ended, status, summary "
            "FROM iterations WHERE project_id = ? AND COALESCE(ended, started, updated_at) >= ? ORDER BY started DESC",
            (project_id, since)).fetchall()
    out = []
    for r in rows:
        try:
            s = json.loads(r["summary"])
        except ValueError:
            continue
        out.append({"kind": "iteration", "id": r["id"], "at": r["started"], "ended": r["ended"], "agent": r["agent"],
                    "model": r["model"], "role": r["role"], "objective_id": r["objective_id"],
                    "objective_title": r["objective_title"], "mode": r["mode"], "status": r["status"], **s})
    return out


def _search_text(it: dict) -> str:
    if it["kind"] == "candidate":
        parts = [f"#{it['seq']}", it["id"], it["model"], it["mode"], it["status"], it["rationale"], it["score_note"],
                 it["error_line"], it["objective_title"], it.get("agent"), it["lookahead"]]
    else:
        parts = [it["id"], it["agent"], it["model"], it["mode"], it["status"], it["outcome"], it.get("outcome_text"),
                 it.get("reason"), it["hypothesis"], it["objective_title"], " ".join(it["tools"]),
                 " ".join(f"#{s.get('seq')}" for s in it["submissions"]), " ".join(e["line"] for e in it["errors"])]
    return " ".join(str(p) for p in parts if p).lower()


def _matches(it: dict, agent: str, outcome: str, kind: str, q: str) -> bool:
    if kind and it["kind"] != kind:
        return False
    if agent and agent not in (it.get("agent"), it.get("model")):
        return False
    if outcome:
        if outcome == "tool_errors":
            # What stayed broken: a failure the agent fixed later in its iteration is not one.
            hit = (it["outcome"] == "error" and not it.get("recovered")) if it["kind"] == "candidate" \
                else (it["tool_errors"] + it["chat_errors"]) > 0
            if not hit:
                return False
        elif it["outcome"] != outcome:
            return False
    return not q or q in _search_text(it)


def _summary(iters: list[dict], cands: list[dict]) -> dict:
    by_outcome = Counter(i["outcome"] for i in iters)
    groups: dict[str, dict] = {}

    def note(line: str, source: str, at: float | None) -> None:
        if not line:
            return
        g = groups.setdefault(error_key(line), {"line": line, "count": 0, "sources": Counter(), "last_at": 0.0})
        g["count"] += 1
        g["sources"][source] += 1
        g["last_at"] = max(g["last_at"], float(at or 0))

    # Only what stayed broken: an error the agent fixed later in the same iteration (recovered),
    # or the runner repaired (auto-repaired), is counted apart, not grouped here.
    for i in iters:
        for e in i["errors"]:
            if e.get("state", "failed") == "failed":
                note(e["line"], e.get("tool") or "tool", e.get("at") or i["at"])
    for c in cands:
        if c["status"] == "error" and not c.get("recovered"):
            note(c["error_line"], "candidate", c["at"])
    errors = sorted(groups.values(), key=lambda g: (-g["count"], -g["last_at"]))[:15]
    best: dict[str, dict] = {}
    for c in cands:
        if c["status"] != "ok" or c["score"] is None or c["lookahead"] in ("fail", "error"):
            continue
        b = best.get(c["objective_id"])
        if b is None or (c["score"] > b["score"] if c["higher_is_better"] else c["score"] < b["score"]):
            best[c["objective_id"]] = {k: c[k] for k in ("objective_id", "objective_title", "id", "seq", "model", "score",
                                                          "is_score", "holdout", "higher_is_better", "at")}
    names = Counter()
    for it in iters + cands:
        for n in {it.get("agent"), it.get("model")}:
            if n:
                names[n] += 1
    return {
        "iterations": len(iters),
        "by_outcome": dict(by_outcome),
        "candidates": len(cands),
        "candidates_ok": sum(1 for c in cands if c["status"] == "ok"),
        # Failed candidates their iteration did not make good: no later submission of it ran, and
        # the runner did not repair it ...
        "candidates_error": sum(1 for c in cands if c["status"] == "error" and not c.get("recovered")),
        # ... and the failed ones it did (recovered).
        "candidates_error_recovered": sum(1 for c in cands if c["status"] == "error" and c.get("recovered")),
        "tool_calls": sum(i["tool_calls"] for i in iters),
        "experiments": sum(i["experiments"] for i in iters),
        "experiments_failed": sum(i["experiments_failed"] for i in iters),
        "experiments_recovered": sum(i.get("experiments_recovered", 0) for i in iters),
        "experiments_refused": sum(i.get("experiments_refused", 0) for i in iters),
        "refusals": sum(i.get("refusals", 0) for i in iters),
        "refusals_by_kind": dict(sum((Counter(i.get("refusals_by_kind") or {}) for i in iters), Counter())),
        "refusals_recovered": sum(i.get("refusals_recovered", 0) for i in iters),
        "tool_errors": sum(i["tool_errors"] for i in iters),
        "tool_errors_recovered": sum(i.get("tool_errors_recovered", 0) for i in iters),
        "tool_auto_repaired": sum(i.get("tool_auto_repaired", 0) for i in iters),
        "chat_errors": sum(i["chat_errors"] for i in iters),
        "chat_errors_soft": sum(i.get("chat_errors_soft", 0) for i in iters),
        "truncations": sum(i["truncations"] for i in iters),
        "best": sorted(best.values(), key=lambda b: -b["at"]),
        "errors": [{**g, "sources": dict(g["sources"])} for g in errors],
        "agents": sorted(names),
    }


def work(project_id: str, since: float, limit: int = 150, agent: str = "", outcome: str = "", kind: str = "",
         q: str = "", include_cleared: bool = False) -> dict:
    """The work log for one project since `since`: a summary of the whole window, and the
    newest `limit` rows that pass the filters. With a clear marker set (and `include_cleared`
    off), the window starts at the marker instead when that is later."""
    # An iteration whose agent stopped reporting long ago is not running, whether or not the
    # inspector still holds it (it keeps only the last few per agent).
    close_running(time.time() - agent_activity.SILENT_AFTER_S, agent_activity.SILENT_REASON)
    cleared = cleared_at(project_id)
    hiding = cleared is not None and not include_cleared and cleared > since
    iters = _iterations(project_id, since)
    cands = _candidates(project_id, since)
    by_cid = {s["candidate_id"]: i["agent"] for i in iters for s in i["submissions"] if s.get("candidate_id")}
    fixed = {cid for i in iters for cid in i.get("recovered_candidates") or ()}
    windows = [(i["objective_id"], i["model"], w) for i in iters for w in i.get("recovered_submits") or ()]
    for c in cands:
        c["agent"] = by_cid.get(c["id"])
        # A failed candidate whose iteration went on to submit one that ran: a later submission
        # did, or the runner repaired this one (the crashed run, and any failed repair attempt,
        # stay rows of their own).
        c["recovered"] = c["status"] == "error" and (
            c["id"] in fixed or any(made_by(c, o, m, w) for o, m, w in windows))
    if hiding:
        iters = [i for i in iters if _after_clear(i, cleared)]
        cands = [c for c in cands if _after_clear(c, cleared)]
    summary = _summary(iters, cands)
    q = q.strip().lower()
    rows = [it for it in iters + cands if _matches(it, agent, outcome, kind, q)]
    rows.sort(key=lambda it: it["at"] or 0, reverse=True)
    for it in rows[:limit]:
        if it["kind"] == "iteration":
            # A row shows what stayed broken; the iteration's detail has the recovered ones too.
            broken = [e for e in it["errors"] if e.get("state", "failed") == "failed"]
            it["error_count"] = len(broken)
            it["recovered_count"] = sum(1 for e in it["errors"] if e.get("state") in ("recovered", "auto_repaired"))
            it["refused_count"] = sum(1 for e in it["errors"] if e.get("state") == "refused")
            it["soft_count"] = sum(1 for e in it["errors"] if e.get("state") == "soft")
            it["errors"] = broken[-ROW_ERRORS:]
            it.pop("recovered_candidates", None)
            it.pop("recovered_submits", None)
    return {"now": time.time(), "since": max(since, cleared) if hiding else since, "window_since": since,
            "cleared_at": cleared, "hiding_cleared": hiding, "include_cleared": include_cleared,
            "summary": summary, "total": len(rows), "items": rows[:limit],
            "counts": {"all": len(iters) + len(cands), "iteration": len(iters), "candidate": len(cands)}}


def iteration(rid: str) -> dict | None:
    """One iteration in full: its record, each tool call marked with whether it failed and why."""
    with _lock:
        r = db().execute("SELECT * FROM iterations WHERE id = ?", (rid,)).fetchone()
    if r is None:
        return None
    rec = json.loads(zlib.decompress(r["record"]))
    tl = rec.get("timeline") or []
    states = recovery(tl, running=rec.get("status") == "running")
    windows = side_windows(rec)
    for c in rec.get("chats") or []:
        _mark_soft_chat(c, windows)
    for i, e in enumerate(tl):
        if e.get("kind") == "chat":
            _mark_soft_chat(e, windows)
        if e.get("kind") != "tool":
            continue
        st = states.get(i)
        why = refusal(e) if st in ("refused", "recovered") else None
        if why:
            # A policy refusal: not a failed call. "recovered": the agent resent it and it ran.
            e["refused"] = True
            e["refusal"] = why
            e["recovered"] = st == "recovered"
            e["error_line"], e["error_tail"] = _tool_error(e)
            continue
        if soft_step(e):
            res = e.get("result")
            err = (res.get("soft_error") or res.get("error")) if isinstance(res, dict) else None
            if err is None and isinstance(res, str) and not e.get("ok", True):
                err = _as_text(res)
            if err:
                e["soft"] = True
                e["soft_error"] = error_line(err) or str(err)[:200]
            continue
        if st == "auto_repaired":
            info = auto_repair(e) or {}
            e["auto_repaired"] = True
            e["repair_attempts"] = info.get("attempts")
            e["repair_errors"] = repaired_errors(info)
        elif st is not None:
            e["failed"] = True
            e["recovered"] = st == "recovered"
            e["error_line"], e["error_tail"] = _tool_error(e)
    return {"id": r["id"], "project_id": r["project_id"], "agent": r["agent"], "model": r["model"], "role": r["role"],
            "summary": json.loads(r["summary"]), "record": rec}


def _mark_soft_chat(c: dict, windows: list[tuple[float, float, str]]) -> None:
    """A failed side request (see side_purpose), as the detail shows it: soft, its error under
    "soft_error" (not "error", the runner's own convention for soft steps), "skipped (model busy)"."""
    if not isinstance(c, dict) or not c.get("error"):
        return
    side = side_purpose(c, windows)
    if side:
        c["soft"] = True
        c["purpose"] = side
        c["soft_error"] = c.pop("error")
        c["soft_note"] = SOFT_CHAT_NOTE


def records(ids: list[str]) -> dict[str, dict]:
    """Archived iteration records by id (the full timeline), for those the log still has."""
    out: dict[str, dict] = {}
    ids = [str(i) for i in dict.fromkeys(ids) if i]
    with _lock:
        con = db()
        for k in range(0, len(ids), 500):
            chunk = ids[k:k + 500]
            for r in con.execute(f"SELECT id, record FROM iterations WHERE id IN ({','.join('?' * len(chunk))})",
                                 chunk).fetchall():
                try:
                    out[r["id"]] = json.loads(zlib.decompress(r["record"]))
                except (ValueError, zlib.error):
                    continue
    return out


def candidate(cid: str) -> dict | None:
    """One candidate in full: code, rationale, scores, look-ahead and audit verdicts, output."""
    con = _objectives_db()
    if con is None:
        return None
    with _lock:
        r = con.execute(
            "SELECT c.*, p.seq AS parent_seq, o.title AS objective_title, o.metric AS metric, o.project_id AS project_id "
            "FROM candidates c JOIN objectives o ON o.id = c.objective_id LEFT JOIN candidates p ON p.id = c.parent_id "
            "WHERE c.id = ?", (cid,)).fetchone()
    if r is None:
        return None
    higher, kind = _metric(r["metric"])
    try:
        metrics = json.loads(r["metrics"] or "{}")
    except ValueError:
        metrics = {}
    keep = {k: metrics[k] for k in ("in_sample", "holdout", "full", "rank", "change", "features_used", "warning")
            if k in metrics}
    with _lock:
        iters = [x["id"] for x in db().execute("SELECT id FROM iterations WHERE candidate_ids LIKE ?",
                                               (f"%|{cid}|%",)).fetchall()]
    return {
        "id": r["id"], "objective_id": r["objective_id"], "objective_title": r["objective_title"],
        "project_id": r["project_id"], "seq": r["seq"], "created_at": r["created_at"], "model": r["model"],
        "mode": r["mode"], "parent_id": r["parent_id"], "parent_seq": r["parent_seq"], "rationale": r["rationale"] or "",
        "code": _tail(r["code"], 60_000), "status": r["status"], "score": r["score"], "is_score": r["is_score"],
        "holdout": _holdout(metrics, kind), "higher_is_better": higher, "score_note": r["score_note"] or "",
        "metrics": keep, "lookahead": r["lookahead"], "lookahead_detail": _clip(r["lookahead_detail"], 4000),
        "audit": r["audit"], "audit_notes": _clip(r["audit_notes"], 4000), "eval_seconds": r["eval_seconds"],
        "idea": r["idea_id"], "error_line": error_line(r["stderr"]) if r["status"] == "error" else "",
        "stderr": _tail(r["stderr"], 8000), "stdout": _tail(r["stdout"], 4000), "iterations": iters,
    }


# =======================================================================================
# Routes (mounted under /api)
# =======================================================================================
@router.get("/projects/{project_id}/work")
async def work_list(project_id: str, hours: float = Query(24.0, gt=0, le=MAX_WINDOW_H),
                    since: float | None = Query(None), limit: int = Query(150, ge=1, le=1000),
                    agent: str = Query("", max_length=300),
                    outcome: str = Query("", pattern="^(|ok|error|no_submission|interrupted|running|done|tool_errors)$"),
                    kind: str = Query("", pattern="^(|iteration|candidate)$"), q: str = Query("", max_length=300),
                    include_cleared: bool = Query(False)) -> dict:
    """Everything the swarm did in the window, newest first: iterations and candidates. After a
    clear, only the work since it unless `include_cleared`."""
    # Records the inspector has not flushed yet (the iteration running now) are copied first --
    # by a full flush, so its own file never lags behind what this log shows.
    await asyncio.to_thread(agent_activity.flush)
    start = since if since is not None else time.time() - hours * 3600.0
    return await asyncio.to_thread(work, project_id, start, limit, agent, outcome, kind, q, include_cleared)


@router.post("/projects/{project_id}/work/clear")
async def work_clear(project_id: str) -> dict:
    """Hide the work so far from the page (a view marker: nothing is deleted)."""
    return {"cleared_at": await asyncio.to_thread(set_cleared, project_id)}


@router.delete("/projects/{project_id}/work/clear")
async def work_unclear(project_id: str) -> dict:
    """Undo the clear."""
    await asyncio.to_thread(unclear, project_id)
    return {"cleared_at": None}


@router.get("/work/iterations/{rid}")
async def work_iteration(rid: str) -> dict:
    doc = await asyncio.to_thread(iteration, rid)
    if doc is None:
        raise HTTPException(status_code=404, detail="no such iteration in the work log")
    return doc


@router.get("/work/candidates/{cid}")
async def work_candidate(cid: str) -> dict:
    doc = await asyncio.to_thread(candidate, cid)
    if doc is None:
        raise HTTPException(status_code=404, detail="no such candidate")
    return doc
