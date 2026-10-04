"""The worker that runs queued swarm tasks -- inside a project, with that project's tools.

The message board is a *pull* queue: `POST /mb/tasks` only records a task; it stays open
until something calls `POST /mb/tasks/claim`. This process is that consumer.

**A swarm runs inside a project.** A project owns a message board, a data folder, an
optional read-only SQL Server connection, a set of active connectors (MCP), and a choice of
which loaded models it may use. For every project, this runner keeps one agent per allowed
LLM, and each agent answers that project's tasks using that project's resources only.

**Agents use real tool calls.** The agent's LLM is given OpenAI-style tools and decides for
itself when to use them; the runner executes each call against the control plane and feeds
the result back, for up to MAX_TOOL_ROUNDS rounds:

* ``list_data`` / ``describe_data`` / ``query_data`` -- SQL over the data folder's parquet and
  CSV files (DuckDB, confined to that folder by DuckDB itself);
* ``list_sql_tables`` / ``describe_sql_table`` / ``query_sql`` -- the project's SQL Server
  tables, through a login that SQL Server restricts to SELECT on the chosen tables;
* ``list_forecasters`` / ``forecast`` -- the loaded time-series models;
* ``ask_model`` -- delegate a sub-question to another loaded LLM the project allows;
* ``<connector>__<tool>`` -- every tool of the project's active MCP connectors.

Every tool call is posted to the project's board, so the board shows which model did what,
with which tool, and when. Loading models stays a human action: agents only use what is
already loaded.

**Standing objectives.** When no one-off task is queued, an agent works on its project's
running objective (app/objectives.py): one ITERATION at a time, forever, until the operator
pauses it. An iteration is a small evolutionary step --

1. fetch the context: the objective, how it is scored, the operator's steering notes, the
   team's lessons, the leaderboard, recent attempts, and an assignment chosen by the control
   plane: IMPROVE a parent picked by tournament from the top ranks, or EXPLORE a new idea;
2. investigate with the usual tools -- which, in objective mode, see only IN-SAMPLE data --
   and test code with run_python;
3. submit_candidate: the control plane runs it in the sandbox and scores it (holdout rank,
   look-ahead test); a failed run may be fixed and resubmitted (MAX_SUBMITS);
4. reflect: write one lesson for the team (KEEP / AVOID / TRY), which every later iteration
   reads. Lessons are periodically consolidated by an agent into a short list.

Chores come first when the control plane hands one out: auditing a would-be champion's code
(done by a DIFFERENT loaded model when there is one), and consolidating lessons. This is the
"agents get smarter" loop: better parents to build on, a growing memory of what works and
what does not, and a critic between a good number and the title.

stdlib only (urllib + threading), matching the connectors.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import http.client
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

CONTROL_PLANE = os.getenv("FREESWARM_API_URL", "http://127.0.0.1:8500").rstrip("/")
BOARD = os.getenv("FREESWARM_BOARD_URL", "http://127.0.0.1:8510").rstrip("/")

STATIC_TOKEN = os.getenv("FREESWARM_API_TOKEN", "").strip()
AGENT_USER = os.getenv("FREESWARM_AGENT_USER", "").strip()
AGENT_PASSWORD = os.getenv("FREESWARM_AGENT_PASSWORD", "")

POLL_IDLE_S = float(os.getenv("FREESWARM_SWARM_POLL_S", "3"))
ENGINE_RESYNC_S = float(os.getenv("FREESWARM_SWARM_RESYNC_S", "15"))
# After a start, the best free model (the one that mentors once the local engines are up) is not
# given search agents for this long: it is often the only model ready at first, and every search
# iteration it began was retired minutes later, unfinished, when it became the mentor (bug #8 --
# DeepSeek never submitted once in 1838 candidates).
STARTUP_GRACE_S = float(os.getenv("FREESWARM_SWARM_STARTUP_GRACE_S", "900"))
HEARTBEAT_S = 20.0
LEASE_S = int(os.getenv("FREESWARM_SWARM_LEASE_S", "300"))
# Graceful drain (ui\restart-swarm.cmd, stop-services.cmd --drain). A restart used to kill the
# runner outright, and with it every iteration in flight (median 45 min of work each; ~50 were
# lost in one day). While DRAIN_FILE exists no agent starts anything new -- no iteration, chore,
# mentor pass or task -- and what is running finishes normally. Once nothing runs, the runner
# deletes the file and exits 0; after DRAIN_MAX_S it exits anyway, logging what it cut. Progress
# goes to the log, to DRAIN_STATUS_FILE (read by the restart script) and to the board key
# "swarm_drain". Deleting the file cancels the drain.
DRAIN_FILE = os.getenv("FREESWARM_DRAIN_FILE") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".swarm_drain")
DRAIN_MAX_S = float(os.getenv("FREESWARM_DRAIN_MAX_S", "5400"))
DRAIN_POLL_S = 5.0
DRAIN_REPORT_S = float(os.getenv("FREESWARM_DRAIN_REPORT_S", "180"))
# 16K: a coding model that reasons before it writes (Qwen3.6) ran out mid-script at 8K --
# "spent its 8192-token budget reasoning and never answered". Still capped by the window.
MAX_TOKENS = int(os.getenv("FREESWARM_SWARM_MAX_TOKENS", "16384"))
GENERATION_TIMEOUT_S = int(os.getenv("FREESWARM_SWARM_TIMEOUT_S", "1800"))
# A run may first have to build the forecasts its script asks for with ft.forecast (up to 3).
FORECAST_BUILD_ALLOWANCE_S = 1800
MAX_TOOL_ROUNDS = int(os.getenv("FREESWARM_SWARM_TOOL_ROUNDS", "10"))
# A tool result is fed back into the model's context, so it must stay small: a SELECT *
# on a big table would otherwise crowd out everything else the model is holding. This is the
# CEILING; the real limit is scaled to the engine's context window (see _result_budget).
TOOL_RESULT_CHARS = 12_000

# --- context window ------------------------------------------------------------------------
# An engine's usable context is min(model max, KV pages x page size) -- often FAR below the
# model's advertised maximum (an engine launched with 8192 KV pages holds 8K tokens even though
# the model supports 262K). Every tool result is fed back into the prompt, so an agent that
# does not budget against the real window overflows after two or three calls: the first real
# failure was a 15,605-token prompt against an 8,192-token engine, after the agent described
# one wide table twice.
#
# Token counts are ESTIMATED from characters (no tokenizer in this process). JSON and code
# tokenize denser than prose, so the ratio is deliberately conservative.
CHARS_PER_TOKEN = 3.0
DEFAULT_CONTEXT = 8192
ANSWER_RESERVE = 1024  # room the model needs to actually say something after its tools
# The output-token floor for a chat request. Anything smaller silently truncates a tool call
# (Groq will happily return 44-token half-JSON with finish "tool_calls", or 256 tokens of prose
# with finish "length") -- one experiment burned on `bar = ft.load('sql_exports_db` because the
# runner's own `max_tokens` came out to 256 when it thought the window was DEFAULT_CONTEXT and
# the real prompt was 25,720 tokens. A 1024-token floor still fits inside ANSWER_RESERVE + 128
# after successful compaction; when it doesn't (an unknown or wrong context), we send it anyway
# and let the provider's real overflow refusal correct the window instead of half-answering.
MIN_OUTPUT = 1024
_context: dict[str, int] = {}  # model -> usable tokens, from engine stats or a refusal
# Provider hard cap on generated tokens per request (Groq/OpenRouter publish this in their
# catalogs; local engines have none, capped only by the context window). None/absent means
# "no known cap -- MAX_TOKENS is the ceiling".
_max_output: dict[str, int] = {}
# Windows a refusal told us exactly: engine stats and the sync loop never overwrite these.
_learned: set[str] = set()
# A paired computer's model whose engine misreports its window (DeepSeek-V4's 128K came back
# as 1024). Guessing SMALL never corrects itself -- the model never overflows, so the guess
# stands, and a 24K prompt against an 8K guess left the 256-token floor for the answer: every
# DeepSeek search iteration was cut off at 255 tokens and never submitted. Guessing LARGE does:
# the first oversize prompt is refused with the real window, which _learned then keeps.
UNVERIFIED_CONTEXT = 131072
# model -> engine tokens per ESTIMATED token. The character estimate is only a starting point:
# every reply reports the engine's real prompt_tokens and every refusal states the real size,
# so the estimate is calibrated against the tokenizer that actually counts. Without this, a
# refusal was "handled" by compacting to the runner's own (3x low) estimate -- which judged
# the prompt already small enough, trimmed nothing, and resent the identical prompt.
_scale: dict[str, float] = {}


def context_for(model: str) -> int:
    return _context.get(model, DEFAULT_CONTEXT)


def max_output_for(model: str) -> int | None:
    """Provider hard cap on tokens generated per request, or None if unknown."""
    return _max_output.get(model)


def _est_tokens(messages: list[dict], tools: list[dict], scale: float = 1.0) -> int:
    chars = len(json.dumps(messages, default=str)) + len(json.dumps(tools, default=str))
    return int((chars / CHARS_PER_TOKEN + 64) * scale)


# What a reasoning model may spend thinking before a tool-less answer (an audit verdict, a list
# of lessons), on top of the answer's own size. max_tokens is a ceiling, not a cost: a model
# that does not think stops early and pays nothing for the room.
REASONING_ROOM = 4096


def _toolless_budget(model: str, messages: list[dict], want: int) -> int:
    """max_tokens for a tool-less request whose answer needs about `want` tokens: that plus
    REASONING_ROOM where the window has it, never less than `want`, never more than the
    provider's cap on generated tokens or the runner's own (MAX_TOKENS)."""
    return min(max_output_for(model) or MAX_TOKENS, MAX_TOKENS,
               max(want, min(_output_room(model, messages), want + REASONING_ROOM)))


def _output_room(model: str, messages: list[dict]) -> int:
    """Tokens the model's window leaves for a reply to `messages` (estimated, calibrated)."""
    return context_for(model) - int(_est_tokens(messages, []) * _scale.get(model, 1.0)) - 128


def _result_budget(ctx: int) -> int:
    """Chars one tool result may occupy: ~12% of the window, between 1.5K and the ceiling."""
    return max(1500, min(TOOL_RESULT_CHARS, int(ctx * 0.12 * CHARS_PER_TOKEN)))


_OMITTED = "[earlier result omitted to fit the context window -- call the tool again if needed]"


def _compact(messages: list[dict], tools: list[dict], limit: int, core_tools: list[dict],
             scale: float = 1.0) -> list[dict]:
    """Shrink the prompt to fit `limit` tokens, oldest material first. Returns the tool list
    to use (possibly reduced). Mutates `messages`.

    Order of sacrifice: old tool results (the model already acted on them), then connector
    tool schemas (the built-in tools are what a data task needs), then the latest results.
    """
    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    # Keep the most recent round's results for last: the model is mid-way through using them.
    last_assistant = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    older = [i for i in tool_idx if i < last_assistant]
    for i in older:
        if _est_tokens(messages, tools, scale) <= limit:
            return tools
        if messages[i]["content"] != _OMITTED:
            messages[i]["content"] = _OMITTED
    if _est_tokens(messages, tools, scale) > limit and len(tools) > len(core_tools):
        tools = core_tools
    for i in [i for i in tool_idx if i > last_assistant]:
        if _est_tokens(messages, tools, scale) <= limit:
            break
        content = messages[i]["content"]
        keep = max(400, len(content) // 3)
        messages[i]["content"] = content[:keep] + "\n...[cut to fit the context window]"
    return tools


def _parse_overflow(message: str) -> tuple[int, int] | None:
    """'prompt is too long: 15605 tokens > 8192 maximum' -> (15605, 8192): the prompt's REAL
    size by the engine's tokenizer, and the window."""
    m = re.search(r"(\d+)\s+tokens\s*>\s*(\d+)\s+maximum", message)
    return (int(m.group(1)), int(m.group(2))) if m else None


# --- hosted providers: transient refusals and the daily budget ------------------------------
# A hosted model's iteration is many paid rounds. Before, ANY failed request ended the
# iteration and threw all of them away: Qwen on Groq spent $10 over two days and produced no
# candidate, its iterations dying on Groq's tokens-per-minute 429 ("Please try again in
# 166.08ms" -- three agents share one TPM allowance) and on Groq's 400 tool_use_failed (a
# malformed tool call the model would get right on a second try). Both are now retried in place.
RATE_LIMIT_RETRIES = 6
RATE_LIMIT_MAX_WAIT_S = 60.0      # longer hints are daily quotas (TPD): waiting will not help
BAD_TOOL_CALL_RETRIES = 2         # per round
# The console's own refusal once today's external budget is spent (app/external.py). Nothing
# changes until midnight or until the operator raises the limit, so the agent waits quietly.
BUDGET_BACKOFF_S = float(os.getenv("FREESWARM_SWARM_BUDGET_BACKOFF_S", "600"))
# Same-args-same-error re-issues of a failed tool call. In agent_activity, one Qwen/Qwen3-0.6B
# agent re-sent the same failing library_save 7 times in one iteration (bug #12); a second
# iteration by the same model repeated 3 times. Every other tool had zero same-args repeats
# across 104 iterations. So the intercept starts on the FIRST duplicate (return an error
# without executing so the tool budget is not spent) and the iteration ends after this many
# distinct intercepted duplicates -- past this the model is not learning from the error text.
REPEAT_TOOL_LIMIT = int(os.getenv("FREESWARM_SWARM_REPEAT_TOOL_LIMIT", "3"))


def _spending_limited(message: str) -> bool:
    return "spending limit" in message.lower()


def _rate_limit_wait(message: str) -> float | None:
    """Seconds to wait before resending after a provider rate-limit 429, else None.

    Groq says "Please try again in 166.08ms" / "in 1.5s" / "in 7m12.5s". The wait is floored
    at 1 s (three agents share one allowance -- resending at +166 ms just collides again) and
    jittered so they do not return in lockstep. A hint beyond RATE_LIMIT_MAX_WAIT_S is a daily
    quota, not a burst, and is not retried.
    """
    low = message.lower()
    if _spending_limited(message) or not ("rate_limit_exceeded" in low or "rate limit" in low):
        return None
    m = re.search(r"try again in\s+((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)", message, re.I)
    if m:
        units = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
        hint = sum(float(v) * units[u.lower()]
                   for v, u in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", m.group(1), re.I))
        if hint > RATE_LIMIT_MAX_WAIT_S * 5:
            return None
    else:
        hint = 5.0
    return min(RATE_LIMIT_MAX_WAIT_S, max(1.0, hint)) + random.uniform(0.25, 1.5)


def _bad_tool_call(message: str) -> str | None:
    """The model's malformed output when the provider rejected its tool call (Groq's 400
    tool_use_failed / "tool call validation failed"), '' if none was returned, else None."""
    low = message.lower()
    if "tool_use_failed" not in low and "tool call validation failed" not in low:
        return None
    # request() raises "<path> -> 400: <body>"; the body is the provider's JSON error.
    start = message.find("{")
    try:
        body = json.loads(message[start:]) if start >= 0 else {}
    except ValueError:
        body = {}
    err = body.get("error") if isinstance(body, dict) else None
    return str(err.get("failed_generation") or "") if isinstance(err, dict) else ""


_XML_INVOKE = re.compile(r'<[\w:.-]*invoke\s+name="([\w.-]+)"\s*>(.*?)(?:</[\w:.-]*invoke>|$)', re.S)
_XML_PARAM = re.compile(r'<[\w:.-]*parameter\s+name="([\w.-]+)"[^>]*>(.*?)</[\w:.-]*parameter>', re.S)
_HARMONY = re.compile(r'to=(?:functions\.)?([\w-]+)[^{]*?<\|message\|>', re.S)
# Qwen3-Coder / Qwen3.6's own syntax: <tool_call><function=NAME><parameter=KEY>value</parameter>
# ...</function></tool_call>.
_QWEN_FUNCTION = re.compile(r'<function=([\w.-]+)>(.*?)(?:</function>|$)', re.S)
_QWEN_PARAM = re.compile(r'<parameter=([\w.-]+)>(.*?)</parameter>', re.S)


def _text_tool_calls(text: str, names: set[str]) -> list[tuple[str, dict]]:
    """Tool calls a model wrote as TEXT instead of making them -- recovered so the work counts.

    Seen from the swarm's models: gpt-oss's own channel syntax (``<|start|> to=submit_candidate
    <|message|>{...}``), XML-ish ``<invoke name=...><parameter name=...>`` blocks (sometimes both
    at once), Qwen's ``<function=...><parameter=...>`` blocks, and a bare ``{"name": ...,
    "arguments": {...}}``. Only known tool names count, and an XML parameter only when it is
    closed -- output cut off mid-script is not submitted.

    Qwen3.6 writes its call in its own syntax in the final round (which offers no tools and asks
    for submit_candidate): none of the others matched it, so a complete, closed submission was
    dropped and the iteration ended "no submission" (bug #31).
    """
    out: list[tuple[str, dict]] = []
    for pattern, param in ((_XML_INVOKE, _XML_PARAM), (_QWEN_FUNCTION, _QWEN_PARAM)):
        for name, body in pattern.findall(text or ""):
            if name in names:
                args = {k: v.strip("\n") for k, v in param.findall(body)}
                if args:
                    out.append((name, args))
        if out:
            return out
    dec = json.JSONDecoder()
    for m in _HARMONY.finditer(text or ""):
        start = text.find("{", m.end())
        if m.group(1) in names and start >= 0:
            try:
                args, _ = dec.raw_decode(text[start:])
            except ValueError:
                continue
            if isinstance(args, dict):
                out.append((m.group(1), args))
    if out:
        return out
    start = (text or "").find("{")
    while start >= 0:
        try:
            obj, end = dec.raw_decode(text[start:])
        except ValueError:
            start = text.find("{", start + 1)
            continue
        if isinstance(obj, dict) and obj.get("name") in names:
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = None
            if isinstance(args, dict):
                out.append((obj["name"], args))
        start = text.find("{", start + end)
    return out


MCP_TOOLS_TTL_S = 300.0
OBJECTIVE_TOOL_ROUNDS = int(os.getenv("FREESWARM_SWARM_OBJECTIVE_ROUNDS", "14"))
# Submissions per iteration: the first, and fixes of a run that failed. One fix was too few: on
# the night of 2026-09-30 Qwen3.6-35B crashed on both of its tries (an unknown trend_exits
# keyword, a polars Series/expression mix-up -- 2-3 s each) in 6 of 7 iterations and lost the
# ~40 minutes of research behind each. A clean run still ends the iteration at once.
MAX_SUBMITS = int(os.getenv("FREESWARM_SWARM_MAX_SUBMITS", "4"))
# Experiments per iteration. Without a cap Qwen spent whole iterations in 10+ private
# run_python experiments and never saved or submitted anything the team could use.
# The cap of 6 ran too tight for the productive models: Qwen3.6-35B and Muse-Glimmer-30B
# both hit the wall after 5-6 legitimate experiments (mostly successful) and only just
# submitted afterwards. 8 gives one or two runs of head-room; a platform failure inside
# run_python (network / harness) does not count against it (see ObjectiveWorld.call).
MAX_EXPERIMENTS = int(os.getenv("FREESWARM_SWARM_EXPERIMENTS", "8"))
OBJECTIVE_POLL_S = 10.0

_LOG_LOCK = threading.Lock()

# The runner lives in a cmd.exe window whose console encoding is cp1252/cp437. Printing a
# task title with an arrow, an emoji or non-Latin text raised UnicodeEncodeError inside log()
# -- AFTER the task was claimed -- so the task sat stuck until its lease expired, then was
# claimed and crashed again. Never let logging be what fails a task.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def log(msg: str) -> None:
    with _LOG_LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def drain_requested() -> bool:
    """Whether a graceful drain was asked for (see DRAIN_FILE)."""
    try:
        return os.path.exists(DRAIN_FILE)
    except (OSError, ValueError):
        return False


# =======================================================================================
# HTTP + auth
# =======================================================================================
class Auth:
    """Bearer token holder. Thread-safe; refreshes at most one login at a time."""

    def __init__(self) -> None:
        self._token = STATIC_TOKEN
        self._lock = threading.Lock()

    def header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def refresh(self) -> bool:
        if STATIC_TOKEN or not AGENT_USER:
            return False
        with self._lock:
            body = urllib.parse.urlencode({"username": AGENT_USER, "password": AGENT_PASSWORD}).encode()
            req = urllib.request.Request(
                f"{CONTROL_PLANE}/auth/token", data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    self._token = json.loads(resp.read().decode()).get("access_token", "")
            except (urllib.error.URLError, ValueError) as exc:
                log(f"auth: login failed: {exc}")
                return False
            return bool(self._token)


AUTH = Auth()


def request(base: str, path: str, payload: dict | None = None, *, timeout: int = 30,
            retry_auth: bool = True, method: str | None = None):
    """JSON request (GET without a payload, POST with one, unless `method` says otherwise).
    Retries once after refreshing credentials on a 401."""
    headers = {"Content-Type": "application/json", **AUTH.header(), **_agent_header()}
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"{base}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        if exc.code == 401 and retry_auth and AUTH.refresh():
            return request(base, path, payload, timeout=timeout, retry_auth=False, method=method)
        body = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(body).get("detail", body)
        except (ValueError, AttributeError):
            detail = body
        raise RuntimeError(f"{path} -> {exc.code}: {_validation_text(detail)}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach {base}{path}: {exc.reason}") from None
    except (TimeoutError, OSError, http.client.HTTPException, ValueError) as exc:
        # A read that times out AFTER the connection is established raises socket.timeout
        # straight from getresponse(), not URLError -- as do a reset mid-read and a truncated
        # body. Every caller guards against RuntimeError, so transport failures are normalised
        # to that here. Letting one escape killed the whole runner: a single slow reply from a
        # busy control plane took the swarm down with it.
        raise RuntimeError(f"{path} failed: {type(exc).__name__}: {exc}") from None


def _validation_text(detail: Any) -> Any:
    """A FastAPI 422 body as the line an agent can act on. The raw pydantic list read
    "[{'type': 'less_than_equal', 'loc': ['body', 'horizon'], 'msg': 'Input should be less than
    or equal to 256', 'input': 360, ...}]" (forecast, 10-01) -- now "horizon: Input should be
    less than or equal to 256 (you sent 360)"."""
    if not (isinstance(detail, list) and detail and all(isinstance(d, dict) and "msg" in d for d in detail)):
        return detail
    lines = []
    for d in detail[:6]:
        where = ".".join(str(x) for x in d.get("loc") or [] if x not in ("body", "query")) or "the request"
        got = d.get("input")
        sent = (f" (you sent {(repr(got) if isinstance(got, str) else str(got))[:80]})"
                if d.get("type") != "missing" and got is not None and not isinstance(got, (dict, list)) else "")
        lines.append(f"{where}: {d['msg']}{sent}")
    return "; ".join(lines)


def q(value: str) -> str:
    return urllib.parse.quote(value, safe="")


EXTERNAL_SUFFIXES = ("@groq", "@openrouter")


def is_external(model: str) -> bool:
    """A hosted, pay-per-token model (see app/external.py)."""
    return model.endswith(EXTERNAL_SUFFIXES)


def permitted(project: dict, model: str) -> bool:
    """The console's rule (app/swarm_policy.py): "all loaded models" never includes a model
    that costs money -- an external model has to be ticked for the project."""
    allowed = project.get("models")
    if is_external(model):
        return allowed is not None and model in allowed
    return allowed is None or model in allowed


# =======================================================================================
# Model -> tier
# =======================================================================================
_TIER_ORDER = ("small", "mid", "hard")


def infer_tier(model: str) -> str:
    """Tier from the parameter count in the name. `auto` tasks go to any tier, so a wrong
    guess costs routing quality, never throughput."""
    sizes = [float(n) for n in re.findall(r"(\d+(?:\.\d+)?)\s*[bB]\b", model)]
    largest = max(sizes) if sizes else 0.0
    if largest >= 70:
        return "hard"
    if largest >= 25:
        return "mid"
    if largest > 0:
        return "small"
    return "mid"


# =======================================================================================
# The project's world: what an agent may use, and the tools that expose it
# =======================================================================================
_mcp_cache: dict[str, tuple[float, list[dict]]] = {}


def _mcp_tools(project_id: str) -> list[dict]:
    """The project's connector tools. Probing spawns every server, so it is cached."""
    hit = _mcp_cache.get(project_id)
    if hit and time.time() - hit[0] < MCP_TOOLS_TTL_S:
        return hit[1]
    try:
        tools = request(CONTROL_PLANE, f"/api/mcp/tools?project_id={q(project_id)}", timeout=120).get("tools", [])
    except RuntimeError as exc:
        log(f"mcp tools for {project_id}: {exc}")
        tools = []
    _mcp_cache[project_id] = (time.time(), tools)
    return tools


# Letters that look Latin but are not: Qwen wrote "decі_plot" (Cyrillic i) on 10-01 15:10, twice, and got
# "unknown tool". Tool names are ASCII, so a look-alike can only mean its Latin letter.
_LOOKALIKES = str.maketrans({
    "а": "a", "е": "e", "і": "i", "о": "o", "р": "p", "с": "c", "у": "y",
    "х": "x", "ѕ": "s", "ј": "j", "һ": "h", "ԁ": "d", "ɡ": "g", "ı": "i",
    "ο": "o", "α": "a", "ι": "i", "ρ": "p", "ν": "v", "Α": "A", "Β": "B",
    "Ε": "E", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Χ": "X", "А": "A", "В": "B", "Е": "E", "К": "K",
    "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
})


def _plain_name(name: str) -> str:
    """A tool name as the model meant it: width/compatibility forms folded (NFKC), look-alike Cyrillic
    and Greek letters mapped to Latin, stray whitespace dropped."""
    import unicodedata

    plain = unicodedata.normalize("NFKC", name).translate(_LOOKALIKES).strip()
    return plain if plain.isascii() else name


def _edits(a: str, b: str) -> int:
    """Edit distance counting a swap of neighbours as one edit (optimal string alignment)."""
    d = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


def _resolve_tool_name(name: str, names: set[str]) -> tuple[str, str | None]:
    """(the tool meant, a note saying so) for a name that is not offered but can only be one
    tool: the same letters without the separators ("deciplot") or one typo away ("decie_plot",
    "decpi_plot"), or two in a longer name ("decide_plot") -- 10 'unknown tool' calls on the
    board 09-29..10-01. Short names and names near two tools stay as they are: the 'unknown tool' reply then lists the
    candidates."""
    if name in names or len(name) < 6:
        return name, None

    def squash(s: str) -> str:
        return re.sub(r"[\s_\-.]", "", s).lower()

    hits = [n for n in names if squash(n) == squash(name)]
    if len(hits) != 1:
        # One typo, or two in a longer name -- and no second tool that close.
        near = [n for n in names if _edits(name.lower(), n.lower()) <= (2 if len(name) >= 8 else 1)]
        hits = near if len(near) == 1 else []
    if len(hits) != 1:
        return name, None
    return hits[0], f"there is no tool {name!r}; ran {hits[0]}, the only tool that name can mean"


def _coerce_scalar(v: Any, kind: str | None) -> Any:
    if isinstance(v, str) and kind in ("integer", "number"):
        try:
            f = float(v.strip())
            return int(f) if kind == "integer" and f.is_integer() else f
        except ValueError:
            return v
    if isinstance(v, str) and kind == "boolean" and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    return v


def _coerce_args(args: dict, props: dict) -> tuple[dict, list[str]]:
    """`args` read the way the tool's schema declares them, plus a note per argument that had
    to be decoded. Models send arrays as JSON strings -- deci_plot(timeframes='["5min",
    "15min"]', horizons='[1, 3, 6, 12, 0]') and gex__task_sample_rows(columns='["t", ...]'),
    10-01 -- and numbers as strings; read literally, a string horizons list was iterated a
    character at a time ("[" -> int() -> crash)."""
    import ast

    out, notes = dict(args), []
    for key, v in args.items():
        spec = props.get(key) if isinstance(props.get(key), dict) else {}
        if "type" not in spec:
            # A connector's optional parameter: {"anyOf": [{"type": "array", ...}, {"type": "null"}]}.
            spec = next((s for s in spec.get("anyOf") or spec.get("oneOf") or []
                         if isinstance(s, dict) and s.get("type") not in (None, "null")), spec)
        kind = spec.get("type")
        item_kind = (spec.get("items") or {}).get("type") if isinstance(spec.get("items"), dict) else None
        if kind == "array" and isinstance(v, str):
            s = v.strip()
            parsed: Any = None
            if s.startswith("["):
                for parse in (json.loads, ast.literal_eval):
                    try:
                        parsed = parse(s)
                        break
                    except (ValueError, SyntaxError):
                        continue
            if not isinstance(parsed, (list, tuple)):
                parsed = [p.strip().strip("'\"") for p in s.strip("[]").split(",") if p.strip().strip("'\"")]
            out[key] = [_coerce_scalar(x, item_kind) for x in parsed]
            notes.append(f"`{key}` arrived as a string; read it as the list {json.dumps(out[key])[:200]} "
                         "(send a JSON array next time)")
        elif kind == "array" and isinstance(v, list):
            out[key] = [_coerce_scalar(x, item_kind) for x in v]
        elif kind == "array" and isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = [v]
        elif kind == "object" and isinstance(v, str) and v.strip().startswith("{"):
            try:
                out[key] = json.loads(v)
            except ValueError:
                pass
        else:
            out[key] = _coerce_scalar(v, kind)
    return out, notes


def _module_name(raw: str) -> tuple[str, bool]:
    """(the library name `raw` means, whether it changed). The library takes lowercase
    identifiers of at most 48 characters; "signal_charm_Imb_adaptive_gate" and a 52-character
    name were refused outright (8 library_save calls, 09-21..09-29) though the intent is plain."""
    name = re.sub(r"[^0-9a-z_]+", "_", raw.strip().lower()).strip("_")
    if name[:1].isdigit():
        name = f"m_{name}"
    name = name[:48].rstrip("_")
    if not name or name in ("ft", "lib"):
        return raw, False                     # nothing to infer: the endpoint says what is wrong
    return name, name != raw


def _with_note(out: Any, note: str | None) -> Any:
    """A tool result carrying a note on how the call was read."""
    if not note:
        return out
    if isinstance(out, dict):
        prev = out.get("note")
        return {**out, "note": f"{note}. {prev}" if prev else note}
    return {"result": out, "note": note}


# SQL that ran off the end of the reply: Groq/Qwen3.8 sent "... CASE WHEN" (09-30) and
# '... "HistVol" FROM sql_exports_dbo_gexbar10s ORDER BY .' (09-26, 10 calls), which came back as
# sqlglot's "Required keyword: 'true' missing for If" -- nothing an agent can map to "resend it".
_SQL_DANGLING = re.compile(
    r"(?:\b(?:select|from|where|and|or|not|when|then|else|case|on|join|by|as|in|having|union|"
    r"over|partition|between|like|is|with|distinct)|[,(=<>+\-*/.|])\s*$", re.I)


def _sql_cut_off(sql: str) -> str | None:
    """The tail of a query that was evidently cut off mid-statement, else None."""
    s = str(sql or "").strip().rstrip(";").rstrip()
    if not s:
        return None
    # One pass over the text: literals become a placeholder, comments go; what is left open at
    # the end (a quote, a block comment) is itself proof of a cut.
    bare, state, i = [], "", 0
    while i < len(s):
        ch, two = s[i], s[i:i + 2]
        if state == "":
            if ch in "'\"":
                state = ch
            elif two == "--":
                state = "line"
            elif two == "/*":
                state, i = "block", i + 1
            else:
                bare.append(ch)
        elif state in "'\"" and ch == state:
            if s[i + 1:i + 2] == ch:
                i += 1                                    # '' / "" inside a literal
            else:
                state = ""
                bare.append("x")
        elif state == "line" and ch == "\n":
            state = ""
            bare.append(" ")
        elif state == "block" and two == "*/":
            state, i = "", i + 1
            bare.append(" ")
        i += 1
    text = "".join(bare).rstrip()
    if state in ("'", '"', "block") or text.count("(") > text.count(")") or (text and _SQL_DANGLING.search(text)):
        return s[-60:]
    return None


def _sql_refusal(args: dict) -> dict | None:
    tail = _sql_cut_off(args.get("sql", ""))
    if tail is None:
        return None
    return {"error": (f"your SQL arrived cut off -- it ends with {tail!r}, mid-statement (the reply ran out "
                      "of room). Nothing was run. Resend the COMPLETE query, shorter: fewer columns, "
                      "aggregate in SQL.")}


def _fn(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or []},
        },
    }


class ProjectWorld:
    """Everything one task may touch, gathered fresh when the task starts."""

    def __init__(self, project: dict, self_model: str, llms: list[str], forecasters: list[dict]):
        self.project = project
        self.pid = project["id"]
        self.self_model = self_model
        allowed = project.get("models")
        self.peers = [m for m in llms if m != self_model and permitted(project, m)]
        self.forecasters = [
            f for f in forecasters if allowed is None or f["model_id"] in allowed
        ]
        try:
            self.files = request(CONTROL_PLANE, f"/api/projects/{q(self.pid)}/data/catalog").get("files", [])
        except RuntimeError:
            self.files = []
        self.sql = project.get("sql") or None
        self.mcp = _mcp_tools(self.pid)
        try:
            self.research = [d for d in request(CONTROL_PLANE, f"/api/research/docs?project_id={q(self.pid)}")
                             .get("docs", []) if d.get("status") == "ready"]
        except RuntimeError:
            self.research = []

    # -- what the model is told ----------------------------------------------------------
    def briefing(self) -> str:
        lines = [f"Project: {self.project['name']}."]
        if self.files:
            shown = ", ".join(f"{f['view']} ({f['path']})" for f in self.files[:25])
            more = f" and {len(self.files) - 25} more" if len(self.files) > 25 else ""
            lines.append(f"Data files (query with query_data; each is also a view): {shown}{more}.")
        else:
            lines.append("The project data folder has no queryable files.")
        if self.sql:
            lines.append(
                f"SQL Server database {self.sql['database']} (read-only), tables: "
                f"{', '.join(self.sql['tables'])}. Query with query_sql (T-SQL, SELECT only)."
            )
        if self.forecasters:
            lines.append(
                "Time-series forecasters (call forecast): "
                + ", ".join(f["model_id"] for f in self.forecasters) + "."
            )
        if self.peers:
            lines.append(f"Other models you can delegate to with ask_model: {', '.join(self.peers)}.")
        if self.mcp:
            lines.append(f"Connector tools: {', '.join(t['function']['name'] for t in self.mcp[:30])}.")
        if self.research:
            lines.append(f"Research library ({len(self.research)} documents: "
                         + "; ".join(d["title"][:60] for d in self.research[:6])
                         + "): search it with research_search, read a document or chunk with research_get.")
        return "\n".join(lines)

    def tools(self) -> list[dict]:
        out = [
            _fn("list_data", "List the queryable data files (parquet/csv/json) in the project folder.", {}),
            _fn("describe_data", "Columns, row count and sample rows of one data file.",
                {"view": {"type": "string", "description": "view name or relative path"}}, ["view"]),
            _fn("query_data",
                "Run ONE read-only SQL SELECT (DuckDB dialect) over the project's data files. "
                "Refer to a file by its view name or as 'relative/path.parquet'. Aggregate in SQL "
                "rather than pulling raw rows: at most 200 rows come back.",
                {"sql": {"type": "string"}}, ["sql"]),
        ]
        if self.sql:
            out += [
                _fn("list_sql_tables", "List the SQL Server tables this project may read.", {}),
                _fn("describe_sql_table", "Columns and sample rows of one allowed SQL Server table.",
                    {"table": {"type": "string", "description": "schema.table"}}, ["table"]),
                _fn("query_sql",
                    "Run ONE read-only T-SQL SELECT against the allowed tables. Use TOP / "
                    "aggregation; at most 200 rows come back.",
                    {"sql": {"type": "string"}}, ["sql"]),
            ]
        if self.forecasters:
            out += [
                _fn("list_forecasters", "List loaded time-series forecasting models and what each suits.", {}),
                _fn("forecast",
                    "Forecast a numeric time series with a loaded time-series model. Returns the "
                    "median and quantiles for each future step. Give EITHER `sql` -- a query_data "
                    "SELECT returning the history as one numeric column, oldest first (e.g. SELECT "
                    "close FROM (SELECT ts, close FROM prices ORDER BY ts DESC LIMIT 512) ORDER BY ts) "
                    "-- OR `series`, the numbers themselves.",
                    {
                        "sql": {"type": "string", "description": "SELECT returning the history, oldest first"},
                        "series": {"type": "array", "items": {"type": "number"}},
                        "horizon": {"type": "integer", "description": "steps ahead"},
                        "model": {"type": "string", "description": "optional forecaster name"},
                    },
                    ["horizon"]),
            ]
        if self.research:
            out += [
                _fn("research_search",
                    "Search the research library -- reports and papers the operator added -- for findings, tables, "
                    "figures, code and extracted trading ideas. Hybrid semantic + keyword search: ask in words "
                    "('overnight drift when dealers are long gamma') or by identifier ('wall_pos', 'causal_rank'). "
                    "The numbers in a document are its authors', on their data: test before trusting them.",
                    {"query": {"type": "string"},
                     "kinds": {"type": "array", "items": {"type": "string", "enum": ["text", "table", "figure", "code", "idea"]},
                               "description": "optional filter"},
                     "k": {"type": "integer", "description": "results (default 8, max 20)"}},
                    ["query"]),
                _fn("research_get",
                    "Read from the research library: `chunk` = one chunk in full (a whole code listing, table or passage, "
                    "by the chunk id search returned); `doc` = a document's map -- outline, extracted ideas, code listings "
                    "with their import lines, tables and figures. A Python listing is importable in run_python and in "
                    "candidates as `from research.<doc package> import <module>`.",
                    {"chunk": {"type": "integer"}, "doc": {"type": "string", "description": "document id (d_...)"}}),
            ]
        if self.peers:
            out.append(
                _fn("ask_model",
                    "Delegate a self-contained question to another loaded model and get its answer. "
                    "It sees only what you send.",
                    {"model": {"type": "string", "enum": self.peers},
                     "prompt": {"type": "string"}},
                    ["model", "prompt"]))
        return out + self.mcp

    # -- execution -----------------------------------------------------------------------
    def call(self, name: str, args: dict) -> Any:
        pid = q(self.pid)
        if name == "list_data":
            return self.files or {"files": [], "note": "no queryable files in the data folder"}
        if name == "describe_data":
            return request(CONTROL_PLANE, f"/api/projects/{pid}/data/describe?view={q(str(args.get('view', '')))}")
        if name == "query_data":
            if _sql_refusal(args):
                return _sql_refusal(args)
            return request(CONTROL_PLANE, f"/api/projects/{pid}/data/query",
                           {"sql": args.get("sql", ""), "max_rows": int(args.get("_max_rows") or 200)}, timeout=180)
        if name == "list_sql_tables" and self.sql:
            return {"database": self.sql["database"], "tables": self.sql["tables"]}
        if name == "describe_sql_table" and self.sql:
            return request(CONTROL_PLANE, f"/api/projects/{pid}/sql/describe?table={q(str(args.get('table', '')))}")
        if name == "query_sql" and self.sql:
            if _sql_refusal(args):
                return _sql_refusal(args)
            return request(CONTROL_PLANE, f"/api/projects/{pid}/sql/query",
                           {"sql": args.get("sql", ""), "max_rows": 200}, timeout=180)
        if name == "list_forecasters":
            return [
                {"model": f["model_id"], "gpu": f["gpu"], **{k: (f.get("health") or {}).get(k) for k in
                 ("family", "context_length", "native_horizon", "channels", "probabilistic")}}
                for f in self.forecasters
            ]
        if name == "forecast":
            model = args.get("model") or None
            if model and model not in {f["model_id"] for f in self.forecasters}:
                return {"error": f"{model} is not loaded for this project",
                        "loaded": [f["model_id"] for f in self.forecasters]}
            series = args.get("series") or []
            if args.get("sql"):
                # The model names the history instead of pasting thousands of numbers.
                got = self.call("query_data", {"sql": args["sql"], "_max_rows": 500})
                rows = got.get("rows") if isinstance(got, dict) else None
                if not rows:
                    return {"error": "the sql returned no rows", "detail": got}
                col = next((i for i, v in enumerate(rows[0]) if isinstance(v, (int, float))), None)
                if col is None:
                    return {"error": "the sql returned no numeric column"}
                series = [float(r[col]) for r in rows if isinstance(r[col], (int, float))]
            return request(CONTROL_PLANE, "/api/ts/forecast", {
                "model": model or self.forecasters[0]["model_id"],
                "series": series,
                "horizon": int(args.get("horizon") or 12),
                "quantiles": [0.1, 0.5, 0.9],
            }, timeout=180)
        if name == "ask_model":
            model = args.get("model")
            if model not in self.peers:
                return {"error": f"{model} is not available", "available": self.peers}
            body = {"max_tokens": 4096, "stream": False,
                    "messages": [{"role": "user", "content": str(args.get("prompt", ""))}]}
            note = None
            try:
                r = request(CONTROL_PLANE, "/v1/chat/completions", {"model": model, **body}, timeout=GENERATION_TIMEOUT_S)
            except RuntimeError as exc:
                # Today's external budget is spent (10-01 14:09, qwen3.8@groq -> 429): the question is
                # still worth an answer, so a free model takes it instead of a failed call.
                free = [p for p in self.peers if not is_external(p)]
                if "spending limit" not in str(exc) or not is_external(model) or not free:
                    raise
                note = f"{model} is out of today's external budget -- {free[0]} answered instead"
                model = free[0]
                r = request(CONTROL_PLANE, "/v1/chat/completions", {"model": model, **body}, timeout=GENERATION_TIMEOUT_S)
            choice = (r.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            return {"model": model, "answer": msg.get("content") or "", **({"note": note} if note else {}),
                    "finish_reason": choice.get("finish_reason"), "usage": r.get("usage")}
        if name == "research_search":
            got = request(CONTROL_PLANE, "/api/research/search", {
                "query": str(args.get("query") or "")[:2000], "project_id": self.pid,
                "kinds": [k for k in args.get("kinds") or [] if k in ("text", "table", "figure", "code", "idea")] or None,
                "k": max(1, min(20, int(args.get("k") or 8)))}, timeout=120)
            return [{k: v for k, v in h.items() if k not in ("score", "cosine", "image_url") and v not in (None, "")}
                    for h in got.get("hits", [])] or {"hits": [], "note": "nothing matched -- try other words"}
        if name == "research_get":
            if args.get("chunk"):
                c = request(CONTROL_PLANE, f"/api/research/chunks/{int(args['chunk'])}")
                return {k: c.get(k) for k in ("id", "doc_id", "doc_title", "kind", "section", "page", "label", "text", "import")
                        if c.get(k) not in (None, "")}
            if args.get("doc"):
                return request(CONTROL_PLANE, f"/api/research/docs/{q(str(args['doc']))}?compact=true")
            return {"documents": [{"doc": d["id"], "title": d["title"], "ideas": d.get("ideas"),
                                   "code": (d.get("chunks") or {}).get("code", 0)} for d in self.research]}
        name = self._mcp_name(name)
        if "__" in name and any(t["function"]["name"] == name for t in self.mcp):
            args, note, refusal = self._only_task(name, args)
            if refusal:
                return refusal
            out = request(CONTROL_PLANE, f"/api/mcp/call?project_id={pid}",
                          {"tool": name, "arguments": args}, timeout=300)
            # A connector's in-band failure is a failed call: without "error" the repeat guard
            # never saw it, and Muse-Glimmer sent task_describe({}) 13 times in one iteration.
            if isinstance(out, dict) and out.get("is_error"):
                return _with_note({"error": str(out.get("content") or "the connector reported an error")[:4000]}, note)
            return _with_note(out, note)
        # A misspelt tool ("decipic" for deci_plot, 2026-10-01) gets the closest names, not a dead end.
        import difflib

        tools = self.tools()
        names = [t["function"]["name"] for t in tools]
        # Arguments that fit exactly one tool's parameters say which tool was meant: Qwen3-0.6B
        # called "regime_detector" with {"name": "regime", "version": 1} -- library_get's (09-30).
        keys = set(args) if isinstance(args, dict) else set()
        shaped = []
        for t in tools:
            p = t["function"].get("parameters") or {}
            props, req = set(p.get("properties") or {}), set(p.get("required") or [])
            if keys and req <= keys <= props:
                shaped.append(t["function"]["name"])
        near = list(dict.fromkeys(shaped[:2] + difflib.get_close_matches(name, names, n=3, cutoff=0.4)))[:4]
        return {"error": f"unknown tool {name!r}" + (f" -- did you mean {', '.join(near)}?" if near else
                                                     f"; the tools are: {', '.join(names)}")}

    def _only_task(self, name: str, args: dict) -> tuple[dict, str | None, dict | None]:
        """(args, note, refusal) for a connector tool that requires `task` and was called without
        it (gex__task_describe({}) in a dataset objective, 09-30 -- "task: Field required"). When the
        server's task_list names exactly one task, that is the one meant; with several, the agent
        is told which exist instead of getting pydantic's error."""
        tool = next((t["function"] for t in self.mcp if t["function"]["name"] == name), {})
        params = tool.get("parameters") or {}
        if "task" not in (params.get("required") or []) or str(args.get("task") or "").strip():
            return args, None, None
        server = name.split("__", 1)[0]
        lister = f"{server}__task_list"
        if lister == name or not any(t["function"]["name"] == lister for t in self.mcp):
            return args, None, None
        cache = self.__dict__.setdefault("_task_names", {})
        if server not in cache:
            names: list[str] = []
            try:
                got = request(CONTROL_PLANE, f"/api/mcp/call?project_id={q(self.pid)}",
                              {"tool": lister, "arguments": {}}, timeout=120)
                body = got.get("structured") if isinstance(got.get("structured"), dict) else json.loads(got.get("content") or "{}")
                names = [str(t["name"]) for t in (body.get("tasks") or []) if isinstance(t, dict) and t.get("name")]
            except (RuntimeError, ValueError, TypeError, AttributeError):
                names = []
            cache[server] = names
        names = cache[server]
        if len(names) == 1:
            return ({**args, "task": names[0]},
                    f"`task` was missing; {server} has one task, {names[0]!r}, so it was used", None)
        if names:
            return args, None, {"error": f"{name} needs `task` -- one of: {', '.join(names)} (see {lister})"}
        return args, None, None

    def _mcp_name(self, name: str) -> str:
        """A connector tool called without its server prefix ("task_describe" for gex__task_describe,
        Qwen 10-01 12:53) is that tool when exactly one server has it and no tool of ours has the name."""
        if "__" in name or any(t["function"]["name"] == name for t in self.tools()):
            return name
        hits = [t["function"]["name"] for t in self.mcp if t["function"]["name"].endswith(f"__{name}")]
        return hits[0] if len(hits) == 1 else name


class ObjectiveWorld(ProjectWorld):
    """The project's tools, re-pointed for an objective iteration.

    Data access is IN-SAMPLE only (the holdout must stay unseen or ranking on it means
    nothing): query_data goes through the objective's in-sample views, run_python mounts the
    truncated data, and query_sql is withdrawn while a split is set (the SQL Server table has
    every row; its parquet export is covered by the views).
    """

    def __init__(self, project: dict, self_model: str, llms: list[str], forecasters: list[dict],
                 objective: dict, on_submit) -> None:
        super().__init__(project, self_model, llms, forecasters)
        self.objective = objective
        self.oid = objective["id"]
        self.split = objective.get("split_date")
        self.on_submit = on_submit
        self.experiments = 0
        # Last run_python code whose sandbox call reported ok=True. If the agent burns
        # every experiment we can hand this back in the refusal so submit_candidate has
        # something to submit -- otherwise the model tries to reconstruct it from context
        # and often just stops.
        self.best_code: str | None = None
        self.saved: list[str] = []   # library modules saved this iteration
        self.sent: list[dict] = []   # team_post messages this iteration
        self.inbox_seqs: set[int] = set()   # the MESSAGES TO YOU of this iteration's brief
        self.agent_name: str | None = None  # which of the agents on this model is posting
        self.author_id: str | None = None
        self.feature_views = set()
        # fix(tool, code, crash) -> corrected code or None: the worker's own model, asked to repair
        # a script that crashed (see _auto_repair). None (the default) leaves failures as they are.
        self.repairer = None
        if self.split:
            self.sql = None

    def tools(self) -> list[dict]:
        base = [t for t in super().tools() if t["function"]["name"] not in ("query_data", "forecast")]
        split_note = f" Sees only rows BEFORE {self.split} (the holdout is hidden)." if self.split else ""
        base.insert(2, _fn(
            "query_data",
            "Run ONE read-only SQL SELECT (DuckDB) over the project's datasets, by VIEW NAME." + split_note +
            " Aggregate in SQL; at most 200 rows come back.",
            {"sql": {"type": "string"}}, ["sql"]))
        base += [
            _fn("run_python",
                "Run an experimental Python script in the offline sandbox (polars, numpy, scipy -- use polars, "
                "not pandas: it is several times faster on these rows). "
                + ("Load the task's rows with `import ft; rows = ft.rows_pl()` (ft.task() describes them)."
                   if _is_task(self.objective) else "Load data with `import ft; df = ft.load_pl('<view>')`.")
                + split_note + " Print what you want to see; nothing is scored. ft.quick_score(positions, rows) "
                "gives an approximate score (both halves), ft.sweep(...) scores a grid of variants in one run and "
                "ft.direction_scan(rows) finds fields that tell the rest of the day's direction.",
                {"code": {"type": "string"}}, ["code"]),
            _fn("get_candidate",
                "Full code and in-sample results of an earlier candidate, by its number (seq) or id "
                "(a mentor idea number is not a candidate number).",
                {"candidate": {"type": "string"}}, ["candidate"]),
            *([_fn("trade_review",
                   "An earlier candidate's TRADES, in-sample: how many were big winners, big losers and scratch, "
                   "what the market looked like at the entry of its big winners vs the rest (per side, per field), "
                   "the time of day, and its best and worst trades. Use it to find the filter that keeps the "
                   "winners and drops the rest.",
                   {"candidate": {"type": "string", "description": "its number (seq) or id"}}, ["candidate"])]
              if _is_task(self.objective) else []),
            *([
                _fn("forecast",
                    "Look at one forecast of a column's most recent IN-SAMPLE values from a loaded "
                    "time-series model (median and 10/90% quantiles per step). For exploring only -- a "
                    "strategy uses forecasts through forecast_feature.",
                    {"column": {"type": "string"}, "dataset": {"type": "string", "description": "view name; default: the objective's dataset"},
                     "horizon": {"type": "integer", "description": "steps ahead, 1..256 (default 12)"},
                     "context": {"type": "integer", "description": "history points (default 512)"},
                     "model": {"type": "string"}},
                    ["column"]),
                _fn("forecast_feature",
                    "Turn a time-series model into a FEATURE your script can use. Runs the forecaster over "
                    "a series CAUSALLY (the forecast stored at bar t uses only bars up to and including t) "
                    "every `every` bars and saves it as a dataset your script loads with ft.load('fc_<name>'), "
                    "joined with pd.merge_asof(direction='backward'). A series is a column (GEX, IntrVol, "
                    "Gamma_NormalizedValue...) or an expression over columns (GEX / Pinning_TotalAbsGex, "
                    "Charm_NormalizedValue * IntrVol); give several in `columns` to forecast each (the model is "
                    "univariate: each is forecast on its own, stored side by side with a series prefix). "
                    "Columns: t, last, fc_median, fc_q10, fc_q90, fc_path_mean, fc_change. Returns each "
                    "series' in-sample SKILL vs a no-change forecast and its direction accuracy -- only use "
                    "series the model can actually forecast. Skill differs by horizon: build the same series at "
                    "a few horizons (e.g. 3, 6, 12, 30) and compare before choosing one or combining them. Takes "
                    "up to a few minutes; cached after.",
                    {"column": {"type": "string", "description": "one series: a column or an expression"},
                     "columns": {"type": "array", "items": {"type": "string"}, "description": "several series"},
                     "dataset": {"type": "string", "description": "view name; default: the objective's dataset"},
                     "horizon": {"type": "integer", "description": "bars ahead, 1..256 (default 12)"},
                     "every": {"type": "integer", "description": "bars between forecasts (0 = automatic, ~20k forecasts)"},
                     "context": {"type": "integer", "description": "history bars per forecast (default 512)"},
                     "model": {"type": "string"},
                     "bar": {"type": "string", "description": "bar size to resample to first, e.g. 1min, 5min, 30s (Kronos: candle size)"},
                     "covariates": {"type": "array", "items": {"type": "string"},
                                    "description": "Chronos-2 only: other columns the forecast READS as inputs (e.g. GEX, Pressure_Below, Imb_OINet_D0)"},
                     "calendar": {"type": "boolean", "description": "Chronos-2 only: add time-of-day and weekday as inputs known ahead"},
                     "name": {"type": "string", "description": "short name; the view becomes fc_<name>"}},
                    []),
            ] if self.forecasters else []),
            _fn("team_board",
                "Read what the team just did: recent posts from your teammates -- plans in #planning, "
                "results and new bests in #results, failures in #errors. Read it before planning so you "
                "build on their work and do not repeat their failures.",
                {"channel": {"type": "string", "enum": ["all", "planning", "results", "errors", "general"]},
                 "n": {"type": "integer", "description": "how many recent posts (default 25)"}}),
            _fn("team_post",
                "Talk to the team on the message board. Announce your plan in #planning BEFORE the expensive "
                "work (hypothesis, timeframe, regime, fields, modules) so teammates pick different directions; "
                "share a finding in #results; or message ONE teammate directly with `to` -- ask a question about "
                "their module or candidate, propose splitting the work, hand over a finding they can use. Answer "
                "messages addressed to you with `reply_to` set to the message number.",
                {"text": {"type": "string"},
                 "channel": {"type": "string", "enum": ["planning", "results", "team"]},
                 "to": {"type": "string", "description": "teammate model name, or 'all' (default)"},
                 "reply_to": {"type": "integer", "description": "the number of the message you are answering or "
                              "acting on (from MESSAGES TO YOU); cite further ones in the text as #<number>"}},
                ["text"]),
            *([_fn("field_scan",
                   "Screen EVERY field of the dataset (all the greeks, walls, imbalances, surface, IV...) against "
                   "the forward return on in-sample data: rank correlation (IC) of each field's level and of its "
                   "change over the horizon, optionally within each regime of a regime module. Returns the top "
                   "fields; the full scan is stored for the team.",
                   {"horizon": {"type": "integer", "description": "bars ahead (10 s bars; default 30 = 5 min)"},
                    "regime": {"type": "string", "description": "optional regime module to split the IC by"},
                    "columns": {"type": "array", "items": {"type": "string"}, "description": "optional subset"}})]
              if _analysis_ok(self.objective) else []),
            *([_fn("deci_plot",
                   "Decile study of ONE signal on in-sample data: the mean forward return (bps), hit rate and t "
                   "in each of the signal's 10 deciles, on 10s/20s/30s/1min/5min bars at 1/3/6/12 bars ahead AND to "
                   "the session CLOSE (horizon 0 -- the horizon of a strategy that trades a few times a day and holds), "
                   "with monotonicity (Spearman), the top-minus-bottom spread and its t, and whether it holds in "
                   "each of 3 sub-periods. Deciles use ROLLING edges from past sessions only (no look-ahead). "
                   "`condition` limits the study to a regime -- either a boolean expression "
                   "(e.g. -GEX > 0 for dealers short gamma, or GEX < 0 AND IntrVol > 0.2) or a numeric one "
                   "(e.g. -GEX) kept where it is > 0. Run the signal both with and without a condition to "
                   "see whether a regime changes it. "
                   "Stored and cached: a study already run comes back at once. Call with no signal to list the "
                   "studies the team already has -- check them before running a new one.",
                   {"signal": {"type": "string", "description": "a column (GEX), an expression (GEX / Pinning_TotalAbsGex), "
                                                                "or a forecast feature column fc_<name>:fc_change"},
                    "condition": {"type": "string", "description": "optional regime: a boolean (-GEX > 0, GEX < 0 AND IntrVol > 0.2) or a numeric expression kept where >0 (-GEX)"},
                    "timeframes": {"type": "array", "items": {"type": "string"}, "description": "default 10s,20s,30s,1min,5min"},
                    "horizons": {"type": "array", "items": {"type": "integer"},
                                 "description": "bars of the timeframe ahead, 0 = to the session close; default 1,3,6,12,0"},
                    "window_days": {"type": "integer", "description": "past sessions the decile edges come from (default 20)"}})]
              if _analysis_ok(self.objective) else []),
            # Combining verified candidates (ensembles): daily-return objectives only.
            *([_fn("correlations",
                   "IN-SAMPLE daily-return correlation matrix among candidates (the given numbers, or the top "
                   "`top` ranked ones), each one's in-sample Sharpe and whether it may join an ensemble, plus "
                   "suggested low-correlation sets of 2-4 with their average |rho| and what an equal-weight blend "
                   "did in-sample. Call this before combine_candidates.",
                   {"seqs": {"type": "array", "items": {"type": "integer"}, "description": "candidate numbers (optional)"},
                    "top": {"type": "integer", "description": "how many top-ranked candidates when no seqs (default 12)"}}),
               _fn("combine_candidates",
                   "Combine 2-8 VERIFIED candidates (scored, look-ahead pass, not disqualified, not ensembles) into "
                   "one ENSEMBLE candidate: a portfolio whose daily return is the weighted sum of the members' net "
                   "daily returns (no code runs; weights use only returns BEFORE each day). weighting 'equal' = 1/n; "
                   "'inverse_vol' = in proportion to 1/volatility of each member over the last lookback_days. It is "
                   "scored and ranked like any candidate; returns its in-sample result.",
                   {"members": {"type": "array", "items": {"type": "integer"}, "description": "candidate numbers"},
                    "weighting": {"type": "string", "enum": ["equal", "inverse_vol"]},
                    "lookback_days": {"type": "integer", "description": "inverse_vol window in trading days (default 20, 5..250)"},
                    "rationale": {"type": "string", "description": "why these members: low correlation, different signals/regimes"}},
                   ["members", "rationale"])]
              if self.objective["metric"].get("kind") in ("sharpe", "sortino", "calmar", "total_return", "cagr",
                                                          "max_drawdown") else []),
            *([_fn("explore_forecast_inputs",
                   "Chronos-2: systematically find which INPUT columns make a forecast of `target` better, in-sample. "
                   "Runs in the background: target alone, then each candidate input alone, then greedy forward "
                   "selection that only adds an input beating the current best beyond +/- 2 SE, then a leave-one-out "
                   "prune. Every combination is stored, so none is ever forecast twice; results appear in your "
                   "brief (FORECAST INPUTS) and on the console. Returns what is already known for the target now.",
                   {"target": {"type": "string", "description": "column to forecast, e.g. Close, IntrVol, GEX"},
                    "candidates": {"type": "array", "items": {"type": "string"},
                                   "description": "inputs to consider (default: field-scan leaders and monotone decile signals)"},
                    "horizon": {"type": "integer", "description": "rows ahead (default 30)"},
                    "bar": {"type": "string", "description": "optional bar size to resample to first, e.g. 1min"},
                    "budget": {"type": "integer", "description": "new combinations to forecast (default 30)"}},
                   ["target"])]
              if any((f.get("health") or {}).get("supports_covariates") or "chronos-2" in str(f.get("model_id", "")).lower()
                     for f in self.forecasters) else []),
            _fn("library_list",
                "The project's code library: reusable modules (regime detectors, signals, risk rules, "
                "utils) with evidence -- how many candidates used each, how they scored, look-ahead "
                "failures -- and comment counts.", {}),
            _fn("library_get",
                "One library module: code, description, versions, comments (works/broken/note) and the "
                "candidates that used it. For a regime module, also its latest regime map.",
                {"name": {"type": "string"}, "version": {"type": "integer"}}, ["name"]),
            _fn("library_save",
                "Save a reusable module to the library (a new module, or a new version of one). It is "
                "smoke-tested in the sandbox first (imported, then your test_code runs on in-sample data); "
                "a module that fails is not saved. Contracts: kind='regime' defines detect(df) -> one label "
                "per row; kind='signal' defines signal(df) -> one position per row in [-1, 1]; both CAUSAL "
                "(row t uses rows <= t only) and aligned to df's rows (sorted by time). A regime or signal "
                "module is also causality-tested: its output is recomputed with the data cut at many times "
                "(including mid-bar), and if any earlier row changes the save is REFUSED with the row, the "
                "cut and the likely cause. After resampling, use ft.resample/ft.align or shift by one full "
                "bar -- never attach a bar's aggregate to rows inside that same bar. A quarantined module "
                "only becomes usable again when a new version passes. Import in scripts "
                "with `from lib import <name>`.",
                {"name": {"type": "string", "description": "lowercase identifier"},
                 "kind": {"type": "string", "enum": ["regime", "signal", "risk", "util"]},
                 "description": {"type": "string", "description": "what it does, inputs, when to use it"},
                 "code": {"type": "string"},
                 "test_code": {"type": "string", "description": "optional script run after `from lib import <name>`; use ft.load"},
                 "note": {"type": "string", "description": "what changed in this version"}},
                ["name", "kind", "code"]),
            _fn("library_comment",
                "Comment on a library module so the team learns: verdict 'works' or 'broken' (with the "
                "evidence -- candidate number, numbers, the failure) or 'note'.",
                {"name": {"type": "string"}, "verdict": {"type": "string", "enum": ["works", "broken", "note"]},
                 "text": {"type": "string"}, "candidate": {"type": "string", "description": "candidate number as proof"}},
                ["name", "verdict", "text"]),
            *([_fn("regime_map",
                   "Measure every signal module (plus buy-and-hold) inside every regime a regime module "
                   "detects, on in-sample data: share of bars, Sharpe, mean bps per bar and hit rate per "
                   "(regime, signal). Stored for the team. Use it to decide which library functions to "
                   "route each regime to with ft.route.",
                   {"regime": {"type": "string", "description": "regime module name"},
                    "signals": {"type": "array", "items": {"type": "string"},
                                "description": "signal modules to test (default: all)"}},
                   ["regime"])] if _analysis_ok(self.objective) else []),
            *([_fn("regime_lab",
                   "REGIME LAB: which VERIFIED candidate works in which market regime, and a router that trades each "
                   "regime with the one that works there. Regimes are a crossing of fields (default GEX x IntrVol "
                   "terciles: each field smoothed, then ranked causally against its own trailing sessions -- exactly "
                   "ft.regime_grid) or a library regime module. Members are verified candidates (default: the best "
                   "ranked that are not near-copies). Every member's net P&L is split by regime, in-sample and per "
                   "in-sample half. Routes start from the best single member everywhere; a regime switches to another "
                   "member only when it beats that one by 0.5 Sharpe in BOTH halves (flat only when the baseline "
                   "loses in both), and the router is re-measured exactly WITH its switching costs. Returns the table, the router's in-sample result vs the best single "
                   "member, and a ready SCRIPT -- submit it (or edit the routes) with submit_candidate. Slow the first "
                   "time (each member is replayed once), fast after. Try different fields: the regime that separates "
                   "your strategies best is the finding.",
                   {"fields": {"type": "array", "items": {"type": "string"},
                               "description": "1-2 numeric columns to split by, e.g. ['GEX', 'IntrVol'] or ['Pressure_Total', 'HistVol']"},
                    "buckets": {"type": "integer", "description": "buckets per field: 2 (low/high), 3 (default) up to 5"},
                    "module": {"type": "string", "description": "a library regime module instead of fields"},
                    "members": {"type": "array", "items": {"type": "integer"}, "description": "candidate numbers (2-8; default automatic)"},
                    "smooth": {"type": "integer", "description": "trailing-mean bars before ranking (default 360 = 1 h of 10 s bars)"},
                    "window_days": {"type": "number", "description": "sessions each field is ranked against (default 20)"}})]
              if self.objective["metric"].get("price_column") and self.objective.get("dataset") else []),
            _fn("submit_candidate",
                "Submit your candidate for scoring. Returns the evaluation (in-sample metrics, "
                "look-ahead verdict, rank). Call once your script is complete.",
                {"code": {"type": "string", "description": "the complete Python script"},
                 "rationale": {"type": "string", "description": "the hypothesis: what you changed or tried, and why it should generalise"},
                 "answer": {"type": "string", "description": "for judged objectives: the answer text"},
                 "idea": {"type": "integer", "description": "the number of the mentor idea this candidate tests, if any"},
                 "parent": {"type": "string", "description": "the candidate number (seq) your script starts from or "
                            "builds on -- a teammate's or your own -- if any; the team's lineage records it"}},
                ["rationale"]),
        ]
        if _is_task(self.objective):
            # The task server's rows are the only data a task candidate can read: tools over the
            # project's datasets, forecasts and dataset studies would mislead, so they go.
            base = [t for t in base if t["function"]["name"] not in TASK_HIDDEN_TOOLS]
        return base

    def call(self, name: str, args: dict) -> Any:
        oid = q(self.oid)
        if name == "query_data":
            if _sql_refusal(args):
                return _sql_refusal(args)
            return request(CONTROL_PLANE, f"/api/objectives/{oid}/data/query",
                           {"sql": args.get("sql", ""), "max_rows": 200}, timeout=180)
        if name == "get_candidate":
            hit, miss = self._candidate(args.get("candidate"))
            if hit is None:
                return miss
            full = request(CONTROL_PLANE, f"/api/objectives/{oid}/candidates/{q(hit['id'])}")
            return {"seq": full["seq"], "model": full["model"], "rationale": full["rationale"],
                    "code": full["code"], "status": full["status"],
                    "in_sample": (full.get("metrics") or {}).get("in_sample"),
                    "lookahead": full.get("lookahead"), "problem": full.get("score_note")}
        if name == "trade_review":
            hit, miss = self._candidate(args.get("candidate"))
            if hit is None:
                return miss
            return request(CONTROL_PLANE, f"/api/objectives/{oid}/candidates/{q(hit['id'])}/trade-review", timeout=180)
        if name == "forecast":
            return request(CONTROL_PLANE, f"/api/objectives/{oid}/forecast", {
                k: v for k, v in {"column": args.get("column"), "dataset": args.get("dataset") or None,
                                  "horizon": int(args.get("horizon") or 12), "context": int(args.get("context") or 512),
                                  "model": args.get("model") or None}.items() if v is not None}, timeout=300)
        if name == "forecast_feature":
            body = {"column": args.get("column") or None, "columns": args.get("columns") or None,
                    "dataset": args.get("dataset") or None,
                    "horizon": int(args.get("horizon") or 12), "every": int(args.get("every") or 0),
                    "context": int(args.get("context") or 512), "model": args.get("model") or None,
                    "name": args.get("name") or None, "bar": args.get("bar") or None,
                    "covariates": args.get("covariates") or None, "calendar": bool(args.get("calendar"))}
            return request(CONTROL_PLANE, f"/api/objectives/{oid}/features",
                           {k: v for k, v in body.items() if v is not None}, timeout=1800)
        lib = f"/api/projects/{q(self.pid)}/library"
        if name == "run_python":
            self.experiments += 1
            if self.experiments > MAX_EXPERIMENTS:
                refusal: dict[str, Any] = {"error": (
                    f"experiment budget used ({MAX_EXPERIMENTS} runs this iteration). Turn what works "
                    "into a library module with library_save (with a test) and call submit_candidate "
                    "with a script that imports it.")}
                # Hand back the last code that actually ran so the agent has something concrete
                # to pass to submit_candidate -- otherwise it stalls trying to remember it.
                if self.best_code:
                    refusal["best_working_code"] = self.best_code
                    refusal["hint"] = ("Pass best_working_code above (or a small change to it) to "
                                       "submit_candidate now; further run_python calls will be refused.")
                return refusal
            left = MAX_EXPERIMENTS - self.experiments

            def run(source: str) -> dict:
                return request(CONTROL_PLANE, f"/api/objectives/{oid}/python",
                               {"code": source, "timeout_s": 180}, timeout=400 + FORECAST_BUILD_ALLOWANCE_S)

            try:
                out = run(str(args.get("code", "")))
            except RuntimeError:
                # Platform failure (network / control-plane / harness). The agent never got a
                # chance to prove or disprove anything, so refund the experiment.
                self.experiments -= 1
                raise
            # A crash goes back to the model for a fix; its reruns are part of this one experiment.
            out = _auto_repair("run_python", str(args.get("code", "")), out, self.repairer, run,
                               who=self.self_model)
            if out.get("ok") is True:
                code = str(out.get("code_ran") or args.get("code", "")).strip()
                if code:
                    self.best_code = code
            reply: dict[str, Any] = {**out, "experiments_left": left}
            # Warn the agent BEFORE the wall so it can save + submit. Silent budget-exhaust
            # was the pattern behind bug #11: an agent that thought it had one more run left
            # and stopped when the refusal came without any pointer to what to submit.
            if left == 0:
                # The next turn is a submit turn (Worker.iterate's focus): only submit_candidate
                # -- and library_save in a BUILD iteration that has not saved its module yet.
                reply["note"] = ("last run_python this iteration -- your next call is submit_candidate with your "
                                 "complete script (a BUILD iteration with nothing saved yet: library_save the "
                                 "module first). Any further run_python will be refused.")
            elif left == 1:
                reply["note"] = ("one run_python left after this -- wrap up: save reusable pieces "
                                 "with library_save and prepare to call submit_candidate.")
            return reply
        if name == "describe_data" and str(args.get("view", "")).startswith("fc_"):
            view = str(args["view"])
            got = request(CONTROL_PLANE, f"/api/objectives/{oid}/data/query",
                          {"sql": f'SELECT * FROM "{view}" LIMIT 5', "max_rows": 5})
            n = request(CONTROL_PLANE, f"/api/objectives/{oid}/data/query",
                        {"sql": f'SELECT count(*) FROM "{view}"', "max_rows": 1})
            return {"view": view, "kind": "forecast feature", "rows_in_sample": (n.get("rows") or [[None]])[0][0],
                    "columns": got.get("columns"), "sample": got.get("rows")}
        if name == "team_board":
            ch = args.get("channel") or "all"
            n = max(5, min(60, int(args.get("n") or 25)))
            path = f"/mb/messages?project_id={q(self.pid)}&tail={n * (3 if ch == 'all' else 1)}"
            if ch != "all":
                path += f"&channel={q(ch)}"
            entries = request(BOARD, path).get("entries", [])
            # The tool-call trace ("-> query_data: ...") is noise here; keep what people SAID.
            keep = [e for e in entries if not str(e.get("content", "")).startswith("→ ")][-n:]
            return [{"when": time.strftime("%H:%M", time.localtime(e["ts"])), "who": e["author"],
                     "channel": e["channel"], "kind": e["kind"], "text": str(e["content"])[:700]} for e in keep]
        if name == "team_post":
            to = str(args.get("to") or "all").strip()
            peers = {p.split("/")[-1].lower(): p for p in self.peers}
            if to.lower() not in ("all", "") and to not in self.peers:
                to = peers.get(to.split("/")[-1].lower(), to)
            ch = args.get("channel") if args.get("channel") in ("planning", "results", "team") else (
                "team" if to not in ("all", "") else "planning")
            text = str(args.get("text", ""))[:4000]
            meta = {"objective_id": self.oid, "team": True, **({"agent": self.agent_name} if self.agent_name else {})}
            if to not in ("all", ""):
                meta["to"] = to
            reply_to = _int_or_none(args.get("reply_to"))
            # The inbox messages this post acts on: its reply_to, and any it cites as #<number>.
            # A plan that says "acting on #75840" answers it (10-01: plans acted on the mentor's
            # notes, never with reply_to, and every one counted as unanswered).
            answers = sorted({n for n in [reply_to, *_cited(text)] if n is not None and n in self.inbox_seqs})
            if reply_to is None and answers:
                reply_to = answers[0]
            if reply_to is not None:
                meta["reply_to"] = reply_to
            if answers:
                meta["answers"] = answers
            doc = request(BOARD, "/mb/messages", {
                "project_id": self.pid, "channel": ch, "author": self.self_model, "kind": "chat",
                **({"author_id": self.author_id} if self.author_id else {}),
                "content": (f"@{to.split('/')[-1]} " if meta.get("to") else "") + text, "meta": meta,
                **({"reply_to": meta["reply_to"]} if meta.get("reply_to") else {})})
            self.sent.append({"to": meta.get("to", "all"), "channel": ch, "reply_to": meta.get("reply_to"),
                              "text": text[:160], **({"answers": answers} if answers else {})})
            return {"posted": ch, "to": meta.get("to", "all"), "seq": (doc or {}).get("seq"),
                    **({"answers": [f"#{n}" for n in answers]} if answers else {})}
        if name == "field_scan":
            return request(CONTROL_PLANE, f"/api/objectives/{oid}/field-scan", {
                "horizon": int(args.get("horizon") or 30), "regime": args.get("regime") or None,
                "columns": args.get("columns") or None, "author": self.self_model}, timeout=600)
        if name == "deci_plot":
            if not str(args.get("signal") or "").strip():
                return request(CONTROL_PLANE, f"/api/objectives/{oid}/deci-plots?compact=true")
            body = {"signal": str(args["signal"]), "condition": str(args.get("condition") or "").strip() or None,
                    "timeframes": args.get("timeframes") or None,
                    "horizons": [0 if str(h).strip().lower() in ("close", "to_close", "eod") else int(h)
                                 for h in args.get("horizons") or []] or None,
                    "window_days": int(args.get("window_days") or 20), "author": self.self_model, "compact": True}
            return request(CONTROL_PLANE, f"/api/objectives/{oid}/deci-plots",
                           {k: v for k, v in body.items() if v is not None}, timeout=700)
        if name == "correlations":
            # In-sample only: the endpoint computes everything from dates before the split.
            seqs = ",".join(str(int(str(s).lstrip("#"))) for s in args.get("seqs") or [] if str(s).lstrip("#").isdigit())
            path = f"/api/objectives/{oid}/correlations?top={max(2, min(40, int(args.get('top') or 12)))}"
            return request(CONTROL_PLANE, path + (f"&seqs={q(seqs)}" if seqs else ""), timeout=120)
        if name == "combine_candidates":
            return request(CONTROL_PLANE, f"/api/objectives/{oid}/ensembles", {
                "members": [str(m) for m in args.get("members") or []],
                "weighting": str(args.get("weighting") or "equal"),
                "lookback_days": int(args.get("lookback_days") or 20),
                "rationale": str(args.get("rationale") or "")[:4000], "model": self.self_model}, timeout=120)
        if name == "explore_forecast_inputs":
            target = str(args.get("target") or "").strip()
            job = request(CONTROL_PLANE, "/api/tslab/explore", {
                "project_id": self.pid, "objective_id": self.oid, "target": target,
                "inputs": [str(c) for c in args.get("candidates") or []][:40], "horizon": int(args.get("horizon") or 30),
                "bar": args.get("bar") or None, "budget": max(3, min(120, int(args.get("budget") or 30))),
                "author": self.self_model})
            known = request(CONTROL_PLANE, f"/api/tslab/combos?project_id={q(self.pid)}&objective_id={oid}&target={q(target)}")
            return {"job": {k: job.get(k) for k in ("id", "phase", "total", "already_running")},
                    "known_for_target": [{k: g.get(k) for k in ("horizon", "bar", "model", "tested", "helpful", "hurts", "useless")}
                                         | {"baseline_skill": (g.get("baseline") or {}).get("skill"),
                                            "best": {k: (g.get("best") or {}).get(k) for k in ("inputs", "skill", "gain", "se")}
                                            if g.get("best") else None}
                                         for g in known.get("groups", [])[:4]],
                    "note": "runs in the background; its findings appear in your brief under FORECAST INPUTS"}
        if name == "library_list":
            mods = request(CONTROL_PLANE, lib).get("modules", [])
            return [{"name": m["name"], "kind": m["kind"], "version": m["version"], "status": m["status"],
                     "description": m["description"], "evidence": m["evidence"], "comments": m["comments"],
                     # A quarantined module is still listed, with the reason, so the team learns
                     # from it instead of rediscovering the defect.
                     **({"WARNING": f"DO NOT USE -- {m['warning']}"} if m.get("warning") else {})}
                    for m in mods] or {"modules": [], "note": "the library is empty -- save the first module"}
        if name == "library_get":
            path = f"{lib}/{q(str(args.get('name', '')))}"
            if args.get("version"):
                path += f"?version={int(args['version'])}"
            m = request(CONTROL_PLANE, path)
            ev = m.get("evidence") or {}
            return {"name": m["name"], "kind": m["kind"], "version": m["shown_version"], "latest": m["version"],
                    "status": m["status"], "description": m["description"], "code": m["code"],
                    **({"WARNING": f"DO NOT USE -- {m['warning']}"} if m.get("warning") else {}),
                    "test_output": (m.get("test_output") or "")[-800:],
                    "comments": [{k: c[k] for k in ("verdict", "author", "text", "candidate_id", "version")}
                                 for c in m.get("comments", [])[:15]],
                    "evidence": {k: v for k, v in ev.items() if k != "best_holdout"},
                    "regime_map": (m.get("regime_map") or {}).get("result") and _compact_regime(m["regime_map"]["result"])}
        if name == "library_save":
            raw = str(args.get("name") or args.get("module_name") or "").strip()
            mod, renamed = _module_name(raw)
            test_code = str(args.get("test_code", ""))
            if renamed and raw.isidentifier():
                # The test imports the module under its saved name, so the test's own references follow.
                test_code = re.sub(rf"\b{re.escape(raw)}\b", mod, test_code)
            code = next((str(args[k]) for k in ("code", "source", "module_code", "script")
                         if str(args.get(k) or "").strip()), "")
            if not code.strip():
                return {"error": ("library_save needs the module's source in `code`; this call had "
                                  f"{', '.join(sorted(args)) or 'no arguments'}. Nothing was saved.")}
            body = {"name": mod, "kind": args.get("kind") or "util",
                    "description": str(args.get("description", ""))[:2000], "code": code,
                    "test_code": test_code, "note": str(args.get("note", ""))[:2000],
                    "author": self.self_model, "objective_id": self.oid}
            out = request(CONTROL_PLANE, lib, body, timeout=400)
            # A smoke test that crashed: the module goes back to the model for a fix (the test stays).
            fix = self.repairer
            if fix is not None and test_code.strip():
                tests = (f"THE SMOKE TEST (it imports the module as `{mod}`; it stays as it is -- fix the MODULE):\n"
                         f"```python\n{test_code[:4000]}\n```")
                fix = (lambda tool, src, crash, _f=self.repairer: _f(tool, src, crash, tests))
            out = _auto_repair("library_save", code, out, fix,
                               lambda src: request(CONTROL_PLANE, lib, {**body, "code": src}, timeout=400),
                               who=self.self_model)
            if isinstance(out, dict) and out.get("saved"):
                self.saved.append(out.get("name") or mod)
            if renamed and mod:
                out = _with_note(out, (f"module names are lowercase identifiers of at most 48 characters: {raw!r} "
                                       f"was saved as {mod!r} -- import it with `from lib import {mod}`"))
            return out
        if name == "library_comment":
            cid = None
            if str(args.get("candidate") or "").strip():
                cid = (self._candidate(args.get("candidate"))[0] or {}).get("id")
            return request(CONTROL_PLANE, f"{lib}/{q(str(args.get('name', '')))}/comments", {
                "verdict": args.get("verdict") or "note", "text": str(args.get("text", ""))[:8000],
                "author": self.self_model, "candidate_id": cid})
        if name == "regime_map":
            return request(CONTROL_PLANE, f"/api/objectives/{oid}/regime-map", {
                "regime": str(args.get("regime", "")), "signals": args.get("signals") or None,
                "author": self.self_model}, timeout=600)
        if name == "regime_lab":
            n = max(2, min(5, int(args.get("buckets") or 3)))
            fields = [str(f) for f in args.get("fields") or []][:2]
            if args.get("module"):
                split = {"kind": "module", "module": str(args["module"])}
            else:
                split = {"kind": "fields", "fields": [{"field": f, "n": n} for f in (fields or ["GEX", "IntrVol"])],
                         "smooth": int(args.get("smooth") if args.get("smooth") is not None else 360),
                         "window_days": float(args.get("window_days") or 20)}
            members = [int(str(m).lstrip("#")) for m in args.get("members") or [] if str(m).lstrip("#").isdigit()]
            return request(CONTROL_PLANE, f"/api/objectives/{oid}/regime-lab", {
                "split": split, "members": members or None, "author": self.self_model,
                "wait": True, "compact": True}, timeout=1800)
        if name == "submit_candidate":
            return self.on_submit(args)
        name = self._mcp_name(name)
        out = super().call(name, self._task_args(name, args))
        if isinstance(out, dict) and str(out.get("error", "")).startswith("unknown tool") and "__" not in name:
            # A library module called as if it were a tool ("regime_detector", "signal",
            # "composite_skew_oinet_vwap_mod" -- 7 calls on the board): say how a module is used.
            try:
                mods = {m["name"]: m.get("kind") for m in request(CONTROL_PLANE, lib).get("modules", [])}
            except RuntimeError:
                mods = {}
            if name in mods:
                use = {"regime": f"{name}.detect(df)", "signal": f"{name}.signal(df)"}.get(mods[name], f"{name}.<function>(...)")
                return {"error": (
                    f"{name!r} is a library module, not a tool. Use it in run_python / submit_candidate code: "
                    f"`from lib import {name}` then {use}; read its code with library_get(name={name!r})"
                    + (f"; measure it per regime with regime_map(regime={name!r})" if mods[name] == "regime" else "")
                    + ".")}
        return out

    def _candidate(self, ref: Any) -> tuple[dict | None, dict | None]:
        """(the candidate `ref` names, None) or (None, the error to return). `ref` is a number
        ("123", "#123", "c123", "candidate 123") or an id (or a unique prefix of one).

        The old lookup read the 500 most recent candidates and stripped every leading "c" and
        "#": with 1,545 candidates an older number such as #1233 was "no candidate" (3 board
        errors), and an id that starts with "c" lost its first letter. Most of the 151 misses on
        the board were mentor IDEA numbers ([idea 1104]) passed as candidates -- the error says so."""
        s = str(ref or "").strip()
        m = re.fullmatch(r"(?:candidate|cand|seq|c)?\s*#?\s*(\d+)", s.lstrip("#").strip(), re.I)
        cands = request(CONTROL_PLANE, f"/api/objectives/{q(self.oid)}/candidates?order=recent&limit=5000"
                        ).get("candidates", [])
        hit = next((c for c in cands if c["id"] == s.lstrip("#")), None)
        if hit is None and m:
            hit = next((c for c in cands if str(c["seq"]) == m.group(1)), None)
        if hit is None and len(s) >= 4:
            pre = [c for c in cands if str(c["id"]).startswith(s.lstrip("#"))]
            hit = pre[0] if len(pre) == 1 else None
        if hit is not None:
            return hit, None
        seqs = sorted((int(c["seq"]) for c in cands if str(c.get("seq", "")).isdigit()), reverse=True)
        span = f"candidate numbers here run up to #{seqs[0]}; the newest are {', '.join(f'#{n}' for n in seqs[:5])}" \
            if seqs else "this objective has no candidates yet"
        return None, {"error": (f"no candidate {s!r} in this objective -- {span}. Pass a candidate's number (seq) or "
                                "id; a mentor idea number ([idea N]) is not a candidate number.")}

    def _task_args(self, name: str, args: dict) -> dict:
        """`args` with the objective's task filled in for its own task server's tools: there is
        only one task an iteration can mean, and Qwen sent task_sample_rows without it (10-01
        12:09, "task: Field required") -- a wasted call for a value the runner already knows."""
        m = self.objective.get("metric") or {}
        server, task = m.get("task_server"), m.get("task")
        if not (_is_task(self.objective) and server and task) or not name.startswith(f"{server}__") \
                or args.get("task") not in (None, ""):
            return args
        tool = next((t["function"] for t in self.mcp if t["function"]["name"] == name), None)
        props = ((tool or {}).get("parameters") or {}).get("properties") or {}
        return {**args, "task": task} if "task" in props else args


def _compact_regime(result: dict) -> dict:
    out = {}
    for label, row in (result.get("regimes") or {}).items():
        sigs = sorted(((k, v) for k, v in row["signals"].items() if v.get("sharpe") is not None),
                      key=lambda kv: kv[1]["sharpe"], reverse=True)
        out[label] = {"share": round(row["share"], 3),
                      "net_sharpe_by_signal": {k: round(v["sharpe"], 2) for k, v in sigs},
                      "gross_sharpe": {k: round(v["sharpe_gross"], 2) for k, v in sigs if v.get("sharpe_gross") is not None},
                      "trades_per_day": {k: round(v.get("trades_per_day") or 0, 1) for k, v in sigs}}
    return out


def _regime_lab_lines(runs: list[dict] | None, kind: str) -> list[str]:
    """The REGIME LAB section of the brief: the latest runs (in-sample only), the newest with its
    router script, or -- before the first run -- the nudge to make one."""
    if kind not in ("sharpe", "sortino", "calmar", "total_return", "cagr", "max_drawdown"):
        return []
    if not runs:
        return ["", "REGIME LAB -- not run yet for this objective. Nobody knows which of the team's verified "
                "strategies works in which regime. Call regime_lab (default GEX x IntrVol terciles) and submit the "
                "router it writes; then try other splits (Pressure_Total, HistVol, pinning, skew fields)."]
    lines = ["", "REGIME LAB (verified candidates measured inside regimes; in-sample daily Sharpe while each regime "
             "was in force, net of costs; routes switch away from the best single member only on evidence from BOTH "
             "in-sample halves; a router that does not clearly beat that member in-sample is not worth submitting):"]
    for i, run in enumerate(runs):
        r = run.get("router_in_sample") or {}
        lines.append(f"- run {run.get('run')}: {run.get('regime')} -- router IS Sharpe {r.get('sharpe')} "
                     f"(halves {', '.join(str(h) for h in r.get('halves') or [])}) vs best single {r.get('vs_best_single')}"
                     + (f"; submitted as {run['submitted_as']}" if run.get("submitted_as") else "; NOT submitted yet"))
        for reg in run.get("regimes") or []:
            if reg["regime"] in ("warmup", "unknown"):
                continue
            share = f"{(reg.get('share') or 0) * 100:.0f}%"
            lines.append(f"    {reg['regime']} ({share}, {reg.get('days')} d) -> {reg['route']}; best: "
                         + "; ".join(reg.get("best") or []))
        if i == 0 and run.get("code"):
            lines.append("  Its router script (submit as is, or change routes / fields and argue why):")
            lines += ["    " + ln for ln in run["code"].splitlines()]
    return lines


def _clip(value: Any, limit: int = TOOL_RESULT_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars -- narrow the query or aggregate]"


# Python SyntaxError messages that only ever come from source that ran off the end -- an open
# string, an open bracket, a hanging indent. Match against SyntaxError.msg, lowercased. Missing
# colons, wrong indents and other "the model wrote bad code" errors are not here on purpose: a
# real bug in the model's code should reach the sandbox and come back as a normal error the
# model can learn from, not be silently retried.
_CUTOFF_SYNTAX_MARKERS = (
    "eol while scanning string literal",   # Python < 3.12
    "unterminated",                         # 3.12+: "unterminated string literal",
                                            #        "unterminated triple-quoted string literal"
    "unexpected eof",                       # unclosed brackets
    "was never closed",                     # 3.10+: "'(' was never closed"
    "expected an indented block",           # a def/if/for whose body was cut off
    "unexpected character after line continuation",  # trailing backslash
)
# Characters that a truncated line ends on when the last token was cut mid-expression: an
# opener, a dot before an attribute, a bare operator, a trailing comma. A model-written
# "missing colon" ends with a letter or digit, so those still reach the sandbox.
_CUTOFF_TRAILING_CHARS = "([{,.=+-*/%|&^<>@"
# Tools whose arguments carry Python source. library_save's `test_code` is optional but must
# also parse; submit_candidate's `code` is the whole strategy. All three lose real work when a
# cut-off argument gets through -- an experiment for run_python, a saved module for
# library_save, a candidate slot for submit_candidate.
_CODE_TOOLS = frozenset({"run_python", "library_save", "submit_candidate"})
_CODE_ARG_KEYS = ("code", "test_code")


def _truncated_code_call(name: str, args: Any, args_json_ok: bool, finish: str | None) -> bool:
    """A code-carrying tool call whose Python source was cut off mid-token by the output budget.

    Seen on qwen/qwen3.8-27b@groq (48 sightings): the reply's finish is "length" (or
    "tool_calls" -- Groq reports that even when max_tokens hit mid-argument),
    completion_tokens are 44-215, and the code ends unclosed (``bar = ft.load('sql_exports_db``,
    ``print("GEX:", g.quantile``, ``rows['SkewRR_Value'].to_np.``). Running it just wastes an
    experiment on a SyntaxError the model already implicitly knows about, so the guard refuses
    the call and asks for a shorter resend WITHOUT charging an experiment (or a save slot, or a
    candidate slot). A genuine model-written syntax error (missing colon, bad indent) is NOT
    flagged: the SyntaxError markers we match, and the trailing-char check that backs them up,
    only fire on source that ran off the end -- a "def foo()" ends with a letter or digit, not
    an opener, dot or bare operator.
    """
    if name not in _CODE_TOOLS or not isinstance(args, dict):
        return False
    # library_save carries both `code` and (optional) `test_code`; either being cut is proof.
    pieces = [str(args.get(k) or "") for k in _CODE_ARG_KEYS]
    code = "\n".join(p for p in pieces if p)
    if not code:
        return finish == "length" or not args_json_ok
    try:
        compile(code, "<candidate>", "exec")
        return False
    except SyntaxError as se:
        msg = (se.msg or "").lower()
        looks_cut = any(m in msg for m in _CUTOFF_SYNTAX_MARKERS)
        if not looks_cut:
            # Groq reports finish "tool_calls" even for arguments that were sliced mid-token,
            # so a SyntaxError whose lowest-level signal is only "invalid syntax" still needs
            # a second read. When the trailing non-whitespace char is one that a well-formed
            # statement never ends on (``.``, ``=``, ``(``, ...), the source was cut.
            tail = code.rstrip()
            if tail and tail[-1] in _CUTOFF_TRAILING_CHARS:
                looks_cut = True
    # The SyntaxError alone is not proof (the model may have written bad Python). Combine it
    # with a signal that the reply itself was truncated: provider-reported length, a JSON
    # envelope that would not parse, or an error that only comes from cut-off source.
    return looks_cut or finish == "length" or not args_json_ok


# =======================================================================================
# Auto-repair: a script that crashed goes back to the model that wrote it
# =======================================================================================
# Every one of the 93 failed run_python results in agent_activity.sqlite3 (bug #11) carried the
# agent's own traceback -- a polars Series/expression mix-up, an unknown keyword, a typo. Each
# cost the agent a round to read the traceback and a round to resend, and showed up on the
# Work and Bugs pages as an error even when the next call fixed it. Now the runner does that
# round trip inside the same tool call: the failing code, the error and the traceback go back
# to the SAME model with "fix this, change as little as possible"; the fix runs the same way;
# a success is what the agent sees, marked `auto_repaired` so the pages count the call as
# recovered. A repair never costs an extra experiment, candidate slot or save.
AUTO_REPAIR = os.getenv("FREESWARM_AUTO_REPAIR", "1").strip().lower() not in ("0", "false", "no", "off")
AUTO_REPAIR_ATTEMPTS = 2
# Longer scripts are not sent back: the prompt (and the full script the model must return)
# would cost more than the round the agent spends fixing it itself.
AUTO_REPAIR_MAX_CODE_CHARS = 24_000
AUTO_REPAIR_TRACE_LINES = 40
AUTO_REPAIR_TRACE_CHARS = 4_000
# A "fix" that keeps less than half the original lines most likely deleted the failing part.
AUTO_REPAIR_MIN_KEEP = 0.5
# Wall time one tool call may spend on repairs (model replies; a rerun already started finishes).
# 10-01 17:13, Muse-Glimmer: three repair replies of 5-7k tokens took ~13 minutes of a ~45-minute
# iteration, two of them cut off before any code.
AUTO_REPAIR_MAX_S = float(os.getenv("FREESWARM_AUTO_REPAIR_MAX_S", "360"))
# Fixes tried without the model first (see _mechanical_fix), not counted as model attempts.
AUTO_REPAIR_MECHANICAL = 3
_EXC_LINE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt|Warning)\b")
_FENCE = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\r?\n(.*?)```", re.S | re.I)


def _crash_text(out: Any) -> str:
    """The stderr of a tool result whose script failed: run_python (ok false), a candidate
    that failed to run (status error), a library module whose smoke test failed. Not a
    causality (look-ahead) refusal, a budget refusal or any other error without a script run."""
    if not isinstance(out, dict):
        return ""
    if out.get("ok") is False:
        return str(out.get("stderr") or "")
    if out.get("status") == "error":
        return str(out.get("stderr_tail") or "")
    if (out.get("saved") is False and "causality" not in out
            and str(out.get("error") or "").startswith("the smoke test failed")):
        return str(out.get("test_output") or "")
    return ""


def _script_crash(out: Any) -> dict | None:
    """{"error", "traceback", "hint"} when `out` is a script that raised a Python exception --
    the kind of failure the author can fix. A run killed at the time or memory limit is not."""
    text = _crash_text(out)
    if "Traceback (most recent call last)" not in text or "[killed:" in text or out.get("timed_out"):
        return None
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    error = next((ln.strip() for ln in reversed(lines) if _EXC_LINE.match(ln.strip())), lines[-1].strip())
    hint = str(out.get("hint") or "").strip()
    note = str(out.get("error") or "").strip()      # a candidate's score_note, library_save's message
    if note and note != error and note not in hint:
        hint = f"{hint}\n{note}".strip()
    return {"error": error[:400], "traceback": "\n".join(lines[-AUTO_REPAIR_TRACE_LINES:])[-AUTO_REPAIR_TRACE_CHARS:],
            "hint": hint[:1200]}


def _tool_failed(out: Any) -> bool:
    """A result that did not do its job (what the Work page counts as a failed call)."""
    return isinstance(out, dict) and (bool(out.get("error")) or out.get("ok") is False
                                      or out.get("saved") is False or out.get("status") == "error")


def _code_lines(code: str) -> int:
    return sum(1 for ln in code.splitlines() if ln.strip())


def _fenced_code(text: str) -> str | None:
    """The script in a repair reply: the longest ```python block, or the whole reply when it
    is bare Python. None when there is no complete block (a reply cut off mid-block)."""
    blocks = [b for b in _FENCE.findall(text or "") if b.strip()]
    if blocks:
        return max(blocks, key=len).strip("\n")
    bare = (text or "").strip()
    if bare and "```" not in bare and "\n" in bare:
        try:
            compile(bare, "<repair>", "exec")
            return bare
        except (SyntaxError, ValueError):
            return None
    return None


# =======================================================================================
# A submission written as text, and the forced submit turn
# =======================================================================================
# 2026-09-30..10-01: 22 explore/build iterations ended "no submission". Every one with a reply
# had used all OBJECTIVE_TOOL_ROUNDS and was then asked, in a final round that offered NO tools,
# to "call submit_candidate NOW": Muse-Glimmer answered with its script in a ```python block
# (7 times), Qwen wrote the call in its own <function=...> syntax, and Muse-Glimmer once wrote
# a harmony/XML run_python call that the salvage then RAN -- spending the last experiment and
# the last round on a check instead of the submission. The final round now offers
# submit_candidate itself (tool_choice), a script written as text is submitted as what it is,
# and an iteration that still ends without one gets one forced submit turn.
REPORT_CALL = re.compile(r"\bft\s*\.\s*report(?:_\w+)?\s*\(")
FORCED_SCRIPT_CHARS = 14_000        # a longer default script is named, not pasted, in the prompt
_NO_TOOL_CHOICE: set[str] = set()   # models whose server refused a named tool_choice
CONVERSE_NO_ANSWER = "tool loop ended without an answer"


def _reports(code: str) -> bool:
    """A script that compiles and reports its result through ft.report_* (a candidate, not a probe)."""
    if not code or not REPORT_CALL.search(code):
        return False
    try:
        compile(code, "<candidate>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def _text_script(text: str) -> str | None:
    """The complete candidate script in a reply written as text: the longest closed ```python
    block that compiles and calls ft.report_*. None when there is none (a probe that only
    prints, a block cut off mid-way, prose)."""
    blocks = sorted((b.strip("\n") for b in _FENCE.findall(text or "") if b.strip()), key=len, reverse=True)
    return next((b for b in blocks if _reports(b)), None)


def _text_rationale(text: str, limit: int = 600) -> str:
    """One line of rationale for a salvaged script: the reply's prose around the code block."""
    prose = " ".join(_FENCE.sub(" ", text or "").split()).strip()
    prose = prose.replace("```", "").strip()
    return (prose[:limit] if prose else "") or "script written as text in the final reply (submitted by the runner)"


# --- how fast each model answers ----------------------------------------------------------
# 10-01 19:43: Qwen (~33 tok/s with two agents and their extra calls on one engine) timed out
# both a repair and an answer_feedback step. The runner now measures each model's effective
# speed (completion tokens over the wall time of the whole request, prompt processing and
# queueing included) and sizes those waits by it.
SPEED_MIN_TOKENS = 64          # shorter replies say more about latency than about speed
_speed: dict[str, float] = {}  # model -> completion tokens per second (moving average)
_reply_s: dict[str, float] = {}  # "<kind>|<model>" -> seconds a reply of that kind takes (moving average)
_speed_lock = threading.Lock()


def _note_speed(model: str | None, tokens: Any, seconds: float) -> None:
    try:
        tokens = int(tokens or 0)
    except (TypeError, ValueError):
        return
    if not model or tokens < SPEED_MIN_TOKENS or seconds <= 0:
        return
    tps = tokens / seconds
    with _speed_lock:
        prev = _speed.get(model)
        _speed[model] = tps if prev is None else 0.7 * prev + 0.3 * tps


def _note_reply_s(kind: str, model: str | None, seconds: float) -> None:
    if not model or seconds <= 0:
        return
    key = f"{kind}|{model}"
    with _speed_lock:
        prev = _reply_s.get(key)
        _reply_s[key] = seconds if prev is None else 0.7 * prev + 0.3 * seconds


def _expected_reply_s(model: str | None, tokens: int, kind: str | None = None) -> float | None:
    """About how long `model` takes to write a reply of `tokens` tokens -- or, for `kind`, the
    time such replies actually took, when that is longer. None: not measured yet."""
    tps = _speed.get(model or "")
    est = tokens / tps if tps else None
    seen = _reply_s.get(f"{kind}|{model}") if kind else None
    known = [x for x in (est, seen) if x]
    return max(known) if known else None


# --- repairs that need no model ---------------------------------------------------------------
# 10-01 19:43: Qwen's run_python failed with "NameError: name 'pl_col' is not defined" (pl.col)
# and the repair request to the model timed out. A misspelled name has one obvious fix; it is
# made here, and the model is asked only when that does not work.
_ALIAS_MODULES = {"pl": "polars", "np": "numpy", "pd": "pandas", "ft": "ft", "math": "math"}
_NAME_ERROR = re.compile(r"NameError: name '([A-Za-z_]\w*)' is not defined")
MECHANICAL_CUTOFF = 0.85
_module_attrs: dict[str, set] = {}


def _attrs_of(module: str) -> set:
    """Public names of one of the modules in _ALIAS_MODULES; the sandbox's `ft` is read from its
    source (it is not importable here)."""
    if module in _module_attrs:
        return _module_attrs[module]
    names: set = set()
    try:
        if module == "ft":
            import ast
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            with open(os.path.join(here, "sandbox", "ft.py"), encoding="utf-8") as fh:
                src = fh.read()
            for node in ast.parse(src).body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(node.name)
                elif isinstance(node, ast.Assign):
                    names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        else:
            import importlib
            names = {n for n in dir(importlib.import_module(module)) if not n.startswith("_")}
    except Exception:  # noqa: BLE001 -- unknown module, no fix from it
        names = set()
    names = {n for n in names if not n.startswith("_")}
    _module_attrs[module] = names
    return names


def _script_names(code: str) -> tuple[set, dict]:
    """(names the script defines or imports, {alias: module} of its `import <module> as <alias>`
    for the modules in _ALIAS_MODULES)."""
    import ast
    tree = ast.parse(code)
    names: set = set()
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Import):
            for a in node.names:
                bound = a.asname or a.name.split(".")[0]
                names.add(bound)
                if a.name in _ALIAS_MODULES.values():
                    aliases[bound] = a.name
        elif isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names if a.name != "*")
    return names, aliases


def _mechanical_fix(code: str, crash: dict) -> tuple[str, str] | None:
    """(fixed code, what was changed) for a NameError with one obvious fix, else None:
      - `pl_col` where the script imports polars as pl and polars has `col` -> `pl.col` (also np,
        pd, ft, math; an attribute that is a close misspelling of exactly one of the module's);
      - a name within MECHANICAL_CUTOFF of exactly one name the script defines or imports."""
    import difflib
    m = _NAME_ERROR.search(f"{crash.get('error') or ''}\n{crash.get('traceback') or ''}")
    if not m:
        return None
    bad = m.group(1)
    use = re.compile(rf"(?<![\w.'\"]){re.escape(bad)}\b")
    if not use.search(code):
        return None
    try:
        names, aliases = _script_names(code)
    except (SyntaxError, ValueError):
        return None
    new = None
    a = re.match(r"(pl|np|pd|ft|math)_(\w+)$", bad)
    if a and a.group(1) in aliases:
        attrs = _attrs_of(aliases[a.group(1)])
        attr = a.group(2)
        if attr not in attrs:
            close = difflib.get_close_matches(attr, sorted(attrs), n=2, cutoff=MECHANICAL_CUTOFF)
            attr = close[0] if len(close) == 1 else None
        if attr:
            new = f"{a.group(1)}.{attr}"
    if new is None:
        # Python's own suggestion ("NameError: name 'roll_70' is not defined. Did you mean: 'roll_q70'?",
        # 10-02 00:39) settles it even when difflib sees two near names (roll_q30 / roll_q70).
        said = re.search(rf"name '{re.escape(bad)}' is not defined\. Did you mean: '(\w+)'\?",
                         f"{crash.get('error') or ''}\n{crash.get('traceback') or ''}")
        if said and said.group(1) in names:
            new = said.group(1)
    if new is None:
        close = difflib.get_close_matches(bad, sorted(names - {bad}), n=2, cutoff=MECHANICAL_CUTOFF)
        if len(close) == 1:
            new = close[0]
    if new is None:
        return None
    fixed = use.sub(new, code)
    try:
        compile(fixed, "<repair>", "exec")
    except (SyntaxError, ValueError):
        return None
    return fixed, f"{bad} -> {new}"


def _repair_prompt(tool: str, code: str, crash: dict, extra: str = "") -> str:
    what = {"library_save": "library module (its smoke test failed)",
            "submit_candidate": "candidate script (it failed to run in the evaluation harness)"}.get(tool, "Python script")
    parts = [f"Your {what} raised an error. Fix it.\n",
             f"ERROR: {crash['error']}\n",
             f"TRACEBACK (last lines):\n{crash['traceback']}\n"]
    if crash.get("hint"):
        parts.append(f"HINT FROM THE PLATFORM:\n{crash['hint']}\n")
    if extra:
        parts.append(extra.rstrip() + "\n")
    parts.append(f"THE CODE THAT FAILED:\n```python\n{code}\n```\n")
    parts.append(
        "Reply with ONLY the complete corrected code in ONE ```python block -- no explanation before or "
        "after it. Change as little as possible: fix what the error names and keep everything else the "
        "same (same intent, data, logic, parameters and printed output). Do not delete, skip or stub out "
        "the failing part, and do not wrap it in try/except to hide the error -- make it work.")
    return "\n".join(parts)


def _auto_repair(tool: str, code: str, out: Any, fix: Any, rerun: Any, *, who: str = "?") -> Any:
    """`out` as returned to the agent, after trying to repair a crashed script.

    `fix(tool, code, crash) -> str | None` asks the model for a corrected script; `rerun(code)`
    runs it exactly the way the original ran (without charging the budget again). Up to
    AUTO_REPAIR_ATTEMPTS fixes are tried, each one from the latest version that ran. On
    success the rerun's result comes back with

        "auto_repaired": {"attempts": n, "errors": ["<error line of each failed run>", ...],
                          "original_code_lines": k}

    as its FIRST key, the corrected code in `code_ran` (last) and a note at the top of stdout
    (or in `note`). Otherwise the original failure comes back unchanged, plus a brief
    `auto_repair_failed` at the end. No attempt starts after AUTO_REPAIR_MAX_S, and a `fix` that
    raises RuntimeError (a failed or cut-off repair reply) ends the repairs.
    """
    if not AUTO_REPAIR or fix is None or not str(code or "").strip():
        return out
    crash = _script_crash(out)
    if crash is None:
        return out
    if len(code) > AUTO_REPAIR_MAX_CODE_CHARS:
        log(f"{who}: {tool} crashed ({crash['error'][:120]}); {len(code)} chars is too long to auto-repair")
        return out
    # The repair model call reads the deadline (Worker._repair_code) to cap its own wait.
    deadline = time.time() + AUTO_REPAIR_MAX_S
    _TL.repair_deadline = deadline
    try:
        return _repair_loop(tool, code, out, crash, fix, rerun, who, deadline)
    finally:
        _TL.repair_deadline = None


def _repair_loop(tool: str, code: str, out: Any, crash: dict, fix: Any, rerun: Any, who: str,
                 deadline: float) -> Any:
    original_lines = _code_lines(code)
    errors = [crash["error"]]
    seen = {" ".join(code.split())}
    current, failure, last = code, crash, ""
    attempt = 0                       # model calls
    mechanical: list[str] = []        # fixes made without one (_mechanical_fix)
    while True:
        if time.time() >= deadline:
            last = f"the repair time limit ({AUTO_REPAIR_MAX_S:.0f}s) was used up"
            break
        mech = _mechanical_fix(current, failure) if len(mechanical) < AUTO_REPAIR_MECHANICAL else None
        if mech and " ".join(mech[0].split()) not in seen:
            fixed = mech[0]
            mechanical.append(mech[1])
            log(f"{who}: {tool} crashed ({failure['error'][:160]}); fixed without the model: {mech[1]}")
        else:
            if attempt >= AUTO_REPAIR_ATTEMPTS:
                break
            attempt += 1
            log(f"{who}: {tool} crashed ({failure['error'][:160]}); auto-repair {attempt}/{AUTO_REPAIR_ATTEMPTS}")
            try:
                fixed = fix(tool, current, failure)
            except RuntimeError as exc:
                last = f"the repair request failed: {str(exc)[:200]}"
                break
            if not fixed or not fixed.strip():
                last = "no corrected code came back"
                continue
            if " ".join(fixed.split()) in seen:
                last = "the 'fix' was the same code"
                continue
        seen.add(" ".join(fixed.split()))
        if _code_lines(fixed) < AUTO_REPAIR_MIN_KEEP * original_lines:
            last = (f"the 'fix' dropped most of the code ({_code_lines(fixed)} of {original_lines} lines) "
                    "and was not run")
            continue
        try:
            compile(fixed, "<repair>", "exec")
        except (SyntaxError, ValueError) as exc:
            last = f"the fix did not compile: {exc}"[:300]
            continue
        try:
            res = rerun(fixed)
        except RuntimeError as exc:
            last = f"the repaired run could not be started: {str(exc)[:200]}"
            break
        again = _script_crash(res)
        if again is None and not _tool_failed(res):
            note = (f"Your script failed with {crash['error'][:300]}; it was repaired automatically -- the code "
                    "that ran is in `code_ran`. Use the repaired version from now on.")
            info: dict[str, Any] = {"attempts": attempt + len(mechanical), "errors": errors,
                                    "original_code_lines": original_lines}
            if mechanical:
                # "mechanical": true -- no model call was needed at all
                info.update(mechanical=attempt == 0, mechanical_fixes=mechanical, model_attempts=attempt)
            fixed_out: dict[str, Any] = {"auto_repaired": info}
            fixed_out.update(res if isinstance(res, dict) else {"result": res})
            if isinstance(fixed_out.get("stdout"), str):
                fixed_out["stdout"] = f"[{note}]\n" + fixed_out["stdout"]
            else:
                fixed_out = _with_note(fixed_out, note)
            fixed_out["code_ran"] = fixed
            log(f"{who}: {tool} auto-repaired on attempt {info['attempts']}"
                + (f", {len(mechanical)} without the model" if mechanical else "") + f" ({crash['error'][:120]})")
            return fixed_out
        if again is None:
            # It ran but failed some other way (killed at the limit, a look-ahead refusal): nothing
            # a traceback-driven fix can work on.
            last = str(res.get("error") or "the repaired run failed")[:300] if isinstance(res, dict) else "failed"
            break
        errors.append(again["error"])
        last = again["error"]
        current, failure = fixed, again
    log(f"{who}: {tool} auto-repair gave up ({last[:160]})")
    if isinstance(out, dict):
        return {**out, "auto_repair_failed": (f"an automatic repair was tried and did not work -- last attempt: "
                                              f"{last[:300]}. Fix the original error yourself.")}
    return out


def _summarize_args(name: str, args: dict) -> str:
    if name == "submit_candidate":
        return str(args.get("rationale", ""))[:300]
    if name == "run_python":
        code = str(args.get("code", ""))
        return f"{len(code.splitlines())} lines"
    if "sql" in args:
        return " ".join(str(args["sql"]).split())[:300]
    if name == "forecast":
        return f"{len(args.get('series') or [])} points, horizon {args.get('horizon')}"
    if name == "ask_model":
        return f"{args.get('model')}: {str(args.get('prompt', ''))[:200]}"
    return json.dumps(args, default=str)[:300]


# =======================================================================================
# Activity record for the agent inspector (app/agent_activity.py)
# =======================================================================================
# Each agent keeps a record of its current job -- the assignment, the prompts it was sent,
# every tool call with its inputs, each chat request's tokens, what it submitted -- and posts
# it to the control plane whenever it changes. One background thread does the posting and
# sends only each agent's latest state, at most every ACTIVITY_POST_S. Nothing here may slow
# down or fail an iteration, so every recording method swallows its own errors.
ACTIVITY_POST_S = 1.5
ACTIVITY_PROMPT_CHARS = 250_000  # per prompt: the inspector shows the full text (a 128k context is ~400k chars)
ACTIVITY_ARG_CHARS = 16_000      # per string argument (a script is the big one)
ACTIVITY_RESULT_CHARS = 6_000
ACTIVITY_TIMELINE = 150
_TL = threading.local()  # the agent this thread works for, sent as X-FreeSwarm-Agent


def _agent_header() -> dict[str, str]:
    agent = getattr(_TL, "agent", None)
    return {"X-FreeSwarm-Agent": urllib.parse.quote(agent)} if agent else {}


def _head_tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.75)
    return (text[:head] + f"\n\n...[{len(text) - limit:,} characters omitted from the middle]...\n\n"
            + text[-(limit - head):])


def _trim(v: Any, limit: int = ACTIVITY_ARG_CHARS) -> Any:
    if isinstance(v, str):
        return _head_tail(v, limit)
    if isinstance(v, dict):
        return {str(k): _trim(x, limit) for k, x in list(v.items())[:60]}
    if isinstance(v, (list, tuple)):
        return [_trim(x, limit) for x in list(v)[:200]]
    return v if v is None or isinstance(v, (bool, int, float)) else str(v)[:limit]


_SUBMIT_KEYS = ("auto_repaired", "candidate_id", "seq", "status", "in_sample_score", "lookahead", "lookahead_detail",
                "rank", "not_ranked", "contender_for_best", "judge_score", "error")


def _result_brief(name: str, out: Any) -> Any:
    if isinstance(out, dict) and name == "submit_candidate" and "error" not in out:
        return {k: _trim(out.get(k), 600) for k in _SUBMIT_KEYS if out.get(k) is not None}
    text = out if isinstance(out, str) else json.dumps(out, default=str)
    return _head_tail(text, ACTIVITY_RESULT_CHARS)


def _tool_call_hash(name: str, args: Any) -> str:
    """A stable digest of a tool call, for spotting a re-issue of one that already failed.
    Whitespace inside string values is collapsed and dict keys sorted, so `library_save` with
    the same code and description hashes identically across cosmetic edits."""
    def norm(v: Any) -> Any:
        if isinstance(v, dict):
            return {str(k): norm(v[k]) for k in sorted(v, key=str)}
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        if isinstance(v, str):
            return " ".join(v.split())
        return v
    payload = json.dumps({"n": name, "a": norm(args if isinstance(args, (dict, list)) else {})},
                         sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _err_signature(out: Any) -> str:
    """The stable part of a tool-error string: enough that two failures with the same cause
    hash to the same signature, but with numbers/paths stripped so trivial differences (a
    timestamp, a request id) don't count as new errors."""
    text = ""
    if isinstance(out, dict):
        text = str(out.get("error") or "")
    elif isinstance(out, str):
        text = out
    text = re.sub(r"\b0x[0-9a-fA-F]+\b|\b\d[\d.,]{3,}\b", "#", text)
    return " ".join(text.split())[:160]


class _ActivityPoster(threading.Thread):
    """Posts the latest record of each agent; a burst of changes becomes one post."""

    def __init__(self) -> None:
        super().__init__(name="activity-poster", daemon=True)
        self._pending: dict[str, dict] = {}
        self._cv = threading.Condition()
        self._warned = False
        self._inflight = False  # a batch taken off _pending and not yet posted

    def put(self, key: str, doc: dict) -> None:
        with self._cv:
            self._pending[key] = doc
            self._cv.notify()

    def flush(self, timeout: float) -> bool:
        """Wait (up to `timeout`) until every record put so far is posted -- so a runner that
        exits after a drain leaves the inspector with each iteration's final state."""
        end = time.time() + timeout
        while time.time() < end:
            with self._cv:
                if not self._pending and not self._inflight:
                    return True
            time.sleep(0.2)
        return False

    def run(self) -> None:
        while True:
            with self._cv:
                while not self._pending:
                    self._cv.wait()
                batch, self._pending = self._pending, {}
                self._inflight = True
            for key, doc in batch.items():
                try:
                    request(CONTROL_PLANE, "/api/agents/activity", doc, timeout=10)
                except RuntimeError as exc:
                    # An older control plane has no inspector, or it is restarting: keep the
                    # latest state and try again in a minute rather than on every change.
                    if not self._warned:
                        log(f"agent inspector: posting activity failed ({exc}); will keep trying quietly")
                        self._warned = True
                    with self._cv:
                        for k, d in batch.items():
                            self._pending.setdefault(k, d)
                    self._inflight = False
                    time.sleep(60)
                    break
            self._inflight = False
            time.sleep(ACTIVITY_POST_S)


_POSTER: _ActivityPoster | None = None
_POSTER_LOCK = threading.Lock()


def _poster() -> _ActivityPoster:
    global _POSTER
    with _POSTER_LOCK:
        if _POSTER is None:
            _POSTER = _ActivityPoster()
            _POSTER.start()
    return _POSTER


class _Activity:
    """One agent's current job, as the inspector shows it. Owned by the worker's thread."""

    def __init__(self, worker) -> None:
        self.w = worker
        self.rec: dict | None = None
        self._asked: dict[str, int] = {}   # (system, first user message) -> index in rec["asked"]
        self._seen: set = set()            # follow-up user messages already on the timeline
        self._chat_t0 = 0.0
        self._tool_t0 = 0.0

    # -- lifecycle ------------------------------------------------------------------------
    def begin(self, mode: str, obj: dict | None = None, **extra) -> None:
        try:
            self._close("interrupted")
            now = time.time()
            self.rec = {"id": f"{int(now * 1000)}-{random.randint(0, 9999)}", "mode": mode, "status": "running",
                        "objective": {"id": obj["id"], "title": obj.get("title")} if obj else None,
                        "started_at": now, "ended_at": None, "pending": None, "idea": None,
                        "asked": [], "timeline": [], "chats": [], "submissions": [],
                        "tokens": {"prompt": 0, "completion": 0, "chats": 0}, **extra}
            self._asked, self._seen = {}, set()
            self._push()
        except Exception as exc:  # noqa: BLE001 -- the inspector must never fail the work
            log(f"{self.w.agent_name}: activity record: {exc!r}")

    def begin_iteration(self, obj: dict, ctx: dict) -> None:
        """An objective iteration (or the chore the control plane handed out instead)."""
        try:
            audit = ctx.get("audit")
            mode = ("audit" if audit else "consolidate" if ctx.get("consolidate")
                    else "practices" if ctx.get("refresh_practices") else str(ctx.get("mode") or "?"))
            parent = ctx.get("parent") if mode in ("explore", "improve", "build") else None
            self.begin(
                mode, obj,
                parent={k: parent.get(k) for k in ("id", "seq", "rank", "in_sample_score", "diagnosis")} if parent else None,
                audit_of={k: audit.get(k) for k in ("id", "seq", "model", "score", "is_score")} if audit else None,
                ideas_offered=[{"id": i.get("id"), "model": i.get("model"), "tried": i.get("tried"),
                                "text": str(i.get("text") or "")[:400]} for i in (ctx.get("ideas") or [])[:6]],
                context={"lessons": len(ctx.get("lessons") or []), "notes": len(ctx.get("notes") or []),
                         "leaderboard": len(ctx.get("leaderboard") or []), "recent": len(ctx.get("recent") or []),
                         "features": len(ctx.get("features") or []), "library": len(ctx.get("library") or []),
                         "total_candidates": ctx.get("total_candidates")})
        except Exception as exc:  # noqa: BLE001
            log(f"{self.w.agent_name}: activity record: {exc!r}")

    def end(self, status: str | None = None, reason: str | None = None) -> None:
        try:
            self._close(status, reason)
        except Exception as exc:  # noqa: BLE001
            log(f"{self.w.agent_name}: activity record: {exc!r}")

    def _close(self, status: str | None, reason: str | None = None) -> None:
        rec = self.rec
        if rec is None or rec.get("ended_at"):
            return
        rec["ended_at"] = time.time()
        rec["pending"] = None
        subs = rec["submissions"]
        if status:
            rec["status"] = status
        elif subs:
            rec["status"] = "submitted"
            rec["outcome"] = f"#{subs[-1].get('seq')} {subs[-1].get('status')}"
        else:
            rec["status"] = "done" if rec["mode"] not in ("explore", "improve", "build") else "no submission"
        if reason:
            rec["reason"] = str(reason)[:400]
        self._push()

    # -- what it was asked, and what the model said ---------------------------------------
    def chat_start(self, payload: dict) -> None:
        try:
            if self.rec is None or self.rec.get("ended_at"):
                self.begin("chat")
            rec, now = self.rec, time.time()
            msgs = payload.get("messages") or []
            model = payload.get("model")
            system = next((str(m.get("content") or "") for m in msgs if m.get("role") == "system"), "")
            first_i = next((i for i, m in enumerate(msgs) if m.get("role") == "user"), None)
            first = str(msgs[first_i].get("content") or "") if first_i is not None else ""
            key = f"{model}|{hash(system)}|{hash(first)}"
            if key not in self._asked:
                if len(rec["asked"]) < 8:
                    self._asked[key] = len(rec["asked"])
                    rec["asked"].append({"at": now, "model": model, "system": _head_tail(system, ACTIVITY_PROMPT_CHARS),
                                         "system_chars": len(system), "prompt": _head_tail(first, ACTIVITY_PROMPT_CHARS),
                                         "prompt_chars": len(first), "tools": [
                                             (t.get("function") or {}).get("name") for t in payload.get("tools") or []]})
                    self._event({"kind": "asked", "at": now, "index": self._asked[key], "model": model})
            elif msgs and msgs[-1].get("role") == "user" and len(msgs) - 1 != first_i and (key, len(msgs)) not in self._seen:
                # A nudge, a final-round prompt, the reflection request, a malformed-call notice.
                self._seen.add((key, len(msgs)))
                self._event({"kind": "message", "at": now, "text": _head_tail(str(msgs[-1].get("content") or ""), 4000)})
            rec["pending"] = {"kind": "chat", "model": model, "since": now}
            self._chat_t0 = now
            self._push()
        except Exception as exc:  # noqa: BLE001
            log(f"{self.w.agent_name}: activity record: {exc!r}")

    def chat_done(self, payload: dict, result: dict | None, error: str | None = None) -> None:
        try:
            rec, now = self.rec, time.time()
            if rec is None:
                return
            u = (result or {}).get("usage") or {}
            choice = ((result or {}).get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            # `max_tokens` is what the runner ASKED FOR: pairs with prompt_tokens + finish so
            # the inspector shows when a reply was cut off by our own budget (finish "length"
            # AND completion_tokens == max_tokens) versus by the model (finish "stop"). Before
            # this was recorded, one Groq bug -- max_tokens computed to 256 because the context
            # window fell back to the DEFAULT for a mislabelled external model -- was invisible.
            chat = {"at": self._chat_t0, "model": payload.get("model"), "seconds": round(now - self._chat_t0, 1),
                    "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                    "max_tokens": payload.get("max_tokens"),
                    "finish": choice.get("finish_reason"), "error": (error or "")[:600] or None,
                    # a side request, not the agent's own conversation: the Work page counts its timeout as
                    # soft ("skipped, model busy") instead of a chat error (app/work.py reads this first)
                    "side": ("answer_feedback" if getattr(self.w, "_gen_deadline", None)
                             else "auto_repair" if getattr(_TL, "repair_deadline", None) else None)}
            rec["chats"] = (rec["chats"] + [chat])[-80:]
            rec["tokens"]["prompt"] += int(u.get("prompt_tokens") or 0)
            rec["tokens"]["completion"] += int(u.get("completion_tokens") or 0)
            rec["tokens"]["chats"] += 1
            self._event({"kind": "chat", **chat,
                         "tool_calls": [(tc.get("function") or {}).get("name") for tc in msg.get("tool_calls") or []],
                         "said": _head_tail(str(msg.get("content") or "").strip(), 2000) or None,
                         "reasoning": _head_tail(str(msg.get("reasoning_content") or "").strip(), 1200) or None})
            rec["pending"] = None
            self._push()
        except Exception as exc:  # noqa: BLE001
            log(f"{self.w.agent_name}: activity record: {exc!r}")

    # -- its inputs: every tool call -----------------------------------------------------
    def tool_start(self, name: str, args: Any) -> None:
        try:
            if self.rec is None:
                return
            self._tool_t0 = time.time()
            self.rec["pending"] = {"kind": "tool", "name": name, "since": self._tool_t0, "args": _trim(args, 2000)}
            self._push()
        except Exception as exc:  # noqa: BLE001
            log(f"{self.w.agent_name}: activity record: {exc!r}")

    def tool_done(self, name: str, args: Any, out: Any, ok: bool, **extra) -> None:
        try:
            rec = self.rec
            if rec is None:
                return
            e = {"kind": "tool", "at": self._tool_t0, "name": name, "args": _trim(args), "ok": bool(ok),
                 "seconds": round(time.time() - self._tool_t0, 1), "result": _result_brief(name, out), **extra}
            if isinstance(out, dict) and isinstance(out.get("auto_repaired"), dict):
                # The script crashed and the runner's repair worked (see _auto_repair): also kept
                # whole here, since `result` may be a head-and-tail cut of the JSON.
                e["auto_repaired"] = _trim(out["auto_repaired"], 600)
            self._event(e)
            if name == "submit_candidate" and isinstance(out, dict) and (out.get("seq") or out.get("candidate_id")):
                a = args if isinstance(args, dict) else {}
                sub = {k: out.get(k) for k in _SUBMIT_KEYS if out.get(k) is not None and k != "lookahead_detail"}
                sub.update(at=time.time(), idea=a.get("idea"), rationale=str(a.get("rationale") or "")[:600])
                rec["submissions"].append(sub)
                if a.get("idea") not in (None, ""):
                    rec["idea"] = a.get("idea")
            rec["pending"] = None
            self._push()
        except Exception as exc:  # noqa: BLE001
            log(f"{self.w.agent_name}: activity record: {exc!r}")

    def step(self, name: str, args: Any, out: Any, ok: bool, at: float, **extra) -> None:
        """A step the runner took for the agent (not a model tool call) -- e.g. answering its
        feedback before the work -- shown on the timeline like a tool call."""
        try:
            if self.rec is None:
                return
            self._event({"kind": "tool", "at": at, "name": name, "args": _trim(args), "ok": bool(ok),
                         "seconds": round(time.time() - at, 1), "result": _result_brief(name, out), **extra})
            self._push()
        except Exception as exc:  # noqa: BLE001
            log(f"{self.w.agent_name}: activity record: {exc!r}")

    # -- plumbing -------------------------------------------------------------------------
    def _event(self, e: dict) -> None:
        tl = self.rec["timeline"]
        tl.append(e)
        if len(tl) > ACTIVITY_TIMELINE:
            # Keep the "asked" markers: they anchor the prompts shown above the timeline.
            drop = next((i for i, x in enumerate(tl) if x.get("kind") != "asked"), 0)
            del tl[drop]
            self.rec["timeline_dropped"] = self.rec.get("timeline_dropped", 0) + 1

    def _push(self) -> None:
        w = self.w
        doc = {"agent": w.agent_name, "model": w.model, "role": w.role, "slot": w.slot,
               "project_id": w.pid, "record": self.rec}
        _poster().put(f"{w.pid}|{w.agent_name}", json.loads(json.dumps(doc, default=str)))


def _act(worker) -> _Activity:
    """The worker's activity record (made on first use). Also marks the calling thread as
    working for this agent, so its control-plane requests say who is asking."""
    a = worker.__dict__.get("_activity")
    if a is None:
        a = worker.__dict__["_activity"] = _Activity(worker)
    _TL.agent = worker.agent_name
    return a


# =======================================================================================
# Worker: one agent = one (project, model)
# =======================================================================================
SYSTEM = (
    "You are an autonomous agent working on a task queue for a project. Complete the task "
    "fully and return the finished work itself -- not a plan to do it.\n"
    "Use your tools to get facts: query the data, the database or a forecaster instead of "
    "guessing, and never invent numbers you could have looked up. Prefer aggregating in SQL "
    "over pulling raw rows. If a tool errors, read the error and correct the call.\n"
    "If the task asks for code, return complete, runnable code in a fenced block with the "
    "language tag.\n\n"
)


ITERATE_SYSTEM = (
    "You are one agent in a research swarm working continuously on a single objective. The swarm "
    "improves by evolution: each iteration an agent builds ONE candidate -- by improving a strong "
    "earlier one or by exploring a new idea -- a trusted harness scores it, and the best survive "
    "to be built on. You also inherit the team's lessons; respect them.\n"
    "Work like a careful quant researcher: form a concrete hypothesis, check it against the data "
    "with your tools, keep the rules simple enough to generalise, and never assume a number you "
    "could measure. Your candidate must be a complete, runnable script.\n\n"
)


def _fmt(v: Any) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"


def _pct(v: Any) -> str:
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "n/a"


def _hname(h: Any) -> str:
    """A decile study's horizon as the brief shows it: h<bars>, or to-close for 0."""
    return "to-close" if h in (0, "0") else f"h{h}"


def _knowledge_lines(deci: dict | None, inputs: list[dict] | None) -> list[str]:
    """The DECILE STUDIES and FORECAST INPUTS blocks (app/deciplot.py, app/tslab.py): what the
    team has already measured about signals and forecast inputs, for searchers and the mentor."""
    lines: list[str] = []
    if deci:
        lines += ["", f"DECILE STUDIES ({deci.get('studied')} signals studied in-sample; rolling decile edges from past "
                  "sessions, forward returns within the session; t overlap-adjusted). Check these -- deci_plot() with no "
                  "signal lists them all -- before running a new study; a repeated study is served from the store:"]
        for tf, rows in (deci.get("best_by_timeframe") or {}).items():
            lines.append(f"- {tf}: " + "; ".join(
                f"{r['signal']} {_hname(r['h'])} top-bottom {(r['spread_bps'] or 0):+.2f} bps (t {r['t']}, rho {r['rho']}, "
                f"{r['verdict']}, same sign in {_pct(r['consistency'])} of periods)" for r in rows))
        if deci.get("flat"):
            lines.append("- FLAT (no decile relationship at any timeframe -- do not build on these alone): "
                         + ", ".join(deci["flat"]))
        if deci.get("unstable"):
            lines.append("- UNSTABLE (sign flips between sub-periods): " + ", ".join(deci["unstable"]))
        if deci.get("shapes"):
            # The curve, not just its ends: a straight line is tradeable in proportion to the
            # signal; a U or a single-decile effect only at the extremes. And a spread smaller
            # than the round-trip cost cannot pay for trading on that signal alone.
            lines.append("- SHAPES -- mean forward return (bps) per decile, lowest signal -> highest, at each "
                         "signal's timeframe/horizon with the largest solid spread (|t| >= 3). Trade where the curve says, "
                         "and compare the spread with the ~2x cost_bps a round trip pays:")
            for s in deci["shapes"]:
                curve = " ".join("." if v is None else f"{v:+.2f}" for v in s["means"])
                lines.append(f"  {s['signal']} {s['timeframe']} {_hname(s['h'])}: [{curve}] {s['shape']}; top-bottom "
                             f"{(s['spread_bps'] or 0):+.2f} bps, t {s['t']}")
    for g in inputs or []:
        if not lines or not any(x.startswith("FORECAST INPUTS") for x in lines):
            lines += ["", "FORECAST INPUTS (explored Chronos-2 input combinations, in-sample, paired vs the target alone; "
                      "explore_forecast_inputs adds more -- tested combinations are never re-run):"]
        best = (f"best {', '.join(g['best_inputs'])} skill {_fmt(g.get('best_skill'))} (+{_fmt(g.get('best_gain'))} "
                f"+/- {_fmt(2 * (g.get('best_se') or 0))})" if g.get("best_inputs") else "no combination beats it beyond noise")
        lines.append(f"- {g['target']} h{g['horizon']}{' @' + g['bar'] if g.get('bar') else ''}: alone skill "
                     f"{_fmt(g.get('baseline_skill'))}; {best}; {g['tested']} tested. Inputs that help: "
                     f"{', '.join(g.get('helpful') or []) or 'none'}"
                     + (f"; hurt: {', '.join(g['hurts'])}" if g.get("hurts") else "")
                     + (f"; tested and useless: {', '.join(g['useless'])}" if g.get("useless") else ""))
    return lines


def _json_object(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, dict) else None
    except ValueError:
        return None


def _json_array(text: str) -> list | None:
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, list) else None
    except ValueError:
        return None


def _ens(c: dict) -> str:
    """' (ensemble of #a+#b)' for an ensemble candidate in the brief; '' for a script."""
    members = c.get("ensemble")
    return f" (ensemble of {'+'.join('#' + str(s) for s in members)})" if members else ""


# Tools that work on the project's own datasets (not a task server's rows): hidden from agents
# working on a task objective.
TASK_HIDDEN_TOOLS = {
    "list_sql_tables", "describe_sql_table", "query_sql", "list_forecasters", "forecast", "forecast_feature",
    "correlations", "combine_candidates", "explore_forecast_inputs", "regime_lab",
}


def _is_task(objective: dict) -> bool:
    return (objective.get("metric") or {}).get("kind") == "task"


def _analysis_ok(objective: dict) -> bool:
    """Whether field_scan / deci_plot / regime_map can run: a price column to measure moves
    against -- or, for a task objective, its target over the in-sample view of the task's rows."""
    m = objective.get("metric") or {}
    if m.get("price_column"):
        return True
    return _is_task(objective) and bool(m.get("target") and objective.get("dataset"))


def _task_lines(ctx: dict) -> list[str]:
    """HOW "BETTER" IS MEASURED and the candidate contract for a task objective: the task server
    defines the rows, what an action means and the score; this spells it out for the agent."""
    o = ctx["objective"]
    m = o["metric"]
    t = ctx.get("task") or m.get("task_info") or {}
    act = t.get("action") or {}
    sc = t.get("score") or {}
    server = m.get("task_server")
    bounds = ""
    if act.get("min") is not None or act.get("max") is not None:
        bounds = f" Bounds: {act.get('min')} .. {act.get('max')} (values outside are clipped)."
    lines = [
        f"- This objective is a TASK served by the task server '{server}' (task '{m.get('task')}'): "
        f"{t.get('title') or ''}".rstrip(),
    ]
    if t.get("description"):
        lines.append("- THE PROBLEM: " + " ".join(str(t["description"]).split()))
    lines += [
        f"- ROWS: time-aligned rows, one per time step, sorted by the timestamp column `t`"
        + (f" ({t['rows']} in all, {t['in_sample_rows']} of them in-sample)" if t.get("rows") and t.get("in_sample_rows") else "")
        + f". TARGET column: `{t.get('target') or m.get('target') or '(see ft.task())'}`.",
        f"- ACTION: one number per row -- {act.get('description') or act.get('kind') or 'see ft.task()[\"action\"]'}.{bounds} The action decided at "
        f"row t may use rows up to and including t and takes effect from row t to row t+1. Actions may be sparse: a "
        f"row without one keeps the previous action (before the first: {act.get('initial', 0)}).",
        f"- SCORE: {sc.get('name') or 'score'} ({'higher' if sc.get('higher_is_better', True) else 'LOWER'} is better), "
        + next((f"the value function '{f.get('title') or f['name']}': {f.get('description')}. "
                for f in t.get("value_functions") or [] if f.get("name") == t.get("value_function") and f.get("description")), "")
        + "Computed by the task server from your actions. The leaderboard ranks the WEAKER of the in-sample and "
        "holdout scores, so a strategy must work in both. After each submission you get the in-sample score, the "
        "server's in-sample diagnostics and notes -- read them before your next change.",
    ]
    if t.get("brief"):
        lines.append("- TASK RULES: " + " ".join(str(t["brief"]).split()))
    # Advice only the data/action MCP can give (its data, actions and valuation), verbatim.
    for g in t.get("guidance") or []:
        if isinstance(g, dict) and str(g.get("text") or "").strip():
            lines.append(f"- {str(g.get('title') or 'Note').upper()}: " + " ".join(str(g["text"]).split()))
    total = float(m.get("min_trades") or 0.0)
    if total:
        lines.append(f"- TRADE FLOOR (REQUIRED): at least {total:g} trades in-sample IN ALL, or the candidate is not "
                     "ranked -- enough for its score to be more than luck. It is NOT a daily quota: a selective "
                     "strategy that skips most days is welcome. Your diagnostics report trades per side.")
    floor = float(m.get("min_trades_per_day") or 0.0)
    if floor:
        lines.append(f"- TRADE FLOOR (REQUIRED): at least {floor:g} trades per day on average in-sample, or the "
                     "candidate is not ranked. A strategy that decides once a day (at the open, on the overnight "
                     "gap, at one fixed time) cannot reach it: check the signal throughout the session and "
                     "re-enter whenever it fires again after an exit. Your diagnostics report trades_per_day.")
    cols = t.get("columns") or []
    if cols:
        lines.append("- COLUMNS: " + "; ".join(
            f"{c['name']}" + (f" ({c['role']})" if c.get("role") not in (None, "signal") else "")
            + (f" = {c['description']}" if c.get("description") else "") for c in cols[:200]))
    view = o.get("dataset")
    lines.append(f"- Explore the rows with run_python (in-sample rows only) or the task server's tools: "
                 f"{server}__task_query (SQL over the in-sample rows as the table `rows`, e.g. SELECT hour, "
                 f"avg(target) FROM rows GROUP BY hour), {server}__task_sample_rows, {server}__task_column_stats.")
    if view and _analysis_ok(o):
        lines.append(f"- EVERY FIELD of the schema is also open to the analysis tools, on the in-sample rows as the "
                     f"dataset `{view}` (time column `t`, measured against the target `{t.get('target') or m.get('target')}`): "
                     f"deci_plot (does a field sort the next moves into deciles?), field_scan (screen them all), "
                     f"regime_map (which signals work in which regime), query_data (SQL over `{view}`).")
    return lines


def _task_contract(ctx: dict) -> list[str]:
    o = ctx["objective"]
    return ["", "CANDIDATE CONTRACT",
            "- A complete Python script run offline (POLARS, numpy, scipy, pyarrow). USE POLARS, NOT PANDAS: the "
            "rows are 700k+ bars and polars is several times faster (expressions, .over('session'), "
            ".rolling_mean(), .shift(), .fill_null(strategy='forward')). Read the rows ONLY with "
            "`import ft, polars as pl; rows = ft.rows_pl()` (a polars DataFrame sorted by `t`) and the task's "
            "description with `ft.task()`. There is no other data. On a large task load only the columns you use -- "
            "`ft.rows_pl(columns=['Close', 'GEX'])` -- the sandbox has 4 GB of RAM.",
            "- POLARS TYPES (most failed runs on 2026-09-30/10-01 were these): the time column (`t`, `SlotUtc`) is already a "
            "Datetime -- use pl.col('t').dt.date() / .dt.hour(), never .str.strptime; pl.col('x') is an EXPRESSION, used only inside "
            "select / with_columns / filter (.over('session'), .alias() live there), while df['x'] is a SERIES of data "
            "(.to_numpy(), .mean() -> a number); polars has no sort_values / reset_index / iloc / copy / cum_mean / nth "
            "(sort, with_row_index, row / slice, clone, cum_sum()/cum_count(), get); wrap every comparison in "
            "parentheses before & or |.",
            "- Report one action per row with `ft.report_actions(values, t=rows['t'])` (values: a polars Series or "
            "numpy array, one per row). Do not compute or report the score yourself -- the task server does.",
            "- Optionally ft.report(name=value, ...) extra numbers. Print a short summary.",
            "- The ft helpers below take polars frames and series as they are and return numpy-convertible "
            "results (.to_numpy()); pandas stays available but only where you truly need it (pandas 3: .ffill(), "
            "not fillna(method=...)).",
            "- Causal helpers for intraday trading on the rows (all take the rows and return one value per row; "
            "clock times are New York): ft.clock(rows['t']) -> (session, minute of day); ft.session_vwap(rows); "
            "ft.gamma_regime(rows) (-1 dealers short gamma / +1 long); ft.trend_exits(entries, rows, ...) turns "
            "+1/-1 entry signals into positions with a trailing stop, breakeven stop, re-entries, an optional daily "
            "trade limit and a clock exit (no fixed target) -- its knobs: size, stop_mult (the 'ATR multiple'), "
            "vol_window (the 'ATR lookback', in rows), stop_pct, trail, breakeven_at, vwap_exit, flat_at, "
            "no_entry_before, no_entry_after, max_trades_per_day, reverse, retrigger -- nothing else; ft.noise_area_breakout(rows, ...) is a published SPY intraday-momentum baseline; "
            "ft.decision_points(rows, times=[...]) gives the rows of fixed decision times; ft.admit(scores, sessions, "
            "per_day=3) keeps the best few candidate entries a day causally (a bar set from past sessions, taken in "
            "time order). For research in run_python "
            "ONLY: ft.label_outcomes(rows, points) (what happened after each point -- it reads the future); a strategy "
            "may use those labels only through ft.meta_filter(X, labels, sessions), which learns from sessions that "
            "have already closed. Each has a docstring: help(ft.trend_exits).",
            "- FAST RESEARCH in run_python (one experiment answers what took a whole iteration): "
            "ft.direction_scan(rows) tests every field at 10:00-12:00 for the DIRECTION of the rest of the day "
            "(the swarm's winners were trades pointing with the day's move; minute-scale signals lose to costs) -- a "
            "lead is positive in both halves with t_stat >= 3; ft.quick_score(positions, rows) scores a position "
            "series like the server does (within a few hundredths of Sharpe), with each half of the sessions and "
            "the trade floors; ft.sweep(make_positions, {param: [values]}, rows) scores up to 48 variants at once, "
            "ranked by the weaker half. Check a candidate with quick_score BEFORE submitting it.",
            f"- Limits: {o.get('eval_timeout_s', 300)}s, 4 GB RAM, no network."]


def _direction_rule(m: dict) -> str:
    """The objective's allowed sides, as the brief states them."""
    d = m.get("direction") or "both"
    if d == "long":
        return ("LONG ONLY: positions must be 0 or positive; the harness holds any negative position as flat.")
    if d == "short":
        return ("SHORT ONLY: positions must be 0 or negative; the harness holds any positive position as flat.")
    lev = m.get("max_leverage", 1)
    rule = (f"LONG AND SHORT: positions run from -{lev} (full short) to +{lev} (full long); a negative position "
            "is a short. Design every entry with its MIRROR: when the conditions that open a long appear "
            "reversed (forecast below zero instead of above, price under its average instead of over, the GEX "
            "regime pointing down), open a short of the same size logic. Do not clip or floor positions at 0 "
            "(no clip(0, ...), clip(lower=0), max(0, ...) or size bounds like (0.5, 3.0) applied to the "
            "signed position) -- bound |position| instead, keeping the sign. Shorts should fire just like "
            "longs, whenever the signal points down and only then.")
    share = float(m.get("min_side_share") or 0.0)
    if share:
        rule += (f" REQUIRED: longs and shorts must EACH be at least {share:.0%} of your in-sample trades, or the "
                 "candidate is not ranked (a one-sided strategy is riding the market's drift, not reading the "
                 "signal). The harness reports your long/short trade counts after each submission.")
    return rule


def _trade_book_lines(ctx: dict) -> list[str]:
    """The trade classes and the swarm's trade book (task objectives): what to optimise for, and what
    the big winners of every candidate so far had in common at entry."""
    tb = ctx.get("trade_book") or {}
    if not tb:
        return []
    lines = ["", f"TRADE CLASSES -- WHAT TO OPTIMISE FOR (big = at least {tb.get('threshold')} per unit of size, after "
             "costs, in-sample)", tb.get("goal") or ""]
    if tb.get("pool"):
        lines += ["What the swarm's trades so far say -- conditions that held in both halves of the in-sample period. "
                  "Build entries where trades EARN MORE and away from where they LOSE MORE; in a BIG-MOVE state "
                  "enter only with a condition that picks the direction there (the ++ pairs). A condition that shows "
                  "up for several fields is one market state measured several ways:", tb["pool"]]
    if tb.get("dataset"):
        lines.append(f"Every in-sample trade of every candidate is the dataset `{tb['dataset']}` (seq, entry, exit in "
                     "UTC, side +1/-1, size, bars, net, unit = net per unit of size, cls = big_winner / big_loser / "
                     "scratch). In run_python: `trades = ft.load_pl('" + tb["dataset"] + "')`, join the rows as of "
                     "each entry (`t = trades.sort('entry').join_asof(ft.rows_pl(columns=[...]).sort('t'), "
                     "left_on='entry', right_on='t', strategy='backward')`) and test a filter before you submit: how "
                     "many big winners does it keep, how many scratch and big losers does it drop? "
                     "`t.filter((pl.col('side') == -1) & (pl.col('GEX') < 0)).group_by('cls').len()` counts each class "
                     "a filter keeps. These are POLARS frames: select rows with .filter(...), never df[mask] (that "
                     "selects COLUMNS in polars and fails), no .copy(), and iter_rows(named=True) yields dicts. The "
                     "trade_review tool shows any candidate's review.")
    return lines


def iteration_prompt(ctx: dict) -> str:
    """The standing brief for one iteration, rebuilt from the control plane's context."""
    o = ctx["objective"]
    m = o["metric"]
    kind = m["kind"]
    lines = [f"OBJECTIVE: {o['title']}"]
    if o.get("description"):
        lines.append(o["description"])
    lines += ["", "HOW \"BETTER\" IS MEASURED"]
    positions = bool(m.get("price_column") and o.get("dataset"))
    if kind == "task":
        lines += _task_lines(ctx)
    elif kind in ("sharpe", "sortino", "calmar", "total_return", "cagr", "max_drawdown"):
        if positions:
            lines.append(
                f"- Score: {ctx['metric_label']} of DAILY returns, higher is better. Your script reports "
                f"POSITIONS with ft.report_positions(values, t=df['t']) (polars) or a pandas series indexed by bar "
                f"timestamp -- the position decided "
                f"at the close of each bar. The harness holds each position until your next one, marks it to "
                f"market on column '{m['price_column']}' of '{o['dataset']}', charges {m.get('cost_bps', 0)} bps "
                f"per unit of position change, caps |position| at {m.get('max_leverage', 1)}, compounds per day "
                f"and computes the score itself. Do not compute or report returns yourself.")
            lines.append(
                f"- SIZING is a lever, not just direction: |position| may go up to {m.get('max_leverage', 1)}. "
                "Inverse-volatility sizing -- bigger when recent volatility is low, smaller when it is high -- gives "
                "each trade similar risk, so a few violent days stop dominating the result and the equity curve gets "
                "smoother (the ranking rewards a smooth curve). Size by conviction (signal strength) the same way. "
                "Choose the size when a trade OPENS and hold it; do not rescale every bar. "
                "`pos = ft.size(direction, ft.inverse_vol(bars['Close'], lookback=60), base=2.0, "
                "max_leverage=...)` does this causally (rebalance='entry' by default; rebalance='band', band=0.5 "
                "resizes only when the target drifts that far). Compare against the same strategy at a constant "
                "size and say which you used in your rationale.")
            lines.append(
                "- COSTS DECIDE MOST RESULTS HERE. Every change in position size is a trade: a position rescaled "
                "every bar (by volatility, by a continuous signal) pays costs every bar and loses even when the "
                "signal is right. Prefer discrete positions held for many bars. After each submission the harness "
                "reports your in-sample score before costs, after costs and FLIPPED (every sign reversed); its "
                "`diagnosis` says which one to fix -- read it before your next change. " + _direction_rule(m))
            lines.append(
                "- SWING CAPTURE: the harness cuts the price into its swing legs (high/low pivots within each day) "
                "and reports `swings_in_sample`: up legs you were long through, down legs you were short through "
                "(hits), and the opposite (misses); net = hits - misses. Being always long nets about zero -- a good "
                "strategy catches up legs AND down legs. Use it to see which side of your logic is failing.")
            if m.get("intraday"):
                lines.append(
                    "- INTRADAY ONLY: every trade must open and close on the SAME DAY. The harness forces your "
                    "position flat at each day's last bar, so nothing is held overnight -- a position still open "
                    "then is closed at that bar's price and pays the exit cost. Close positions yourself before "
                    "the session ends BY THE CLOCK (e.g. flat from 15:30 New York; timestamps are UTC) rather than "
                    "relying on the forced exit -- never by the day's last bar (groupby(day).max()), which is not "
                    "known until the day is over and fails the look-ahead test, "
                    "and do not expect an overnight gap to pay: a position wanted again next morning is a new "
                    "trade entered at that day's bars.")
        else:
            lines.append(f"- Score: {ctx['metric_label']} of the DAILY returns you report with "
                         "ft.report_returns(series indexed by date), net of costs; higher is better.")
        lines.append(f"- A candidate needs at least {m.get('min_active_days', 20)} days with a position in the "
                     "scored period to be ranked.")
    elif kind == "reported":
        lines.append("- Score: the number your script reports with ft.report_score(x); "
                     + ("higher" if m.get("higher_is_better", True) else "LOWER") + " is better.")
    else:
        lines.append(f"- Score: a judge model rates your answer 0-10. Rubric: {m.get('rubric') or 'overall quality'}. "
                     "Put the answer in submit_candidate's `answer` (code optional).")
    if o.get("split_date"):
        lines.append(
            f"- Data is split at {o['split_date']}. Ranking uses ONLY the period from {o['split_date']} on, which "
            f"you never see: query_data and run_python show only earlier rows. Fitting the in-sample period "
            f"harder does not help -- prefer few parameters and rules with a reason to work.")
    if o.get("lookahead_check"):
        lines.append(
            f"- Look-ahead test: every submission is re-run with the data cut seconds to minutes AFTER its own "
            f"{'action changes' if kind == 'task' else 'trades'} (and at the split); if any earlier "
            f"{'action' if kind == 'task' else 'position'} changes, it is rejected and never ranked. Decide "
            f"each {'action' if kind == 'task' else 'position'} from rows up to and including its timestamp only: "
            "no shift(-1), no centred windows, no "
            "full-sample mean/std/quantiles or models fit on all rows -- use rolling or expanding windows. With "
            "resampled bars, date each bar's values at the moment they are COMPLETE: a 15-min bar built from rows in "
            "[T, T+15m) is known at T+15m, so its position must be stamped at T+15m or later (resample with "
            "label='right', closed='left'); stamping it at T leaks up to 15 minutes of the future.")
    finance_from = len(lines)
    lines += ["", "CANDIDATE CONTRACT",
              "- A complete Python script run offline (POLARS, numpy, scipy, pyarrow). USE POLARS, NOT PANDAS: load "
              "data ONLY with `import ft; df = ft.load_pl(\"<view>\", columns=[...])` (several times faster on the "
              "700k-bar data; the ft helpers take polars as it is). Datasets: " + ", ".join(
                  (ctx.get("datasets") or []) + [f["view"] for f in ctx.get("features") or []]) + ".",
              "- KNOW WHICH FRAME YOU HOLD: ft.load_pl() gives POLARS; ft.load() and every ft helper that returns a "
              "frame or series (ft.resample, ft.inverse_vol, ft.size, ft.align, ...) give PANDAS, also when you pass "
              "polars in. Convert with x.to_pandas() / pl.from_pandas(x). A library module takes the kind of frame "
              "its code was written for -- read it with library_get before calling it.",
              "- POLARS TYPES (most failed runs on 2026-09-30/10-01 were these): the time column (`t`, `SlotUtc`) is already a "
              "Datetime -- use pl.col('t').dt.date() / .dt.hour(), never .str.strptime; pl.col('x') is an EXPRESSION, used only inside "
              "select / with_columns / filter (.over('session'), .alias() live there), while df['x'] is a SERIES of data "
              "(.to_numpy(), .mean() -> a number); polars has no sort_values / reset_index / iloc / copy / cum_mean / nth "
              "(sort, with_row_index, row / slice, clone, cum_sum()/cum_count(), get); wrap every comparison in "
              "parentheses before & or |.",
              "- Optionally ft.report(name=value, ...) extra numbers (trades, turnover). Print a short summary.",
              f"- Limits: {o.get('eval_timeout_s', 300) if 'eval_timeout_s' in o else 300}s, 4 GB RAM, no network."]
    fcs = ctx.get("forecasters") or []
    if fcs:
        lines += ["", "TIME-SERIES FORECASTING MODELS (loaded on a GPU, available to you now):"]
        for f in fcs:
            lines.append(f"- {f['model']} ({f.get('family') or 'forecaster'}, context {f.get('context_length')}, "
                         f"native horizon {f.get('native_horizon')})")
        lines.append("  Call them FROM YOUR SCRIPT: fc = ft.forecast(\"<column>\", inputs=[...], horizon=6, "
                     "model=\"<name>\", join=df, time_col=TIME) attaches <column>_fc_median / _fc_q10 / _fc_q90 / "
                     "_fc_change to df, made causally (the forecast at bar t read data up to t) and joined the only "
                     "safe way. The first run of a new recipe is built by the harness and re-run automatically; the "
                     "recipe is stored, so the same call always returns the same forecasts. forecast_feature(...) "
                     "builds one ahead of time and reports its skill; forecast eyeballs a single forecast.")
        if any((f.get("family") or "") == "chronos2" for f in fcs):
            lines.append("  Chronos-2 takes INPUTS: forecast_feature(model=\"amazon/chronos-2\", column=\"Close\", "
                         "covariates=[\"Pressure_Below\", \"Imb_OINet_D0\"], calendar=true, bar=\"1min\", horizon=30) forecasts "
                         "Close while reading those columns. The feature records its lift over the same forecast without the "
                         "inputs (with a +/- 2 SE range) -- keep inputs only if the lift is significant.")
        if any((f.get("family") or "") == "kronos" for f in fcs):
            lines.append("  Kronos is a CANDLE model: it reads whole OHLCV bars (not one column) and forecasts future "
                         "candles -- call forecast_feature(model=\"<kronos name>\", bar=\"1min\", horizon=10); no column "
                         "needed. It is slow (generates bar by bar), so a feature is capped at 3000 forecasts. Its "
                         "feature adds fc_high_q90 / fc_low_q10 (forecast range). Compare its skill with Chronos'.")
        lines.append("  Example: forecast_feature(columns=[\"IntrVol\", \"GEX\"], horizon=30, every=30) forecasts both "
                     "30 bars (5 min) ahead every 30 bars (5 min). The grid is capped at 60,000 forecasts per "
                     "feature (every >= ~12 on this data); a smaller `every` is raised automatically. Check the "
                     "returned in-sample skill before building on it.")
        lines.append("  TRY SEVERAL HORIZONS. A model that is a coin flip 30 bars out can be right 3 bars out, or the "
                     "reverse -- do not settle on the first horizon you try. Build the same series at a few horizons "
                     "(e.g. 3, 6, 12, 30) and compare their skill and direction accuracy. Then either trade the "
                     "horizon that forecasts best, or combine them into a LIKELY FAN: at each bar, a vote on the "
                     "direction across horizons, each weighted by how accurate that horizon has been (its in-sample "
                     "direction accuracy minus 0.5, or better its ROLLING hit rate). A fan whose horizons agree and "
                     "whose accurate horizons point the same way is a conviction signal (size by it, or trade only "
                     "then); horizons that disagree say stand aside. Keep it causal: a forecast made at t is only "
                     "scored once t + horizon has passed, so a rolling hit rate at t may use only forecasts made at "
                     "or before t - horizon. Drop horizons with no skill from the fan instead of letting them dilute it.")
    lines += ["", "TIMEFRAMES",
              "- The data is 10-second bars. Signals often work better on slower bars: try 20s, 30s, 1min, "
              "5min, 15min (and combinations -- e.g. a 5min trend filter with 30s entries). "
              "`bars = ft.resample(df, '5min')` builds OHLCV bars stamped at their LAST underlying 10s bar (the "
              "moment they are complete); decide on them, then carry positions back onto the 10s grid with "
              "`ft.align(pos, bars[TIME], df[TIME])` before ft.report_positions. Say which timeframe you used "
              "in your rationale.",
              "", "REGIMES",
              "- Markets change character (volatility, gamma sign, trend vs chop), and a signal that works in one "
              "regime often loses in another. Label regimes CAUSALLY -- `reg = ft.regimes(df['IntrVol'], n=3, "
              "window=6*2340).rename('vol_regime')` ranks each bar against only the bars before it -- or with a "
              "library regime module; measure which signals work in each regime on in-sample data (regime_map); "
              "then trade each regime with its own signal: `pos = ft.route(reg, {'low': mom.rename('momentum'), "
              "'high': mr.rename('meanrev')})` (unlisted regimes stay flat). Regimes that exist in-sample are the "
              "ones you can learn; a regime rule tuned to one stretch of days is overfitting.",
              "- ft.route records which regime each bar was in, and the equity chart colours each stretch by it. "
              "Add `ft.report_regime(reg, signal=df['IntrVol'])` to also plot the series the regime came from.",
              "- MULTI-STRATEGY BY REGIME: the team's verified candidates are building blocks. "
              "`ft.candidate_positions(1233)` runs candidate #1233's own script inside yours and returns its "
              "positions (time-indexed, any timeframe); `reg = ft.regime_grid(df, {'GEX': 3, 'IntrVol': 3}, "
              "time=TIME)` labels every bar causally ('GEX:high|IntrVol:low', ...); `ft.route(reg, {label: "
              "positions})` aligns and routes them. The regime_lab tool measures which candidate works in which "
              "regime and writes that script for you -- run it with fields that plausibly change market "
              "behaviour (gamma sign/level, volatility, pressure, pinning) and submit the router it suggests."]
    if kind in ("sharpe", "sortino", "calmar", "total_return", "cagr", "max_drawdown"):
        lines += ["", "COMBINING STRATEGIES",
                  "- Two strategies whose daily returns are nearly uncorrelated (|rho| < 0.3 in-sample) lose on different "
                  "days, so a portfolio of both has a SMOOTHER equity curve than either -- and the ranking rewards a "
                  "smooth curve. combine_candidates(members=[a, b, ...], weighting, lookback_days, rationale) records "
                  "such a portfolio as an ENSEMBLE candidate: its daily return is the weighted sum of the members' own "
                  "net daily returns (nothing re-runs, no netting), scored and ranked like any candidate.",
                  "- Call correlations() FIRST: it gives the in-sample correlation matrix and suggested low-|rho| sets. "
                  "Combine members with different signals, timeframes or regimes, each with a positive in-sample score; "
                  "two copies of the same idea are correlated and add nothing.",
                  "- weighting 'equal' gives each member 1/n; 'inverse_vol' sizes each by 1/(its volatility over the "
                  "last lookback_days, from returns BEFORE each day), so a calm member is not drowned out by a wild one. "
                  "Members must already be verified: scored, look-ahead passed, not disqualified, not ensembles. This "
                  "is an extra action -- still submit your own candidate this iteration."]
    if kind == "task":
        # The contract, forecasters, 10-second bars and GEX regimes above are about the project's
        # own datasets; a task candidate reads only the task server's rows.
        del lines[finance_from:]
        lines += _task_contract(ctx)
    # Stable first (the engine's prefix cache reuses it across iterations), what changes
    # every iteration last: the leaderboard, recent attempts, messages, teammates, the assignment.
    fields = ctx.get("fields") or {}
    if fields:
        total = sum(len(f["columns"]) for f in fields.values())
        if kind == "task":
            lines += ["", f"FIELD GUIDE -- all {total} columns the task server serves, as it describes them; every one is "
                      "usable (ft.rows(columns=[...]) loads a subset) and open to field_scan / deci_plot:"]
        else:
            lines += ["", f"FIELD GUIDE -- all {total} columns of the dataset; every one is usable (ft.load(view, columns=[...]) "
                      "to load a subset). Most GEX/greek fields are untested -- screen them with field_scan:"]
        for fam, f in fields.items():
            lines.append(f"- {fam}: {f['about'] or ''} -> {', '.join(f['columns'])}")
    scan = ctx.get("field_scan")
    if scan:
        lines += ["", f"LATEST FIELD SCAN ({scan['horizon_bars']} bars ahead, {scan['fields_scanned']} fields, in-sample IC "
                  "of the level / of the change):"]
        for r in scan["top"][:12]:
            reg = f" by regime {r['ic_by_regime']}" if r.get("ic_by_regime") else ""
            lines.append(f"- {r['field']}: {r['ic_level']} / {r['ic_change']}{reg}")
    lab = ctx.get("forecast_lab")
    if lab:
        verdict = ("beats the target alone beyond noise" if lab.get("best_significant") and (lab.get("best_gain") or 0) > 0
                   else "is within noise of the target alone -- treat input effects as unproven")
        lines += ["", f"FORECAST LAB (operator's input analysis, {lab.get('model')}, target {lab.get('target')}, "
                  f"{lab.get('points')} in-sample points): best inputs {', '.join(lab.get('best_inputs') or []) or 'none'} "
                  f"(skill {_fmt(lab.get('best_skill'))} vs {_fmt(lab.get('baseline_skill'))} alone) {verdict}."]
        if lab.get("significant_impacts"):
            lines.append("  Inputs with significant impact: " + ", ".join(f"{k} {v:+.4f}" for k, v in lab["significant_impacts"]))
    lines += _knowledge_lines(ctx.get("deci_studies"), ctx.get("forecast_inputs"))
    lib = ctx.get("library") or []
    lines += ["", "CODE LIBRARY (shared, reusable -- `from lib import <name>`)"]
    if lib:
        for m in lib:
            ev = f"used by {m['used_by']} ({m['ok']} ok, {m['errors']} errors"
            ev += f", {m['lookahead_fails']} look-ahead fails" if m["lookahead_fails"] else ""
            ev += f", {m['champions']} champions" if m["champions"] else ""
            ev += f", best in-sample {_fmt(m['best_in_sample'])})" if m.get("best_in_sample") is not None else ")"
            by = f" by {_short_name(m['author'])}" if m.get("author") else ""
            lines.append(f"- {m['name']} [{m['kind']} v{m['version']}]{by}: {m['description'][:200]} -- {ev}")
            for c in m.get("comments") or []:
                lines.append(f"    {c['verdict'].upper()} ({c['author']}): {c['text'][:200]}")
    else:
        lines.append("- (empty)")
    lines.append(
        "- Build reusable pieces here with library_save instead of burying them in one script: regime "
        "detectors (detect(df) -> label per row, from GEX sign/level/changes, volatility (HistVol, IntrVol, "
        "realised), pinning, skew, charm/vanna...), signals (signal(df) -> position per row) and risk rules. "
        "Then regime_map shows which signal works in which regime, and a strategy routes them: "
        "`pos = ft.route(regime_mod.detect(df), {'neg_gamma_volatile': mom.signal(df), 'pos_gamma_calm': mr.signal(df)})`. "
        "Leave library_comment verdicts (works/broken + the candidate number as evidence) on modules you use.")
    for rm in ctx.get("regime_maps") or []:
        lines += ["", f"REGIME MAP from {rm['regime_module']} v{rm['version']} (in-sample Sharpe NET of costs; "
                  "gross and trades/day in brackets):"]
        for label, row in (rm.get("regimes") or {}).items():
            top = list((row.get("by_signal") or {}).items())[:4]
            lines.append(f"- {label} ({row['share'] * 100:.0f}% of bars): "
                         + ", ".join(f"{k} {v['sharpe']} [{v.get('gross')}, {v.get('trades_per_day')}/d]" for k, v in top))
    lines += _regime_lab_lines(ctx.get("regime_lab"), kind)
    board = ctx.get("forecast_board") or []
    feats = [] if board else (ctx.get("features") or [])
    if board:
        lines += ["", "FORECAST SCOREBOARD (built forecasts; skill > 0 beats 'no change', lift_from_inputs > 0 = the "
                  "input columns helped; used_by / helped = how candidates that loaded it scored in-sample vs "
                  "candidates using no forecast). Build on forecasts with skill and a positive `helped`; drop the rest:"]
        for f in board:
            lines.append(f"- {f['view']}: {f.get('model')} forecast of {', '.join(f.get('series') or ['?'])}"
                         + (f" reading {', '.join(f['inputs'])}" if f.get("inputs") else "")
                         + f", {f.get('horizon')} bars ahead. Skill {json.dumps(f.get('skill'), default=str)[:260]}; "
                         f"used by {f.get('used_by')}, helped {f.get('helped')}"
                         # vs its own parent where possible (tslab.forecast_report): the fairer test
                         + (f", verdict {f['verdict']}" if f.get("verdict") else ""))
    if feats:
        lines += ["", "FORECAST FEATURES ALREADY BUILT (load with ft.load(view)):"]
        for f in feats:
            pr = f.get("params") or {}
            sk = "; ".join(f"{k}: skill {v.get('skill_vs_no_change')}, direction {v.get('direction_accuracy')}"
                           for k, v in (f.get("skill") or {}).items())
            lines.append(f"- {f['view']}: {pr.get('model')} forecast of {', '.join(pr.get('series') or [pr.get('column') or '?'])} "
                         f"in {pr.get('dataset')}, {pr.get('horizon')} bars ahead, every {pr.get('every')} bars, "
                         f"{f.get('rows')} rows. In-sample {sk}")
    notes = ctx.get("notes") or []
    if notes:
        lines += ["", "OPERATOR STEERING -- follow it (newest first):"]
        lines += [f"- {n['text']}" for n in notes]
    ideas = ctx.get("ideas") or []
    if ideas:
        lines += ["", "DIRECTIONS FROM THE MENTOR / STRONGER MODELS -- concepts to test, each with the experiment "
                  "that would falsify it. Prefer testing one of these (especially one tried little) over another "
                  "variation of the leader, and pass its number as `idea` to submit_candidate so the team learns "
                  "whether the IDEA works:"]
        for i in ideas:
            lines += [f"[idea {i['id']}] from {i['model']}, tried {i.get('tried', 0)} times so far:", i["text"]]
    coaching = ctx.get("coaching")
    if coaching:
        # The mentor's read of the last results, posted at the end of its #planning note -- which
        # TEAMMATES clipped to 400 characters, so it never reached a searcher until now.
        lines += ["", f"MENTOR COACHING ({coaching['minutes_ago']} min ago, {coaching['by']}) -- what the evidence says "
                  "the team should stop, keep or build on:", coaching["text"]]
    research = (ctx.get("research") or {}).get("documents") or []
    if research:
        lines += ["", "RESEARCH LIBRARY -- documents the operator added (research_search / research_get). Their ideas "
                  "reach you above as RESEARCH IDEA; their Python code imports as `from research.<package> import "
                  "<module>` in run_python and in candidates. Their numbers are the authors', on their data:"]
        lines += [f"- {d['id']} \"{d['title'][:90]}\" (package {d['package']}): {d.get('ideas', 0)} ideas, "
                  f"{d.get('code', 0)} code listings" for d in research if d.get("status") == "ready"][:10]
    if ctx.get("lessons"):
        lines += ["", "TEAM LESSONS (shared memory, newest first):"]
        lines += [f"- {x}" for x in ctx["lessons"]]
    if ctx.get("liked"):
        lines += ["", "OPERATOR FAVOURITES -- runs the operator flagged as the SHAPE they want (a steady equity curve "
                  "that keeps climbing). Build on these and aim for more like them:"]
        for c in ctx["liked"]:
            note = f" -- operator: {c['operator_note']}" if c.get("operator_note") else ""
            lines.append(f"- candidate {c['seq']} by {c['model']}: in-sample {_fmt(c['in_sample_score'])}{note} -- "
                         f"{c['rationale'][:300]}")
    if ctx.get("leaderboard"):
        lines += ["", "LEADERBOARD (ranked on consistency: the weaker of in-sample and the hidden holdout, times how "
                  "smooth the whole equity curve is -- steady gains in BOTH periods win; in-sample score shown):"]
        for c in ctx["leaderboard"]:
            lines.append(f"#{c['rank']}  candidate {c['seq']}{_ens(c)} by {c['model']}: in-sample {_fmt(c['in_sample_score'])} -- {c['rationale']}")
    if ctx.get("promising"):
        lines += ["", "PROMISING -- no ranked candidate makes money yet; these have a POSITIVE in-sample score. Fix what "
                  "keeps them off the leaderboard or build on them (get_candidate for the code; pass the number as "
                  "`parent` to submit_candidate):"]
        for c in ctx["promising"]:
            why = f" -- not ranked: {c['problem'].rstrip('. ')}" if c.get("problem") else (f", rank {c['rank']}" if c.get("rank") else "")
            lines.append(f"- candidate {c['seq']} by {c['model']}: in-sample {_fmt(c['in_sample_score'])}{why}. "
                         f"{c['rationale'][:240]}")
    if ctx.get("recent"):
        lines += ["", "RECENT ATTEMPTS (do not repeat these):"]
        for c in ctx["recent"]:
            status = c["status"] if not c.get("problem") else f"{c['status']}: {c['problem']}"
            lines.append(f"- candidate {c['seq']}{_ens(c)} ({c['model']}, {status}, look-ahead {c.get('lookahead')}"
                         f"{', rank ' + str(c['rank']) if c.get('rank') else ''}): {c['rationale'][:200]}")
    inbox = ctx.get("inbox") or []
    if inbox:
        lines += ["", "MESSAGES TO YOU from the mentor and teammates -- act on them or say why not. Link what you do: "
                  "pass reply_to=<the number> in your #planning post (cite any others in its text as #<number>), "
                  "or answer the sender directly with team_post(to=<sender>, reply_to=<number>). A question gets an "
                  "answer, a suggestion gets your result or your reason to do something else:"]
        lines += [f"- #{m['seq']} from {m['from']} ({m['minutes_ago']} min ago"
                  + (f", about your candidate {m['candidate']}" if m.get("candidate") else "")
                  + (f", replying to your #{m['reply_to']}" if m.get("reply_to") else "") + f"): {m['text']}"
                  for m in inbox]
    mates = ctx.get("teammates") or []
    lines += ["", "TEAMMATES RIGHT NOW (#planning, last 45 min) -- pick a different direction or build on theirs:"]
    lines += [f"- {m['who']} ({m['minutes_ago']} min ago): {m['text']}" for m in mates] or ["- (no plans posted)"]
    lines += _trade_book_lines(ctx)
    # The agent's own answers to its feedback (Worker.answer_feedback), nearest the assignment
    # so the work that follows is shaped by them.
    lines += _commitment_lines(ctx.get("commitments"))
    lines += ["", "YOUR ASSIGNMENT THIS ITERATION"]
    parent = ctx.get("parent")
    if ctx.get("mode") == "build":
        lines.append(
            "BUILD FOR THE LIBRARY. Write ONE reusable, tested module the team is missing -- or a better version of "
            "one that exists: a regime detector from greek fields (detect(df)), a signal (signal(df)), a filter or a "
            "sizing/risk rule. Check library_list and team_board first so it is not a duplicate. Save it with "
            "library_save (kind, a clear description, a short test that loads data and prints its output), fix it "
            "if the smoke test fails, run regime_map if it is a regime or signal module, then submit_candidate with "
            "a strategy that imports it -- the candidate is the module's evidence. Comment on modules you reused.")
        if parent:
            lines.append(f"The current leader is candidate {parent['seq']}; factoring its best idea into a module is a "
                         "good BUILD -- an entry FILTER that keeps its big winners and drops its scratch and big "
                         "losers is the best module of all. Its code:")
            lines.append("```python\n" + (parent.get("code") or "")[:7000] + "\n```")
            if parent.get("trade_review"):
                lines.append(parent["trade_review"])
    elif parent:
        standing = f"rank {parent['rank']}" if parent.get("rank") else "not ranked yet"
        who = f" by {parent['model']}" if parent.get("model") else ""
        lines.append(
            f"IMPROVE candidate {parent['seq']}{who} ({standing}, in-sample {_fmt(parent.get('in_sample_score'))}). "
            "Make ONE focused change you expect to generalise -- a better signal, a filter, a regime condition, "
            "position sizing (e.g. inverse-volatility or conviction sizing set at entry with ft.size) or a risk "
            "rule -- and keep what works. Its rationale: " + (parent.get("rationale") or "")[:600])
        if parent.get("problem"):
            lines.append(f"It scored well in-sample but is NOT RANKED: {parent['problem']} Fix that first -- it is "
                         "what stands between this idea and the leaderboard -- without losing what made it score.")
        if parent.get("diagnosis"):
            lines.append("What its result says (in-sample, computed by the harness): " + parent["diagnosis"])
        if parent.get("trade_review"):
            lines.append(parent["trade_review"])
            # "Require a condition where its trades earn more" made 8 filters: in-sample +0.96 on
            # average, holdout -1.48 (7 of 8 worse). A condition picked on the same trades it is
            # judged on is fitted to them; only one checked on trades it was not picked on means much.
            half = o["metric"].get("mid_cut") or "the middle of the in-sample period"
            lines.append(f"If you add a filter from this review, PICK it on the parent's trades before {half} and "
                         f"CHECK it on the trades after: keep it only if the kept trades beat the rest in BOTH "
                         "halves, and give both halves' averages in your rationale. A condition picked on all the "
                         "trades is fitted to them and fails the holdout; most of these listed ones are chance.")
        lines.append("```python\n" + (parent.get("code") or parent.get("answer") or "")[:9000] + "\n```")
    else:
        lines.append("EXPLORE: test an idea the team has not tested yet -- a mentor direction above, or a different "
                     "signal family, horizon or feature of the data. Do not repeat what already failed (RECENT ATTEMPTS, "
                     "the AVOID lessons). Starting from a teammate's candidate or module is welcome when it serves the "
                     "idea (a direction, practice or message that names one): read it with get_candidate / "
                     "library_get and pass the candidate's number as `parent` to submit_candidate. Start by looking "
                     "at the data if you need to.")
    lines += ["", "Steps: (1) team_board, library_list and the lessons -- learn from the team first; (2) team_post "
              "your plan to #planning, naming the teammate's candidate or library module you build on"
              + (" and the message(s) above you act on (reply_to=<number>)" if inbox else "")
              + (" and how it carries out YOUR COMMITMENTS" if ctx.get("commitments") else "")
              + "; (3) investigate with your tools (at most "
              f"{MAX_EXPERIMENTS} run_python experiments); (4) save reusable parts with library_save, and import "
              "modules that work (`from lib import <name>`) instead of copying their code; (5) call "
              "submit_candidate with the complete script, your hypothesis and `parent` = the candidate you started "
              "from, if any. If the script failed, read the error "
              f"and its hint, fix it and resubmit (up to {MAX_SUBMITS - 1} fixes). Then stop."]
    return "\n".join(lines)


MENTOR_IDLE_S = 60.0
MENTOR_MAX_DIRECTIONS = 3
MENTOR_MAX_FORECASTS = 2


def _j(x: Any, n: int = 1600) -> str:
    return json.dumps(x, default=str)[:n]


def mentor_prompt(brief: dict, inbox: list[dict]) -> str:
    """What the mentor reads: the team's memory as evidence, and what it must produce."""
    o = brief["objective"]
    lines = [
        "You are the MENTOR of a team of AI agents searching for a trading strategy. You do not write "
        "candidates; you think for the team. Your notes are read by every agent before its next iteration. "
        "The team learns fastest when it tests CONCEPTS with a clear hypothesis, not when it nudges thresholds "
        "(brute force overfits the in-sample period and teaches nothing). Everything below is in-sample; the "
        "holdout is hidden from you and from them.",
        "", f"OBJECTIVE: {o['title']}", o.get("description") or "",
        *(_mentor_task_lines(o) if _is_task(o) else [
            f"Metric: {brief['metric_label']} of daily returns; positions are marked to market by the harness on "
            f"{o['metric'].get('price_column')} with {o['metric'].get('cost_bps')} bps per unit of position change.",
            "", "TEAM HABITS over the last candidates (counts):", _j(brief.get("habits"), 800),
            "(results: 'edge given away by costs' = right direction but trades too often; 'points the wrong way' = "
            "flipping every position would score better; 'no edge' = the idea does not work. changes_vs_parent: "
            "'parameters only' = only numbers changed.)",
            "", "LEADERBOARD (ranked on consistency: the weaker of in-sample and the hidden holdout, times equity-curve "
            "smoothness; in-sample shown, with the harness's diagnosis):"]),
    ]
    lines += [f"- #{c['seq']} {c['model']} in-sample {c['in_sample']}: {c['rationale'][:220]} || {c.get('diagnosis') or ''}"
              for c in brief.get("leaderboard") or []]
    lines += ["", "RECENT ATTEMPTS (newest first):"]
    # The author on every line: without it the mentor guessed, and addressed its feedback to
    # "agent", "team", "ok", "lowk" or the wrong model -- names that reach nobody's inbox.
    lines += [f"- #{c['seq']} by {c.get('model') or '?'} {c['status']} in-sample {c['in_sample']} idea={c.get('idea_id')} change={c.get('change')} "
              f"forecasts={c.get('forecasts_used')}: {c['rationale'][:200]}"
              + (f" || {c['diagnosis'][:200]}" if c.get("diagnosis") else "")
              + (f" || failed: {c['problem']}" if c.get("problem") else "")
              for c in brief.get("recent") or []]
    lines += ["", "IDEAS SCOREBOARD (your earlier directions and what the candidates that tested them scored):"]
    lines += [f"- idea {i['id']} ({i['minutes_ago']} min ago): tried {i['tried']}, ran {i['ran']}, best {i['best_in_sample']} "
              f"(#{i['best_seq']}, {i['best_diagnosis']}), median {i['median_in_sample']}: {i['idea'][:260]}"
              for i in brief.get("ideas") or []] or ["- (none yet)"]
    lines += ["", "FORECAST SCOREBOARD (skill > 0 beats 'no change'; direction 0.5 = coin flip; lift_from_inputs > 0 "
              "means the input columns helped; helped = median in-sample of candidates using it minus those using "
              "no forecast):"]
    lines += [f"- {f['view']} [{f['model']}] series={f['series']} inputs={f['inputs']} h={f['horizon']}: "
              f"skill {_j(f['skill'], 300)}; used by {f['used_by']}, helped {f['helped']}"
              for f in (brief.get("forecasts") or [])[:20]] or ["- (none built yet)"]
    lines += ["", "FORECASTERS LOADED: " + _j(brief.get("forecasters"), 600)]
    if brief.get("field_scan"):
        lines += ["", "FIELD SCAN (information coefficient of each field with future returns):", _j(brief["field_scan"], 2500)]
    # Decile studies and explored forecast inputs: point the team at what is proven, away from what is flat.
    lines += _knowledge_lines(brief.get("deci_studies"), brief.get("forecast_inputs"))
    lines += ["", "TEAM LESSONS:"] + [f"- {x}" for x in (brief.get("lessons") or [])[:25]]
    if inbox:
        # Agents' plain acknowledgements ("accept") are not listed (Worker.inbox): replying to each
        # one made an endless back-and-forth that took the start of every iteration (10-01).
        lines += ["", "MESSAGES TO YOU (answer a question or a new point in `replies` with its number as reply_to; "
                      "an agent's REJECT of your feedback needs a reply only if its evidence is wrong or you have a "
                      "correction -- otherwise leave it, the thread is done):"]
        lines += [f"- [{m['seq']}] from {m['from']}"
                  + (f" ({m['feedback_reply'].upper()} of your feedback)" if m.get("feedback_reply") else "")
                  + f": {m['text']}" for m in inbox]
    lines += [
        "", "Reply with ONE JSON object and nothing else:",
        '{"directions": [{"idea": "...", "hypothesis": "why it should work, in market terms", '
        '"test": "the first concrete experiment and what result would FALSIFY it", "avoid": "the brute-force trap to avoid"}],',
        ' "coaching": "3-6 short lines to the whole team: what the evidence says they are doing wrong or should '
        'stop, which ideas/forecasts to build on or drop, citing numbers from above",',
        ' "replies": [{"candidate": <candidate number>, "text": "feedback for the agent who wrote it"}, '
        '{"reply_to": <message number>, "text": "your answer to a message above"}],',
        ' "forecasts": [{"column": "<series to forecast>", "inputs": ["<columns the model reads>"], "horizon": 6, '
        '"every": 0, "model": "<a loaded forecaster>", "why": "what building it would teach us"}]}',
        "`replies` reach one agent's inbox: give `candidate` (a number from the lists above) for feedback on a "
        "candidate -- it goes to the agent who wrote it -- or `reply_to` (a number from MESSAGES TO YOU) to answer a "
        "message. Do not invent names; advice for everyone belongs in `coaching`.",
        f"At most {MENTOR_MAX_DIRECTIONS} directions -- conceptually different from each other and from what failed; "
        f"at most {MENTOR_MAX_FORECASTS} forecasts, only where the scoreboard suggests one could help (prefer "
        "Chronos-2 with input columns; do not repeat a recipe already on the scoreboard). When a series looks "
        "forecastable, request it at a horizon the scoreboard does not have yet, so the team can see which "
        "horizons have skill and build a fan of the accurate ones (a direction vote across horizons, weighted by "
        "each horizon's direction accuracy). Be concrete and brief.",
    ]
    if _is_task(o):
        # A task candidate reads only the task server's rows: forecasts of project datasets don't reach it.
        lines[-4] = lines[-4].rstrip().rstrip(",") + "}"          # the reply shape, without "forecasts"
        lines[-3:] = [lines[-2], f"At most {MENTOR_MAX_DIRECTIONS} directions -- conceptually different from each "
                      "other and from what failed. Be concrete and brief."]
    return "\n".join(lines)


def _mentor_task_lines(o: dict) -> list[str]:
    """The metric and leaderboard frame for a team scored by a data/action MCP."""
    m = o.get("metric") or {}
    t = m.get("task_info") or {}
    act = t.get("action") or {}
    return [
        f"Task: '{m.get('task')}' served by the data/action MCP '{m.get('task_server')}'. Candidates report one "
        f"ACTION per row ({act.get('description') or act.get('kind') or 'see the task'}); the server manages the "
        f"actions and values them with '{m.get('value_function') or (t.get('score') or {}).get('name') or 'its score'}' "
        f"against the target `{m.get('target') or t.get('target')}`.",
        *[f"- {str(g.get('title') or 'Note').upper()}: " + " ".join(str(g.get("text") or "").split())
          for g in (t.get("guidance") or []) if isinstance(g, dict) and g.get("text")],
        "", "TEAM HABITS over the last candidates (counts):", "(not measured for task objectives: read the server's "
        "in-sample diagnostics in the attempts below)",
        "", "LEADERBOARD (ranked on consistency: the weaker of the in-sample and hidden holdout values; in-sample shown):"]


def parse_mentor(text: str) -> dict:
    """The mentor's JSON, from a reply that may wrap it in prose or code fences."""
    dec = json.JSONDecoder()
    start = (text or "").find("{")
    while start >= 0:
        try:
            obj, _ = dec.raw_decode(text[start:])
            if isinstance(obj, dict) and ("directions" in obj or "coaching" in obj):
                return obj
        except ValueError:
            pass
        start = text.find("{", start + 1)
    return {"directions": [], "coaching": (text or "").strip()[:3000], "replies": [], "forecasts": []}


# =======================================================================================
# Collaboration plumbing: who a message is for, what a script reused, what it built on
# =======================================================================================
# How far back a fresh agent (just started or restarted) reads its inbox. It was one hour:
# with a restart every ~25 minutes on 10-01, a teammate's message older than that was never
# shown to anyone.
INBOX_WINDOW_S = 6 * 3600.0
INBOX_TAIL = 500          # the board's maximum; #general alone gets ~20 posts an hour
INBOX_CAP = 8
COACHING_FRESH_S = 12 * 3600.0

# app/library.py's IMPORT_RE: the runner's own `from\s+lib\s+import\s+([\w\s,]+)` missed
# `from lib import x as y` and the parenthesised form, so most reuse went unrecorded.
_LIB_IMPORT_RE = re.compile(
    r"from\s+lib\s+import\s+\(([^)]*)\)"
    r"|from\s+lib\s+import\s+([^\n(#]+)"
    r"|from\s+lib\.(\w+)\s+import"
    r"|import\s+lib\.(\w+)"
)
_CITE_RE = re.compile(r"#\s?(\d{1,7})\b")
# "Building on candidate 75", "Improvement over candidate 75", "improve #94", "mirror #131".
_BUILT_ON_RE = re.compile(
    r"\b(?:build(?:s|ing)?\s+on|built\s+on|improv(?:e|es|ing)|improvement\s+(?:on|over|of|to)|extend(?:s|ing)?|"
    r"fix(?:es|ing)?|mirror(?:s|ing)?|start(?:s|ing)?\s+from|based\s+on|variant\s+of|on\s+top\s+of|"
    r"refin(?:e|es|ing))\s+(?:the\s+)?(?:leader(?:'s)?\s+)?(?:candidate\s*#?|#)(\d{1,6})\b", re.I)


def _lib_imports(code: str) -> list[str]:
    """Library module names a script imports (the same reading as app/library.py)."""
    names: set[str] = set()
    for m in _LIB_IMPORT_RE.finditer(code or ""):
        for clause in (m.group(1) or m.group(2) or "").split(","):
            name = clause.strip().split(" as ")[0].strip()
            if re.fullmatch(r"[a-z_][a-z0-9_]{0,47}", name):
                names.add(name)
        names.update(g for g in (m.group(3), m.group(4)) if g)
    return sorted(names)


def _cited(text: str) -> list[int]:
    """Message / candidate numbers a text cites as #N."""
    return [int(n) for n in _CITE_RE.findall(str(text or ""))]


def _built_on_ref(rationale: str) -> int | None:
    """The candidate a rationale says it starts from ("Building on candidate 75 ..."), if any."""
    m = _BUILT_ON_RE.search(str(rationale or "")[:1500])
    return int(m.group(1)) if m else None


def _short_name(model: Any) -> str:
    """'org/Model-X #2' -> 'model-x': the name the board's @mentions and `to` fields use."""
    return str(model or "").split("/")[-1].split(" #")[0].split(" (")[0].strip().lower()


def _int_or_none(v: Any) -> int | None:
    try:
        return int(str(v).strip().lstrip("#")) if v not in (None, "", False) else None
    except (TypeError, ValueError):
        return None


def _board_reply_to(e: dict) -> int | None:
    return _int_or_none(e.get("reply_to") if e.get("reply_to") is not None else (e.get("meta") or {}).get("reply_to"))


def _addressed_to(e: dict, model: str) -> bool:
    """A board message is for `model`: `meta.to` names it (full or short name, or one of its
    agents "<model> #2"), or the text @mentions its short name."""
    short = _short_name(model)
    to = (e.get("meta") or {}).get("to")
    if to and (to == model or _short_name(to) == short):
        return True
    return f"@{short}" in str(e.get("content", "")).lower()


def _feedback_depth(meta: dict) -> int:
    """How deep in a feedback thread an agent's answer is: 1 answers a message, 2 answers a
    reply to one of its answers (answers posted before the depth was recorded count as 1)."""
    d = _int_or_none(meta.get("feedback_depth"))
    return d if d is not None and d > 0 else (1 if meta.get("feedback_reply") else 0)


def _inbox_entries(entries: list[dict], model: str, since: float, now: float | None = None, *,
                   acks: bool = True) -> list[dict]:
    """The messages `model` should answer: addressed to it, or replies to one of its own
    messages, posted after `since` by someone else -- minus those its model already answered
    (two agents share a model; a restart re-reads the window). `acks=False` (the mentor) also
    leaves out agents' plain "accept" answers: an acknowledgement needs no reply, and replying
    to each one made an endless mentor <-> agent back-and-forth (10-01).

    Each entry may carry, for answer_feedback: `feedback_reply` (the message is an agent's
    answer to feedback, with this verdict), `thread_depth` (it replies to one of this model's
    feedback answers, at that depth) and `candidate_by` ({"id", "agent"}: which of this
    model's agents ran the candidate it is about -- from that agent's collaboration record)."""
    now = time.time() if now is None else now
    own = {e["seq"]: e for e in entries if e.get("author") == model}
    mine = set(own)
    answered = {_board_reply_to(e) for e in entries if e.get("author") == model}
    answered |= {n for e in entries if e.get("author") == model for n in (e.get("meta") or {}).get("answers") or []}
    ran: dict[int, dict] = {}
    for e in own.values():
        c = ((e.get("meta") or {}).get("collab") or {}).get("candidate")
        if isinstance(c, int) and e.get("author_id"):
            ran[c] = {"id": e["author_id"], "agent": (e.get("meta") or {}).get("agent") or model}
    out = []
    for e in entries:
        if e["ts"] <= since or e.get("author") == model or e["seq"] in answered:
            continue
        rt = _board_reply_to(e)
        if _addressed_to(e, model) or (rt is not None and rt in mine):
            meta = e.get("meta") or {}
            verdict = meta.get("feedback_reply")
            if verdict == "accept" and not acks:
                continue
            cand = meta["candidate_seq"] if isinstance(meta.get("candidate_seq"), int) else None
            depth = _feedback_depth(own[rt].get("meta") or {}) if rt in mine else 0
            out.append({"seq": e["seq"], "from": e["author"], "channel": e["channel"],
                        "minutes_ago": int((now - e["ts"]) / 60), "text": str(e["content"])[:600],
                        **({"candidate": cand} if cand is not None else {}),
                        **({"reply_to": rt} if rt in mine else {}),
                        **({"feedback_reply": verdict} if verdict in ("accept", "reject", "question") else {}),
                        **({"thread_depth": depth} if depth else {}),
                        **({"candidate_by": ran[cand]} if cand in ran else {})})
    return out[-INBOX_CAP:]


def _coaching_from(entries: list[dict], now: float | None = None) -> dict | None:
    """The newest mentor coaching on the board (meta.coaching, or the COACHING: part of a
    mentor-notes post), if it is fresh and reads as prose rather than a failed JSON reply."""
    now = time.time() if now is None else now
    for e in reversed(entries):
        meta = e.get("meta") or {}
        if not meta.get("mentor") or now - e["ts"] > COACHING_FRESH_S:
            continue
        text = str(meta.get("coaching") or "")
        if not text and "COACHING:\n" in str(e.get("content", "")):
            text = str(e["content"]).split("COACHING:\n", 1)[1]
        text = text.strip()
        if text and not text.startswith(("{", "[", "```")):
            return {"by": e["author"], "minutes_ago": int((now - e["ts"]) / 60), "text": text[:2500]}
    return None


def _mentor_reply(pid: str, author: str, oid: str, r: Any, brief: dict, inbox: list[dict]) -> dict | None:
    """One of the mentor's `replies` as a board message, addressed by the runner rather than by
    the mentor's guess. From 09-30 to 10-01 its replies went "to" agent / team / ok / lowk /
    "team member who ran candidate 131" / the wrong model, with a CANDIDATE number as reply_to
    (so the thread pointed at an unrelated board message): feedback nobody's inbox matched.
    Now `candidate` (or a reply_to that is a candidate number, not a message in its inbox)
    goes to that candidate's author; reply_to answers a message in its inbox, to its sender;
    a name is kept only if it is a teammate's. None: nothing to post."""
    if not isinstance(r, dict) or not str(r.get("text") or "").strip():
        return None
    authors = {int(c["seq"]): c.get("model") for c in [*(brief.get("leaderboard") or []), *(brief.get("recent") or [])]
               if isinstance(c, dict) and str(c.get("seq", "")).isdigit()}
    asked = {m["seq"]: m for m in inbox}
    rt, cand = _int_or_none(r.get("reply_to")), _int_or_none(r.get("candidate"))
    to = ""
    if rt is not None and rt in asked:
        to = asked[rt]["from"]
    else:
        if rt is not None and cand is None and rt in authors:
            cand = rt
        rt = None
    if cand is not None and authors.get(cand):
        to = authors[cand]
    if not to and r.get("to"):
        known = {_short_name(m): m for m in [*authors.values(), *(m["from"] for m in inbox)] if m}
        to = known.get(_short_name(r["to"]), "")
    if to == author:
        to = ""
    meta: dict = {"objective_id": oid, "team": True}
    if to:
        meta["to"] = to
    if cand is not None and cand in authors:
        meta["candidate_seq"] = cand
    prefix = (f"@{to.split('/')[-1]} " if to else "") + (f"[candidate #{cand}] " if "candidate_seq" in meta else "")
    body = {"project_id": pid, "channel": "team", "author": author, "kind": "chat",
            "content": prefix + str(r["text"])[:3000], "meta": meta}
    if rt is not None:
        body["reply_to"] = meta["reply_to"] = rt
    return body


# =======================================================================================
# Answering feedback before the work
# =======================================================================================
# Agents read the mentor's per-candidate coaching and their teammates' messages in the brief
# and almost never answered them (Muse-Glimmer: 67 unanswered, 0 answered by 10-01) -- nor,
# mostly, acted on them. So before an explore/build/improve iteration the agent's own model
# answers each one in a separate short call: accept (the concrete change it will make this
# iteration) or reject (why, with evidence). The replies go on the board to the sender, and
# the accepted changes become the iteration's commitments in its prompt.
ANSWER_FEEDBACK = os.getenv("FREESWARM_ANSWER_FEEDBACK", "1").strip().lower() not in ("0", "false", "no", "off")
# 10-01 19:04-19:30: 1 of 10 answering steps hit the old 300 s cap -- with 6 messages, which
# were 3 follow-ups duplicated because both agents of one model had answered the same message.
FEEDBACK_MAX_S = float(os.getenv("FREESWARM_FEEDBACK_MAX_S", "480"))   # wall time of the answering call
FEEDBACK_MAX_MESSAGES = 4            # newest first; the rest stay in the brief's MESSAGES TO YOU
FEEDBACK_MAX_AGE_S = 24 * 3600.0     # older messages are skipped, not answered
COMMITMENTS_CHARS = 1500             # the YOUR COMMITMENTS section of the iteration prompt
FEEDBACK_VERDICTS = ("accept", "reject", "question")
# A thread ends at agent answer -> reply -> agent answer: a reply to an answer at this depth
# is not answered (the mentor replied to every answer, so each iteration began by answering it).
FEEDBACK_MAX_DEPTH = 2
# One agent answers a message: it claims "fbclaim:<seq>" on the board's blackboard by
# compare-and-set first. Two agents share each model, and both answered the same messages
# within a minute (10-01 19:02, 19:18) -- then the mentor replied to both answers. A claim
# expires after FEEDBACK_CLAIM_TTL_S so a crashed agent's messages are answered by its twin.
FEEDBACK_CLAIM_TTL_S = 1800.0
# A message about a candidate waits this long for the agent that ran the candidate to claim it.
FEEDBACK_ROUTE_WAIT_S = 1800.0


# A model measured slow gets more time, up to FEEDBACK_HARD_MAX_S, and fewer messages when even
# that is not enough (see _feedback_budget).
FEEDBACK_HARD_MAX_S = float(os.getenv("FREESWARM_FEEDBACK_HARD_MAX_S", "900"))
FEEDBACK_TOKENS_PER_MESSAGE = 250    # one reply of one or two sentences, in the JSON list
FEEDBACK_SLACK = 1.5                 # the wait allowed over the expected reply time


def _feedback_budget(model: str, n: int) -> tuple[int, float]:
    """(how many messages to answer, at most `n`; the answering call's wall-time cap) for
    `model` at its measured speed: FEEDBACK_MAX_S unless the expected reply needs longer,
    never over FEEDBACK_HARD_MAX_S, with fewer messages when even that is too short."""
    def need(k: int) -> float | None:
        s = _expected_reply_s(model, k * FEEDBACK_TOKENS_PER_MESSAGE + REASONING_ROOM, kind="feedback")
        return None if s is None else FEEDBACK_SLACK * s
    if need(n) is None:
        return n, FEEDBACK_MAX_S
    k = n
    while k > 1 and need(k) > FEEDBACK_HARD_MAX_S:
        k -= 1
    return k, min(FEEDBACK_HARD_MAX_S, max(FEEDBACK_MAX_S, need(k)))


def _feedback_closed(m: dict) -> str | None:
    """Why inbox message `m` needs no answer at all, or None."""
    if m.get("feedback_reply") == "accept":
        return "an acknowledgement (accept) -- it needs no answer"
    if (m.get("thread_depth") or 0) >= FEEDBACK_MAX_DEPTH:
        return "the thread is closed: it replies to your answer to a reply"
    return None


def _feedback_due(inbox: list[dict]) -> tuple[list[dict], list[dict]]:
    """(the inbox messages to answer now, those skipped as older than FEEDBACK_MAX_AGE_S): the
    newest FEEDBACK_MAX_MESSAGES fresh ones that need an answer, oldest first."""
    fresh = [m for m in inbox if (m.get("minutes_ago") or 0) * 60 <= FEEDBACK_MAX_AGE_S and not _feedback_closed(m)]
    old = [m for m in inbox if (m.get("minutes_ago") or 0) * 60 > FEEDBACK_MAX_AGE_S]
    return fresh[-FEEDBACK_MAX_MESSAGES:], old


def _feedback_route(m: dict, agent_id: str | None, my_candidates: set) -> str | None:
    """None if this agent may claim message `m`; else who should answer it: the agent of this
    model that ran the candidate it is about, while the message is fresh enough to wait for it."""
    cand, by = m.get("candidate"), m.get("candidate_by") or {}
    if cand is None or cand in my_candidates or not by.get("id") or by.get("id") == agent_id:
        return None
    if (m.get("minutes_ago") or 0) * 60 >= FEEDBACK_ROUTE_WAIT_S:
        return None
    return str(by.get("agent") or "the agent that ran it")


def _candidate_line(ctx: dict, seq: Any) -> str | None:
    """The brief's one-line standing of candidate `seq`, if the context already holds it."""
    pool = [ctx.get("parent") or {}, *(ctx.get("leaderboard") or []), *(ctx.get("promising") or []),
            *(ctx.get("recent") or [])]
    c = next((x for x in pool if isinstance(x, dict) and x.get("seq") == seq), None)
    if c is None:
        return None
    parts = [f"candidate {seq}" + (f" by {c['model']}" if c.get("model") else "")]
    if c.get("status"):
        parts.append(str(c["status"]))
    parts.append(f"in-sample {_fmt(c.get('in_sample_score'))}")
    parts.append(f"rank {c['rank']}" if c.get("rank") else "not ranked")
    if c.get("lookahead"):
        parts.append(f"look-ahead {c['lookahead']}")
    line = ", ".join(parts)
    if c.get("problem"):
        line += f" -- {str(c['problem'])[:200]}"
    return line


def feedback_prompt(obj: dict, ctx: dict, due: list[dict], agent: str) -> str:
    lines = [f"You are {agent}, one agent in a research swarm working on \"{obj.get('title') or obj.get('id')}\". "
             "Before you start this iteration, answer the feedback you were sent. Answering it -- and then doing "
             "what you accept -- is part of how you and the team learn.", "", "MESSAGES TO YOU (oldest first):"]
    for m in due:
        about = ""
        if m.get("candidate"):
            standing = _candidate_line(ctx, m["candidate"])
            about = f", about your candidate {m['candidate']}" + (f" [{standing}]" if standing else "")
        lines.append(f"#{m['seq']} from {m['from']} ({m.get('minutes_ago', 0)} min ago{about}"
                     + (f", replying to your #{m['reply_to']}" if m.get("reply_to") else "") + f"): {m['text']}")
    lines += ["", "For EACH message, reply in one or two sentences:",
              "- accept: the concrete change you will make THIS iteration (which script or candidate, which rule, "
              "parameter or data);",
              "- reject: why not, with evidence (a number, a result, a candidate that showed it);",
              "- question: only if you cannot act at all without an answer -- ask exactly what you need.",
              "Reply with ONLY a JSON list, one object per message:",
              '[{"reply_to": <message number>, "verdict": "accept"|"reject"|"question", "text": "..."}]']
    return "\n".join(lines)


def _json_dict_list(text: str) -> list | None:
    """The last JSON list of objects in `text` (a reasoning model may draft one before its answer)."""
    text = text or ""
    found = None
    got = _json_array(text)
    if isinstance(got, list) and got and all(isinstance(x, dict) for x in got):
        found = got
    dec = json.JSONDecoder()
    for m in re.finditer(r"\[", text):
        try:
            v, _ = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            found = v
    return found


def _verdict(v: Any) -> str | None:
    v = str(v or "").strip().lower()
    if v.startswith("accept") or v in ("agree", "yes"):
        return "accept"
    if v.startswith(("reject", "declin", "disagree")) or v == "no":
        return "reject"
    if v.startswith(("question", "ask", "clarif")):
        return "question"
    return None


def _parse_feedback_replies(text: str, seqs: set) -> list[dict]:
    """[{"reply_to", "verdict", "text"}] from the model's reply: one per message number in
    `seqs` (the first wins), with a usable verdict and some text."""
    out: list[dict] = []
    seen: set = set()
    for r in _json_dict_list(text) or []:
        n = _int_or_none(r.get("reply_to"))
        if n is None or n not in seqs or n in seen:
            continue
        body = " ".join(str(r.get("text") or "").split())
        verdict = _verdict(r.get("verdict"))
        m = re.match(r"(accept(?:ed)?|reject(?:ed)?|question)\s*[:\-]+\s*", body, re.I)
        if m:
            verdict = verdict or _verdict(m.group(1))
            body = body[m.end():]
        if verdict is None or not body:
            continue
        seen.add(n)
        out.append({"reply_to": n, "verdict": verdict, "text": body[:600]})
    return out


def _feedback_reply_body(pid: str, model: str, agent: str, agent_id: str | None, oid: str, m: dict,
                         r: dict) -> dict:
    """A reply to inbox message `m` as a board post, linked the way team_post links one: to the
    sender, reply_to and meta.answers = the message, so record_collaboration counts it answered
    and the sender's inbox shows it."""
    n, to = r["reply_to"], m["from"]
    meta = {"objective_id": oid, "team": True, "agent": agent, "to": to, "reply_to": n, "answers": [n],
            "feedback_reply": r["verdict"],
            # 1: answers a message; 2: answers a reply to one of its answers (see FEEDBACK_MAX_DEPTH)
            "feedback_depth": int(m.get("thread_depth") or 0) + 1}
    content = (f"@{to.split('/')[-1]} re #{n}" + (f" [candidate #{m['candidate']}]" if m.get("candidate") else "")
               + f" -- {r['verdict']}: {r['text']}")
    return {"project_id": pid, "channel": "team", "author": model, "kind": "chat", "content": content,
            "meta": meta, "reply_to": n, **({"author_id": agent_id} if agent_id else {})}


def _commitment_lines(commitments: list[dict] | None) -> list[str]:
    """YOUR COMMITMENTS for the iteration prompt: accepted changes first, then rejections and
    questions, within COMMITMENTS_CHARS."""
    if not commitments:
        return []
    head = ("YOUR COMMITMENTS THIS ITERATION (from the feedback you just answered -- your replies are on the "
            "board; do what you accepted, and let the candidate show it):")
    label = {"accept": "ACCEPTED", "reject": "REJECTED", "question": "ASKED"}
    out, used = [head], len(head)
    ordered = sorted(commitments, key=lambda c: FEEDBACK_VERDICTS.index(c["verdict"]))
    for i, c in enumerate(ordered):
        who = str(c.get("from") or "?").split("/")[-1]
        line = (f"- {label[c['verdict']]} #{c['reply_to']} ({who}"
                + (f", candidate {c['candidate']}" if c.get("candidate") else "") + f"): {c['text'][:400]}")
        if used + len(line) > COMMITMENTS_CHARS:
            out.append(f"- (+{len(ordered) - i} more on the board)")
            break
        out.append(line)
        used += len(line) + 1
    return ["", *out]


class Worker(threading.Thread):
    def __init__(self, project: dict, model: str, stop: threading.Event, sync, slot: int = 0,
                 role: str = "search") -> None:
        super().__init__(name=f"swarm-{project['slug']}-{model}-{role}-{slot}", daemon=True)
        self.project = project
        self.model = model
        # "search" writes candidates; "mentor" thinks for the team (app/mentor.py).
        self.role = role
        # Several agents may share one hosted model (see main()); each needs its own name on
        # the board, which keys identity by name. Candidates and messages still carry the model.
        self.slot = slot
        self.agent_name = (f"{model} (mentor)" if role == "mentor"
                           else model if slot == 0 else f"{model} #{slot + 1}")
        self.tier = infer_tier(model)
        self._stop = stop
        self._sync = sync  # () -> (llms, forecasters) -- the supervisor's latest view
        self.agent_id: str | None = None
        self._last_beat = 0.0
        self._status = "idle"
        self.retired = threading.Event()
        # An iteration (model thinking, sandbox runs, a minute of scoring) outlasts the board's
        # 90 s staleness window, so a heartbeat sent only when work starts made every busy agent
        # look offline mid-iteration. This keeps the last status fresh while the work runs.
        threading.Thread(target=self._keepalive, name=f"beat-{self.agent_name}", daemon=True).start()
        self._objectives: list[dict] = []
        self._obj_checked = 0.0
        self._obj_turn = -1
        self._inbox_since = 0.0
        # Set when the console refused this agent's own model for today's spending limit;
        # run() then waits BUDGET_BACKOFF_S instead of starting the next iteration.
        self._budget_block: str | None = None
        self._budget_streak = 0  # back-to-back backoffs; only the first is posted to the board
        # What this agent is doing right now (an iteration, a mentor pass, a task), or None
        # between units of work. The supervisor's drain waits until every agent is None.
        self.busy: str | None = None
        self.busy_since = 0.0

    @property
    def pid(self) -> str:
        return self.project["id"]

    # -- board plumbing -----------------------------------------------------------------
    def register(self) -> bool:
        try:
            doc = request(BOARD, "/mb/agents/register", {
                "project_id": self.pid, "name": self.agent_name,
                "role": f"{self.tier}-tier model", "model": self.model,
                "capabilities": ["chat", "code", "tools", self.tier],
            })
        except RuntimeError as exc:
            log(f"{self.project['slug']}/{self.model}: register failed: {exc}")
            return False
        self.agent_id = doc.get("agent_id")
        log(f"{self.project['slug']}/{self.agent_name}: registered {self.agent_id} (tier {self.tier})")
        return bool(self.agent_id)

    def beat(self, status: str, force: bool = False) -> None:
        now = time.time()
        if not force and status == "idle" and now - self._last_beat < HEARTBEAT_S:
            return
        self._last_beat = now
        self._status = status
        try:
            request(BOARD, f"/mb/agents/{self.agent_id}/heartbeat", {"status": status})
        except RuntimeError as exc:
            log(f"{self.model}: heartbeat failed ({exc}); re-registering")
            self.agent_id = None

    def _keepalive(self) -> None:
        while not self._stop.wait(HEARTBEAT_S):
            if self.retired.is_set():
                return
            if self.agent_id and self._status != "idle" and time.time() - self._last_beat >= HEARTBEAT_S:
                self.beat(self._status, force=True)

    def say(self, channel: str, kind: str, content: str, meta: dict | None = None) -> None:
        # channel is stored WITHOUT '#'; kind must be one of the board's literals.
        try:
            request(BOARD, "/mb/messages", {
                "project_id": self.pid, "channel": channel, "author": self.model,
                "author_id": self.agent_id, "kind": kind, "content": content, "meta": meta or {},
            })
        except RuntimeError as exc:
            log(f"{self.model}: could not post to {channel}: {exc}")

    def claim(self) -> dict | None:
        try:
            doc = request(BOARD, "/mb/tasks/claim", {
                "project_id": self.pid, "agent_id": self.agent_id, "lease_s": LEASE_S, "tier": self.tier,
            })
        except RuntimeError as exc:
            log(f"{self.model}: claim failed: {exc}")
            return None
        return doc.get("task")

    # -- generation ------------------------------------------------------------------------
    def _generate(self, payload: dict) -> dict:
        """POST a chat completion. A spending-limit refusal of THIS agent's model is noted
        so run() backs off, then re-raised like any other failure."""
        timeout = getattr(self, "_gen_timeout", None) or GENERATION_TIMEOUT_S
        # A step with a wall-time cap over all its requests (answer_feedback sets _gen_deadline).
        deadline = getattr(self, "_gen_deadline", None)
        if deadline is not None:
            left = deadline - time.time()
            if left < 5:
                raise RuntimeError("no time left before this step's deadline")
            timeout = min(timeout, int(left))
        act = _act(self)  # the agent inspector: what was asked, tokens, what came back
        act.chat_start(payload)
        t0 = time.time()
        try:
            # A repair's wait is capped by its deadline (_repair_code sets _gen_timeout).
            r = request(CONTROL_PLANE, "/v1/chat/completions", payload, timeout=timeout)
        except RuntimeError as exc:
            act.chat_done(payload, None, str(exc))
            if payload.get("model") == self.model and _spending_limited(str(exc)):
                self._budget_block = str(exc)
            raise
        act.chat_done(payload, r)
        if isinstance(r, dict):
            _note_speed(payload.get("model"), (r.get("usage") or {}).get("completion_tokens"), time.time() - t0)
        self._budget_streak = 0
        return r

    def hold_for_budget(self) -> None:
        """Today's external budget refused this agent: wait BUDGET_BACKOFF_S, quietly.

        Before, every iteration failed at once and posted an error and a thought, so three
        agents on one hosted model wrote ~2000 board messages an hour until midnight. One
        message per streak of backoffs now; the console's plan normally retires the agent
        within a resync anyway (swarm_policy moves spent models out of `search`), which ends
        the wait at once.
        """
        reason, self._budget_block = self._budget_block or "", None
        self._budget_streak += 1
        mins = max(1, round(BUDGET_BACKOFF_S / 60))
        log(f"{self.agent_name}: spending limit reached; pausing {mins} min")
        if self._budget_streak == 1:
            detail = reason.split(": ", 1)[-1] if " -> 429: " in reason else reason
            self.say("errors", "system", f"Paused for {mins} min -- {detail[:600]}",
                     {"model": self.model, "budget": True})
        until = time.time() + BUDGET_BACKOFF_S
        while time.time() < until and not self._stop.is_set() and not self.retired.is_set():
            if self.agent_id:
                self.beat("blocked", force=True)
            self._stop.wait(min(HEARTBEAT_S, max(0.0, until - time.time())))

    # -- the tool loop ------------------------------------------------------------------
    def converse(self, messages: list[dict], tools: list[dict], call, *, tag: dict,
                 max_rounds: int = MAX_TOOL_ROUNDS, done=lambda: False,
                 final_prompt: Any = "Tool budget used up. Give your final answer now from what you have.",
                 nudge=lambda text: None,
                 final_tools: Any = None,
                 focus=None,
                 text_call=None,
                 force_tool: bool = False,
                 ) -> tuple[bool, str, dict]:
        """Run the model with tools until it answers (or `done()` says the job is finished).

        `messages` is extended in place, so a caller can continue the conversation after.
        `tag` is attached to every board post (task_id / objective_id).

        Steering for a turn that must end in a particular call (an iteration's submission):
        `final_prompt` (str, or a callable returning one) is what the last round says;
        `final_tools` (names, or a callable returning them) are the tools that last round offers
        -- by default none -- and the only ones it runs, a call written as text included;
        `focus()` returns (tool names, prompt) when a round must be restricted to those tools
        (the prompt is said once), else None; `text_call(text)` turns a reply written as text
        into a (tool, args) call when that reply would otherwise end the turn; `force_tool`
        restricts every round to the given tools. A restricted round with ONE tool asks for it
        with a named tool_choice.
        """
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "tool_calls": 0, "rounds": 0}
        all_tools = list(tools)
        focus_said: set[str] = set()
        core_tools = [t for t in tools if "__" not in t["function"]["name"]]
        tool_names = {t["function"]["name"] for t in tools}
        # Each tool's declared parameters, to read arguments the way the schema means them.
        schemas = {t["function"]["name"]: ((t["function"].get("parameters") or {}).get("properties") or {})
                   for t in tools}
        # One retry per turn for a model that spends its whole output budget thinking (a
        # reasoning model such as DeepSeek-V4): told to act, with the rest of the window to do it.
        length_retry = {"used": False, "boost": False}
        # Same-args, same-error re-issues: hash -> {"name", "err", "count"}. The first attempt
        # is executed as normal; every duplicate is intercepted (see the tool loop) instead of
        # burning another round on the same failure. `_intercepted` is the total number of
        # duplicates blocked this converse, and past REPEAT_TOOL_LIMIT we end the iteration.
        failed_calls: dict[str, dict] = {}
        intercepted = 0

        for rnd in range(max_rounds + 1):
            # Switching the swarm off, or retiring this agent, has to be felt inside a turn
            # as well as between turns. An agent turn is many rounds of thinking and tool
            # calls and can run for minutes; checking only at the top of the worker loop
            # meant "swarm off" left every agent generating until it happened to finish,
            # which reads as the switch doing nothing.
            if self._stop.is_set() or self.retired.is_set():
                log(f"{self.model}: stopping mid-turn (round {rnd})")
                reason = ("retired: model unloaded or worker replaced"
                          if self.retired.is_set() else "stopped: swarm halted")
                _act(self).end("interrupted", reason)
                return False, "", usage
            final_round = rnd == max_rounds or not tools
            if rnd == max_rounds and tools:
                said = final_prompt() if callable(final_prompt) else final_prompt
                messages.append({"role": "user", "content": said})
            # Tools this round is restricted to (None: no restriction).
            allowed: set[str] | None = None
            if final_round:
                if final_tools is not None and tools:
                    allowed = set(final_tools() if callable(final_tools) else final_tools) & tool_names
            else:
                f = focus() if focus is not None else None
                if f:
                    allowed = set(f[0]) & tool_names
                    if f[1] and f[1] not in focus_said:
                        focus_said.add(f[1])
                        messages.append({"role": "user", "content": f[1]})
                elif force_tool:
                    allowed = set(tool_names)
            restricted = [t for t in all_tools if t["function"]["name"] in allowed] if allowed else None
            # What may run this round: the restriction, or nothing at all in a final round that
            # was given final_tools none of which exist (a call written as text included).
            gate = allowed
            if gate is None and final_round and final_tools is not None:
                gate = set()
            result = None
            choice_dropped = False
            # Three independent retry budgets for one round: context overflow (compact and
            # resend, up to 3 times), provider rate limit (wait the hinted time and resend
            # the same prompt), malformed tool call (tell the model and let it try again).
            overflows = waits = bad_calls = 0
            while True:
                ctx = context_for(self.model)
                scale = _scale.get(self.model, 1.0)
                offered = restricted if restricted else [] if final_round else tools
                # Fit the prompt to the window (in calibrated tokens), leaving room to answer.
                offered = _compact(messages, offered, max(1024, ctx - ANSWER_RESERVE),
                                   restricted or ([] if final_round else core_tools), scale)
                if not final_round and restricted is None:
                    tools = offered
                raw = _est_tokens(messages, offered)  # uncalibrated, for learning the ratio
                est = int(raw * scale)
                if overflows > 0 and est > ctx:
                    # Even fully compacted it cannot fit: the instructions and tool definitions
                    # alone exceed the window. Resending would be refused identically, so stop
                    # and say what actually fixes it.
                    floor = _est_tokens(messages[:2], offered, scale)
                    return False, (
                        f"{self.model}'s context window is {ctx} tokens, too small for an agent: "
                        f"the instructions and tool definitions alone take ~{floor}, before any "
                        f"data. Unload it and reload with more KV pages (32768 or more) on the "
                        f"Models page -- 64K context costs about 1-2 GiB of VRAM."
                    ), usage
                budget = MAX_TOKENS * 2 if length_retry["boost"] else MAX_TOKENS
                # `available` is the room left in the window after the prompt. When the ctx we
                # think we have is wrong (e.g. an external model whose real 128K window we hadn't
                # populated), this can go negative -- the OLD floor of 256 tokens then sent the
                # request anyway, the provider accepted it (its real window is huge), and the
                # answer or tool call came back cut off. MIN_OUTPUT is the new floor: below it,
                # a chat request cannot produce a full tool call or a coherent reply, so we
                # would rather have the provider correct us via an overflow refusal (which the
                # retry loop below already handles) than silently truncate. `provider_cap` is
                # the provider's own hard limit on generated tokens (Groq caps Qwen3.8-27b at
                # 16,384; a request past that is 400'd).
                available = ctx - est - 128
                provider_cap = max_output_for(self.model) or budget
                max_tokens_out = min(provider_cap, budget, max(MIN_OUTPUT, available))
                payload = {
                    "model": self.model, "messages": messages, "stream": False,
                    "max_tokens": max_tokens_out,
                }
                if offered:
                    payload["tools"] = offered
                    if (restricted and len(offered) == 1 and not choice_dropped
                            and self.model not in _NO_TOOL_CHOICE):
                        # The one call this round is for (an OpenAI-style named tool_choice:
                        # FreeToken shows the model only that tool; Groq/OpenRouter force it).
                        payload["tool_choice"] = {"type": "function",
                                                  "function": {"name": offered[0]["function"]["name"]}}
                try:
                    result = self._generate(payload)
                    break
                except RuntimeError as exc:
                    err = str(exc)
                    if (payload.get("tool_choice") and re.search(r"\b(400|422)\b", err)
                            and _parse_overflow(err) is None and _bad_tool_call(err) is None
                            and _rate_limit_wait(err) is None):
                        # A server that does not take a named tool_choice: the restricted tool
                        # list alone still says what to call. Not asked again for this model.
                        choice_dropped = True
                        _NO_TOOL_CHOICE.add(self.model)
                        log(f"{self.model}: tool_choice refused ({err[:160]}); resending without it")
                        continue
                    wait = _rate_limit_wait(err)
                    if wait is not None and waits < RATE_LIMIT_RETRIES:
                        # Provider rate limit (not our budget): the prompt and every paid round
                        # before it are still good -- wait and resend instead of discarding them.
                        waits += 1
                        log(f"{self.agent_name}: provider rate limit; retrying in {wait:.1f}s "
                            f"({waits}/{RATE_LIMIT_RETRIES})")
                        if self._stop.wait(wait) or self.retired.is_set():
                            reason = ("retired: model unloaded or worker replaced"
                                      if self.retired.is_set() else "stopped: swarm halted")
                            _act(self).end("interrupted", reason)
                            return False, "", usage
                        continue
                    failed = _bad_tool_call(err)
                    if failed is not None:
                        # The provider rejected the whole reply as a bad tool call (Groq's 400
                        # tool_use_failed): its parser took the model's output for a tool call
                        # whose arguments did not match any tool's schema. The model still
                        # PRODUCED something (`failed_generation`) -- prose it wanted to say, or
                        # a tool call in a syntax Groq could not parse. Never drop it silently.
                        text = failed or ""
                        recovered = _text_tool_calls(text, tool_names) if (offered and text) else []
                        if recovered:
                            # A tool call written in text: pretend it came back on the wire so
                            # the normal path runs it and the round's work counts. `usage`
                            # stays empty (the 400 had no billed tokens).
                            result = {"choices": [{"finish_reason": "tool_calls", "message": {
                                "content": "", "tool_calls": [
                                    {"id": f"salvage_{rnd}_{i}", "type": "function",
                                     "function": {"name": n, "arguments": json.dumps(a)}}
                                    for i, (n, a) in enumerate(recovered)]}}], "usage": {}}
                            log(f"{self.agent_name}: Groq rejected reply as bad tool call; recovered "
                                f"{len(recovered)} call(s) from failed_generation "
                                f"({', '.join(n for n, _ in recovered)})")
                            break
                        if offered and bad_calls < BAD_TOOL_CALL_RETRIES:
                            bad_calls += 1
                            # Preserve the rejected prose as the assistant's turn so the retry
                            # sees what it already said and does not repeat itself.
                            if text.strip():
                                messages.append({"role": "assistant", "content": text[:6000]})
                            snippet = f"\nYour rejected output began: {text[:400]}" if text else ""
                            messages.append({"role": "user", "content": (
                                "Your last reply was rejected by the provider (tool_use_failed): its "
                                "parser took the output for a tool call whose arguments did not match "
                                "any tool's schema. If you meant to call a tool, emit exactly ONE valid "
                                "tool call whose arguments are a JSON object matching that tool's "
                                "parameters -- no text around it, no XML or markdown wrappers, and do "
                                "not reference function-call syntax in prose. If your work here is done, "
                                "answer in plain prose without referring to function names." + snippet)})
                            log(f"{self.agent_name}: malformed tool call rejected by the provider; "
                                f"asking again ({bad_calls}/{BAD_TOOL_CALL_RETRIES})")
                            continue
                        # Retries used up, or no tools offered this round: rather than throw the
                        # whole iteration away, treat the rejected prose as the model's text
                        # answer. The outer loop then delivers it (or nudges it) normally.
                        if text.strip():
                            log(f"{self.agent_name}: tool_use_failed after {bad_calls} retries -- "
                                f"keeping the rejected prose as the assistant's answer")
                            result = {"choices": [{"finish_reason": "stop", "message": {
                                "content": text[:16000], "tool_calls": []}}], "usage": {}}
                            break
                        # Nothing to salvage at all: fall through to the generic failure path.
                    over = _parse_overflow(err)
                    if over is None or overflows == 3:
                        return False, f"generation failed: {exc}", usage
                    overflows += 1
                    actual, limit = over
                    # The engine told us both its window AND how big this prompt really was.
                    # Learn the true ratio (plus a margin) so the next compaction aims correctly.
                    _context[self.model] = limit
                    _learned.add(self.model)
                    _scale[self.model] = max(scale, actual / max(1, raw) * 1.1)
                    log(f"{self.model}: prompt was {actual} tokens for a {limit}-token window "
                        f"(estimated {est}); recalibrated x{_scale[self.model]:.2f}, compacting and retrying")
                    self.say("errors", "system",
                             f"Context full: prompt was {actual} tokens, window is {limit}. "
                             "Trimming older tool results and retrying.",
                             {**tag, "context": limit, "prompt_tokens": actual})
            u = result.get("usage") or {}
            real = int(u.get("prompt_tokens") or 0)
            if real and raw:
                # Keep the calibration current: an even blend, never trusting below half.
                _scale[self.model] = max(0.5, 0.5 * _scale.get(self.model, 1.0) + 0.5 * real / raw)
            usage["prompt_tokens"] += real
            usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
            usage["rounds"] = rnd + 1

            choice = (result.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            calls = msg.get("tool_calls") or []
            salvaged = False
            salvage_how: str | None = None
            if not calls and tool_names:
                # A tool call written as text (gpt-oss does this, and every model does in the
                # final round, which offers no tools but asks for submit_candidate): run it.
                found = _text_tool_calls(msg.get("content") or "", tool_names)
                if found:
                    salvaged = True
                    salvage_how = "tool call written as text"
                    calls = [{"id": f"text_{rnd}_{i}", "type": "function",
                              "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(found)]
                    log(f"{self.model}: recovered {len(calls)} tool call(s) written as text "
                        f"({', '.join(n for n, _ in found)})")
            dropped: list[str] = []
            if calls and gate is not None:
                # A restricted round runs only what it is for. 10-01 21:36: in the final round
                # Muse-Glimmer wrote a run_python call as harmony/XML text, the salvage ran it --
                # the last experiment and the last round went on a check, and nothing was submitted.
                keep = [tc for tc in calls if _resolve_tool_name(
                    _plain_name((tc.get("function") or {}).get("name") or "?"), tool_names)[0] in gate]
                dropped = [str((tc.get("function") or {}).get("name") or "?") for tc in calls if tc not in keep]
                if dropped:
                    log(f"{self.model}: not run -- only {', '.join(sorted(gate)) or 'an answer'} "
                        f"this round: {', '.join(dropped)}")
                calls = keep
                if not calls:
                    salvaged, salvage_how = False, None
            content_now = (msg.get("content") or "").strip()
            push_early: str | None = None
            nudged_early = False
            if not calls and text_call is not None and content_now:
                # A script written as text where a call was due: in a restricted or final round,
                # or a reply that would otherwise end the turn (no nudge left to send).
                due = final_round or gate is not None
                if not due:
                    push_early, nudged_early = nudge(content_now), True
                    due = push_early is None
                conv = text_call(content_now) if due else None
                if conv and (gate is None or conv[0] in gate):
                    salvaged, salvage_how = True, "script written as text"
                    calls = [{"id": f"script_{rnd}", "type": "function",
                              "function": {"name": conv[0], "arguments": json.dumps(conv[1])}}]
                    log(f"{self.model}: reply carried its script as text -- calling {conv[0]} with it")
                    self.say("general", "thought",
                             f"(the script was written as text -- the runner called {conv[0]} with it)", tag)
            if dropped and not calls and not final_round:
                messages.append({"role": "assistant", "content": content_now or "(tool call)"})
                messages.append({"role": "user", "content": (
                    f"{', '.join(dropped)} is not available now -- NOT run. Call "
                    f"{' or '.join(sorted(gate)) or 'nothing more'} as a TOOL CALL.")})
                continue
            if calls and (not final_round or salvaged or gate):
                messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
                finish = choice.get("finish_reason")
                for i, tc in enumerate(calls):
                    fn = tc.get("function") or {}
                    name, notes = _resolve_tool_name(_plain_name(fn.get("name") or "?"), tool_names)
                    notes = [notes] if notes else []
                    args_json_ok = True
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except ValueError:
                        args = {}
                        args_json_ok = False
                    if isinstance(args, dict) and schemas.get(name):
                        args, arg_notes = _coerce_args(args, schemas[name])
                        notes += arg_notes
                    usage["tool_calls"] += 1
                    self.say("general", "thought", f"→ {name}: {_summarize_args(name, args)}",
                             {**tag, "tool": name, "round": rnd + 1})
                    _act(self).tool_start(name, args)
                    t0 = time.time()
                    call_hash = _tool_call_hash(name, args if isinstance(args, dict) else {})
                    prev = failed_calls.get(call_hash)
                    try:
                        if prev is not None:
                            # Same tool + same args already failed this iteration. Intercept
                            # instead of executing (the failure was in the tool result content,
                            # not transient), and tell the model to CHANGE something -- the ten
                            # sightings of one Qwen agent re-sending an identical library_save
                            # in bug #12 wasted the whole tool budget on this pattern.
                            prev["count"] += 1
                            intercepted += 1
                            out = {"error": (
                                f"you already called {name} this iteration with these exact "
                                "arguments and it failed the same way -- NOT run again. Previous "
                                f"error: {prev['err'][:400]}. Change what that error names (a "
                                "missing function, a bad field, the wrong 'kind', ...) or use a "
                                "different tool. Repeating the same call will end the iteration.")}
                            ok = False
                            log(f"{self.model}: {name} intercepted (repeat #{prev['count']}; "
                                f"{intercepted}/{REPEAT_TOOL_LIMIT} intercepts this iteration)")
                        elif _truncated_code_call(name, args, args_json_ok, finish):
                            # Code came back cut off mid-token (Groq/Qwen3.8-27b did this 48 times:
                            # `bar = ft.load('sql_exports_db`, `print(g.quantile`,
                            # `rows['X'].to_np.`, half-written signal() bodies to library_save).
                            # Refuse instead of consuming the tool's slot (an experiment, a saved
                            # module or a candidate); ask the model to resend shorter. `call()`
                            # never runs, so nothing is charged for the truncated attempt.
                            code_tail = str(args.get("code") or "")[-160:] if isinstance(args, dict) else ""
                            slot = ("This experiment has NOT been consumed." if name == "run_python"
                                    else "The candidate slot has NOT been used." if name == "submit_candidate"
                                    else "Nothing was saved to the library.")
                            out = {"error": (
                                f"your {name} call arrived TRUNCATED (the code did not compile, "
                                "and either the reply's finish_reason was 'length' or the arguments "
                                f"were cut mid-token). {slot} Resend the tool call with a SHORTER "
                                "script -- move helpers into library_save modules and import them, "
                                "drop debug prints that aren't essential, and split the work across "
                                "multiple calls if needed. Last chunk received: ..." + code_tail)}
                            ok = False
                            log(f"{self.model}: {name} code arrived truncated "
                                f"(finish={finish!r}, json_ok={args_json_ok}); asked model to resend shorter")
                        else:
                            out = call(name, args if isinstance(args, dict) else {})
                            ok = not (isinstance(out, dict) and "error" in out)
                    except RuntimeError as exc:
                        out, ok = {"error": str(exc)}, False
                    out = _with_note(out, "; ".join(notes))
                    _act(self).tool_done(name, args, out, ok,
                                         **({"salvaged": salvage_how} if salvage_how else {}))
                    if not ok:
                        self.say("errors", "error", f"{name} failed: {str(out.get('error'))[:500]}",
                                 {**tag, "tool": name})
                        # Remember the first failure for this exact (name, args) so a repeat is
                        # intercepted next time (the intercept itself is not re-registered).
                        if prev is None:
                            failed_calls[call_hash] = {
                                "name": name, "err": _err_signature(out), "count": 1}
                    else:
                        log(f"{self.model}: {name} ok in {time.time() - t0:.1f}s")
                    messages.append({"role": "tool", "tool_call_id": tc.get("id") or f"call_{rnd}_{i}",
                                     "name": name,
                                     "content": _clip(out, _result_budget(context_for(self.model)))})
                if intercepted >= REPEAT_TOOL_LIMIT:
                    # The model kept re-issuing calls it had already been told failed. Ending
                    # the iteration is cheaper than another round it will spend the same way.
                    top = max(failed_calls.values(), key=lambda x: x["count"], default={})
                    reason = (f"stopped: {intercepted} identical repeat calls (same tool, same "
                              f"args, same error) -- most-repeated {top.get('name', '?')} x"
                              f"{top.get('count', 0)}")
                    log(f"{self.model}: {reason}")
                    self.say("errors", "error", reason, tag)
                    _act(self).end("stopped", reason)
                    return False, reason, usage
                if done():
                    return True, "", usage
                continue

            content = (msg.get("content") or "").strip()
            reasoning = (msg.get("reasoning_content") or "").strip()
            if (choice.get("finish_reason") == "length" and not content and not length_retry["used"]
                    and not final_round):
                length_retry.update(used=True, boost=True)
                messages.append({"role": "assistant", "content": reasoning[-1500:] or "(no reply)"})
                messages.append({"role": "user", "content": (
                    "You used your whole output budget thinking and produced no answer or tool call. "
                    "Stop deliberating: decide from what you already worked out and make the next TOOL CALL "
                    "now (submit_candidate if your script is ready). Keep any further reasoning short.")})
                self.say("general", "thought",
                         "(ran out of output tokens while thinking -- asked to act now, with a larger budget)", tag)
                continue
            # The model stopped calling tools. If the job is not done, say so and go on:
            # gpt-oss in particular "answers" with prose or writes its next tool call as text.
            if final_round:
                push = None
            elif gate is not None:
                # A restricted round answered in prose: the call is still what it is for.
                push = (f"That was text, so nothing happened. Call {' or '.join(sorted(gate))} now -- a TOOL "
                        "CALL with the complete arguments, not text.") if gate else None
            else:
                push = push_early if nudged_early else nudge(content or reasoning)
            if push:
                messages.append({"role": "assistant", "content": content or reasoning[-1500:] or "(no reply)"})
                messages.append({"role": "user", "content": push})
                self.say("general", "thought", "(nudged: stopped before finishing -- asked to continue with a tool call)", tag)
                continue
            if content:
                messages.append({"role": "assistant", "content": content})
                return True, content, usage
            if choice.get("finish_reason") == "length":
                cap = (payload or {}).get("max_tokens", MAX_TOKENS)
                return False, (f"{self.model} spent its whole output budget ({cap} tokens) reasoning and never answered"
                               + (", even after being told to act with twice the budget" if length_retry["used"] else "")
                               + f". Raise FREESWARM_SWARM_MAX_TOKENS (now {MAX_TOKENS}).\n\n{reasoning[-2000:]}"), usage
            if reasoning:
                return True, f"(reasoning only, no final answer)\n\n{reasoning}", usage
            return False, f"{self.model} returned nothing (finish_reason={choice.get('finish_reason')}).", usage
        return False, CONVERSE_NO_ANSWER, usage

    def answer(self, task: dict) -> tuple[bool, str, dict]:
        llms, forecasters = self._sync()
        world = ProjectWorld(self.project, self.model, llms, forecasters)
        title = (task.get("title") or "").strip()
        description = (task.get("description") or "").strip()
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM + world.briefing()},
            {"role": "user", "content": title if not description else f"{title}\n\n{description}"},
        ]
        return self.converse(messages, world.tools(), world.call, tag={"task_id": task["id"]})

    # -- objectives -----------------------------------------------------------------------
    def next_objective(self) -> dict | None:
        """This project's running objectives, taken in turn."""
        now = time.time()
        if now - self._obj_checked > OBJECTIVE_POLL_S:
            self._obj_checked = now
            try:
                self._objectives = request(
                    CONTROL_PLANE, f"/api/projects/{q(self.pid)}/objectives?status=running").get("objectives", [])
            except RuntimeError as exc:
                log(f"{self.model}: objectives: {exc}")
                self._objectives = []
        if not self._objectives:
            return None
        self._obj_turn = (self._obj_turn + 1) % len(self._objectives)
        return self._objectives[self._obj_turn]

    def _ask(self, model: str, msgs: list[dict], max_tokens: int) -> dict:
        """One tool-less completion's message, given room to finish.

        `max_tokens` is the size of the ANSWER wanted. A reasoning model's thinking is counted
        against the same limit, so the request asks for that plus REASONING_ROOM; a reply that is
        still cut off (finish "length") is asked for once more with twice the room. Sent with
        the bare number, Muse-Glimmer's 15-lesson consolidation stopped mid-list at 2,999 of
        3,000 tokens every time it was due (no JSON to parse, nothing consolidated, four minutes
        of engine time each), and audits came back as 3,000 tokens of thinking and no verdict."""
        first = _toolless_budget(model, msgs, max_tokens)
        payload = {"model": model, "messages": msgs, "stream": False, "max_tokens": first}
        r = self._generate(payload)
        choice = (r.get("choices") or [{}])[0]
        again = min(2 * first, max_output_for(model) or MAX_TOKENS, MAX_TOKENS, max(first, _output_room(model, msgs)))
        if choice.get("finish_reason") == "length" and again > first:
            log(f"{self.model}: reply from {model} cut off at {first} tokens; asking again with {again}")
            try:
                choice = (self._generate({**payload, "max_tokens": again}).get("choices") or [{}])[0]
            except RuntimeError as exc:                # keep the cut-off reply: no worse than before
                log(f"{self.model}: second try failed ({str(exc)[:200]}); using the cut-off reply")
        self._last_finish = choice.get("finish_reason")   # _repair_code: was the reply cut off?
        return choice.get("message") or {}

    def _chat(self, prompt: str, max_tokens: int = 2048, system: str | None = None) -> str:
        """One tool-less completion: audits, lessons, judging, consolidation."""
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        msg = self._ask(self.model, msgs, max_tokens)
        return (msg.get("content") or msg.get("reasoning_content") or "").strip()

    def _repair_code(self, tool: str, code: str, crash: dict, extra: str = "") -> str | None:
        """This agent's own model, asked to fix a script of its that crashed (see _auto_repair).

        The reply gets the runner's normal budget (up to MAX_TOKENS, as the window allows): a
        reasoning model thinks first and only the code block is used. Sized to the script alone
        (len(code)//3 + 512, plus REASONING_ROOM), Muse-Glimmer's repairs ran out at ~5.4k tokens
        mid-thought, were asked again, and were cut off again (10-01 17:13). A reply still cut off
        with no complete code block ends the repair (RuntimeError -> _auto_repair stops) rather
        than spending another attempt; the wait is capped by _auto_repair's deadline."""
        if self._stop.is_set() or self.retired.is_set():
            return None
        prompt = _repair_prompt(tool, code, crash, extra)
        room = _output_room(getattr(self, "model", ""), [{"role": "user", "content": prompt}])
        want = min(MAX_TOKENS, max(MIN_OUTPUT, len(code) // 3 + 512, room))
        deadline = getattr(_TL, "repair_deadline", None)
        model = getattr(self, "model", "")
        if deadline is not None:
            left = deadline - time.time()
            if left < 30:
                raise RuntimeError("no repair time left")
            # A slow model (measured) that cannot write the fixed script in the time left is not
            # asked: the request would only time out (Qwen, 10-01 19:43).
            need = _expected_reply_s(model, len(code) // 3 + 512, kind="repair")
            if need is not None and left < need:
                raise RuntimeError(f"no repair time left: a fix from {model} takes ~{need:.0f}s "
                                   f"({_speed.get(model, 0):.0f} tok/s measured), {left:.0f}s left")
            self._gen_timeout = int(left)
        self._last_finish = None
        t0 = time.time()
        try:
            text = self._chat(prompt, max_tokens=want)
        finally:
            self._gen_timeout = None
        _note_reply_s("repair", model, time.time() - t0)
        got = _fenced_code(text)
        if got is None and getattr(self, "_last_finish", None) == "length":
            raise RuntimeError(f"the repair reply was cut off at the {want}-token limit before any complete code block")
        return got

    def _peer_chat(self, prompt: str, max_tokens: int = 3000) -> tuple[str, str]:
        """Ask a DIFFERENT loaded model when one is allowed -- a critic that did not write
        the code. Falls back to this agent's own model."""
        llms, _ = self._sync()
        # Free models first: a critique is routine work, not worth a paid call when a free
        # peer can do it.
        peers = sorted((m for m in llms if m != self.model and permitted(self.project, m)), key=is_external)
        model = peers[0] if peers else self.model
        msgs = [{"role": "user", "content": prompt}]
        try:
            msg = self._ask(model, msgs, max_tokens)
        except RuntimeError as exc:
            # A hosted peer out of today's budget is no reason to skip the critique: this
            # agent's own model can still do it (if it is also hosted and refused, _generate
            # notes that and run() backs off).
            if model == self.model or not _spending_limited(str(exc)):
                raise
            model = self.model
            msg = self._ask(model, msgs, max_tokens)
        return model, (msg.get("content") or msg.get("reasoning_content") or "").strip()

    def audit(self, obj: dict, cand: dict) -> None:
        o = obj
        m = o["metric"]
        body = cand.get("code") or cand.get("answer") or ""
        prompt = (
            "You are auditing a candidate solution before it can become the champion of this objective.\n\n"
            f"OBJECTIVE: {o['title']}\n{o.get('description') or ''}\n\n"
            f"It scored {((cand.get('metrics') or {}).get('holdout') or {}).get(m['kind'], cand.get('score'))} "
            f"({m['kind']}) on a hidden holdout and {cand.get('is_score')} in-sample (ranking score {cand.get('score')}). "
            "The harness already verified mechanically that its positions do not change when future rows "
            "are removed, and it computes returns itself from prices.\n\n"
            "Check the code for problems the mechanical tests cannot catch:\n"
            "1. Hard-coded dates, prices or thresholds that encode knowledge of the specific data "
            "(e.g. 'go long after 2023-03-01', a parameter that only makes sense in hindsight).\n"
            "2. Reading files or data other than through ft.load, or touching /work/.ft/ files.\n"
            "3. Degenerate or fragile logic (e.g. trades only a handful of days, extreme leverage "
            "flipping, relies on a single outlier).\n"
            "4. Anything else that makes the score untrustworthy -- shown in the code, not supposed.\n\n"
            "A FAIL disqualifies the result for good, so it needs a CONCRETE defect you can point to in the code "
            "(quote the line). These are NOT defects and must not fail a candidate: ordinary parameter choices "
            "(thresholds, windows, hold times) with no sign of being fitted to specific dates; constant position "
            "size; simple exit rules; generic worries that it 'may overfit' or 'might not generalise'. The author "
            "never saw the holdout, so a parameter cannot have been tuned to it. Mention such concerns in notes and "
            "PASS. Reply with ONLY a JSON object: "
            '{"passed": true|false, "issues": ["..."], "notes": "one or two sentences"}\n\n'
            f"CODE:\n```python\n{body[:16000]}\n```"
        )
        try:
            auditor, text = self._peer_chat(prompt)
        except RuntimeError as exc:
            log(f"{self.model}: audit call failed: {exc}")
            return
        verdict = _json_object(text)
        if verdict is None or "passed" not in verdict:
            log(f"{self.model}: auditor reply unparseable; leaving candidate {cand['id']} pending")
            return
        passed = bool(verdict.get("passed"))
        notes = (verdict.get("notes") or "") + ("" if not verdict.get("issues") else
                                                 " Issues: " + "; ".join(map(str, verdict["issues"]))[:1500])
        try:
            res = request(CONTROL_PLANE, f"/api/objectives/{q(o['id'])}/candidates/{q(cand['id'])}/audit",
                          {"passed": passed, "notes": notes, "model": auditor})
        except RuntimeError as exc:
            log(f"{self.model}: posting audit failed: {exc}")
            return
        tag = {"objective_id": o["id"], "candidate_id": cand["id"]}
        if res.get("champion"):
            self.announce(o, cand["id"])
        else:
            self.say("general", "thought",
                     f"Audit of #{cand.get('seq')} by {auditor}: {'passed' if passed else 'FAILED'} -- {notes[:400]}", tag)

    def announce(self, obj: dict, cid: str) -> None:
        try:
            c = request(CONTROL_PLANE, f"/api/objectives/{q(obj['id'])}/candidates/{q(cid)}")
        except RuntimeError:
            return
        m = c.get("metrics") or {}
        ho, ins = m.get("holdout") or {}, m.get("in_sample") or {}
        kind = obj["metric"]["kind"]
        line = (f"New best for \"{obj['title']}\": candidate #{c['seq']} by {c['model']} -- "
                f"ranking score {_fmt(c.get('score'))}, {kind} {_fmt(ho.get(kind, c.get('score')))} on the holdout "
                f"(in-sample {_fmt(c.get('is_score'))})")
        if ho:
            line += (f", holdout return {_pct(ho.get('total_return'))}, max drawdown {_pct(ho.get('max_drawdown'))}, "
                     f"{ho.get('active_days')} active days")
        self.say("results", "result", f"{line}.\n\nHypothesis: {c.get('rationale') or '(none)'}",
                 {"objective_id": obj["id"], "candidate_id": cid, "champion": True, "score": c.get("score")})
        log(f"{self.model}: NEW BEST #{c['seq']} {kind}={c.get('score')}")

    def consolidate(self, obj: dict, lessons: list[str]) -> None:
        prompt = (
            f"These are lessons a team of agents recorded while working on: {obj['title']}.\n\n"
            + "\n".join(f"- {x}" for x in lessons) +
            "\n\nMerge them into at most 15 lessons: drop duplicates and ones contradicted by later "
            "evidence, keep specifics (parameters, features, failure causes), keep the KEEP:/AVOID:/TRY: "
            'prefixes. Reply with ONLY a JSON array of strings.'
        )
        try:
            text = self._chat(prompt, max_tokens=3000)
        except RuntimeError as exc:
            log(f"{self.model}: consolidation failed: {exc}")
            return
        items = _json_array(text)
        if not items:
            return
        try:
            request(CONTROL_PLANE, f"/api/objectives/{q(obj['id'])}/lessons/replace",
                    {"lessons": [str(x)[:2000] for x in items[:15]], "model": self.model})
            self.say("general", "thought", f"Consolidated {len(lessons)} lessons into {len(items[:15])}.",
                     {"objective_id": obj["id"]})
        except RuntimeError as exc:
            log(f"{self.model}: posting consolidated lessons failed: {exc}")

    def inbox(self) -> list[dict]:
        """Board messages addressed to this agent (by `to` or an @mention) or replying to one of
        its own, since its last iteration (the last INBOX_WINDOW_S after a start), that its
        model has not answered yet -- a teammate's question or hand-over lands in the next brief."""
        try:
            entries = request(BOARD, f"/mb/messages?project_id={q(self.pid)}&tail={INBOX_TAIL}").get("entries", [])
        except RuntimeError:
            return []
        # The mentor does not reply to plain acknowledgements (see _inbox_entries).
        return _inbox_entries(entries, self.model, self._inbox_since or (time.time() - INBOX_WINDOW_S),
                              acks=self.role != "mentor")

    # -- one agent answers each message ------------------------------------------------------
    def _claim_feedback(self, seq: int) -> tuple[str | None, int | None]:
        """Claim inbox message `seq` for this agent to answer: compare-and-set of the board's
        blackboard key "fbclaim:<seq>" (absent, expired, or already this agent's). Returns
        (None, version) when this agent holds it, (holder, None) when another agent does. A
        board that cannot be asked does not block the answer: the claim is then assumed."""
        key, now = f"fbclaim:{seq}", time.time()
        path = f"/mb/state/{q(key)}"
        try:
            try:
                cur = request(BOARD, f"{path}?project_id={q(self.pid)}")
            except RuntimeError as exc:
                if "-> 404" not in str(exc):
                    raise
                cur = {}
            value = cur.get("value") if isinstance(cur.get("value"), dict) else {}
            version = int(cur.get("version") or 0)
            holder = value.get("agent")
            mine = holder == self.agent_name or (self.agent_id and value.get("agent_id") == self.agent_id)
            if holder and not mine and float(value.get("until") or 0) > now:
                return str(holder), None
            try:
                got = request(BOARD, path, {"project_id": self.pid, "updated_by": self.agent_name,
                                            "expect_version": version,
                                            "value": {"seq": seq, "agent": self.agent_name, "agent_id": self.agent_id,
                                                      "at": now, "until": now + FEEDBACK_CLAIM_TTL_S}},
                              method="PUT")
            except RuntimeError as exc:
                if "-> 409" not in str(exc):
                    raise
                # Another agent claimed it between the read and the write.
                try:
                    won = request(BOARD, f"{path}?project_id={q(self.pid)}").get("value") or {}
                except RuntimeError:
                    won = {}
                return str(won.get("agent") or "a teammate"), None
            return None, _int_or_none((got or {}).get("version"))
        except Exception as exc:  # noqa: BLE001 -- a claim must never stop the answer, nor the work
            log(f"{self.agent_name}: claiming #{seq} failed ({exc}); answering it anyway")
            return None, None

    def _release_feedback(self, seq: int, version: int | None) -> None:
        """Give up the claim on `seq` (its answer failed), so either agent may answer it next."""
        if version is None:
            return
        try:
            request(BOARD, f"/mb/state/{q(f'fbclaim:{seq}')}",
                    {"project_id": self.pid, "updated_by": self.agent_name, "expect_version": version,
                     "value": {"seq": seq, "agent": self.agent_name, "agent_id": self.agent_id,
                               "until": 0, "released": True}}, method="PUT")
        except Exception as exc:  # noqa: BLE001 -- the claim then simply expires
            log(f"{self.agent_name}: releasing the claim on #{seq} failed: {exc}")

    def answer_feedback(self, obj: dict, ctx: dict, inbox: list[dict], world) -> list[dict]:
        """Have this agent's own model answer its inbox before the work starts: one tool-less
        call, one accept/reject/question reply per message, each posted to the sender. Returns
        the posted replies ({"reply_to", "verdict", "text", "from", "candidate"?}) -- the
        iteration's commitments. A failed, cut-off or unparseable answer is logged and the
        iteration goes on without it; nothing here may stop the work.

        Only messages this agent claimed are answered (newest first, at most
        FEEDBACK_MAX_MESSAGES): one claimed by the other agent of its model, or about a
        candidate that agent ran, is left to it. Acknowledgements and replies deeper than
        FEEDBACK_MAX_DEPTH are not answered. Those messages, with why, are kept in
        self.feedback_skipped ({seq: why}) so the iteration's brief leaves them out too."""
        self.feedback_skipped: dict[int, str] = {}
        if not ANSWER_FEEDBACK or not inbox or getattr(self, "_budget_block", None) \
                or self._stop.is_set() or self.retired.is_set():
            return []
        old = [m for m in inbox if (m.get("minutes_ago") or 0) * 60 > FEEDBACK_MAX_AGE_S]
        if old:
            log(f"{self.agent_name}: not answering {len(old)} message(s) older than "
                f"{FEEDBACK_MAX_AGE_S / 3600:.0f}h: " + ", ".join(f"#{m['seq']}" for m in old))
        fresh = []
        for m in inbox:
            if (m.get("minutes_ago") or 0) * 60 > FEEDBACK_MAX_AGE_S:
                continue
            why = _feedback_closed(m)
            if why:
                self.feedback_skipped[m["seq"]] = why
            else:
                fresh.append(m)
        mine_cands = getattr(self, "_my_candidates", None) or set()
        most, max_s = _feedback_budget(self.model, FEEDBACK_MAX_MESSAGES)
        due, claims = [], {}
        for m in reversed(fresh):                       # newest first
            if len(due) >= most:
                break
            owner = _feedback_route(m, self.agent_id, mine_cands)
            if owner:
                self.feedback_skipped[m["seq"]] = f"left to {owner}, who ran candidate {m['candidate']}"
                continue
            holder, version = self._claim_feedback(m["seq"])
            if holder:
                self.feedback_skipped[m["seq"]] = f"being answered by {holder}"
                continue
            claims[m["seq"]] = version
            due.append(m)
        due.reverse()                                   # the prompt lists them oldest first
        if self.feedback_skipped:
            log(f"{self.agent_name}: not answering " + "; ".join(f"#{n}: {w}" for n, w in self.feedback_skipped.items()))
        if not due:
            return []
        t0 = time.time()
        prompt = feedback_prompt(obj, ctx, due, self.agent_name)
        room = _output_room(self.model, [{"role": "user", "content": prompt}])
        want = min(MAX_TOKENS, max(MIN_OUTPUT, room))
        self._last_finish, error, text = None, None, ""
        self._gen_deadline = t0 + max_s
        try:
            text = self._chat(prompt, max_tokens=want)
            _note_reply_s("feedback", self.model, time.time() - t0)
        except RuntimeError as exc:
            error = f"the answering call failed: {str(exc)[:300]}"
        finally:
            self._gen_deadline = None
        replies: list[dict] = []
        if error is None:
            replies = _parse_feedback_replies(text, {m["seq"] for m in due})
            if not replies:
                error = ("the reply was cut off before a complete JSON list of replies"
                         if getattr(self, "_last_finish", None) == "length"
                         else "the reply held no usable JSON list of replies")
        by_seq = {m["seq"]: m for m in due}
        posted: list[dict] = []
        for r in replies:
            m = by_seq[r["reply_to"]]
            try:
                request(BOARD, "/mb/messages", _feedback_reply_body(self.pid, self.model, self.agent_name,
                                                                    self.agent_id, obj["id"], m, r))
            except RuntimeError as exc:
                log(f"{self.agent_name}: posting the reply to #{r['reply_to']} failed: {exc}")
                continue
            # What record_collaboration reads: these messages are answered.
            world.sent.append({"to": m["from"], "channel": "team", "reply_to": r["reply_to"], "text": r["text"][:160],
                               "answers": [r["reply_to"]]})
            posted.append({**r, "from": m["from"], **({"candidate": m["candidate"]} if m.get("candidate") else {})})
        # Unanswered messages go back: either agent may answer them in its next iteration.
        done = {p["reply_to"] for p in posted}
        for n, version in claims.items():
            if n not in done:
                self._release_feedback(n, version)
        counts = {v: sum(1 for p in posted if p["verdict"] == v) for v in FEEDBACK_VERDICTS}
        if error:
            log(f"{self.agent_name}: answering {len(due)} message(s) before the work: {error}; going on without it")
        else:
            log(f"{self.agent_name}: answered {len(posted)} of {len(due)} message(s) before the work "
                f"({counts['accept']} accepted, {counts['reject']} rejected, {counts['question']} questions)")
        # A soft step: a failed answer is recorded ok=True with its reason under "soft_error" (not
        # "error"), so the Work page (app/work.py tool_failed/recovery) and the bug monitor do not
        # count it as an unrecovered tool failure -- the iteration goes on without it, by design.
        _act(self).step(
            "answer_feedback",
            {"messages": [f"#{m['seq']} from {m['from']}: {str(m['text'])[:200]}" for m in due],
             "time_limit_s": int(max_s),
             **({"skipped_older_than_24h": [m["seq"] for m in old]} if old else {}),
             **({"not_answered": [f"#{n}: {w}" for n, w in self.feedback_skipped.items()]}
                if self.feedback_skipped else {})},
            {"answered": 0, "soft": True, "soft_error": error, "released": sorted(claims)} if error
            else {"answered": len(posted), **counts,
                  "replies": [f"#{p['reply_to']} {p['verdict']}: {p['text']}" for p in posted]},
            ok=True, at=t0, **({"soft": True} if error else {}))
        return posted

    def mentor_coaching(self) -> dict | None:
        """The mentor's latest coaching for the whole team. It is posted at the END of the
        mentor's #planning note, which the brief clipped to 400 characters -- so no searcher
        ever read it."""
        try:
            entries = request(BOARD, f"/mb/messages?project_id={q(self.pid)}&channel=planning&tail=60").get("entries", [])
        except RuntimeError:
            return None
        return _coaching_from(entries)

    def _submission_parent(self, world, args: dict, ctx: dict) -> dict | None:
        """The candidate a submission builds on, as {"id", "seq"}: the `parent` it names, else
        the parent this iteration was assigned, else one its rationale names ("Building on
        candidate 75 ..."). Explore iterations had no way to say it: #132 and #133 built on
        Muse's #75 and were stored, and counted, as starting from nothing."""
        assigned = ctx.get("parent") or {}
        assigned = {"id": assigned["id"], "seq": assigned.get("seq")} if assigned.get("id") else None
        ref = args.get("parent")
        if ref in (None, "", 0, "0", "none", "None"):
            if assigned:
                return assigned
            ref = _built_on_ref(str(args.get("rationale") or ""))
            if ref is None:
                return None
        try:
            hit, _ = world._candidate(ref)
        except RuntimeError:
            hit = None
        return {"id": hit["id"], "seq": hit.get("seq")} if hit else assigned

    def record_collaboration(self, obj: dict, ctx: dict, world, last: dict, team_note: str, inbox: list[dict],
                             built_on: dict | None = None) -> None:
        """Post, to #team, how this iteration used and helped the rest of the team -- built from
        what actually happened, plus the agent's own one-line note. `built_on` is the parent
        the submission named ({"id", "seq"}); without it, the assigned parent."""
        me = self.model.split("/")[-1]
        tag = {"objective_id": obj["id"], "candidate_id": last.get("candidate_id")}
        parts: list[str] = []
        collab: dict = {"agent": self.model, "mode": ctx.get("mode"), "candidate": last.get("seq"),
                        "built_on": None, "reused": [], "contributed": list(world.saved),
                        "messages_sent": world.sent, "answered": [], "inbox": len(inbox),
                        # which messages: the console lists the unanswered ones (app/team_threads.py)
                        "inbox_seqs": [m["seq"] for m in inbox],
                        "teammates_seen": [m["who"] for m in ctx.get("teammates") or []]}
        parent = built_on or ctx.get("parent")
        if parent and parent.get("id"):
            try:
                pc = request(CONTROL_PLANE, f"/api/objectives/{q(obj['id'])}/candidates/{q(parent['id'])}")
                collab["built_on"] = {"seq": parent["seq"], "by": pc.get("model")}
                who = "its own" if pc.get("model") == self.model else f"{(pc.get('model') or '?').split('/')[-1]}'s"
                parts.append(f"built on {who} candidate #{parent['seq']}")
            except RuntimeError:
                pass
        code = ""
        try:
            if last.get("candidate_id"):
                code = request(CONTROL_PLANE, f"/api/objectives/{q(obj['id'])}/candidates/{q(last['candidate_id'])}").get("code") or ""
        except RuntimeError:
            pass
        names = set(_lib_imports(code))
        authors = {}
        try:
            for m in request(CONTROL_PLANE, f"/api/projects/{q(self.pid)}/library").get("modules", []):
                authors[m["name"]] = m.get("author") or "?"
        except RuntimeError:
            pass
        for n in sorted(names):
            if n in authors:
                collab["reused"].append({"module": n, "by": authors[n]})
        others = [r for r in collab["reused"] if r["by"] != self.model]
        if others:
            parts.append("reused " + ", ".join(f"lib.{r['module']} ({(r['by'] or '?').split('/')[-1]})" for r in others))
        mine = [r for r in collab["reused"] if r["by"] == self.model]
        if mine:
            parts.append("reused its own " + ", ".join(f"lib.{r['module']}" for r in mine))
        if world.saved:
            parts.append("contributed " + ", ".join(f"lib.{n}" for n in world.saved) + " to the library")
        # Answered: a team_post with reply_to = the message, or one that cites it as #<number>
        # (agents acted on the mentor's notes in their #planning post without linking them).
        replied = {m["reply_to"] for m in world.sent if m.get("reply_to")}
        replied |= {n for m in world.sent for n in m.get("answers") or []}
        collab["answered"] = [m for m in inbox if m["seq"] in replied]
        direct = [m for m in world.sent if m["to"] != "all"]
        if direct:
            parts.append("messaged " + ", ".join(sorted({m["to"].split("/")[-1] for m in direct})))
        if collab["answered"]:
            parts.append("answered " + ", ".join(f"{m['from'].split('/')[-1]} (#{m['seq']})" for m in collab["answered"]))
        elif inbox:
            parts.append(f"left {len(inbox)} message(s) to it unanswered")
        if not world.sent:
            parts.append("posted no plan")
        outcome = (f"candidate #{last.get('seq')}: in-sample {_fmt(last.get('in_sample_score'))}, look-ahead "
                   f"{last.get('lookahead')}, rank {last.get('rank')}" if last.get("status") == "ok"
                   else f"candidate #{last.get('seq')} failed to run")
        text = f"{me} ({ctx.get('mode')}): " + "; ".join(parts or ["worked alone"]) + f". Result: {outcome}."
        if team_note:
            text += f"\n{me}: \"{team_note}\""
        collab["note"] = team_note
        # meta.agent: which of the model's agents ran the candidate -- feedback about it is
        # routed to that agent (_inbox_entries' candidate_by).
        self.say("team", "result", text, {**tag, "agent": self.agent_name, "collab": collab})

    def teammates(self) -> list[dict]:
        """What the other agents announced in #planning in the last 45 minutes. The other agent
        on this same model is a teammate too (its plans were hidden: they share the author); the
        mentor's notes and practice updates reach the brief whole, elsewhere, not clipped here."""
        try:
            entries = request(BOARD, f"/mb/messages?project_id={q(self.pid)}&channel=planning&tail=20").get("entries", [])
        except RuntimeError:
            return []
        now = time.time()
        out = []
        for e in entries:
            meta = e.get("meta") or {}
            if now - e["ts"] >= 2700 or meta.get("practices") or meta.get("mentor"):
                continue
            agent = meta.get("agent")
            if agent == self.agent_name or (e.get("author_id") and e.get("author_id") == self.agent_id) \
                    or (e["author"] == self.model and not agent and not e.get("author_id")):
                continue
            out.append({"who": agent or e["author"], "minutes_ago": int((now - e["ts"]) / 60),
                        "text": str(e["content"])[:400]})
        return out[-8:]

    def rewrite_practices(self, obj: dict, ctx: dict) -> None:
        """The recursive step: rewrite the team's own working instructions from the evidence."""
        oid = obj["id"]
        try:
            cands = request(CONTROL_PLANE, f"/api/objectives/{q(oid)}/candidates?order=recent&limit=40").get("candidates", [])
            board = request(BOARD, f"/mb/messages?project_id={q(self.pid)}&tail=80").get("entries", [])
        except RuntimeError as exc:
            log(f"{self.model}: practices rewrite skipped: {exc}")
            return
        pb = ctx.get("playbook") or {}
        hist = "\n".join(
            f"#{c['seq']} {c['model']} [{c['mode']}] {c['status']} in-sample={_fmt(c.get('is_score'))} "
            f"look-ahead={c['lookahead']}{' CHAMPION' if c.get('champion_at') else ''}: {(c.get('rationale') or c.get('score_note') or '')[:180]}"
            for c in cands)
        errs = "\n".join(f"- {str(e['content'])[:200]}" for e in board if e["channel"] == "errors")[-3000:]
        try:
            team = request(BOARD, f"/mb/messages?project_id={q(self.pid)}&channel=team&tail=30").get("entries", [])
        except RuntimeError:
            team = []
        collab = "\n".join(f"- {str(e['content'])[:260]}" for e in team)[-5000:]
        lib = "\n".join(f"- {m['name']} [{m['kind']}] used {m['used_by']}, ok {m['ok']}, champions {m['champions']}; "
                         + "; ".join(f"{c['verdict']}: {c['text'][:80]}" for c in m.get("comments") or [])
                         for m in ctx.get("library") or [])
        prompt = (
            f"You maintain the TEAM PRACTICES section of the playbook for a swarm of agents working on: {obj['title']}.\n"
            "Team practices are about HOW the team should work -- process, what to check first, sequencing, how to "
            "use the tools, library and each other -- learned from what actually happened. They are read by every "
            "agent at the start of every iteration, so they must be short, concrete and correct.\n\n"
            f"CURRENT PRACTICES (v{pb.get('practices_version') or 0}):\n{pb.get('practices') or '(none yet)'}\n\n"
            f"RECENT CANDIDATES (newest first):\n{hist}\n\nLESSONS:\n" + "\n".join(f"- {x}" for x in ctx.get("lessons") or [])
            + f"\n\nLIBRARY:\n{lib or '(empty)'}\n\nRECENT ERRORS:\n{errs or '(none)'}\n\n"
            f"HOW THE AGENTS COLLABORATED (one line per iteration, newest last):\n{collab or '(no record yet)'}\n\n"
            "Rewrite the practices: keep what the evidence supports, drop what it contradicts, add what the "
            "failures and the successes teach (e.g. which tool calls wasted iterations, which process preceded "
            "improvements, which modules to start from). Include how the agents should collaborate: which kinds "
            "of hand-over, reuse and division of work preceded improvements, what was duplicated or ignored "
            "(unanswered messages, reinvented modules, plans nobody read). At most 12 numbered points, each one "
            "line. Reply with the practices only."
        )
        try:
            text = self._chat(prompt, max_tokens=4000)
        except RuntimeError as exc:
            log(f"{self.model}: practices rewrite failed: {exc}")
            return
        text = re.sub(r"^```\w*|```$", "", text.strip(), flags=re.M).strip()
        if len(text) < 40:
            return
        try:
            request(CONTROL_PLANE, f"/api/projects/{q(self.pid)}/playbook",
                    {"part": "practices", "text": text[:12000], "author": self.model,
                     "note": f"rewritten after {len(cands)} recent candidates"}, timeout=60)
        except RuntimeError as exc:
            log(f"{self.model}: saving practices failed: {exc}")
            return
        self.say("planning", "result", f"Updated the team practices (the playbook every agent follows):\n\n{text[:3000]}",
                 {"objective_id": oid, "practices": True})

    def mentor(self, obj: dict) -> None:
        """One mentoring pass: read the evidence, give the team directions, coaching, replies
        and forecasts to build; rewrite the team practices when they are due."""
        oid = obj["id"]
        tag = {"objective_id": oid, "mentor": True}
        try:
            brief = request(CONTROL_PLANE, f"/api/objectives/{q(oid)}/mentor/brief?model={q(self.model)}", timeout=120)
        except RuntimeError as exc:
            log(f"{self.model}: mentor brief for {oid}: {exc}")
            self._stop.wait(MENTOR_IDLE_S)
            return
        if not brief.get("due"):
            self._stop.wait(MENTOR_IDLE_S)
            return
        self.beat("working", force=True)
        _act(self).begin("mentor", obj, why=brief.get("why"))
        if brief.get("refresh_practices"):
            self.rewrite_practices(obj, brief)
        inbox = self.inbox()
        self._inbox_since = time.time()
        self.say("general", "thought", f"Mentoring: reading the team's results ({brief.get('why')}).", tag)
        try:
            text = self._chat(mentor_prompt(brief, inbox), max_tokens=MAX_TOKENS)
        except RuntimeError as exc:
            self.say("errors", "error", f"Mentor pass failed: {str(exc)[:400]}", tag)
            self._stop.wait(MENTOR_IDLE_S)
            return
        notes = parse_mentor(text)

        posted = []
        for d in (notes.get("directions") or [])[:MENTOR_MAX_DIRECTIONS]:
            if not isinstance(d, dict) or not str(d.get("idea") or "").strip():
                continue
            body = "\n".join(f"{k.upper()}: {str(d[k]).strip()}" for k in ("idea", "hypothesis", "test", "avoid")
                             if str(d.get(k) or "").strip())
            try:
                got = request(CONTROL_PLANE, f"/api/objectives/{q(oid)}/ideas",
                              {"model": self.model, "text": body[:8000], "trigger": "mentor"})
                posted.append((got.get("id"), body))
            except RuntimeError as exc:
                log(f"{self.model}: storing an idea failed: {exc}")
        coaching = str(notes.get("coaching") or "").strip()
        if posted or coaching:
            parts = [f"[idea {i}] {b}" for i, b in posted]
            if coaching:
                parts.append("COACHING:\n" + coaching)
            self.say("planning", "result",
                     "Mentor notes -- test one of these ideas and pass its number as `idea` to submit_candidate:\n\n"
                     + "\n\n".join(parts), {**tag, "ideas": [i for i, _ in posted],
                                            # read whole into every brief (Worker.mentor_coaching)
                                            **({"coaching": coaching[:4000]} if coaching else {})})
        for r in (notes.get("replies") or [])[:8]:
            body = _mentor_reply(self.pid, self.model, oid, r, brief, inbox)
            if body is None:
                continue
            try:
                request(BOARD, "/mb/messages", body)
            except RuntimeError as exc:
                log(f"{self.model}: reply failed: {exc}")
        # Forecasts last: building one can take minutes, and the notes above should not wait.
        for f in ([] if _is_task(brief["objective"]) else (notes.get("forecasts") or [])[:MENTOR_MAX_FORECASTS]):
            if not isinstance(f, dict) or not f.get("column"):
                continue
            recipe = {"column": str(f["column"]), "covariates": [str(c) for c in f.get("inputs") or []] or None,
                      "horizon": int(f.get("horizon") or 12), "every": int(f.get("every") or 0),
                      "model": f.get("model") or None}
            try:
                meta = request(CONTROL_PLANE, f"/api/objectives/{q(oid)}/features",
                               {k: v for k, v in recipe.items() if v is not None}, timeout=1800)
            except RuntimeError as exc:
                self.say("errors", "error", f"Mentor's forecast {recipe['column']} could not be built: {str(exc)[:400]}", tag)
                continue
            self.say("team", "result",
                     f"Built forecast {meta.get('view')} for the team ({str(f.get('why') or '')[:300]}). "
                     f"Measured in-sample skill: {_j(meta.get('skill'), 700)}. Load it with "
                     f"ft.load(\"{meta.get('view')}\", prefix=...) or the same recipe via ft.forecast(...).",
                     {**tag, "feature": meta.get("view")})
        self.beat("idle", force=True)

    def judge(self, obj: dict, view: dict, answer: str) -> dict:
        rubric = obj["metric"].get("rubric") or "How well does this answer achieve the objective?"
        prompt = (
            f"Score this candidate for the objective below from 0 (useless) to 10 (outstanding).\n\n"
            f"OBJECTIVE: {obj['title']}\n{obj.get('description') or ''}\n\nRUBRIC: {rubric}\n\n"
            f"CANDIDATE:\n{answer[:14000]}\n\nOUTPUT OF ITS CODE (if any):\n{view.get('stdout_tail', '')}\n\n"
            'Reply with ONLY a JSON object: {"score": <0-10>, "notes": "what would make it better"}'
        )
        try:
            judge_model, text = self._peer_chat(prompt, 1500)
        except RuntimeError as exc:
            return {**view, "judge_error": str(exc)}
        verdict = _json_object(text) or {}
        try:
            score = max(0.0, min(10.0, float(verdict.get("score"))))
        except (TypeError, ValueError):
            return {**view, "judge_error": "judge reply unparseable"}
        return request(CONTROL_PLANE, f"/api/objectives/{q(obj['id'])}/candidates/{q(view['candidate_id'])}/judge",
                       {"score": score, "notes": str(verdict.get("notes", ""))[:4000], "model": judge_model}) | {
                           "judge_score": score, "judge_notes": verdict.get("notes")}

    def iterate(self, obj: dict) -> None:
        oid = obj["id"]
        tag = {"objective_id": oid}
        try:
            ctx = request(CONTROL_PLANE, f"/api/objectives/{q(oid)}/context?model={q(self.model)}")
        except RuntimeError as exc:
            log(f"{self.model}: context for {oid}: {exc}")
            self._stop.wait(POLL_IDLE_S)
            return
        self.beat("working", force=True)
        _act(self).begin_iteration(obj, ctx)
        # Chores first -- they gate everyone else's progress.
        if ctx.get("audit"):
            self.say("general", "thought", f"Auditing contender #{ctx['audit'].get('seq')} before it can take the title.", tag)
            self.audit(obj, ctx["audit"])
            return
        if ctx.get("consolidate"):
            self.consolidate(obj, ctx.get("lessons") or [])
            return
        if ctx.get("refresh_practices"):
            self.rewrite_practices(obj, ctx)
            return

        llms, forecasters = self._sync()
        submits: list[dict] = []
        parents: list[dict | None] = []   # what each submission built on ({"id", "seq"}), in step

        def on_submit(args: dict) -> Any:
            if len(submits) >= MAX_SUBMITS:
                # 13 board errors (09-24..09-29) were further submit calls after the last slot,
                # some in the same reply: say what is left to do instead of a bare refusal.
                left = ("library_save the reusable module -- this BUILD iteration ends when one is saved"
                        if build and not world.saved else "answer in one line with what you learned; no more tool "
                        "calls are needed")
                return {"error": (f"submission limit reached: {MAX_SUBMITS} candidates were submitted this iteration "
                                  f"(last: #{submits[-1].get('seq')}, {submits[-1].get('status')}). This one was NOT "
                                  f"submitted. Now {left}.")}
            # Qwen's text-format tool calls (recovered by _text_tool_calls) often name the script
            # `script` or `source`; read as missing, each one was a 400 (bug #5).
            code = next((str(args[k]) for k in ("code", "script", "source") if str(args.get(k) or "").strip()), "")
            if not code and obj["metric"]["kind"] != "judge":
                return {"error": ("submit_candidate needs the complete Python script in `code`; this call had "
                                  f"{', '.join(sorted(args)) or 'no arguments'}. Call it again with code=<script>. "
                                  "The candidate slot has NOT been used.")}
            built_on = self._submission_parent(world, args, ctx)
            payload = {"code": code, "answer": str(args.get("answer") or ""),
                       "rationale": str(args.get("rationale") or "")[:8000], "model": self.model,
                       "mode": ctx["mode"], "parent_id": (built_on or {}).get("id")}
            try:
                idea = int(args.get("idea")) if args.get("idea") not in (None, "") else None
            except (TypeError, ValueError):
                idea = None
            if idea is not None and idea in {i.get("id") for i in ctx.get("ideas") or []}:
                payload["idea_id"] = idea
            # + time for the harness to build forecasts the script asks for (ft.forecast)
            eval_timeout = int(obj.get("eval_timeout_s") or 300) * 4 + 120 + FORECAST_BUILD_ALLOWANCE_S
            view = request(CONTROL_PLANE, f"/api/objectives/{q(oid)}/candidates", payload, timeout=eval_timeout)
            # A script that crashed goes back to the model for a fix and is evaluated again, in the
            # same submission slot. The harness look-ahead-tests the repaired script like any other.
            view = _auto_repair(
                "submit_candidate", code, view, self._repair_code,
                lambda src: request(CONTROL_PLANE, f"/api/objectives/{q(oid)}/candidates", {**payload, "code": src},
                                    timeout=eval_timeout),
                who=self.agent_name)
            if view.get("code_ran"):
                payload["code"] = view["code_ran"]
            if obj["metric"]["kind"] == "judge" and view.get("status") == "ok":
                repaired = {k: view[k] for k in ("auto_repaired", "code_ran") if view.get(k)}
                view = self.judge(obj, view, payload["answer"] or payload["code"])
                if repaired.get("auto_repaired"):
                    view = {"auto_repaired": repaired["auto_repaired"], **view, "code_ran": repaired["code_ran"]}
            submits.append(view)
            parents.append(built_on)
            if isinstance(view.get("seq"), int):
                # feedback about this candidate is this agent's to answer (_feedback_route)
                self.__dict__.setdefault("_my_candidates", set()).add(view["seq"])
            self.report_eval(obj, view)
            if view.get("status") == "error" and len(submits) < MAX_SUBMITS:
                view = {**view, "next": (f"This run failed, so the iteration is not over: read the error (and its "
                                         f"hint), fix the script and call submit_candidate again -- "
                                         f"{MAX_SUBMITS - len(submits)} more tries this iteration. Test a doubtful "
                                         "line with run_python first if you have experiments left.")}
            return view

        world = ObjectiveWorld(self.project, self.model, llms, forecasters, obj, on_submit)
        world.repairer = self._repair_code
        pb = ctx.get("playbook") or {}
        playbook = (pb.get("charter") or "").strip()
        if (pb.get("practices") or "").strip():
            playbook += (f"\n\n# Team practices (v{pb.get('practices_version')}, written by the team from its own "
                         f"results -- follow them)\n" + pb["practices"].strip())
        # Last, so it is the nearest thing to the task: results the operator disqualified by
        # hand after the automated checks passed them. These are not style preferences.
        if (pb.get("pitfalls") or "").strip():
            playbook += ("\n\n# Pitfalls -- results DISQUALIFIED after review (the harness passed them; a human or "
                         "an external model caught them). Never reproduce these patterns. Before you submit, check "
                         "your own code against every line here.\n" + pb["pitfalls"].strip())
        ctx["teammates"] = self.teammates()
        ctx["coaching"] = self.mentor_coaching()
        inbox = self.inbox()
        self._inbox_since = time.time()
        world.agent_name, world.author_id = self.agent_name, self.agent_id
        world.inbox_seqs = {m["seq"] for m in inbox}
        # Answer the feedback first (the user's "answer these before moving ahead"): the replies
        # are posted, and what the agent accepted becomes its commitments for this iteration.
        try:
            commitments = self.answer_feedback(obj, ctx, inbox, world)
        except Exception as exc:  # noqa: BLE001 -- answering is a step of the work, never a stop to it
            log(f"{self.agent_name}: answering feedback failed: {exc!r}")
            commitments = []
        if self._budget_block:
            # Today's spending limit refused the answering call; the iteration would be refused
            # too. run() backs off.
            return
        # Messages its twin answers (or that need no answer) leave this agent's brief and record.
        skipped = getattr(self, "feedback_skipped", None) or {}
        if skipped:
            inbox = [m for m in inbox if m["seq"] not in skipped]
            world.inbox_seqs = {m["seq"] for m in inbox}
        answered = {c["reply_to"] for c in commitments}
        ctx["inbox"] = [m for m in inbox if m["seq"] not in answered]
        ctx["commitments"] = commitments
        messages = [
            {"role": "system", "content": ITERATE_SYSTEM + playbook + "\n\n" + world.briefing()},
            {"role": "user", "content": iteration_prompt(ctx)},
        ]
        mode = ctx["mode"]
        parent = ctx.get("parent")
        self.say("general", "thought",
                 f"Iteration on \"{obj['title']}\": "
                 + ("BUILD -- adding a reusable module to the library" if mode == "build"
                    else f"improving #{parent['seq']} ({'rank ' + str(parent['rank']) if parent.get('rank') else 'unranked'})"
                    if parent else "exploring a new approach"),
                 tag)
        nudges = {"n": 0}

        build = ctx.get("mode") == "build"

        def nudge(text: str) -> str | None:
            if nudges["n"] >= 3:
                return None
            if build and not world.saved:
                nudges["n"] += 1
                return ("This is a BUILD iteration and nothing is in the library yet. Use a TOOL CALL, not text: "
                        "library_save the reusable module (kind, description, code, a short test), fix it if the "
                        "smoke test fails, then submit_candidate with a script that imports it.")
            if submits:
                return None
            nudges["n"] += 1
            return ("You stopped without calling submit_candidate, so nothing was evaluated. Use a TOOL CALL, "
                    "not text. If a tool returned an error, read it, fix the arguments and call it again. "
                    "If your script is ready, call submit_candidate with the complete script and rationale now.")

        judged = obj["metric"]["kind"] == "judge"

        def submit_tools() -> list[str]:
            """What a submit turn may call: submit_candidate, and library_save in a BUILD
            iteration that has saved nothing yet."""
            names = ["library_save"] if build and not world.saved else []
            return names + (["submit_candidate"] if len(submits) < MAX_SUBMITS else [])

        def script_call(text: str) -> tuple[str, dict] | None:
            """A reply that carries the candidate script as text (a ```python block that compiles
            and calls ft.report_*): submit it as what it is."""
            if judged or len(submits) >= MAX_SUBMITS:
                return None
            code = _text_script(text)
            return ("submit_candidate", {"code": code, "rationale": _text_rationale(text)}) if code else None

        def submit_prompt(why: str, reply: str = "") -> str:
            """The firm submit instruction, offering a default script: the one the reply wrote as
            text, else the last run_python script that ran."""
            lines = [why]
            if build and not world.saved:
                lines.append("This BUILD iteration has nothing in the library yet: library_save the reusable module "
                             "(with a short test) if you can, and submit_candidate in any case.")
            lines.append("Call submit_candidate NOW -- a TOOL CALL, not text; no other tool is available. "
                         + ("answer = your final answer; rationale = one line." if judged else
                            "code = your best complete script (it must report its result with the ft.report_* call "
                            "the task asks for); rationale = one line."))
            if world.saved:
                lines.append(f"Library modules you saved this iteration: {', '.join(world.saved)} -- a script can "
                             "import them with `from lib import <name>`.")
            if not judged:
                blocks = sorted((b.strip("\n") for b in _FENCE.findall(reply or "") if b.strip()), key=len, reverse=True)
                code, label = ((blocks[0], "the script you wrote as text in your last reply") if blocks
                               else (world.best_code, "your last run_python script that ran") if world.best_code
                               else (None, ""))
                if code and len(code) <= FORCED_SCRIPT_CHARS:
                    fix = "" if _reports(code) else (" -- it does not report a result yet: end main() with the "
                                                     "ft.report_* call the task asks for")
                    lines.append(f"If you have nothing better, submit {label}{fix}:\n```python\n{code}\n```")
                elif code:
                    lines.append(f"If you have nothing better, submit {label} (too long to repeat here).")
            return "\n".join(lines)

        def focus() -> tuple[list[str], str] | None:
            # The experiment budget is spent and nothing is submitted: the next turn is the submission.
            if submits or world.experiments < MAX_EXPERIMENTS:
                return None
            return submit_tools(), submit_prompt(
                f"All {MAX_EXPERIMENTS} run_python experiments of this iteration are used and nothing is submitted yet.")

        ok, text, usage = self.converse(
            messages, world.tools(), world.call, tag=tag, max_rounds=OBJECTIVE_TOOL_ROUNDS, nudge=nudge,
            # Finished once a submission ran cleanly or the budget of submissions is used.
            done=lambda: bool(submits) and (submits[-1].get("status") == "ok" or len(submits) >= MAX_SUBMITS)
            and (not build or bool(world.saved)),
            final_prompt=lambda: submit_prompt("Tool budget used up: this is the last round of the iteration."),
            final_tools=submit_tools, focus=focus, text_call=script_call,
        )
        if (not submits and (ok or text == CONVERSE_NO_ANSWER) and submit_tools()
                and not (self._stop.is_set() or self.retired.is_set() or self._budget_block)):
            # Still nothing to evaluate after a turn that ended normally: one forced submit turn.
            reply = next((str(m.get("content") or "") for m in reversed(messages) if m.get("role") == "assistant"), "")
            why = ("You ended the iteration without calling submit_candidate, so nothing was evaluated."
                   + (" You wrote your script as text." if _FENCE.search(reply) else ""))
            _act(self).step("forced_submit", {"why": why, "tools": submit_tools(),
                                              "default": "reply script" if _FENCE.search(reply)
                                              else "last run_python" if world.best_code else None},
                            {"forced": True}, ok=True, at=time.time())
            log(f"{self.agent_name}: no submission at the end of the turn -- one forced submit turn")
            self.say("general", "thought", "(ended without a submission -- the runner asked for one forced submit turn)", tag)
            messages.append({"role": "user", "content": submit_prompt(why, reply)})
            only = set(submit_tools())
            ok2, text2, usage2 = self.converse(
                messages, [t for t in world.tools() if t["function"]["name"] in only], world.call, tag=tag,
                max_rounds=1, force_tool=True,
                done=lambda: bool(submits),
                final_prompt=lambda: submit_prompt("Last chance: call submit_candidate now."),
                final_tools=submit_tools, text_call=script_call)
            for k in ("prompt_tokens", "completion_tokens", "tool_calls"):
                usage[k] = usage.get(k, 0) + usage2.get(k, 0)
            text = text2 or text
        if not submits:
            # A turn abandoned because the swarm was switched off is not a failure worth
            # posting to #errors -- it would fill the board every time the operator stops.
            # Nor is a turn refused by today's spending limit: run() posts that once and waits.
            if not (self._stop.is_set() or self.retired.is_set() or self._budget_block):
                self.say("errors", "error", f"Iteration ended without a submission: {(text or '')[:500]}", tag)
            return
        last = submits[-1]
        # Reflection: one lesson for the team, from what this attempt actually showed.
        messages.append({"role": "user", "content": (
            "Write ONE lesson for the team from this attempt, in one or two sentences, starting with "
            "KEEP:, AVOID: or TRY:. Be specific (features, parameters, timeframe, regime, failure causes, what "
            "the numbers showed). If the script used library modules, add one line per module: "
            "`LIB <name>: works|broken -- <evidence>`. Then one line starting `TEAM:` saying how you worked with "
            "your teammates this time (whose work you built on, what you handed them, what you would ask of them "
            "next). Reply with only these lines.")})
        try:
            _, lesson, _ = self.converse(messages, [], world.call, tag=tag, max_rounds=0)
        except Exception as exc:  # noqa: BLE001 -- a lesson is a bonus, not a requirement
            lesson = ""
            log(f"{self.model}: reflection failed: {exc!r}")
        lesson = (lesson or "").strip().strip('"')
        # `LIB name: works|broken -- evidence` lines become comments on those modules, tied to
        # this candidate as the proof; the rest is the lesson.
        kept, team_note, verdicts = _reflection_lines(lesson)
        for name, verdict, text in verdicts:
            try:
                request(CONTROL_PLANE, f"/api/projects/{q(self.pid)}/library/{q(name)}/comments", {
                    "verdict": verdict, "text": text, "author": self.model,
                    "candidate_id": last.get("candidate_id")})
            except RuntimeError as exc:
                log(f"{self.model}: library comment on {name} failed: {exc}")
        lesson = "\n".join(kept).strip()
        if lesson and len(lesson) > 8 and not lesson.startswith("(reasoning only"):
            try:
                request(CONTROL_PLANE, f"/api/objectives/{q(oid)}/lessons",
                        {"text": lesson[:1500], "model": self.model, "candidate_id": last.get("candidate_id")})
            except RuntimeError as exc:
                log(f"{self.model}: lesson not saved: {exc}")
        try:
            self.record_collaboration(obj, ctx, world, last, team_note, inbox,
                                      built_on=parents[-1] if parents else None)
        except Exception as exc:  # noqa: BLE001 -- the record is a bonus, never a failure
            log(f"{self.model}: collaboration record failed: {exc!r}")
        log(f"{self.model}: iteration done ({mode}), candidate #{last.get('seq')} {last.get('status')}, "
            f"{usage['tool_calls']} tool calls")

    def report_eval(self, obj: dict, view: dict) -> None:
        tag = {"objective_id": obj["id"], "candidate_id": view.get("candidate_id")}
        seq = view.get("seq")
        if view.get("status") == "error":
            self.say("errors", "error", f"Candidate #{seq} failed to run: {view.get('error')}\n{(view.get('stderr_tail') or '')[-600:]}", tag)
            return
        parts = [f"Candidate #{seq}: in-sample {obj['metric']['kind']} {_fmt(view.get('in_sample_score'))}",
                 f"look-ahead {view.get('lookahead')}", f"rank {view.get('rank')}"]
        if view.get("not_ranked"):
            parts.append(f"not ranked: {view['not_ranked']}")
        if view.get("judge_score") is not None:
            parts.append(f"judge {view['judge_score']}")
        if view.get("contender_for_best"):
            parts.append("contender for best -- audit pending" if obj.get("require_audit") else "NEW BEST")
        self.say("general", "thought", " · ".join(parts), tag)
        if view.get("lookahead") == "fail":
            self.say("errors", "error", f"Candidate #{seq} rejected: {view.get('lookahead_detail')}", tag)
        if view.get("contender_for_best") and not obj.get("require_audit"):
            self.announce(obj, view["candidate_id"])

    def run_task(self, task: dict) -> None:
        task_id = task["id"]
        title = task.get("title") or task_id
        log(f"{self.project['slug']}/{self.model}: claimed {task_id} -- {title[:60]}")
        self.beat("working", force=True)
        self.say("general", "thought", f"Working on: {title}", {"task_id": task_id})

        done = threading.Event()

        def keep_lease() -> None:
            while not done.wait(LEASE_S / 3):
                try:
                    request(BOARD, f"/mb/tasks/{task_id}/extend", {"agent_id": self.agent_id, "lease_s": LEASE_S})
                except RuntimeError as exc:
                    log(f"{self.model}: lease extend failed for {task_id}: {exc}")
                    return

        keeper = threading.Thread(target=keep_lease, daemon=True)
        keeper.start()
        started = time.time()
        try:
            ok, result, usage = self.answer(task)
        finally:
            done.set()
            keeper.join(timeout=5)
        elapsed = time.time() - started

        try:
            request(BOARD, f"/mb/tasks/{task_id}/complete",
                    {"agent_id": self.agent_id, "status": "done" if ok else "failed", "result": result})
        except RuntimeError as exc:
            log(f"{self.model}: could not complete {task_id}: {exc}")
            self.say("errors", "error", f"Lost the lease on {title}: {exc}", {"task_id": task_id})
            self.beat("idle", force=True)
            return

        log(f"{self.model}: {'done' if ok else 'FAILED'} {task_id} in {elapsed:.1f}s "
            f"({usage['tool_calls']} tool calls, {usage['prompt_tokens']}+{usage['completion_tokens']} tokens)")
        self.say("results" if ok else "errors", "result" if ok else "error",
                 f"{'Completed' if ok else 'Failed'}: {title} ({elapsed:.0f}s, "
                 f"{usage['tool_calls']} tool calls)\n\n{result}",
                 {"task_id": task_id, "ok": ok, "seconds": round(elapsed, 1), **usage})
        self.beat("idle", force=True)

    def _start_work(self, label: str) -> bool:
        """Mark this agent busy, THEN look for a drain request. In that order the supervisor
        can never see the agent idle in the instant between its check and the work starting;
        a drain seen here undoes the mark and nothing starts."""
        self.busy, self.busy_since = label, time.time()
        if drain_requested():
            self.busy = None
            return False
        return True

    def _turn(self) -> float:
        """One unit of work: a claimed task, a mentor pass or an objective iteration (or the
        chore the control plane hands out instead). Returns the seconds to rest after it."""
        # The mentor leaves one-off tasks to the searchers: its passes are the standing work.
        task = self.claim() if self.role != "mentor" else None
        if task is not None:
            self.busy = f"task {task.get('title') or task.get('id')}"[:120]
            _act(self).begin("task", task={"id": task.get("id"), "title": task.get("title")})
            self.run_task(task)  # one-off tasks take priority over the standing work
            _act(self).end()
            return 0.0
        obj = self.next_objective()
        if obj is None:
            return POLL_IDLE_S
        what = str(obj.get("title") or obj.get("id"))[:80]
        if self.role == "mentor":
            self.busy = f"mentor pass on {what}"
            self.mentor(obj)
            _act(self).end()
            return 0.0
        self.busy = f"iteration on {what}"
        self.iterate(obj)
        _act(self).end()
        self.beat("idle", force=True)
        return float(obj.get("cooldown_s") or 0)

    def run(self) -> None:
        while not self._stop.is_set() and not self.retired.is_set():
            if not self.agent_id and not self.register():
                self._stop.wait(POLL_IDLE_S * 2)
                continue
            pause = 0.0
            try:
                if self._budget_block:
                    # The last generation was refused for today's spending limit. Starting the
                    # next iteration would fail the same way within seconds.
                    self.hold_for_budget()
                    continue
                self.beat("idle")
                if not self.agent_id:
                    continue
                # A drain (restart-swarm.cmd): start nothing new; the supervisor exits once
                # every agent's running work has finished.
                if not self._start_work("starting"):
                    self._stop.wait(POLL_IDLE_S)
                    continue
                try:
                    pause = self._turn()
                finally:
                    self.busy = None
            except Exception as exc:  # noqa: BLE001 -- a worker must never die on one task
                log(f"{self.model}: unexpected error: {exc!r}")
                # An exception used to leave the record "running" with a pending chat/tool
                # until the NEXT iteration's begin() closed it -- for a retired worker, forever.
                _act(self).end("error", f"unexpected error: {exc!r}"[:400])
                pause = POLL_IDLE_S
            if pause > 0:
                self._stop.wait(pause)
        # The outer loop exited because we were stopped or retired mid-turn (converse also
        # closes the record when it feels the flag, but a converse that never entered -- e.g.
        # a chore-only iteration -- would otherwise leak). Idempotent: end() no-ops on an
        # already-closed record.
        reason = ("retired: model unloaded or worker replaced"
                  if self.retired.is_set() else "stopped: swarm halted")
        _act(self).end("interrupted", reason)
        log(f"{self.project['slug']}/{self.model}: stopped")


# =======================================================================================
# Supervisor
# =======================================================================================
class State:
    """The supervisor's latest view of what is loaded, shared read-only with workers."""

    def __init__(self) -> None:
        self.llms: list[str] = []
        self.forecasters: list[dict] = []
        self._lock = threading.Lock()

    def set(self, llms: list[str], forecasters: list[dict]) -> None:
        with self._lock:
            self.llms, self.forecasters = llms, forecasters

    def get(self) -> tuple[list[str], list[dict]]:
        with self._lock:
            return list(self.llms), list(self.forecasters)


def _mins(seconds: float) -> str:
    return f"{max(0.0, seconds) / 60:.0f} min"


def _bullets(items) -> str:
    return "".join(f"\n    - {x}" for x in items)


class Drain:
    """The supervisor's side of a graceful drain (see DRAIN_FILE).

    tick() is called every few seconds with the live workers and returns True when the runner
    should exit: nothing is running any more, or DRAIN_MAX_S has passed (then `cut` names the
    workers whose work is abandoned). Workers gate themselves (Worker._start_work); this only
    watches, reports and decides when to leave.
    """

    def __init__(self, max_s: float | None = None, report_s: float | None = None) -> None:
        self.max_s = DRAIN_MAX_S if max_s is None else max_s
        self.report_s = DRAIN_REPORT_S if report_s is None else report_s
        self._reset()

    def _reset(self) -> None:
        self.started: float | None = None
        self._reported = 0.0
        self.waited: dict[int, tuple[str, str, float]] = {}  # id(worker) -> (agent, work, since)
        self.cut: list = []
        self.outcome: str | None = None  # "idle" | "timeout" once tick() said exit

    @property
    def active(self) -> bool:
        return self.started is not None

    @staticmethod
    def busy(workers) -> list:
        return [w for w in workers if w.is_alive() and getattr(w, "busy", None)]

    @staticmethod
    def _describe(w, now: float) -> str:
        return f"{w.project.get('slug') or w.pid}/{w.agent_name}: {w.busy} ({_mins(now - (w.busy_since or now))})"

    def status(self, workers, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        busy = self.busy(workers)
        return {"draining": self.active and self.outcome is None, "since": self.started,
                "deadline": (self.started + self.max_s) if self.started else None,
                "running": len(busy), "updated_at": now, "outcome": self.outcome,
                "iterations": [{"agent": w.agent_name, "project": w.pid, "work": w.busy,
                                "since": w.busy_since} for w in busy]}

    def tick(self, workers, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        workers = list(workers)
        if not drain_requested():
            if self.active and self.outcome is None:
                log("drain cancelled (the drain file was removed): agents start new work again")
                self._publish(workers, {**self.status(workers, now), "draining": False, "outcome": "cancelled"})
                self._reset()
            return False
        busy = self.busy(workers)
        for w in busy:
            self.waited.setdefault(id(w), (w.agent_name, w.busy, w.busy_since))
        listed = _bullets(self._describe(w, now) for w in busy)
        if not self.active:
            self.started = self._reported = now
            log(f"DRAIN requested: no new iterations, chores, mentor passes or tasks start; waiting for "
                f"{len(busy)} running to finish (at most {_mins(self.max_s)}, FREESWARM_DRAIN_MAX_S){listed}")
            self._publish(workers, self.status(workers, now))
        if not busy:
            self.outcome = "idle"
            done = _bullets(f"{a}: {work} (ran {_mins(now - since)})" for a, work, since in self.waited.values())
            log(f"DRAIN complete after {_mins(now - self.started)}: nothing running. "
                f"Waited for {len(self.waited)}{done or '.'}" + "\n  exiting (code 0)")
            self._finish(workers, now)
            return True
        if now - self.started >= self.max_s:
            self.outcome = "timeout"
            self.cut = busy
            log(f"DRAIN max wait ({_mins(self.max_s)}) reached with {len(busy)} still running; "
                f"exiting anyway and cutting:{listed}")
            self._finish(workers, now)
            return True
        self._write_status(self.status(workers, now))
        if now - self._reported >= self.report_s:
            self._reported = now
            log(f"draining: {len(busy)} still running, {_mins(now - self.started)} in, "
                f"{_mins(self.started + self.max_s - now)} left before the max wait{listed}")
            self._publish(workers, self.status(workers, now))
        return False

    def _finish(self, workers, now: float) -> None:
        self._publish(workers, {**self.status(workers, now), "draining": False})
        for path in (DRAIN_FILE, DRAIN_FILE + ".status.json"):
            try:
                os.remove(path)
            except OSError:
                pass

    @staticmethod
    def _write_status(doc: dict) -> None:
        """The drain's progress for restart-swarm.cmd (which prints it while it waits)."""
        path = DRAIN_FILE + ".status.json"
        try:
            with open(path + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(doc, fh, default=str)
            os.replace(path + ".tmp", path)
        except OSError:
            pass

    def _publish(self, workers, doc: dict) -> None:
        """Status file, plus board key "swarm_drain" in every project with agents, for the
        console and the coordinator."""
        self._write_status(doc)
        for pid in sorted({w.pid for w in workers}):
            try:
                request(BOARD, "/mb/state/swarm_drain", {"project_id": pid, "value": doc, "updated_by": "swarm runner"},
                        timeout=10, method="PUT")
            except RuntimeError as exc:
                log(f"drain: could not publish the state to the board for {pid} ({exc})")


def _reflection_lines(text: str) -> tuple[list[str], str, list[tuple[str, str, str]]]:
    """A reflection split into (KEEP/AVOID/TRY lesson lines, the TEAM note, LIB verdicts as
    (module, verdict, evidence)). Anything else is dropped. Markdown bullets and bold
    ("- **KEEP:** ...", "* LIB x: works") used to defeat the matching, and TEAM notes and LIB
    verdicts landed in the lessons every model reads."""
    kept, team, verdicts = [], "", []
    for raw in (text or "").splitlines():
        line = re.sub(r"^[\s>*#\-•\d.)]+", "", raw).replace("**", "").replace("__", "").strip()
        if line.upper().startswith("TEAM:"):
            team = line[5:].strip()[:400]
            continue
        m = re.match(r"LIB\s+([a-z_][a-z0-9_]*)\s*:\s*(works|broken|note)\b[\s\-–—:,]*(.*)", line, re.I)
        if m:
            verdicts.append((m.group(1), m.group(2).lower(), m.group(3)[:2000] or m.group(2)))
        elif re.match(r"(KEEP|AVOID|TRY)\b", line, re.I):
            kept.append(line)
    return kept, team, verdicts


def _mentor_to_be(plan: dict, model: str) -> bool:
    """Whether `model` searches now only because the models that would let it mentor are not
    ready yet: the free head of the ideas ladder, on Auto (an operator's Search/Both is kept)."""
    ladder = plan.get("ladder") or []
    if not ladder or ladder[0].get("model") != model or ladder[0].get("kind") == "external":
        return False
    entry = next((m for m in plan.get("search") or [] if m.get("model") == model), {})
    return not str(entry.get("why") or "").startswith("you set it")


_INSTANCE_LOCK = None                    # held for the process's life (see _single_instance)


def _single_instance(path: str | None = None) -> bool:
    """True if this is the only runner. Two runners double every agent (10-01 20:32: a restart left a
    second one running for a minute); an OS file lock is released by the OS when the process dies,
    so a killed runner never blocks the next one. The holder's pid is written after the locked byte:
    the same process may take it again (tests load the module more than once)."""
    global _INSTANCE_LOCK
    path = path or os.environ.get("FREESWARM_RUNNER_LOCK") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".swarm_runner.lock")
    fh = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        try:
            with open(path, "r") as other:
                other.seek(16)
                holder = other.read(16).strip()
        except OSError:
            holder = ""
        return holder == str(os.getpid())
    fh.seek(16)
    fh.truncate()
    fh.write(f"{os.getpid():<16}")
    fh.flush()
    _INSTANCE_LOCK = fh
    return True


def main() -> int:
    if not _single_instance():
        log("swarm runner: another runner is already running here -- not starting a second one "
            "(restart it with ui\\restart-swarm.cmd)")
        return 3
    log(f"swarm runner: control plane {CONTROL_PLANE}, board {BOARD}")
    if AGENT_USER and not STATIC_TOKEN:
        AUTH.refresh()
    stop = threading.Event()
    state = State()
    workers: dict[tuple[str, str, int], Worker] = {}
    warned_empty = False
    started_at = time.time()
    stale_closed = False
    drain = Drain()
    retiring: list[Worker] = []  # retired agents still finishing (or abandoning) a turn
    if drain_requested():
        # Left by a runner that was killed while draining; this start is the restart it asked for.
        log(f"removing a drain request left from before this start ({DRAIN_FILE})")
        for path in (DRAIN_FILE, DRAIN_FILE + ".status.json"):
            try:
                os.remove(path)
            except OSError:
                pass

    def live() -> list[Worker]:
        retiring[:] = [w for w in retiring if w.is_alive()]
        return list(workers.values()) + retiring

    try:
        while True:
            try:
                projects = request(CONTROL_PLANE, "/api/projects").get("projects", [])
                if not stale_closed:
                    # Iterations an earlier runner left "running" are over: close them, or the
                    # monitor reads them as agents stuck mid-iteration (bugs #69, #70, #128, #156).
                    try:
                        n = request(CONTROL_PLANE, "/api/agents/activity/close-stale",
                                    {"before": started_at, "reason": "the swarm runner was restarted"},
                                    timeout=30).get("closed", 0)
                        if n:
                            log(f"closed {n} iteration record(s) an earlier runner left open")
                        stale_closed = True
                    except RuntimeError as exc:
                        log(f"could not close stale iteration records ({exc}); will retry")
                loaded = request(CONTROL_PLANE, "/api/engines").get("loaded", []) or []
                llms = [m.get("model") or m.get("served_name") for m in loaded if m.get("ready", True)]
                llms = [m for m in llms if m]
                # Models on paired computers report their window with the list; local ones
                # are read from the console's engine stats below. External (hosted) models carry
                # both their context and a provider-published `max_output` cap -- we need both,
                # because without them the runner falls back to DEFAULT_CONTEXT=8192 and the
                # `max_tokens = ctx - est - 128` formula floors at 256 the moment the real
                # prompt (25,720 tokens for one Groq bug) exceeds the guessed window. The
                # provider still accepts the prompt (its true window is 128K), but returns
                # 256 tokens of prose or a mid-token tool call.
                for m in loaded:
                    mid = m.get("model")
                    if not mid or mid in _learned:
                        continue
                    ctx_val = int(m.get("context") or 0)
                    if m.get("remote"):
                        # Below 4K (or missing) is a misreport, not a window an agent could use
                        # (an engine once gave DeepSeek-V4's 128K as "1024"): squeezing every
                        # prompt into it and capping answers at 256 tokens is worse than
                        # learning the real window from the first overflow error.
                        if ctx_val >= 4096:
                            _context[mid] = ctx_val
                        else:
                            _context[mid] = UNVERIFIED_CONTEXT
                    elif m.get("external"):
                        # Groq/OpenRouter publish exact numbers; take them as truth.
                        if ctx_val >= 4096:
                            _context[mid] = ctx_val
                        cap = int(m.get("max_output") or 0)
                        if cap > 0:
                            _max_output[mid] = cap
                try:
                    ts = [i for i in request(CONTROL_PLANE, "/api/ts").get("instances", []) if i.get("state") == "running"]
                except RuntimeError:
                    ts = []
                # The usable window per engine: min(model max, KV pages x page size).
                try:
                    for e in request(CONTROL_PLANE, "/api/console").get("engines", []) or []:
                        st = e.get("stats") or {}
                        kv = st.get("kv") or {}
                        model_ctx = (st.get("model") or {}).get("ctx")
                        kv_tokens = (kv.get("total_pages") or 0) * (kv.get("page_size") or 1)
                        window = min(x for x in (model_ctx, kv_tokens) if x) if (model_ctx or kv_tokens) else None
                        if e.get("model_id") and window:
                            _context[e["model_id"]] = int(window)
                except RuntimeError:
                    pass
                state.set(llms, ts)
            except RuntimeError as exc:
                log(f"cannot sync ({exc}); retrying")
                projects = None

            if projects is not None:
                wanted: dict[tuple[str, str, int], dict] = {}
                mentor_slot = -1  # key slot of a model's mentor agent (searchers use 0..n-1)
                for project in projects:
                    # A project whose swarm is switched off gets no agents; any it had are
                    # retired below because they drop out of `wanted`.
                    if not project.get("swarm_enabled", True):
                        continue
                    # Which models search is the console's policy (app/swarm_policy.py): every free
                    # model the project allows, plus ticked external models whose SWE-bench score
                    # matches the free ones. Stronger externals are kept for escalation.
                    # Hosted models run several agents each (plan "agents"), local engines one.
                    try:
                        plan = request(CONTROL_PLANE, f"/api/projects/{q(project['id'])}/swarm/plan")
                        searchers = [(m["model"], int(m.get("agents") or 1)) for m in plan.get("search", [])]
                        mentors = [m["model"] for m in plan.get("mentors", [])]
                        if time.time() - started_at < STARTUP_GRACE_S:
                            searchers = [(m, n) for m, n in searchers if not _mentor_to_be(plan, m)]
                    except RuntimeError as exc:
                        log(f"swarm plan for {project['id']} unavailable ({exc}); using free models only")
                        searchers = [(m, 1) for m in llms if not is_external(m) and permitted(project, m)]
                        mentors = []
                    for model, n in searchers:
                        if model in llms:
                            for slot in range(max(1, n)):
                                wanted[(project["id"], model, slot)] = project
                    for model in mentors:
                        if model in llms:
                            wanted[(project["id"], model, mentor_slot)] = project
                for key, project in wanted.items():
                    w = workers.get(key)
                    if (w is None or not w.is_alive()) and drain_requested():
                        continue  # draining: no new agents either
                    if w is None or not w.is_alive():
                        w = Worker(project, key[1], stop, state.get, slot=max(0, key[2]),
                                   role="mentor" if key[2] == mentor_slot else "search")
                        workers[key] = w
                        w.start()
                    else:
                        w.project = project  # pick up renamed projects / new SQL / model lists
                for key in [k for k in workers if k not in wanted]:
                    log(f"{key[1]}: no longer allowed/loaded for project {key[0]}; retiring agent")
                    w = workers.pop(key)
                    w.retired.set()
                    retiring.append(w)
                if not wanted and not warned_empty:
                    log("no models loaded -- queued tasks will wait until one is loaded")
                    warned_empty = True
                elif wanted:
                    warned_empty = False
            # Wait for the next resync, watching for a drain every few seconds meanwhile.
            resync_at = time.time() + ENGINE_RESYNC_S
            leave = False
            while not leave:
                leave = drain.tick(live())
                left = resync_at - time.time()
                if leave or left <= 0:
                    break
                leave = stop.wait(min(DRAIN_POLL_S, left))
            if leave:
                break
    except KeyboardInterrupt:
        log("shutting down")
    finally:
        stop.set()
        for w in drain.cut:
            # Abandoned at the drain's max wait: close the record now rather than leave it to
            # the next runner's close-stale sweep.
            _act(w).end("interrupted", f"cut: the swarm runner's drain reached its max wait ({_mins(drain.max_s)})")
        for w in workers.values():
            w.join(timeout=3)
        if drain.outcome and _POSTER is not None and not _POSTER.flush(15):
            log("drain: some agent-inspector updates were not posted before exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
