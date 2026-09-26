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
   look-ahead test); a failed run may be fixed and resubmitted once;
4. reflect: write one lesson for the team (KEEP / AVOID / TRY), which every later iteration
   reads. Lessons are periodically consolidated by an agent into a short list.

Chores come first when the control plane hands one out: auditing a would-be champion's code
(done by a DIFFERENT loaded model when there is one), and consolidating lessons. This is the
"agents get smarter" loop: better parents to build on, a growing memory of what works and
what does not, and a critic between a good number and the title.

stdlib only (urllib + threading), matching the connectors.
"""

from __future__ import annotations

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

CONTROL_PLANE = os.getenv("FREESWARM_API_URL", "http://127.0.0.1:8000").rstrip("/")
BOARD = os.getenv("FREESWARM_BOARD_URL", "http://127.0.0.1:8100").rstrip("/")

STATIC_TOKEN = os.getenv("FREESWARM_API_TOKEN", "").strip()
AGENT_USER = os.getenv("FREESWARM_AGENT_USER", "").strip()
AGENT_PASSWORD = os.getenv("FREESWARM_AGENT_PASSWORD", "")

POLL_IDLE_S = float(os.getenv("FREESWARM_SWARM_POLL_S", "3"))
ENGINE_RESYNC_S = float(os.getenv("FREESWARM_SWARM_RESYNC_S", "15"))
HEARTBEAT_S = 20.0
LEASE_S = int(os.getenv("FREESWARM_SWARM_LEASE_S", "300"))
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
_context: dict[str, int] = {}  # model -> usable tokens, from engine stats or a refusal
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


def _est_tokens(messages: list[dict], tools: list[dict], scale: float = 1.0) -> int:
    chars = len(json.dumps(messages, default=str)) + len(json.dumps(tools, default=str))
    return int((chars / CHARS_PER_TOKEN + 64) * scale)


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


def _text_tool_calls(text: str, names: set[str]) -> list[tuple[str, dict]]:
    """Tool calls a model wrote as TEXT instead of making them -- recovered so the work counts.

    Seen from the swarm's models: gpt-oss's own channel syntax (``<|start|> to=submit_candidate
    <|message|>{...}``), XML-ish ``<invoke name=...><parameter name=...>`` blocks (sometimes both
    at once), and a bare ``{"name": ..., "arguments": {...}}``. Only known tool names count, and
    an XML parameter only when it is closed -- output cut off mid-script is not submitted.
    """
    out: list[tuple[str, dict]] = []
    for name, body in _XML_INVOKE.findall(text or ""):
        if name in names:
            args = {k: v.strip("\n") for k, v in _XML_PARAM.findall(body)}
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
MAX_SUBMITS = 2  # a failed run may be fixed and resubmitted once per iteration
# Experiments per iteration. Without a cap Qwen spent whole iterations in 10+ private
# run_python experiments and never saved or submitted anything the team could use.
MAX_EXPERIMENTS = int(os.getenv("FREESWARM_SWARM_EXPERIMENTS", "6"))
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
            retry_auth: bool = True):
    """JSON request. Retries once after refreshing credentials on a 401."""
    headers = {"Content-Type": "application/json", **AUTH.header(), **_agent_header()}
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"{base}{path}", data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        if exc.code == 401 and retry_auth and AUTH.refresh():
            return request(base, path, payload, timeout=timeout, retry_auth=False)
        body = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(body).get("detail", body)
        except ValueError:
            detail = body
        raise RuntimeError(f"{path} -> {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach {base}{path}: {exc.reason}") from None
    except (TimeoutError, OSError, http.client.HTTPException, ValueError) as exc:
        # A read that times out AFTER the connection is established raises socket.timeout
        # straight from getresponse(), not URLError -- as do a reset mid-read and a truncated
        # body. Every caller guards against RuntimeError, so transport failures are normalised
        # to that here. Letting one escape killed the whole runner: a single slow reply from a
        # busy control plane took the swarm down with it.
        raise RuntimeError(f"{path} failed: {type(exc).__name__}: {exc}") from None


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
            return request(CONTROL_PLANE, f"/api/projects/{pid}/data/query",
                           {"sql": args.get("sql", ""), "max_rows": int(args.get("_max_rows") or 200)}, timeout=180)
        if name == "list_sql_tables" and self.sql:
            return {"database": self.sql["database"], "tables": self.sql["tables"]}
        if name == "describe_sql_table" and self.sql:
            return request(CONTROL_PLANE, f"/api/projects/{pid}/sql/describe?table={q(str(args.get('table', '')))}")
        if name == "query_sql" and self.sql:
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
            r = request(CONTROL_PLANE, "/v1/chat/completions", {
                "model": model, "max_tokens": 4096, "stream": False,
                "messages": [{"role": "user", "content": str(args.get("prompt", ""))}],
            }, timeout=GENERATION_TIMEOUT_S)
            choice = (r.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            return {"model": model, "answer": msg.get("content") or "",
                    "finish_reason": choice.get("finish_reason"), "usage": r.get("usage")}
        if "__" in name and any(t["function"]["name"] == name for t in self.mcp):
            return request(CONTROL_PLANE, f"/api/mcp/call?project_id={pid}",
                           {"tool": name, "arguments": args}, timeout=300)
        return {"error": f"unknown tool {name!r}"}


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
        self.saved: list[str] = []   # library modules saved this iteration
        self.sent: list[dict] = []   # team_post messages this iteration
        self.feature_views = set()
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
                "Run an experimental Python script in the offline sandbox (pandas, numpy, scipy). "
                "Load data with `import ft; df = ft.load('<view>')`." + split_note +
                " Print what you want to see; nothing is scored.",
                {"code": {"type": "string"}}, ["code"]),
            _fn("get_candidate",
                "Full code and in-sample results of an earlier candidate, by its number (seq) or id.",
                {"candidate": {"type": "string"}}, ["candidate"]),
            *([
                _fn("forecast",
                    "Look at one forecast of a column's most recent IN-SAMPLE values from a loaded "
                    "time-series model (median and 10/90% quantiles per step). For exploring only -- a "
                    "strategy uses forecasts through forecast_feature.",
                    {"column": {"type": "string"}, "dataset": {"type": "string", "description": "view name; default: the objective's dataset"},
                     "horizon": {"type": "integer"}, "context": {"type": "integer", "description": "history points (default 512)"},
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
                    "series the model can actually forecast. Takes up to a few minutes; cached after.",
                    {"column": {"type": "string", "description": "one series: a column or an expression"},
                     "columns": {"type": "array", "items": {"type": "string"}, "description": "several series"},
                     "dataset": {"type": "string", "description": "view name; default: the objective's dataset"},
                     "horizon": {"type": "integer", "description": "bars ahead (default 12)"},
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
                 "reply_to": {"type": "integer", "description": "message number you are answering"}},
                ["text"]),
            *([_fn("field_scan",
                   "Screen EVERY field of the dataset (all the greeks, walls, imbalances, surface, IV...) against "
                   "the forward return on in-sample data: rank correlation (IC) of each field's level and of its "
                   "change over the horizon, optionally within each regime of a regime module. Returns the top "
                   "fields; the full scan is stored for the team.",
                   {"horizon": {"type": "integer", "description": "bars ahead (10 s bars; default 30 = 5 min)"},
                    "regime": {"type": "string", "description": "optional regime module to split the IC by"},
                    "columns": {"type": "array", "items": {"type": "string"}, "description": "optional subset"}})]
              if self.objective["metric"].get("price_column") else []),
            *([_fn("deci_plot",
                   "Decile study of ONE signal on in-sample data: the mean forward return (bps), hit rate and t "
                   "in each of the signal's 10 deciles, on 10s/20s/30s/1min/5min bars at 1/3/6/12 bars ahead, "
                   "with monotonicity (Spearman), the top-minus-bottom spread and its t, and whether it holds in "
                   "each of 3 sub-periods. Deciles use ROLLING edges from past sessions only (no look-ahead). "
                   "Stored and cached: a study already run comes back at once. Call with no signal to list the "
                   "studies the team already has -- check them before running a new one.",
                   {"signal": {"type": "string", "description": "a column (GEX), an expression (GEX / Pinning_TotalAbsGex), "
                                                                "or a forecast feature column fc_<name>:fc_change"},
                    "timeframes": {"type": "array", "items": {"type": "string"}, "description": "default 10s,20s,30s,1min,5min"},
                    "horizons": {"type": "array", "items": {"type": "integer"}, "description": "bars of the timeframe ahead; default 1,3,6,12"},
                    "window_days": {"type": "integer", "description": "past sessions the decile edges come from (default 20)"}})]
              if self.objective["metric"].get("price_column") else []),
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
                   ["regime"])] if self.objective["metric"].get("price_column") else []),
            _fn("submit_candidate",
                "Submit your candidate for scoring. Returns the evaluation (in-sample metrics, "
                "look-ahead verdict, rank). Call once your script is complete.",
                {"code": {"type": "string", "description": "the complete Python script"},
                 "rationale": {"type": "string", "description": "the hypothesis: what you changed or tried, and why it should generalise"},
                 "answer": {"type": "string", "description": "for judged objectives: the answer text"},
                 "idea": {"type": "integer", "description": "the number of the mentor idea this candidate tests, if any"}},
                ["rationale"]),
        ]
        return base

    def call(self, name: str, args: dict) -> Any:
        oid = q(self.oid)
        if name == "query_data":
            return request(CONTROL_PLANE, f"/api/objectives/{oid}/data/query",
                           {"sql": args.get("sql", ""), "max_rows": 200}, timeout=180)
        if name == "get_candidate":
            ref = str(args.get("candidate", "")).lstrip("#c ")
            cands = request(CONTROL_PLANE, f"/api/objectives/{oid}/candidates?order=recent&limit=500").get("candidates", [])
            hit = next((c for c in cands if c["id"] == ref or str(c["seq"]) == ref), None)
            if hit is None:
                return {"error": f"no candidate {ref!r}"}
            full = request(CONTROL_PLANE, f"/api/objectives/{oid}/candidates/{q(hit['id'])}")
            return {"seq": full["seq"], "model": full["model"], "rationale": full["rationale"],
                    "code": full["code"], "status": full["status"],
                    "in_sample": (full.get("metrics") or {}).get("in_sample"),
                    "lookahead": full.get("lookahead"), "problem": full.get("score_note")}
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
                return {"error": (f"experiment budget used ({MAX_EXPERIMENTS} runs this iteration). Turn what works "
                                  "into a library module with library_save (with a test) and call submit_candidate "
                                  "with a script that imports it.")}
            left = MAX_EXPERIMENTS - self.experiments
            out = request(CONTROL_PLANE, f"/api/objectives/{oid}/python",
                          {"code": str(args.get("code", "")), "timeout_s": 180}, timeout=400 + FORECAST_BUILD_ALLOWANCE_S)
            return {**out, "experiments_left": left}
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
            meta = {"objective_id": self.oid, "team": True}
            if to not in ("all", ""):
                meta["to"] = to
            if args.get("reply_to"):
                meta["reply_to"] = int(args["reply_to"])
            doc = request(BOARD, "/mb/messages", {
                "project_id": self.pid, "channel": ch, "author": self.self_model, "kind": "chat",
                "content": (f"@{to.split('/')[-1]} " if meta.get("to") else "") + text, "meta": meta,
                **({"reply_to": meta["reply_to"]} if meta.get("reply_to") else {})})
            self.sent.append({"to": meta.get("to", "all"), "channel": ch, "reply_to": meta.get("reply_to"),
                              "text": text[:160]})
            return {"posted": ch, "to": meta.get("to", "all"), "seq": (doc or {}).get("seq")}
        if name == "field_scan":
            return request(CONTROL_PLANE, f"/api/objectives/{oid}/field-scan", {
                "horizon": int(args.get("horizon") or 30), "regime": args.get("regime") or None,
                "columns": args.get("columns") or None, "author": self.self_model}, timeout=600)
        if name == "deci_plot":
            if not str(args.get("signal") or "").strip():
                return request(CONTROL_PLANE, f"/api/objectives/{oid}/deci-plots?compact=true")
            body = {"signal": str(args["signal"]), "timeframes": args.get("timeframes") or None,
                    "horizons": [int(h) for h in args.get("horizons") or []] or None,
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
            out = request(CONTROL_PLANE, lib, {
                "name": str(args.get("name", "")).strip(), "kind": args.get("kind") or "util",
                "description": str(args.get("description", ""))[:2000], "code": str(args.get("code", "")),
                "test_code": str(args.get("test_code", "")), "note": str(args.get("note", ""))[:2000],
                "author": self.self_model, "objective_id": self.oid}, timeout=400)
            if isinstance(out, dict) and out.get("saved"):
                self.saved.append(out.get("name") or str(args.get("name")))
            return out
        if name == "library_comment":
            cid = None
            ref = str(args.get("candidate") or "").lstrip("#c ")
            if ref:
                cands = request(CONTROL_PLANE, f"/api/objectives/{oid}/candidates?order=recent&limit=500").get("candidates", [])
                cid = next((c["id"] for c in cands if c["id"] == ref or str(c["seq"]) == ref), None)
            return request(CONTROL_PLANE, f"{lib}/{q(str(args.get('name', '')))}/comments", {
                "verdict": args.get("verdict") or "note", "text": str(args.get("text", ""))[:8000],
                "author": self.self_model, "candidate_id": cid})
        if name == "regime_map":
            return request(CONTROL_PLANE, f"/api/objectives/{oid}/regime-map", {
                "regime": str(args.get("regime", "")), "signals": args.get("signals") or None,
                "author": self.self_model}, timeout=600)
        if name == "submit_candidate":
            return self.on_submit(args)
        return super().call(name, args)


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


def _clip(value: Any, limit: int = TOOL_RESULT_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars -- narrow the query or aggregate]"


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


_SUBMIT_KEYS = ("candidate_id", "seq", "status", "in_sample_score", "lookahead", "lookahead_detail", "rank",
                "not_ranked", "contender_for_best", "judge_score", "error")


def _result_brief(name: str, out: Any) -> Any:
    if isinstance(out, dict) and name == "submit_candidate" and "error" not in out:
        return {k: _trim(out.get(k), 600) for k in _SUBMIT_KEYS if out.get(k) is not None}
    text = out if isinstance(out, str) else json.dumps(out, default=str)
    return _head_tail(text, ACTIVITY_RESULT_CHARS)


class _ActivityPoster(threading.Thread):
    """Posts the latest record of each agent; a burst of changes becomes one post."""

    def __init__(self) -> None:
        super().__init__(name="activity-poster", daemon=True)
        self._pending: dict[str, dict] = {}
        self._cv = threading.Condition()
        self._warned = False

    def put(self, key: str, doc: dict) -> None:
        with self._cv:
            self._pending[key] = doc
            self._cv.notify()

    def run(self) -> None:
        while True:
            with self._cv:
                while not self._pending:
                    self._cv.wait()
                batch, self._pending = self._pending, {}
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
                    time.sleep(60)
                    break
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

    def end(self, status: str | None = None) -> None:
        try:
            self._close(status)
        except Exception as exc:  # noqa: BLE001
            log(f"{self.w.agent_name}: activity record: {exc!r}")

    def _close(self, status: str | None) -> None:
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
            chat = {"at": self._chat_t0, "model": payload.get("model"), "seconds": round(now - self._chat_t0, 1),
                    "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                    "finish": choice.get("finish_reason"), "error": (error or "")[:600] or None}
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

    def tool_done(self, name: str, args: Any, out: Any, ok: bool) -> None:
        try:
            rec = self.rec
            if rec is None:
                return
            self._event({"kind": "tool", "at": self._tool_t0, "name": name, "args": _trim(args), "ok": bool(ok),
                         "seconds": round(time.time() - self._tool_t0, 1), "result": _result_brief(name, out)})
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
                f"{r['signal']} h{r['h']} top-bottom {(r['spread_bps'] or 0):+.2f} bps (t {r['t']}, rho {r['rho']}, "
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
                lines.append(f"  {s['signal']} {s['timeframe']} h{s['h']}: [{curve}] {s['shape']}; top-bottom "
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
    if kind in ("sharpe", "sortino", "calmar", "total_return", "cagr", "max_drawdown"):
        if positions:
            lines.append(
                f"- Score: {ctx['metric_label']} of DAILY returns, higher is better. Your script reports "
                f"POSITIONS with ft.report_positions(series indexed by bar timestamp) -- the position decided "
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
                "`diagnosis` says which one to fix -- read it before your next change. Going short instead of "
                "long (flipping) is allowed: positions may be negative.")
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
            "- Look-ahead test: every submission is re-run with the data cut seconds to minutes AFTER its own "
            "trades (and at the split); if any earlier position changes, it is rejected and never ranked. Decide "
            "each position from rows up to and including its timestamp only: no shift(-1), no centred windows, no "
            "full-sample mean/std/quantiles or models fit on all rows -- use rolling or expanding windows. With "
            "resampled bars, date each bar's values at the moment they are COMPLETE: a 15-min bar built from rows in "
            "[T, T+15m) is known at T+15m, so its position must be stamped at T+15m or later (resample with "
            "label='right', closed='left'); stamping it at T leaks up to 15 minutes of the future.")
    lines += ["", "CANDIDATE CONTRACT",
              "- A complete Python script run offline (pandas, polars, numpy, scipy, pyarrow). Load data ONLY with "
              "`import ft; df = ft.load(\"<view>\")` (pandas) or `ft.load_pl(\"<view>\", columns=[...])` (polars -- "
              "several times faster on the 700k-bar data; .to_pandas() where a helper needs pandas). Datasets: " + ", ".join(
                  (ctx.get("datasets") or []) + [f["view"] for f in ctx.get("features") or []]) + ".",
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
              "Add `ft.report_regime(reg, signal=df['IntrVol'])` to also plot the series the regime came from."]
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
    # Stable first (the engine's prefix cache reuses it across iterations), what changes
    # every iteration last: the leaderboard, recent attempts, messages, teammates, the assignment.
    fields = ctx.get("fields") or {}
    if fields:
        total = sum(len(f["columns"]) for f in fields.values())
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
            lines.append(f"- {m['name']} [{m['kind']} v{m['version']}]: {m['description'][:200]} -- {ev}")
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
    if ctx.get("recent"):
        lines += ["", "RECENT ATTEMPTS (do not repeat these):"]
        for c in ctx["recent"]:
            status = c["status"] if not c.get("problem") else f"{c['status']}: {c['problem']}"
            lines.append(f"- candidate {c['seq']}{_ens(c)} ({c['model']}, {status}, look-ahead {c.get('lookahead')}"
                         f"{', rank ' + str(c['rank']) if c.get('rank') else ''}): {c['rationale'][:200]}")
    inbox = ctx.get("inbox") or []
    if inbox:
        lines += ["", "MESSAGES TO YOU from teammates -- answer them with team_post(to=<sender>, reply_to=<number>):"]
        lines += [f"- #{m['seq']} from {m['from']} ({m['minutes_ago']} min ago): {m['text']}" for m in inbox]
    mates = ctx.get("teammates") or []
    lines += ["", "TEAMMATES RIGHT NOW (#planning, last 45 min) -- pick a different direction or build on theirs:"]
    lines += [f"- {m['who']} ({m['minutes_ago']} min ago): {m['text']}" for m in mates] or ["- (no plans posted)"]
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
                         "good BUILD. Its code:")
            lines.append("```python\n" + (parent.get("code") or "")[:7000] + "\n```")
    elif parent:
        lines.append(
            f"IMPROVE candidate {parent['seq']} (rank {parent['rank']}, in-sample {_fmt(parent.get('in_sample_score'))}). "
            "Make ONE focused change you expect to generalise -- a better signal, a filter, a regime condition, "
            "position sizing (e.g. inverse-volatility or conviction sizing set at entry with ft.size) or a risk "
            "rule -- and keep what works. Its rationale: " + (parent.get("rationale") or "")[:600])
        if parent.get("diagnosis"):
            lines.append("What its result says (in-sample, computed by the harness): " + parent["diagnosis"])
        lines.append("```python\n" + (parent.get("code") or parent.get("answer") or "")[:9000] + "\n```")
    else:
        lines.append("EXPLORE: propose an approach genuinely different from those above -- a different signal "
                     "family, horizon or feature of the data. Start by looking at the data if you need to.")
    lines += ["", "Steps: (1) team_board, library_list and the lessons -- learn from the team first; (2) team_post "
              "your plan to #planning; (3) investigate with your tools (at most "
              f"{MAX_EXPERIMENTS} run_python experiments); (4) save reusable parts with library_save; (5) call "
              "submit_candidate with the complete script and your hypothesis. If the script failed you may fix it "
              "and resubmit once. Then stop."]
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
        f"Metric: {brief['metric_label']} of daily returns; positions are marked to market by the harness on "
        f"{o['metric'].get('price_column')} with {o['metric'].get('cost_bps')} bps per unit of position change.",
        "", "TEAM HABITS over the last candidates (counts):", _j(brief.get("habits"), 800),
        "(results: 'edge given away by costs' = right direction but trades too often; 'points the wrong way' = "
        "flipping every position would score better; 'no edge' = the idea does not work. changes_vs_parent: "
        "'parameters only' = only numbers changed.)",
        "", "LEADERBOARD (ranked on consistency: the weaker of in-sample and the hidden holdout, times equity-curve "
        "smoothness; in-sample shown, with the harness's diagnosis):",
    ]
    lines += [f"- #{c['seq']} {c['model']} in-sample {c['in_sample']}: {c['rationale'][:220]} || {c.get('diagnosis') or ''}"
              for c in brief.get("leaderboard") or []]
    lines += ["", "RECENT ATTEMPTS (newest first):"]
    lines += [f"- #{c['seq']} {c['status']} in-sample {c['in_sample']} idea={c.get('idea_id')} change={c.get('change')} "
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
        lines += ["", "MESSAGES TO YOU (answer each in `replies`):"]
        lines += [f"- [{m['seq']}] from {m['from']}: {m['text']}" for m in inbox]
    lines += [
        "", "Reply with ONE JSON object and nothing else:",
        '{"directions": [{"idea": "...", "hypothesis": "why it should work, in market terms", '
        '"test": "the first concrete experiment and what result would FALSIFY it", "avoid": "the brute-force trap to avoid"}],',
        ' "coaching": "3-6 short lines to the whole team: what the evidence says they are doing wrong or should '
        'stop, which ideas/forecasts to build on or drop, citing numbers from above",',
        ' "replies": [{"reply_to": <message number>, "to": "<author>", "text": "..."}],',
        ' "forecasts": [{"column": "<series to forecast>", "inputs": ["<columns the model reads>"], "horizon": 6, '
        '"every": 0, "model": "<a loaded forecaster>", "why": "what building it would teach us"}]}',
        f"At most {MENTOR_MAX_DIRECTIONS} directions -- conceptually different from each other and from what failed; "
        f"at most {MENTOR_MAX_FORECASTS} forecasts, only where the scoreboard suggests one could help (prefer "
        "Chronos-2 with input columns; do not repeat a recipe already on the scoreboard). Be concrete and brief.",
    ]
    return "\n".join(lines)


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
        act = _act(self)  # the agent inspector: what was asked, tokens, what came back
        act.chat_start(payload)
        try:
            r = request(CONTROL_PLANE, "/v1/chat/completions", payload, timeout=GENERATION_TIMEOUT_S)
        except RuntimeError as exc:
            act.chat_done(payload, None, str(exc))
            if payload.get("model") == self.model and _spending_limited(str(exc)):
                self._budget_block = str(exc)
            raise
        act.chat_done(payload, r)
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
                 final_prompt: str = "Tool budget used up. Give your final answer now from what you have.",
                 nudge=lambda text: None,
                 ) -> tuple[bool, str, dict]:
        """Run the model with tools until it answers (or `done()` says the job is finished).

        `messages` is extended in place, so a caller can continue the conversation after.
        `tag` is attached to every board post (task_id / objective_id).
        """
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "tool_calls": 0, "rounds": 0}
        core_tools = [t for t in tools if "__" not in t["function"]["name"]]
        tool_names = {t["function"]["name"] for t in tools}
        # One retry per turn for a model that spends its whole output budget thinking (a
        # reasoning model such as DeepSeek-V4): told to act, with the rest of the window to do it.
        length_retry = {"used": False, "boost": False}

        for rnd in range(max_rounds + 1):
            # Switching the swarm off, or retiring this agent, has to be felt inside a turn
            # as well as between turns. An agent turn is many rounds of thinking and tool
            # calls and can run for minutes; checking only at the top of the worker loop
            # meant "swarm off" left every agent generating until it happened to finish,
            # which reads as the switch doing nothing.
            if self._stop.is_set() or self.retired.is_set():
                log(f"{self.model}: stopping mid-turn (round {rnd})")
                return False, "", usage
            final_round = rnd == max_rounds or not tools
            if rnd == max_rounds and tools:
                messages.append({"role": "user", "content": final_prompt})
            result = None
            # Three independent retry budgets for one round: context overflow (compact and
            # resend, up to 3 times), provider rate limit (wait the hinted time and resend
            # the same prompt), malformed tool call (tell the model and let it try again).
            overflows = waits = bad_calls = 0
            while True:
                ctx = context_for(self.model)
                scale = _scale.get(self.model, 1.0)
                offered = [] if final_round else tools
                # Fit the prompt to the window (in calibrated tokens), leaving room to answer.
                offered = _compact(messages, offered, max(1024, ctx - ANSWER_RESERVE),
                                   [] if final_round else core_tools, scale)
                if not final_round:
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
                payload = {
                    "model": self.model, "messages": messages, "stream": False,
                    "max_tokens": max(256, min(budget, ctx - est - 128)),
                }
                if offered:
                    payload["tools"] = offered
                try:
                    result = self._generate(payload)
                    break
                except RuntimeError as exc:
                    err = str(exc)
                    wait = _rate_limit_wait(err)
                    if wait is not None and waits < RATE_LIMIT_RETRIES:
                        # Provider rate limit (not our budget): the prompt and every paid round
                        # before it are still good -- wait and resend instead of discarding them.
                        waits += 1
                        log(f"{self.agent_name}: provider rate limit; retrying in {wait:.1f}s "
                            f"({waits}/{RATE_LIMIT_RETRIES})")
                        if self._stop.wait(wait) or self.retired.is_set():
                            return False, "", usage
                        continue
                    failed = _bad_tool_call(err)
                    if failed is not None and bad_calls < BAD_TOOL_CALL_RETRIES and offered:
                        # The provider rejected a malformed tool call before it reached us.
                        # Say so and let the model try again; the round's context is intact.
                        bad_calls += 1
                        snippet = f"\nYour rejected output began: {failed[:400]}" if failed else ""
                        messages.append({"role": "user", "content": (
                            "Your last tool call was malformed and the provider rejected it (it did not "
                            "match the tool's schema). Emit exactly ONE valid tool call whose arguments are "
                            "a JSON object matching that tool's parameters -- no text around it." + snippet)})
                        log(f"{self.agent_name}: malformed tool call rejected by the provider; "
                            f"asking again ({bad_calls}/{BAD_TOOL_CALL_RETRIES})")
                        continue
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
            if not calls and tool_names:
                # A tool call written as text (gpt-oss does this, and every model does in the
                # final round, which offers no tools but asks for submit_candidate): run it.
                found = _text_tool_calls(msg.get("content") or "", tool_names)
                if found:
                    salvaged = True
                    calls = [{"id": f"text_{rnd}_{i}", "type": "function",
                              "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(found)]
                    log(f"{self.model}: recovered {len(calls)} tool call(s) written as text "
                        f"({', '.join(n for n, _ in found)})")
            if calls and (not final_round or salvaged):
                messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
                for i, tc in enumerate(calls):
                    fn = tc.get("function") or {}
                    name = fn.get("name") or "?"
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except ValueError:
                        args = {}
                    usage["tool_calls"] += 1
                    self.say("general", "thought", f"→ {name}: {_summarize_args(name, args)}",
                             {**tag, "tool": name, "round": rnd + 1})
                    _act(self).tool_start(name, args)
                    t0 = time.time()
                    try:
                        out = call(name, args if isinstance(args, dict) else {})
                        ok = not (isinstance(out, dict) and "error" in out)
                    except RuntimeError as exc:
                        out, ok = {"error": str(exc)}, False
                    _act(self).tool_done(name, args, out, ok)
                    if not ok:
                        self.say("errors", "error", f"{name} failed: {str(out.get('error'))[:500]}",
                                 {**tag, "tool": name})
                    else:
                        log(f"{self.model}: {name} ok in {time.time() - t0:.1f}s")
                    messages.append({"role": "tool", "tool_call_id": tc.get("id") or f"call_{rnd}_{i}",
                                     "name": name,
                                     "content": _clip(out, _result_budget(context_for(self.model)))})
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
            push = None if final_round else nudge(content or reasoning)
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
        return False, "tool loop ended without an answer", usage

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

    def _chat(self, prompt: str, max_tokens: int = 2048, system: str | None = None) -> str:
        """One tool-less completion: audits, lessons, judging, consolidation."""
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        r = self._generate({"model": self.model, "messages": msgs, "stream": False, "max_tokens": max_tokens})
        msg = ((r.get("choices") or [{}])[0].get("message") or {})
        return (msg.get("content") or msg.get("reasoning_content") or "").strip()

    def _peer_chat(self, prompt: str, max_tokens: int = 3000) -> tuple[str, str]:
        """Ask a DIFFERENT loaded model when one is allowed -- a critic that did not write
        the code. Falls back to this agent's own model."""
        llms, _ = self._sync()
        # Free models first: a critique is routine work, not worth a paid call when a free
        # peer can do it.
        peers = sorted((m for m in llms if m != self.model and permitted(self.project, m)), key=is_external)
        model = peers[0] if peers else self.model
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "stream": False, "max_tokens": max_tokens}
        try:
            r = self._generate(payload)
        except RuntimeError as exc:
            # A hosted peer out of today's budget is no reason to skip the critique: this
            # agent's own model can still do it (if it is also hosted and refused, _generate
            # notes that and run() backs off).
            if model == self.model or not _spending_limited(str(exc)):
                raise
            model = self.model
            r = self._generate({**payload, "model": model})
        msg = ((r.get("choices") or [{}])[0].get("message") or {})
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
        """Board messages addressed to this agent (by `to` or an @mention) since its last
        iteration -- a teammate's question or hand-over lands in the next brief."""
        try:
            entries = request(BOARD, f"/mb/messages?project_id={q(self.pid)}&tail=150").get("entries", [])
        except RuntimeError:
            return []
        short = self.model.split("/")[-1].lower()
        since = self._inbox_since or (time.time() - 3600)
        out = []
        for e in entries:
            meta = e.get("meta") or {}
            if e["ts"] <= since or e["author"] == self.model:
                continue
            if meta.get("to") == self.model or f"@{short}" in str(e.get("content", "")).lower():
                out.append({"seq": e["seq"], "from": e["author"], "channel": e["channel"],
                            "minutes_ago": int((time.time() - e["ts"]) / 60), "text": str(e["content"])[:600]})
        return out[-8:]

    def record_collaboration(self, obj: dict, ctx: dict, world, last: dict, team_note: str, inbox: list[dict]) -> None:
        """Post, to #team, how this iteration used and helped the rest of the team -- built from
        what actually happened, plus the agent's own one-line note."""
        me = self.model.split("/")[-1]
        tag = {"objective_id": obj["id"], "candidate_id": last.get("candidate_id")}
        parts: list[str] = []
        collab: dict = {"agent": self.model, "mode": ctx.get("mode"), "candidate": last.get("seq"),
                        "built_on": None, "reused": [], "contributed": list(world.saved),
                        "messages_sent": world.sent, "answered": [], "inbox": len(inbox),
                        # which messages: the console lists the unanswered ones (app/team_threads.py)
                        "inbox_seqs": [m["seq"] for m in inbox],
                        "teammates_seen": [m["who"] for m in ctx.get("teammates") or []]}
        parent = ctx.get("parent")
        if parent:
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
        used = set(re.findall(r"from\s+lib\s+import\s+([\w\s,]+)", code))
        names = {n.strip() for grp in used for n in grp.split(",") if n.strip()} | set(re.findall(r"lib\.(\w+)", code))
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
        replied = {m["reply_to"] for m in world.sent if m.get("reply_to")}
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
        self.say("team", "result", text, {**tag, "collab": collab})

    def teammates(self) -> list[dict]:
        """What the other agents announced in #planning in the last 45 minutes."""
        try:
            entries = request(BOARD, f"/mb/messages?project_id={q(self.pid)}&channel=planning&tail=12").get("entries", [])
        except RuntimeError:
            return []
        now = time.time()
        return [{"who": e["author"], "minutes_ago": int((now - e["ts"]) / 60), "text": str(e["content"])[:400]}
                for e in entries if now - e["ts"] < 2700 and e["author"] != self.model]

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
                     + "\n\n".join(parts), {**tag, "ideas": [i for i, _ in posted]})
        for r in (notes.get("replies") or [])[:8]:
            if not isinstance(r, dict) or not str(r.get("text") or "").strip():
                continue
            meta = {"objective_id": oid, "team": True, "to": str(r.get("to") or "")}
            body = {"project_id": self.pid, "channel": "team", "author": self.model, "kind": "chat",
                    "content": (f"@{meta['to'].split('/')[-1]} " if meta["to"] else "") + str(r["text"])[:3000],
                    "meta": meta}
            if isinstance(r.get("reply_to"), int):
                body["reply_to"] = meta["reply_to"] = r["reply_to"]
            try:
                request(BOARD, "/mb/messages", body)
            except RuntimeError as exc:
                log(f"{self.model}: reply failed: {exc}")
        # Forecasts last: building one can take minutes, and the notes above should not wait.
        for f in (notes.get("forecasts") or [])[:MENTOR_MAX_FORECASTS]:
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

        def on_submit(args: dict) -> Any:
            if len(submits) >= MAX_SUBMITS:
                return {"error": "submission limit for this iteration reached"}
            payload = {"code": str(args.get("code") or ""), "answer": str(args.get("answer") or ""),
                       "rationale": str(args.get("rationale") or "")[:8000], "model": self.model,
                       "mode": ctx["mode"], "parent_id": (ctx.get("parent") or {}).get("id")}
            try:
                idea = int(args.get("idea")) if args.get("idea") not in (None, "") else None
            except (TypeError, ValueError):
                idea = None
            if idea is not None and idea in {i.get("id") for i in ctx.get("ideas") or []}:
                payload["idea_id"] = idea
            view = request(CONTROL_PLANE, f"/api/objectives/{q(oid)}/candidates", payload,
                           # + time for the harness to build forecasts the script asks for (ft.forecast)
                           timeout=int(obj.get("eval_timeout_s") or 300) * 4 + 120 + FORECAST_BUILD_ALLOWANCE_S)
            if obj["metric"]["kind"] == "judge" and view.get("status") == "ok":
                view = self.judge(obj, view, payload["answer"] or payload["code"])
            submits.append(view)
            self.report_eval(obj, view)
            return view

        world = ObjectiveWorld(self.project, self.model, llms, forecasters, obj, on_submit)
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
        inbox = self.inbox()
        ctx["inbox"] = inbox
        self._inbox_since = time.time()
        messages = [
            {"role": "system", "content": ITERATE_SYSTEM + playbook + "\n\n" + world.briefing()},
            {"role": "user", "content": iteration_prompt(ctx)},
        ]
        mode = ctx["mode"]
        parent = ctx.get("parent")
        self.say("general", "thought",
                 f"Iteration on \"{obj['title']}\": "
                 + ("BUILD -- adding a reusable module to the library" if mode == "build"
                    else f"improving #{parent['seq']} (rank {parent['rank']})" if parent else "exploring a new approach"),
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

        ok, text, usage = self.converse(
            messages, world.tools(), world.call, tag=tag, max_rounds=OBJECTIVE_TOOL_ROUNDS, nudge=nudge,
            # Finished once a submission ran cleanly or the budget of submissions is used.
            done=lambda: bool(submits) and (submits[-1].get("status") == "ok" or len(submits) >= MAX_SUBMITS)
            and (not build or bool(world.saved)),
            final_prompt="Tool budget nearly used up. Call submit_candidate NOW with your best complete script.",
        )
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
        kept = []
        team_note = ""
        for line in lesson.splitlines():
            if line.strip().upper().startswith("TEAM:"):
                team_note = line.strip()[5:].strip()[:400]
                continue
            m = re.match(r"\s*LIB\s+([a-z_][a-z0-9_]*)\s*:\s*(works|broken|note)\b\W*(.*)", line, re.I)
            if not m:
                kept.append(line)
                continue
            try:
                request(CONTROL_PLANE, f"/api/projects/{q(self.pid)}/library/{q(m.group(1))}/comments", {
                    "verdict": m.group(2).lower(), "text": m.group(3)[:2000] or m.group(2),
                    "author": self.model, "candidate_id": last.get("candidate_id")})
            except RuntimeError as exc:
                log(f"{self.model}: library comment on {m.group(1)} failed: {exc}")
        lesson = "\n".join(kept).strip()
        if lesson and len(lesson) > 8 and not lesson.startswith("(reasoning only"):
            try:
                request(CONTROL_PLANE, f"/api/objectives/{q(oid)}/lessons",
                        {"text": lesson[:1500], "model": self.model, "candidate_id": last.get("candidate_id")})
            except RuntimeError as exc:
                log(f"{self.model}: lesson not saved: {exc}")
        try:
            self.record_collaboration(obj, ctx, world, last, team_note, inbox)
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

    def run(self) -> None:
        while not self._stop.is_set() and not self.retired.is_set():
            if not self.agent_id and not self.register():
                self._stop.wait(POLL_IDLE_S * 2)
                continue
            try:
                if self._budget_block:
                    # The last generation was refused for today's spending limit. Starting the
                    # next iteration would fail the same way within seconds.
                    self.hold_for_budget()
                    continue
                self.beat("idle")
                if not self.agent_id:
                    continue
                # The mentor leaves one-off tasks to the searchers: its passes are the standing work.
                task = self.claim() if self.role != "mentor" else None
                if task is not None:
                    _act(self).begin("task", task={"id": task.get("id"), "title": task.get("title")})
                    self.run_task(task)  # one-off tasks take priority over the standing work
                    _act(self).end()
                    continue
                obj = self.next_objective()
                if obj is None:
                    self._stop.wait(POLL_IDLE_S)
                    continue
                if self.role == "mentor":
                    self.mentor(obj)
                    _act(self).end()
                    continue
                self.iterate(obj)
                _act(self).end()
                self.beat("idle", force=True)
                self._stop.wait(float(obj.get("cooldown_s") or 0))
            except Exception as exc:  # noqa: BLE001 -- a worker must never die on one task
                log(f"{self.model}: unexpected error: {exc!r}")
                self._stop.wait(POLL_IDLE_S)
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


def main() -> int:
    log(f"swarm runner: control plane {CONTROL_PLANE}, board {BOARD}")
    if AGENT_USER and not STATIC_TOKEN:
        AUTH.refresh()
    stop = threading.Event()
    state = State()
    workers: dict[tuple[str, str, int], Worker] = {}
    warned_empty = False
    try:
        while True:
            try:
                projects = request(CONTROL_PLANE, "/api/projects").get("projects", [])
                loaded = request(CONTROL_PLANE, "/api/engines").get("loaded", []) or []
                llms = [m.get("model") or m.get("served_name") for m in loaded if m.get("ready", True)]
                llms = [m for m in llms if m]
                # Models on paired computers report their window with the list; local ones
                # are read from the console's engine stats below.
                for m in loaded:
                    if m.get("remote") and m.get("model"):
                        # Below 4K (or missing) is a misreport, not a window an agent could use
                        # (an engine once gave DeepSeek-V4's 128K as "1024"): squeezing every
                        # prompt into it and capping answers at 256 tokens is worse than
                        # learning the real window from the first overflow error.
                        if m["model"] in _learned:
                            pass
                        elif int(m.get("context") or 0) >= 4096:
                            _context[m["model"]] = int(m["context"])
                        else:
                            _context[m["model"]] = UNVERIFIED_CONTEXT
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
                    if w is None or not w.is_alive():
                        w = Worker(project, key[1], stop, state.get, slot=max(0, key[2]),
                                   role="mentor" if key[2] == mentor_slot else "search")
                        workers[key] = w
                        w.start()
                    else:
                        w.project = project  # pick up renamed projects / new SQL / model lists
                for key in [k for k in workers if k not in wanted]:
                    log(f"{key[1]}: no longer allowed/loaded for project {key[0]}; retiring agent")
                    workers.pop(key).retired.set()
                if not wanted and not warned_empty:
                    log("no models loaded -- queued tasks will wait until one is loaded")
                    warned_empty = True
                elif wanted:
                    warned_empty = False
            if stop.wait(ENGINE_RESYNC_S):
                break
    except KeyboardInterrupt:
        log("shutting down")
    finally:
        stop.set()
        for w in workers.values():
            w.join(timeout=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
