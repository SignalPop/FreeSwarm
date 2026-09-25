"""External (hosted, pay-per-token) LLM providers: Groq and OpenRouter.

Both speak the OpenAI chat-completions protocol, so a hosted model is used exactly like a
local engine once it is *enabled*. Enabled models are named ``<provider model id>@<provider>``
-- ``openai/gpt-oss-120b@groq``, ``anthropic/claude-opus-5.5@openrouter`` -- the same
``name@where`` shape as a model on a paired computer, and appear in the same lists (chat,
project model picker, swarm). Nothing is enabled by default.

**Money.** Every call is priced from the provider's per-token rates and written to a ledger
(``backend/external_usage.sqlite3``). A daily limit (USD, default $5) is checked before each
call; once it is reached, external calls are refused until midnight local time. A project
whose model list is "all loaded models" never picks up an external model on its own -- it has
to be ticked for that project.

Part of the limit (``ideas_reserve_usd``, default $1) is **held for ideas when stuck**: routine
search, chat and chores stop once ``limit - reserve`` is spent, and only escalation calls
(purpose ``ideas:<objective id>``) may spend the rest. Without it, three hosted search agents
spent a whole $5 day in 23 minutes after midnight, and the strong model on the escalation
ladder -- the one the budget most needed to reach -- was refused every time it became due.

**Catalogs.** OpenRouter publishes its models, prices, context windows and Artificial
Analysis scores without a key (``GET /api/v1/models``). Groq's model list needs a key and
carries no prices, so Groq's prices and speeds come from the table below (console.groq.com/
docs/models, collected 2026-09-23).

Keys are stored with the other secrets in ``auth/secrets.json`` (owner-only), passed only to
the provider they belong to, and never returned to the browser.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import ratings
from .auth import AUTH_DIR, _restrict_permissions

logger = logging.getLogger("freetoken.external")
router = APIRouter(tags=["external"])

SECRETS_FILE = AUTH_DIR / "secrets.json"
BACKEND_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BACKEND_DIR / "external.json"
LEDGER_PATH = BACKEND_DIR / "external_usage.sqlite3"

PROVIDERS: dict[str, dict] = {
    "groq": {"label": "Groq", "base": "https://api.groq.com/openai/v1",
             "keys_url": "https://console.groq.com/keys", "site": "https://console.groq.com/docs/models"},
    "openrouter": {"label": "OpenRouter", "base": "https://openrouter.ai/api/v1",
                   "keys_url": "https://openrouter.ai/settings/keys", "site": "https://openrouter.ai/models"},
}

DEFAULT_CONFIG = {
    "enabled": [],            # model names, "<id>@<provider>"
    "daily_limit_usd": 5.0,
    # Held back from search for the escalation ladder (clamped to 0..daily_limit_usd). Search
    # stops at limit - reserve; only ``ideas:*`` calls may spend the reserve.
    "ideas_reserve_usd": 1.0,
    # A hosted model serves many requests at once, so each one that searches for a project runs
    # this many agents in parallel.
    "parallel_agents": 3,
    # Agents per free model (this computer's engines and paired computers'). An engine batches
    # concurrent requests, so 2 can raise its throughput -- at the cost of KV-cache memory.
    "parallel_local_agents": 2,
    # When the swarm's search stalls, ask the strongest models for new directions (see
    # escalation.py). The thresholds are how long "stuck" has to last before each step.
    "escalation": {"enabled": True, "stuck_candidates": 40, "stuck_minutes": 60,
                   "step_candidates": 25, "step_minutes": 45},
}

# Groq has no pricing API. Speed is Groq's published output tokens/s. None = "contact sales"
# (Enterprise-only), which the console shows but cannot price, so such models cannot be enabled.
GROQ_COLLECTED = "2026-09-23"
GROQ_MODELS: list[dict] = [
    {"id": "openai/gpt-oss-120b", "name": "GPT-OSS 120B", "tier": "production", "speed": 500,
     "price_in": 0.15, "price_out": 0.60, "context": 131_072, "max_output": 65_536},
    {"id": "openai/gpt-oss-20b", "name": "GPT-OSS 20B", "tier": "production", "speed": 1000,
     "price_in": 0.075, "price_out": 0.30, "context": 131_072, "max_output": 65_536},
    {"id": "qwen/qwen3.8-27b", "name": "Qwen3.8 27B", "tier": "preview", "speed": 450,
     "price_in": 0.80, "price_out": 4.00, "context": 131_072, "max_output": 16_384},
    {"id": "openai/gpt-oss-safeguard-20b", "name": "GPT-OSS Safeguard 20B", "tier": "preview", "speed": 1000,
     "price_in": 0.075, "price_out": 0.30, "context": 131_072, "max_output": 65_536},
    {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "tier": "production (enterprise)", "speed": 280,
     "price_in": None, "price_out": None, "context": 131_072, "max_output": 32_768},
    {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B", "tier": "production (enterprise)", "speed": 560,
     "price_in": None, "price_out": None, "context": 131_072, "max_output": 131_072},
    {"id": "minimaxai/minimax-m2.7", "name": "MiniMax M2.7", "tier": "preview (enterprise)", "speed": 260,
     "price_in": None, "price_out": None, "context": 196_608, "max_output": 131_072},
]

_lock = threading.RLock()
_catalog: dict[str, dict] = {}         # provider -> {"at": ts, "rows": [...], "error": str|None}
CATALOG_TTL_S = 600


# =======================================================================================
# Keys and config
# =======================================================================================
def _read_secrets() -> dict:
    try:
        return json.loads(SECRETS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def api_key(provider: str) -> str | None:
    return _read_secrets().get(f"{provider}:api_key") or None


def _write_key(provider: str, value: str | None) -> None:
    with _lock:
        data = _read_secrets()
        if value:
            data[f"{provider}:api_key"] = value
        else:
            data.pop(f"{provider}:api_key", None)
        AUTH_DIR.mkdir(parents=True, exist_ok=True)
        SECRETS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        _restrict_permissions(SECRETS_FILE)
    with _lock:
        _catalog.pop(provider, None)  # what a key can see may differ


def config() -> dict:
    try:
        saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    cfg = {**DEFAULT_CONFIG, **saved}
    cfg["escalation"] = {**DEFAULT_CONFIG["escalation"], **(saved.get("escalation") or {})}
    return cfg


def _save_config(cfg: dict) -> None:
    with _lock:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# =======================================================================================
# Names
# =======================================================================================
def split(name: str | None) -> tuple[str, str] | None:
    """(provider, provider model id) for ``<id>@groq`` / ``<id>@openrouter``, else None."""
    if not name or "@" not in name:
        return None
    mid, _, provider = name.rpartition("@")
    return (provider, mid) if provider in PROVIDERS and mid else None


def is_external(name: str | None) -> bool:
    return split(name) is not None


# =======================================================================================
# Catalogs
# =======================================================================================
def _per_million(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f * 1_000_000, 4) if f >= 0 else None


async def _fetch_openrouter(client: httpx.AsyncClient) -> list[dict]:
    r = await client.get(PROVIDERS["openrouter"]["base"] + "/models", timeout=30.0)
    r.raise_for_status()
    rows = []
    for m in r.json().get("data") or []:
        arch = m.get("architecture") or {}
        if "text" not in (arch.get("output_modalities") or ["text"]):
            continue
        price = m.get("pricing") or {}
        aa = ((m.get("benchmarks") or {}).get("artificial_analysis") or {})
        top = m.get("top_provider") or {}
        rows.append({
            "id": m["id"], "name": m.get("name") or m["id"],
            "context": m.get("context_length") or top.get("context_length"),
            "max_output": top.get("max_completion_tokens"),
            "price_in": _per_million(price.get("prompt")), "price_out": _per_million(price.get("completion")),
            "speed": None,  # OpenRouter publishes no per-model throughput without a key
            "aa_live": aa.get("intelligence_index"), "aa_coding_live": aa.get("coding_index"),
            "created": m.get("created"), "tier": "free" if m["id"].endswith(":free") else "",
            "reasoning": bool(m.get("reasoning")),
        })
    return rows


async def _fetch_groq(client: httpx.AsyncClient) -> tuple[list[dict], str | None]:
    """Groq's static price table, marked with what this key can actually reach."""
    rows = [dict(r) for r in GROQ_MODELS]
    key = api_key("groq")
    if not key:
        return rows, "no Groq API key -- showing Groq's published list"
    r = await client.get(PROVIDERS["groq"]["base"] + "/models",
                         headers={"Authorization": f"Bearer {key}"}, timeout=20.0)
    if r.status_code == 401:
        return rows, "Groq rejected the API key"
    r.raise_for_status()
    live = {m["id"]: m for m in r.json().get("data") or []}
    for row in rows:
        m = live.get(row["id"])
        row["available"] = bool(m and m.get("active", True))
        if m and m.get("context_window"):
            row["context"] = m["context_window"]
    # Chat models the key can reach that the price table does not know yet: listed, unpriced.
    known = {r["id"] for r in rows}
    for mid, m in live.items():
        if mid not in known and not any(k in mid for k in ("whisper", "tts", "guard", "embed", "orpheus")):
            rows.append({"id": mid, "name": mid, "tier": "unpriced", "speed": None, "price_in": None,
                         "price_out": None, "context": m.get("context_window"), "max_output": None,
                         "available": bool(m.get("active", True))})
    return rows, None


def _decorate(provider: str, row: dict) -> dict:
    name = f"{row['id']}@{provider}"
    sc = ratings.scores(row["id"], row.get("name"))
    aa = sc["aa"] if sc["aa"] is not None else row.get("aa_live")
    price_in, price_out = row.get("price_in"), row.get("price_out")
    return {**row, "provider": provider, "model": name, "swe": sc["swe"], "aa": aa,
            "rating_label": sc["rating_label"],
            # Blended $/Mtok at 3 input tokens per output token -- the usual agent mix.
            "price_blended": round((3 * price_in + price_out) / 4, 4)
            if price_in is not None and price_out is not None else None,
            "priced": price_in is not None and price_out is not None,
            "enabled": name in config()["enabled"]}


async def catalog(provider: str, client: httpx.AsyncClient, refresh: bool = False) -> dict:
    if provider not in PROVIDERS:
        raise HTTPException(status_code=404, detail=f"unknown provider {provider!r}")
    with _lock:
        hit = _catalog.get(provider)
    if hit and not refresh and time.time() - hit["at"] < CATALOG_TTL_S:
        rows, note = hit["rows"], hit["error"]
    else:
        note = None
        try:
            if provider == "openrouter":
                rows = await _fetch_openrouter(client)
            else:
                rows, note = await _fetch_groq(client)
        except (httpx.HTTPError, ValueError) as exc:
            rows = hit["rows"] if hit else ([dict(r) for r in GROQ_MODELS] if provider == "groq" else [])
            note = f"could not reach {PROVIDERS[provider]['label']}: {exc}"
        with _lock:
            _catalog[provider] = {"at": time.time(), "rows": rows, "error": note}
    return {"provider": provider, "rows": [_decorate(provider, r) for r in rows], "note": note,
            "key_set": bool(api_key(provider)),
            "prices_as_of": GROQ_COLLECTED if provider == "groq" else "live"}


def _row(name: str) -> dict | None:
    """The cached catalog row for an enabled model (None until the catalog has been read)."""
    sp = split(name)
    if sp is None:
        return None
    with _lock:
        hit = _catalog.get(sp[0])
    for r in (hit or {}).get("rows") or ([dict(x) for x in GROQ_MODELS] if sp[0] == "groq" else []):
        if r["id"] == sp[1]:
            return _decorate(sp[0], r)
    return None


def loaded() -> list[dict]:
    """Enabled external models in the shape of EngineManager.loaded_models(), plus `external`.

    A provider without a key contributes nothing -- its models cannot answer.
    """
    out = []
    for name in config()["enabled"]:
        sp = split(name)
        if sp is None or not api_key(sp[0]):
            continue
        row = _row(name) or {}
        out.append({
            "model": name, "served_name": name, "ready": True, "context": row.get("context"),
            "external": {"provider": sp[0], "provider_label": PROVIDERS[sp[0]]["label"], "id": sp[1],
                         "price_in": row.get("price_in"), "price_out": row.get("price_out"),
                         "price_blended": row.get("price_blended"), "speed": row.get("speed")},
            "swe": row.get("swe"), "aa": row.get("aa"),
        })
    return out


async def warm(client: httpx.AsyncClient) -> None:
    """Read both catalogs once at startup so enabled models have prices and windows."""
    for p in PROVIDERS:
        try:
            await catalog(p, client)
        except Exception:  # noqa: BLE001 -- the console starts regardless
            logger.exception("could not read the %s catalog", p)


# =======================================================================================
# Spend ledger and limit
# =======================================================================================
_ledger: sqlite3.Connection | None = None


def _db() -> sqlite3.Connection:
    global _ledger
    if _ledger is None:
        _ledger = sqlite3.connect(LEDGER_PATH, check_same_thread=False, timeout=30.0)
        _ledger.execute("PRAGMA journal_mode=WAL")
        _ledger.execute("""CREATE TABLE IF NOT EXISTS calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, day TEXT NOT NULL,
            provider TEXT NOT NULL, model TEXT NOT NULL, purpose TEXT NOT NULL DEFAULT '',
            prompt_tokens INTEGER, completion_tokens INTEGER, usd REAL NOT NULL DEFAULT 0)""")
        _ledger.commit()
    return _ledger


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def spent_today() -> float:
    with _lock:
        v = _db().execute("SELECT COALESCE(SUM(usd),0) FROM calls WHERE day=?", (_today(),)).fetchone()[0]
    return float(v or 0)


def _record(name: str, usage: dict | None, purpose: str) -> float:
    sp = split(name)
    if sp is None:
        return 0.0
    usage = usage or {}
    pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
    usd = usage.get("cost")  # OpenRouter reports the exact charge when asked (usage.include)
    if usd is None:
        row = _row(name) or {}
        if row.get("priced") and (pt or ct):
            usd = ((pt or 0) * row["price_in"] + (ct or 0) * row["price_out"]) / 1_000_000
    usd = float(usd or 0)
    with _lock:
        _db().execute("INSERT INTO calls (ts, day, provider, model, purpose, prompt_tokens, completion_tokens, usd) "
                      "VALUES (?,?,?,?,?,?,?,?)", (time.time(), _today(), sp[0], sp[1], purpose, pt, ct, usd))
        _db().commit()
    return usd


IDEAS_PURPOSE = "ideas:"  # escalation.py's purpose prefix -- the only calls that may spend the reserve


def is_ideas(purpose: str | None) -> bool:
    return (purpose or "").startswith(IDEAS_PURPOSE)


def limits() -> tuple[float, float]:
    """(daily limit, ideas reserve) in USD. The reserve is clamped into [0, limit] so a
    reserve larger than the limit simply means "ideas only", never a negative search budget."""
    cfg = config()
    limit = max(0.0, float(cfg["daily_limit_usd"]))
    reserve = min(limit, max(0.0, float(cfg.get("ideas_reserve_usd") or 0)))
    return limit, reserve


def search_budget_left() -> float:
    """USD that routine (non-ideas) calls may still spend today; 0 once search must stop."""
    limit, reserve = limits()
    return round(max(0.0, limit - reserve - spent_today()), 4)


def check_budget(purpose: str = "chat", name: str | None = None) -> None:
    """Refuse a call that today's spend no longer allows (429).

    ``ideas:*`` calls may spend up to the full daily limit; everything else stops at
    ``limit - reserve``. Both messages contain "spending limit" -- the swarm runner matches
    that phrase to back off quietly instead of retrying every few seconds. A refusal is also
    noted against ``name`` so the console can show why a model is silent.
    """
    limit, reserve = limits()
    spent = spent_today()
    if spent >= limit:
        detail = (f"today's external-model spending limit is reached (${spent:.2f} of ${limit:.2f}). "
                  "Raise it in Settings -> External models, or wait until midnight.")
        short = "daily limit reached -- resumes at midnight"
    elif not is_ideas(purpose) and spent >= limit - reserve:
        detail = (f"today's external-model spending limit for search is reached (${spent:.2f} of "
                  f"${limit - reserve:.2f}); the remaining ${reserve:.2f} of the ${limit:.2f} daily limit is "
                  "held for ideas when stuck. Search resumes at midnight, or raise the limit / lower the "
                  "reserve in Settings -> External models.")
        short = "daily search budget spent -- resumes at midnight"
    else:
        return
    if name:
        _note_refusal(name, short, detail, "budget")
    raise HTTPException(status_code=429, detail=detail)


# ---- refusals ---------------------------------------------------------------------------
# Why a model is silent: calls refused by the budget, and non-200 answers from the provider
# (rate limits, malformed tool calls, outages). Kept in memory -- it answers "what is going
# wrong right now"; the ledger already holds what was paid for. Counts reset at midnight.
_refusals: dict[str, dict] = {}   # model name -> {"day", "count", "short", "reason", "kind", "ts"}


def _note_refusal(name: str, short: str, reason: str, kind: str) -> None:
    with _lock:
        r = _refusals.get(name)
        if r is None or r["day"] != _today():
            r = {"day": _today(), "count": 0}
        r.update(count=r["count"] + 1, short=short[:300], reason=reason[:1000], kind=kind, ts=time.time())
        _refusals[name] = r


def _note_provider_error(name: str, status: int, body: Any) -> None:
    """A non-200 from the provider, condensed to its message ("Groq 429: Rate limit reached...")."""
    sp = split(name)
    label = PROVIDERS[sp[0]]["label"] if sp else "provider"
    msg = body
    if isinstance(body, (bytes, str)):
        try:
            msg = json.loads(body)
        except ValueError:
            msg = body.decode("utf-8", "replace") if isinstance(body, bytes) else body
    if isinstance(msg, dict):
        err = msg.get("error")
        msg = (err.get("message") or err.get("code") or err) if isinstance(err, dict) else (err or msg)
    text = f"{label} {status}: {str(msg).strip()}"
    _note_refusal(name, text[:300], text, "provider")


def refusals(name: str) -> dict:
    """Today's refusals for a model: count, last short reason, last full reason, last ts."""
    with _lock:
        r = dict(_refusals.get(name) or {})
    if r.get("day") != _today():
        return {"count": 0, "short": None, "reason": None, "kind": None, "ts": None}
    return {k: r.get(k) for k in ("count", "short", "reason", "kind", "ts")}


def usage_by_model() -> dict[str, dict]:
    """Ledger totals per model name (``<id>@<provider>``): today and all time.

    The ledger is global -- one hosted model serving two projects is one bill -- so these are
    not split by project; candidates and ideas (objectives.sqlite3) are.
    """
    day = _today()
    with _lock:
        rows = _db().execute(
            "SELECT provider, model, COUNT(*), COALESCE(SUM(usd),0), MAX(ts), "
            "SUM(day=?), COALESCE(SUM(CASE WHEN day=? THEN usd END),0), "
            "COALESCE(SUM(CASE WHEN day=? THEN prompt_tokens END),0), "
            "COALESCE(SUM(CASE WHEN day=? THEN completion_tokens END),0), "
            "SUM(day=? AND purpose LIKE 'ideas:%') "
            "FROM calls GROUP BY provider, model", (day, day, day, day, day)).fetchall()
    return {f"{m}@{p}": {"calls_total": n, "usd_total": round(u, 4), "last_call_ts": last,
                         "calls_today": int(nt or 0), "usd_today": round(ut, 4),
                         "prompt_tokens_today": int(pt or 0), "completion_tokens_today": int(ct or 0),
                         "ideas_calls_today": int(it or 0)}
            for p, m, n, u, last, nt, ut, pt, ct, it in rows}


def spend_summary() -> dict:
    with _lock:
        rows = _db().execute(
            "SELECT provider, model, COUNT(*), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0), "
            "COALESCE(SUM(usd),0) FROM calls WHERE day=? GROUP BY provider, model ORDER BY 6 DESC", (_today(),)).fetchall()
        days = _db().execute("SELECT day, COALESCE(SUM(usd),0) FROM calls GROUP BY day ORDER BY day DESC LIMIT 14").fetchall()
    limit, reserve = limits()
    return {"today": round(spent_today(), 4), "limit": limit, "ideas_reserve": reserve,
            "search_left": search_budget_left(),
            "by_model": [{"provider": p, "model": m, "calls": n, "prompt_tokens": a, "completion_tokens": b,
                          "usd": round(u, 4)} for p, m, n, a, b, u in rows],
            "days": [{"day": d, "usd": round(u, 4)} for d, u in days]}


# =======================================================================================
# Calls
# =======================================================================================
def _headers(provider: str) -> dict:
    key = api_key(provider)
    if not key:
        raise HTTPException(status_code=409, detail=f"no {PROVIDERS[provider]['label']} API key -- add it in Settings")
    h = {"Authorization": f"Bearer {key}"}
    if provider == "openrouter":
        h |= {"X-OpenRouter-Title": "FreeSwarm", "X-Title": "FreeSwarm"}
    return h


def _prepare(name: str, payload: dict, purpose: str = "chat") -> tuple[str, str, dict]:
    sp = split(name)
    if sp is None:
        raise HTTPException(status_code=404, detail=f"{name!r} is not an external model")
    if name not in config()["enabled"]:
        raise HTTPException(status_code=403, detail=f"{name} is not enabled -- enable it on the External page")
    row = _row(name)
    if row is not None and not row.get("priced"):
        raise HTTPException(status_code=403, detail=f"{name} has no published price; it cannot be metered")
    check_budget(purpose, name)
    provider, mid = sp
    body = {**payload, "model": mid}
    if provider == "openrouter":
        body["usage"] = {"include": True}  # the response then carries the exact cost
    if body.get("stream"):
        body["stream_options"] = {"include_usage": True}
    return provider, mid, body


async def complete(client: httpx.AsyncClient, name: str, payload: dict, purpose: str = "chat") -> Any:
    """Forward an OpenAI-style chat completion to the model's provider and meter it."""
    provider, _, body = _prepare(name, payload, purpose)
    url = PROVIDERS[provider]["base"] + "/chat/completions"
    headers = _headers(provider)
    if body.get("stream"):
        async def relay():
            usage: dict | None = None
            buf = b""
            try:
                async with client.stream("POST", url, json=body, headers=headers,
                                         timeout=httpx.Timeout(None, connect=10.0)) as r:
                    if r.status_code != 200:
                        text = (await r.aread()).decode("utf-8", "replace")
                        _note_provider_error(name, r.status_code, text)
                        yield f'data: {{"error": {text!r}}}\n\n'.encode()
                        return
                    async for chunk in r.aiter_raw():
                        yield chunk
                        buf = (buf + chunk)[-65536:]
                        # The usage block rides on the last data chunk (Groq also nests it in x_groq).
                        for line in buf.split(b"\n"):
                            if line.startswith(b"data: {") and b'"usage"' in line:
                                try:
                                    d = json.loads(line[6:])
                                except ValueError:
                                    continue
                                u = d.get("usage") or (d.get("x_groq") or {}).get("usage")
                                if u:
                                    usage = u
            except httpx.HTTPError as exc:
                _note_refusal(name, f"{PROVIDERS[provider]['label']} unreachable", str(exc), "provider")
                yield f'data: {{"error": "{PROVIDERS[provider]["label"]} failed: {exc}"}}\n\n'.encode()
            finally:
                _record(name, usage, purpose)

        return StreamingResponse(relay(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    try:
        r = await client.post(url, json=body, headers=headers, timeout=600.0)
    except httpx.HTTPError as exc:
        _note_refusal(name, f"{PROVIDERS[provider]['label']} unreachable", str(exc), "provider")
        raise HTTPException(status_code=502, detail=f"{PROVIDERS[provider]['label']} unreachable: {exc}") from None
    try:
        data = r.json()
    except ValueError:
        _note_provider_error(name, r.status_code, r.text[:500])
        raise HTTPException(status_code=502, detail=f"{PROVIDERS[provider]['label']} returned {r.status_code}") from None
    if r.status_code == 200:
        _record(name, data.get("usage"), purpose)
    else:
        _note_provider_error(name, r.status_code, data)
    return JSONResponse(data, status_code=r.status_code)


async def complete_text(client: httpx.AsyncClient, name: str, messages: list[dict], max_tokens: int,
                        purpose: str) -> str:
    resp = await complete(client, name, {"messages": messages, "max_tokens": max_tokens, "stream": False}, purpose)
    data = json.loads(resp.body)
    if resp.status_code != 200:
        err = (data.get("error") or {}) if isinstance(data, dict) else {}
        raise HTTPException(status_code=502, detail=f"{name}: {err.get('message') or data}")
    msg = ((data.get("choices") or [{}])[0].get("message") or {})
    return (msg.get("content") or msg.get("reasoning") or "").strip()


# =======================================================================================
# Console API
# =======================================================================================
_http: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=8.0))
    return _http


@router.get("/external")
async def overview() -> dict:
    cfg = config()
    return {"providers": [{"id": p, **meta, "key_set": bool(api_key(p))} for p, meta in PROVIDERS.items()],
            "enabled": cfg["enabled"], "daily_limit_usd": cfg["daily_limit_usd"],
            "ideas_reserve_usd": cfg["ideas_reserve_usd"],
            "parallel_agents": cfg["parallel_agents"], "parallel_local_agents": cfg["parallel_local_agents"],
            "escalation": cfg["escalation"], "spend": spend_summary()}


@router.get("/external/{provider}/models")
async def models(provider: str, refresh: bool = False) -> dict:
    return await catalog(provider, _client(), refresh)


class EnableReq(BaseModel):
    model: str = Field(..., max_length=300)
    enabled: bool


@router.put("/external/enabled")
async def set_enabled(req: EnableReq) -> dict:
    sp = split(req.model)
    if sp is None:
        raise HTTPException(status_code=400, detail="expected <model id>@groq or <model id>@openrouter")
    cfg = config()
    names = [n for n in cfg["enabled"] if n != req.model]
    if req.enabled:
        await catalog(sp[0], _client())
        row = _row(req.model)
        if row is None:
            raise HTTPException(status_code=404, detail=f"{PROVIDERS[sp[0]]['label']} does not list {sp[1]}")
        if not row["priced"]:
            raise HTTPException(status_code=400, detail=f"{sp[1]} has no published price, so its use cannot be metered")
        names.append(req.model)
    cfg["enabled"] = names
    _save_config(cfg)
    return {"enabled": names}


class ConfigReq(BaseModel):
    # Write-only keys: None leaves a key alone, "" removes it.
    groq_api_key: str | None = Field(None, max_length=400)
    openrouter_api_key: str | None = Field(None, max_length=400)
    daily_limit_usd: float | None = Field(None, ge=0, le=10_000)
    ideas_reserve_usd: float | None = Field(None, ge=0, le=10_000)
    parallel_agents: int | None = Field(None, ge=1, le=8)
    parallel_local_agents: int | None = Field(None, ge=1, le=4)
    escalation: dict | None = None


@router.put("/external/config")
async def write_config(req: ConfigReq) -> dict:
    cfg = config()
    if req.daily_limit_usd is not None:
        cfg["daily_limit_usd"] = req.daily_limit_usd
    if req.ideas_reserve_usd is not None:
        cfg["ideas_reserve_usd"] = req.ideas_reserve_usd
    if float(cfg.get("ideas_reserve_usd") or 0) > float(cfg["daily_limit_usd"]):
        # A reserve above the limit means nothing more than "all of it" (limits() clamps it
        # anyway); store what is actually in force so the settings page shows the truth.
        cfg["ideas_reserve_usd"] = cfg["daily_limit_usd"]
    if req.parallel_agents is not None:
        cfg["parallel_agents"] = req.parallel_agents
    if req.parallel_local_agents is not None:
        cfg["parallel_local_agents"] = req.parallel_local_agents
    if req.escalation is not None:
        esc = cfg["escalation"]
        for k in DEFAULT_CONFIG["escalation"]:
            if k in req.escalation:
                v = req.escalation[k]
                esc[k] = bool(v) if k == "enabled" else max(1, int(v))
    _save_config(cfg)
    for provider, value in (("groq", req.groq_api_key), ("openrouter", req.openrouter_api_key)):
        if value is not None:
            _write_key(provider, value.strip() or None)
    return await overview()


@router.post("/external/{provider}/test")
async def test_key(provider: str) -> dict:
    """Prove the key works -- a free call (list models / key info), not a paid completion."""
    if provider not in PROVIDERS:
        raise HTTPException(status_code=404, detail=f"unknown provider {provider!r}")
    headers = _headers(provider)
    url = PROVIDERS[provider]["base"] + ("/models" if provider == "groq" else "/key")
    try:
        r = await _client().get(url, headers=headers, timeout=20.0)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"could not reach {PROVIDERS[provider]['label']}: {exc}") from None
    if r.status_code in (401, 403):
        raise HTTPException(status_code=401, detail=f"{PROVIDERS[provider]['label']} rejected the API key")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"{PROVIDERS[provider]['label']} returned {r.status_code}")
    d = r.json()
    if provider == "groq":
        return {"ok": True, "detail": f"key works -- {len(d.get('data') or [])} models available"}
    info = d.get("data") or {}
    limit = info.get("limit")
    left = info.get("limit_remaining")
    return {"ok": True, "detail": "key works -- " + (f"${left:.2f} of ${limit:.2f} credit left" if limit is not None and left is not None
                                                    else "no credit limit set on the key")}


async def shutdown() -> None:
    global _http
    if _http is not None:
        await _http.aclose()
        _http = None
    await asyncio.sleep(0)
