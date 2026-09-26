"""Tokens processed per model -- this computer's engines, paired computers' models and hosted
(external) models -- kept across restarts in ``backend/token_usage.sqlite3``.

Where each source's figures come from:

* **local** -- the engine's own lifetime counters (``/v1/stats`` ``requests.prompt_tokens_total``
  / ``completion_tokens_total`` / ``completed``), diffed between polls. The engine feeds them
  from the same numbers it reports as ``usage``, and they see EVERY request it serves: the
  console's chat, the swarm through ``/v1``, and paired computers through the federation
  gateway, which talks to the engine directly and never passes through this server's proxy.
  Counting ``usage`` at the proxy instead would miss that last path entirely.
* **network** -- the ``usage`` block of each reply relayed from a paired computer (or a
  configured remote backend). Streams are asked for it with ``stream_options.include_usage``;
  a reply that still carries none is counted as a request with ``unmetered`` set.
* **external** -- Groq / OpenRouter calls are already written, one row per call, to the spend
  ledger (``external_usage.sqlite3``); that is read as-is rather than written twice. Calls
  this server makes to Anthropic directly (the external reviewer) have no ledger, so they are
  recorded here under ``<model>@anthropic``.

Writes are cheap: every observation lands in an in-memory accumulator and is upserted into
hourly buckets every ``FLUSH_S`` seconds (and on shutdown), so a busy swarm costs one small
transaction per flush, not one per request. Hourly buckets are what make the ``since`` window
possible; a window is rounded out to whole hours.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

from fastapi import APIRouter, Query

logger = logging.getLogger("freetoken.tokens")
router = APIRouter(tags=["usage"])

DB_PATH = Path(__file__).resolve().parent.parent / "token_usage.sqlite3"
SOURCES = ("local", "network", "external")
BUCKET_S = 3600
FLUSH_S = 10.0
# How often engines are sampled when nothing else is asking. The console's 1 Hz poll samples
# too, and every stop/move samples first, so this only bounds what an unwatched crash can lose.
SAMPLE_S = 5.0

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
# (bucket, source, model) -> [prompt, completion, requests, unmetered, first_ts, last_ts], unwritten
_pending: dict[tuple[int, str, str], list] = {}
# engine identity (instance id + start time) -> (the last counters seen from it, when)
_marks: dict[str, tuple[tuple[int, int, int], float]] = {}
# How long a stopped engine's baseline is kept. A reading already in flight when it stopped
# may still arrive; without its baseline it would be counted again from zero.
FORGET_AFTER_S = 600.0


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.execute("""CREATE TABLE IF NOT EXISTS usage (
            bucket INTEGER NOT NULL, source TEXT NOT NULL, model TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL DEFAULT 0, completion_tokens INTEGER NOT NULL DEFAULT 0,
            requests INTEGER NOT NULL DEFAULT 0, unmetered INTEGER NOT NULL DEFAULT 0,
            first_ts REAL NOT NULL, last_ts REAL NOT NULL,
            PRIMARY KEY (bucket, source, model))""")
        _conn.commit()
    return _conn


def close() -> None:
    """Flush and drop the connection (shutdown, and tests that point DB_PATH elsewhere)."""
    global _conn
    flush()
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


# =======================================================================================
# Recording
# =======================================================================================
def record(source: str, model: str, prompt: int, completion: int, requests: int = 1,
           unmetered: int = 0, ts: float | None = None) -> None:
    if not model or source not in SOURCES:
        return
    ts = time.time() if ts is None else ts
    key = (int(ts // BUCKET_S) * BUCKET_S, source, model)
    with _lock:
        row = _pending.get(key)
        if row is None:
            _pending[key] = [int(prompt), int(completion), int(requests), int(unmetered), ts, ts]
        else:
            row[0] += int(prompt)
            row[1] += int(completion)
            row[2] += int(requests)
            row[3] += int(unmetered)
            row[4], row[5] = min(row[4], ts), max(row[5], ts)


def _tokens(usage: Any) -> tuple[int, int] | None:
    """(prompt, completion) from an OpenAI (`prompt_tokens`) or Anthropic (`input_tokens`)
    usage block, or None when there is nothing to count."""
    if not isinstance(usage, dict):
        return None
    pt = usage.get("prompt_tokens", usage.get("input_tokens"))
    ct = usage.get("completion_tokens", usage.get("output_tokens"))
    if pt is None and ct is None:
        return None
    try:
        return max(0, int(pt or 0)), max(0, int(ct or 0))
    except (TypeError, ValueError):
        return None


def record_usage(source: str, model: str, usage: Any) -> None:
    """One request. A reply without a usable usage block still counts, as `unmetered`."""
    t = _tokens(usage)
    if t is None:
        record(source, model, 0, 0, unmetered=1)
    else:
        record(source, model, *t)


def usage_from_body(body: Any) -> dict | None:
    """The usage block of a non-streamed chat completion (a dict, or its JSON bytes)."""
    if isinstance(body, (bytes, str)):
        try:
            body = json.loads(body)
        except ValueError:
            return None
    return body.get("usage") if isinstance(body, dict) else None


async def metered(chunks: AsyncIterator[bytes], on_done: Callable[[dict | None], None]) -> AsyncIterator[bytes]:
    """Pass an SSE stream through untouched, then report the last usage block it carried.

    The usage chunk is the final data event (``choices: []``), so only the tail of the stream
    is kept and it is parsed once, at the end. `on_done` runs however the stream ends --
    finished, failed or abandoned by the client -- so an interrupted generation still counts
    as a request.
    """
    tail = b""
    try:
        async for chunk in chunks:
            yield chunk
            tail = (tail + chunk)[-65536:]
    finally:
        try:
            on_done(usage_from_sse(tail))
        except Exception:  # noqa: BLE001 -- accounting must never break a reply
            logger.debug("token accounting failed", exc_info=True)


def usage_from_sse(tail: bytes) -> dict | None:
    """The last usage block in (the tail of) an SSE stream. Groq nests it in `x_groq`."""
    for line in reversed(tail.split(b"\n")):
        line = line.strip()
        if line.startswith(b"data: {") and b'"usage"' in line:
            try:
                d = json.loads(line[6:])
            except ValueError:
                continue
            u = d.get("usage") or (d.get("x_groq") or {}).get("usage")
            if u:
                return u
    return None


def with_stream_usage(payload: dict) -> dict:
    """Ask a streamed request to end with a usage chunk, unless the caller already chose."""
    if payload.get("stream") and "stream_options" not in payload:
        return {**payload, "stream_options": {"include_usage": True}}
    return payload


def observe_engine(identity: str, model: str, stats: Any) -> None:
    """Fold one ``/v1/stats`` sample from a local engine into the totals.

    `identity` must change when the engine restarts (instance id + start time): the counters
    are per process, so a new process starts a new baseline. Within one identity the counters
    only grow; a sample that reads lower is an older poll answered late (the console and the
    sampler poll concurrently) and is ignored rather than mistaken for a restart.
    """
    req = (stats or {}).get("requests") if isinstance(stats, dict) else None
    if not isinstance(req, dict) or not model:
        return
    try:
        now = (int(req["prompt_tokens_total"]), int(req["completion_tokens_total"]), int(req.get("completed") or 0))
    except (KeyError, TypeError, ValueError):
        return  # an engine that does not publish lifetime counters
    with _lock:
        prev = _marks.get(identity, ((0, 0, 0), 0.0))[0]
        _marks[identity] = (tuple(max(n, p) for n, p in zip(now, prev)), time.time())
    dp, dc, dn = (max(0, n - p) for n, p in zip(now, prev))
    if dp or dc or dn:
        record("local", model, dp, dc, requests=dn)


def forget_engines(keep: set[str]) -> None:
    """Drop the baselines of engines long gone, so the map does not grow forever."""
    cutoff = time.time() - FORGET_AFTER_S
    with _lock:
        for k in [k for k, (_, seen) in _marks.items() if k not in keep and seen < cutoff]:
            _marks.pop(k, None)


def flush() -> None:
    with _lock:
        if not _pending:
            return
        rows = [(b, s, m, *v) for (b, s, m), v in _pending.items()]
        _pending.clear()
        try:
            db = _db()
            db.executemany(
                "INSERT INTO usage (bucket, source, model, prompt_tokens, completion_tokens, requests, unmetered, "
                "first_ts, last_ts) VALUES (?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(bucket, source, model) DO UPDATE SET "
                "prompt_tokens = prompt_tokens + excluded.prompt_tokens, "
                "completion_tokens = completion_tokens + excluded.completion_tokens, "
                "requests = requests + excluded.requests, unmetered = unmetered + excluded.unmetered, "
                "first_ts = MIN(first_ts, excluded.first_ts), last_ts = MAX(last_ts, excluded.last_ts)", rows)
            db.commit()
        except sqlite3.Error:
            # Put them back: a locked or briefly unwritable file must not lose the counts.
            for b, s, m, pt, ct, n, un, first, last in rows:
                row = _pending.setdefault((b, s, m), [0, 0, 0, 0, first, last])
                row[0] += pt
                row[1] += ct
                row[2] += n
                row[3] += un
                row[4], row[5] = min(row[4], first), max(row[5], last)
            raise


async def run(sample: Callable[[], Awaitable[None]]) -> None:
    """Background loop: sample the engines every SAMPLE_S, flush every FLUSH_S."""
    flushed = time.monotonic()
    try:
        while True:
            try:
                await sample()
                if time.monotonic() - flushed >= FLUSH_S:
                    await asyncio.to_thread(flush)
                    flushed = time.monotonic()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 -- a poller must never die
                logger.debug("token sampling failed", exc_info=True)
            await asyncio.sleep(SAMPLE_S)
    finally:
        with contextlib.suppress(Exception):
            await asyncio.to_thread(flush)


# =======================================================================================
# Reading
# =======================================================================================
def _rows(since: float | None) -> list[dict]:
    flush()
    start = int(since // BUCKET_S) * BUCKET_S if since else 0
    with _lock:
        got = _db().execute(
            "SELECT source, model, SUM(prompt_tokens), SUM(completion_tokens), SUM(requests), SUM(unmetered), "
            "MIN(first_ts), MAX(last_ts) FROM usage WHERE bucket >= ? GROUP BY source, model", (start,)).fetchall()
    return [{"source": s, "model": m, "prompt_tokens": int(pt or 0), "completion_tokens": int(ct or 0),
             "requests": int(n or 0), "unmetered": int(un or 0), "first_seen": first, "last_seen": last}
            for s, m, pt, ct, n, un, first, last in got]


def summary(since: float | None = None, extra: list[dict] | None = None) -> dict:
    """Per-model rows (largest first) and totals, overall and per source.

    `extra` is rows kept elsewhere in the same shape (the external spend ledger); a model
    present in both is summed.
    """
    merged: dict[tuple[str, str], dict] = {}
    for r in _rows(since) + list(extra or []):
        key = (r["source"], r["model"])
        cur = merged.get(key)
        if cur is None:
            merged[key] = dict(r)
            continue
        for k in ("prompt_tokens", "completion_tokens", "requests", "unmetered"):
            cur[k] += r[k]
        firsts = [x for x in (cur["first_seen"], r["first_seen"]) if x]
        cur["first_seen"] = min(firsts) if firsts else None
        cur["last_seen"] = max(cur["last_seen"] or 0, r["last_seen"] or 0) or None
    models = []
    for r in merged.values():
        r["total_tokens"] = r["prompt_tokens"] + r["completion_tokens"]
        models.append(r)
    models.sort(key=lambda r: (-r["total_tokens"], -r["requests"], r["model"]))

    def total(rows: list[dict]) -> dict:
        return {k: sum(r[k] for r in rows)
                for k in ("prompt_tokens", "completion_tokens", "total_tokens", "requests", "unmetered")}

    return {"since": since, "models": models, "totals": total(models),
            "by_source": {s: total([r for r in models if r["source"] == s]) for s in SOURCES},
            "ts": time.time()}


WINDOWS = {"all": None, "24h": 86_400, "7d": 7 * 86_400, "30d": 30 * 86_400}


@router.get("/usage/tokens")
async def usage_tokens(since: str = Query("all", pattern="^(all|24h|7d|30d)$")) -> dict:
    """Tokens processed per model -- local engines, paired computers and external providers --
    over all time or a recent window (rounded out to whole hours)."""
    from . import external

    span = WINDOWS[since]
    start = time.time() - span if span else None

    def gather() -> dict:
        extra: list[dict] = []
        try:
            extra = external.token_rows(start)
        except sqlite3.Error:  # the ledger is someone else's file; its absence is not ours to fail on
            logger.warning("could not read the external ledger", exc_info=True)
        return summary(start, extra)

    return {**await asyncio.to_thread(gather), "window": since}
