"""Live events for the console: what the swarm just did, pushed instead of polled.

The Work page used to learn about progress by polling every 15 s, and the runner's activity
posts reached it only on the next poll. This module is an in-process fan-out: the handlers
that learn something (an agent's activity post, a candidate scored, a new champion, a
steering message queued or delivered) call ``publish``; every console tab watching that
project gets the event over Server-Sent Events from ``GET /api/projects/{id}/events``.

Events are hints, not state: a client that misses some (it reconnected, its queue was full)
loses nothing, because each event makes it re-read the real thing from the normal endpoints.
So the stream is lossy on purpose -- a slow tab is dropped events, never allowed to back up
the handlers that publish them.

``publish`` is safe from any thread (the activity flush runs in a worker thread) and never
raises: an event that cannot be delivered must not fail the work it reports.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import threading
import time
from typing import Any, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

logger = logging.getLogger("freetoken.events")
router = APIRouter(tags=["events"])

# Events buffered per subscriber before newer ones are dropped (a tab in the background).
QUEUE_MAX = 500
# A comment line this often keeps proxies (and the Next.js rewrite) from closing an idle stream.
KEEPALIVE_S = 15.0

_lock = threading.Lock()
_ids = itertools.count(1)
# subscriber id -> (project_id or None for every project, queue, the loop the queue belongs to)
_subs: dict[int, tuple[str | None, asyncio.Queue, asyncio.AbstractEventLoop]] = {}
_seq = itertools.count(1)


def publish(project_id: str | None, kind: str, data: dict[str, Any] | None = None) -> None:
    """Tell every subscriber of `project_id` (and every all-projects subscriber) that `kind`
    happened. Thread-safe; never raises; drops the event for a subscriber whose queue is full."""
    try:
        event = {"id": next(_seq), "kind": kind, "project_id": project_id, "at": time.time(), **(data or {})}
        with _lock:
            targets = [(q, loop) for pid, q, loop in _subs.values() if pid is None or pid == project_id]
        for q, loop in targets:
            try:
                loop.call_soon_threadsafe(_offer, q, event)
            except RuntimeError:
                pass  # that subscriber's loop is closed; it unsubscribes itself
    except Exception as exc:  # noqa: BLE001 -- an event must never fail the work it reports
        logger.debug("events: publish %s failed: %s", kind, exc)


def _offer(q: asyncio.Queue, event: dict) -> None:
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        pass


def subscribers() -> int:
    with _lock:
        return len(_subs)


async def stream(project_id: str | None, keepalive_s: float = KEEPALIVE_S) -> AsyncIterator[bytes]:
    """SSE frames for one subscriber until the client goes away."""
    q: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)
    sid = next(_ids)
    with _lock:
        _subs[sid] = (project_id, q, asyncio.get_running_loop())
    try:
        # First frame at once: the client knows the stream is up before anything happens.
        yield f"event: hello\ndata: {json.dumps({'at': time.time(), 'project_id': project_id})}\n\n".encode()
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=keepalive_s)
            except asyncio.TimeoutError:
                yield b": keepalive\n\n"
                continue
            yield (f"id: {event['id']}\nevent: {event['kind']}\n"
                   f"data: {json.dumps(event, default=str)}\n\n").encode()
    finally:
        with _lock:
            _subs.pop(sid, None)


@router.get("/projects/{project_id}/events")
async def project_events(project_id: str) -> StreamingResponse:
    """Server-Sent Events for one project: activity, candidate, champion, steer, steer_delivered."""
    return StreamingResponse(stream(project_id), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
