"""Windows-native runtime shims for FreeToken. Imported at package init; no-op elsewhere.

pyzmq's ``zmq.asyncio`` requires a selector event loop. Python 3.13 on Windows defaults to
the Proactor loop, which lacks the ``add_reader`` family and breaks the frontend's delivery
of generated tokens from the scheduler. We force the selector policy process-wide (this runs
in every process that imports freetoken: the API frontend and the spawned worker processes).
"""
from __future__ import annotations

import sys


def apply() -> None:
    if sys.platform != "win32":
        return
    import asyncio

    policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy is None:
        return
    try:
        asyncio.set_event_loop_policy(policy())
    except Exception:
        pass


apply()
