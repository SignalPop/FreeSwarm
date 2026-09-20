"""FreeSwarm's version -- the one place it is set.

Two numbers, because they answer different questions:

* ``APP_VERSION`` (semver) -- which release this is. Shown in the console, reported to other
  computers. Two computers on different app versions can still work together.
* ``FEDERATION_PROTOCOL`` -- the wire contract between computers (endpoints, token grants,
  model list shape, relay format). Bump it when a change would break an older peer, and set
  ``MIN_FEDERATION_PROTOCOL`` to the oldest protocol this build still speaks. Two computers
  are compatible when each one's protocol is at least the other's minimum.

Release checklist: bump APP_VERSION here (and ui/frontend/package.json to match); bump the
protocol only for a breaking federation change.
"""

from __future__ import annotations

APP_VERSION = "1.1.0"
FEDERATION_PROTOCOL = 1
MIN_FEDERATION_PROTOCOL = 1


def info() -> dict:
    return {"app_version": APP_VERSION, "protocol": FEDERATION_PROTOCOL, "min_protocol": MIN_FEDERATION_PROTOCOL}


def compatible(peer_protocol: int | None, peer_min_protocol: int | None) -> tuple[bool, str]:
    """(compatible?, reason) for a peer that speaks `peer_protocol` and accepts down to
    `peer_min_protocol`. A peer that reports nothing predates versioning: protocol 1."""
    theirs = int(peer_protocol or 1)
    their_min = int(peer_min_protocol or theirs)
    if theirs < MIN_FEDERATION_PROTOCOL:
        return False, (f"the other computer speaks federation protocol {theirs}; this one needs at least "
                       f"{MIN_FEDERATION_PROTOCOL} -- update FreeSwarm on the other computer")
    if FEDERATION_PROTOCOL < their_min:
        return False, (f"the other computer needs federation protocol {their_min} or newer; this one speaks "
                       f"{FEDERATION_PROTOCOL} -- update FreeSwarm on this computer")
    return True, "compatible"


def describe_difference(peer_app: str | None) -> str | None:
    """A note when app versions differ (still compatible), None when they match."""
    if not peer_app or peer_app == APP_VERSION:
        return None
    return f"different release (this computer {APP_VERSION}, other {peer_app}) -- compatible, but consider updating both"
