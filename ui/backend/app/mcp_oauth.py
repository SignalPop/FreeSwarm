"""OAuth 2.1 for remote MCP servers.

A remote MCP server (http / sse transport) usually sits behind OAuth rather than a static
key. The MCP authorization spec builds on plain OAuth 2.1: discover the authorization
server from the protected resource, optionally register this client dynamically
(RFC 7591), then run authorization-code-with-PKCE and send the access token as a bearer.
The SDK implements all of that in `OAuthClientProvider`; what it cannot do is drive a
browser, so this module supplies the two halves it delegates:

  * a **redirect handler** -- which here does not open a browser, it parks the URL so the
    HTTP layer can hand it to the operator, and
  * a **callback handler** -- which waits for the loopback redirect to arrive.

The flow is therefore: the UI asks to connect, gets an authorization URL, the operator
approves in their browser, the provider's redirect lands on `/api/mcp/oauth/callback`, and
the waiting flow completes the token exchange.

**Token storage is the sensitive part.** Refresh tokens are long-lived credentials for a
third-party account, so they are written to `ui/backend/mcp_tokens.json` with the same
owner-only ACL treatment as the JWT signing key, never logged, and never returned by any
API -- the UI only ever learns *whether* a server is authorised, not with what.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("freetoken.mcp.oauth")

TOKENS_PATH = Path(__file__).resolve().parent.parent / "mcp_tokens.json"

# The loopback redirect this control plane listens on. RFC 8252 (OAuth for native apps)
# blesses a loopback URI for exactly this case; the port must match the control plane's.
DEFAULT_REDIRECT_PATH = "/api/mcp/oauth/callback"

# How long an operator has to finish approving in the browser before the flow is dropped.
AUTH_TIMEOUT_S = 300.0


# =======================================================================================
# Persistent token store
# =======================================================================================
_store_lock = threading.Lock()


def _restrict(path: Path) -> None:
    """Owner-only ACLs. These are third-party refresh tokens, not app config."""
    import os
    import subprocess
    import sys

    if sys.platform != "win32":
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return
    try:
        subprocess.run(
            ["icacls", str(path), "/inheritance:r",
             "/grant:r", f"{os.environ.get('USERNAME', '')}:F", "/grant:r", "SYSTEM:F"],
            capture_output=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def _read_all() -> dict[str, Any]:
    if not TOKENS_PATH.is_file():
        return {}
    try:
        data = json.loads(TOKENS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        logger.warning("mcp_tokens.json unreadable; treating every server as unauthorised")
        return {}


def _write_all(data: dict[str, Any]) -> None:
    TOKENS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _restrict(TOKENS_PATH)


class FileTokenStorage:
    """`TokenStorage` for one server, persisted to disk.

    The SDK's protocol is async; the file work is trivial and synchronous, so these just
    do it inline rather than pretending to be I/O-bound.
    """

    def __init__(self, server: str, preset_client: Any | None = None) -> None:
        self.server = server
        # When the operator supplied client_id/secret in mcp_servers.json, hand those to
        # the SDK instead of letting it attempt Dynamic Client Registration. Most real
        # providers (GitHub among them) have no registration endpoint and answer 404.
        self.preset_client = preset_client

    def _entry(self) -> dict:
        return _read_all().get(self.server, {})

    def _update(self, **fields: Any) -> None:
        with _store_lock:
            data = _read_all()
            entry = data.get(self.server, {})
            entry.update(fields)
            data[self.server] = entry
            _write_all(data)

    async def get_tokens(self):
        from mcp.shared.auth import OAuthToken

        raw = self._entry().get("tokens")
        if not raw:
            return None
        try:
            return OAuthToken.model_validate(raw)
        except Exception:  # noqa: BLE001 - a corrupt entry must not block re-authorising
            return None

    async def set_tokens(self, tokens) -> None:
        self._update(
            tokens=json.loads(tokens.model_dump_json(exclude_none=True)),
            obtained_at=time.time(),
        )

    async def get_client_info(self):
        from mcp.shared.auth import OAuthClientInformationFull

        if self.preset_client is not None:
            return self.preset_client
        raw = self._entry().get("client_info")
        if not raw:
            return None
        try:
            return OAuthClientInformationFull.model_validate(raw)
        except Exception:  # noqa: BLE001
            return None

    async def set_client_info(self, client_info) -> None:
        # Dynamic client registration result -- reused on later flows so the server does
        # not accumulate a new client registration per connection attempt. Skipped when a
        # client was preset, since there is nothing dynamic to remember.
        if self.preset_client is not None:
            return
        self._update(client_info=json.loads(client_info.model_dump_json(exclude_none=True)))


def authorised_servers() -> set[str]:
    """Names with a stored token. Used only to show connected/disconnected in the UI."""
    return {name for name, entry in _read_all().items() if entry.get("tokens")}


def token_summary(server: str) -> dict:
    """Non-secret description of a stored token, safe to return over HTTP."""
    entry = _read_all().get(server, {})
    tokens = entry.get("tokens") or {}
    if not tokens:
        return {"authorised": False}
    obtained = entry.get("obtained_at")
    expires_in = tokens.get("expires_in")
    return {
        "authorised": True,
        "scope": tokens.get("scope"),
        "has_refresh_token": bool(tokens.get("refresh_token")),
        "obtained_at": obtained,
        # Never the token itself -- only when it lapses.
        "expires_at": (obtained + expires_in) if (obtained and expires_in) else None,
    }


def forget(server: str) -> bool:
    """Drop stored credentials for a server (the UI's Disconnect)."""
    with _store_lock:
        data = _read_all()
        if server not in data:
            return False
        data.pop(server)
        _write_all(data)
    return True


# =======================================================================================
# In-flight authorisation flows
# =======================================================================================
@dataclass
class PendingFlow:
    """One browser round-trip, waiting between 'we made a URL' and 'the code came back'."""

    server: str
    state: str
    created_at: float = field(default_factory=time.time)
    authorization_url: str | None = None
    # Set by the redirect handler once the SDK produces the URL.
    url_ready: asyncio.Event = field(default_factory=asyncio.Event)
    # Set by the HTTP callback once the provider returns.
    code_ready: asyncio.Event = field(default_factory=asyncio.Event)
    code: str | None = None
    returned_state: str | None = None
    error: str | None = None


_flows: dict[str, PendingFlow] = {}
_flows_lock = threading.Lock()


def _reap_expired() -> None:
    cutoff = time.time() - AUTH_TIMEOUT_S
    with _flows_lock:
        for key in [k for k, f in _flows.items() if f.created_at < cutoff]:
            _flows.pop(key, None)


def register_flow(server: str) -> PendingFlow:
    _reap_expired()
    flow = PendingFlow(server=server, state=secrets.token_urlsafe(24))
    with _flows_lock:
        _flows[flow.state] = flow
    return flow


def get_flow(state: str) -> PendingFlow | None:
    _reap_expired()
    with _flows_lock:
        return _flows.get(state)


def finish_flow(state: str, code: str | None, error: str | None = None) -> bool:
    """Called from the HTTP callback route when the browser comes back."""
    flow = get_flow(state)
    if flow is None:
        return False
    flow.code = code
    flow.returned_state = state
    flow.error = error
    flow.code_ready.set()
    return True


def drop_flow(state: str) -> None:
    with _flows_lock:
        _flows.pop(state, None)


def preset_client_info(spec: Any, redirect_uri: str):
    """`OAuthClientInformationFull` from configured credentials, or None for DCR."""
    client_id = getattr(spec, "client_id", None)
    if not client_id:
        return None
    from mcp.shared.auth import OAuthClientInformationFull

    return OAuthClientInformationFull(
        client_id=client_id,
        client_secret=spec.resolved_client_secret(),
        redirect_uris=[redirect_uri],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method=(
            "client_secret_post" if spec.resolved_client_secret() else "none"
        ),
        scope=" ".join(spec.scopes) if getattr(spec, "scopes", None) else None,
    )


def build_provider(
    server_name: str,
    server_url: str,
    redirect_uri: str,
    flow: PendingFlow,
    spec: Any | None = None,
):
    """An `OAuthClientProvider` wired to this module's browser round-trip."""
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientMetadata

    async def redirect_handler(authorization_url: str) -> None:
        # Deliberately NOT webbrowser.open(): the control plane may be running headless
        # or over an SSH tunnel, where opening a browser on the *server* is useless. The
        # URL is handed back over HTTP so it opens on whichever machine the operator is at.
        flow.authorization_url = authorization_url
        flow.url_ready.set()

    async def callback_handler():
        from mcp.client.auth import AuthorizationCodeResult

        try:
            await asyncio.wait_for(flow.code_ready.wait(), timeout=AUTH_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"no authorization callback within {AUTH_TIMEOUT_S:.0f}s"
            ) from None
        if flow.error:
            raise RuntimeError(f"authorization failed: {flow.error}")
        if not flow.code:
            raise RuntimeError("authorization callback carried no code")
        return AuthorizationCodeResult(code=flow.code, state=flow.returned_state)

    preset = preset_client_info(spec, redirect_uri) if spec is not None else None
    metadata = OAuthClientMetadata(
        client_name="FreeSwarm Console",
        redirect_uris=[redirect_uri],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="client_secret_post",
        scope=" ".join(spec.scopes) if spec is not None and getattr(spec, "scopes", None) else None,
    )

    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=metadata,
        storage=FileTokenStorage(server_name, preset),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )


def build_stored_provider(
    server_name: str, server_url: str, redirect_uri: str, spec: Any | None = None
):
    """A provider for ordinary use: refreshes silently, never starts a browser flow.

    If the stored refresh token has lapsed the SDK would otherwise try to open an
    interactive authorization, which is meaningless during a background tool call. Both
    handlers therefore fail fast so the caller reports "needs authorisation" instead of
    hanging until the flow times out.
    """
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientMetadata

    async def redirect_handler(authorization_url: str) -> None:
        raise RuntimeError(
            "this MCP server needs authorisation; connect it from the Connectors page"
        )

    async def callback_handler():
        raise RuntimeError(
            "this MCP server needs authorisation; connect it from the Connectors page"
        )

    preset = preset_client_info(spec, redirect_uri) if spec is not None else None
    metadata = OAuthClientMetadata(
        client_name="FreeSwarm Console",
        redirect_uris=[redirect_uri],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="client_secret_post",
    )
    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=metadata,
        storage=FileTokenStorage(server_name, preset),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )
