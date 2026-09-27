"""OAuth 2.1 for a task server's HTTP transport -- so the control plane connects securely.

The server is its own authorization server AND resource server, with the smallest surface that
is still standard OAuth 2.1 (what the control plane's MCP client already speaks):

* ONE pre-registered confidential client -- the FreeSwarm control plane -- with a client id, a
  client secret and a fixed loopback redirect URI. Dynamic client registration is off.
* Authorization code + PKCE (S256). /authorize sends the operator to a consent page served by
  this server; it approves only with the APPROVAL PASSPHRASE (stored as an scrypt hash; five
  wrong tries lock approvals for ten minutes).
* Access tokens: HMAC-SHA256 signed, 1 hour, bound to this server's resource URL and the
  `tasks` scope. Refresh tokens: 256-bit random, stored only as SHA-256 hashes, 30 days, ROTATED
  on every use; both revocable (/revoke).
* Every /mcp request needs a valid bearer token (the SDK's middleware checks it).

Secrets live in <server>/.oauth/ (owner-only file permissions; git-ignored), created by the
server's make_oauth_secrets.py -- which also hands the control plane its half (see `generate`).
Serve over HTTPS (a reverse proxy) before exposing a server beyond this machine: OAuth only
allows plain http on loopback.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

SCOPE = "tasks"
ACCESS_TTL = 3600
REFRESH_TTL = 30 * 86400
CODE_TTL = 300
CONSENT_TTL = 600
MAX_FAILURES, LOCKOUT_S = 5, 600


# ---------------------------------------------------------------------------------------------
# Files: owner-only
# ---------------------------------------------------------------------------------------------
def restrict(path: Path) -> None:
    """Owner-only permissions on a secrets file (icacls on Windows, 0600 elsewhere)."""
    if sys.platform != "win32":
        os.chmod(path, 0o600)
        return
    user = os.environ.get("USERNAME") or ""
    if user:
        subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"],
                       capture_output=True, check=False)


def write_secret(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data if isinstance(data, str) else json.dumps(data, indent=1), encoding="utf-8")
    restrict(tmp)
    tmp.replace(path)
    restrict(path)


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _hash_passphrase(passphrase: str, salt: bytes) -> str:
    return hashlib.scrypt(passphrase.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32).hex()


# ---------------------------------------------------------------------------------------------
# The provider (mcp.server.auth.provider.OAuthAuthorizationServerProvider)
# ---------------------------------------------------------------------------------------------
class TaskServerOAuth:
    """Authorization server + token verifier for one task server. `oauth_dir` holds server.json
    (made by `generate`); `base_url` is where this server is reached (issuer)."""

    def __init__(self, oauth_dir: Path, base_url: str, name: str):
        self.dir = Path(oauth_dir)
        cfg = json.loads((self.dir / "server.json").read_text(encoding="utf-8"))
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.resource = self.base_url + "/mcp"
        self.key = bytes.fromhex(cfg["signing_key"])
        self.approval = cfg["approval"]
        self.client_cfg = cfg["client"]
        self._lock = threading.Lock()
        self._consents: dict[str, tuple[float, Any, Any]] = {}      # tx -> (expiry, client, params)
        self._codes: dict[str, Any] = {}                             # code -> AuthorizationCode
        self._failures: list[float] = []
        self._store_path = self.dir / "tokens.json"

    # ---- persistence of refresh tokens and revocations ------------------------------------
    def _store(self) -> dict[str, Any]:
        try:
            s = json.loads(self._store_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            s = {}
        now = time.time()
        s["refresh"] = {h: r for h, r in (s.get("refresh") or {}).items() if r["exp"] > now}
        s["revoked"] = {j: e for j, e in (s.get("revoked") or {}).items() if e > now}
        return s

    def _save(self, s: dict[str, Any]) -> None:
        write_secret(self._store_path, s)

    # ---- clients -------------------------------------------------------------------------
    async def get_client(self, client_id: str):
        from mcp.shared.auth import OAuthClientInformationFull

        c = self.client_cfg
        if not hmac.compare_digest(client_id.encode(), c["client_id"].encode()):
            return None
        return OAuthClientInformationFull(
            client_id=c["client_id"], client_secret=c["client_secret"], redirect_uris=c["redirect_uris"],
            grant_types=["authorization_code", "refresh_token"], response_types=["code"],
            token_endpoint_auth_method="client_secret_post", scope=SCOPE,
            client_name="FreeSwarm control plane")

    async def register_client(self, client_info) -> None:
        raise NotImplementedError("dynamic client registration is disabled -- use make_oauth_secrets.py")

    # ---- authorization code + PKCE -----------------------------------------------------------
    async def authorize(self, client, params) -> str:
        tx = secrets.token_urlsafe(24)
        with self._lock:
            now = time.time()
            self._consents = {k: v for k, v in self._consents.items() if v[0] > now}
            self._consents[tx] = (now + CONSENT_TTL, client, params)
        return f"{self.base_url}/oauth/consent?tx={tx}"

    def consent_page(self, tx: str, error: str = "") -> str:
        with self._lock:
            entry = self._consents.get(tx)
        if not entry or entry[0] < time.time():
            return _page("This approval request has expired", "<p>Start the connection again from the FreeSwarm console.</p>")
        _, client, params = entry
        scopes = " ".join(params.scopes or [SCOPE])
        body = (f"<p><b>{html.escape(client.client_name or client.client_id)}</b> asks to use the task server "
                f"<b>{html.escape(self.name)}</b> (scope <code>{html.escape(scopes)}</code>).</p>"
                f"<p>It will be redirected back to <code>{html.escape(str(params.redirect_uri))}</code>.</p>"
                + (f"<p class=err>{html.escape(error)}</p>" if error else "")
                + f'<form method="post" action="/oauth/consent"><input type="hidden" name="tx" value="{html.escape(tx)}">'
                  '<label>Approval passphrase<br><input type="password" name="passphrase" autofocus autocomplete="off"></label>'
                  '<div class=row><button name="decision" value="approve">Approve</button> '
                  '<button name="decision" value="deny" class=secondary>Deny</button></div></form>'
                  "<p class=small>The passphrase was created by this server's make_oauth_secrets.py.</p>")
        return _page(f"Connect to {self.name}", body)

    def consent_submit(self, tx: str, passphrase: str, decision: str) -> tuple[str | None, str | None]:
        """(redirect URL, error to show on the page)."""
        from mcp.server.auth.provider import AuthorizationCode, construct_redirect_uri

        with self._lock:
            entry = self._consents.get(tx)
            now = time.time()
            if not entry or entry[0] < now:
                return None, "This approval request has expired -- start again from the console."
            _, client, params = entry
            if decision != "approve":
                del self._consents[tx]
                return construct_redirect_uri(str(params.redirect_uri), error="access_denied", state=params.state), None
            self._failures = [f for f in self._failures if f > now - LOCKOUT_S]
            if len(self._failures) >= MAX_FAILURES:
                return None, "Too many wrong passphrases -- approvals are locked for ten minutes."
            want = self.approval["hash"]
            got = _hash_passphrase(passphrase or "", bytes.fromhex(self.approval["salt"]))
            if not hmac.compare_digest(want, got):
                self._failures.append(now)
                return None, "Wrong passphrase."
            del self._consents[tx]
            code = secrets.token_urlsafe(32)
            self._codes[code] = AuthorizationCode(
                code=code, scopes=params.scopes or [SCOPE], expires_at=now + CODE_TTL, client_id=client.client_id,
                code_challenge=params.code_challenge, redirect_uri=params.redirect_uri,
                redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
                resource=params.resource, subject="operator")
            return construct_redirect_uri(str(params.redirect_uri), code=code, state=params.state), None

    async def load_authorization_code(self, client, authorization_code: str):
        with self._lock:
            c = self._codes.get(authorization_code)
        if c is None or c.client_id != client.client_id or c.expires_at < time.time():
            return None
        return c

    async def exchange_authorization_code(self, client, authorization_code):
        with self._lock:
            if self._codes.pop(authorization_code.code, None) is None:        # one use only
                from mcp.server.auth.provider import TokenError
                raise TokenError("invalid_grant", "authorization code already used")
        return self._issue(client.client_id, authorization_code.scopes)

    # ---- tokens --------------------------------------------------------------------------------
    def _issue(self, client_id: str, scopes: list[str]):
        from mcp.shared.auth import OAuthToken

        now = int(time.time())
        payload = {"jti": secrets.token_urlsafe(12), "cid": client_id, "scp": scopes, "exp": now + ACCESS_TTL,
                   "aud": self.resource, "iat": now}
        body = _b64(json.dumps(payload, separators=(",", ":")).encode())
        access = f"fsat.{body}.{_b64(hmac.new(self.key, body.encode(), hashlib.sha256).digest())}"
        refresh = secrets.token_urlsafe(32)
        with self._lock:
            s = self._store()
            s["refresh"][hashlib.sha256(refresh.encode()).hexdigest()] = {
                "cid": client_id, "scp": scopes, "exp": now + REFRESH_TTL}
            self._save(s)
        return OAuthToken(access_token=access, expires_in=ACCESS_TTL, scope=" ".join(scopes), refresh_token=refresh)

    async def load_refresh_token(self, client, refresh_token: str):
        from mcp.server.auth.provider import RefreshToken

        with self._lock:
            r = self._store()["refresh"].get(hashlib.sha256(refresh_token.encode()).hexdigest())
        if not r or r["cid"] != client.client_id:
            return None
        return RefreshToken(token=refresh_token, client_id=r["cid"], scopes=r["scp"], expires_at=int(r["exp"]),
                            resource=self.resource, subject="operator")

    async def exchange_refresh_token(self, client, refresh_token, scopes: list[str]):
        with self._lock:
            s = self._store()
            if s["refresh"].pop(hashlib.sha256(refresh_token.token.encode()).hexdigest(), None) is None:
                from mcp.server.auth.provider import TokenError
                raise TokenError("invalid_grant", "refresh token already used or revoked")
            self._save(s)                                                      # rotation: the old one dies
        return self._issue(client.client_id, [x for x in (scopes or refresh_token.scopes) if x in refresh_token.scopes])

    def verify(self, token: str) -> dict[str, Any] | None:
        try:
            prefix, body, sig = token.split(".")
        except ValueError:
            return None
        if prefix != "fsat":
            return None
        want = hmac.new(self.key, body.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(want, _unb64(sig)):
            return None
        p = json.loads(_unb64(body))
        if p["exp"] < time.time() or p["aud"] != self.resource:
            return None
        with self._lock:
            if p["jti"] in self._store()["revoked"]:
                return None
        return p

    async def load_access_token(self, token: str):
        from mcp.server.auth.provider import AccessToken

        p = self.verify(token)
        if p is None:
            return None
        return AccessToken(token=token, client_id=p["cid"], scopes=p["scp"], expires_at=int(p["exp"]),
                           resource=self.resource, subject="operator")

    async def revoke_token(self, token) -> None:
        with self._lock:
            s = self._store()
            s["refresh"].pop(hashlib.sha256(token.token.encode()).hexdigest(), None)
            p = self.verify(token.token) if token.token.startswith("fsat.") else None
            if p:
                s["revoked"][p["jti"]] = p["exp"]
            self._save(s)


def _page(title: str, body: str) -> str:
    return ("<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>"
            ":root{color-scheme:light dark;--bg:#fff;--fg:#1a1a1a;--dim:#666;--acc:#2563eb;--err:#b91c1c}"
            "@media (prefers-color-scheme:dark){:root{--bg:#111418;--fg:#e8e8e8;--dim:#9aa;--acc:#60a5fa;--err:#f87171}}"
            "body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:480px;margin:10vh auto;padding:0 16px}"
            "input{width:100%;padding:8px;margin-top:4px;font:inherit}button{padding:8px 16px;font:inherit;background:var(--acc);"
            "color:#fff;border:0;border-radius:6px;cursor:pointer}.secondary{background:transparent;color:var(--fg);border:1px solid var(--dim)}"
            ".row{margin-top:16px}.err{color:var(--err)}.small{color:var(--dim);font-size:13px}code{font-size:13px}"
            f"</style></head><body><h2>{html.escape(title)}</h2>{body}</body></html>")


def mount(mcp, provider: TaskServerOAuth) -> None:
    """The consent page routes (not behind the bearer check -- they are part of the flow)."""
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, RedirectResponse

    @mcp.custom_route("/oauth/consent", methods=["GET", "POST"])
    async def consent(request: Request):
        headers = {"Cache-Control": "no-store", "X-Frame-Options": "DENY",
                   "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'"}
        if request.method == "GET":
            return HTMLResponse(provider.consent_page(request.query_params.get("tx", "")), headers=headers)
        form = await request.form()
        tx = str(form.get("tx", ""))
        url, error = provider.consent_submit(tx, str(form.get("passphrase", "")), str(form.get("decision", "")))
        if url:
            return RedirectResponse(url, status_code=302, headers=headers)
        return HTMLResponse(provider.consent_page(tx, error or ""), status_code=400, headers=headers)


def auth_kwargs(oauth_dir: Path, base_url: str, name: str) -> tuple[dict[str, Any], TaskServerOAuth]:
    """The MCPServer(...) keyword arguments that switch OAuth on, and the provider."""
    from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions

    provider = TaskServerOAuth(oauth_dir, base_url, name)
    settings = AuthSettings(issuer_url=base_url, resource_server_url=base_url.rstrip("/") + "/mcp",
                            required_scopes=[SCOPE], client_registration_options=ClientRegistrationOptions(enabled=False),
                            revocation_options=RevocationOptions(enabled=True),
                            validate_token_resource=True)       # a token for another server is refused
    return {"auth": settings, "auth_server_provider": provider}, provider


# ---------------------------------------------------------------------------------------------
# make_oauth_secrets.py: create the server's secrets and hand the control plane its half
# ---------------------------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
CONTROL_PLANE = REPO / "ui" / "backend"


def generate(server_dir: Path, name: str, port: int, argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=f"Create OAuth secrets for the task server '{name}' and register it "
                                             "with the FreeSwarm control plane.")
    ap.add_argument("--rotate", action="store_true", help="replace existing secrets (connected clients must reconnect)")
    ap.add_argument("--host", default="127.0.0.1", help="host the server is reached at (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=port, help=f"HTTP port of the server (default {port})")
    ap.add_argument("--control-plane", default="http://127.0.0.1:8000",
                    help="the control plane's own URL (its OAuth callback is registered as the only redirect)")
    ap.add_argument("--passphrase", help="use this approval passphrase instead of a generated one")
    ap.add_argument("--no-register", action="store_true", help="do not touch ui/backend/mcp_servers.json")
    ap.add_argument("--quiet", action="store_true", help="do not print the passphrase (it is saved owner-only)")
    a = ap.parse_args(argv)

    odir = Path(server_dir) / ".oauth"
    if (odir / "server.json").is_file() and not a.rotate:
        print(f"{odir / 'server.json'} exists -- pass --rotate to replace it (clients must then reconnect).")
        sys.exit(1)
    passphrase = a.passphrase or "-".join(secrets.choice(_WORDS) for _ in range(6))
    salt = secrets.token_bytes(16)
    client_id = f"freeswarm-{name}-{secrets.token_hex(4)}"
    client_secret = secrets.token_urlsafe(32)
    redirect = a.control_plane.rstrip("/") + "/api/mcp/oauth/callback"
    write_secret(odir / "server.json", {
        "server": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "client": {"client_id": client_id, "client_secret": client_secret, "redirect_uris": [redirect]},
        "approval": {"salt": salt.hex(), "hash": _hash_passphrase(passphrase, salt)},
        "signing_key": secrets.token_hex(32),
    })
    (odir / "tokens.json").unlink(missing_ok=True)                      # old tokens die with old keys
    write_secret(odir / "approval_passphrase.txt",
                 f"{passphrase}\n\nApproval passphrase for task server '{name}'. Store it somewhere safe and delete this file.\n")
    secret_file = CONTROL_PLANE / "auth" / "mcp_clients" / f"{name}.secret"
    write_secret(secret_file, client_secret)
    url = f"http://{a.host}:{a.port}/mcp"
    if not a.no_register:
        cfg_path = CONTROL_PLANE / "mcp_servers.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {"servers": []}
        servers = [s for s in cfg.get("servers", []) if s.get("name") != name]
        servers.append({"name": name, "kind": "task", "transport": "http", "url": url, "oauth": True, "client_id": client_id,
                        "client_secret_file": str(secret_file), "scopes": [SCOPE], "enabled": True})
        cfg["servers"] = servers
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"OAuth secrets for task server '{name}':")
    print(f"  server secrets   {odir / 'server.json'}   (owner-only; never share)")
    print(f"  client secret    {secret_file}   (owner-only; read by the control plane)")
    print(f"  registered       {'no (--no-register)' if a.no_register else 'ui/backend/mcp_servers.json -> ' + name + ' (http, oauth)'}")
    print(f"  approval phrase  {'saved to ' + str(odir / 'approval_passphrase.txt') if a.quiet else passphrase}")
    print(f"\nStart the server over HTTP ({url}), then in the console open Connectors -> {name} -> Connect,")
    print("and approve with the passphrase. The connection then refreshes itself for 30 days of use.")


# Six words from this list = ~41 bits, stretched by scrypt and rate-limited to 5 tries per 10 minutes.
_WORDS = ("amber anchor apple arrow aspen atlas autumn badge bamboo banner barley basin beacon birch bison "
          "blossom boulder bramble breeze bridge bronze brook cabin cactus canyon carbon cedar chalk cherry cinder "
          "citrus clover cobalt comet copper coral cotton crane crystal cypress dawn delta desert dolphin dune eagle "
          "ember falcon fern fjord flint forest fossil frost garnet glacier granite grove harbor hazel heron hollow "
          "horizon iris island ivory jade jasper juniper kelp kestrel lagoon lantern larch lava lemon lilac linen "
          "lotus lumen maple marble meadow mesa meteor mint mist monsoon moss nectar nickel nova oak oasis ocean "
          "olive onyx opal orchid otter pebble pepper pine plume polar poppy prairie quartz quill raven reed ridge "
          "river robin saffron sage salt sierra silver slate spruce storm summit sunset tango thistle thunder "
          "tide timber topaz tulip tundra valley velvet violet walnut willow winter zephyr").split()
