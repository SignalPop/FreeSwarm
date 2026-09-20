"""LAN federation: use models running on other FreeSwarm computers as if they were local.

Two roles, and a computer can play both:

* **Sharing** (the machine with the model, "B"). When the operator turns sharing on, this
  process starts a second listener -- the *gateway* -- on the LAN (default port 8443), TLS
  only, with a small surface: node info, the OAuth endpoints, the list of models the operator
  chose to share, and chat completions for those models. Nothing else of the control plane
  is reachable from the network. It also broadcasts a discovery beacon.
* **Using** (the machine that wants the model, "A"). A listens for beacons, lists the
  computers it hears, and pairs with one. Once paired, B's shared models appear here as
  ``<model>@<computer>`` -- in the model lists, the project model picker and the swarm's
  agents -- and ``/v1`` requests for them are relayed to B.

**Security** (LAN or not, the connection is treated as hostile):

* *Transport.* Every federation request is HTTPS. Each node has a self-signed certificate;
  A pins B's exact certificate at pairing time (trust on first use), and both screens show
  its SHA-256 fingerprint so the operator can confirm they match. A later certificate change
  is refused, not silently accepted.
* *Authorization: OAuth 2.0 Device Authorization Grant (RFC 8628).* A asks B for a device
  code; B shows the request -- with A's name, address and a short user code -- on B's OWN
  console (loopback only), and nothing is granted until B's operator approves it there.
  A polls B's token endpoint and receives an access token (1 h) and a refresh token (30 d,
  rotated on every use). B stores only SHA-256 hashes of tokens. Scope: ``inference``.
* *Least privilege.* A token can only list and call the models B's operator ticked, only
  ``chat/completions`` and ``completions``, with a per-client concurrency cap. The gateway
  refuses any source address that is not private, loopback or link-local.
* *Revocation.* B's operator can revoke a client at any time (its tokens stop working
  immediately); A's operator can disconnect, which also revokes at B.

State lives in ui/backend/auth/federation.json (permissions restricted like the other
secrets); the certificate and key beside it.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import ipaddress
import json
import logging
import os
import re
import secrets
import socket
import ssl
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import version as V

logger = logging.getLogger("freetoken.federation")

AUTH_DIR = Path(__file__).resolve().parent.parent / "auth"
STATE_PATH = AUTH_DIR / "federation.json"
CERT_PATH = AUTH_DIR / "federation_cert.pem"
KEY_PATH = AUTH_DIR / "federation_key.pem"

GATEWAY_PORT = int(os.getenv("FREESWARM_FED_PORT", "8443"))
DISCOVERY_PORT = int(os.getenv("FREESWARM_FED_DISCOVERY_PORT", "19191"))
BEACON_S = 5.0
PEER_STALE_S = 30.0
PROTOCOL = V.FEDERATION_PROTOCOL

ACCESS_TTL_S = 3600
REFRESH_TTL_S = 30 * 24 * 3600
DEVICE_TTL_S = 600
POLL_INTERVAL_S = 3
# A swarm routinely puts more than a handful of agents on one model, and the engine batches
# them anyway; 4 made a shared model the bottleneck of an otherwise parallel team. The using
# side queues to this number rather than overrunning it (see _slots).
MAX_CONCURRENT_PER_CLIENT = 8
# What protocol 1 shipped with, before /fed/info and /fed/models advertised the number. What
# an un-upgraded peer still enforces, and so what this side must assume when asking one.
LEGACY_MAX_CONCURRENT = 4
SCOPE = "inference"
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
ALLOWED_PATHS = {"chat/completions", "completions"}

_lock = threading.RLock()


# =======================================================================================
# Persistent state
# =======================================================================================
def _restrict(path: Path) -> None:
    try:
        from .auth import _restrict_permissions  # noqa: PLC0415

        _restrict_permissions(path)
    except Exception:  # noqa: BLE001 -- best effort, same as the other secret files
        pass


def _load() -> dict:
    with _lock:
        try:
            st = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st = {}
        changed = False
        if not st.get("node_id"):
            st["node_id"] = uuid.uuid4().hex
            changed = True
        st.setdefault("name", socket.gethostname())
        st.setdefault("sharing", False)
        st.setdefault("shared_models", [])
        st.setdefault("clients", {})   # served side: client_id -> record (token hashes only)
        st.setdefault("peers", {})     # using side: node_id -> connection (pinned cert, tokens)
        if changed:
            _save(st)
        return st


def _save(st: dict) -> None:
    with _lock:
        AUTH_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=2), encoding="utf-8")
        os.replace(tmp, STATE_PATH)
        _restrict(STATE_PATH)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def node_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")[:32] or "node"


# =======================================================================================
# Certificate
# =======================================================================================
def ensure_cert(name: str) -> tuple[str, str]:
    """(cert pem, fingerprint). Self-signed ECDSA P-256, created once and kept."""
    if not CERT_PATH.is_file() or not KEY_PATH.is_file():
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID

        key = ec.generate_private_key(ec.SECP256R1())
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"FreeToken {name}"[:64]),
                             x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FreeToken federation")])
        now = _dt.datetime.now(_dt.timezone.utc)
        ski = x509.SubjectKeyIdentifier.from_public_key(key.public_key())
        # A proper self-signed CA certificate: Python 3.13 verifies with X509_STRICT, which
        # rejects a pinned trust anchor missing these extensions.
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject).issuer_name(subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - _dt.timedelta(days=1))
            .not_valid_after(now + _dt.timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                         content_commitment=False, key_encipherment=False,
                                         data_encipherment=False, key_agreement=False,
                                         encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(ski, critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(ski), critical=False)
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(socket.gethostname()),
                                                        x509.DNSName("localhost")]), critical=False)
            .sign(key, hashes.SHA256())
        )
        AUTH_DIR.mkdir(parents=True, exist_ok=True)
        KEY_PATH.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                               serialization.NoEncryption()))
        _restrict(KEY_PATH)
        CERT_PATH.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    pem = CERT_PATH.read_text(encoding="ascii")
    return pem, fingerprint(pem)


def fingerprint(pem: str) -> str:
    der = ssl.PEM_cert_to_DER_cert(pem)
    h = hashlib.sha256(der).hexdigest().upper()
    return ":".join(h[i:i + 2] for i in range(0, len(h), 2))


def short_fp(fp: str) -> str:
    return fp.replace(":", "")[:16]


_ctx_cache: dict[str, ssl.SSLContext] = {}


def pinned_context(pem: str) -> ssl.SSLContext:
    """Trust exactly this certificate and nothing else. Hostname is not checked -- the
    pinned certificate IS the identity, and LAN addresses change.

    Cached per certificate: building one parses the PEM and seeds a fresh trust store, which
    costs more than the request it protects. A swarm fanning out onto a shared model was
    spending most of its wall clock here. An SSLContext is immutable once configured and safe
    to share across connections; the cache is keyed by the pinned PEM, so re-pairing to a new
    certificate gets a new context rather than silently reusing the old trust."""
    ctx = _ctx_cache.get(pem)
    if ctx is None:
        ctx = ssl.create_default_context(cadata=pem)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_REQUIRED
        if len(_ctx_cache) > 32:      # a LAN's worth of peers; never unbounded
            _ctx_cache.clear()
        _ctx_cache[pem] = ctx
    return ctx


# =======================================================================================
# Hooks into the control plane (set by main at import)
# =======================================================================================
_local_models: Callable[[], list[dict]] = lambda: []   # [{name, port, served_name, ready}]


def configure(local_models: Callable[[], list[dict]]) -> None:
    global _local_models
    _local_models = local_models


# =======================================================================================
# Served side: pending device grants, clients, tokens
# =======================================================================================
_pending: dict[str, dict] = {}        # device_code hash -> grant
_inflight: dict[str, int] = {}        # client_id -> requests in flight

# Per-request bookkeeping (how many, last seen) kept in memory and written back occasionally.
# It used to be saved on EVERY relayed request, and _save re-applies the file's ACL with a
# synchronous icacls call -- ~0.3s, under the global lock, on the gateway's event loop. That
# serialised federated inference completely: a fan-out of twelve took as long as running them
# one after another. None of this is security state; losing the last few seconds of a counter
# on a hard kill costs nothing.
_usage: dict[str, dict] = {}          # client_id -> {"requests": n, "last_seen": ts} unwritten
_usage_flushed = 0.0
USAGE_FLUSH_S = 30.0


def _note_request(cid: str) -> None:
    u = _usage.setdefault(cid, {"requests": 0, "last_seen": 0.0})
    u["requests"] += 1
    u["last_seen"] = time.time()


def _flush_usage(force: bool = False) -> None:
    global _usage_flushed
    if not _usage or (not force and time.time() - _usage_flushed < USAGE_FLUSH_S):
        return
    with _lock:
        st = _load()
        for cid, u in list(_usage.items()):
            c = st["clients"].get(cid)
            if c is not None:                      # revoked mid-flight: drop the counts
                c["requests"] = c.get("requests", 0) + u["requests"]
                c["last_seen"] = max(c.get("last_seen") or 0, u["last_seen"])
            _usage.pop(cid, None)
        _save(st)
        _usage_flushed = time.time()


def _user_code() -> str:
    alphabet = "BCDFGHJKLMNPQRSTVWXZ"  # no vowels: no accidental words, no 0/O 1/I confusion
    raw = "".join(secrets.choice(alphabet) for _ in range(8))
    return f"{raw[:4]}-{raw[4:]}"


def _issue(st: dict, client_id: str) -> dict:
    access, refresh = secrets.token_urlsafe(32), secrets.token_urlsafe(40)
    c = st["clients"][client_id]
    now = time.time()
    c["access"] = {h: exp for h, exp in (c.get("access") or {}).items() if exp > now}
    c["access"][_hash(access)] = now + ACCESS_TTL_S
    c["refresh"] = _hash(refresh)
    c["refresh_exp"] = now + REFRESH_TTL_S
    return {"access_token": access, "refresh_token": refresh, "token_type": "Bearer",
            "expires_in": ACCESS_TTL_S, "scope": SCOPE}


def _client_for_token(token: str) -> str | None:
    h, now = _hash(token), time.time()
    st = _load()
    for cid, c in st["clients"].items():
        if (c.get("access") or {}).get(h, 0) > now:
            return cid
    return None


# 100.64.0.0/10 is carrier-grade NAT space -- and what Tailscale assigns. Python does not count
# it as private, but a Tailscale peer is a WireGuard-authenticated device on YOUR tailnet, which
# is a better place for federation than a shared LAN.
_CGNAT = ipaddress.ip_network("100.64.0.0/10")


def _private(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or (ip.version == 4 and ip in _CGNAT)


def _oauth_error(code: str, desc: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": code, "error_description": desc}, status_code=status)


def _param(body: dict, form: dict, key: str) -> str:
    return str(body.get(key) or form.get(key) or "")


def build_gateway() -> FastAPI:
    """The LAN-facing app. Deliberately tiny: nothing of the control plane is mounted here."""
    gw = FastAPI(title="FreeSwarm federation gateway", docs_url=None, redoc_url=None, openapi_url=None)

    @gw.middleware("http")
    async def only_lan(request: Request, call_next):
        host = request.client.host if request.client else ""
        if not _private(host):
            return JSONResponse({"detail": "federation is limited to private networks"}, status_code=403)
        response = await call_next(request)
        # Every answer says which release and protocol produced it, so a client notices an
        # upgrade (or downgrade) on the other side without a separate call.
        response.headers["X-FreeToken-Version"] = V.APP_VERSION
        response.headers["X-FreeToken-Protocol"] = f"{V.FEDERATION_PROTOCOL};min={V.MIN_FEDERATION_PROTOCOL}"
        return response

    async def _body(request: Request) -> tuple[dict, dict]:
        ctype = request.headers.get("content-type", "")
        if "application/json" in ctype:
            try:
                return (await request.json()) or {}, {}
            except ValueError:
                return {}, {}
        form = await request.form()
        return {}, {k: str(v) for k, v in form.items()}

    @gw.get("/fed/info")
    async def info() -> dict:
        st = _load()
        pem, fp = ensure_cert(st["name"])
        return {**V.info(), "node_id": st["node_id"], "name": st["name"], "fingerprint": fp,
                "sharing": st["sharing"], "shared_models": len(st["shared_models"]),
                # How many requests at once this node will take from one client. The using
                # side queues to this number instead of being refused, so a swarm can fan out
                # onto a shared model the same way it does onto a local one. Absent on an
                # older peer -- the client then assumes the protocol-1 default.
                "max_concurrent": MAX_CONCURRENT_PER_CLIENT}

    @gw.post("/fed/oauth/device_authorization")
    async def device_authorization(request: Request):
        body, form = await _body(request)
        client_id = _param(body, form, "client_id")[:64]
        client_name = _param(body, form, "client_name")[:80] or "unknown computer"
        if not re.fullmatch(r"[a-f0-9]{32}", client_id):
            return _oauth_error("invalid_client", "client_id must be the requesting node id")
        if _param(body, form, "scope") not in ("", SCOPE):
            return _oauth_error("invalid_scope", f"only '{SCOPE}' is offered")
        theirs = int(_param(body, form, "protocol") or 1)
        ok, _why = V.compatible(theirs, int(_param(body, form, "min_protocol") or 0) or None)
        if not ok:
            # Worded for the REQUESTING computer's screen, naming both sides explicitly.
            me = _load()["name"]
            newer = "the requesting computer" if theirs < V.MIN_FEDERATION_PROTOCOL else me
            return _oauth_error("unsupported_version", (
                f"{me} runs FreeSwarm {V.APP_VERSION} (federation protocol {V.FEDERATION_PROTOCOL}, accepts "
                f"{V.MIN_FEDERATION_PROTOCOL}+); {client_name} speaks protocol {theirs}. Update FreeSwarm on {newer}."))
        addr = request.client.host if request.client else "?"
        now = time.time()
        for k in [k for k, g in _pending.items() if g["expires"] < now]:
            _pending.pop(k, None)
        if sum(1 for g in _pending.values() if g["addr"] == addr and g["status"] == "pending") >= 3:
            return _oauth_error("slow_down", "too many pending requests from this address", 429)
        device_code = secrets.token_urlsafe(32)
        grant = {"user_code": _user_code(), "client_id": client_id, "client_name": client_name,
                 "client_fp": _param(body, form, "client_fingerprint")[:120], "addr": addr, "scope": SCOPE,
                 "client_version": _param(body, form, "app_version")[:40] or None,
                 "created": now, "expires": now + DEVICE_TTL_S, "status": "pending", "last_poll": 0.0}
        _pending[_hash(device_code)] = grant
        st = _load()
        logger.warning("federation: pairing request from %s (%s) -- code %s", client_name, addr, grant["user_code"])
        return {"device_code": device_code, "user_code": grant["user_code"],
                "verification_uri": f"FreeSwarm on {st['name']} > Network > Requests to use this computer",
                "expires_in": DEVICE_TTL_S, "interval": POLL_INTERVAL_S}

    @gw.post("/fed/oauth/token")
    async def token(request: Request):
        body, form = await _body(request)
        grant_type = _param(body, form, "grant_type")
        client_id = _param(body, form, "client_id")
        st = _load()
        if grant_type == DEVICE_GRANT:
            g = _pending.get(_hash(_param(body, form, "device_code")))
            if g is None or g["client_id"] != client_id:
                return _oauth_error("invalid_grant", "unknown device code")
            now = time.time()
            if g["expires"] < now:
                return _oauth_error("expired_token", "the request expired -- start pairing again")
            if now - g["last_poll"] < POLL_INTERVAL_S - 0.5:
                g["last_poll"] = now
                return _oauth_error("slow_down", "poll no faster than the interval")
            g["last_poll"] = now
            if g["status"] == "denied":
                return _oauth_error("access_denied", "the operator of this computer declined the request")
            if g["status"] != "approved":
                return _oauth_error("authorization_pending", "waiting for approval on this computer's console")
            _pending.pop(_hash(_param(body, form, "device_code")), None)
            with _lock:
                st = _load()
                st["clients"][client_id] = {"name": g["client_name"], "addr": g["addr"], "scope": SCOPE,
                                            "created": now, "last_seen": now, "client_fp": g["client_fp"],
                                            "client_version": g.get("client_version")}
                tokens = _issue(st, client_id)
                _save(st)
            return tokens
        if grant_type == "refresh_token":
            h = _hash(_param(body, form, "refresh_token"))
            with _lock:
                st = _load()
                cid = next((k for k, c in st["clients"].items() if c.get("refresh") == h), None)
                if cid is None or cid != client_id or st["clients"][cid].get("refresh_exp", 0) < time.time():
                    return _oauth_error("invalid_grant", "refresh token unknown, revoked or expired")
                tokens = _issue(st, cid)          # rotation: the old refresh token dies here
                st["clients"][cid]["last_seen"] = time.time()
                _save(st)
            return tokens
        return _oauth_error("unsupported_grant_type", "use the device code or refresh token grant")

    @gw.post("/fed/oauth/revoke")
    async def revoke(request: Request):
        body, form = await _body(request)
        h = _hash(_param(body, form, "token"))
        with _lock:
            st = _load()
            for cid, c in list(st["clients"].items()):
                if c.get("refresh") == h or h in (c.get("access") or {}):
                    st["clients"].pop(cid)
            _save(st)
        return {}

    def _authorized(request: Request) -> str:
        auth = request.headers.get("authorization", "")
        tok = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        cid = _client_for_token(tok) if tok else None
        if cid is None:
            raise HTTPException(status_code=401, detail="invalid or expired token",
                                headers={"WWW-Authenticate": 'Bearer error="invalid_token"'})
        return cid

    def _shared_now() -> list[dict]:
        st = _load()
        if not st["sharing"]:
            return []
        wanted = set(st["shared_models"])
        return [m for m in _local_models() if m["name"] in wanted]

    @gw.get("/fed/models")
    async def models(request: Request) -> dict:
        _authorized(request)
        out = []
        async with httpx.AsyncClient(timeout=3.0) as c:
            for m in _shared_now():
                ctx = None
                try:
                    stats = (await c.get(f"http://127.0.0.1:{m['port']}/v1/stats")).json()
                    kv = stats.get("kv") or {}
                    model_ctx = (stats.get("model") or {}).get("ctx")
                    kv_tokens = (kv.get("total_pages") or 0) * (kv.get("page_size") or 1)
                    ctx = min(x for x in (model_ctx, kv_tokens) if x) if (model_ctx or kv_tokens) else None
                    tps = (stats.get("throughput") or {}).get("decode_tps")
                    active = (stats.get("requests") or {}).get("active")
                except Exception:  # noqa: BLE001
                    tps = active = None
                out.append({"name": m["name"], "ready": m["ready"], "context": ctx, "decode_tps": tps,
                            "active": active})
        return {"node": _load()["name"], "models": out, "max_concurrent": MAX_CONCURRENT_PER_CLIENT}

    @gw.post("/fed/v1/{path:path}")
    async def infer(path: str, request: Request):
        cid = _authorized(request)
        if path not in ALLOWED_PATHS:
            raise HTTPException(status_code=404, detail=f"/v1/{path} is not shared")
        try:
            payload = await request.json()
        except ValueError:
            raise HTTPException(status_code=400, detail="body must be JSON") from None
        name = str(payload.get("model") or "")
        m = next((x for x in _shared_now() if x["name"] == name), None)
        if m is None or not m["ready"]:
            raise HTTPException(status_code=404, detail=f"{name!r} is not shared by this computer (or not loaded)")
        if _inflight.get(cid, 0) >= MAX_CONCURRENT_PER_CLIENT:
            raise HTTPException(status_code=429, detail="too many concurrent requests from this client")
        payload["model"] = m["served_name"] or name
        url = f"http://127.0.0.1:{m['port']}/v1/{path}"
        _note_request(cid)
        await asyncio.to_thread(_flush_usage)
        _inflight[cid] = _inflight.get(cid, 0) + 1
        if payload.get("stream"):
            async def relay():
                try:
                    async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5.0)) as c:
                        async with c.stream("POST", url, json=payload) as r:
                            async for chunk in r.aiter_raw():
                                yield chunk
                finally:
                    _inflight[cid] = max(0, _inflight.get(cid, 1) - 1)
            return StreamingResponse(relay(), media_type="text/event-stream")
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=5.0)) as c:
                r = await c.post(url, json=payload)
            return JSONResponse(r.json(), status_code=r.status_code)
        finally:
            _inflight[cid] = max(0, _inflight.get(cid, 1) - 1)

    return gw


# =======================================================================================
# Gateway server + beacon (served side) and discovery listener (using side)
# =======================================================================================
class _Gateway:
    def __init__(self) -> None:
        self.server = None
        self.thread: threading.Thread | None = None
        self.error: str | None = None

    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def start(self) -> None:
        if self.running():
            return
        import uvicorn

        st = _load()
        ensure_cert(st["name"])
        config = uvicorn.Config(build_gateway(), host="0.0.0.0", port=GATEWAY_PORT, log_level="warning",
                                ssl_certfile=str(CERT_PATH), ssl_keyfile=str(KEY_PATH), lifespan="off",
                                access_log=False)
        self.server = uvicorn.Server(config)
        self.error = None

        def run() -> None:
            try:
                # Its own thread and loop: uvicorn only installs signal handlers on the main
                # thread, so the control plane's own Ctrl+C handling is untouched.
                self.server.run()
            except (OSError, SystemExit) as exc:
                self.error = f"gateway could not start on port {GATEWAY_PORT}: {exc}"
                logger.error(self.error)

        self.thread = threading.Thread(target=run, name="federation-gateway", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=8)
        self.server = self.thread = None


gateway = _Gateway()
_discovered: dict[str, dict] = {}
# Another computer announcing THIS computer's node id: ui/backend/auth/federation.json was
# copied between the two. Keyed by the clone's certificate fingerprint, which is what actually
# distinguishes them. Never a peer -- a node id must be unique for pairing to mean anything.
_clones: dict[str, dict] = {}
_stop = threading.Event()


def _local_addresses() -> list[str]:
    addrs = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addrs.add(info[4][0])
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        addrs.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(a for a in addrs if not a.startswith("127.") and _private(a))


def _beacon_loop() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    while not _stop.wait(BEACON_S):
        st = _load()
        if not st["sharing"] or not gateway.running():
            continue
        _, fp = ensure_cert(st["name"])
        msg = json.dumps({"freetoken": PROTOCOL, "min_protocol": V.MIN_FEDERATION_PROTOCOL,
                          "app_version": V.APP_VERSION, "node_id": st["node_id"], "name": st["name"],
                          "port": GATEWAY_PORT, "fingerprint": fp}).encode()
        for target in {"255.255.255.255", *[a.rsplit(".", 1)[0] + ".255" for a in _local_addresses()]}:
            try:
                sock.sendto(msg, (target, DISCOVERY_PORT))
            except OSError:
                pass


def _listen_loop() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", DISCOVERY_PORT))
    except OSError as exc:
        logger.error("federation discovery: cannot listen on udp/%d: %s", DISCOVERY_PORT, exc)
        return
    sock.settimeout(1.0)
    st0 = _load()
    me = st0["node_id"]
    _, my_fp = ensure_cert(st0["name"])
    while not _stop.is_set():
        try:
            data, (addr, _) = sock.recvfrom(2048)
        except socket.timeout:
            continue
        except OSError:
            continue
        if not _private(addr):
            continue
        try:
            msg = json.loads(data.decode())
        except ValueError:
            continue
        # Any FreeSwarm beacon is listed -- an incompatible one too, marked, so an out-of-date
        # computer shows up as "update needed" instead of silently never appearing.
        if not isinstance(msg.get("freetoken"), int):
            continue
        their_fp = str(msg.get("fingerprint") or "")[:120]
        if msg.get("node_id") == me:
            # Our own beacon coming back to us -- unless the certificate differs, in which case
            # it is a second computer running a copy of our identity file. Silently ignoring it
            # would leave that computer invisible here AND unpairable, with no hint why.
            if their_fp and their_fp != my_fp:
                _clones[their_fp] = {"name": str(msg.get("name"))[:80], "address": addr,
                                     "fingerprint": their_fp, "last_seen": time.time()}
            continue
        nid = str(msg.get("node_id"))[:64]
        ok, why = V.compatible(msg.get("freetoken"), msg.get("min_protocol"))
        _discovered[nid] = {"node_id": nid, "name": str(msg.get("name"))[:80], "address": addr,
                            "port": int(msg.get("port") or GATEWAY_PORT),
                            "fingerprint": str(msg.get("fingerprint"))[:120], "last_seen": time.time(),
                            "app_version": str(msg.get("app_version") or "")[:40] or None,
                            "protocol": msg.get("freetoken"), "compatible": ok,
                            "version_note": why if not ok else V.describe_difference(msg.get("app_version"))}


# =======================================================================================
# Using side: pairing, token refresh, remote model sync, routing
# =======================================================================================
_pairings: dict[str, dict] = {}       # pairing id -> progress (in memory; the tokens land in state)
_remote_models: dict[str, dict] = {}  # node_id -> {"models": [...], "status": ..., "checked": ts}

# The peer caps concurrent requests per client, and every request this computer makes to a
# given peer shares one client id. Holding a semaphore of exactly that size means a fan-out
# (a swarm putting six agents on one shared model) QUEUES here instead of collecting 429s
# there -- the cap becomes backpressure, which is what makes a remote model usable like a
# local one. Sized from the peer's own answer; recreated when that number changes.
_peer_slots: dict[str, tuple[int, asyncio.Semaphore]] = {}


def _slots(node_id: str) -> asyncio.Semaphore:
    # A peer that does not advertise a cap is running a build from before it was advertised,
    # where the cap was 4. Falling back to THIS build's number would overrun that peer and
    # earn 429s -- the failure the queue exists to prevent.
    want = int((_remote_models.get(node_id) or {}).get("max_concurrent") or LEGACY_MAX_CONCURRENT)
    have = _peer_slots.get(node_id)
    if have is None or have[0] != want:
        _peer_slots[node_id] = (want, asyncio.Semaphore(want))
    return _peer_slots[node_id][1]


def _peer_client(peer: dict, timeout: float = 30.0) -> httpx.AsyncClient:
    """A one-shot client, for pairing and teardown. Use `_pooled` on the hot paths."""
    return httpx.AsyncClient(base_url=f"https://{peer['address']}:{peer['port']}",
                             verify=pinned_context(peer["pem"]), timeout=timeout)


# One keep-alive pool per peer, because a fresh client means a fresh TLS handshake, and the
# handshakes serialise on the peer's single gateway loop: a fan-out of twelve spent more time
# shaking hands than generating. Keyed on what identifies the connection -- a peer that moves
# address or presents a new pinned certificate gets a new pool rather than a stale one.
_peer_clients: dict[str, tuple[tuple, httpx.AsyncClient]] = {}


def _pooled(node_id: str, peer: dict) -> httpx.AsyncClient:
    key = (peer["address"], peer["port"], peer["pem"])
    have = _peer_clients.get(node_id)
    if have is not None:
        if have[0] == key:
            return have[1]
        asyncio.create_task(have[1].aclose())
    cap = MAX_CONCURRENT_PER_CLIENT + 4          # slots, plus room for syncs and refreshes
    c = httpx.AsyncClient(base_url=f"https://{peer['address']}:{peer['port']}",
                          verify=pinned_context(peer["pem"]),
                          timeout=httpx.Timeout(1800.0, connect=8.0),
                          limits=httpx.Limits(max_connections=cap, max_keepalive_connections=cap))
    _peer_clients[node_id] = (key, c)
    return c


def _drop_pool(node_id: str) -> None:
    have = _peer_clients.pop(node_id, None)
    if have is not None:
        asyncio.create_task(have[1].aclose())


async def _fetch_cert(address: str, port: int) -> str:
    """The certificate the peer presents, for trust-on-first-use pinning."""
    return await asyncio.to_thread(ssl.get_server_certificate, (address, port), timeout=6)


def _same_node_id_detail(info: dict, address: str, fp: str, my_fp: str, st: dict) -> str:
    """Same node id as this computer: either it really is this computer, or -- far more often --
    ui/backend/auth/federation.json was copied from one machine to the other, which carries the
    node id with it."""
    if fp == my_fp:
        return f"that is this computer ({address} is one of its own addresses)"
    return (f"{info.get('name') or address} announces the same node id as this computer "
            f"({st['node_id'][:8]}...) but a different certificate, so it is a different machine "
            f"running a copy of this one's identity -- ui/backend/auth/federation.json was copied "
            f"between them. Two computers cannot federate while they share a node id: neither can "
            f"discover the other, and pairing cannot complete. On ONE of them, stop FreeSwarm, "
            f"delete ui/backend/auth/federation.json, and start it again -- it writes a fresh node "
            f"id and name (any pairings that computer had must be made again).")


async def connect(address: str, port: int = GATEWAY_PORT, expect_fp: str | None = None) -> dict:
    """Start pairing with another computer: pin its certificate and ask it for a device code."""
    st = _load()
    my_pem, my_fp = ensure_cert(st["name"])
    try:
        pem = await _fetch_cert(address, port)
    except (OSError, ssl.SSLError) as exc:
        raise HTTPException(status_code=502, detail=f"cannot reach https://{address}:{port}: {exc}") from None
    fp = fingerprint(pem)
    if expect_fp and expect_fp != fp:
        raise HTTPException(status_code=409, detail=(
            f"certificate fingerprint mismatch: the beacon announced {short_fp(expect_fp)}... but "
            f"{address} presented {short_fp(fp)}... -- refusing (possible impersonation)"))
    peer = {"address": address, "port": port, "pem": pem}
    try:
        async with _peer_client(peer, 10.0) as c:
            info = (await c.get("/fed/info")).json()
            ok, why = V.compatible(info.get("protocol"), info.get("min_protocol"))
            if not ok:
                raise HTTPException(status_code=409, detail=(
                    f"{info.get('name', address)} runs FreeSwarm {info.get('app_version') or '(unversioned)'}: {why}"))
            # Identity before the ask: anything that makes pairing impossible has to fail HERE,
            # or the other computer shows an approval prompt that this side will never poll --
            # the operator approves it and nothing happens, with nothing on screen saying why.
            if info.get("node_id") == st["node_id"]:
                raise HTTPException(status_code=400, detail=_same_node_id_detail(info, address, fp, my_fp, st))
            r = await c.post("/fed/oauth/device_authorization", json={
                "client_id": st["node_id"], "client_name": st["name"], "scope": SCOPE,
                "client_fingerprint": my_fp, **V.info()})
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"pairing request failed: {exc}") from None
    if r.status_code != 200:
        try:
            body = r.json()
            msg = body.get("error_description") or body.get("detail") or r.text[:300]
        except ValueError:
            msg = r.text[:300]
        raise HTTPException(status_code=409 if "version" in r.text else 502,
                            detail=f"{info.get('name', address)} refused the pairing request: {msg}")
    grant = r.json()
    pid = secrets.token_hex(6)
    _pairings[pid] = {"id": pid, "node_id": info["node_id"], "name": info["name"], "address": address,
                      "app_version": info.get("app_version"),
                      "port": port, "fingerprint": fp, "user_code": grant["user_code"],
                      "expires": time.time() + int(grant.get("expires_in", DEVICE_TTL_S)),
                      "status": "waiting", "error": None}
    asyncio.create_task(_poll_pairing(pid, pem, grant["device_code"], int(grant.get("interval", POLL_INTERVAL_S))))
    return _pairings[pid]


async def _poll_pairing(pid: str, pem: str, device_code: str, interval: int) -> None:
    p = _pairings[pid]
    st = _load()
    peer = {"address": p["address"], "port": p["port"], "pem": pem}
    while time.time() < p["expires"] and p["status"] == "waiting":
        await asyncio.sleep(interval)
        try:
            async with _peer_client(peer, 10.0) as c:
                r = await c.post("/fed/oauth/token", json={"grant_type": DEVICE_GRANT, "device_code": device_code,
                                                           "client_id": st["node_id"]})
        except httpx.HTTPError as exc:
            p["error"] = f"unreachable: {exc}"
            continue
        p["error"] = None          # it answered; don't leave a stale failure on screen
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code == 200 and body.get("access_token"):
            with _lock:
                st = _load()
                st["peers"][p["node_id"]] = {
                    "node_id": p["node_id"], "name": p["name"], "address": p["address"], "port": p["port"],
                    "app_version": p.get("app_version"),
                    "pem": pem, "fingerprint": p["fingerprint"], "access_token": body["access_token"],
                    "access_exp": time.time() + int(body.get("expires_in", ACCESS_TTL_S)),
                    "refresh_token": body["refresh_token"], "connected_at": time.time(), "enabled": True}
                _save(st)
            p["status"] = "connected"
            asyncio.create_task(sync_peer(p["node_id"]))
            return
        err = body.get("error")
        if err == "slow_down":
            interval += 2
        elif err == "authorization_pending":
            continue
        else:
            p["status"] = "denied" if err == "access_denied" else "failed"
            p["error"] = body.get("error_description") or err or r.text[:200]
            return
    if p["status"] == "waiting":
        p["status"] = "expired"


async def _fresh_token(node_id: str) -> str | None:
    st = _load()
    peer = st["peers"].get(node_id)
    if not peer:
        return None
    if peer["access_exp"] - time.time() > 120:
        return peer["access_token"]
    async with _peer_client(peer, 10.0) as c:
        r = await c.post("/fed/oauth/token", json={"grant_type": "refresh_token", "client_id": st["node_id"],
                                                   "refresh_token": peer["refresh_token"]})
    if r.status_code != 200:
        _remote_models[node_id] = {"models": [], "status": "revoked",
                                   "error": r.json().get("error_description", r.text[:200]), "checked": time.time()}
        return None
    body = r.json()
    with _lock:
        st = _load()
        if node_id in st["peers"]:
            st["peers"][node_id].update(access_token=body["access_token"], refresh_token=body["refresh_token"],
                                        access_exp=time.time() + int(body.get("expires_in", ACCESS_TTL_S)))
            _save(st)
    return body["access_token"]


async def sync_peer(node_id: str) -> None:
    peer = _load()["peers"].get(node_id)
    if not peer or not peer.get("enabled", True):
        return
    # A peer that moved to a new address is followed via its beacon (same node id, same pin).
    seen = _discovered.get(node_id)
    if seen and (seen["address"], seen["port"]) != (peer["address"], peer["port"]):
        with _lock:
            st = _load()
            st["peers"][node_id].update(address=seen["address"], port=seen["port"])
            _save(st)
            peer = st["peers"][node_id]
    try:
        tok = await _fresh_token(node_id)
        if tok is None:
            return
        async with _peer_client(peer, 8.0) as c:
            r = await c.get("/fed/models", headers={"Authorization": f"Bearer {tok}"})
        # The other computer may have been upgraded since pairing: re-check on every sync.
        app_v = r.headers.get("x-freetoken-version")
        proto = r.headers.get("x-freetoken-protocol", "")
        try:
            p_now = int(proto.split(";")[0]) if proto else None
            p_min = int(proto.split("min=")[1]) if "min=" in proto else None
        except (ValueError, IndexError):
            p_now = p_min = None
        if app_v and app_v != peer.get("app_version"):
            with _lock:
                st = _load()
                if node_id in st["peers"]:
                    st["peers"][node_id]["app_version"] = app_v
                    _save(st)
        ok, why = V.compatible(p_now, p_min)
        if not ok:
            _remote_models[node_id] = {"models": [], "status": "incompatible version", "error": why,
                                       "checked": time.time()}
            return
        if r.status_code == 401:
            _remote_models[node_id] = {"models": [], "status": "revoked", "error": "access revoked by the other computer",
                                       "checked": time.time()}
            return
        r.raise_for_status()
        doc = r.json()
        _remote_models[node_id] = {"models": doc.get("models", []), "status": "ok", "error": None,
                                   "max_concurrent": doc.get("max_concurrent"), "checked": time.time()}
    except ssl.SSLError as exc:
        _remote_models[node_id] = {"models": [], "status": "certificate changed",
                                   "error": f"the pinned certificate no longer matches ({exc}); re-pair to trust a new one",
                                   "checked": time.time()}
    except (httpx.HTTPError, ValueError, OSError) as exc:
        # httpx wraps the TLS failure; the verify error survives only in the message chain.
        chain, e = [], exc
        while e is not None and len(chain) < 6:
            chain.append(e)
            e = e.__cause__ or e.__context__
        pinned_fail = any(isinstance(x, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(x)
                          for x in chain)
        status = "certificate changed" if pinned_fail else "unreachable"
        _remote_models[node_id] = {"models": [], "status": status, "error": str(exc)[:300], "checked": time.time()}


async def _sync_loop() -> None:
    while True:
        try:
            for nid in list(_load()["peers"]):
                await sync_peer(nid)
        except Exception:  # noqa: BLE001 -- a poller must never die
            logger.debug("federation sync failed", exc_info=True)
        await asyncio.sleep(15)


def remote_loaded() -> list[dict]:
    """Remote models in the same shape as EngineManager.loaded_models(), plus `remote`."""
    out = []
    st = _load()
    for nid, peer in st["peers"].items():
        rm = _remote_models.get(nid) or {}
        if rm.get("status") != "ok" or not peer.get("enabled", True):
            continue
        for m in rm.get("models", []):
            name = f"{m['name']}@{node_slug(peer['name'])}"
            out.append({"instance_id": f"remote:{nid}:{m['name']}", "model": name, "served_name": name,
                        "state": "running" if m.get("ready") else "starting", "port": None, "gpus": None,
                        "ready": bool(m.get("ready")), "context": m.get("context"),
                        "remote": {"node_id": nid, "node": peer["name"], "address": peer["address"],
                                   "decode_tps": m.get("decode_tps"), "active": m.get("active")}})
    return out


def route(model: str) -> tuple[str, str] | None:
    """(node_id, the model's name on that node) for `name@node`, or None."""
    if "@" not in (model or ""):
        return None
    for m in remote_loaded():
        if m["model"] == model:
            return m["remote"]["node_id"], model.rsplit("@", 1)[0]
    return None


async def relay(node_id: str, remote_name: str, path: str, payload: dict):
    st = _load()
    peer = st["peers"].get(node_id)
    tok = await _fresh_token(node_id)
    if peer is None or tok is None:
        raise HTTPException(status_code=502, detail="the other computer revoked access or is not paired")
    payload = {**payload, "model": remote_name}
    headers = {"Authorization": f"Bearer {tok}"}
    slots = _slots(node_id)
    c = _pooled(node_id, peer)
    if payload.get("stream"):
        async def gen():
            # Held for the whole generation, not just the request: the peer counts a stream
            # as in flight until its last chunk.
            async with slots:
                try:
                    async with c.stream("POST", f"/fed/v1/{path}", json=payload, headers=headers,
                                        timeout=httpx.Timeout(None, connect=8.0)) as r:
                        if r.status_code != 200:
                            body = (await r.aread()).decode("utf-8", "replace")
                            yield f'data: {{"error": {body!r}}}\n\n'.encode()
                            return
                        async for chunk in r.aiter_raw():
                            yield chunk
                except httpx.HTTPError as exc:
                    yield f'data: {{"error": "{peer["name"]} failed: {exc}"}}\n\n'.encode()
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    async with slots:
        try:
            r = await c.post(f"/fed/v1/{path}", json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"{peer['name']} unreachable: {exc}") from None
    return JSONResponse(r.json(), status_code=r.status_code)


# =======================================================================================
# Lifecycle
# =======================================================================================
_threads: list[threading.Thread] = []
_sync_task: asyncio.Task | None = None


def start() -> None:
    global _sync_task
    st = _load()
    ensure_cert(st["name"])
    _stop.clear()
    for target, name in ((_listen_loop, "federation-listen"), (_beacon_loop, "federation-beacon")):
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        _threads.append(t)
    if st["sharing"]:
        gateway.start()
    _sync_task = asyncio.get_event_loop().create_task(_sync_loop())


def stop() -> None:
    _stop.set()
    _flush_usage(force=True)      # the last few seconds of counters, on the way out
    gateway.stop()
    if _sync_task is not None:
        _sync_task.cancel()


# =======================================================================================
# Console API (loopback control plane, behind the operator's auth)
# =======================================================================================
router = APIRouter(prefix="/federation", tags=["federation"])


_fw_cache: tuple[float, dict] = (0.0, {})


def firewall_status() -> dict:
    """Whether the Windows Firewall rules from ui/federation-firewall.cmd exist, and whether
    each network is Private (the rules only apply there). Cached: it shells out."""
    global _fw_cache
    if time.time() - _fw_cache[0] < 60:
        return _fw_cache[1]
    import subprocess
    import sys

    out: dict = {"platform": sys.platform, "tcp_rule": None, "udp_rule": None, "networks": [],
                 "tcp_port": GATEWAY_PORT, "udp_port": DISCOVERY_PORT}
    if sys.platform == "win32":
        def rule(name: str) -> bool:
            r = subprocess.run(["netsh", "advfirewall", "firewall", "show", "rule", f"name={name}"],
                               capture_output=True, text=True, timeout=15, check=False)
            return r.returncode == 0 and "No rules match" not in r.stdout
        try:
            out["tcp_rule"] = rule(f"FreeToken federation gateway (TCP {GATEWAY_PORT})")
            out["udp_rule"] = rule(f"FreeToken federation discovery (UDP {DISCOVERY_PORT})")
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                "Get-NetConnectionProfile | ForEach-Object { $_.Name + '|' + $_.NetworkCategory }"],
                               capture_output=True, text=True, timeout=20, check=False)
            out["networks"] = [{"name": ln.split("|")[0], "category": ln.split("|")[-1]}
                               for ln in r.stdout.splitlines() if "|" in ln]
        except (OSError, subprocess.SubprocessError):
            pass
    _fw_cache = (time.time(), out)
    return out


def _overview() -> dict:
    st = _load()
    pem, fp = ensure_cert(st["name"])
    now = time.time()
    local = _local_models()
    peers = []
    for nid, p in st["peers"].items():
        rm = _remote_models.get(nid) or {}
        peers.append({"node_id": nid, "name": p["name"], "address": p["address"], "port": p["port"],
                      "fingerprint": p["fingerprint"], "connected_at": p.get("connected_at"),
                      "enabled": p.get("enabled", True), "status": rm.get("status", "checking"),
                      "app_version": p.get("app_version"),
                      "version_note": V.describe_difference(p.get("app_version")),
                      "error": rm.get("error"), "models": rm.get("models", []),
                      "slug": node_slug(p["name"])})
    return {
        "node": {"node_id": st["node_id"], "name": st["name"], "fingerprint": fp, "addresses": _local_addresses(),
                 "port": GATEWAY_PORT, "discovery_port": DISCOVERY_PORT, **V.info()},
        "sharing": {"enabled": st["sharing"], "gateway_running": gateway.running(), "error": gateway.error,
                    "shared_models": st["shared_models"],
                    "available_models": [{"name": m["name"], "ready": m["ready"]} for m in local]},
        "requests": [{"user_code": g["user_code"], "client_name": g["client_name"], "address": g["addr"],
                      "app_version": g.get("client_version"),
                      "client_fingerprint": g["client_fp"], "expires_in": int(g["expires"] - now)}
                     for g in _pending.values() if g["status"] == "pending" and g["expires"] > now],
        # Counts not yet flushed to disk are folded in here, so the console stays live even
        # though the file is only written every USAGE_FLUSH_S.
        "clients": [{"client_id": cid, "name": c["name"], "address": c["addr"], "created": c["created"],
                     "app_version": c.get("client_version"),
                     "last_seen": max(c.get("last_seen") or 0,
                                      (_usage.get(cid) or {}).get("last_seen", 0)) or None,
                     "requests": c.get("requests", 0) + (_usage.get(cid) or {}).get("requests", 0),
                     "scope": c.get("scope")}
                    for cid, c in st["clients"].items()],
        "discovered": [d for d in _discovered.values() if now - d["last_seen"] < PEER_STALE_S
                       and d["node_id"] not in st["peers"]],
        "peers": peers,
        "pairings": [p for p in _pairings.values() if p["status"] != "connected" or now - p["expires"] < 60],
        "firewall": firewall_status(),
        "identity_conflicts": [c for c in _clones.values() if now - c["last_seen"] < PEER_STALE_S],
    }


@router.get("")
async def overview() -> dict:
    return await asyncio.to_thread(_overview)


class SharingReq(BaseModel):
    enabled: bool | None = None
    shared_models: list[str] | None = None
    name: str | None = Field(None, min_length=1, max_length=60)


@router.put("/sharing")
async def set_sharing(req: SharingReq) -> dict:
    with _lock:
        st = _load()
        if req.shared_models is not None:
            st["shared_models"] = [m for m in req.shared_models if m][:50]
        if req.name:
            st["name"] = req.name.strip()
        if req.enabled is not None:
            st["sharing"] = req.enabled
        _save(st)
    if st["sharing"]:
        await asyncio.to_thread(gateway.start)
    else:
        await asyncio.to_thread(gateway.stop)
    return _overview()


class Decide(BaseModel):
    approve: bool


@router.post("/requests/{user_code}")
async def decide(user_code: str, req: Decide) -> dict:
    """Approve or deny a pairing request -- only possible from THIS computer's console."""
    g = next((g for g in _pending.values() if g["user_code"] == user_code.upper() and g["status"] == "pending"), None)
    if g is None:
        raise HTTPException(status_code=404, detail="no pending request with that code (it may have expired)")
    g["status"] = "approved" if req.approve else "denied"
    return _overview()


@router.delete("/clients/{client_id}")
async def revoke_client(client_id: str) -> dict:
    with _lock:
        st = _load()
        st["clients"].pop(client_id, None)
        _save(st)
    return _overview()


class ConnectReq(BaseModel):
    node_id: str | None = None
    address: str | None = Field(None, max_length=200)
    port: int = Field(GATEWAY_PORT, ge=1, le=65535)


@router.post("/peers")
async def connect_peer(req: ConnectReq) -> dict:
    if req.node_id:
        d = _discovered.get(req.node_id)
        if d is None:
            raise HTTPException(status_code=404, detail="that computer is no longer announcing itself")
        if d.get("compatible") is False:
            raise HTTPException(status_code=409, detail=f"{d['name']} runs FreeSwarm {d.get('app_version')}: {d['version_note']}")
        return await connect(d["address"], d["port"], d["fingerprint"])
    if not req.address:
        raise HTTPException(status_code=400, detail="give a discovered computer or an address")
    host = req.address.strip()
    if ":" in host and host.count(":") == 1:
        host, port = host.split(":")
        req.port = int(port)
    try:
        ip = socket.gethostbyname(host)
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"cannot resolve {host}: {exc}") from None
    if not _private(ip):
        raise HTTPException(status_code=400, detail="federation is limited to private network addresses")
    return await connect(ip, req.port)


@router.delete("/peers/{node_id}")
async def disconnect_peer(node_id: str) -> dict:
    st = _load()
    peer = st["peers"].get(node_id)
    if peer:
        try:  # tell the other side to revoke; forget locally either way
            async with _peer_client(peer, 5.0) as c:
                await c.post("/fed/oauth/revoke", json={"token": peer["refresh_token"]})
        except (httpx.HTTPError, OSError, ssl.SSLError):
            pass
        with _lock:
            st = _load()
            st["peers"].pop(node_id, None)
            _save(st)
    _remote_models.pop(node_id, None)
    _peer_slots.pop(node_id, None)
    _drop_pool(node_id)
    return _overview()


def status_for_models() -> dict[str, dict]:
    """{remote model name: remote info} for decorating lists in the console."""
    return {m["model"]: m["remote"] for m in remote_loaded()}

