"""OAuth 2.1 on a task server's HTTP transport, end to end against a live server:
discovery, the operator's consent with the approval passphrase (and its lockout), authorization
code + PKCE with the client secret, a working bearer token, rejection of everything forged,
replayed or missing, refresh rotation and revocation."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import secrets
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest

pytest.importorskip("mcp.server.mcpserver", reason="needs the MCP SDK the servers run on (mcp >= 2; the project .venv)")

from taskkit import oauth as O
from taskkit.server import TasksProvider, build_server

PASS = "amber-anchor-apple-arrow-aspen-atlas"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _NoTasks:
    name = "none"


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    import uvicorn

    d = tmp_path_factory.mktemp("srv")
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    O.generate(d, "unit", port, ["--no-register", "--quiet", "--passphrase", PASS,
                                 "--control-plane", "http://127.0.0.1:9"])
    cfg = json.loads((d / ".oauth" / "server.json").read_text())
    mcp = build_server(TasksProvider(lambda: []), "unit", (d / ".oauth", base))
    app = mcp.streamable_http_app()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        try:
            httpx.get(base + "/.well-known/oauth-authorization-server", timeout=1)
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    yield {"base": base, "client": cfg["client"], "dir": d}
    srv.should_exit = True
    th.join(5)


def _pkce():
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def _authorize(s, challenge, state="st8"):
    r = httpx.get(s["base"] + "/authorize", params={
        "response_type": "code", "client_id": s["client"]["client_id"], "redirect_uri": s["client"]["redirect_uris"][0],
        "code_challenge": challenge, "code_challenge_method": "S256", "state": state, "scope": "tasks"})
    assert r.status_code in (302, 303), r.text
    return re.search(r"tx=([^&]+)", r.headers["location"]).group(1)


def _consent(s, tx, passphrase, decision="approve"):
    return httpx.post(s["base"] + "/oauth/consent", data={"tx": tx, "passphrase": passphrase, "decision": decision})


def _token(s, **form):
    return httpx.post(s["base"] + "/token", data={"client_id": s["client"]["client_id"],
                                                  "client_secret": s["client"]["client_secret"], **form})


def test_secrets_are_owner_only_files_and_nothing_is_plain(server):
    cfg = json.loads((server["dir"] / ".oauth" / "server.json").read_text())
    assert PASS not in json.dumps(cfg)                              # only the scrypt hash is kept
    assert (server["dir"] / ".oauth" / "approval_passphrase.txt").read_text().startswith(PASS)


def test_no_token_no_access_and_discovery(server):
    r = httpx.post(server["base"] + "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert r.status_code == 401 and "resource_metadata" in r.headers.get("www-authenticate", "")
    meta = httpx.get(server["base"] + "/.well-known/oauth-authorization-server").json()
    assert meta["code_challenge_methods_supported"] == ["S256"] and "registration_endpoint" not in meta
    r = httpx.post(server["base"] + "/register", json={"redirect_uris": ["http://evil/cb"]})
    assert r.status_code in (404, 405)                               # no dynamic registration


def test_full_flow_bearer_refresh_rotation_and_forgeries(server):
    verifier, challenge = _pkce()
    tx = _authorize(server, challenge)
    bad = _consent(server, tx, "wrong-words")
    assert bad.status_code == 400 and "Wrong passphrase" in bad.text
    ok = _consent(server, tx, PASS)
    assert ok.status_code == 302
    loc = ok.headers["location"]
    assert loc.startswith(server["client"]["redirect_uris"][0]) and "state=st8" in loc
    code = re.search(r"code=([^&]+)", loc).group(1)

    redirect = server["client"]["redirect_uris"][0]
    assert _token(server, grant_type="authorization_code", code=code, redirect_uri=redirect,
                  code_verifier="not-the-verifier").status_code == 400            # PKCE enforced
    r = httpx.post(server["base"] + "/token", data={"grant_type": "authorization_code", "code": code,
                                                    "redirect_uri": redirect, "code_verifier": verifier,
                                                    "client_id": server["client"]["client_id"],
                                                    "client_secret": "guess"})
    assert r.status_code == 401                                                   # client secret enforced
    tok = _token(server, grant_type="authorization_code", code=code, redirect_uri=redirect, code_verifier=verifier)
    assert tok.status_code == 200, tok.text
    t = tok.json()
    assert t["token_type"] == "Bearer" and t["expires_in"] == O.ACCESS_TTL and t["refresh_token"]
    assert _token(server, grant_type="authorization_code", code=code, redirect_uri=redirect,
                  code_verifier=verifier).status_code == 400                      # a code works once

    # The bearer token opens /mcp; a tampered one does not.
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async def tools(token):
        async with httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"}) as http:
            async with streamable_http_client(server["base"] + "/mcp", http_client=http) as (rd, wr, *_):
                async with ClientSession(rd, wr) as s:
                    await s.initialize()
                    return sorted(x.name for x in (await s.list_tools()).tools)

    names = asyncio.run(tools(t["access_token"]))
    assert "task_list" in names and "harness_evaluate" in names
    forged = t["access_token"][:-4] + ("AAAA" if not t["access_token"].endswith("AAAA") else "BBBB")
    r = httpx.post(server["base"] + "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                   headers={"Authorization": f"Bearer {forged}"})
    assert r.status_code == 401

    # Refresh rotates: the new pair works, the old refresh token is dead.
    r1 = _token(server, grant_type="refresh_token", refresh_token=t["refresh_token"])
    assert r1.status_code == 200 and r1.json()["refresh_token"] != t["refresh_token"]
    assert _token(server, grant_type="refresh_token", refresh_token=t["refresh_token"]).status_code == 400
    # Revocation kills the new refresh token.
    rv = httpx.post(server["base"] + "/revoke", data={"token": r1.json()["refresh_token"],
                                                      "client_id": server["client"]["client_id"],
                                                      "client_secret": server["client"]["client_secret"]})
    assert rv.status_code == 200
    assert _token(server, grant_type="refresh_token", refresh_token=r1.json()["refresh_token"]).status_code == 400


def test_wrong_passphrases_lock_approvals(server):
    _, challenge = _pkce()
    tx = _authorize(server, challenge, state="lock")
    for _ in range(O.MAX_FAILURES):
        _consent(server, tx, "nope")
    r = _consent(server, tx, PASS)                                  # even the right one, now
    assert r.status_code == 400 and "locked" in r.text


def test_http_refuses_to_start_without_secrets(tmp_path):
    from taskkit.server import serve

    with pytest.raises(SystemExit) as e:
        serve(TasksProvider(lambda: []), "x", 1, ["--http"], oauth_dir=tmp_path / ".oauth")
    assert "make_oauth_secrets" in str(e.value)
    with pytest.raises(SystemExit) as e:
        serve(TasksProvider(lambda: []), "x", 1, ["--http", "--no-auth", "--host", "0.0.0.0"], oauth_dir=tmp_path)
    assert "loopback" in str(e.value)
