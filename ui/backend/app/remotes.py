"""Remote FreeSwarm backends -- one console, several machines.

The intended deployment is a small fleet: this box serves the small/mid models, another
box (with an Ada or newer GPU) serves something this hardware cannot run, and agents want
a single OpenAI endpoint that reaches whichever machine holds the model they asked for.

A remote is just another control plane, so the only thing needed here is:

  * where it is (`url`),
  * how to authenticate to it (`token` -- an access token minted by *that* machine's
    `usercli token`), and
  * which models it serves (discovered from its `/v1/models`, cached briefly).

Routing is by model name. `resolve(model)` returns None when the local engine serves it,
or the remote that does. That keeps the fast path -- the common case of a locally served
model -- free of any extra network call.

Declared in `ui/backend/backends.json`:

    {"backends": [
      {"name": "ada", "url": "https://ada-box.lan:8000", "token": "eyJ...", "enabled": true}
    ]}

**Use https, or an SSH tunnel to loopback.** The token in this file is a bearer credential
for the other machine, and plain http puts it on the wire in clear text on every request.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger("freetoken.remotes")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "backends.json"

# Model lists change only when someone starts a different engine, so a short cache keeps
# routing off the network without going stale in a way anyone would notice.
_MODEL_CACHE_TTL_S = 20.0
_model_cache: dict[str, tuple[float, list[str]]] = {}


class RemoteBackend(BaseModel):
    name: str
    url: str
    token: str = ""
    enabled: bool = True
    # Optional static model list. Set it when the remote is reachable but slow to probe, or
    # when you want routing to work before it has ever answered.
    models: list[str] = Field(default_factory=list)

    @property
    def base(self) -> str:
        return self.url.rstrip("/")

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    @property
    def is_plaintext(self) -> bool:
        """http:// to something that is not loopback -- the token crosses the wire bare."""
        if not self.base.startswith("http://"):
            return False
        host = self.base.removeprefix("http://").split("/")[0].split(":")[0]
        return host not in {"127.0.0.1", "localhost", "::1"}


def load_backends() -> list[RemoteBackend]:
    if not CONFIG_PATH.is_file():
        return []
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"backends.json is invalid: {exc}") from exc
    out: list[RemoteBackend] = []
    for entry in raw.get("backends", []) if isinstance(raw, dict) else []:
        try:
            out.append(RemoteBackend(**entry))
        except Exception as exc:  # noqa: BLE001 - one bad entry must not hide the rest
            logger.error("skipping malformed backend %r: %s", entry, exc)
    return out


def write_example_config() -> Path:
    if CONFIG_PATH.is_file():
        return CONFIG_PATH
    CONFIG_PATH.write_text(
        json.dumps(
            {
                "_comment": (
                    "Remote FreeSwarm control planes. `token` is an access token minted on "
                    "THAT machine (python -m app.usercli token <user>). Prefer https:// or "
                    "an SSH tunnel to 127.0.0.1 -- over plain http the token is readable on "
                    "the wire."
                ),
                "backends": [
                    {
                        "name": "ada",
                        "url": "http://127.0.0.1:8001",
                        "token": "",
                        "enabled": False,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return CONFIG_PATH


async def remote_models(client: httpx.AsyncClient, backend: RemoteBackend) -> list[str]:
    """Model ids the remote currently serves. Falls back to its static list on failure."""
    cached = _model_cache.get(backend.name)
    now = time.monotonic()
    if cached and now - cached[0] < _MODEL_CACHE_TTL_S:
        return cached[1]
    try:
        r = await client.get(
            f"{backend.base}/v1/models", headers=backend.headers, timeout=8.0
        )
        if r.status_code == 200:
            ids = [m["id"] for m in r.json().get("data", []) if isinstance(m, dict) and m.get("id")]
            _model_cache[backend.name] = (now, ids)
            return ids
    except (httpx.HTTPError, ValueError, KeyError):
        logger.debug("model probe failed for %s", backend.name, exc_info=True)
    # Cache the fallback too, so an unreachable remote is not re-probed on every request.
    _model_cache[backend.name] = (now, backend.models)
    return backend.models


async def status(client: httpx.AsyncClient, backend: RemoteBackend) -> dict:
    """Health + model list for the Backends view."""
    doc: dict = {
        "name": backend.name,
        "url": backend.base,
        "enabled": backend.enabled,
        "insecure": backend.is_plaintext,
        "ok": False,
        "error": None,
        "models": [],
        "engine_state": None,
    }
    if not backend.enabled:
        return doc
    try:
        r = await client.get(f"{backend.base}/api/engine", headers=backend.headers, timeout=8.0)
        if r.status_code == 401:
            doc["error"] = "unauthorised -- the token is missing, wrong, or expired"
            return doc
        if r.status_code != 200:
            doc["error"] = f"HTTP {r.status_code}"
            return doc
        doc["engine_state"] = r.json().get("state")
        doc["ok"] = True
    except httpx.HTTPError as exc:
        doc["error"] = f"{type(exc).__name__}: {exc}"
        return doc
    doc["models"] = await remote_models(client, backend)
    return doc


async def resolve(
    client: httpx.AsyncClient, model: str, local_models: set[str]
) -> RemoteBackend | None:
    """Which backend serves `model`? None means 'this machine'.

    Local always wins: if the locally loaded engine serves the name, there is no reason to
    cross the network for it.
    """
    if not model or model in local_models:
        return None
    for backend in load_backends():
        if not backend.enabled:
            continue
        if model in backend.models:
            return backend
        if model in await remote_models(client, backend):
            return backend
    return None
