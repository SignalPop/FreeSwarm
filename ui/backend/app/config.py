"""Runtime configuration for the FreeSwarm control-plane server.

Everything is overridable by environment variable so the launcher script can point the
control plane at a different venv / model root without editing code. Defaults assume the
layout this repo ships with: a `.venv` at the repo root and models under `models/`.
"""

from __future__ import annotations

import os
from pathlib import Path

# ui/backend/app/config.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


class Settings:
    # --- process layout -------------------------------------------------------------
    repo_root: Path = REPO_ROOT
    venv_python: Path = _env_path(
        "FREESWARM_UI_PYTHON", REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    )

    # Directories scanned for local checkpoints. A model may only be launched if it
    # resolves inside one of these roots -- see catalog.resolve_model_path. This is the
    # boundary that keeps the (unauthenticated, loopback) start endpoint from turning
    # into "spawn a process pointed at any path on disk".
    model_roots: list[Path] = [
        p
        for p in (
            _env_path("FREESWARM_MODELS_DIR", REPO_ROOT / "models"),
            Path(
                os.getenv("HF_HOME")
                or (Path.home() / ".cache" / "huggingface")
            ).expanduser()
            / "hub",
        )
    ]

    # --- networking -----------------------------------------------------------------
    # Loopback by default. The engine ships no authentication of any kind, so exposing it
    # directly would publish an open inference endpoint; the control plane is the only
    # thing that may listen off-loopback, and only with a token set (see validate()).
    host: str = os.getenv("FREESWARM_UI_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port: int = _env_int("FREESWARM_UI_PORT", 8000)

    # TLS for a non-loopback bind. Bearer tokens are replayable credentials, so serving
    # them over plain HTTP on a shared network hands anyone who can sniff the wire a
    # working login. Both must be set to enable TLS.
    ssl_certfile: str = os.getenv("FREESWARM_UI_SSL_CERT", "").strip()
    ssl_keyfile: str = os.getenv("FREESWARM_UI_SSL_KEY", "").strip()

    # Escape hatch for "my LAN is a trusted lab segment". Off by default on purpose.
    allow_insecure_bind: bool = os.getenv(
        "FREESWARM_UI_ALLOW_INSECURE", ""
    ).strip().lower() in {"1", "true", "yes", "on"}

    engine_host: str = "127.0.0.1"
    engine_port: int = _env_int("FREETOKEN_ENGINE_PORT", 1919)

    # Python sandbox container (ui/sandbox). Loopback-published by run-sandbox.bat; the
    # container itself sits on an --internal docker network with no route out, so this is
    # the only way to reach it and the only way it can be reached.
    sandbox_url: str = os.getenv(
        "FREESWARM_SANDBOX_URL", "http://127.0.0.1:8200"
    ).strip().rstrip("/") or "http://127.0.0.1:8200"

    frontend_origins: list[str] = [
        o.strip()
        for o in os.getenv(
            "FREESWARM_UI_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        ).split(",")
        if o.strip()
    ]

    # --- engine defaults ------------------------------------------------------------
    # GPU 0 on this box is the display adapter (WDDM). The compute cards are 1 and 2.
    # Overridable; an empty string means "leave CUDA_VISIBLE_DEVICES alone".
    visible_devices: str = os.getenv("FREETOKEN_VISIBLE_DEVICES", "1,2")

    # Captured stdout/stderr lines kept for the Logs view.
    log_ring_size: int = _env_int("FREESWARM_UI_LOG_LINES", 4000)

    @property
    def engine_base_url(self) -> str:
        return f"http://{self.engine_host}:{self.engine_port}"

    @property
    def is_loopback(self) -> bool:
        return self.host in {"127.0.0.1", "::1", "localhost"}

    @property
    def tls_enabled(self) -> bool:
        return bool(self.ssl_certfile and self.ssl_keyfile)

    def validate(self) -> list[str]:
        """Refuse an unsafe bind. Returns advisory warnings; raises on a hard stop.

        The control plane can start and stop processes and serve a model with no rate
        limit, so exposing it off-loopback without a login is not a configuration choice
        worth supporting silently. Two interlocks:

        1. Non-loopback bind with no user accounts -> refuse outright.
        2. Non-loopback bind over plain HTTP -> refuse unless explicitly overridden,
           because bearer tokens are replayable and would cross the wire in clear.
        """
        from .auth import auth_enabled  # local import: auth imports settings

        warnings: list[str] = []
        if self.is_loopback:
            if not auth_enabled():
                warnings.append(
                    "No user accounts exist; the API is open to anything on this machine. "
                    "That is fine for a loopback bind. Run `python -m app.usercli add <name>` "
                    "to require a login."
                )
            return warnings

        if not auth_enabled():
            raise RuntimeError(
                f"Refusing to bind to {self.host}: no user accounts exist, so every "
                "endpoint -- including the one that starts and stops engine processes -- "
                "would be open to the network. Create one first:\n"
                "    .venv\\Scripts\\python -m app.usercli add <username>\n"
                "Or bind to 127.0.0.1 and reach it over an SSH tunnel instead."
            )

        if not self.tls_enabled:
            if not self.allow_insecure_bind:
                raise RuntimeError(
                    f"Refusing to bind to {self.host} over plain HTTP: OAuth bearer tokens "
                    "are replayable credentials and would be readable by anyone who can "
                    "capture traffic on that network.\n"
                    "Fix one of these:\n"
                    "  - set FREESWARM_UI_SSL_CERT / FREESWARM_UI_SSL_KEY, or\n"
                    "  - keep the bind on 127.0.0.1 and use an SSH tunnel "
                    "(ssh -L 8000:127.0.0.1:8000 user@host), or\n"
                    "  - set FREESWARM_UI_ALLOW_INSECURE=1 if this really is a trusted "
                    "isolated segment."
                )
            warnings.append(
                f"Bound to {self.host} over plain HTTP with FREESWARM_UI_ALLOW_INSECURE set. "
                "Access tokens cross the network in clear text."
            )
        return warnings


settings = Settings()
