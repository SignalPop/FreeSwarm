"""A transient federation sync failure must not retire the swarm agents mid-iteration.

Bug #8: DeepSeek-V4-Flash-0731@lambda999 iterations consistently ended after their fifth
tool call with status "no submission". Every ~5-8 minutes -- roughly one sync tick's worth of
flakes on a busy LAN peer -- ``sync_peer`` timed out on /fed/models, ``remote_loaded()`` blanked
the peer's model list, /api/engines dropped the model from ``loaded``, and the swarm runner's
supervisor set ``retired`` on every worker using it. The current iteration finished its round,
the top of the next round saw ``retired.is_set()``, and the worker returned silently -- all its
tool-call work lost, no candidate submitted, no error on the board (iterate() suppresses the
"iteration ended without submission" post when retired is set, on purpose so a legitimate
retirement does not spam #errors).

The fix keeps the last known model list on ``unreachable`` and serves it for REMOTE_GRACE_S,
so a single dropped sync tick no longer retires the agent. A permanent state (certificate
changed, revoked, incompatible version) is still fatal at once.
"""

from __future__ import annotations

import pytest

from app import federation


@pytest.fixture
def peer(monkeypatch):
    """One paired computer (`lambda999`) sharing two ready models. Sync state is per-test."""
    real_load = federation._load
    monkeypatch.setattr(federation, "_load", lambda: {
        **real_load(),
        "peers": {"n1": {"name": "lambda999", "address": "10.0.0.25", "port": 8443, "enabled": True}}})
    monkeypatch.setattr(federation, "_remote_models", {}, raising=False)
    yield


MODELS_OK = [{"name": "DeepSeek-V4-Flash-0731", "ready": True, "context": 131072},
             {"name": "Muse-Glimmer-30B-NVFP4", "ready": True, "context": 65536}]


def _served(peer_names=("DeepSeek-V4-Flash-0731@lambda999", "Muse-Glimmer-30B-NVFP4@lambda999")) -> set[str]:
    return {m["model"] for m in federation.remote_loaded()}


def test_ok_status_serves_models(peer):
    """Baseline: a healthy peer serves its models."""
    federation._remote_models["n1"] = {"status": "ok", "models": MODELS_OK, "last_ok": 100.0,
                                       "checked": 100.0}
    assert _served() == {"DeepSeek-V4-Flash-0731@lambda999", "Muse-Glimmer-30B-NVFP4@lambda999"}


def test_transient_unreachable_within_grace_still_serves(peer, monkeypatch):
    """Bug #8 root cause: one dropped sync tick set status='unreachable' and the swarm
    supervisor retired every agent on those models. With the grace period, remote_loaded()
    keeps serving the last known list for REMOTE_GRACE_S -- long enough for the next sync to
    recover the peer, and long enough for a swarm iteration to finish its work.
    """
    monkeypatch.setattr(federation.time, "time", lambda: 200.0)
    # Successful sync at t=180, sync at t=200 timed out but is still within the 60s grace.
    federation._remote_models["n1"] = {"status": "unreachable", "models": MODELS_OK,
                                       "last_ok": 180.0, "checked": 200.0,
                                       "error": "ConnectTimeout"}
    assert _served() == {"DeepSeek-V4-Flash-0731@lambda999", "Muse-Glimmer-30B-NVFP4@lambda999"}


def test_unreachable_after_grace_drops_models(peer, monkeypatch):
    """A peer really down for a while (past REMOTE_GRACE_S since last success) does stop
    appearing in remote_loaded(); the supervisor should then retire its agents. This ensures
    the grace is bounded, not permanent."""
    monkeypatch.setattr(federation.time, "time", lambda: 400.0)
    federation._remote_models["n1"] = {"status": "unreachable", "models": MODELS_OK,
                                       "last_ok": 180.0, "checked": 400.0,
                                       "error": "ConnectTimeout"}
    assert _served() == set()


def test_certificate_changed_never_graced(peer, monkeypatch):
    """A pinned certificate mismatch is not a transient network failure -- silently routing
    to a possibly impersonated peer is exactly what pinning prevents. No grace."""
    monkeypatch.setattr(federation.time, "time", lambda: 200.0)
    federation._remote_models["n1"] = {"status": "certificate changed", "models": MODELS_OK,
                                       "last_ok": 199.0, "checked": 200.0,
                                       "error": "the pinned certificate no longer matches"}
    assert _served() == set()


def test_revoked_never_graced(peer, monkeypatch):
    """Access revoked by the other computer means every request will fail with 401 anyway,
    and continuing to route wastes the agent's turn on a certainty."""
    monkeypatch.setattr(federation.time, "time", lambda: 200.0)
    federation._remote_models["n1"] = {"status": "revoked", "models": MODELS_OK,
                                       "last_ok": 199.0, "checked": 200.0,
                                       "error": "access revoked by the other computer"}
    assert _served() == set()


def test_incompatible_version_never_graced(peer, monkeypatch):
    """A protocol mismatch is not going to fix itself in the grace window."""
    monkeypatch.setattr(federation.time, "time", lambda: 200.0)
    federation._remote_models["n1"] = {"status": "incompatible version", "models": MODELS_OK,
                                       "last_ok": 199.0, "checked": 200.0,
                                       "error": "peer is too old"}
    assert _served() == set()


def test_unreachable_without_prior_ok_does_not_serve(peer, monkeypatch):
    """A peer that has never synced successfully (no last_ok) is not graced -- there is
    nothing to grace to. Prevents an incorrect first-boot behaviour."""
    monkeypatch.setattr(federation.time, "time", lambda: 100.0)
    federation._remote_models["n1"] = {"status": "unreachable", "models": [],
                                       "checked": 100.0, "error": "ConnectTimeout"}
    assert _served() == set()


def test_disabled_peer_never_served(peer, monkeypatch):
    """The operator disabling a peer is honoured immediately, even within the grace period."""
    monkeypatch.setattr(federation, "_load", lambda: {
        "peers": {"n1": {"name": "lambda999", "address": "10.0.0.25", "port": 8443, "enabled": False}}})
    monkeypatch.setattr(federation.time, "time", lambda: 200.0)
    federation._remote_models["n1"] = {"status": "ok", "models": MODELS_OK, "last_ok": 200.0,
                                       "checked": 200.0}
    assert _served() == set()


# ------------------------------------------------------------------------------------------
# sync_peer preserves the model list on transient failures so remote_loaded() can grace
# them. Two prior successful syncs, then one that fails: the models must remain.
# ------------------------------------------------------------------------------------------
class _RecordingSync:
    """Drives sync_peer() with monkeypatched HTTP + token so we can exercise the failure
    branches without a real peer or event loop."""

    def __init__(self, monkeypatch, node_id: str, peer_addr: dict):
        self.node_id = node_id
        self.peer_addr = peer_addr
        monkeypatch.setattr(federation, "_load", lambda: {"peers": {node_id: peer_addr}})

        async def _tok(_nid):
            return "tok"
        monkeypatch.setattr(federation, "_fresh_token", _tok)

    def install_response(self, monkeypatch, *, models=None, raise_exc=None):
        """Make the next /fed/models call return `models` (status 200) or raise `raise_exc`."""

        class _Resp:
            headers = {"x-freetoken-version": "1.0", "x-freetoken-protocol": "1;min=1"}
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"models": models or [], "max_concurrent": 8}

        class _Client:
            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *_):
                return False

            async def get(self_inner, path, headers=None):
                if raise_exc is not None:
                    raise raise_exc
                return _Resp()

        monkeypatch.setattr(federation, "_peer_client", lambda peer, timeout: _Client())


def test_sync_peer_preserves_models_on_transient_failure(monkeypatch):
    """The regression the fix defends: one httpx ConnectTimeout used to overwrite the models
    with []. After the fix, the previous list is preserved so remote_loaded()'s grace window
    can serve it for REMOTE_GRACE_S."""
    import asyncio
    import httpx

    monkeypatch.setattr(federation, "_remote_models", {}, raising=False)
    driver = _RecordingSync(monkeypatch, "n1",
                            {"name": "lambda999", "address": "10.0.0.25", "port": 8443,
                             "enabled": True, "pem": "", "app_version": "1.0"})
    # First sync succeeds and populates the list.
    monkeypatch.setattr(federation.time, "time", lambda: 100.0)
    driver.install_response(monkeypatch, models=MODELS_OK)
    asyncio.run(federation.sync_peer("n1"))
    rm = federation._remote_models["n1"]
    assert rm["status"] == "ok" and rm["last_ok"] == 100.0
    assert [m["name"] for m in rm["models"]] == ["DeepSeek-V4-Flash-0731", "Muse-Glimmer-30B-NVFP4"]

    # Second sync fails with a transient ConnectTimeout -- models must be KEPT.
    monkeypatch.setattr(federation.time, "time", lambda: 115.0)
    driver.install_response(monkeypatch, raise_exc=httpx.ConnectTimeout("timed out"))
    asyncio.run(federation.sync_peer("n1"))
    rm = federation._remote_models["n1"]
    assert rm["status"] == "unreachable"
    assert rm["last_ok"] == 100.0                          # the previous success is remembered
    assert [m["name"] for m in rm["models"]] == ["DeepSeek-V4-Flash-0731", "Muse-Glimmer-30B-NVFP4"]


def test_sync_peer_blanks_models_on_pinned_cert_failure(monkeypatch):
    """A pinned certificate verification failure is permanent (until the operator re-pairs),
    so ``models`` is blanked at once and the grace does not apply."""
    import asyncio
    import ssl

    monkeypatch.setattr(federation, "_remote_models", {}, raising=False)
    driver = _RecordingSync(monkeypatch, "n1",
                            {"name": "lambda999", "address": "10.0.0.25", "port": 8443,
                             "enabled": True, "pem": "", "app_version": "1.0"})
    monkeypatch.setattr(federation.time, "time", lambda: 100.0)
    driver.install_response(monkeypatch, models=MODELS_OK)
    asyncio.run(federation.sync_peer("n1"))
    assert federation._remote_models["n1"]["models"]

    monkeypatch.setattr(federation.time, "time", lambda: 115.0)
    driver.install_response(monkeypatch, raise_exc=ssl.SSLError("CERTIFICATE_VERIFY_FAILED"))
    asyncio.run(federation.sync_peer("n1"))
    rm = federation._remote_models["n1"]
    assert rm["status"] == "certificate changed"
    assert rm["models"] == []                              # no grace: refuse to route to it
