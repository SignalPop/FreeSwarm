"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _own_runner_lock(tmp_path, monkeypatch):
    """Each test gets its own swarm-runner instance lock: the live runner holds the real one
    (ui/backend/.swarm_runner.lock), and a test's main() must not be refused because of it."""
    monkeypatch.setenv("FREESWARM_RUNNER_LOCK", str(tmp_path / "runner.lock"))
