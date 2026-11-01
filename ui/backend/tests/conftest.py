"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _own_runner_lock(tmp_path, monkeypatch):
    """Each test gets its own swarm-runner instance lock: the live runner holds the real one
    (ui/backend/.swarm_runner.lock), and a test's main() must not be refused because of it."""
    monkeypatch.setenv("FREESWARM_RUNNER_LOCK", str(tmp_path / "runner.lock"))


@pytest.fixture(autouse=True)
def _own_checkpoints_and_steering(tmp_path, monkeypatch):
    """Iteration checkpoints (swarm_runner) and steering messages (app/steering.py) go to this
    test's own files, never the live ones next to the code."""
    monkeypatch.setenv("FREESWARM_CHECKPOINT_DB", str(tmp_path / "checkpoints.sqlite3"))
    from app import steering

    monkeypatch.setattr(steering, "DB_PATH", tmp_path / "steering.sqlite3")
    monkeypatch.setattr(steering, "_conn", None)
