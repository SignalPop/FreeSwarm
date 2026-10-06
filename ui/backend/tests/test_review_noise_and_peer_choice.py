"""#409 / #410: a small reviewing model quoted ft's own "[ft] the frame had no 'session' column --
added as ..." note -- from a run_python that worked -- and filed it twice, as "empty tables" and
"constant columns". #70 (10-05): every Muse-Glimmer audit went to the first loaded peer, Qwen3.6,
offloaded and serving five agents at ~6 tok/s; one audit ran 1,232 s into its token limit."""

from __future__ import annotations

import json
import time

import pytest

from app import monitor as M

RESULT = json.dumps({
    "ok": True, "stdout": "both_any_gex_109: active_days=90, sharpe=2.141\n  long_pct=N/A, short_pct=N/A\n",
    "stderr": "[ft] the frame had no 'session' column -- added as the New York session date of 't' (what ft.clock gives)\n",
    "artifacts": [], "duration_s": 7.59, "experiments_left": 2})


def _record(result: str) -> tuple[dict, dict]:
    now = time.time()
    e = {"kind": "tool", "at": now, "name": "run_python", "args": {"code": "print(1)"}, "ok": True, "seconds": 7.6,
         "result": result}
    r = {"id": "r1", "mode": "explore", "status": "done", "objective": {"id": "o1", "title": "o"},
         "started_at": now - 100, "ended_at": now - 1, "pending": None, "timeline": [e]}
    return {"agent": "m1", "model": "m1", "project_id": "p1", "updated_at": now, "records": [r]}, r


def test_an_ft_note_is_not_review_evidence():
    a, r = _record(RESULT)
    quote = "the session column having no 'session' column -- added as the New York session date of 't'"
    assert not M._platform_evidence(quote, a, r)
    # the platform's own output in the same result still counts
    assert M._platform_evidence("both_any_gex_109: active_days=90, sharpe=2.141", a, r)


@pytest.fixture
def R(monkeypatch):
    import swarm_runner as R

    monkeypatch.setattr(R, "_speed", {})
    monkeypatch.setattr(R, "_reply_s", {})
    monkeypatch.setattr(R, "_reply_at", {})
    monkeypatch.setattr(R, "context_for", lambda m: 131_072)
    monkeypatch.setattr(R, "max_output_for", lambda m: None)
    return R


MSGS = [{"role": "user", "content": "audit this " * 200}]


def test_a_peer_too_slow_to_answer_in_time_is_passed_over(R):
    R._speed["Qwen3.6-35B-A3B"] = 7095 / 1232.3          # measured 10-05 16:30
    assert R._pick_peer("Muse-Glimmer-30B-NVFP4", ["Qwen3.6-35B-A3B"], MSGS, 3000) == "Muse-Glimmer-30B-NVFP4"
    # ...and not to a paid peer instead: the old rule never paid while a free one was loaded
    paid = "qwen/qwen3.8-27b@groq"
    assert R.is_external(paid)
    assert R._pick_peer("Muse-Glimmer-30B-NVFP4", ["Qwen3.6-35B-A3B", paid], MSGS, 3000) == "Muse-Glimmer-30B-NVFP4"
    assert R._pick_peer("Muse-Glimmer-30B-NVFP4", [paid], MSGS, 3000) == paid
    R._speed["Qwen3.6-35B-A3B"] = 60.0
    assert R._pick_peer("Muse-Glimmer-30B-NVFP4", ["Qwen3.6-35B-A3B", paid], MSGS, 3000) == "Qwen3.6-35B-A3B"


def test_the_fastest_peer_wins_but_never_a_smaller_tier(R):
    R._speed.update({"Qwen/Qwen3-0.6B": 300.0, "gpt-oss-120b": 40.0, "Qwen3.6-35B-A3B": 20.0})
    peers = ["Qwen3.6-35B-A3B", "Qwen/Qwen3-0.6B", "gpt-oss-120b"]
    assert R._pick_peer("Muse-Glimmer-30B-NVFP4", peers, MSGS, 3000) == "gpt-oss-120b"
    # a peer not measured yet is still a peer
    assert R._pick_peer("Muse-Glimmer-30B-NVFP4", ["gemma-4-26B-A4B-it"], MSGS, 3000) == "gemma-4-26B-A4B-it"
    assert R._pick_peer("Muse-Glimmer-30B-NVFP4", ["Qwen/Qwen3-0.6B"], MSGS, 3000) == "Muse-Glimmer-30B-NVFP4"
