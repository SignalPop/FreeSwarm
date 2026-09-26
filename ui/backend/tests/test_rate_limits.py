"""External-model rate limits: pace on the provider's headers, wait out a 429, fail fast on a
daily quota -- instead of handing every collision back to the caller as a refusal."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi import HTTPException

from app import external as X

NAME = "qwen/qwen3.8-27b@groq"


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    X._rates.clear()
    X._throttles.clear()
    X._refusals.clear()
    monkeypatch.setattr(X, "_prepare", lambda name, payload, purpose="chat": ("groq", "m", {**payload, "model": "m"}))
    monkeypatch.setattr(X, "_headers", lambda provider: {})
    monkeypatch.setattr(X, "_record", lambda *a, **k: 0.0)
    slept: list[float] = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(X.asyncio, "sleep", fake_sleep)
    yield slept


def _client(responses: list[httpx.Response]) -> tuple[httpx.AsyncClient, list]:
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return responses.pop(0)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


OK = {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}


def test_a_429_is_waited_out_and_resent(clean):
    client, seen = _client([
        httpx.Response(429, headers={"retry-after": "2"}, json={"error": {"message": "Rate limit reached ... "
                                                                                   "Please try again in 2s"}}),
        httpx.Response(200, json=OK),
    ])
    r = asyncio.run(X.complete(client, NAME, {"messages": [{"role": "user", "content": "x"}], "max_tokens": 10}))
    assert r.status_code == 200 and len(seen) == 2
    assert any(1.9 < s < 3.5 for s in clean)                    # waited ~Retry-After before resending
    assert X.refusals(NAME)["count"] == 0                        # a wait, not a refusal
    assert X.throttles(NAME)["count"] == 1


def test_try_again_in_message_is_understood_without_headers():
    assert X._retry_wait({}, "Rate limit reached ... Please try again in 2.33208s.") == pytest.approx(2.33208)
    assert X._retry_wait({}, "try again in 7m12.5s") == pytest.approx(432.5)
    assert X._retry_wait({"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "1m3s"}, "") == 63.0


def test_a_request_that_does_not_fit_waits_for_the_refill(clean):
    client, seen = _client([httpx.Response(200, json=OK), httpx.Response(200, json=OK)])
    # The first response says 100 tokens are left of the minute, refilling in 5 s.
    client_first = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(
        200, json=OK, headers={"x-ratelimit-limit-tokens": "250000", "x-ratelimit-remaining-tokens": "100",
                               "x-ratelimit-reset-tokens": "5s"})))
    asyncio.run(X.complete(client_first, NAME, {"messages": [], "max_tokens": 10}))
    clean.clear()
    big = {"messages": [{"role": "user", "content": "x" * 70_000}], "max_tokens": 1000}   # ~21k tokens
    asyncio.run(X.complete(client, NAME, big))
    assert clean and 4.5 < clean[0] < 5.5                       # paced until the window refilled
    assert len(seen) == 1                                       # and never sent early to fail


def test_a_daily_quota_blocks_and_fails_fast(clean):
    client, seen = _client([
        httpx.Response(429, json={"error": {"message": "Rate limit reached for tokens per day (TPD). "
                                                       "Please try again in 7m12s."}}),
    ])
    r = asyncio.run(X.complete(client, NAME, {"messages": [], "max_tokens": 10}))
    assert r.status_code == 429 and len(seen) == 1              # not resent: a 7-minute wait is a quota
    with pytest.raises(HTTPException) as e:                     # the next call does not even go out
        asyncio.run(X.complete(client, NAME, {"messages": [], "max_tokens": 10}))
    assert e.value.status_code == 429 and "try again in" in e.value.detail
    assert len(seen) == 1


def test_duration_parser():
    assert X._duration("250ms") == pytest.approx(0.25)
    assert X._duration("2m59.56s") == pytest.approx(179.56)
    assert X._duration("12") == 12.0
    assert X._duration(None) is None
