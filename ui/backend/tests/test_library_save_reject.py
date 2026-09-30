"""library_save's rejection message must be actionable (bug #12).

Ten of the eleven sightings of "a signal module must define signal(df, ...)" in the console
bug list were the same Qwen/Qwen3-0.6B agent re-submitting essentially the same module every
few minutes: it defined ``main()`` / ``detect(df)`` under ``kind='signal'`` and the old error
just repeated the rule ("returning one position per row") without telling the model WHAT its
code had -- so the retry looked identical.

These tests pin down that the rejection now:
  * names exactly which defs the code has and which one was expected;
  * says the module's ``kind`` chooses the required function;
  * carries a minimal ``def signal(df, ...) -> positions`` (or ``def detect(...)``) template
    the model can rewrite against.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from app import library as L


class _Req:
    """Stand-in for the ``SaveModule`` model (only the fields the guard reads)."""

    def __init__(self, name: str, kind: str, code: str) -> None:
        self.name = name
        self.kind = kind
        self.code = code


@pytest.fixture(autouse=True)
def _stub_project(monkeypatch):
    # The rejection fires before any DB/sandbox work, so a bare project object is enough.
    monkeypatch.setattr(L, "_project", lambda pid: {"id": pid})


def _reject(name: str, kind: str, code: str) -> str:
    with pytest.raises(HTTPException) as ei:
        asyncio.run(L.save_module("p", _Req(name, kind, code)))
    assert ei.value.status_code == 400
    return ei.value.detail


# ---- 'kind' vs top-level defs --------------------------------------------------------------

def test_signal_kind_but_only_detect_and_main_names_what_it_has():
    # The bug's exact shape: kind=signal, top-level defs are main() and detect().
    detail = _reject("qwen_gex_oi", "signal", (
        "import pandas as pd\n"
        "def main(): pass\n"
        "def detect(df):\n"
        "    return [0] * len(df)\n"))
    assert "signal(df, ...)" in detail
    assert "signal" in detail
    # Names the two defs the code actually has, so the retry knows what to change.
    assert "main()" in detail and "detect()" in detail
    # Says the KIND drives it (the fix in the bug: rename the entry point, or save as util).
    assert "'kind'" in detail or "kind" in detail
    assert "util" in detail   # the escape hatch suggestion
    # And gives a runnable template.
    assert "def signal(df" in detail


def test_regime_kind_but_signal_only_names_missing_detect_and_gives_template():
    detail = _reject("regime_x", "regime", (
        "import pandas as pd\n"
        "def signal(df):\n"
        "    return [0] * len(df)\n"))
    assert "detect(df, ...)" in detail
    assert "signal()" in detail
    assert "def detect(df" in detail
    assert "regime label per row" in detail.lower()


def test_util_kind_has_no_required_function():
    # kind=util imposes no def contract, so a random helper module is accepted -- past this
    # check, the smoke test runs (mocked away here). Just verify the guard does NOT reject.
    # We short-circuit before the smoke test by monkeypatching _smoke to raise a marker.
    async def _boom(project, req):
        raise RuntimeError("SMOKE_TEST_REACHED")

    from app import library as LL
    orig = LL._smoke
    LL._smoke = _boom
    try:
        with pytest.raises(RuntimeError, match="SMOKE_TEST_REACHED"):
            asyncio.run(LL.save_module("p", _Req(
                "helper_util", "util",
                "def add(a, b):\n    return a + b\n")))
    finally:
        LL._smoke = orig


def test_signal_defined_passes_the_contract_check():
    # Sanity: correctly defined signal(df, ...) sails past the contract check into the smoke.
    async def _boom(project, req):
        raise RuntimeError("SMOKE_TEST_REACHED")

    from app import library as LL
    orig = LL._smoke
    LL._smoke = _boom
    try:
        with pytest.raises(RuntimeError, match="SMOKE_TEST_REACHED"):
            asyncio.run(LL.save_module("p", _Req(
                "good_signal", "signal",
                "import pandas as pd\n"
                "def signal(df, **k):\n"
                "    return [0] * len(df)\n")))
    finally:
        LL._smoke = orig


# ---- naming --------------------------------------------------------------------------------

def test_bad_name_says_what_it_got():
    detail = _reject("Not-A-Name!", "signal", "def signal(df): pass\n")
    assert "lowercase python identifier" in detail
    assert "'Not-A-Name!'" in detail   # not just the rule -- the actual value


def test_reserved_name_ft_or_lib_is_rejected():
    for reserved in ("ft", "lib"):
        detail = _reject(reserved, "signal", "def signal(df): pass\n")
        assert reserved in detail


# ---- template ------------------------------------------------------------------------------

def test_template_helper_returns_matching_kind():
    sig = L._module_template("signal", "my_sig")
    reg = L._module_template("regime", "my_reg")
    assert "def signal(df" in sig and "positions" in sig.lower() or "position" in sig
    assert "def detect(df" in reg and "regime" in reg.lower()
    # Both are non-empty, importable-looking Python.
    assert "import pandas" in sig and "import pandas" in reg
