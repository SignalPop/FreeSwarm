"""Bug pass 2026-10-01: an engine stopped by a service restart is not a crash (bugs #352, #353,
#377 were all "forrtl ... window-CLOSE event" at restart times), and triage falls back to a loaded
local model when the configured one is not loaded (the banner "the configured model is not loaded"
left triage off since Qwen/Qwen3-0.6B went with the 12:57 restart)."""

from __future__ import annotations

from app import diagnose as D
from app import monitor as M

# The tail the engine left at the 15:32 restart (bug #377's evidence, ANSI codes kept).
RESTART_TAIL = [
    "\x1b[1m[2026-10-01|15:32:29|core|rank=0]\x1b[0m \x1b[32mINFO    \x1b[0m Decode batch, #running-req: 1, "
    "#token: 45964, token usage: 0.35, gen throughput (token/s): 34.50, #queue-req: 3",
    "forrtl: error (200): program aborting due to window-CLOSE event",
    "Image              PC                Routine            Line        Source             ",
    "KERNELBASE.dll     00007FF8CB75C213  Unknown               Unknown  Unknown",
    "\x1b[1m[2026-10-01|15:32:31|FrontendAPI]\x1b[0m \x1b[31mERROR   \x1b[0m Backend supervisor: backend worker "
    "freetoken-detokenizer-0 exited",
]

OOM_TAIL = [
    "Traceback (most recent call last):",
    '  File "x.py", line 3, in <module>',
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
    "\x1b[31mERROR   \x1b[0m Backend supervisor: backend worker freetoken-scheduler-0 exited",
]


def _engine(diag: dict, model: str = "Qwen3.6-35B-A3B") -> dict:
    return {"state": "error", "model_id": model, "started_at": 1.0, "error": diag.get("summary"),
            "diagnosis": {**diag, "exit_code": 2}, "gpus": "2", "port": 1921, "model_path": "x", "command": "c"}


def test_a_console_close_is_diagnosed_as_a_stop_not_a_crash():
    d = D.diagnose(RESTART_TAIL, 2)
    assert d.stopped and d.as_dict()["stopped"] is True
    assert "stopped from outside" in d.summary and "detokenizer" not in d.summary
    for event in ("control-C", "control-BREAK", "logoff", "shutdown"):
        assert D.stopped_from_outside([f"forrtl: error (200): program aborting due to {event} event"])
    assert D.stopped_from_outside(["Traceback (most recent call last):", "KeyboardInterrupt"])


def test_a_real_crash_is_still_one():
    d = D.diagnose(OOM_TAIL, 1)
    assert not d.stopped and d.as_dict()["stopped"] is False
    assert "OutOfMemoryError" in d.summary
    # A traceback that merely mentions the word is not an interrupt.
    assert not D.stopped_from_outside(["ValueError: KeyboardInterrupt handling failed"])


def test_the_monitor_files_no_bug_for_an_engine_a_restart_stopped():
    stopped = _engine(D.diagnose(RESTART_TAIL, 2).as_dict())
    crashed = _engine(D.diagnose(OOM_TAIL, 1).as_dict(), model="Muse-Glimmer-30B-NVFP4")
    found = M.engine_findings([stopped, crashed])
    assert [f["model"] for f in found] == ["Muse-Glimmer-30B-NVFP4"]
    assert found[0]["fingerprint"].startswith("engine:Muse-Glimmer-30B-NVFP4:")
    # A diagnosis made before `stopped` existed (no flag): the tail still tells.
    legacy = {"summary": "ERROR    Backend supervisor: backend worker freetoken-detokenizer-0 exited",
              "tail": RESTART_TAIL}
    assert M.engine_findings([_engine(legacy)]) == []
    # Running / stopped (unloaded on purpose) engines are never findings.
    assert M.engine_findings([{**crashed, "state": "running"}, {**crashed, "state": "stopped"}]) == []


LOADED = [
    {"model": "Muse-Glimmer-30B-NVFP4", "ready": True, "aa": 30},
    {"model": "Qwen3.6-35B-A3B", "ready": True, "aa": 40},
    {"model": "DeepSeek-V4-Flash-0731@lambda999", "ready": True, "aa": 60, "remote": {"computer": "lambda999"}},
    {"model": "kimi@groq", "ready": True, "aa": 70, "external": {"provider": "groq"}},
]


def test_triage_falls_back_to_a_loaded_local_model(monkeypatch):
    monkeypatch.setattr(M, "_loaded", lambda: LOADED)
    # The configured model is not loaded: the best LOCAL one, not the remote or the external one.
    assert M.pick_model({"model": "Qwen/Qwen3-0.6B"}) == "Qwen3.6-35B-A3B"
    note = M.triage_note("Qwen/Qwen3-0.6B", "Qwen3.6-35B-A3B")
    assert "Qwen/Qwen3-0.6B is not loaded" in note and "using Qwen3.6-35B-A3B" in note
    # Loaded: it is used, whatever it is (an external one only when picked).
    assert M.pick_model({"model": "Muse-Glimmer-30B-NVFP4"}) == "Muse-Glimmer-30B-NVFP4"
    assert M.pick_model({"model": "kimi@groq"}) == "kimi@groq"
    assert M.triage_note("kimi@groq", "kimi@groq") is None
    assert M.pick_model({"model": ""}) == "Qwen3.6-35B-A3B" and M.triage_note("", "Qwen3.6-35B-A3B") is None


def test_triage_never_falls_back_to_an_external_model(monkeypatch):
    monkeypatch.setattr(M, "_loaded", lambda: [LOADED[3]])
    assert M.pick_model({"model": "Qwen/Qwen3-0.6B"}) is None
    assert "load Qwen/Qwen3-0.6B or any local model" in M.triage_note("Qwen/Qwen3-0.6B", None)
    assert M.triage_note("", None) == "no local model loaded"
    # A paired computer's model is the second choice, before nothing.
    monkeypatch.setattr(M, "_loaded", lambda: [LOADED[2], LOADED[3]])
    assert M.pick_model({"model": "Qwen/Qwen3-0.6B"}) == "DeepSeek-V4-Flash-0731@lambda999"


# --- ft's compatibility shims are not harness faults --------------------------------------------
def _stack(agent_line: str, ft_fn: str, ft_line: str, tail: str) -> str:
    return ('Traceback (most recent call last):\n'
            '  File "script.py", line 9, in <module>\n    exec(compile(_src, "candidate.py", "exec"))\n'
            f'  File "candidate.py", line 20, in <module>\n    {agent_line}\n'
            f'  File "/work/.ft/ft.py", line 354, in {ft_fn}\n    {ft_line}\n' + tail)


POLARS_RAISE = ('  File "/usr/local/lib/python3.12/site-packages/polars/functions/lit.py", line 228, in lit\n'
                '    return wrap_expr(plr.lit(item, allow_object, is_scalar=True))\n'
                'TypeError: cannot create expression literal for value of type Series.\n')
PANDAS_KEY = ('  File "/usr/local/lib/python3.12/site-packages/pandas/core/frame.py", line 4378, in __getitem__\n'
              '    indexer = self.columns.get_loc(key)\n'
              "KeyError: 't'\n")


def test_an_error_under_ft_s_method_shim_is_the_agent_s():
    tb = M.traceback_of(_stack("rows = rows.with_columns(vwap = vwap)", "with_columns",
                               "out = orig_wc(base, *exprs, **named)", POLARS_RAISE))
    assert tb["origin"] == "agent"
    # The shim's own refusal on the agent's object joins the mix-up family, not "ft keeps refusing".
    refusal = M.traceback_of(_stack("x = s.to_numpy().nunique2()", "__getattr__",
                                    "raise AttributeError(f\"'numpy.ndarray' object has no attribute '{name}'\")",
                                    "AttributeError: 'numpy.ndarray' object has no attribute 'nunique2'\n"))
    assert refusal["origin"] == "agent"


def test_an_ft_helper_called_by_name_stays_the_harness_s():
    for line in ("score = ft.quick_score(positions, rows)", "score = quick_score(aligned.values, rows)"):
        tb = M.traceback_of(_stack(line, "quick_score", "session, _ = clock(_to_pandas(rows)[time], tz)",
                                   PANDAS_KEY))
        assert tb["origin"] == "harness", line
    # ft's own code breaking (not a raise, nothing third-party below it) is the harness's even
    # under a shim.
    own = M.traceback_of(_stack("rows = rows.with_columns(a=1)", "with_columns", "out = orig_wcc(base)",
                                "NameError: name 'orig_wcc' is not defined\n"))
    assert own["origin"] == "harness"
