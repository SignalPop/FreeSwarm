"""ui/restart-swarm.cmd, ui/swarm-drain.ps1, ui/run-swarm.bat: a restart ends with ONE runner.

10-01 20:32: `restart-swarm.cmd --hard` left two runners. It killed only the runner's python
processes; the old `cmd /k run-swarm.bat` window then went on reading its batch file, which had
been edited since it started (cmd re-reads a batch file from disk after each command and resumes
at the old byte offset), landed before the launch line in the new text and started a second
runner beside the one the restart started.

These run the real scripts against a FAKE runner (a script under a unique name, in a temp copy
of the layout with its own venv, so the venv launcher + child pair is real) and never touch a
real runner: the scripts' process and window patterns are pointed at the fake through
FREESWARM_RUNNER_MATCH / FREESWARM_RUNNER_WINDOW_MATCH.

They open console windows and take ~1 min, so they only run with FREESWARM_SCRIPT_TESTS=1:
    set FREESWARM_SCRIPT_TESTS=1 && .venv\\Scripts\\python -m pytest ui\\backend\\tests\\test_restart_swarm_scripts.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.skipif(sys.platform != "win32", reason="Windows batch scripts"),
    pytest.mark.skipif(os.environ.get("FREESWARM_SCRIPT_TESTS") != "1",
                       reason="opens console windows; set FREESWARM_SCRIPT_TESTS=1"),
]

# run-swarm.bat as it was before the fix (HEAD 1cb5eb6): the launch line is followed by more
# lines, so cmd comes back to the file after the runner ends.
OLD_RUN_SWARM = """@echo off
setlocal
rem Swarm task runner (the pre-fix shape: launch, then more lines read after it returns).

set "ROOT=%~dp0.."
set "PY=%ROOT%\\.venv\\Scripts\\python.exe"

if not exist "%PY%" (
  echo [!] venv not found at %PY%
  exit /b 1
)

cd /d "%ROOT%\\ui\\backend"
echo Swarm runner -^> claiming tasks
"%PY%" swarm_runner.py
echo.
echo Swarm runner exited (code %ERRORLEVEL%).
"""

FAKE_RUNNER = """import json, os, sys, time
# Honours the drain file like swarm_runner.py: report, remove the request, exit 0.
while True:
    if os.path.exists('.swarm_drain'):
        with open('.swarm_drain.status.json', 'w') as fh:
            json.dump({'running': 0, 'iterations': []}, fh)
        time.sleep(1)
        os.remove('.swarm_drain')
        sys.exit(0)
    time.sleep(0.2)
"""


def _edit_while_running(text: str) -> str:
    """The 20:12 edit: lines added near the top, sized so that the byte offset just after the
    launch line in the old text is the start of the `cd /d` line in the new one -- the old
    window then runs cd, echo and the launch again."""
    start = text.index('cd /d "%ROOT%')
    end = text.index("\n", text.index('"%PY%" ', start)) + 1
    span = end - start
    pad = "rem " + "x" * (span - 5) + "\n"
    i = text.index("\n", text.index("setlocal")) + 1
    return text[:i] + pad + text[i:]


def _procs(token: str) -> list[dict]:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "@(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*%s*' } | "
         "Select-Object ProcessId,ParentProcessId,Name,CommandLine) | ConvertTo-Json -Compress" % token],
        capture_output=True, text=True, timeout=60).stdout.strip()
    if not out:
        return []
    data = json.loads(out)
    return data if isinstance(data, list) else [data]


@pytest.fixture()
def layout(tmp_path):
    token = "frs" + uuid.uuid4().hex[:10]
    root = tmp_path / token
    (root / "ui" / "backend").mkdir(parents=True)
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(root / ".venv")], check=True, timeout=120)
    for name in ("restart-swarm.cmd", "swarm-drain.ps1"):
        shutil.copy(UI / name, root / "ui" / name)
    script = f"fake_swarm_runner_{token}.py"
    (root / "ui" / "backend" / script).write_text(FAKE_RUNNER)
    env = dict(os.environ,
               FREESWARM_RUNNER_MATCH=re.escape(script),
               FREESWARM_RUNNER_WINDOW_MATCH=re.escape(token) + r".*run-swarm\.bat")
    env.pop("FREESWARM_DRAIN_FILE", None)

    class L:
        pass
    lay = L()
    lay.token, lay.root, lay.env, lay.script = token, root, env, script
    lay.bat = root / "ui" / "run-swarm.bat"
    lay.named = lambda text: text.replace("swarm_runner.py", script)
    lay.batch = lambda text: lay.bat.write_bytes(lay.named(text).encode())
    lay.current = (UI / "run-swarm.bat").read_text()
    try:
        yield lay
    finally:
        for p in _procs(token):
            subprocess.run(["taskkill", "/PID", str(p["ProcessId"]), "/F"], capture_output=True)


def _open_window(lay) -> int:
    """A runner window the way start-services.cmd opens one (cmd /k run-swarm.bat)."""
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 7  # SW_SHOWMINNOACTIVE
    w = subprocess.Popen(["cmd", "/k", str(lay.bat)], creationflags=subprocess.CREATE_NEW_CONSOLE,
                         startupinfo=si, env=lay.env)
    return w.pid


def _runners(lay) -> list[dict]:
    return [p for p in _procs(lay.token) if p["Name"] == "python.exe" and lay.script in (p["CommandLine"] or "")]


def _windows(lay) -> list[dict]:
    return [p for p in _procs(lay.token) if p["Name"] == "cmd.exe" and "run-swarm.bat" in (p["CommandLine"] or "")]


def _wait_runner(lay, timeout=20) -> list[dict]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = _runners(lay)
        if len(r) >= 2:
            return r
        time.sleep(0.5)
    raise AssertionError(f"runner pair did not start: {_procs(lay.token)}")


def _restart(lay, *args) -> str:
    log = lay.root / "restart.log"
    with open(log, "w") as out:
        # stdin nul and output to a file, as the 20:32 run did: the new window `start` opens must
        # not hold a pipe this call waits on.
        rc = subprocess.run(["cmd", "/c", str(lay.root / "ui" / "restart-swarm.cmd"), *args],
                            stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                            env=lay.env, timeout=300).returncode
    text = log.read_text(errors="replace")
    assert rc == 0, text
    return text


def _assert_one_runner(lay, old_windows: set[int]) -> None:
    time.sleep(4)  # long enough for an old window to have re-run its batch file
    _wait_runner(lay)
    time.sleep(3)
    r = _runners(lay)
    pids = {p["ProcessId"] for p in r}
    assert len(r) == 2, f"expected one launcher+child pair, got {r}"
    launcher = [p for p in r if p["ParentProcessId"] not in pids]
    child = [p for p in r if p["ParentProcessId"] in pids]
    assert len(launcher) == 1 and len(child) == 1, r
    wins = _windows(lay)
    assert len(wins) == 1, f"expected one runner window, got {wins}"
    assert wins[0]["ProcessId"] not in old_windows, "the runner is in an OLD window"
    assert launcher[0]["ParentProcessId"] == wins[0]["ProcessId"], (launcher, wins)


@pytest.mark.parametrize("hard", [True, False], ids=["hard", "graceful"])
def test_restart_of_a_window_whose_batch_was_edited_leaves_one_runner(layout, hard):
    lay = layout
    lay.batch(OLD_RUN_SWARM)
    w = _open_window(lay)
    _wait_runner(lay)
    stale = _open_window_stale(lay)
    lay.batch(_edit_while_running(lay.named(OLD_RUN_SWARM)))   # edited while it runs (20:12)
    text = _restart(lay, *(["--hard"] if hard else []))
    assert "[close]" in text, text
    _assert_one_runner(lay, {w, stale})


def _open_window_stale(lay) -> int:
    """A leftover window whose runner is long gone (the 17:09 one): its batch failed and
    returned to the prompt. Same command line shape, so a restart must close it too."""
    bad = lay.root / "stale" / "ui"
    bad.mkdir(parents=True)
    (bad / "run-swarm.bat").write_text(OLD_RUN_SWARM)   # no venv there: exits at once
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 7
    return subprocess.Popen(["cmd", "/k", str(bad / "run-swarm.bat")], creationflags=subprocess.CREATE_NEW_CONSOLE,
                            startupinfo=si).pid


def test_the_10_01_failure_reproduces_with_the_old_batch_and_a_python_only_kill(layout):
    """Control for the tests above: what the old --hard did (kill python, leave the window)
    to a window running the old run-swarm.bat after an edit makes that window start a new
    runner by itself."""
    lay = layout
    lay.batch(OLD_RUN_SWARM)
    w = _open_window(lay)
    r = _wait_runner(lay)
    lay.batch(_edit_while_running(lay.named(OLD_RUN_SWARM)))
    for p in r:
        subprocess.run(["taskkill", "/PID", str(p["ProcessId"]), "/F"], capture_output=True)
    respawned = _wait_runner(lay)
    assert {p["ProcessId"] for p in respawned}.isdisjoint({p["ProcessId"] for p in r})
    assert any(p["ParentProcessId"] == w for p in respawned), "respawned in the old window"


def test_fixed_run_swarm_never_rereads_itself_after_the_runner_ends(layout):
    """Even when the runner is killed the old way (python only, window left open)."""
    lay = layout
    lay.batch(lay.current)
    _open_window(lay)
    r = _wait_runner(lay)
    lay.batch(_edit_while_running(lay.named(lay.current)))
    for p in r:
        subprocess.run(["taskkill", "/PID", str(p["ProcessId"]), "/F"], capture_output=True)
    time.sleep(5)
    assert _runners(lay) == []


def test_check_reports_a_running_runner(layout):
    lay = layout
    ps = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
          str(lay.root / "ui" / "swarm-drain.ps1"), "-Check"]
    assert subprocess.run(ps, env=lay.env, capture_output=True, timeout=60).returncode == 0
    lay.batch(lay.current)
    _open_window(lay)
    _wait_runner(lay)
    res = subprocess.run(ps, env=lay.env, capture_output=True, text=True, timeout=60)
    assert res.returncode == 1 and "already running" in res.stdout
