@echo off
setlocal
rem ===================================================================================
rem  Restart ONLY the swarm runner, gracefully.
rem
rem    restart-swarm.cmd          drain, then restart: no agent starts new work, every
rem                               iteration already running finishes (submits) normally, the
rem                               runner exits on its own, and a fresh one is started the way
rem                               start-services.cmd does. Waits up to FREESWARM_DRAIN_MAX_S
rem                               (default 5400 s) plus a margin; past that it is killed.
rem    restart-swarm.cmd --hard   kill it now (in-flight iterations are lost), then start.
rem                               Needed once for a runner started before drain support.
rem
rem  Either way it ends with exactly one runner, in one new window: every old runner window
rem  is closed (a graceful one keeps its console until the runner exits) and any other runner
rem  process is killed before the new one starts.
rem
rem  Progress: this window, the runner's own window ("DRAIN requested" / "draining: N still
rem  running" every few minutes), ui\backend\.swarm_drain.status.json, and board state key
rem  "swarm_drain". Cancel a drain by deleting ui\backend\.swarm_drain. Never prompts.
rem ===================================================================================

set "UI=%~dp0"
if "%UI:~-1%"=="\" set "UI=%UI:~0,-1%"
set "ROOT=%UI%\.."
for %%r in ("%ROOT%") do set "ROOT=%%~fr"

echo.
echo   FreeSwarm - restarting the swarm runner
echo   ------------------------------------------------------------------

rem Both paths close every old runner window (cmd /k run-swarm.bat) BEFORE the runner ends, and
rem leave no runner process behind: on 10-01 a --hard restart killed only the python processes,
rem the old window then re-ran its (meanwhile edited) run-swarm.bat, and two runners ran. See
rem swarm-drain.ps1.
set "MODE=-CloseWindow"
if /i "%~1"=="--hard" set "MODE=-Hard"
powershell -NoProfile -ExecutionPolicy Bypass -File "%UI%\swarm-drain.ps1" %MODE%
set "RC=%ERRORLEVEL%"
if "%RC%"=="3" (
  echo.
  echo   Not restarted: the running runner ignores drain requests. Run once:
  echo       ui\restart-swarm.cmd --hard
  exit /b 3
)
if not "%RC%"=="0" if not "%RC%"=="2" (
  echo   [X] stopping the runner failed ^(exit %RC%^) - runner not restarted
  exit /b %RC%
)

rem Last guard: never start a second runner.
powershell -NoProfile -ExecutionPolicy Bypass -File "%UI%\swarm-drain.ps1" -Check
if errorlevel 1 (
  echo   [X] a swarm runner is still running - not starting another. Try:  ui\restart-swarm.cmd --hard
  exit /b 4
)
start "FreeSwarm swarm runner" cmd /k "%ROOT%\ui\run-swarm.bat"
echo   [ok] swarm runner started in a new window
echo.
exit /b 0
