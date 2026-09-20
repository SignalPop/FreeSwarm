@echo off
setlocal EnableDelayedExpansion
rem ===================================================================================
rem  FreeToken - stop everything
rem
rem  Order matters. The control plane reaps the engine on shutdown, so it goes first and
rem  gets a moment to do that cleanly. Anything still holding an engine port afterwards
rem  is an orphaned worker, and those are killed with /T because the engine spawns
rem  scheduler and tokenizer workers as SEPARATE processes under the base interpreter -
rem  killing only a parent leaves them holding the port and their VRAM.
rem ===================================================================================

echo.
echo   FreeToken - stopping services
echo   ------------------------------------------------------------------

rem ---- 1. control plane, politely: its lifespan hook stops the engine --------------
call :killport 8000 "control plane"
if defined KILLED (
  echo   ... waiting for the engine to be reaped
  "%SystemRoot%\System32\timeout.exe" /t 6 /nobreak >nul 2>&1
)

call :killport 8100 "message board"
call :killport 3000 "console"

rem ---- 1b. swarm runner + time-series servers ---------------------------------------
rem Matched by command line: the swarm runner binds no port, and the forecasting servers
rem (ports 1960-1999) are normally reaped by the control plane, so this only catches strays.
powershell -NoProfile -Command ^
  "$p = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'swarm_runner|tsfm_server' };" ^
  "if ($p) { $p | ForEach-Object { Write-Host ('   [kill] swarm runner (pid ' + $_.ProcessId + ')'); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } } else { Write-Host '   [ -- ] swarm runner not running' }" 2>nul

rem ---- 1c. sandbox containers --------------------------------------------------------
rem Each run is its own --rm container named ft-sandbox-<id>, so normally there is nothing
rem here; this only catches one still executing when you stopped everything.
where docker >nul 2>&1
if not errorlevel 1 (
  docker info >nul 2>&1
  if not errorlevel 1 (
    for /f %%c in ('docker ps -q -f name^=ft-sandbox- 2^>nul') do (
      echo   [kill] sandbox run container %%c
      docker stop %%c >nul 2>&1
    )
  )
)

rem ---- 2. anything still on the engine ports is an orphan --------------------------
rem The manager runs one engine per GPU, allocating port pairs upward from 1919
rem (1919/1920, 1921/1922, ...). Sweep the range so a second or third engine is not left
rem behind holding its VRAM.
for %%p in (1919 1920 1921 1922 1923 1924 1925 1926) do call :killport %%p "engine port %%p"

rem ---- 3. belt and braces: engine processes that never bound a port ----------------
rem Matches both the parent (`-m freetoken`) and the spawned workers, which run under
rem the BASE interpreter with a multiprocessing.spawn command line and therefore do not
rem match a search for the venv python.
powershell -NoProfile -Command ^
  "$p = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match '-m freetoken|freetoken\.server|spawn_main' };" ^
  "if ($p) { $p | ForEach-Object { Write-Host ('   [kill] stray engine process ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }" 2>nul

echo.
echo   ------------------------------------------------------------------
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader 2>nul
echo   ------------------------------------------------------------------
echo   Done. VRAM above should be back to idle (a few hundred MiB on the
echo   display GPU, ~10 MiB on the compute cards).
echo.
goto :eof

rem ---------------------------------------------------------------------------------
:killport
rem %1 = port, %2 = label. Kills the whole process tree of whatever is LISTENING.
set "KILLED="
for /f "tokens=5" %%p in ('netstat -ano -p TCP ^| findstr /r /c:":%~1 .*LISTENING" 2^>nul') do (
  if not "%%p"=="0" (
    echo   [kill] %~2 ^(pid %%p, port %~1^)
    taskkill /PID %%p /T /F >nul 2>&1
    set "KILLED=1"
  )
)
if not defined KILLED echo   [ -- ] %~2 not running
exit /b 0
