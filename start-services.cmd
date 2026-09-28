@echo off
setlocal EnableDelayedExpansion
rem ===================================================================================
rem  FreeToken - start everything
rem
rem    Console        http://localhost:3000     the web UI
rem    Control plane  http://127.0.0.1:8000     engine lifecycle, telemetry, MCP, /v1
rem    Message board  http://127.0.0.1:8100     agent coordination
rem    Data/action    http://127.0.0.1:82xx/mcp every task server under mcp\ (a folder with server.py
rem    MCPs                                     and make_oauth_secrets.py): gex 8200, battery 8201,
rem                                             tables 8202 -- over HTTP with OAuth (mcp\README.md)
rem    Swarm runner   (no port)                 claims queued tasks, one agent per model
rem    Sandbox        (no port)                 per-run docker container for chat's Python
rem
rem  The inference engine itself is NOT started here - it is a child process of the
rem  control plane, launched from the Models page (or POST /api/engine/start), so that
rem  stopping the control plane always reaps it and its VRAM.
rem
rem  Everything binds 127.0.0.1. See ui\README.md before exposing any of it.
rem ===================================================================================

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "PY=%ROOT%\.venv\Scripts\python.exe"

echo.
echo   FreeToken - starting services
echo   ------------------------------------------------------------------

rem ---- preflight: interpreter ---------------------------------------------------
if not exist "%PY%" (
  echo   [X] Python venv not found at:
  echo       %PY%
  echo.
  echo       Create it and install FreeToken first - see ui\README.md
  goto :fail
)

rem freetoken and its compiled extensions must be importable, otherwise the control plane
rem starts but every engine launch fails at model load. `import freetoken` alone passes
rem without the extensions, because they are loaded lazily.
"%PY%" -c "import freetoken.kernel._pinned_tensor, freetoken.kernel._cpu_moe" >nul 2>&1
if errorlevel 1 (
  echo   [X] The engine's compiled kernels are missing or do not import.
  echo       Build them with:  build-kernel.cmd
  goto :fail
)
echo   [ok] venv                %PY%

rem ---- preflight: node -----------------------------------------------------------
where node >nul 2>&1
if errorlevel 1 (
  echo   [X] node not found on PATH - the console needs Node.js 18+.
  goto :fail
)
for /f "tokens=*" %%v in ('node --version') do echo   [ok] node               %%v

rem ---- preflight: CUDA toolchain -------------------------------------------------
rem Only a warning: the control plane and board run fine without it, but no model will
rem serve, because the engine JIT-compiles CUDA kernels on first use.
set "CU13=%ROOT%\.venv\Lib\site-packages\nvidia\cu13\bin\nvcc.exe"
if exist "%CU13%" (
  echo   [ok] CUDA toolchain     venv cu13 ^(nvcc 13.x^)
) else (
  echo   [warn] CUDA 13 wheels not found in the venv. Models will fail to serve.
  echo        pip install nvidia-cuda-nvcc nvidia-nvvm nvidia-cuda-crt nvidia-cuda-runtime
)

rem ---- preflight: docker (optional) ----------------------------------------------
rem Only the chat Run button needs it. Everything else - engines, board, console - runs
rem without Docker, so a stopped daemon must not block startup.
set "DOCKER_OK="
where docker >nul 2>&1
if errorlevel 1 (
  echo   [warn] docker not on PATH - the chat Run button will be disabled.
) else (
  docker info >nul 2>&1
  if errorlevel 1 (
    echo   [warn] Docker is installed but not running - start Docker Desktop to enable
    echo          the chat Run button, then run ui\run-sandbox.bat.
  ) else (
    set "DOCKER_OK=1"
    echo   [ok] docker               sandbox enabled
  )
)

rem ---- preflight: ports ----------------------------------------------------------
rem A port already in use usually means these services are already running, or an
rem engine was killed without its process tree and a worker still holds 1919/1920.
rem One PowerShell call for all six ports. Get-NetTCPConnection is used rather than
rem `netstat | findstr` because the obvious findstr pattern (":8000 .*LISTENING") has a
rem space in it, which cmd splits into a second argument - findstr then treats it as a
rem filename and the check silently misreports.
rem The PowerShell call is a PLAIN statement redirected to a temp file, NOT nested inside
rem for /f (...). cmd re-parses the embedded double quotes of a for /f command string and
rem mangles it into "The system cannot find the file powershell." - the check then silently
rem reports every port free, which is worse than not checking at all.
set "PORTCHK=%TEMP%\freetoken_ports_%RANDOM%.txt"
powershell -NoProfile -Command "@(8000,8100,8200,8201,8202,3000,1919,1920) | Where-Object { Get-NetTCPConnection -LocalPort $_ -State Listen -ErrorAction SilentlyContinue }" > "%PORTCHK%" 2>nul
set "BUSY="
for /f "usebackq delims=" %%p in ("%PORTCHK%") do (
  echo   [warn] port %%p is already in use
  set "BUSY=1"
)
del "%PORTCHK%" >nul 2>&1
if defined BUSY (
  echo.
  echo   ------------------------------------------------------------------
  echo   Some ports are already in use ^(listed above^).
  echo   If the services are already running, just open http://localhost:3000
  echo   To stop them first, run:   stop-services.cmd
  echo.
  rem /d N /t 20 gives a safe default; errorlevel 255 means choice could not read a
  rem console (piped or scripted run), and proceeding is the useful behaviour there.
  choice /c YN /n /d N /t 20 /m "   Start anyway? [Y/N] "
  if errorlevel 255 goto :launch
  if errorlevel 2 goto :done
)

:launch

rem ---- sandbox image --------------------------------------------------------------
rem Built by build-services.cmd. Verified here only: each run gets its own container
rem with --network none (see ui\run-sandbox.bat), so there is nothing to start.
if defined DOCKER_OK (
  docker image inspect freeswarm-sandbox:latest >nul 2>&1
  if errorlevel 1 (
    echo   [warn] sandbox image not built - the chat Run button will be disabled.
    echo          Build it with:  build-services.cmd --sandbox
  ) else (
    echo   [ok] sandbox             freeswarm-sandbox:latest
  )
)

rem ---- console build --------------------------------------------------------------
rem The console is SERVED here, not built here: `next start` against a build made by
rem build-services.cmd. Building at startup meant the first request compiled for ten
rem seconds and a syntax error showed up as a blank page rather than a build failure.
if not exist "%ROOT%\ui\frontend\node_modules" (
  echo   [X] Console dependencies are not installed.
  echo       Run:  build-services.cmd
  goto :fail
)
if not exist "%ROOT%\ui\frontend\.next\BUILD_ID" (
  echo   [X] The console has no production build.
  echo       Run:  build-services.cmd
  echo       ^(or work on the UI with hot reload:  ui\run-frontend.bat --dev^)
  goto :fail
)
echo   [ok] console             production build present

rem ---- data/action MCPs -------------------------------------------------------------
rem Every folder under mcp\ with a server.py and a make_oauth_secrets.py is a task server. One
rem that has never run gets its OAuth secrets here, one at a time (each registers itself in
rem ui\backend\mcp_servers.json, so they must not race), before the control plane reads them.
set "MCPS="
for /d /r "%ROOT%\mcp" %%d in (*) do (
  if exist "%%d\server.py" if exist "%%d\make_oauth_secrets.py" call :mcp_prepare "%%d"
)

rem ---- launch --------------------------------------------------------------------
echo.
echo   Launching service windows...
start "FreeSwarm control plane" cmd /k "%ROOT%\ui\run-control-plane.bat"
start "FreeSwarm message board" cmd /k "%ROOT%\ui\run-msgboard.bat"
for %%d in (!MCPS!) do start "FreeSwarm MCP %%~nxd" cmd /k ""%ROOT%\ui\run-mcp.bat" "%%~d""
start "FreeSwarm swarm runner"  cmd /k "%ROOT%\ui\run-swarm.bat"
start "FreeSwarm console"       cmd /k "%ROOT%\ui\run-frontend.bat"

rem A production build serves immediately -- this is only the few seconds `next start`
rem needs to bind the port, not the ten the dev server spent compiling.
"%SystemRoot%\System32\timeout.exe" /t 4 /nobreak >nul 2>&1
start "" http://localhost:3000

echo.
echo   ------------------------------------------------------------------
echo     Console        http://localhost:3000
echo     Control plane  http://127.0.0.1:8000/docs
echo     Message board  http://127.0.0.1:8100/docs
for %%d in (!MCPS!) do echo     MCP %%~nxd      ^(%%~d^)
echo     Data/action MCPs connect once from Connectors -^> Connect ^(passphrase in the server's .oauth\^)
echo.
echo     Start a model from the Models page in the console.
echo     Queued Swarm tasks are answered by the swarm runner, one agent per model.
echo     Python from chat runs in the sandbox container - press Run on a code block.
echo     Stop everything with:  stop-services.cmd
echo   ------------------------------------------------------------------
echo.
goto :done

rem ---------------------------------------------------------------------------------
:mcp_prepare
rem %1 = a task server's folder. Creates its OAuth secrets on the first start, then queues it.
if not exist "%~1\.oauth\server.json" (
  "%PY%" "%~1\make_oauth_secrets.py" --quiet >nul 2>&1
  if errorlevel 1 (
    echo   [warn] MCP %~nx1: could not create its OAuth secrets - run make_oauth_secrets.py in %~1
    exit /b 0
  )
  echo   [ok] MCP %~nx1            OAuth secrets created; connect it once from Connectors
  echo        ^(approval passphrase: %~1\.oauth\approval_passphrase.txt^)
) else (
  echo   [ok] MCP %~nx1            %~1
)
set "MCPS=!MCPS! "%~1""
exit /b 0

rem ---------------------------------------------------------------------------------
:fail
echo.
echo   Startup aborted.
echo.
pause
exit /b 1

:done
endlocal
exit /b 0
