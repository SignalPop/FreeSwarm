@echo off
setlocal
rem ===================================================================================
rem  Python sandbox - prepares the image that runs the code chat writes.
rem
rem  There is no long-lived container to start. The control plane runs ONE CONTAINER PER
rem  RUN with --network none, because that is the only arrangement on Docker Desktop that
rem  actually blocks outbound traffic:
rem
rem    --internal network              published ports stop working; unreachable
rem    bridge, masquerade disabled     port works, but the WSL2 VM's own NAT still routes
rem                                    out - 1.1.1.1:53 and 8.8.8.8:443 were reachable
rem    --network none                  egress blocked outright (verified), and no
rem                                    listening socket exists to begin with
rem
rem  So this script builds the image and checks it can run. start-services.cmd calls it;
rem  running it by hand is how you rebuild after editing sandbox\requirements.txt.
rem ===================================================================================

set "ROOT=%~dp0.."
set "IMAGE=freeswarm-sandbox:latest"

docker info >nul 2>&1
if errorlevel 1 (
  echo.
  echo   [X] Docker is not running.
  echo       Start Docker Desktop and run this again. Everything else in FreeToken
  echo       works without it - only the chat Run button needs the sandbox.
  echo.
  exit /b 1
)

if /i "%~1"=="--rebuild" (
  echo   Rebuilding %IMAGE% ...
  docker build --no-cache -t %IMAGE% "%ROOT%\ui\sandbox" || exit /b 1
  goto :verify
)

docker image inspect %IMAGE% >nul 2>&1
if errorlevel 1 (
  echo   Building %IMAGE% ^(first run only - installs pandas/matplotlib/openpyxl, a few minutes^)...
  docker build -t %IMAGE% "%ROOT%\ui\sandbox"
  if errorlevel 1 (
    echo   [X] Image build failed.
    exit /b 1
  )
) else (
  echo   [ok] %IMAGE% already built
)

:verify
rem Prove a run actually works, rather than only that the image exists - a half-built or
rem architecture-mismatched image fails here instead of at the user's first Run click.
for /f "delims=" %%v in ('docker run --rm --network none %IMAGE% python -c "import pandas,matplotlib,openpyxl,docx;print('ok')" 2^>nul') do set "PROBE=%%v"
if /i not "%PROBE%"=="ok" (
  echo   [X] The image is present but a test run failed. Try:  ui\run-sandbox.bat --rebuild
  exit /b 1
)

echo   [ok] sandbox ready - Run buttons in chat are live
exit /b 0
