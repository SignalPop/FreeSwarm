@echo off
setlocal
rem ===================================================================================
rem  FreeSwarm console. Proxies /api -> :8000 and /mb -> :8100, so the browser only ever
rem  talks to this one origin and neither backend needs to be network-reachable.
rem
rem  Serves the PRODUCTION build (`next start`). The build is made by build-services.cmd,
rem  not here: compiling on first request made the console look hung for ten seconds and
rem  turned a typo into a blank page instead of a build error.
rem
rem    run-frontend.bat           serve the production build   (what start-services uses)
rem    run-frontend.bat --dev     the dev server, with hot reload, for working on the UI
rem ===================================================================================

set "ROOT=%~dp0.."
cd /d "%ROOT%\ui\frontend"

if not exist node_modules (
  echo   [X] Console dependencies are not installed.
  echo       Run:  build-services.cmd
  exit /b 1
)

if /i "%~1"=="--dev" (
  echo   Console ^(dev, hot reload^) -^> http://localhost:3000
  call npm run dev
  exit /b %errorlevel%
)

rem `next start` refuses to run without a build, with a message that does not say which
rem command makes one. Say it here instead.
if not exist ".next\BUILD_ID" (
  echo.
  echo   [X] No production build of the console found.
  echo.
  echo       Build it first:   build-services.cmd
  echo       Or work on the UI with hot reload:   ui\run-frontend.bat --dev
  echo.
  exit /b 1
)

echo   Console -^> http://localhost:3000
call npm start
