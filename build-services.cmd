@echo off
setlocal EnableDelayedExpansion
rem ===================================================================================
rem  FreeSwarm - build everything start-services.cmd then just runs
rem
rem  Split out from start-services.cmd so that starting is fast and predictable: the
rem  console is served as a production build (`next start`), not the dev server, so
rem  nothing compiles on the first request and a syntax error surfaces here rather than
rem  as a blank page at 3am.
rem
rem  Run this after: a fresh clone, `git pull`, an edit to ui\frontend, an edit to
rem  ui\sandbox\requirements.txt, or an edit to python\freetoken\kernel\csrc. You do NOT
rem  need it to change other Python code -- the control plane, board and swarm runner
rem  run from source.
rem
rem    build-services.cmd              engine kernels + console + sandbox image
rem    build-services.cmd --kernel     engine kernels only  (same as build-kernel.cmd)
rem    build-services.cmd --console    console only
rem    build-services.cmd --sandbox    sandbox image only
rem ===================================================================================

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "PY=%ROOT%\.venv\Scripts\python.exe"

set "DO_KERNEL=1"
set "DO_CONSOLE=1"
set "DO_SANDBOX=1"
if /i "%~1"=="--kernel"  ( set "DO_CONSOLE=" & set "DO_SANDBOX=" )
if /i "%~1"=="--console" ( set "DO_KERNEL=" & set "DO_SANDBOX=" )
if /i "%~1"=="--sandbox" ( set "DO_KERNEL=" & set "DO_CONSOLE=" )

echo.
echo   FreeSwarm - building
echo   ------------------------------------------------------------------

rem ---- engine kernels --------------------------------------------------------------
rem `import freetoken` succeeds without the compiled extensions, so an import check is
rem not enough: build-kernel.cmd checks the .pyd files themselves and is a no-op when
rem they are current.
if defined DO_KERNEL (
  call "%ROOT%\build-kernel.cmd" --nopause
  if errorlevel 1 goto :fail
)

rem ---- console -------------------------------------------------------------------
if defined DO_CONSOLE (
  where node >nul 2>&1
  if errorlevel 1 (
    echo   [X] node not found on PATH - the console needs Node.js 18+.
    goto :fail
  )

  pushd "%ROOT%\ui\frontend"

  if not exist node_modules (
    echo.
    echo   Installing console dependencies...
    call npm install
    if errorlevel 1 ( popd & echo   [X] npm install failed. & goto :fail )
  ) else (
    rem package.json newer than node_modules means a dependency changed since the last
    rem install. Cheap to check, and skipping it produces a build failure that reads
    rem like a code error.
    for /f %%i in ('powershell -NoProfile -Command "if ((Get-Item package.json).LastWriteTime -gt (Get-Item node_modules).LastWriteTime) { 'stale' }"') do set "DEPS=%%i"
    if "!DEPS!"=="stale" (
      echo.
      echo   package.json changed since the last install - reinstalling...
      call npm install
      if errorlevel 1 ( popd & echo   [X] npm install failed. & goto :fail )
    ) else (
      echo   [ok] console deps       node_modules up to date
    )
  )

  echo.
  echo   Building the console ^(next build^)...
  call npm run build
  if errorlevel 1 (
    popd
    echo.
    echo   [X] The console build failed - see the errors above.
    goto :fail
  )
  popd
  echo   [ok] console             production build in ui\frontend\.next
)

rem ---- sandbox image -------------------------------------------------------------
if defined DO_SANDBOX (
  where docker >nul 2>&1
  if errorlevel 1 (
    echo   [skip] docker not on PATH - the chat Run button stays disabled. Everything
    echo          else works without it.
  ) else (
    docker info >nul 2>&1
    if errorlevel 1 (
      echo   [skip] Docker is installed but not running - start Docker Desktop and run
      echo          build-services.cmd --sandbox to enable the chat Run button.
    ) else (
      echo.
      call "%ROOT%\ui\run-sandbox.bat"
      if errorlevel 1 (
        echo   [warn] sandbox image not ready - the chat Run button will be disabled.
      )
    )
  )
)

echo.
echo   ------------------------------------------------------------------
echo     Build complete. Start everything with:  start-services.cmd
echo   ------------------------------------------------------------------
echo.
goto :done

:fail
echo.
echo   Build aborted.
echo.
pause
exit /b 1

:done
endlocal
exit /b 0
