@echo off
setlocal
rem FreeToken control plane -- engine lifecycle, telemetry, chat proxy, MCP connectors.
rem Binds 127.0.0.1 by default. To expose it on the LAN, set FREESWARM_UI_HOST *and*
rem create an account (python -m app.usercli add <name>) *and* configure TLS -- the server
rem refuses an unauthenticated or plaintext non-loopback bind on purpose.

set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [!] venv not found at %PY%
  echo     Create it first - see ui\README.md
  exit /b 1
)

rem The engine subprocess needs MSVC + CUDA 13 for its runtime kernel JIT; the control
rem plane resolves both itself (app\winenv.py) and injects them when it spawns the engine,
rem so this window does not need a vcvars prompt.
set "CUDA_PATH=%ROOT%\.venv\Lib\site-packages\nvidia\cu13"
set "CUDA_HOME=%CUDA_PATH%"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"

rem GPU 0 is the WDDM display adapter; 1 and 2 are the TCC A6000s. Override as needed.
if "%FREETOKEN_VISIBLE_DEVICES%"=="" set "FREETOKEN_VISIBLE_DEVICES=1,2"

cd /d "%ROOT%\ui\backend"
echo Control plane -^> http://127.0.0.1:8000
"%PY%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
