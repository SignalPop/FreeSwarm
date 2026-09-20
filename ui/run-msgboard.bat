@echo off
setlocal
rem Agent coordination bus: messages, task queue with atomic claim, shared blackboard.
rem Shares the control plane's token issuer, so one login covers both services.

set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [!] venv not found at %PY%
  exit /b 1
)

cd /d "%ROOT%\ui\backend"
echo Message board -^> http://127.0.0.1:8100
"%PY%" -m uvicorn app.msgboard:app --host 127.0.0.1 --port 8100
