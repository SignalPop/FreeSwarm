@echo off
setlocal
rem Swarm task runner: registers one board agent per resident model and answers queued
rem tasks through it. Without this the board's task queue has no consumer -- tasks sit
rem "open" forever, because /mb/tasks/claim is pull-based and nothing was calling it.
rem
rem Safe to start before the control plane or before any model is loaded: it polls and
rem picks models up as they appear.

set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [!] venv not found at %PY%
  exit /b 1
)

cd /d "%ROOT%\ui\backend"
echo Swarm runner -^> claiming tasks from http://127.0.0.1:8100
"%PY%" swarm_runner.py
