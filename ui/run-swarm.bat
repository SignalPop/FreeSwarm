@echo off
setlocal
rem Swarm task runner: registers one board agent per resident model and answers queued
rem tasks through it. Without this the board's task queue has no consumer -- tasks sit
rem "open" forever, because /mb/tasks/claim is pull-based and nothing was calling it.
rem
rem Safe to start before the control plane or before any model is loaded: it polls and
rem picks models up as they appear.
rem
rem To restart it without losing in-flight iterations use ui\restart-swarm.cmd: it drains the
rem runner (no new work starts; running iterations finish; it then exits 0) and starts it again.

set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [!] venv not found at %PY%
  exit /b 1
)

cd /d "%ROOT%\ui\backend"
echo Swarm runner -^> claiming tasks from http://127.0.0.1:8510
rem The launch and everything after it is ONE parenthesised block ending in exit /b, so cmd has
rem parsed all of it before the runner starts and never reads this file again. cmd runs a batch
rem file by re-opening it after each command and seeking to the byte offset where it left off:
rem when this file was edited while a runner ran (10-01 20:12), killing that runner (restart
rem --hard, 20:32) made its old window resume mid-file in the NEW text and start a SECOND runner.
rem Keep it the last thing in the file. A window started from an older copy of this file is still
rem exposed, so ui\swarm-drain.ps1 closes runner windows BEFORE it kills or drains the runner.
(
  "%PY%" swarm_runner.py
  echo.
  call echo Swarm runner exited, code %%ERRORLEVEL%%.
  exit /b
)
