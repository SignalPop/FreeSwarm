@echo off
setlocal
rem The GEX task server (mcp\gex): SPY 10-second GEX bars as FreeSwarm tasks -- data loading and
rem caching (polars), position management and valuation (numpy). See mcp\README.md.
rem Registered in ui\backend\mcp_servers.json as
rem   {"name": "gex", "transport": "http", "url": "http://127.0.0.1:8200/mcp"}

set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [!] venv not found at %PY%
  exit /b 1
)
if not exist "%ROOT%\mcp\gex\server.py" (
  echo [!] mcp\gex is not present on this machine -- nothing to serve
  exit /b 1
)

cd /d "%ROOT%\mcp\gex"
echo GEX task server -^> http://127.0.0.1:8200/mcp
"%PY%" server.py --http --host 127.0.0.1 --port 8200
