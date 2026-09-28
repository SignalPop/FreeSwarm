@echo off
setlocal
rem One data/action MCP (task server) from the mcp\ folder, over HTTP with OAuth:
rem
rem   run-mcp.bat <server folder>        e.g.  run-mcp.bat ..\mcp\test\battery
rem
rem The folder holds server.py and make_oauth_secrets.py (mcp\README.md). The port is the
rem server's own default (gex 8200, battery 8201, tables 8202). On the first start the server's
rem OAuth secrets are created and it is registered in ui\backend\mcp_servers.json (http + oauth);
rem connect it once from the console (Connectors -> Connect) with the approval passphrase saved in
rem <folder>\.oauth\approval_passphrase.txt. start-services.cmd starts every such folder.

set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"
if "%~1"=="" (
  echo usage: run-mcp.bat ^<mcp server folder^>
  exit /b 1
)
set "DIR=%~f1"
set "NAME=%~nx1"

if not exist "%PY%" (
  echo [!] venv not found at %PY%
  exit /b 1
)
if not exist "%DIR%\server.py" (
  echo [!] %DIR% has no server.py -- nothing to serve
  exit /b 1
)
if not exist "%DIR%\.oauth\server.json" (
  if not exist "%DIR%\make_oauth_secrets.py" (
    echo [!] %DIR% has no OAuth secrets and no make_oauth_secrets.py to create them
    exit /b 1
  )
  echo First start: creating OAuth secrets and registering %NAME% with the control plane...
  "%PY%" "%DIR%\make_oauth_secrets.py" --quiet
  if errorlevel 1 exit /b 1
)

title FreeSwarm MCP %NAME%
cd /d "%DIR%"
echo Data/action MCP %NAME% (%DIR%) over HTTP with OAuth
"%PY%" "%DIR%\server.py" --http
