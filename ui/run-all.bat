@echo off
rem Launch the services in their own windows.
setlocal
set "HERE=%~dp0"
start "FreeToken control plane" cmd /k "%HERE%run-control-plane.bat"
start "FreeToken message board" cmd /k "%HERE%run-msgboard.bat"
rem Every data/action MCP under mcp\ (a folder with server.py and make_oauth_secrets.py), serially
rem prepared: a first start creates its OAuth secrets (see run-mcp.bat / start-services.cmd).
for /d /r "%HERE%..\mcp" %%d in (*) do (
  if exist "%%d\server.py" if exist "%%d\make_oauth_secrets.py" (
    if not exist "%%d\.oauth\server.json" "%HERE%..\.venv\Scripts\python.exe" "%%d\make_oauth_secrets.py" --quiet
    start "FreeToken MCP %%~nxd" cmd /k ""%HERE%run-mcp.bat" "%%d""
  )
)
start "FreeToken console"       cmd /k "%HERE%run-frontend.bat"
echo.
echo   Console        http://localhost:3000
echo   Control plane  http://127.0.0.1:8500/docs
echo   Message board  http://127.0.0.1:8510/docs
echo   MCPs           mcp\*  ^(gex 8520, battery 8521, tables 8522^)
echo.
