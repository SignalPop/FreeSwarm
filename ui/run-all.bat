@echo off
rem Launch all three services in their own windows.
setlocal
set "HERE=%~dp0"
start "FreeToken control plane" cmd /k "%HERE%run-control-plane.bat"
start "FreeToken message board" cmd /k "%HERE%run-msgboard.bat"
start "FreeToken console"       cmd /k "%HERE%run-frontend.bat"
echo.
echo   Console        http://localhost:3000
echo   Control plane  http://127.0.0.1:8000/docs
echo   Message board  http://127.0.0.1:8100/docs
echo.
