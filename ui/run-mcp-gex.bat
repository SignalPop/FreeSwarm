@echo off
rem The GEX task server (mcp\gex) on http://127.0.0.1:8520/mcp -- see run-mcp.bat, which serves
rem any data/action MCP folder; start-services.cmd starts them all.
call "%~dp0run-mcp.bat" "%~dp0..\mcp\gex"
