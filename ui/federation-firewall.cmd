@echo off
REM ---------------------------------------------------------------------------------------
REM  Windows Firewall rules for FreeToken LAN federation (sharing models between computers).
REM
REM    federation-firewall.cmd            add the rules
REM    federation-firewall.cmd remove     remove them
REM
REM  Two inbound rules, both limited to the PRIVATE network profile and to addresses on
REM  your LOCAL SUBNET -- so they never open anything on public Wi-Fi or to the internet:
REM
REM    TCP 8443   the federation gateway (TLS). Needed on a computer that SHARES its models.
REM    UDP 19191  discovery beacons. Needed on a computer that wants to FIND others
REM               automatically; without it, use "Connect by address" on the Network page.
REM
REM  Override the ports with FREESWARM_FED_PORT / FREESWARM_FED_DISCOVERY_PORT (set the
REM  same values for the console). Must be run as Administrator.
REM ---------------------------------------------------------------------------------------
setlocal
if "%FREESWARM_FED_PORT%"=="" (set TCPPORT=8443) else (set TCPPORT=%FREESWARM_FED_PORT%)
if "%FREESWARM_FED_DISCOVERY_PORT%"=="" (set UDPPORT=19191) else (set UDPPORT=%FREESWARM_FED_DISCOVERY_PORT%)
set RULE_TCP=FreeToken federation gateway (TCP %TCPPORT%)
set RULE_UDP=FreeToken federation discovery (UDP %UDPPORT%)

net session >nul 2>&1
if errorlevel 1 (
  echo This needs Administrator rights. Right-click the file and choose "Run as administrator",
  echo or run it from an elevated Command Prompt.
  exit /b 1
)

if /i "%~1"=="remove" (
  netsh advfirewall firewall delete rule name="%RULE_TCP%" >nul
  netsh advfirewall firewall delete rule name="%RULE_UDP%" >nul
  echo Removed the FreeToken federation firewall rules.
  exit /b 0
)

REM Replace rather than duplicate if run twice.
netsh advfirewall firewall delete rule name="%RULE_TCP%" >nul 2>&1
netsh advfirewall firewall delete rule name="%RULE_UDP%" >nul 2>&1

netsh advfirewall firewall add rule name="%RULE_TCP%" dir=in action=allow protocol=TCP localport=%TCPPORT% profile=private remoteip=localsubnet description="FreeToken: other computers on this LAN use the models you share (TLS, OAuth-approved clients only)." || exit /b 1
netsh advfirewall firewall add rule name="%RULE_UDP%" dir=in action=allow protocol=UDP localport=%UDPPORT% profile=private remoteip=localsubnet description="FreeToken: discovery beacons from other FreeToken computers on this LAN." || exit /b 1

echo.
echo Added:  %RULE_TCP%
echo         %RULE_UDP%
echo Both apply to the Private network profile and your local subnet only.
echo.
powershell -NoProfile -Command "Get-NetConnectionProfile | ForEach-Object { '  network \"' + $_.Name + '\" is ' + $_.NetworkCategory }"
echo.
echo If your network shows as Public, the rules do not apply there. To mark a trusted home or
echo office network Private: Settings ^> Network ^& internet ^> (your connection) ^> Private network.
endlocal
