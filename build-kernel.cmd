@echo off
setlocal EnableDelayedExpansion
rem ===================================================================================
rem  FreeSwarm - build the engine's native extensions
rem
rem  freetoken.kernel._pinned_tensor and freetoken.kernel._cpu_moe are C++ extensions
rem  compiled by setup.py during `pip install -e .`. They are not in git, so a fresh
rem  clone (or a copied tree) has none, and `import freetoken` still succeeds without
rem  them -- the engine only fails once a model load asks for pinned memory.
rem
rem  Rebuilds when either .pyd is missing, fails to import, or is older than its sources
rem  (csrc\ or setup.py). Otherwise it is a no-op, so build-services.cmd calls it every time.
rem
rem    build-kernel.cmd              build if missing or stale
rem    build-kernel.cmd --force      always rebuild
rem
rem  Needs Visual Studio with the C++ workload (found via vswhere; override with
rem  FREETOKEN_VCVARS=<path to vcvars64.bat>) and the CUDA 13 pip wheels in the venv.
rem ===================================================================================

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "PY=%ROOT%\.venv\Scripts\python.exe"
set "KDIR=%ROOT%\python\freetoken\kernel"
set "CU13=%ROOT%\.venv\Lib\site-packages\nvidia\cu13"

set "FORCE="
set "NOPAUSE="
for %%a in (%*) do (
  if /i "%%~a"=="--force" set "FORCE=1"
  if /i "%%~a"=="--nopause" set "NOPAUSE=1"
)

echo.
echo   FreeSwarm - engine kernels
echo   ------------------------------------------------------------------

if not exist "%PY%" (
  echo   [X] no venv at %PY% - see README.md, section 1.
  goto :fail
)

"%PY%" -c "import torch" >nul 2>&1
if errorlevel 1 (
  echo   [X] torch is not installed in the venv. Install it first:
  echo         .venv\Scripts\python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
  goto :fail
)

rem ---- up to date? -----------------------------------------------------------------
if defined FORCE goto :build
call :check_current
if not errorlevel 1 (
  echo   [ok] kernels             _pinned_tensor, _cpu_moe up to date
  goto :done
)

:build
rem ---- CUDA 13 toolchain from the venv ---------------------------------------------
rem A system-wide CUDA_PATH (e.g. 12.x) must not win: _toolchain.py refuses an nvcc
rem major that differs from torch's, and setup.py links against its cudart.
if not exist "%CU13%\bin\nvcc.exe" (
  echo   [X] CUDA 13 wheels not found in the venv. Install them:
  echo         .venv\Scripts\python -m pip install nvidia-cuda-nvcc nvidia-nvvm nvidia-cuda-crt nvidia-cuda-runtime
  goto :fail
)
set "CUDA_PATH=%CU13%"
set "CUDA_HOME=%CU13%"
echo   [ok] CUDA toolchain     %CU13%

rem ---- MSVC ---------------------------------------------------------------------------
where cl >nul 2>&1
if not errorlevel 1 (
  echo   [ok] MSVC                cl.exe already on PATH
  goto :have_msvc
)

set "VCVARS=%FREETOKEN_VCVARS%"
if defined VCVARS goto :call_vcvars

rem vswhere is queried as a plain statement into a temp file, not inside for /f: the
rem "(x86)" in its path makes cmd strip the command's quotes and the lookup silently fails.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSROOT="
if exist "%VSWHERE%" (
  set "VSTMP=%TEMP%\freeswarm_vswhere_%RANDOM%.txt"
  "%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath > "!VSTMP!" 2>nul
  set /p VSROOT=<"!VSTMP!"
  del "!VSTMP!" >nul 2>&1
)
if defined VSROOT set "VCVARS=%VSROOT%\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat"

:call_vcvars
if not exist "%VCVARS%" (
  echo   [X] vcvars64.bat not found at:
  echo         %VCVARS%
  echo       Install the "Desktop development with C++" workload, or set FREETOKEN_VCVARS.
  goto :fail
)
call "%VCVARS%" >nul
where cl >nul 2>&1
if errorlevel 1 (
  echo   [X] vcvars64.bat ran but cl.exe is still not on PATH:
  echo         %VCVARS%
  goto :fail
)
echo   [ok] MSVC                %VCVARS%

:have_msvc
rem Required inside a vcvars shell, otherwise setuptools re-runs its own MSVC discovery.
set "DISTUTILS_USE_SDK=1"
rem setup.py builds with ninja; torch finds it on PATH, and pip puts it in .venv\Scripts.
set "PATH=%ROOT%\.venv\Scripts;%PATH%"
"%PY%" -c "import ninja" >nul 2>&1
if errorlevel 1 (
  echo   Installing ninja...
  "%PY%" -m pip install ninja
  if errorlevel 1 ( echo   [X] could not install ninja. & goto :fail )
)

rem ---- build ------------------------------------------------------------------------
rem A running engine (or anything that imported freetoken.kernel) holds the .pyd open,
rem and setuptools cannot overwrite it. Windows does allow renaming a loaded DLL, so move
rem the current ones aside first; they are restored if the build fails, and deleted on the
rem next run once nothing has them loaded.
set "TAG=%RANDOM%%RANDOM%"
powershell -NoProfile -Command "Get-ChildItem '%KDIR%' -Filter '*.pyd.stale-*' | Remove-Item -ErrorAction SilentlyContinue; Get-ChildItem '%KDIR%' -File | Where-Object Extension -eq '.pyd' | ForEach-Object { Rename-Item $_.FullName ($_.Name + '.stale-%TAG%') }"

echo.
echo   Building _pinned_tensor and _cpu_moe ^(pip install -e .^)...
pushd "%ROOT%"
"%PY%" -m pip install -e . --no-build-isolation --no-deps
set "RC=%errorlevel%"
popd
if not "%RC%"=="0" (
  powershell -NoProfile -Command "Get-ChildItem '%KDIR%' -Filter '*.pyd.stale-%TAG%' | ForEach-Object { $orig = $_.FullName -replace '\.stale-%TAG%$',''; if (-not (Test-Path $orig)) { Rename-Item $_.FullName (Split-Path $orig -Leaf) } }"
  echo.
  echo   [X] The kernel build failed - see the compiler output above.
  echo       The previous kernels, if any, were left in place.
  goto :fail
)
powershell -NoProfile -Command "Get-ChildItem '%KDIR%' -Filter '*.pyd.stale-*' | Remove-Item -ErrorAction SilentlyContinue; if (Get-ChildItem '%KDIR%' -Filter '*.pyd.stale-*') { exit 1 }"
if errorlevel 1 (
  echo   [note] A running engine still has the old kernels loaded. Restart it ^(Unload,
  echo          then load again^) to pick up the new build.
)

rem setuptools copies the .pyd out of build\ with its original mtime, so an incremental
rem build that ninja considered current would still look stale to :check_current.
powershell -NoProfile -Command "Get-ChildItem '%KDIR%' -File | Where-Object Extension -eq '.pyd' | ForEach-Object { $_.LastWriteTime = Get-Date }"

call :check_current
if errorlevel 1 (
  echo   [X] Build reported success but the extensions still do not import:
  "%PY%" -c "import freetoken.kernel._pinned_tensor, freetoken.kernel._cpu_moe"
  goto :fail
)
echo   [ok] kernels             built into python\freetoken\kernel
goto :done

rem ---------------------------------------------------------------------------------
rem errorlevel 0 = both extensions import and neither is older than csrc\ or setup.py.
:check_current
"%PY%" -c "import freetoken.kernel._pinned_tensor, freetoken.kernel._cpu_moe" >nul 2>&1
if errorlevel 1 exit /b 1
powershell -NoProfile -Command "$pyd = Get-ChildItem '%KDIR%' -File | Where-Object Extension -eq '.pyd' | Sort-Object LastWriteTime | Select-Object -First 1; $src = @(Get-ChildItem '%KDIR%\csrc' -Recurse -File) + @(Get-Item '%ROOT%\setup.py') | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if (-not $pyd -or $src.LastWriteTime -gt $pyd.LastWriteTime) { exit 1 } else { exit 0 }"
exit /b %errorlevel%

:fail
echo.
echo   Kernel build aborted.
echo.
if not defined NOPAUSE pause
exit /b 1

:done
endlocal
exit /b 0
