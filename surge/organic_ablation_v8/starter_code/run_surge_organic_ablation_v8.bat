@echo off
setlocal
cd /d "%~dp0\.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_surge_organic_ablation_v8.ps1"
set EXITCODE=%ERRORLEVEL%
echo.
if not "%EXITCODE%"=="0" echo V8 failed with exit code %EXITCODE%.
pause
exit /b %EXITCODE%
