@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
python run_sealed_preflight.py %*
set EXITCODE=%ERRORLEVEL%
pause
exit /b %EXITCODE%
