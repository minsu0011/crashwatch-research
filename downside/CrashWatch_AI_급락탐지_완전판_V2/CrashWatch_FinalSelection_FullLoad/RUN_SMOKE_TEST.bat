@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
python smoke_test.py
set EXITCODE=%ERRORLEVEL%
pause
exit /b %EXITCODE%
