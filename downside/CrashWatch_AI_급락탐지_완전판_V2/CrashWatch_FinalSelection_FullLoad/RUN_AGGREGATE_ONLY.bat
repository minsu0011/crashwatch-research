@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
python run_full_selection.py --stages aggregate %*
set EXITCODE=%ERRORLEVEL%
pause
exit /b %EXITCODE%
