@echo off
setlocal
cd /d "%~dp0\.."
python starter_code\verify_surge_model_zoo_v4.py --output outputs\surge_model_zoo_v4
set EXITCODE=%ERRORLEVEL%
pause
exit /b %EXITCODE%
