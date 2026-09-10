@echo off
setlocal
cd /d "%~dp0.."
python starter_code\run_surge_hardfp_v7.py --package-root . --quick --base-seeds 17 --device cpu --resume
set EXITCODE=%ERRORLEVEL%
echo.
echo Exit code: %EXITCODE%
pause
exit /b %EXITCODE%
