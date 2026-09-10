@echo off
setlocal
cd /d "%~dp0.."
python starter_code\run_surge_hardfp_v7.py --package-root . --base-seeds 17,29,41 --target-precision 0.70 --selection-precision-buffer 0.03 --minimum-precision-lcb 0.60 --minimum-alerts 30 --minimum-alert-days 10 --minimum-recall 0.03 --device auto --allow-cpu-fallback --threads-per-model 8 --xgboost-threads 8 --resume
set EXITCODE=%ERRORLEVEL%
echo.
echo Exit code: %EXITCODE%
pause
exit /b %EXITCODE%
