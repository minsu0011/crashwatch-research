@echo off
setlocal
cd /d "%~dp0\.."
python starter_code\run_surge_tickerwise_correlation_map_v10.py --package-root . --base-backends lightgbm_cpu,xgboost_gpu --probe-backends lightgbm_cpu,xgboost_gpu --device cuda --strict-backend --require-full-439 --run-probe
set EXIT_CODE=%ERRORLEVEL%
echo.
if not "%EXIT_CODE%"=="0" echo Execution failed with exit code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
