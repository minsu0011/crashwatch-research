@echo off
setlocal
cd /d "%~dp0\.."
python starter_code\rebuild_tickerwise_hierarchy_v10_1.py --package-root . --v10-output outputs\surge_tickerwise_correlation_map_v10 --output outputs\surge_tickerwise_correlation_map_v10_1 --prior-strength-grid 20,40,80,120
if errorlevel 1 goto :fail
python starter_code\verify_surge_tickerwise_correlation_map_v10_1.py --output outputs\surge_tickerwise_correlation_map_v10_1
if errorlevel 1 goto :fail
exit /b 0
:fail
echo.
echo V10.1 rebuild failed.
pause
exit /b 1
