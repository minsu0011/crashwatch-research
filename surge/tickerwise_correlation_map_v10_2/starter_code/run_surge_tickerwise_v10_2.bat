@echo off
setlocal
cd /d "%~dp0.."
python starter_code\run_surge_tickerwise_validation_v10_2.py --package-root . --v10-output outputs\surge_tickerwise_correlation_map_v10 --v10-1-output outputs\surge_tickerwise_correlation_map_v10_1 --device cuda --strict-backend --base-backends lightgbm_cpu,xgboost_gpu --probe-backends lightgbm_cpu,xgboost_gpu --prior-strength-grid 20,40,80,120 --probe-top-tickers 8 --probe-discovery-folds 0,1,2 --probe-evaluation-folds 3,4,5,6,7 --run-probe --resume
if errorlevel 1 (
  echo.
  echo V10.2 failed. Check outputs\surge_tickerwise_correlation_map_v10_2\RUN_STATUS.json
  pause
  exit /b 1
)
python starter_code\verify_surge_tickerwise_validation_v10_2.py --output outputs\surge_tickerwise_correlation_map_v10_2
pause
