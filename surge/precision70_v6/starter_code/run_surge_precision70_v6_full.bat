@echo off
setlocal
cd /d "%~dp0\.."
python -m pip install -r starter_code\requirements_surge_precision70_v6.txt
if errorlevel 1 goto :error
python starter_code\run_surge_precision70_v6.py ^
  --package-root . ^
  --precision-seeds 17,29,41 ^
  --target-precision 0.70 ^
  --selection-precision-buffer 0.03 ^
  --minimum-precision-lcb 0.60 ^
  --minimum-alerts-per-fold 30 ^
  --minimum-alert-days-per-fold 10 ^
  --required-selection-fold-pass-rate 1.0 ^
  --required-holdout-fold-pass-rate 1.0 ^
  --minimum-useful-recall 0.05 ^
  --policy-kinds global_threshold,market_threshold ^
  --device auto ^
  --allow-cpu-fallback ^
  --resume
if errorlevel 1 goto :error
python starter_code\verify_surge_precision70_v6.py --output outputs\surge_precision70_v6 --allow-gate-failed
if errorlevel 1 goto :error
echo.
echo V6 finished. Check outputs\surge_precision70_v6\FINAL_RECOMMENDATION_V6.json
pause
exit /b 0
:error
echo.
echo V6 failed. Check RUN_STATUS.json and console output.
pause
exit /b 1
