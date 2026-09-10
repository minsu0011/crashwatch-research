@echo off
setlocal
cd /d "%~dp0.."
where py >nul 2>nul
if %errorlevel%==0 (
  set "PYTHON_CMD=py -3"
) else (
  set "PYTHON_CMD=python"
)

%PYTHON_CMD% starter_code\run_surge_alert_budget_v5.py ^
  --package-root . ^
  --rank-seeds 17,29,41 ^
  --methods equal_recipe_rank,equal_family_rank,optimized_family_rank,hard_negative_meta_lgb ^
  --target-recall 0.70 ^
  --selection-recall-buffer 0.02 ^
  --minimum-precision-lift 1.10 ^
  --max-alert-rate 0.40 ^
  --max-alerts-per-day 20 ^
  --required-selection-fold-pass-rate 1.0 ^
  --daily-fraction-grid 0.20,0.25,0.30,0.35,0.40 ^
  --diagnostic-fractions 0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60 ^
  --random-weight-samples 128 ^
  --threads-per-model 8 ^
  --xgboost-threads 8 ^
  --device auto ^
  --allow-cpu-fallback ^
  --resume

if errorlevel 1 (
  echo.
  echo V5 full run failed. Check RUN_STATUS.json.
  pause
  exit /b 1
)

%PYTHON_CMD% starter_code\verify_surge_alert_budget_v5.py --output outputs\surge_alert_budget_v5
pause
