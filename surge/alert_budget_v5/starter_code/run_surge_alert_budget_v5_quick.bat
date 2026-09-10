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
  --quick ^
  --rank-seeds 17,29 ^
  --target-recall 0.70 ^
  --selection-recall-buffer 0.02 ^
  --minimum-precision-lift 1.10 ^
  --max-alert-rate 0.40 ^
  --max-alerts-per-day 20 ^
  --daily-fraction-grid 0.20,0.25,0.30,0.35,0.40 ^
  --resume

if errorlevel 1 (
  echo.
  echo V5 quick run failed. Check RUN_STATUS.json.
  pause
  exit /b 1
)

%PYTHON_CMD% starter_code\verify_surge_alert_budget_v5.py --output outputs\surge_alert_budget_v5
pause
