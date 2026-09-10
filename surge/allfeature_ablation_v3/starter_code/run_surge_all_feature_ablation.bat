@echo off
setlocal EnableExtensions
chcp 65001 >nul

cd /d "%~dp0\.."

echo ==========================================================
echo CrashWatch Surge V3 - Pre-model gate + Full LightGBM LOO
echo ==========================================================

python starter_code\run_surge_pre_model_gate.py ^
  --package-root . ^
  --correlation-dir outputs\surge_correlation_map_complete ^
  --output outputs\surge_pre_model_gate_v3 ^
  --resume
if errorlevel 1 goto :failed

python starter_code\run_surge_all_feature_ablation.py ^
  --package-root . ^
  --correlation-dir outputs\surge_correlation_map_complete ^
  --pre-model-dir outputs\surge_pre_model_gate_v3 ^
  --output outputs\surge_all_feature_ablation_v3_lightgbm ^
  --backends lightgbm_cpu ^
  --stages baseline,profiles,feature_loo ^
  --workers 5 ^
  --threads-per-worker 3 ^
  --resume
if errorlevel 1 goto :failed

echo.
echo SUCCESS
echo Result: outputs\surge_all_feature_ablation_v3_lightgbm
pause
exit /b 0

:failed
echo.
echo FAILED - RUN_STATUS.json and failed_tasks.csv를 확인하세요.
pause
exit /b 1
