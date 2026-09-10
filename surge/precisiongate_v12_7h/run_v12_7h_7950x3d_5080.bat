@echo off
setlocal
set "ROOT=C:\Users\minsu\Desktop\New Project"
set "PKG=%~dp0"

REM 7950X3D: leave several logical threads for Windows / file IO.
set OMP_NUM_THREADS=24
set MKL_NUM_THREADS=24
set OPENBLAS_NUM_THREADS=24
set NUMEXPR_NUM_THREADS=24
set PYTHONUNBUFFERED=1

python "%PKG%run_surge_precision_gate_v12_7h.py" ^
  --package-root "%ROOT%" ^
  --v10-output "%ROOT%\CrashWatch_Surge_Tickerwise_Correlation_Map_V10\outputs\surge_tickerwise_correlation_map_v10" ^
  --v10-2-output "%ROOT%\CrashWatch_Surge_Tickerwise_Correlation_Map_V10_2_Patch_20260815\outputs\surge_tickerwise_correlation_map_v10_2" ^
  --dataset "%ROOT%\CrashWatch_Surge_3D5_Reference_Package\data\training_dataset_finance11h.parquet" ^
  --target-sidecar "%ROOT%\CrashWatch_Surge_3D5_Reference_Package\data\surge_target_3d5.parquet" ^
  --folds "%ROOT%\CrashWatch_Surge_Correlation_Map_Complete_V2_Patch_20260809\outputs\surge_correlation_map_complete\walk_forward_folds.json" ^
  --feature-profile-manifest "%ROOT%\CrashWatch_Surge_AllFeature_Ablation_V3_Patch_20260809\outputs\surge_pre_model_gate_v3\surge_feature_profiles_corrected.json" ^
  --feature-profile P0_ALL_VALID ^
  --output "%ROOT%\outputs\surge_precision_gate_v12_7h" ^
  --target-hours 7.0 ^
  --cpu-threads 24 ^
  --require-gpu ^
  --search-seeds-per-family 3 ^
  --top-families 8 ^
  --robust-seeds 13 ^
  --max-ab-features-per-ticker 24 ^
  --minimum-alerts 30 ^
  --target-precision 0.70 ^
  --resume

if errorlevel 1 (
  echo.
  echo V12-7H FAILED. Check RUN_STATUS.json and GPU_PREFLIGHT.json.
  exit /b 1
)

python "%PKG%verify_surge_precision_gate_v12_7h.py" --output "%ROOT%\outputs\surge_precision_gate_v12_7h"
endlocal
