@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM ===== CrashWatch V14: 9-hour full-load run for 7950X3D / RTX 5080 / 96GB =====
set "ROOT=C:\Users\minsu\Documents\New Project"
set "V10_OUT=%ROOT%\CrashWatch_Surge_Tickerwise_Correlation_Map_V10\outputs\surge_tickerwise_correlation_map_v10"
set "V10_2_OUT=%ROOT%\CrashWatch_Surge_Tickerwise_Correlation_Map_V10_2_Patch_20260815\outputs\surge_tickerwise_correlation_map_v10_2"
set "V13_OUT=%ROOT%\outputs\surge_magnitude_direction_v13"
set "DATASET=%ROOT%\CrashWatch_Surge_3D5_Reference_Package\data\training_dataset_finance11h.parquet"
set "TARGET=%ROOT%\CrashWatch_Surge_3D5_Reference_Package\data\surge_target_3d5.parquet"
set "FOLDS=%ROOT%\CrashWatch_Surge_Correlation_Map_Complete_V2_Patch_20260809\outputs\surge_correlation_map_complete\walk_forward_folds.json"
set "PROFILE=%ROOT%\CrashWatch_Surge_AllFeature_Ablation_V3_Patch_20260809\outputs\surge_pre_model_gate_v3\surge_feature_profiles_corrected.json"
set "OUT=%ROOT%\outputs\surge_competingrisk_hardfp_v14"

set OMP_NUM_THREADS=32
set MKL_NUM_THREADS=32
set OPENBLAS_NUM_THREADS=32
set NUMEXPR_MAX_THREADS=32
set PYTHONHASHSEED=14014

python run_surge_competingrisk_hardfp_v14.py ^
  --package-root "%ROOT%" ^
  --v10-output "%V10_OUT%" ^
  --v10-2-output "%V10_2_OUT%" ^
  --v13-output "%V13_OUT%" ^
  --dataset "%DATASET%" ^
  --target-sidecar "%TARGET%" ^
  --folds "%FOLDS%" ^
  --feature-profile-manifest "%PROFILE%" ^
  --feature-profile P0_ALL_VALID ^
  --output "%OUT%" ^
  --target-hours 9.0 ^
  --cpu-threads 32 ^
  --fold-workers 4 ^
  --expected-target-valid-rows 91775 ^
  --expected-ticker-count 48 ^
  --require-full-439 ^
  --require-full-base-oof ^
  --require-gpu ^
  --minimum-alerts 30 ^
  --target-precision 0.70 ^
  --robust-seeds 5 ^
  --max-stability-seeds 181 ^
  --resume

if errorlevel 1 (
  echo V14 FAILED. Check %OUT%\RUN_STATUS.json and runtime logs.
  exit /b 1
)
echo V14 COMPLETE: %OUT%
endlocal
