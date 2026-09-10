@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=C:\Users\minsu\Documents\New Project"
set "V10_OUT=%ROOT%\CrashWatch_Surge_Tickerwise_Correlation_Map_V10\outputs\surge_tickerwise_correlation_map_v10"
set "V10_2_OUT=%ROOT%\CrashWatch_Surge_Tickerwise_Correlation_Map_V10_2_Patch_20260815\outputs\surge_tickerwise_correlation_map_v10_2"
python run_surge_magnitude_direction_v13.py ^
  --package-root "%ROOT%" ^
  --v10-output "%V10_OUT%" ^
  --v10-2-output "%V10_2_OUT%" ^
  --dataset "%ROOT%\CrashWatch_Surge_3D5_Reference_Package\data\training_dataset_finance11h.parquet" ^
  --target-sidecar "%ROOT%\CrashWatch_Surge_3D5_Reference_Package\data\surge_target_3d5.parquet" ^
  --folds "%ROOT%\CrashWatch_Surge_Correlation_Map_Complete_V2_Patch_20260809\outputs\surge_correlation_map_complete\walk_forward_folds.json" ^
  --feature-profile-manifest "%ROOT%\CrashWatch_Surge_AllFeature_Ablation_V3_Patch_20260809\outputs\surge_pre_model_gate_v3\surge_feature_profiles_corrected.json" ^
  --output "%ROOT%\outputs\surge_magnitude_direction_v13_smoke" ^
  --target-hours 0.10 ^
  --expected-target-valid-rows 91775 ^
  --expected-ticker-count 48 ^
  --cpu-threads 12 ^
  --fast-mode ^
  --no-require-gpu ^
  --resume
endlocal
