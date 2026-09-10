@echo off
setlocal
REM Edit these paths if your project folders differ.
set ROOT=C:\Users\minsu\Desktop\New Project
python "%~dp0run_surge_precision_gate_v12.py" ^
  --package-root "%ROOT%" ^
  --v10-output "%ROOT%\CrashWatch_Surge_Tickerwise_Correlation_Map_V10\outputs\surge_tickerwise_correlation_map_v10" ^
  --v10-2-output "%ROOT%\CrashWatch_Surge_Tickerwise_Correlation_Map_V10_2_Patch_20260815\outputs\surge_tickerwise_correlation_map_v10_2" ^
  --dataset "%ROOT%\CrashWatch_Surge_3D5_Reference_Package\data\training_dataset_finance11h.parquet" ^
  --target-sidecar "%ROOT%\CrashWatch_Surge_3D5_Reference_Package\data\surge_target_3d5.parquet" ^
  --folds "%ROOT%\CrashWatch_Surge_Correlation_Map_Complete_V2_Patch_20260809\outputs\surge_correlation_map_complete\walk_forward_folds.json" ^
  --feature-profile-manifest "%ROOT%\CrashWatch_Surge_AllFeature_Ablation_V3_Patch_20260809\outputs\surge_pre_model_gate_v3\surge_feature_profiles_corrected.json" ^
  --feature-profile P0_ALL_VALID
