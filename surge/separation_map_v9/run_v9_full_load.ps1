$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BaseRoot = Split-Path -Parent $ProjectRoot
$OutputRoot = Join-Path $ProjectRoot 'outputs\surge_separation_map_v9'
$RuntimeRoot = Join-Path $ProjectRoot 'runtime_logs'
$PythonExe = (Get-Command python -ErrorAction Stop).Source

$Dataset = Join-Path $BaseRoot 'CrashWatch_Surge_3D5_Reference_Package\data\training_dataset_finance11h.parquet'
$Target = Join-Path $BaseRoot 'CrashWatch_Surge_3D5_Reference_Package\data\surge_target_3d5.parquet'
$Folds = Join-Path $BaseRoot 'CrashWatch_Surge_Correlation_Map_Complete_V2_Patch_20260809\outputs\surge_correlation_map_complete\walk_forward_folds.json'
$Profile = Join-Path $BaseRoot 'CrashWatch_Surge_AllFeature_Ablation_V3_Patch_20260809\outputs\surge_pre_model_gate_v3\surge_feature_profiles_corrected.json'
$Correlation = Join-Path $BaseRoot 'CrashWatch_Surge_Correlation_Map_Complete_V2_Patch_20260809\outputs\surge_correlation_map_complete\cluster_basis_combined_abs.csv.gz'
$V8 = Join-Path $BaseRoot 'CrashWatch_Surge_Organic_Ablation_V8\outputs\surge_organic_ablation_v8'
$BaseOof = Join-Path $BaseRoot 'CrashWatch_Surge_Precision70_V6_Patch_20260811\outputs\surge_precision70_v6\precision_candidate_predictions.npz'

New-Item -ItemType Directory -Force -Path $OutputRoot,$RuntimeRoot | Out-Null
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$StdoutLog = Join-Path $RuntimeRoot "separation_v9_full_$Stamp.stdout.log"
$StderrLog = Join-Path $RuntimeRoot "separation_v9_full_$Stamp.stderr.log"

$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:NVIDIA_VISIBLE_DEVICES = '0'

$Arguments = @(
    'starter_code\run_surge_separation_map_v9.py',
    '--package-root', "`"$ProjectRoot`"",
    '--dataset', "`"$Dataset`"",
    '--target-sidecar', "`"$Target`"",
    '--folds', "`"$Folds`"",
    '--feature-profile-manifest', "`"$Profile`"",
    '--feature-profile', 'P0_ALL_VALID',
    '--correlation-matrix', "`"$Correlation`"",
    '--v8-dir', "`"$V8`"",
    '--base-oof', "`"$BaseOof`"",
    '--output', "`"$OutputRoot`"",
    '--base-backends', 'lightgbm_cpu,xgboost_gpu',
    '--device', 'cuda',
    '--target-precision', '0.70',
    '--minimum-alerts', '30',
    '--error-top-quantile', '0.80',
    '--error-low-quantile', '0.50',
    '--controls-per-case', '3',
    '--minimum-positive-rows', '30',
    '--minimum-negative-rows', '60',
    '--minimum-coverage', '0.50',
    '--minimum-orientation-consistency', '0.80',
    '--interaction-top-per-fold', '2500',
    '--interaction-validate-count', '256',
    '--interaction-prefilter-count', '64',
    '--minimum-interaction-fold-presence', '0.50',
    '--prefilter-feature-count', '18',
    '--axis-candidate-count', '24',
    '--horizon-candidate-count', '12',
    '--pair-candidate-count', '16',
    '--threads', '32',
    '--xgboost-threads', '8',
    '--analysis-workers', '1',
    '--pair-workers', '8',
    '--resume'
)

$StartedAt = Get-Date
$Process = Start-Process -FilePath $PythonExe -ArgumentList $Arguments -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog -WindowStyle Hidden -PassThru
try { $Process.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::AboveNormal } catch {}
try { $Process.ProcessorAffinity = [IntPtr]::new([long]4294967295) } catch {}

$Active = [ordered]@{
    schema = 'crashwatch_surge_separation_v9_full_load_launcher_v1'
    launched_at = $StartedAt.ToString('o')
    pid = $Process.Id
    priority_class = $Process.PriorityClass.ToString()
    processor_affinity = 'logical processors 0-31 (all)'
    compute_plan = 'single deterministic feature-map path; 8 spawned processes for pair validation'
    gpu_plan = 'CUDA device 0 available; V6 forward OOF cache avoids redundant model retraining'
    ram_plan = '96GB available; process-local pair contexts fit in memory'
    feature_count = 439
    benchmark_48_feature_seconds = 87.691
    output = $OutputRoot
    stdout_log = $StdoutLog
    stderr_log = $StderrLog
    command_arguments = $Arguments
}
$Active | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $RuntimeRoot 'ACTIVE_RUN.json') -Encoding UTF8
$Active | ConvertTo-Json -Depth 6
