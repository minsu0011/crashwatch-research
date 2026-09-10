$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$outputDir = Join-Path $projectRoot "outputs\surge_tickerwise_correlation_map_v10"
$logDir = Join-Path $projectRoot "runtime_logs"
$consoleLog = Join-Path $logDir "v10_full_load.console.log"
$timingFile = Join-Path $logDir "v10_full_load.timing.json"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Set-Location $projectRoot

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "16"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONWARNINGS = "ignore"

$started = Get-Date
$arguments = @(
    ".\starter_code\run_surge_tickerwise_correlation_map_v10.py",
    "--package-root", ".",
    "--dataset", "C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_3D5_Reference_Package\data\training_dataset_finance11h.parquet",
    "--target-sidecar", "C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_3D5_Reference_Package\data\surge_target_3d5.parquet",
    "--folds", "C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_Correlation_Map_Complete_V2_Patch_20260809\outputs\surge_correlation_map_complete\walk_forward_folds.json",
    "--feature-profile-manifest", "C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_AllFeature_Ablation_V3_Patch_20260809\outputs\surge_pre_model_gate_v3\surge_feature_profiles_corrected.json",
    "--feature-profile", "P0_ALL_VALID",
    "--output", $outputDir,
    "--base-backends", "lightgbm_cpu,xgboost_gpu",
    "--device", "cuda",
    "--strict-backend",
    "--require-full-439",
    "--threads", "24",
    "--xgboost-threads", "8",
    "--model-workers", "1",
    "--parallel-backends",
    "--analysis-workers", "1",
    "--create-plots",
    "--save-full-matrices",
    "--no-run-probe"
)

& python -u @arguments 2>&1 | Tee-Object -FilePath $consoleLog
$exitCode = $LASTEXITCODE
$ended = Get-Date

[ordered]@{
    exit_code = $exitCode
    started_at_kst = $started.ToString("yyyy-MM-dd HH:mm:ss zzz")
    ended_at_kst = $ended.ToString("yyyy-MM-dd HH:mm:ss zzz")
    elapsed_seconds = [math]::Round(($ended - $started).TotalSeconds, 3)
    output = $outputDir
    console_log = $consoleLog
} | ConvertTo-Json | Set-Content -LiteralPath $timingFile -Encoding UTF8

exit $exitCode
