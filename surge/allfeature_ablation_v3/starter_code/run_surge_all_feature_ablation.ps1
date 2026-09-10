param(
    [ValidateSet("gate", "lightgbm", "xgboost_gpu", "both", "dry_run")]
    [string]$Mode = "lightgbm",
    [int]$Workers = 5,
    [int]$ThreadsPerWorker = 3,
    [int]$XgboostThreads = 1,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Starter = $PSScriptRoot
$Root = (Resolve-Path (Join-Path $Starter "..")).Path
Set-Location $Root

$CorrelationDir = Join-Path $Root "outputs\surge_correlation_map_complete"
$GateDir = Join-Path $Root "outputs\surge_pre_model_gate_v3"

& $Python (Join-Path $Starter "run_surge_pre_model_gate.py") `
    --package-root $Root `
    --correlation-dir $CorrelationDir `
    --output $GateDir `
    --resume
if ($LASTEXITCODE -ne 0) {
    throw "Pre-model gate failed. $GateDir\RUN_STATUS.json 확인"
}

if ($Mode -eq "gate") {
    Write-Host "Pre-model gate completed: $GateDir"
    exit 0
}

switch ($Mode) {
    "lightgbm" {
        $Backends = "lightgbm_cpu"
        $OutputDir = Join-Path $Root "outputs\surge_all_feature_ablation_v3_lightgbm"
    }
    "xgboost_gpu" {
        $Backends = "xgboost_gpu"
        $OutputDir = Join-Path $Root "outputs\surge_all_feature_ablation_v3_xgboost_gpu"
    }
    "both" {
        $Backends = "lightgbm_cpu,xgboost_gpu"
        $OutputDir = Join-Path $Root "outputs\surge_all_feature_ablation_v3_both"
    }
    "dry_run" {
        $Backends = "lightgbm_cpu"
        $OutputDir = Join-Path $Root "outputs\surge_all_feature_ablation_v3_dry_run"
    }
}

$Arguments = @(
    (Join-Path $Starter "run_surge_all_feature_ablation.py"),
    "--package-root", $Root,
    "--correlation-dir", $CorrelationDir,
    "--pre-model-dir", $GateDir,
    "--output", $OutputDir,
    "--backends", $Backends,
    "--stages", "baseline,profiles,feature_loo",
    "--workers", "$Workers",
    "--threads-per-worker", "$ThreadsPerWorker",
    "--xgboost-threads", "$XgboostThreads",
    "--resume"
)
if ($Mode -eq "dry_run") {
    $Arguments += "--dry-run"
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Ablation failed. $OutputDir\RUN_STATUS.json 및 failed_tasks.csv 확인"
}

Write-Host "Completed: $OutputDir"
