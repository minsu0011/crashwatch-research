$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
Set-Location $Root

python "$ScriptDir/run_surge_organic_ablation_v8.py" `
  --package-root . `
  --backends lightgbm_cpu,xgboost_gpu `
  --seeds 17 `
  --stages all `
  --cluster-thresholds 0.80,0.90,0.92,0.95,0.98 `
  --primary-cluster-threshold 0.92 `
  --pair-threshold 0.92 `
  --neighborhood-ks 1,3,5 `
  --precision-target 0.70 `
  --precision-min-alerts 30 `
  --workers 5 `
  --threads-per-worker 3 `
  --xgboost-threads 8 `
  --resume
