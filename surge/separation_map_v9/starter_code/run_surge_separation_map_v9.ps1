$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

python starter_code/run_surge_separation_map_v9.py `
  --package-root . `
  --base-backends lightgbm_cpu,xgboost_gpu `
  --device auto `
  --allow-cpu-fallback `
  --target-precision 0.70 `
  --minimum-alerts 30 `
  --prefilter-feature-count 18 `
  --interaction-prefilter-count 64 `
  --resume

if ($LASTEXITCODE -ne 0) {
  throw "V9 failed with exit code $LASTEXITCODE"
}
