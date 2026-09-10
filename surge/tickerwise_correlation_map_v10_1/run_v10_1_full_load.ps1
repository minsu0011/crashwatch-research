$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourceV10 = "C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_Tickerwise_Correlation_Map_V10\outputs\surge_tickerwise_correlation_map_v10"
$outputDir = Join-Path $projectRoot "outputs\surge_tickerwise_correlation_map_v10_1"
$logDir = Join-Path $projectRoot "runtime_logs"
$consoleLog = Join-Path $logDir "v10_1_full_load.console.log"
$timingFile = Join-Path $logDir "v10_1_full_load.timing.json"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Set-Location $projectRoot

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "8"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONWARNINGS = "ignore"

$started = Get-Date
$arguments = @(
    ".\starter_code\rebuild_tickerwise_hierarchy_v10_1.py",
    "--package-root", ".",
    "--v10-output", $sourceV10,
    "--output", $outputDir,
    "--prior-strength-grid", "20,40,80,120",
    "--strength-workers", "4",
    "--precision-min-selection-auc", "0.57",
    "--precision-min-selection-min-auc", "0.52",
    "--precision-min-selection-folds", "2",
    "--precision-min-effective-n", "12",
    "--precision-min-matched-concordance", "0.54",
    "--similarity-feature-count", "180",
    "--similarity-min-node-coverage", "0.75"
)

& python -u @arguments 2>&1 | Tee-Object -FilePath $consoleLog
$exitCode = $LASTEXITCODE
$ended = Get-Date

[ordered]@{
    exit_code = $exitCode
    started_at_kst = $started.ToString("yyyy-MM-dd HH:mm:ss zzz")
    ended_at_kst = $ended.ToString("yyyy-MM-dd HH:mm:ss zzz")
    elapsed_seconds = [math]::Round(($ended - $started).TotalSeconds, 3)
    source_v10 = $sourceV10
    output = $outputDir
    console_log = $consoleLog
} | ConvertTo-Json | Set-Content -LiteralPath $timingFile -Encoding UTF8

exit $exitCode
