$ErrorActionPreference = 'Stop'

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RuntimeDir = Join-Path $ProjectDir 'outputs\surge_organic_ablation_v8\runtime_logs'
$StdoutLog = Join-Path $RuntimeDir 'game_mode_stdout.log'
$StderrLog = Join-Path $RuntimeDir 'game_mode_stderr.log'
$LauncherPidPath = Join-Path $RuntimeDir 'game_mode_launcher.pid'

New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
Set-Content -LiteralPath $LauncherPidPath -Value $PID -Encoding ascii

# Ryzen 9 7950X3D logical CPUs 16-31: the lower-cache CCD requested for PUBG coexistence.
$self = Get-Process -Id $PID
$self.ProcessorAffinity = [IntPtr][Int64]4294901760
$self.PriorityClass = 'BelowNormal'

# Seal CUDA for this process and every spawned Python/LightGBM worker.
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:NVIDIA_VISIBLE_DEVICES = 'none'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:NUMEXPR_NUM_THREADS = '2'

Push-Location $ProjectDir
try {
    & python -u 'tools\run_v8_game_mode_resume.py' --workers 8 --threads-per-worker 2 --progress-every 25 1>> $StdoutLog 2>> $StderrLog
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
