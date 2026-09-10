$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutputRoot = Join-Path $ProjectRoot "outputs\surge_organic_ablation_v8"
$RuntimeRoot = Join-Path $ProjectRoot "runtime_logs"
$PythonExe = (Get-Command python -ErrorAction Stop).Source
$Dataset = "C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_3D5_Reference_Package\data\training_dataset_finance11h.parquet"
$Target = "C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_3D5_Reference_Package\data\surge_target_3d5.parquet"
$Folds = "C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_Correlation_Map_Complete_V2_Patch_20260809\outputs\surge_correlation_map_complete\walk_forward_folds.json"
$Correlation = "C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_Correlation_Map_Complete_V2_Patch_20260809\outputs\surge_correlation_map_complete"
$PreModel = "C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_AllFeature_Ablation_V3_Patch_20260809\outputs\surge_pre_model_gate_v3"

New-Item -ItemType Directory -Path $OutputRoot,$RuntimeRoot -Force | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$StdoutLog = Join-Path $RuntimeRoot "organic_v8_full_$Stamp.stdout.log"
$StderrLog = Join-Path $RuntimeRoot "organic_v8_full_$Stamp.stderr.log"

# Nine 3-thread LightGBM workers plus five 1-thread CUDA workers use all 32
# logical processors while keeping five independent GPU queues in flight.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "3"
$env:OMP_DYNAMIC = "FALSE"
$env:OMP_WAIT_POLICY = "ACTIVE"
$env:MKL_NUM_THREADS = "3"
$env:MKL_DYNAMIC = "FALSE"
$env:OPENBLAS_NUM_THREADS = "3"
$env:NUMEXPR_NUM_THREADS = "3"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:NVIDIA_VISIBLE_DEVICES = "0"

$Arguments = @(
    "starter_code\run_surge_organic_ablation_v8.py",
    "--package-root", "`"$ProjectRoot`"",
    "--dataset", "`"$Dataset`"",
    "--target-sidecar", "`"$Target`"",
    "--folds", "`"$Folds`"",
    "--correlation-dir", "`"$Correlation`"",
    "--pre-model-dir", "`"$PreModel`"",
    "--output", "`"$OutputRoot`"",
    "--backends", "lightgbm_cpu,xgboost_gpu",
    "--seeds", "17",
    "--stages", "all",
    "--cluster-thresholds", "0.80,0.90,0.92,0.95,0.98",
    "--primary-cluster-threshold", "0.92",
    "--pair-threshold", "0.92",
    "--neighborhood-ks", "1,3,5",
    "--error-top-quantile", "0.80",
    "--error-low-quantile", "0.50",
    "--precision-target", "0.70",
    "--precision-min-alerts", "30",
    "--workers", "9",
    "--gpu-workers", "5",
    "--threads-per-worker", "3",
    "--xgboost-threads", "1",
    "--bootstrap-repetitions", "5000",
    "--progress-every", "25",
    "--retry-failed", "1",
    "--resume"
)

$StartedAt = Get-Date
$EstimatedHours = 15.0
$EstimatedFinish = $StartedAt.AddHours($EstimatedHours)
$Process = Start-Process -FilePath $PythonExe -ArgumentList $Arguments -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog -WindowStyle Hidden -PassThru

try {
    $Process.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::AboveNormal
} catch {
    # The workload still defaults to all processors if priority adjustment is unavailable.
}
try {
    $Process.ProcessorAffinity = [IntPtr]::new([long]4294967295)
} catch {
    # Default affinity already includes all available logical processors.
}

$Active = [ordered]@{
    schema = "crashwatch_surge_organic_v8_full_load_launcher_v1"
    launched_at = $StartedAt.ToString("o")
    pid = $Process.Id
    priority_class = $Process.PriorityClass.ToString()
    processor_affinity = "logical processors 0-31 (all)"
    cpu_plan = "9 LightGBM workers x 3 threads + 5 XGBoost-GPU workers x 1 thread"
    gpu_plan = "CUDA device 0, five concurrent XGBoost queues"
    ram_plan = "shared read-only 91,775x439 float32 mmap; 96GB capacity"
    planned_model_tasks = 37808
    benchmark_gpu_tasks_per_second = 0.40415782699403907
    estimated_hours = $EstimatedHours
    estimated_finish_kst = $EstimatedFinish.ToString("yyyy-MM-dd HH:mm:ss zzz")
    output = $OutputRoot
    stdout_log = $StdoutLog
    stderr_log = $StderrLog
    command_arguments = $Arguments
}
$Active | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $RuntimeRoot "ACTIVE_RUN.json") -Encoding UTF8
$Active | ConvertTo-Json -Depth 6
