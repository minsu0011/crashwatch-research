param()

$ErrorActionPreference = 'Stop'

$ProjectRoot = 'C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_HardFP_Pairwise_V7_Patch_20260811'
$PackageRoot = 'C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_Model_Zoo_V4_Patch_20260810'
$PythonExe = 'C:\Users\minsu\anaconda3\python.exe'
$Dataset = 'C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_3D5_Reference_Package\data\training_dataset_finance11h.parquet'
$Target = 'C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_3D5_Reference_Package\data\surge_target_3d5.parquet'
$Folds = 'C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_Correlation_Map_Complete_V2_Patch_20260809\outputs\surge_correlation_map_complete\walk_forward_folds.json'
$Profiles = 'C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_AllFeature_Ablation_V3_Patch_20260809\outputs\surge_pre_model_gate_v3\surge_feature_profiles_corrected.json'
$Recipes = Join-Path $ProjectRoot 'starter_code\default_hardfp_recipes_v7.json'
$Output = Join-Path $ProjectRoot 'outputs\surge_hardfp_v7'
$LogDirectory = Join-Path $ProjectRoot 'runtime_logs'

New-Item -ItemType Directory -Path $Output -Force | Out-Null
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null

$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$StdoutLog = Join-Path $LogDirectory "hardfp_v7_game_${Timestamp}.stdout.log"
$StderrLog = Join-Path $LogDirectory "hardfp_v7_game_${Timestamp}.stderr.log"
$MetadataPath = Join-Path $LogDirectory 'ACTIVE_RUN.json'

$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:NVIDIA_VISIBLE_DEVICES = 'none'
$env:OMP_NUM_THREADS = '16'
$env:OMP_DYNAMIC = 'FALSE'
$env:OMP_WAIT_POLICY = 'PASSIVE'
$env:KMP_BLOCKTIME = '0'
$env:MKL_NUM_THREADS = '16'
$env:OPENBLAS_NUM_THREADS = '16'
$env:NUMEXPR_NUM_THREADS = '16'

$Arguments = @(
    'starter_code\run_surge_hardfp_v7.py',
    '--package-root', ('"{0}"' -f $PackageRoot),
    '--output', ('"{0}"' -f $Output),
    '--dataset', ('"{0}"' -f $Dataset),
    '--target-sidecar', ('"{0}"' -f $Target),
    '--folds', ('"{0}"' -f $Folds),
    '--profiles', ('"{0}"' -f $Profiles),
    '--recipes', ('"{0}"' -f $Recipes),
    '--base-seeds', '17,29,41',
    '--selection-folds', '0,1,2,3,4',
    '--confirmation-folds', '5,6',
    '--recent-folds', '7',
    '--error-top-quantile', '0.80',
    '--error-missed-quantile', '0.50',
    '--error-feature-count', '72',
    '--minimum-contrast-score', '0.12',
    '--target-precision', '0.70',
    '--selection-precision-buffer', '0.03',
    '--minimum-precision-lcb', '0.60',
    '--minimum-alerts', '30',
    '--minimum-alert-days', '10',
    '--minimum-recall', '0.03',
    '--threads-per-model', '16',
    '--xgboost-threads', '16',
    '--device', 'cpu',
    '--resume'
)

$Process = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList $Arguments `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -WindowStyle Hidden `
    -PassThru

$Process.PriorityClass = 'BelowNormal'
$Process.ProcessorAffinity = [IntPtr]([long]4294901760)

$Metadata = [ordered]@{
    schema = 'crashwatch_surge_hardfp_v7_game_mode_launcher_v1'
    launched_at = (Get-Date).ToString('o')
    pid = $Process.Id
    priority_class = $Process.PriorityClass.ToString()
    processor_affinity_hex = '0xFFFF0000'
    logical_processors = '16-31'
    cpu_threads = 16
    gpu_policy = 'SEALED: device=cpu, CUDA_VISIBLE_DEVICES=-1, NVIDIA_VISIBLE_DEVICES=none'
    matrix_cache_source = (Join-Path $PackageRoot 'outputs\surge_model_zoo_v4\matrix_cache')
    output = $Output
    stdout_log = $StdoutLog
    stderr_log = $StderrLog
    command_arguments = $Arguments
}
$Metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $MetadataPath -Encoding UTF8
$Metadata | ConvertTo-Json -Depth 4
