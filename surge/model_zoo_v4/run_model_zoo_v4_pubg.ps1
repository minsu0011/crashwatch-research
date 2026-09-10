param()

$ErrorActionPreference = 'Stop'

$ProjectRoot = 'C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_Model_Zoo_V4_Patch_20260810'
$PythonExe = 'C:\Users\minsu\anaconda3\python.exe'
$Dataset = 'C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_3D5_Reference_Package\data\training_dataset_finance11h.parquet'
$Target = 'C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_3D5_Reference_Package\data\surge_target_3d5.parquet'
$Folds = 'C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_Correlation_Map_Complete_V2_Patch_20260809\outputs\surge_correlation_map_complete\walk_forward_folds.json'
$Profiles = 'C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_AllFeature_Ablation_V3_Patch_20260809\outputs\surge_pre_model_gate_v3\surge_feature_profiles_corrected.json'
$Recipes = Join-Path $ProjectRoot 'starter_code\default_model_zoo_recipes.json'
$Output = Join-Path $ProjectRoot 'outputs\surge_model_zoo_v4'
$LogDirectory = Join-Path $ProjectRoot 'runtime_logs'

New-Item -ItemType Directory -Path $Output -Force | Out-Null
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null

$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$StdoutLog = Join-Path $LogDirectory "model_zoo_v4_pubg_${Timestamp}.stdout.log"
$StderrLog = Join-Path $LogDirectory "model_zoo_v4_pubg_${Timestamp}.stderr.log"
$MetadataPath = Join-Path $LogDirectory 'ACTIVE_RUN.json'

$env:OMP_NUM_THREADS = '16'
$env:MKL_NUM_THREADS = '16'
$env:OPENBLAS_NUM_THREADS = '16'
$env:NUMEXPR_NUM_THREADS = '16'
$env:VECLIB_MAXIMUM_THREADS = '16'
$env:PYTHONUNBUFFERED = '1'

$Arguments = @(
    'starter_code\run_surge_model_zoo_v4.py',
    '--package-root', ('"{0}"' -f $ProjectRoot),
    '--dataset', ('"{0}"' -f $Dataset),
    '--target-sidecar', ('"{0}"' -f $Target),
    '--folds', ('"{0}"' -f $Folds),
    '--profiles', ('"{0}"' -f $Profiles),
    '--recipes', ('"{0}"' -f $Recipes),
    '--output', ('"{0}"' -f $Output),
    '--screen-seeds', '17,29',
    '--final-seeds', '17,29,41,73,101',
    '--finalist-count', '8',
    '--device', 'cpu',
    '--threads-per-model', '16',
    '--xgboost-threads', '16',
    '--target-recall', '0.70',
    '--threshold-recall-buffer', '0.05',
    '--minimum-oof-precision-lift', '1.05',
    '--minimum-holdout-precision-lift', '1.10',
    '--max-alert-rate', '0.65',
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
    schema = 'crashwatch_surge_model_zoo_v4_pubg_launcher_v1'
    launched_at = (Get-Date).ToString('o')
    pid = $Process.Id
    priority_class = $Process.PriorityClass.ToString()
    processor_affinity_hex = '0xFFFF0000'
    logical_processors = '16-31'
    model_threads = 16
    gpu_policy = 'device=cpu; experiment GPU use intentionally near 0% for PUBG frame protection'
    output = $Output
    stdout_log = $StdoutLog
    stderr_log = $StderrLog
    command_arguments = $Arguments
}
$Metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $MetadataPath -Encoding UTF8
$Metadata | ConvertTo-Json -Depth 4
