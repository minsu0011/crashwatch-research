param()

$ErrorActionPreference = 'Stop'

$ProjectRoot = 'C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_Alert_Budget_V5_Patch_20260810'
$PythonExe = 'C:\Users\minsu\anaconda3\python.exe'
$V4Output = 'C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_Model_Zoo_V4_Patch_20260810\outputs\surge_model_zoo_v4'
$Dataset = 'C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_3D5_Reference_Package\data\training_dataset_finance11h.parquet'
$Target = 'C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_3D5_Reference_Package\data\surge_target_3d5.parquet'
$Folds = 'C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_Correlation_Map_Complete_V2_Patch_20260809\outputs\surge_correlation_map_complete\walk_forward_folds.json'
$Profiles = 'C:\Users\minsu\Desktop\New Project\CrashWatch_Surge_AllFeature_Ablation_V3_Patch_20260809\outputs\surge_pre_model_gate_v3\surge_feature_profiles_corrected.json'
$RankRecipes = Join-Path $ProjectRoot 'starter_code\default_alert_budget_recipes_v5.json'
$Output = Join-Path $ProjectRoot 'outputs\surge_alert_budget_v5'
$LogDirectory = Join-Path $ProjectRoot 'runtime_logs'

New-Item -ItemType Directory -Path $Output -Force | Out-Null
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null

$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$StdoutLog = Join-Path $LogDirectory "alert_budget_v5_full_${Timestamp}.stdout.log"
$StderrLog = Join-Path $LogDirectory "alert_budget_v5_full_${Timestamp}.stderr.log"
$MetadataPath = Join-Path $LogDirectory 'ACTIVE_RUN.json'

$env:PYTHONUNBUFFERED = '1'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:CUDA_MODULE_LOADING = 'LAZY'
$env:OMP_NUM_THREADS = '24'
$env:OMP_DYNAMIC = 'FALSE'
$env:MKL_NUM_THREADS = '8'
$env:OPENBLAS_NUM_THREADS = '8'
$env:NUMEXPR_NUM_THREADS = '8'

$Arguments = @(
    'starter_code\run_surge_alert_budget_v5.py',
    '--package-root', ('"{0}"' -f $ProjectRoot),
    '--v4-output', ('"{0}"' -f $V4Output),
    '--output', ('"{0}"' -f $Output),
    '--dataset', ('"{0}"' -f $Dataset),
    '--target-sidecar', ('"{0}"' -f $Target),
    '--folds', ('"{0}"' -f $Folds),
    '--profiles', ('"{0}"' -f $Profiles),
    '--rank-recipes', ('"{0}"' -f $RankRecipes),
    '--rank-seeds', '17,29,41',
    '--methods', 'equal_recipe_rank,equal_family_rank,optimized_family_rank,hard_negative_meta_lgb',
    '--target-recall', '0.70',
    '--selection-recall-buffer', '0.02',
    '--minimum-precision-lift', '1.10',
    '--max-alert-rate', '0.40',
    '--max-alerts-per-day', '20',
    '--required-selection-fold-pass-rate', '1.0',
    '--daily-fraction-grid', '0.20,0.25,0.30,0.35,0.40',
    '--diagnostic-fractions', '0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60',
    '--random-weight-samples', '128',
    '--threads-per-model', '24',
    '--xgboost-threads', '8',
    '--family-parallel',
    '--device', 'cuda',
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

$Process.PriorityClass = 'AboveNormal'
$Process.ProcessorAffinity = [IntPtr]([long]4294967295)

$Metadata = [ordered]@{
    schema = 'crashwatch_surge_alert_budget_v5_full_load_launcher_v1'
    launched_at = (Get-Date).ToString('o')
    pid = $Process.Id
    priority_class = $Process.PriorityClass.ToString()
    processor_affinity_hex = '0xFFFFFFFF'
    logical_processors = '0-31'
    lightgbm_threads = 24
    xgboost_threads = 8
    family_parallel = $true
    gpu_policy = 'CUDA required on RTX 5080'
    output = $Output
    stdout_log = $StdoutLog
    stderr_log = $StderrLog
    command_arguments = $Arguments
}
$Metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $MetadataPath -Encoding UTF8
$Metadata | ConvertTo-Json -Depth 4
