param(
    [int]$MaxAttempts = 3
)

$ErrorActionPreference = 'Stop'
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
$Batch = Join-Path $Project 'run_v14_9h_7950x3d_5080.bat'
$Output = 'C:\Users\minsu\Documents\New Project\outputs\surge_competingrisk_hardfp_v14'
$LogDir = Join-Path $Project 'runtime_logs_v14'
$Desktop = 'C:\Users\minsu\Desktop'
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
New-Item -ItemType Directory -Path $Output -Force | Out-Null
$SupervisorStatus = Join-Path $LogDir 'supervisor_status.json'
$HealthLog = Join-Path $LogDir 'watch_v14_health.jsonl'

function Write-SupervisorStatus([string]$Status, [hashtable]$Extra) {
    $payload = [ordered]@{
        status = $Status
        updated_local = (Get-Date).ToString('o')
    }
    foreach ($key in $Extra.Keys) { $payload[$key] = $Extra[$key] }
    $payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $SupervisorStatus -Encoding UTF8
}

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $stdout = Join-Path $LogDir "official_v14_attempt${attempt}_${stamp}.stdout.log"
    $stderr = Join-Path $LogDir "official_v14_attempt${attempt}_${stamp}.stderr.log"
    Write-SupervisorStatus 'STARTING' @{ attempt = $attempt; max_attempts = $MaxAttempts; stdout = $stdout; stderr = $stderr }
    $process = Start-Process -FilePath $env:ComSpec -ArgumentList @('/d','/c',"`"$Batch`"") -WorkingDirectory $Project -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    try { $process.PriorityClass = 'AboveNormal' } catch {}
    Write-SupervisorStatus 'RUNNING' @{ attempt = $attempt; pid = $process.Id; started_local = (Get-Date).ToString('o'); stdout = $stdout; stderr = $stderr }

    while (-not $process.HasExited) {
        $process.Refresh()
        $worker = $null
        try {
            $workerInfo = Get-CimInstance Win32_Process | Where-Object {
                $_.Name -match '^python' -and $_.CommandLine -match 'run_surge_competingrisk_hardfp_v14.py'
            } | Select-Object -First 1
            if ($workerInfo) { $worker = Get-Process -Id $workerInfo.ProcessId -ErrorAction SilentlyContinue }
        } catch {}
        $gpu = (& nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader 2>$null) -join ' '
        $runStatus = $null
        $runStatusPath = Join-Path $Output 'RUN_STATUS.json'
        if (Test-Path -LiteralPath $runStatusPath) {
            try { $runStatus = (Get-Content -LiteralPath $runStatusPath -Raw | ConvertFrom-Json).status } catch {}
        }
        $health = [ordered]@{
            timestamp_local = (Get-Date).ToString('o')
            attempt = $attempt
            supervisor_child_pid = $process.Id
            worker_pid = if ($worker) { $worker.Id } else { $null }
            alive = $true
            cpu_seconds = if ($worker) { $worker.TotalProcessorTime.TotalSeconds } else { $process.TotalProcessorTime.TotalSeconds }
            working_set_gb = if ($worker) { [math]::Round($worker.WorkingSet64 / 1GB, 3) } else { [math]::Round($process.WorkingSet64 / 1GB, 3) }
            run_status = $runStatus
            gpu = $gpu
            stdout_bytes = if (Test-Path $stdout) { (Get-Item $stdout).Length } else { 0 }
            stderr_bytes = if (Test-Path $stderr) { (Get-Item $stderr).Length } else { 0 }
        }
        ($health | ConvertTo-Json -Compress) | Add-Content -LiteralPath $HealthLog -Encoding UTF8
        Start-Sleep -Seconds 60
        $process.Refresh()
    }

    $process.WaitForExit()
    $exitCode = $process.ExitCode
    $recommendation = Join-Path $Output 'FINAL_RECOMMENDATION_V14.json'
    $verified = $false
    if (Test-Path -LiteralPath $recommendation) {
        try { $verified = ((Get-Content -LiteralPath $recommendation -Raw | ConvertFrom-Json).status -eq 'SUCCESS_VERIFIED') } catch {}
    }
    $runSucceeded = $false
    $verifierPassed = $false
    $runStatusPath = Join-Path $Output 'RUN_STATUS.json'
    $verifierPath = Join-Path $Output 'VERIFIER_RESULTS_V14.json'
    if (Test-Path -LiteralPath $runStatusPath) {
        try { $runSucceeded = ((Get-Content -LiteralPath $runStatusPath -Raw | ConvertFrom-Json).status -eq 'SUCCESS') } catch {}
    }
    if (Test-Path -LiteralPath $verifierPath) {
        try { $verifierPassed = ((Get-Content -LiteralPath $verifierPath -Raw | ConvertFrom-Json).status -eq 'PASS') } catch {}
    }
    if ($verified -and $runSucceeded -and $verifierPassed) {
        Write-SupervisorStatus 'EXPERIMENT_SUCCESS' @{ attempt = $attempt; exit_code = $exitCode; verified = $verified }
        exit 0
    }

    $failure = $null
    $runStatusPath = Join-Path $Output 'RUN_STATUS.json'
    if (Test-Path -LiteralPath $runStatusPath) {
        try { $failure = Get-Content -LiteralPath $runStatusPath -Raw } catch {}
    }
    Write-SupervisorStatus 'ATTEMPT_FAILED' @{ attempt = $attempt; exit_code = $exitCode; verified = $verified; run_status = $failure }
    if ($attempt -lt $MaxAttempts) { Start-Sleep -Seconds 15 }
}

Write-SupervisorStatus 'FAILED_EXHAUSTED_RETRIES' @{ max_attempts = $MaxAttempts }
exit 1
