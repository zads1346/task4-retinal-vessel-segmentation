param([ValidateRange(1, 300)][int]$IntervalSeconds = 10)
# Read-only monitor. Ctrl+C stops this window, not the separate training process.
$ErrorActionPreference = 'Stop'
$v4MonitorRoot = $PSScriptRoot
while ($true) {
    try {
        $v4Progress = Get-Content -LiteralPath (Join-Path $v4MonitorRoot 'runs\cv_progress.json') -Raw | ConvertFrom-Json
        Clear-Host
        Write-Host '========== v4 training monitor ==========' -ForegroundColor Cyan
        $v4TrainingPath = Join-Path $v4Progress.current_run 'training.jsonl'
        if (Test-Path -LiteralPath $v4TrainingPath) {
            $v4Training = Get-Content -LiteralPath $v4TrainingPath -Tail 1 | ConvertFrom-Json
            $v4Best = 'pending'
            $v4ValidationPath = Join-Path $v4Progress.current_run 'validation.jsonl'
            if (Test-Path -LiteralPath $v4ValidationPath) {
                $v4ValidationRows = Get-Content -LiteralPath $v4ValidationPath | ForEach-Object { $_ | ConvertFrom-Json }
                $v4Maximum = ($v4ValidationRows | Measure-Object -Property mean_dice -Maximum).Maximum
                $v4Best = '{0:N4}' -f $v4Maximum
            }
            '{0:HH:mm:ss} | {1} | completed {2}/5 | fold {3}/5 | step {4}/6000 | loss {5:N4} | best Dice {6}' -f `
                (Get-Date), $v4Progress.status, @($v4Progress.completed).Count, ([int]$v4Progress.current_fold + 1), `
                $v4Training.step, $v4Training.loss, $v4Best
        } else {
            Write-Host ('Preparing fold {0}/5...' -f ([int]$v4Progress.current_fold + 1))
        }
        if (@($v4Progress.completed).Count -gt 0) {
            $v4Progress.completed | Select-Object fold, best_step, weight_kind, native_dice_at_0_5 |
                Format-Table -AutoSize | Out-Host
        }
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            Write-Host 'GPU utilization, used/total memory, temperature, power:'
            & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
        }
        if ($v4Progress.status -in @('failed', 'aggregation_failed')) {
            Write-Host 'Run failed; continuing to monitor for recovery. Recent log:' -ForegroundColor Yellow
            Get-Content -LiteralPath (Join-Path $v4Progress.current_run 'console.log') -Tail 8 | Out-Host
        }
        if ($v4Progress.status -eq 'completed') {
            Write-Host 'CV/OOF complete. Waiting for test ensemble, audit and release.'
        }
        $v4Manifest = Get-Content -LiteralPath (Join-Path $v4MonitorRoot 'version_manifest.json') -Raw | ConvertFrom-Json
        if ($v4Manifest.status -eq 'completed') {
            Write-Host 'All v4 training, prediction, audit and release work completed.' -ForegroundColor Green
            Write-Host (Join-Path $v4MonitorRoot 'submission_v4.0.0.csv')
            break
        }
        Write-Host ('Refresh every {0}s. Ctrl+C stops only this monitor.' -f $IntervalSeconds)
    } catch {
        Write-Host ('Waiting for a complete log update: {0}' -f $_.Exception.Message)
    }
    Start-Sleep -Seconds $IntervalSeconds
}
