param([switch]$Watch, [ValidateRange(1,300)][int]$IntervalSeconds = 2)
$ErrorActionPreference = 'Stop'
do {
    Write-Host "`n$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') v5"
    $statusPath = Join-Path $PSScriptRoot 'status.json'
    if (Test-Path -LiteralPath $statusPath) {
        $state = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        $state | Select-Object stage,time,experiment,fold,selected_experiment,error | Format-List
        if ($state.stage -eq 'training') {
            $run = Join-Path $PSScriptRoot "experiments\$($state.experiment)\fold$($state.fold)"
            $trainingLog = Join-Path $run 'training.jsonl'
            $validationLog = Join-Path $run 'validation.jsonl'
            if (Test-Path -LiteralPath $trainingLog) {
                $batchRecord = Get-Content -LiteralPath $trainingLog -Tail 1 | ConvertFrom-Json
                Write-Host ('方案 {0} | fold {1}（第 {2}/5 折）| 最近记录 Batch {3}/6000 | {4:N1}% | Loss {5:N6} | LR {6:E3}' -f $state.experiment, $state.fold, ([int]$state.fold + 1), $batchRecord.step, (100 * $batchRecord.step / 6000), $batchRecord.loss, $batchRecord.lr) -ForegroundColor Cyan
                Write-Host '每 batch 8 个裁块；每 25 batch 记录一次。Ctrl+C 仅停止监控。'
            }
            if (Test-Path -LiteralPath $validationLog) {
                Get-Content -LiteralPath $validationLog -Tail 1 | ConvertFrom-Json |
                    Select-Object step,weight_kind,mean_dice,best_step,best_kind | Format-List
            }
        }
    }
    & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
    if (-not $Watch -or $state.stage -in @('completed','failed')) { break }
    Start-Sleep -Seconds $IntervalSeconds
} while ($true)
