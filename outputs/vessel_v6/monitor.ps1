param([switch]$Watch,[ValidateRange(1,300)][int]$IntervalSeconds=2)
$ErrorActionPreference='Stop'
do {
    try {
        $state=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'status.json') -Raw | ConvertFrom-Json
        Write-Host "`n$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | v6 LoRA | $($state.stage)"
        if($state.stage -eq 'training') {
            $folder=Join-Path $PSScriptRoot "runs\fold$($state.fold)"
            $live=Join-Path $folder 'live.json'
            if(Test-Path -LiteralPath $live) {
                $b=Get-Content -LiteralPath $live -Raw | ConvertFrom-Json
                if($null -eq $b -or $null -eq $b.batch) { throw 'Progress update in flight' }
                Write-Host ('第 {0}/5 折 | Batch {1}/{2} | Loss {3:N6} | LR {4:E3} | 已用 {5:N1}s' -f ([int]$state.fold+1),$b.batch,$b.total_batches,$b.loss,$b.lr,$b.elapsed_seconds) -ForegroundColor Cyan
            }
            $val=Join-Path $folder 'validation.jsonl'
            if(Test-Path -LiteralPath $val) {
                Get-Content -LiteralPath $val -Tail 1 | ConvertFrom-Json | Select-Object step,best_step,mean_dice | Format-List
            }
        } elseif($state.stage -eq 'test_prediction') { Write-Host $state.model }
        if($state.stage -eq 'failed'){Write-Host $state.error -ForegroundColor Red}
        & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    } catch {Write-Host '正在更新进度文件，下一次刷新重试。'}
    if(-not $Watch -or $state.stage -in @('completed','failed')){break}
    Start-Sleep -Seconds $IntervalSeconds
} while($true)
