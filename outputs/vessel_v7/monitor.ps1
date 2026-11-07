param([switch]$Watch)
$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
function Read-SharedText([string]$Path) {
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, ([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
    $reader = [System.IO.StreamReader]::new($stream)
    try { $reader.ReadToEnd() } finally { $reader.Dispose(); $stream.Dispose() }
}
do {
    try {
        $state = Read-SharedText (Join-Path $root 'status.json') | ConvertFrom-Json
        Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  Stage=$($state.stage)"
        if ($state.stage -eq 'training') {
            Write-Host "Experiment=$($state.experiment)  Fold=$($state.fold) (0-4)  PeakLR=$($state.lr)"
            $livePath = Join-Path $state.run_dir 'live.json'
            if (Test-Path -LiteralPath $livePath) {
                $live = Read-SharedText $livePath | ConvertFrom-Json
                Write-Host "Batch=$($live.batch)/$($live.total_batches)  Loss=$($live.loss)  LR=$($live.lr)  Elapsed=$($live.elapsed_seconds)s"
            } else { Write-Host 'Preparing fold / evaluating step 0...' }
            $validation = Join-Path $state.run_dir 'validation.jsonl'
            if (Test-Path -LiteralPath $validation) {
                $v = ((Read-SharedText $validation).Trim() -split "`n")[-1] | ConvertFrom-Json
                Write-Host "Last validation: step=$($v.step)  Dice=$($v.mean_dice)  BestStep=$($v.best_step)"
            }
        } else { $state | ConvertTo-Json -Depth 3 }
        if ($state.stage -in @('completed','failed')) { break }
    } catch { Write-Host "Waiting for status: $_" }
    if ($Watch) { Start-Sleep -Seconds 2 }
} while ($Watch)
